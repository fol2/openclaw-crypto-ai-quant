use serde::{de::DeserializeOwned, Deserialize};
use std::collections::HashMap;
use std::time::Duration;

const HL_INFO_URL: &str = "https://api.hyperliquid.xyz/info";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const USD_QUOTE_COINS: &[&str] = &["USDC", "USDH", "USDE", "USDT0"];

#[derive(Debug, Clone)]
pub struct HlAccountSnapshot {
    pub account_value: f64,
    pub withdrawable: f64,
    pub total_margin_used: f64,
    pub positions: Vec<HlPositionSnapshot>,
}

#[derive(Debug, Clone)]
pub struct HlPositionSnapshot {
    pub symbol: String,
    pub pos_type: String,
    pub open_timestamp: Option<String>,
    pub size: f64,
    pub entry_price: f64,
    pub leverage: f64,
    pub margin_used: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UserAbstractionMode {
    StandardLike,
    UnifiedAccount,
    PortfolioMargin,
}

impl UserAbstractionMode {
    fn from_raw(raw: &str) -> Self {
        match raw.trim() {
            "unifiedAccount" => Self::UnifiedAccount,
            "portfolioMargin" => Self::PortfolioMargin,
            _ => Self::StandardLike,
        }
    }

    fn uses_spot_balance_source(self) -> bool {
        matches!(self, Self::UnifiedAccount | Self::PortfolioMargin)
    }
}

/// Raw HL clearinghouseState response (only fields we need).
#[derive(Debug, Deserialize)]
struct ClearinghouseResponse {
    #[serde(rename = "marginSummary")]
    margin_summary: MarginSummary,
    #[serde(default)]
    withdrawable: Option<StringOrNumber>,
    #[serde(rename = "assetPositions", default)]
    asset_positions: Vec<ClearinghouseAssetPosition>,
}

#[derive(Debug, Deserialize)]
struct SpotClearinghouseResponse {
    #[serde(default)]
    balances: Vec<SpotBalance>,
    #[serde(rename = "tokenToAvailableAfterMaintenance", default)]
    token_to_available_after_maintenance: Vec<(u32, StringOrNumber)>,
}

#[derive(Debug, Deserialize)]
struct SpotBalance {
    coin: String,
    token: u32,
    hold: StringOrNumber,
    total: StringOrNumber,
}

#[derive(Debug, Deserialize)]
struct MarginSummary {
    #[serde(rename = "accountValue")]
    account_value: StringOrNumber,
    #[serde(rename = "totalMarginUsed")]
    total_margin_used: StringOrNumber,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StringOrNumber {
    String(String),
    Number(f64),
    Integer(i64),
    Unsigned(u64),
}

impl StringOrNumber {
    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::String(value) => value.parse().ok(),
            Self::Number(value) => Some(*value),
            Self::Integer(value) => Some(*value as f64),
            Self::Unsigned(value) => Some(*value as f64),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ClearinghouseAssetPosition {
    position: ClearinghousePosition,
}

#[derive(Debug, Deserialize)]
struct ClearinghousePosition {
    coin: String,
    #[serde(rename = "entryPx")]
    entry_px: Option<StringOrNumber>,
    leverage: Option<ClearinghouseLeverage>,
    #[serde(rename = "marginUsed")]
    margin_used: Option<StringOrNumber>,
    szi: StringOrNumber,
}

#[derive(Debug, Deserialize)]
struct ClearinghouseLeverage {
    value: Option<StringOrNumber>,
}

/// Fetch account snapshot from Hyperliquid while respecting unified-account balance semantics.
///
/// Returns `None` on any error (network, parse, timeout) so callers can
/// gracefully fall back to DB-based values.
pub async fn fetch_account_snapshot(
    client: &reqwest::Client,
    address: &str,
) -> Option<HlAccountSnapshot> {
    let abstraction = fetch_user_abstraction(client, address)
        .await
        .unwrap_or(UserAbstractionMode::StandardLike);
    let perps = fetch_info(
        client,
        serde_json::json!({
            "type": "clearinghouseState",
            "user": address,
        }),
    )
    .await?;
    let positions = parse_positions(&perps)?;

    if abstraction.uses_spot_balance_source() {
        let spot = fetch_info(
            client,
            serde_json::json!({
                "type": "spotClearinghouseState",
                "user": address,
            }),
        )
        .await?;
        parse_unified_account_snapshot(&spot, &positions)
    } else {
        parse_perps_account_snapshot(&perps, positions)
    }
}

async fn fetch_user_abstraction(
    client: &reqwest::Client,
    address: &str,
) -> Option<UserAbstractionMode> {
    let raw: String = fetch_info(
        client,
        serde_json::json!({
            "type": "userAbstraction",
            "user": address,
        }),
    )
    .await?;
    Some(UserAbstractionMode::from_raw(&raw))
}

async fn fetch_info<T: DeserializeOwned>(
    client: &reqwest::Client,
    body: serde_json::Value,
) -> Option<T> {
    let resp = client
        .post(HL_INFO_URL)
        .timeout(REQUEST_TIMEOUT)
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            tracing::warn!("HL API request failed: {e}");
            e
        })
        .ok()?;

    if !resp.status().is_success() {
        tracing::warn!("HL API returned status {}", resp.status());
        return None;
    }

    resp.json()
        .await
        .map_err(|e| {
            tracing::warn!("HL API response parse failed: {e}");
            e
        })
        .ok()
}

fn parse_perps_account_snapshot(
    data: &ClearinghouseResponse,
    positions: Vec<HlPositionSnapshot>,
) -> Option<HlAccountSnapshot> {
    let account_value = data.margin_summary.account_value.as_f64()?;
    let total_margin_used = data.margin_summary.total_margin_used.as_f64()?;
    let withdrawable = data
        .withdrawable
        .as_ref()
        .and_then(StringOrNumber::as_f64)
        .unwrap_or(account_value - total_margin_used);

    Some(HlAccountSnapshot {
        account_value,
        withdrawable,
        total_margin_used,
        positions,
    })
}

fn parse_unified_account_snapshot(
    data: &SpotClearinghouseResponse,
    positions: &[HlPositionSnapshot],
) -> Option<HlAccountSnapshot> {
    let available_by_token = data
        .token_to_available_after_maintenance
        .iter()
        .map(|(token, value)| value.as_f64().map(|value| (*token, value)))
        .collect::<Option<HashMap<_, _>>>()?;

    let mut account_value = 0.0;
    let mut withdrawable = 0.0;
    let mut saw_usdc = false;

    for balance in data
        .balances
        .iter()
        .filter(|balance| balance.coin.eq_ignore_ascii_case("USDC"))
    {
        saw_usdc = true;
        let total = balance.total.as_f64()?;
        let hold = balance.hold.as_f64()?;
        account_value += total;
        withdrawable += available_by_token
            .get(&balance.token)
            .copied()
            .unwrap_or((total - hold).max(0.0));
    }

    if !saw_usdc {
        for balance in data.balances.iter().filter(|balance| {
            USD_QUOTE_COINS
                .iter()
                .any(|coin| balance.coin.eq_ignore_ascii_case(coin))
        }) {
            let total = balance.total.as_f64()?;
            let hold = balance.hold.as_f64()?;
            account_value += total;
            withdrawable += available_by_token
                .get(&balance.token)
                .copied()
                .unwrap_or((total - hold).max(0.0));
        }
    }

    let hold_back = (account_value - withdrawable).max(0.0);
    let perps_margin_used = positions
        .iter()
        .map(|position| position.margin_used.max(0.0))
        .sum::<f64>();

    Some(HlAccountSnapshot {
        account_value,
        withdrawable,
        total_margin_used: hold_back.max(perps_margin_used),
        positions: positions.to_vec(),
    })
}

fn parse_positions(data: &ClearinghouseResponse) -> Option<Vec<HlPositionSnapshot>> {
    data.asset_positions
        .iter()
        .filter_map(|asset| {
            let symbol = asset.position.coin.trim().to_ascii_uppercase();
            if symbol.is_empty() {
                return None;
            }
            let signed_size = asset.position.szi.as_f64()?;
            if signed_size.abs() <= f64::EPSILON {
                return None;
            }
            let entry_price = asset
                .position
                .entry_px
                .as_ref()
                .and_then(StringOrNumber::as_f64)
                .unwrap_or(0.0);
            let leverage = asset
                .position
                .leverage
                .as_ref()
                .and_then(|payload| payload.value.as_ref())
                .and_then(StringOrNumber::as_f64)
                .unwrap_or(1.0);
            let margin_used = asset
                .position
                .margin_used
                .as_ref()
                .and_then(StringOrNumber::as_f64)
                .unwrap_or(0.0);

            Some(HlPositionSnapshot {
                symbol,
                pos_type: if signed_size > 0.0 {
                    "LONG".to_string()
                } else {
                    "SHORT".to_string()
                },
                open_timestamp: None,
                size: signed_size.abs(),
                entry_price,
                leverage: leverage.max(1.0),
                margin_used,
            })
        })
        .collect::<Vec<_>>()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_perps_account_snapshot_extracts_positions() {
        let payload = serde_json::json!({
            "marginSummary": {
                "accountValue": "222.032332",
                "totalMarginUsed": "179.856732"
            },
            "withdrawable": "42.1756",
            "assetPositions": [
                {
                    "position": {
                        "coin": "HYPE",
                        "entryPx": "35.7393",
                        "marginUsed": "77.627865",
                        "leverage": { "value": 4 },
                        "szi": "-8.31"
                    }
                },
                {
                    "position": {
                        "coin": "DOGE",
                        "entryPx": "0.094602",
                        "marginUsed": "42.226821",
                        "leverage": { "value": 4 },
                        "szi": "1692.0"
                    }
                }
            ]
        });

        let parsed: ClearinghouseResponse = serde_json::from_value(payload).unwrap();
        let positions = parse_positions(&parsed).unwrap();
        let snapshot = parse_perps_account_snapshot(&parsed, positions.clone()).unwrap();
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0].symbol, "HYPE");
        assert_eq!(positions[0].pos_type, "SHORT");
        assert!((positions[0].size - 8.31).abs() < 1e-9);
        assert!((positions[0].margin_used - 77.627865).abs() < 1e-9);
        assert_eq!(positions[1].symbol, "DOGE");
        assert_eq!(positions[1].pos_type, "LONG");
        assert!((positions[1].entry_price - 0.094602).abs() < 1e-9);
        assert!((snapshot.withdrawable - 42.1756).abs() < 1e-9);
    }

    #[test]
    fn parse_unified_account_snapshot_prefers_spot_balances() {
        let spot: SpotClearinghouseResponse = serde_json::from_value(serde_json::json!({
            "balances": [
                { "coin": "USDC", "token": 0, "total": "208.63", "hold": "0.0" },
                { "coin": "USDH", "token": 360, "total": "0.0", "hold": "0.0" }
            ],
            "tokenToAvailableAfterMaintenance": [[0, "208.63"]]
        }))
        .unwrap();

        let snapshot = parse_unified_account_snapshot(&spot, &[]).unwrap();
        assert!((snapshot.account_value - 208.63).abs() < 1e-9);
        assert!((snapshot.withdrawable - 208.63).abs() < 1e-9);
        assert_eq!(snapshot.total_margin_used, 0.0);
    }
}
