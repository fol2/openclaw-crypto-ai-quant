use serde::Deserialize;
use std::time::Duration;

const HL_INFO_URL: &str = "https://api.hyperliquid.xyz/info";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

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

/// Raw HL clearinghouseState response (only fields we need).
#[derive(Deserialize)]
struct ClearinghouseResponse {
    #[serde(rename = "marginSummary")]
    margin_summary: MarginSummary,
    /// Top-level withdrawable (string numeric).
    #[serde(default)]
    withdrawable: Option<String>,
    #[serde(rename = "assetPositions", default)]
    asset_positions: Vec<ClearinghouseAssetPosition>,
}

#[derive(Deserialize)]
struct MarginSummary {
    #[serde(rename = "accountValue")]
    account_value: String,
    #[serde(rename = "totalMarginUsed")]
    total_margin_used: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StringOrNumber {
    String(String),
    Number(f64),
    Integer(i64),
}

impl StringOrNumber {
    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::String(value) => value.parse().ok(),
            Self::Number(value) => Some(*value),
            Self::Integer(value) => Some(*value as f64),
        }
    }
}

#[derive(Deserialize)]
struct ClearinghouseAssetPosition {
    position: ClearinghousePosition,
}

#[derive(Deserialize)]
struct ClearinghousePosition {
    coin: String,
    #[serde(rename = "entryPx")]
    entry_px: Option<StringOrNumber>,
    leverage: Option<ClearinghouseLeverage>,
    #[serde(rename = "marginUsed")]
    margin_used: Option<StringOrNumber>,
    szi: StringOrNumber,
}

#[derive(Deserialize)]
struct ClearinghouseLeverage {
    value: Option<StringOrNumber>,
}

/// Fetch account snapshot from Hyperliquid clearinghouse API.
///
/// Returns `None` on any error (network, parse, timeout) so callers can
/// gracefully fall back to DB-based values.
pub async fn fetch_account_snapshot(
    client: &reqwest::Client,
    address: &str,
) -> Option<HlAccountSnapshot> {
    let body = serde_json::json!({
        "type": "clearinghouseState",
        "user": address,
    });

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

    let data: ClearinghouseResponse = resp
        .json()
        .await
        .map_err(|e| {
            tracing::warn!("HL API response parse failed: {e}");
            e
        })
        .ok()?;

    parse_account_snapshot(data)
}

fn parse_account_snapshot(data: ClearinghouseResponse) -> Option<HlAccountSnapshot> {
    let account_value: f64 = data.margin_summary.account_value.parse().ok()?;
    let total_margin_used: f64 = data.margin_summary.total_margin_used.parse().ok()?;
    let withdrawable: f64 = data
        .withdrawable
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(account_value - total_margin_used);

    let positions = data
        .asset_positions
        .into_iter()
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
            let leverage: f64 = asset
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
        .collect();

    Some(HlAccountSnapshot {
        account_value,
        withdrawable,
        total_margin_used,
        positions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_account_snapshot_extracts_positions() {
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

        let parsed = parse_account_snapshot(serde_json::from_value(payload).unwrap()).unwrap();
        assert_eq!(parsed.positions.len(), 2);
        assert_eq!(parsed.positions[0].symbol, "HYPE");
        assert_eq!(parsed.positions[0].pos_type, "SHORT");
        assert!((parsed.positions[0].size - 8.31).abs() < 1e-9);
        assert!((parsed.positions[0].margin_used - 77.627865).abs() < 1e-9);
        assert_eq!(parsed.positions[1].symbol, "DOGE");
        assert_eq!(parsed.positions[1].pos_type, "LONG");
        assert!((parsed.positions[1].entry_price - 0.094602).abs() < 1e-9);
    }
}
