use serde::Deserialize;
use std::time::Duration;

const HL_INFO_URL: &str = "https://api.hyperliquid.xyz/info";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub struct HlAccountSnapshot {
    pub account_value: f64,
    pub withdrawable: f64,
    pub total_margin_used: f64,
}

/// Raw HL clearinghouseState response (only fields we need).
#[derive(Deserialize)]
struct ClearinghouseResponse {
    #[serde(rename = "marginSummary")]
    margin_summary: MarginSummary,
    /// Top-level withdrawable (string numeric).
    #[serde(default)]
    withdrawable: Option<String>,
}

#[derive(Deserialize)]
struct MarginSummary {
    #[serde(rename = "accountValue")]
    account_value: String,
    #[serde(rename = "totalMarginUsed")]
    total_margin_used: String,
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

    let account_value: f64 = data.margin_summary.account_value.parse().ok()?;
    let total_margin_used: f64 = data.margin_summary.total_margin_used.parse().ok()?;
    let withdrawable: f64 = data
        .withdrawable
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(account_value - total_margin_used);

    Some(HlAccountSnapshot {
        account_value,
        withdrawable,
        total_margin_used,
    })
}
