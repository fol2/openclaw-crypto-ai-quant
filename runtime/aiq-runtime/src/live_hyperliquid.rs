use anyhow::{Context, Result};
use k256::ecdsa::{RecoveryId, Signature, SigningKey};
use reqwest::blocking::Client;
use reqwest::Url;
use rmpv::{encode::write_value, Utf8String, Value};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha3::{Digest, Keccak256};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::live_secrets::{validate_address, validate_secret_key, LiveSecrets};

const DEFAULT_BASE_URL: &str = "https://api.hyperliquid.xyz";
const DEFAULT_TIMEOUT_S: f64 = 4.0;
const MAX_LEVERAGE: u32 = 50;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HyperliquidSignature {
    pub r: String,
    pub s: String,
    pub v: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyperliquidAccountSnapshot {
    pub account_value_usd: f64,
    pub withdrawable_usd: f64,
    pub total_margin_used_usd: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyperliquidPosition {
    pub symbol: String,
    pub pos_type: String,
    pub size: f64,
    pub entry_price: f64,
    pub leverage: f64,
    pub margin_used: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyperliquidFill {
    pub raw: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HyperliquidAssetMeta {
    pub asset: u32,
    pub symbol: String,
    pub sz_decimals: u32,
    pub max_leverage: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiveOrderType {
    LimitIoc,
    LimitGtc,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrderRequest {
    pub symbol: String,
    pub is_buy: bool,
    pub size: f64,
    pub limit_px: f64,
    pub reduce_only: bool,
    pub cloid: Option<String>,
    pub order_type: LiveOrderType,
}

#[derive(Debug, Deserialize)]
struct ClearinghouseStateResponse {
    #[serde(rename = "marginSummary")]
    margin_summary: MarginSummary,
    withdrawable: Option<StringOrNumber>,
    #[serde(rename = "assetPositions", default)]
    asset_positions: Vec<AssetPosition>,
}

#[derive(Debug, Deserialize)]
struct MarginSummary {
    #[serde(rename = "accountValue")]
    account_value: StringOrNumber,
    #[serde(rename = "totalMarginUsed")]
    total_margin_used: StringOrNumber,
}

#[derive(Debug, Deserialize)]
struct AssetPosition {
    position: PositionPayload,
}

#[derive(Debug, Deserialize)]
struct PositionPayload {
    coin: String,
    #[serde(rename = "entryPx")]
    entry_px: Option<StringOrNumber>,
    leverage: Option<LeveragePayload>,
    #[serde(rename = "marginUsed")]
    margin_used: Option<StringOrNumber>,
    szi: StringOrNumber,
}

#[derive(Debug, Deserialize)]
struct LeveragePayload {
    value: Option<StringOrNumber>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum StringOrNumber {
    String(String),
    Number(f64),
    Integer(i64),
    Unsigned(u64),
}

impl StringOrNumber {
    fn parse_f64(&self, field: &str) -> Result<f64> {
        match self {
            Self::String(raw) => parse_number(raw, field),
            Self::Number(raw) => Ok(*raw),
            Self::Integer(raw) => Ok(*raw as f64),
            Self::Unsigned(raw) => Ok(*raw as f64),
        }
    }
}

pub struct HyperliquidClient {
    client: Client,
    base_url: Url,
    signing_key: SigningKey,
    pub main_address: String,
    asset_map: Mutex<Option<HashMap<String, HyperliquidAssetMeta>>>,
}

impl HyperliquidClient {
    pub fn new(secrets: &LiveSecrets, timeout_s: Option<f64>) -> Result<Self> {
        Self::with_base_url(secrets, DEFAULT_BASE_URL, timeout_s)
    }

    pub fn with_base_url(
        secrets: &LiveSecrets,
        base_url: &str,
        timeout_s: Option<f64>,
    ) -> Result<Self> {
        validate_secret_key(&secrets.secret_key)?;
        validate_address(&secrets.main_address, "main_address")?;
        let stripped_key = secrets
            .secret_key
            .trim()
            .strip_prefix("0x")
            .unwrap_or(secrets.secret_key.trim());
        let key_bytes = hex::decode(stripped_key).context("failed to decode secret_key hex")?;
        let signing_key = SigningKey::from_slice(&key_bytes)
            .context("failed to initialise secp256k1 signing key")?;
        let base_url = Url::parse(base_url).context("invalid Hyperliquid base URL")?;
        let timeout_s = timeout_s.unwrap_or(DEFAULT_TIMEOUT_S).clamp(1.0, 30.0);
        let client = Client::builder()
            .timeout(Duration::from_secs_f64(timeout_s))
            .build()
            .context("failed to build Hyperliquid HTTP client")?;

        Ok(Self {
            client,
            base_url,
            signing_key,
            main_address: secrets.main_address.trim().to_string(),
            asset_map: Mutex::new(None),
        })
    }

    pub fn account_snapshot(&self) -> Result<HyperliquidAccountSnapshot> {
        let response: ClearinghouseStateResponse =
            self.info("clearinghouseState", json!({ "user": self.main_address }))?;
        let account_value_usd = response
            .margin_summary
            .account_value
            .parse_f64("accountValue")?;
        let total_margin_used_usd = response
            .margin_summary
            .total_margin_used
            .parse_f64("totalMarginUsed")?;
        let withdrawable_usd = response
            .withdrawable
            .as_ref()
            .map(|value| value.parse_f64("withdrawable"))
            .transpose()?
            .unwrap_or(account_value_usd - total_margin_used_usd);
        Ok(HyperliquidAccountSnapshot {
            account_value_usd,
            withdrawable_usd,
            total_margin_used_usd,
        })
    }

    pub fn positions(&self) -> Result<Vec<HyperliquidPosition>> {
        let response: ClearinghouseStateResponse =
            self.info("clearinghouseState", json!({ "user": self.main_address }))?;
        parse_positions(response)
    }

    pub fn all_mids(&self) -> Result<HashMap<String, f64>> {
        let response: HashMap<String, StringOrNumber> =
            self.info("allMids", serde_json::Value::Object(Default::default()))?;
        response
            .into_iter()
            .map(|(symbol, value)| {
                value
                    .parse_f64("midPx")
                    .map(|parsed| (symbol.trim().to_ascii_uppercase(), parsed))
            })
            .collect()
    }

    pub fn user_fills_by_time(
        &self,
        start_time_ms: i64,
        end_time_ms: i64,
    ) -> Result<Vec<HyperliquidFill>> {
        let response: Vec<serde_json::Value> = self.info(
            "userFillsByTime",
            json!({
                "user": self.main_address,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "aggregateByTime": false,
            }),
        )?;
        Ok(response
            .into_iter()
            .map(|raw| HyperliquidFill { raw })
            .collect())
    }

    pub fn update_leverage(
        &self,
        symbol: &str,
        leverage: u32,
        is_cross: bool,
    ) -> Result<serde_json::Value> {
        let lev = leverage.clamp(1, MAX_LEVERAGE);
        let asset = self.asset_for_symbol(symbol)?;
        let action = map_value(vec![
            ("type", Value::from("updateLeverage")),
            ("asset", Value::from(i64::from(asset))),
            ("isCross", Value::from(is_cross)),
            ("leverage", Value::from(i64::from(lev))),
        ]);
        self.post_action(action)
    }

    pub fn market_open(
        &self,
        symbol: &str,
        is_buy: bool,
        size: f64,
        px: Option<f64>,
        slippage_pct: f64,
        cloid: Option<&str>,
    ) -> Result<serde_json::Value> {
        let reference_price = px.unwrap_or_else(|| self.mid_for_symbol(symbol).unwrap_or(0.0));
        if reference_price <= 0.0 {
            anyhow::bail!("market_open requires a valid reference price for {symbol}");
        }
        let meta = self.asset_meta(symbol)?;
        let limit_px = market_order_limit_px(
            reference_price,
            is_buy,
            slippage_pct,
            meta.sz_decimals,
            meta.asset >= 10_000,
        )
        .context("failed to derive market_open slippage price")?;
        self.order(OrderRequest {
            symbol: symbol.trim().to_ascii_uppercase(),
            is_buy,
            size,
            limit_px,
            reduce_only: false,
            cloid: cloid.map(str::to_string),
            order_type: LiveOrderType::LimitIoc,
        })
    }

    pub fn market_close(
        &self,
        symbol: &str,
        is_buy: bool,
        size: f64,
        px: Option<f64>,
        slippage_pct: f64,
        cloid: Option<&str>,
    ) -> Result<serde_json::Value> {
        let reference_price = match px {
            Some(value) if value > 0.0 => value,
            _ => self.mid_for_symbol(symbol)?,
        };
        let meta = self.asset_meta(symbol)?;
        let limit_px = market_order_limit_px(
            reference_price,
            is_buy,
            slippage_pct,
            meta.sz_decimals,
            meta.asset >= 10_000,
        )
        .context("failed to derive market_close limit price")?;
        self.order(OrderRequest {
            symbol: symbol.trim().to_ascii_uppercase(),
            is_buy,
            size,
            limit_px,
            reduce_only: true,
            cloid: cloid.map(str::to_string),
            order_type: LiveOrderType::LimitIoc,
        })
    }

    pub fn asset_meta(&self, symbol: &str) -> Result<HyperliquidAssetMeta> {
        let requested = symbol.trim().to_ascii_uppercase();
        let cached = self
            .asset_map
            .lock()
            .map_err(|_| anyhow::anyhow!("asset map mutex poisoned"))?;
        if let Some(map) = cached.as_ref() {
            if let Some(meta) = map.get(&requested) {
                return Ok(meta.clone());
            }
        }
        drop(cached);

        let map = self.load_asset_meta_map()?;
        let meta = map
            .get(&requested)
            .cloned()
            .with_context(|| format!("unknown Hyperliquid symbol: {requested}"))?;
        let mut cached = self
            .asset_map
            .lock()
            .map_err(|_| anyhow::anyhow!("asset map mutex poisoned"))?;
        *cached = Some(map);
        Ok(meta)
    }

    pub fn open_orders(&self) -> Result<Vec<serde_json::Value>> {
        self.info(
            "frontendOpenOrders",
            json!({
                "user": self.main_address,
            }),
        )
    }

    pub fn cancel_order(&self, symbol: &str, oid: u64) -> Result<serde_json::Value> {
        let asset = self.asset_for_symbol(symbol)?;
        let action = map_value(vec![
            ("type", Value::from("cancel")),
            (
                "cancels",
                Value::Array(vec![map_value(vec![
                    ("a", Value::from(i64::from(asset))),
                    ("o", Value::from(oid as i64)),
                ])]),
            ),
        ]);
        self.post_action(action)
    }

    pub fn cancel_order_by_cloid(&self, symbol: &str, cloid: &str) -> Result<serde_json::Value> {
        validate_cloid(cloid)?;
        let asset = self.asset_for_symbol(symbol)?;
        let action = map_value(vec![
            ("type", Value::from("cancelByCloid")),
            (
                "cancels",
                Value::Array(vec![map_value(vec![
                    ("asset", Value::from(i64::from(asset))),
                    ("cloid", Value::from(cloid)),
                ])]),
            ),
        ]);
        self.post_action(action)
    }

    pub fn order(&self, order: OrderRequest) -> Result<serde_json::Value> {
        let asset = self.asset_for_symbol(&order.symbol)?;
        let mut fields = vec![
            ("a", Value::from(i64::from(asset))),
            ("b", Value::from(order.is_buy)),
            ("p", Value::from(float_to_wire(order.limit_px)?)),
            ("s", Value::from(float_to_wire(order.size)?)),
            ("r", Value::from(order.reduce_only)),
            (
                "t",
                match order.order_type {
                    LiveOrderType::LimitIoc => map_value(vec![(
                        "limit",
                        map_value(vec![("tif", Value::from("Ioc"))]),
                    )]),
                    LiveOrderType::LimitGtc => map_value(vec![(
                        "limit",
                        map_value(vec![("tif", Value::from("Gtc"))]),
                    )]),
                },
            ),
        ];
        if let Some(cloid) = order.cloid.as_deref() {
            validate_cloid(cloid)?;
            fields.push(("c", Value::from(cloid)));
        }

        let action = map_value(vec![
            ("type", Value::from("order")),
            ("orders", Value::Array(vec![map_value(fields)])),
            ("grouping", Value::from("na")),
        ]);
        self.post_action(action)
    }

    fn asset_for_symbol(&self, symbol: &str) -> Result<u32> {
        Ok(self.asset_meta(symbol)?.asset)
    }

    fn load_asset_meta_map(&self) -> Result<HashMap<String, HyperliquidAssetMeta>> {
        #[derive(Debug, Deserialize)]
        struct MetaResponse {
            universe: Vec<UniverseAsset>,
        }

        #[derive(Debug, Deserialize)]
        struct UniverseAsset {
            name: String,
            #[serde(rename = "szDecimals")]
            sz_decimals: u32,
            #[serde(rename = "maxLeverage")]
            max_leverage: u32,
        }

        let meta: MetaResponse =
            self.info("meta", serde_json::Value::Object(Default::default()))?;
        Ok(meta
            .universe
            .into_iter()
            .enumerate()
            .map(|(idx, asset)| {
                let symbol = asset.name.trim().to_ascii_uppercase();
                (
                    symbol.clone(),
                    HyperliquidAssetMeta {
                        asset: idx as u32,
                        symbol,
                        sz_decimals: asset.sz_decimals,
                        max_leverage: asset.max_leverage,
                    },
                )
            })
            .collect())
    }

    fn mid_for_symbol(&self, symbol: &str) -> Result<f64> {
        self.all_mids()?
            .remove(&symbol.trim().to_ascii_uppercase())
            .with_context(|| {
                format!(
                    "missing Hyperliquid mid for {}",
                    symbol.trim().to_ascii_uppercase()
                )
            })
    }

    fn info<T: serde::de::DeserializeOwned>(
        &self,
        request_type: &str,
        extra: serde_json::Value,
    ) -> Result<T> {
        let mut body = serde_json::Map::new();
        body.insert(
            "type".to_string(),
            serde_json::Value::String(request_type.to_string()),
        );
        if let serde_json::Value::Object(extra) = extra {
            body.extend(extra);
        }
        let response = self
            .client
            .post(
                self.base_url
                    .join("info")
                    .context("failed to resolve info URL")?,
            )
            .json(&serde_json::Value::Object(body))
            .send()
            .context("Hyperliquid info request failed")?;
        parse_json_response(response, "Hyperliquid info")
    }

    fn post_action(&self, action: Value) -> Result<serde_json::Value> {
        let nonce = current_timestamp_ms();
        let signature = sign_l1_action(&self.signing_key, &action, nonce, true)?;
        let action_json = msgpack_value_to_json(&action);
        let payload = json!({
            "action": action_json,
            "nonce": nonce,
            "signature": signature,
            "vaultAddress": serde_json::Value::Null,
            "expiresAfter": serde_json::Value::Null,
        });
        let response = self
            .client
            .post(
                self.base_url
                    .join("exchange")
                    .context("failed to resolve exchange URL")?,
            )
            .json(&payload)
            .send()
            .context("Hyperliquid exchange request failed")?;
        parse_json_response(response, "Hyperliquid exchange")
    }
}

pub fn response_has_embedded_error(response: &serde_json::Value) -> bool {
    response
        .get("response")
        .and_then(|response| response.get("data"))
        .and_then(|data| data.get("statuses"))
        .and_then(|statuses| statuses.as_array())
        .map(|statuses| {
            statuses.iter().any(|status| {
                status
                    .as_object()
                    .and_then(|status| status.get("error"))
                    .map(|error| !error.is_null())
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

pub fn extract_exchange_order_id(response: &serde_json::Value) -> Option<String> {
    for key in ["oid", "orderId", "order_id", "id"] {
        if let Some(value) = response.get(key) {
            let text = match value {
                serde_json::Value::String(value) => value.to_string(),
                other => other.to_string(),
            };
            if !text.is_empty() && text != "null" {
                return Some(text);
            }
        }
    }

    response
        .get("response")
        .and_then(|response| response.get("data"))
        .and_then(|data| data.get("statuses"))
        .and_then(|statuses| statuses.as_array())
        .and_then(|statuses| {
            statuses.iter().find_map(|status| {
                status.as_object().and_then(|status| {
                    ["filled", "resting"].iter().find_map(|key| {
                        status
                            .get(*key)
                            .and_then(|payload| payload.get("oid"))
                            .map(|value| match value {
                                serde_json::Value::String(value) => value.to_string(),
                                other => other.to_string(),
                            })
                    })
                })
            })
        })
        .filter(|value| !value.is_empty() && value != "null")
}

pub fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_millis() as u64
}

pub fn float_to_wire(value: f64) -> Result<String> {
    let rounded = format!("{value:.8}");
    let rounded_value = rounded
        .parse::<f64>()
        .with_context(|| format!("failed to parse rounded float {rounded}"))?;
    if (rounded_value - value).abs() >= 1e-12 {
        anyhow::bail!("float_to_wire causes rounding for {value}");
    }
    let rounded = if rounded == "-0.00000000" {
        "0.00000000".to_string()
    } else {
        rounded
    };
    let trimmed = rounded
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string();
    Ok(if trimmed.is_empty() || trimmed == "-0" {
        "0".to_string()
    } else {
        trimmed
    })
}

#[cfg(test)]
pub fn local_slippage_limit_px(px: f64, is_buy: bool, slippage_pct: f64) -> Option<f64> {
    market_order_limit_px(px, is_buy, slippage_pct, 0, false)
}

pub fn market_order_limit_px(
    px: f64,
    is_buy: bool,
    slippage_pct: f64,
    sz_decimals: u32,
    is_spot: bool,
) -> Option<f64> {
    if !px.is_finite() || px <= 0.0 {
        return None;
    }
    let slip = slippage_pct.clamp(0.0005, 0.10);
    let adjusted = if is_buy {
        px * (1.0 + slip)
    } else {
        px * (1.0 - slip)
    };
    sdk_slippage_price(adjusted, sz_decimals, is_spot)
}

pub fn normalise_limit_px_for_wire(px: f64, is_buy: bool) -> Option<f64> {
    if !px.is_finite() || px <= 0.0 {
        return None;
    }
    let scaled = px * 100_000_000.0;
    let rounded = if is_buy {
        scaled.ceil()
    } else {
        scaled.floor()
    } / 100_000_000.0;
    if rounded.is_finite() && rounded > 0.0 {
        Some(rounded)
    } else {
        None
    }
}

fn sdk_slippage_price(px: f64, sz_decimals: u32, is_spot: bool) -> Option<f64> {
    if !px.is_finite() || px <= 0.0 {
        return None;
    }
    let sigfig_px = round_to_significant_figures(px, 5)?;
    let decimals = (if is_spot { 8_i32 } else { 6_i32 }) - sz_decimals as i32;
    round_to_decimal_places(sigfig_px, decimals)
}

fn round_to_significant_figures(value: f64, sig_figs: u32) -> Option<f64> {
    if !value.is_finite() || value <= 0.0 || sig_figs == 0 {
        return None;
    }
    let precision = sig_figs.saturating_sub(1) as usize;
    let rounded = format!("{value:.precision$e}").parse::<f64>().ok()?;
    if rounded.is_finite() && rounded > 0.0 {
        Some(rounded)
    } else {
        None
    }
}

fn round_to_decimal_places(value: f64, decimals: i32) -> Option<f64> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let rounded = if decimals >= 0 {
        format!("{value:.precision$}", precision = decimals as usize)
            .parse::<f64>()
            .ok()?
    } else {
        let factor = 10f64.powi(-decimals);
        let scaled = format!("{:.0}", value / factor).parse::<f64>().ok()?;
        scaled * factor
    };
    if rounded.is_finite() && rounded > 0.0 {
        Some(rounded)
    } else {
        None
    }
}

pub fn validate_cloid(cloid: &str) -> Result<()> {
    let trimmed = cloid.trim();
    if trimmed.len() != 34
        || !trimmed.starts_with("0x")
        || !trimmed[2..].chars().all(|ch| ch.is_ascii_hexdigit())
    {
        anyhow::bail!("invalid cloid format; expected a 0x-prefixed 16-byte hex string");
    }
    Ok(())
}

pub fn action_hash(action: &Value, vault_address: Option<&str>, nonce: u64) -> Result<[u8; 32]> {
    let mut payload = Vec::new();
    write_value(&mut payload, action).context("failed to msgpack-encode Hyperliquid action")?;
    payload.extend_from_slice(&nonce.to_be_bytes());
    match vault_address {
        Some(address) => {
            validate_address(address, "vault_address")?;
            payload.push(0x01);
            payload.extend_from_slice(
                &hex::decode(&address[2..]).context("failed to decode vault address")?,
            );
        }
        None => payload.push(0x00),
    }
    let digest = Keccak256::digest(payload);
    Ok(digest.into())
}

pub fn sign_l1_action(
    signing_key: &SigningKey,
    action: &Value,
    nonce: u64,
    is_mainnet: bool,
) -> Result<HyperliquidSignature> {
    let connection_id = action_hash(action, None, nonce)?;
    sign_agent_payload(signing_key, &connection_id, is_mainnet)
}

pub fn sign_agent_payload(
    signing_key: &SigningKey,
    connection_id: &[u8; 32],
    is_mainnet: bool,
) -> Result<HyperliquidSignature> {
    let source_hash = keccak_bytes(if is_mainnet { b"a" } else { b"b" });
    let agent_type_hash = keccak_bytes(b"Agent(string source,bytes32 connectionId)");
    let domain_hash = exchange_domain_separator();

    let mut struct_payload = Vec::with_capacity(32 * 3);
    struct_payload.extend_from_slice(&agent_type_hash);
    struct_payload.extend_from_slice(&source_hash);
    struct_payload.extend_from_slice(connection_id);
    let struct_hash = keccak_bytes(&struct_payload);

    let mut digest = Keccak256::new();
    digest.update([0x19, 0x01]);
    digest.update(domain_hash);
    digest.update(struct_hash);
    let (signature, recovery_id) = signing_key
        .sign_digest_recoverable(digest)
        .context("failed to sign Hyperliquid typed data")?;
    Ok(signature_to_wire(&signature, recovery_id))
}

fn signature_to_wire(signature: &Signature, recovery_id: RecoveryId) -> HyperliquidSignature {
    let r = signature.r().to_bytes();
    let s = signature.s().to_bytes();
    HyperliquidSignature {
        r: format!("0x{}", hex::encode(r)),
        s: format!("0x{}", hex::encode(s)),
        v: 27 + recovery_id.to_byte(),
    }
}

fn exchange_domain_separator() -> [u8; 32] {
    let domain_type_hash = keccak_bytes(
        b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)",
    );
    let name_hash = keccak_bytes(b"Exchange");
    let version_hash = keccak_bytes(b"1");
    let mut payload = Vec::with_capacity(32 * 5);
    payload.extend_from_slice(&domain_type_hash);
    payload.extend_from_slice(&name_hash);
    payload.extend_from_slice(&version_hash);
    payload.extend_from_slice(&u256_word(1337));
    payload.extend_from_slice(&[0u8; 32]);
    keccak_bytes(&payload)
}

fn u256_word(value: u64) -> [u8; 32] {
    let mut word = [0u8; 32];
    word[24..].copy_from_slice(&value.to_be_bytes());
    word
}

fn keccak_bytes(bytes: &[u8]) -> [u8; 32] {
    Keccak256::digest(bytes).into()
}

fn parse_number(raw: &str, field: &str) -> Result<f64> {
    raw.parse::<f64>()
        .with_context(|| format!("failed to parse Hyperliquid numeric field {field}: {raw}"))
}

fn parse_json_response<T: serde::de::DeserializeOwned>(
    response: reqwest::blocking::Response,
    label: &str,
) -> Result<T> {
    let status = response.status();
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        anyhow::bail!("{label} returned HTTP {} {}", status, body.trim());
    }
    response
        .json()
        .with_context(|| format!("failed to parse {label} response"))
}

fn parse_positions(response: ClearinghouseStateResponse) -> Result<Vec<HyperliquidPosition>> {
    let mut positions = Vec::new();
    for asset in response.asset_positions {
        let symbol = asset.position.coin.trim().to_ascii_uppercase();
        if symbol.is_empty() {
            continue;
        }
        let signed_size = asset.position.szi.parse_f64("szi")?;
        if signed_size.abs() <= f64::EPSILON {
            continue;
        }
        let entry_price = asset
            .position
            .entry_px
            .as_ref()
            .map(|value| value.parse_f64("entryPx"))
            .transpose()?
            .unwrap_or(0.0);
        let leverage = asset
            .position
            .leverage
            .as_ref()
            .and_then(|payload| payload.value.as_ref())
            .map(|value| value.parse_f64("leverage"))
            .transpose()?
            .unwrap_or(1.0)
            .max(1.0);
        let margin_used = asset
            .position
            .margin_used
            .as_ref()
            .map(|value| value.parse_f64("marginUsed"))
            .transpose()?
            .unwrap_or(0.0);
        positions.push(HyperliquidPosition {
            symbol,
            pos_type: if signed_size > 0.0 {
                "LONG".to_string()
            } else {
                "SHORT".to_string()
            },
            size: signed_size.abs(),
            entry_price,
            leverage,
            margin_used,
        });
    }
    Ok(positions)
}

fn map_value(entries: Vec<(&str, Value)>) -> Value {
    Value::Map(
        entries
            .into_iter()
            .map(|(key, value)| (Value::String(Utf8String::from(key)), value))
            .collect(),
    )
}

fn msgpack_value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Nil => serde_json::Value::Null,
        Value::Boolean(value) => serde_json::Value::Bool(*value),
        Value::Integer(value) => {
            if let Some(value) = value.as_i64() {
                serde_json::Value::from(value)
            } else if let Some(value) = value.as_u64() {
                serde_json::Value::from(value)
            } else {
                serde_json::Value::from(value.to_string())
            }
        }
        Value::F32(value) => serde_json::Value::from(*value),
        Value::F64(value) => serde_json::Value::from(*value),
        Value::String(value) => serde_json::Value::String(
            value
                .as_str()
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| value.to_string()),
        ),
        Value::Binary(value) => serde_json::Value::String(format!("0x{}", hex::encode(value))),
        Value::Array(values) => {
            serde_json::Value::Array(values.iter().map(msgpack_value_to_json).collect())
        }
        Value::Map(entries) => {
            let mut map = serde_json::Map::new();
            for (key, value) in entries {
                map.insert(map_key_to_json_string(key), msgpack_value_to_json(value));
            }
            serde_json::Value::Object(map)
        }
        Value::Ext(_, value) => serde_json::Value::String(format!("0x{}", hex::encode(value))),
    }
}

fn map_key_to_json_string(key: &Value) -> String {
    match key {
        Value::String(value) => value
            .as_str()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| value.to_string()),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_secrets::LiveSecrets;

    fn sample_client() -> HyperliquidClient {
        HyperliquidClient::with_base_url(
            &LiveSecrets {
                secret_key: "0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                    .to_string(),
                main_address: "0x1111111111111111111111111111111111111111".to_string(),
            },
            "https://api.hyperliquid.xyz",
            Some(4.0),
        )
        .unwrap()
    }

    #[test]
    fn float_to_wire_matches_python_sdk_examples() {
        assert_eq!(float_to_wire(50000.0).unwrap(), "50000");
        assert_eq!(float_to_wire(0.01).unwrap(), "0.01");
        assert_eq!(float_to_wire(49000.12345678).unwrap(), "49000.12345678");
    }

    #[test]
    fn order_action_hash_matches_python_sdk_oracle() {
        let action = map_value(vec![
            ("type", Value::from("order")),
            (
                "orders",
                Value::Array(vec![map_value(vec![
                    ("a", Value::from(0_i64)),
                    ("b", Value::from(true)),
                    ("p", Value::from("50000")),
                    ("s", Value::from("0.01")),
                    ("r", Value::from(false)),
                    (
                        "t",
                        map_value(vec![(
                            "limit",
                            map_value(vec![("tif", Value::from("Ioc"))]),
                        )]),
                    ),
                ])]),
            ),
            ("grouping", Value::from("na")),
        ]);
        let digest = action_hash(&action, None, 1_700_000_000_000).unwrap();
        assert_eq!(
            hex::encode(digest),
            "38f5d1661d7dcafaa8bdb789d272b5ea4482ae6a37179b7254b79f2a2d0d4222"
        );
    }

    #[test]
    fn signatures_match_python_sdk_oracle() {
        let client = sample_client();

        let action = map_value(vec![
            ("type", Value::from("updateLeverage")),
            ("asset", Value::from(1_i64)),
            ("isCross", Value::from(true)),
            ("leverage", Value::from(7_i64)),
        ]);
        let signature =
            sign_l1_action(&client.signing_key, &action, 1_700_000_000_000, true).unwrap();
        assert_eq!(
            signature.r,
            "0x27ec16909b585ebeef3b8a2293595d137838a5b205a463349ab6fe3bffb655c8"
        );
        assert_eq!(
            signature.s,
            "0x5f6f513a5cf8ae7feeb18d6afead2635422c8cb342d7619bdb792957bea97f42"
        );
        assert_eq!(signature.v, 27);
    }

    #[test]
    fn cancel_by_cloid_signature_matches_python_sdk_oracle() {
        let client = sample_client();
        let action = map_value(vec![
            ("type", Value::from("cancelByCloid")),
            (
                "cancels",
                Value::Array(vec![map_value(vec![
                    ("asset", Value::from(5_i64)),
                    ("cloid", Value::from("0x6169715f1234567890abcdef12345678")),
                ])]),
            ),
        ]);
        let digest = action_hash(&action, None, 1_700_000_000_000).unwrap();
        assert_eq!(
            hex::encode(digest),
            "c611e3111bd5ac5302b818abf0efc8c667da26c51bef386334fea19f0ae8a5e5"
        );
        let signature =
            sign_l1_action(&client.signing_key, &action, 1_700_000_000_000, true).unwrap();
        assert_eq!(
            signature.r,
            "0x9b6485f102910e55d5fa58f142ecbb7dd19a9a59f819d4a1dc28240ba54016ed"
        );
        assert_eq!(
            signature.s,
            "0x3ab826465ff3fd7f2ee5ffc38e6a3537911eee800e0afb3a576cd6637225e809"
        );
        assert_eq!(signature.v, 28);
    }

    #[test]
    fn local_slippage_uses_buy_round_up_and_sell_round_down() {
        assert_eq!(local_slippage_limit_px(100.0, true, 0.01), Some(101.0));
        assert_eq!(local_slippage_limit_px(100.0, false, 0.01), Some(99.0));
    }

    #[test]
    fn market_order_limit_px_matches_python_sdk_for_eth_perp() {
        let price = market_order_limit_px(2105.65, false, 0.01, 4, false).unwrap();
        assert!((price - 2084.6).abs() < 1e-9);
        assert_eq!(float_to_wire(price).unwrap(), "2084.6");
    }

    #[test]
    fn market_order_limit_px_matches_python_sdk_for_hype_perp() {
        let price = market_order_limit_px(36.5525, false, 0.01, 2, false).unwrap();
        assert!((price - 36.187).abs() < 1e-9);
        assert_eq!(float_to_wire(price).unwrap(), "36.187");
    }

    #[test]
    fn market_order_limit_px_uses_bankers_rounding_like_python_sdk() {
        let price = market_order_limit_px(0.1555, false, 0.01, 0, false).unwrap();
        assert!((price - 0.15394).abs() < 1e-9);
        assert_eq!(float_to_wire(price).unwrap(), "0.15394");
    }

    #[test]
    fn market_order_limit_px_matches_python_sdk_on_decimal_tie_cases() {
        let price = market_order_limit_px(323.1818181818, false, 0.01, 5, false).unwrap();
        assert!((price - 319.9).abs() < 1e-9);
        assert_eq!(float_to_wire(price).unwrap(), "319.9");
    }

    #[test]
    fn helpers_detect_embedded_errors_and_extract_order_ids() {
        let response = json!({
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        { "resting": { "oid": 123456789 } },
                        { "error": "minimum order value" }
                    ]
                }
            }
        });
        assert!(response_has_embedded_error(&response));
        assert_eq!(
            extract_exchange_order_id(&response).as_deref(),
            Some("123456789")
        );
    }

    #[test]
    fn clearinghouse_positions_accept_numeric_info_fields() {
        let payload = json!({
            "marginSummary": {
                "accountValue": "223.843819",
                "totalMarginUsed": "175.36588"
            },
            "withdrawable": 48.477939,
            "assetPositions": [
                {
                    "position": {
                        "coin": "HYPE",
                        "entryPx": "35.7393",
                        "marginUsed": "75.309375",
                        "leverage": { "value": 4 },
                        "szi": "-8.31"
                    }
                },
                {
                    "position": {
                        "coin": "DOGE",
                        "entryPx": 0.094602,
                        "marginUsed": 41.451039,
                        "leverage": { "value": "4" },
                        "szi": "1692.0"
                    }
                }
            ]
        });

        let response: ClearinghouseStateResponse = serde_json::from_value(payload).unwrap();
        let positions = parse_positions(response).unwrap();
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0].symbol, "HYPE");
        assert_eq!(positions[0].pos_type, "SHORT");
        assert!((positions[0].size - 8.31).abs() < 1e-9);
        assert!((positions[0].leverage - 4.0).abs() < 1e-9);
        assert!((positions[1].entry_price - 0.094602).abs() < 1e-9);
        assert!((positions[1].margin_used - 41.451039).abs() < 1e-9);
    }

    #[test]
    fn msgpack_json_preserves_raw_string_keys_and_values() {
        let action = map_value(vec![
            ("type", Value::from("order")),
            (
                "orders",
                Value::Array(vec![map_value(vec![
                    ("a", Value::from(159_i64)),
                    ("b", Value::from(true)),
                    ("p", Value::from("36.61452001")),
                    ("s", Value::from("2.82")),
                    ("r", Value::from(true)),
                    (
                        "t",
                        map_value(vec![(
                            "limit",
                            map_value(vec![("tif", Value::from("Ioc"))]),
                        )]),
                    ),
                ])]),
            ),
            ("grouping", Value::from("na")),
        ]);

        let json_value = msgpack_value_to_json(&action);
        assert_eq!(json_value["type"], "order");
        assert_eq!(json_value["grouping"], "na");
        assert_eq!(json_value["orders"][0]["p"], "36.61452001");
        assert_eq!(json_value["orders"][0]["t"]["limit"]["tif"], "Ioc");
        let encoded = json_value.to_string();
        assert!(!encoded.contains("\\\"type\\\""));
        assert!(!encoded.contains("\\\"order\\\""));
    }
}
