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

pub struct HyperliquidClient {
    client: Client,
    base_url: Url,
    signing_key: SigningKey,
    pub main_address: String,
    asset_map: Mutex<Option<HashMap<String, u32>>>,
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
        #[derive(Debug, Deserialize)]
        struct MarginSummary {
            #[serde(rename = "accountValue")]
            account_value: String,
            #[serde(rename = "totalMarginUsed")]
            total_margin_used: String,
        }

        #[derive(Debug, Deserialize)]
        struct Response {
            #[serde(rename = "marginSummary")]
            margin_summary: MarginSummary,
            withdrawable: Option<String>,
        }

        let response: Response =
            self.info("clearinghouseState", json!({ "user": self.main_address }))?;
        let account_value_usd =
            parse_number(&response.margin_summary.account_value, "accountValue")?;
        let total_margin_used_usd = parse_number(
            &response.margin_summary.total_margin_used,
            "totalMarginUsed",
        )?;
        let withdrawable_usd = response
            .withdrawable
            .as_deref()
            .map(|value| parse_number(value, "withdrawable"))
            .transpose()?
            .unwrap_or(account_value_usd - total_margin_used_usd);
        Ok(HyperliquidAccountSnapshot {
            account_value_usd,
            withdrawable_usd,
            total_margin_used_usd,
        })
    }

    pub fn positions(&self) -> Result<Vec<HyperliquidPosition>> {
        #[derive(Debug, Deserialize)]
        struct Response {
            #[serde(rename = "assetPositions", default)]
            asset_positions: Vec<AssetPosition>,
        }

        #[derive(Debug, Deserialize)]
        struct AssetPosition {
            position: PositionPayload,
        }

        #[derive(Debug, Deserialize)]
        struct PositionPayload {
            coin: String,
            #[serde(rename = "entryPx")]
            entry_px: Option<String>,
            leverage: Option<LeveragePayload>,
            #[serde(rename = "marginUsed")]
            margin_used: Option<String>,
            szi: String,
        }

        #[derive(Debug, Deserialize)]
        struct LeveragePayload {
            value: Option<String>,
        }

        let response: Response =
            self.info("clearinghouseState", json!({ "user": self.main_address }))?;
        let mut positions = Vec::new();
        for asset in response.asset_positions {
            let symbol = asset.position.coin.trim().to_ascii_uppercase();
            if symbol.is_empty() {
                continue;
            }
            let signed_size = parse_number(&asset.position.szi, "szi")?;
            if signed_size.abs() <= f64::EPSILON {
                continue;
            }
            let entry_price = asset
                .position
                .entry_px
                .as_deref()
                .map(|value| parse_number(value, "entryPx"))
                .transpose()?
                .unwrap_or(0.0);
            let leverage = asset
                .position
                .leverage
                .as_ref()
                .and_then(|payload| payload.value.as_deref())
                .map(|value| parse_number(value, "leverage"))
                .transpose()?
                .unwrap_or(1.0)
                .max(1.0);
            let margin_used = asset
                .position
                .margin_used
                .as_deref()
                .map(|value| parse_number(value, "marginUsed"))
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

    pub fn all_mids(&self) -> Result<HashMap<String, f64>> {
        let response: HashMap<String, String> =
            self.info("allMids", serde_json::Value::Object(Default::default()))?;
        response
            .into_iter()
            .map(|(symbol, value)| {
                parse_number(&value, "midPx")
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
        let limit_px = local_slippage_limit_px(reference_price, is_buy, slippage_pct)
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
        let limit_px = local_slippage_limit_px(reference_price, is_buy, slippage_pct)
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

        let requested = symbol.trim().to_ascii_uppercase();
        let meta: MetaResponse =
            self.info("meta", serde_json::Value::Object(Default::default()))?;
        let (asset, universe_asset) = meta
            .universe
            .into_iter()
            .enumerate()
            .find(|(_, asset)| asset.name.trim().eq_ignore_ascii_case(&requested))
            .with_context(|| format!("unknown Hyperliquid symbol: {requested}"))?;

        Ok(HyperliquidAssetMeta {
            asset: asset as u32,
            symbol: universe_asset.name.trim().to_ascii_uppercase(),
            sz_decimals: universe_asset.sz_decimals,
            max_leverage: universe_asset.max_leverage,
        })
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
        let cached = self
            .asset_map
            .lock()
            .map_err(|_| anyhow::anyhow!("asset map mutex poisoned"))?;
        if let Some(map) = cached.as_ref() {
            if let Some(asset) = map.get(&symbol.trim().to_ascii_uppercase()) {
                return Ok(*asset);
            }
        }
        drop(cached);

        #[derive(Debug, Deserialize)]
        struct MetaResponse {
            universe: Vec<UniverseAsset>,
        }

        #[derive(Debug, Deserialize)]
        struct UniverseAsset {
            name: String,
        }

        let meta: MetaResponse =
            self.info("meta", serde_json::Value::Object(Default::default()))?;
        let map = meta
            .universe
            .into_iter()
            .enumerate()
            .map(|(idx, asset)| (asset.name.trim().to_ascii_uppercase(), idx as u32))
            .collect::<HashMap<_, _>>();
        let requested = symbol.trim().to_ascii_uppercase();
        let asset = map
            .get(&requested)
            .copied()
            .with_context(|| format!("unknown Hyperliquid symbol: {requested}"))?;
        let mut cached = self
            .asset_map
            .lock()
            .map_err(|_| anyhow::anyhow!("asset map mutex poisoned"))?;
        *cached = Some(map);
        Ok(asset)
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
        self.client
            .post(
                self.base_url
                    .join("info")
                    .context("failed to resolve info URL")?,
            )
            .json(&serde_json::Value::Object(body))
            .send()
            .context("Hyperliquid info request failed")?
            .error_for_status()
            .context("Hyperliquid info returned an error status")?
            .json()
            .context("failed to parse Hyperliquid info response")
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
        self.client
            .post(
                self.base_url
                    .join("exchange")
                    .context("failed to resolve exchange URL")?,
            )
            .json(&payload)
            .send()
            .context("Hyperliquid exchange request failed")?
            .error_for_status()
            .context("Hyperliquid exchange returned an error status")?
            .json()
            .context("failed to parse Hyperliquid exchange response")
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

pub fn local_slippage_limit_px(px: f64, is_buy: bool, slippage_pct: f64) -> Option<f64> {
    if !px.is_finite() || px <= 0.0 {
        return None;
    }
    let slip = slippage_pct.clamp(0.0005, 0.10);
    let adjusted = if is_buy {
        px * (1.0 + slip)
    } else {
        px * (1.0 - slip)
    };
    normalise_limit_px_for_wire(adjusted, is_buy)
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
        Value::String(value) => serde_json::Value::String(value.to_string()),
        Value::Binary(value) => serde_json::Value::String(format!("0x{}", hex::encode(value))),
        Value::Array(values) => {
            serde_json::Value::Array(values.iter().map(msgpack_value_to_json).collect())
        }
        Value::Map(entries) => {
            let mut map = serde_json::Map::new();
            for (key, value) in entries {
                map.insert(key.to_string(), msgpack_value_to_json(value));
            }
            serde_json::Value::Object(map)
        }
        Value::Ext(_, value) => serde_json::Value::String(format!("0x{}", hex::encode(value))),
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
}
