use anyhow::{anyhow, bail, Context, Result};
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use sha3::{Digest, Sha3_256};
use std::env;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const DEFAULT_MATCH_TTL_MS: i64 = 10 * 60 * 1_000;
const DEFAULT_DB_TIMEOUT_MS: u64 = 1_000;

static INTENT_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct LiveOmsOptions {
    pub match_ttl_ms: i64,
    pub db_timeout_ms: u64,
    pub cloid_prefix: String,
}

impl Default for LiveOmsOptions {
    fn default() -> Self {
        let cloid_prefix = env::var("AI_QUANT_OMS_CLOID_PREFIX")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "aiq_".to_string());
        Self {
            match_ttl_ms: DEFAULT_MATCH_TTL_MS,
            db_timeout_ms: DEFAULT_DB_TIMEOUT_MS,
            cloid_prefix,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntentHandle {
    pub intent_id: String,
    pub client_order_id: Option<String>,
    pub dedupe_key: Option<String>,
    pub duplicate: bool,
}

#[derive(Debug, Clone)]
pub struct CreateIntentRequest<'a> {
    pub symbol: &'a str,
    pub action: &'a str,
    pub side: &'a str,
    pub requested_size: Option<f64>,
    pub requested_notional: Option<f64>,
    pub leverage: Option<f64>,
    pub decision_ts_ms: Option<i64>,
    pub reason: Option<&'a str>,
    pub confidence: Option<&'a str>,
    pub entry_atr: Option<f64>,
    pub meta: Option<&'a Value>,
    pub dedupe_open: bool,
    pub strategy_version: Option<&'a str>,
    pub strategy_sha1: Option<&'a str>,
}

#[derive(Debug, Clone)]
pub struct SubmitUnknownRequest<'a> {
    pub symbol: &'a str,
    pub side: &'a str,
    pub order_type: &'a str,
    pub reduce_only: bool,
    pub requested_size: Option<f64>,
    pub error: Option<&'a str>,
}

#[derive(Debug, Clone)]
pub struct SentOrderRequest<'a> {
    pub symbol: &'a str,
    pub side: &'a str,
    pub order_type: &'a str,
    pub reduce_only: bool,
    pub requested_size: Option<f64>,
    pub result: Option<&'a Value>,
    pub exchange_order_id: Option<&'a str>,
}

#[derive(Debug, Clone)]
pub struct InsertFillRequest<'a> {
    pub ts_ms: i64,
    pub symbol: &'a str,
    pub intent_id: Option<&'a str>,
    pub order_id: Option<i64>,
    pub action: Option<&'a str>,
    pub side: Option<&'a str>,
    pub pos_type: Option<&'a str>,
    pub price: f64,
    pub size: f64,
    pub notional: f64,
    pub fee_usd: Option<f64>,
    pub fee_token: Option<&'a str>,
    pub fee_rate: Option<f64>,
    pub pnl_usd: Option<f64>,
    pub fill_hash: Option<&'a str>,
    pub fill_tid: Option<i64>,
    pub matched_via: Option<&'a str>,
    pub raw: Option<&'a Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FillMatch {
    pub intent_id: String,
    pub matched_via: &'static str,
}

#[derive(Debug, Clone)]
pub struct LiveOms {
    store: OmsStore,
    match_ttl_ms: i64,
    cloid_prefix: String,
}

impl LiveOms {
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        Self::with_options(db_path, LiveOmsOptions::default())
    }

    pub fn with_options<P: AsRef<Path>>(db_path: P, options: LiveOmsOptions) -> Result<Self> {
        let store = OmsStore::new(db_path.as_ref().to_path_buf(), options.db_timeout_ms);
        store.ensure()?;
        Ok(Self {
            store,
            match_ttl_ms: options.match_ttl_ms.max(0),
            cloid_prefix: normalise_prefix(&options.cloid_prefix),
        })
    }

    pub fn db_path(&self) -> &Path {
        self.store.db_path()
    }

    pub fn create_intent(&self, request: CreateIntentRequest<'_>) -> Result<IntentHandle> {
        let symbol = require_upper("symbol", request.symbol)?;
        let action = require_upper("action", request.action)?;
        let side = require_upper("side", request.side)?;
        let created_ts_ms = now_ms();
        let intent_id = make_intent_id();
        let client_order_id = make_hl_cloid(&intent_id, &self.cloid_prefix);
        let dedupe_key = if request.dedupe_open && action == "OPEN" {
            request
                .decision_ts_ms
                .map(|decision_ts_ms| format!("OPEN:{symbol}:{side}:{decision_ts_ms}"))
        } else {
            None
        };
        let meta_json = serialise_json(request.meta)?;
        let inserted = self.store.insert_intent(&IntentInsert {
            intent_id: &intent_id,
            client_order_id: Some(client_order_id.as_str()),
            created_ts_ms,
            symbol: &symbol,
            action: &action,
            side: &side,
            requested_size: request.requested_size,
            requested_notional: request.requested_notional,
            entry_atr: request.entry_atr,
            leverage: request.leverage,
            decision_ts_ms: request.decision_ts_ms,
            strategy_version: request.strategy_version,
            strategy_sha1: request.strategy_sha1,
            reason: request.reason,
            confidence: request.confidence,
            status: "NEW",
            dedupe_key: dedupe_key.as_deref(),
            meta_json: meta_json.as_deref(),
        })?;

        if inserted {
            return Ok(IntentHandle {
                intent_id,
                client_order_id: Some(client_order_id),
                dedupe_key,
                duplicate: false,
            });
        }

        let Some(dedupe_key_value) = dedupe_key else {
            bail!("failed to insert OMS intent without a dedupe key");
        };

        let Some((existing_id, existing_client_order_id)) =
            self.store.get_intent_by_dedupe_key(&dedupe_key_value)?
        else {
            bail!("dedupe key collision occurred but no existing intent was readable");
        };

        let client_order_id = match existing_client_order_id {
            Some(existing_client_order_id) if is_valid_hl_cloid(&existing_client_order_id) => {
                Some(existing_client_order_id)
            }
            maybe_invalid => {
                let upgraded = make_hl_cloid(&existing_id, &self.cloid_prefix);
                if let Err(error) = self
                    .store
                    .set_intent_client_order_id(&existing_id, upgraded.as_str())
                {
                    if maybe_invalid.as_deref().unwrap_or("").is_empty() {
                        return Err(error);
                    }
                    maybe_invalid
                } else {
                    Some(upgraded)
                }
            }
        };

        Ok(IntentHandle {
            intent_id: existing_id,
            client_order_id,
            dedupe_key: Some(dedupe_key_value),
            duplicate: true,
        })
    }

    pub fn mark_would(&self, intent: &IntentHandle, note: Option<&str>) -> Result<()> {
        let note = note
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("dry_live");
        self.store
            .set_intent_status(&intent.intent_id, "WOULD", note)
    }

    pub fn mark_failed(&self, intent: &IntentHandle, error: &str) -> Result<()> {
        let error = error.trim();
        let error = if error.is_empty() {
            "order_rejected"
        } else {
            error
        };
        self.store
            .set_intent_status(&intent.intent_id, "REJECTED", error)
    }

    pub fn mark_submit_unknown(
        &self,
        intent: &IntentHandle,
        request: SubmitUnknownRequest<'_>,
    ) -> Result<()> {
        let symbol = require_upper("symbol", request.symbol)?;
        let side = require_upper("side", request.side)?;
        let order_type = require_trimmed("order_type", request.order_type)?;
        let sent_ts_ms = now_ms();
        let error = request
            .error
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("submit_unknown");
        let raw_json = json!({
            "kind": "submit_unknown",
            "error": error,
        })
        .to_string();
        self.store.record_order_submission(&OrderSubmission {
            intent_id: &intent.intent_id,
            status: "UNKNOWN",
            sent_ts_ms,
            symbol: &symbol,
            side: &side,
            order_type: &order_type,
            requested_size: request.requested_size,
            reduce_only: request.reduce_only,
            client_order_id: intent.client_order_id.as_deref(),
            exchange_order_id: None,
            last_error: error,
            raw_json: Some(raw_json.as_str()),
        })
    }

    pub fn mark_sent(&self, intent: &IntentHandle, request: SentOrderRequest<'_>) -> Result<()> {
        let symbol = require_upper("symbol", request.symbol)?;
        let side = require_upper("side", request.side)?;
        let order_type = require_trimmed("order_type", request.order_type)?;
        let sent_ts_ms = now_ms();
        let exchange_order_id = request
            .exchange_order_id
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .or_else(|| {
                request
                    .result
                    .and_then(extract_exchange_order_id_from_result)
            });
        let raw_json = serialise_json(request.result)?;
        self.store.record_order_submission(&OrderSubmission {
            intent_id: &intent.intent_id,
            status: "SENT",
            sent_ts_ms,
            symbol: &symbol,
            side: &side,
            order_type: &order_type,
            requested_size: request.requested_size,
            reduce_only: request.reduce_only,
            client_order_id: intent.client_order_id.as_deref(),
            exchange_order_id: exchange_order_id.as_deref(),
            last_error: "",
            raw_json: raw_json.as_deref(),
        })
    }

    pub fn insert_fill(&self, request: InsertFillRequest<'_>) -> Result<bool> {
        self.store.insert_fill(&request)
    }

    pub fn find_pending_intent(
        &self,
        symbol: &str,
        action: &str,
        side: &str,
        ts_ms: i64,
    ) -> Result<Option<String>> {
        self.store.find_pending_intent(
            &require_upper("symbol", symbol)?,
            &require_upper("action", action)?,
            &require_upper("side", side)?,
            ts_ms,
            self.match_ttl_ms,
        )
    }

    pub fn find_pending_intent_with_ttl(
        &self,
        symbol: &str,
        action: &str,
        side: &str,
        ts_ms: i64,
        ttl_ms: i64,
    ) -> Result<Option<String>> {
        self.store.find_pending_intent(
            &require_upper("symbol", symbol)?,
            &require_upper("action", action)?,
            &require_upper("side", side)?,
            ts_ms,
            ttl_ms,
        )
    }

    /// Match a live fill to an in-flight intent using the same precedence as Python OMS.
    pub fn match_intent_for_fill(
        &self,
        fill: &Value,
        symbol: &str,
        action: &str,
        side: &str,
        ts_ms: i64,
    ) -> Result<Option<FillMatch>> {
        if let Some(exchange_order_id) = extract_fill_exchange_order_id(fill) {
            if let Some(intent_id) = self
                .store
                .find_intent_by_exchange_order_id(&exchange_order_id)?
            {
                return Ok(Some(FillMatch {
                    intent_id,
                    matched_via: "exchange_order_id",
                }));
            }
        }

        if let Some(fill_hash) = extract_optional_text(fill, &["hash"]) {
            if let Some(intent_id) = self.store.find_intent_by_fill_hash(&fill_hash)? {
                return Ok(Some(FillMatch {
                    intent_id,
                    matched_via: "fill_hash_sibling",
                }));
            }
        }

        if let Some(client_order_id) =
            extract_optional_text(fill, &["cloid", "clientOrderId", "client_order_id"])
        {
            if let Some(intent_id) = self
                .store
                .find_intent_by_client_order_id(&client_order_id)?
            {
                return Ok(Some(FillMatch {
                    intent_id,
                    matched_via: "client_order_id",
                }));
            }
        }

        if let Some(intent_id) = self.find_pending_intent(symbol, action, side, ts_ms)? {
            return Ok(Some(FillMatch {
                intent_id,
                matched_via: "time_proximity",
            }));
        }

        Ok(None)
    }
}

#[derive(Debug, Clone)]
struct OmsStore {
    db_path: PathBuf,
    busy_timeout: Duration,
}

impl OmsStore {
    fn new(db_path: PathBuf, db_timeout_ms: u64) -> Self {
        Self {
            db_path,
            busy_timeout: Duration::from_millis(db_timeout_ms.max(1)),
        }
    }

    fn db_path(&self) -> &Path {
        &self.db_path
    }

    fn open(&self) -> Result<Connection> {
        let conn = Connection::open(&self.db_path).with_context(|| {
            format!(
                "failed to open OMS SQLite database at {}",
                self.db_path.display()
            )
        })?;
        conn.busy_timeout(self.busy_timeout)
            .context("failed to configure OMS SQLite busy timeout")?;
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            ",
        )
        .context("failed to configure OMS SQLite pragmas")?;
        Ok(conn)
    }

    fn ensure(&self) -> Result<()> {
        let mut conn = self.open()?;
        let tx = conn
            .transaction()
            .context("failed to start OMS schema transaction")?;
        tx.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS oms_intents (
                intent_id TEXT PRIMARY KEY,
                created_ts_ms INTEGER NOT NULL,
                sent_ts_ms INTEGER,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                side TEXT NOT NULL,
                requested_size REAL,
                requested_notional REAL,
                entry_atr REAL,
                leverage REAL,
                decision_ts_ms INTEGER,
                strategy_version TEXT,
                strategy_sha1 TEXT,
                reason TEXT,
                confidence TEXT,
                status TEXT,
                dedupe_key TEXT,
                client_order_id TEXT,
                exchange_order_id TEXT,
                last_error TEXT,
                meta_json TEXT
            );
            CREATE TABLE IF NOT EXISTS oms_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent_id TEXT,
                created_ts_ms INTEGER NOT NULL,
                symbol TEXT,
                side TEXT,
                order_type TEXT,
                requested_size REAL,
                reduce_only INTEGER,
                client_order_id TEXT,
                exchange_order_id TEXT,
                status TEXT,
                raw_json TEXT
            );
            CREATE TABLE IF NOT EXISTS oms_fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER,
                symbol TEXT,
                intent_id TEXT,
                order_id INTEGER,
                action TEXT,
                side TEXT,
                pos_type TEXT,
                price REAL,
                size REAL,
                notional REAL,
                fee_usd REAL,
                fee_token TEXT,
                fee_rate REAL,
                pnl_usd REAL,
                fill_hash TEXT,
                fill_tid INTEGER,
                matched_via TEXT,
                raw_json TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_oms_intents_dedupe ON oms_intents(dedupe_key);
            CREATE INDEX IF NOT EXISTS idx_oms_intents_symbol_status
                ON oms_intents(symbol, status, sent_ts_ms);
            CREATE INDEX IF NOT EXISTS idx_oms_intents_client_order_id
                ON oms_intents(client_order_id);
            CREATE INDEX IF NOT EXISTS idx_oms_orders_intent
                ON oms_orders(intent_id);
            CREATE INDEX IF NOT EXISTS idx_oms_orders_symbol
                ON oms_orders(symbol, created_ts_ms);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_oms_fills_hash_tid
                ON oms_fills(fill_hash, fill_tid);
            CREATE INDEX IF NOT EXISTS idx_oms_fills_intent
                ON oms_fills(intent_id);
            CREATE INDEX IF NOT EXISTS idx_oms_fills_symbol_ts
                ON oms_fills(symbol, ts_ms);
            ",
        )
        .context("failed to create OMS schema")?;
        tx.commit().context("failed to commit OMS schema")?;
        Ok(())
    }

    fn insert_intent(&self, intent: &IntentInsert<'_>) -> Result<bool> {
        let conn = self.open()?;
        let changed = conn
            .execute(
                "
                INSERT OR IGNORE INTO oms_intents (
                    intent_id, created_ts_ms, symbol, action, side,
                    requested_size, requested_notional, entry_atr, leverage,
                    decision_ts_ms, strategy_version, strategy_sha1,
                    reason, confidence, status, dedupe_key, client_order_id, meta_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ",
                params![
                    intent.intent_id,
                    intent.created_ts_ms,
                    intent.symbol,
                    intent.action,
                    intent.side,
                    intent.requested_size,
                    intent.requested_notional,
                    intent.entry_atr,
                    intent.leverage,
                    intent.decision_ts_ms,
                    intent.strategy_version,
                    intent.strategy_sha1,
                    normalise_optional_text(intent.reason),
                    normalise_optional_text(intent.confidence),
                    intent.status,
                    intent.dedupe_key,
                    intent.client_order_id,
                    intent.meta_json,
                ],
            )
            .context("failed to insert OMS intent")?;
        Ok(changed > 0)
    }

    fn get_intent_by_dedupe_key(
        &self,
        dedupe_key: &str,
    ) -> Result<Option<(String, Option<String>)>> {
        if dedupe_key.trim().is_empty() {
            return Ok(None);
        }
        let conn = self.open()?;
        conn.query_row(
            "
            SELECT intent_id, client_order_id
            FROM oms_intents
            WHERE dedupe_key = ?
            LIMIT 1
            ",
            params![dedupe_key],
            |row| {
                let intent_id: String = row.get(0)?;
                let client_order_id: Option<String> = row.get(1)?;
                Ok((intent_id, client_order_id))
            },
        )
        .optional()
        .context("failed to query OMS intent by dedupe key")
    }

    fn set_intent_client_order_id(&self, intent_id: &str, client_order_id: &str) -> Result<()> {
        let conn = self.open()?;
        let changed = conn
            .execute(
                "
                UPDATE oms_intents
                SET client_order_id = ?
                WHERE intent_id = ?
                ",
                params![client_order_id, intent_id],
            )
            .context("failed to update OMS client_order_id")?;
        if changed == 0 {
            bail!("cannot update client_order_id for missing intent {intent_id}");
        }
        Ok(())
    }

    fn set_intent_status(&self, intent_id: &str, status: &str, last_error: &str) -> Result<()> {
        let conn = self.open()?;
        let changed = conn
            .execute(
                "
                UPDATE oms_intents
                SET status = ?, last_error = ?
                WHERE intent_id = ?
                ",
                params![status, last_error, intent_id],
            )
            .with_context(|| format!("failed to update OMS status for intent {intent_id}"))?;
        if changed == 0 {
            bail!("cannot update status for missing intent {intent_id}");
        }
        Ok(())
    }

    fn record_order_submission(&self, submission: &OrderSubmission<'_>) -> Result<()> {
        let mut conn = self.open()?;
        let tx = conn
            .transaction()
            .context("failed to start OMS submission transaction")?;
        let changed = if let Some(exchange_order_id) = submission.exchange_order_id {
            tx.execute(
                "
                UPDATE oms_intents
                SET status = ?, sent_ts_ms = ?, exchange_order_id = ?, last_error = ?
                WHERE intent_id = ?
                ",
                params![
                    submission.status,
                    submission.sent_ts_ms,
                    exchange_order_id,
                    submission.last_error,
                    submission.intent_id
                ],
            )?
        } else {
            tx.execute(
                "
                UPDATE oms_intents
                SET status = ?, sent_ts_ms = ?, last_error = ?
                WHERE intent_id = ?
                ",
                params![
                    submission.status,
                    submission.sent_ts_ms,
                    submission.last_error,
                    submission.intent_id
                ],
            )?
        };

        if changed == 0 {
            return Err(anyhow!(
                "cannot record OMS submission for missing intent {}",
                submission.intent_id
            ));
        }

        tx.execute(
            "
            INSERT INTO oms_orders (
                intent_id, created_ts_ms, symbol, side, order_type, requested_size,
                reduce_only, client_order_id, exchange_order_id, status, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ",
            params![
                submission.intent_id,
                submission.sent_ts_ms,
                submission.symbol,
                submission.side,
                submission.order_type,
                submission.requested_size,
                if submission.reduce_only { 1 } else { 0 },
                submission.client_order_id,
                submission.exchange_order_id,
                submission.status,
                submission.raw_json,
            ],
        )
        .context("failed to insert OMS order")?;

        tx.commit()
            .context("failed to commit OMS submission transaction")
    }

    fn insert_fill(&self, fill: &InsertFillRequest<'_>) -> Result<bool> {
        let conn = self.open()?;
        let raw_json = serialise_json(fill.raw)?;
        let changed = conn
            .execute(
                "
                INSERT OR IGNORE INTO oms_fills (
                    ts_ms, symbol, intent_id, order_id, action, side, pos_type,
                    price, size, notional, fee_usd, fee_token, fee_rate, pnl_usd,
                    fill_hash, fill_tid, matched_via, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ",
                params![
                    fill.ts_ms,
                    require_upper("symbol", fill.symbol)?,
                    normalise_optional_text(fill.intent_id),
                    fill.order_id,
                    fill.action.map(normalise_upper_owned).transpose()?,
                    fill.side.map(normalise_upper_owned).transpose()?,
                    fill.pos_type.map(normalise_upper_owned).transpose()?,
                    fill.price,
                    fill.size,
                    fill.notional,
                    fill.fee_usd,
                    normalise_optional_text(fill.fee_token),
                    fill.fee_rate,
                    fill.pnl_usd,
                    normalise_optional_text(fill.fill_hash),
                    fill.fill_tid,
                    normalise_optional_text(fill.matched_via),
                    raw_json,
                ],
            )
            .context("failed to insert OMS fill")?;
        Ok(changed > 0)
    }

    fn find_intent_by_client_order_id(&self, client_order_id: &str) -> Result<Option<String>> {
        if client_order_id.trim().is_empty() {
            return Ok(None);
        }
        let conn = self.open()?;
        conn.query_row(
            "
            SELECT intent_id
            FROM oms_intents
            WHERE client_order_id = ?
            ORDER BY created_ts_ms DESC
            LIMIT 1
            ",
            params![client_order_id],
            |row| row.get(0),
        )
        .optional()
        .context("failed to query OMS intent by client_order_id")
    }

    fn find_intent_by_exchange_order_id(&self, exchange_order_id: &str) -> Result<Option<String>> {
        if exchange_order_id.trim().is_empty() {
            return Ok(None);
        }
        let conn = self.open()?;
        conn.query_row(
            "
            SELECT intent_id
            FROM oms_intents
            WHERE exchange_order_id = ?
            ORDER BY created_ts_ms DESC
            LIMIT 1
            ",
            params![exchange_order_id],
            |row| row.get(0),
        )
        .optional()
        .context("failed to query OMS intent by exchange_order_id")
    }

    fn find_intent_by_fill_hash(&self, fill_hash: &str) -> Result<Option<String>> {
        if fill_hash.trim().is_empty() {
            return Ok(None);
        }
        let conn = self.open()?;
        conn.query_row(
            "
            SELECT intent_id
            FROM oms_fills
            WHERE fill_hash = ?
              AND intent_id IS NOT NULL
            ORDER BY ts_ms DESC
            LIMIT 1
            ",
            params![fill_hash],
            |row| row.get(0),
        )
        .optional()
        .context("failed to query OMS fill hash siblings")
    }

    fn find_pending_intent(
        &self,
        symbol: &str,
        action: &str,
        side: &str,
        ts_ms: i64,
        ttl_ms: i64,
    ) -> Result<Option<String>> {
        let ttl_ms = ttl_ms.max(0);
        let start = ts_ms.saturating_sub(ttl_ms);
        let end = ts_ms.saturating_add(ttl_ms);
        let conn = self.open()?;
        conn.query_row(
            "
            SELECT intent_id
            FROM oms_intents
            WHERE symbol = ?
              AND action = ?
              AND side = ?
              AND status IN ('SENT', 'PARTIAL', 'UNKNOWN')
              AND sent_ts_ms IS NOT NULL
              AND sent_ts_ms BETWEEN ? AND ?
            ORDER BY ABS(sent_ts_ms - ?) ASC
            LIMIT 1
            ",
            params![symbol, action, side, start, end, ts_ms],
            |row| row.get(0),
        )
        .optional()
        .context("failed to query pending OMS intent by time proximity")
    }
}

#[derive(Debug, Clone)]
struct IntentInsert<'a> {
    intent_id: &'a str,
    client_order_id: Option<&'a str>,
    created_ts_ms: i64,
    symbol: &'a str,
    action: &'a str,
    side: &'a str,
    requested_size: Option<f64>,
    requested_notional: Option<f64>,
    entry_atr: Option<f64>,
    leverage: Option<f64>,
    decision_ts_ms: Option<i64>,
    strategy_version: Option<&'a str>,
    strategy_sha1: Option<&'a str>,
    reason: Option<&'a str>,
    confidence: Option<&'a str>,
    status: &'a str,
    dedupe_key: Option<&'a str>,
    meta_json: Option<&'a str>,
}

#[derive(Debug, Clone)]
struct OrderSubmission<'a> {
    intent_id: &'a str,
    status: &'a str,
    sent_ts_ms: i64,
    symbol: &'a str,
    side: &'a str,
    order_type: &'a str,
    requested_size: Option<f64>,
    reduce_only: bool,
    client_order_id: Option<&'a str>,
    exchange_order_id: Option<&'a str>,
    last_error: &'a str,
    raw_json: Option<&'a str>,
}

fn now_ms() -> i64 {
    Utc::now().timestamp_millis()
}

fn make_intent_id() -> String {
    let counter = INTENT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = Utc::now()
        .timestamp_nanos_opt()
        .unwrap_or_else(|| now_ms().saturating_mul(1_000_000));
    let mut hasher = Sha3_256::new();
    hasher.update(nanos.to_le_bytes());
    hasher.update(std::process::id().to_le_bytes());
    hasher.update(counter.to_le_bytes());
    let digest = hasher.finalize();
    hex::encode(&digest[..16])
}

fn normalise_prefix(prefix: &str) -> String {
    let prefix = prefix.trim();
    if prefix.is_empty() {
        "aiq_".to_string()
    } else {
        prefix.to_string()
    }
}

fn make_hl_cloid(seed_hex: &str, prefix: &str) -> String {
    let mut seed = match hex::decode(seed_hex.trim()) {
        Ok(seed) => seed,
        Err(_) => {
            let digest = Sha3_256::digest(seed_hex.trim().as_bytes());
            digest[..16].to_vec()
        }
    };
    if seed.len() < 16 {
        seed.resize(16, 0);
    } else if seed.len() > 16 {
        seed.truncate(16);
    }

    let mut prefix_bytes = decode_prefix_bytes(prefix);
    if prefix_bytes.len() > 8 {
        prefix_bytes.truncate(8);
    }
    let need = 16usize.saturating_sub(prefix_bytes.len());
    let mut out = prefix_bytes;
    out.extend_from_slice(&seed[..need]);
    if out.len() < 16 {
        out.resize(16, 0);
    }
    format!("0x{}", hex::encode(out))
}

fn decode_prefix_bytes(prefix: &str) -> Vec<u8> {
    let trimmed = prefix.trim();
    if let Some(hex_prefix) = trimmed.strip_prefix("0x") {
        if hex_prefix.len() % 2 == 0 {
            if let Ok(bytes) = hex::decode(hex_prefix) {
                if !bytes.is_empty() {
                    return bytes;
                }
            }
        }
    }
    trimmed
        .bytes()
        .filter(|byte| byte.is_ascii())
        .collect::<Vec<_>>()
}

fn is_valid_hl_cloid(cloid: &str) -> bool {
    let cloid = cloid.trim();
    cloid.starts_with("0x")
        && cloid.len() == 34
        && hex::decode(&cloid[2..])
            .map(|bytes| bytes.len() == 16)
            .unwrap_or(false)
}

fn serialise_json(value: Option<&Value>) -> Result<Option<String>> {
    match value {
        Some(value) => serde_json::to_string(value)
            .map(Some)
            .context("failed to serialise OMS JSON payload"),
        None => Ok(None),
    }
}

fn require_trimmed(name: &str, value: &str) -> Result<String> {
    let value = value.trim();
    if value.is_empty() {
        bail!("{name} cannot be empty");
    }
    Ok(value.to_string())
}

fn require_upper(name: &str, value: &str) -> Result<String> {
    require_trimmed(name, value).map(|value| value.to_ascii_uppercase())
}

fn normalise_upper_owned(value: &str) -> Result<String> {
    require_upper("value", value)
}

fn normalise_optional_text(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn json_to_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(boolean) => Some(boolean.to_string()),
        _ => None,
    }
}

fn extract_optional_text(fill: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| fill.get(*key))
        .and_then(json_to_text)
}

fn extract_fill_exchange_order_id(fill: &Value) -> Option<String> {
    extract_optional_text(fill, &["oid", "orderId", "order_id", "id"])
}

fn extract_exchange_order_id_from_result(result: &Value) -> Option<String> {
    if let Some(exchange_order_id) =
        extract_optional_text(result, &["oid", "orderId", "order_id", "id"])
    {
        return Some(exchange_order_id);
    }

    let statuses = result
        .get("response")
        .and_then(|value| value.get("data"))
        .and_then(|value| value.get("statuses"))
        .and_then(Value::as_array)?;

    for status in statuses {
        for key in ["filled", "resting"] {
            let exchange_order_id = status
                .get(key)
                .and_then(|payload| payload.get("oid"))
                .and_then(json_to_text);
            if exchange_order_id.is_some() {
                return exchange_order_id;
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::OptionalExtension;
    use tempfile::tempdir;

    fn new_oms(match_ttl_ms: i64) -> (tempfile::TempDir, PathBuf, LiveOms) {
        let dir = tempdir().expect("tempdir should exist");
        let db_path = dir.path().join("oms.db");
        let oms = LiveOms::with_options(
            &db_path,
            LiveOmsOptions {
                match_ttl_ms,
                db_timeout_ms: DEFAULT_DB_TIMEOUT_MS,
                cloid_prefix: "aiq_".to_string(),
            },
        )
        .expect("OMS should initialise");
        (dir, db_path, oms)
    }

    #[test]
    fn ensure_creates_core_schema() {
        let (_dir, db_path, _oms) = new_oms(DEFAULT_MATCH_TTL_MS);
        let conn = Connection::open(db_path).expect("OMS db should open");
        let tables = ["oms_intents", "oms_orders", "oms_fills"];
        for table in tables {
            let exists: Option<String> = conn
                .query_row(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
                    params![table],
                    |row| row.get(0),
                )
                .optional()
                .expect("schema lookup should succeed");
            assert_eq!(exists.as_deref(), Some(table));
        }
        let index: Option<String> = conn
            .query_row(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'idx_oms_fills_hash_tid'",
                [],
                |row| row.get(0),
            )
            .optional()
            .expect("index lookup should succeed");
        assert_eq!(index.as_deref(), Some("idx_oms_fills_hash_tid"));
    }

    #[test]
    fn create_intent_generates_valid_hyperliquid_cloid() {
        let (_dir, _db_path, oms) = new_oms(DEFAULT_MATCH_TTL_MS);
        let intent = oms
            .create_intent(CreateIntentRequest {
                symbol: "BTC",
                action: "OPEN",
                side: "BUY",
                requested_size: Some(1.0),
                requested_notional: Some(100.0),
                leverage: Some(3.0),
                decision_ts_ms: Some(1_700_000_000_000),
                reason: Some("test"),
                confidence: Some("high"),
                entry_atr: Some(1.0),
                meta: None,
                dedupe_open: true,
                strategy_version: Some("test"),
                strategy_sha1: None,
            })
            .expect("intent creation should succeed");

        let cloid = intent
            .client_order_id
            .expect("client_order_id should exist");
        assert!(is_valid_hl_cloid(&cloid));
        let bytes = hex::decode(&cloid[2..]).expect("cloid should be hex");
        assert!(bytes.starts_with(b"aiq_"));
        assert!(!intent.duplicate);
    }

    #[test]
    fn create_intent_dedupe_upgrades_invalid_client_order_id() {
        let (_dir, db_path, oms) = new_oms(DEFAULT_MATCH_TTL_MS);
        let existing_id = "0123456789abcdeffedcba9876543210";
        let dedupe_key = "OPEN:BTC:BUY:1700000000000";
        let conn = Connection::open(&db_path).expect("OMS db should open");
        conn.execute(
            "
            INSERT INTO oms_intents (
                intent_id, created_ts_ms, symbol, action, side,
                requested_size, requested_notional, leverage,
                reason, confidence, status, dedupe_key, client_order_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ",
            params![
                existing_id,
                1_700_000_000_000i64,
                "BTC",
                "OPEN",
                "BUY",
                1.0f64,
                100.0f64,
                3.0f64,
                "test",
                "high",
                "NEW",
                dedupe_key,
                "aiq_not_hex"
            ],
        )
        .expect("seed intent should insert");

        let intent = oms
            .create_intent(CreateIntentRequest {
                symbol: "BTC",
                action: "OPEN",
                side: "BUY",
                requested_size: Some(1.0),
                requested_notional: Some(100.0),
                leverage: Some(3.0),
                decision_ts_ms: Some(1_700_000_000_000),
                reason: Some("test"),
                confidence: Some("high"),
                entry_atr: None,
                meta: None,
                dedupe_open: true,
                strategy_version: None,
                strategy_sha1: None,
            })
            .expect("dedupe lookup should succeed");

        assert!(intent.duplicate);
        assert_eq!(intent.intent_id, existing_id);
        assert!(intent
            .client_order_id
            .as_deref()
            .map(is_valid_hl_cloid)
            .unwrap_or(false));

        let updated: Option<String> = conn
            .query_row(
                "SELECT client_order_id FROM oms_intents WHERE intent_id = ?",
                params![existing_id],
                |row| row.get(0),
            )
            .optional()
            .expect("updated cloid should read");
        assert_eq!(updated, intent.client_order_id);
    }

    #[test]
    fn mark_would_and_failed_update_intent_status() {
        let (_dir, db_path, oms) = new_oms(DEFAULT_MATCH_TTL_MS);

        let would_intent = oms
            .create_intent(CreateIntentRequest {
                symbol: "BTC",
                action: "ADD",
                side: "BUY",
                requested_size: Some(1.0),
                requested_notional: Some(100.0),
                leverage: Some(3.0),
                decision_ts_ms: Some(1_700_000_000_000),
                reason: Some("would"),
                confidence: Some("high"),
                entry_atr: None,
                meta: None,
                dedupe_open: false,
                strategy_version: None,
                strategy_sha1: None,
            })
            .expect("intent should insert");
        oms.mark_would(&would_intent, Some("dry_live"))
            .expect("mark_would should succeed");

        let failed_intent = oms
            .create_intent(CreateIntentRequest {
                symbol: "ETH",
                action: "OPEN",
                side: "SELL",
                requested_size: Some(2.0),
                requested_notional: Some(200.0),
                leverage: Some(2.0),
                decision_ts_ms: Some(1_700_000_100_000),
                reason: Some("failed"),
                confidence: Some("medium"),
                entry_atr: None,
                meta: None,
                dedupe_open: false,
                strategy_version: None,
                strategy_sha1: None,
            })
            .expect("intent should insert");
        oms.mark_failed(&failed_intent, "market_open rejected")
            .expect("mark_failed should succeed");

        let conn = Connection::open(db_path).expect("OMS db should open");
        let would_row: (String, String) = conn
            .query_row(
                "SELECT status, last_error FROM oms_intents WHERE intent_id = ?",
                params![would_intent.intent_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("WOULD row should exist");
        assert_eq!(would_row.0, "WOULD");
        assert_eq!(would_row.1, "dry_live");

        let failed_row: (String, String) = conn
            .query_row(
                "SELECT status, last_error FROM oms_intents WHERE intent_id = ?",
                params![failed_intent.intent_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("REJECTED row should exist");
        assert_eq!(failed_row.0, "REJECTED");
        assert_eq!(failed_row.1, "market_open rejected");
    }

    #[test]
    fn mark_submit_unknown_records_order_and_enables_time_proximity_matching() {
        let (_dir, db_path, oms) = new_oms(60_000);
        let intent = oms
            .create_intent(CreateIntentRequest {
                symbol: "BTC",
                action: "ADD",
                side: "BUY",
                requested_size: Some(1.0),
                requested_notional: Some(100.0),
                leverage: Some(3.0),
                decision_ts_ms: Some(1_700_000_000_000),
                reason: Some("test"),
                confidence: Some("high"),
                entry_atr: None,
                meta: None,
                dedupe_open: false,
                strategy_version: None,
                strategy_sha1: None,
            })
            .expect("intent should insert");

        oms.mark_submit_unknown(
            &intent,
            SubmitUnknownRequest {
                symbol: "BTC",
                side: "BUY",
                order_type: "market_open",
                reduce_only: false,
                requested_size: Some(1.0),
                error: Some("timeout"),
            },
        )
        .expect("mark_submit_unknown should succeed");

        let conn = Connection::open(&db_path).expect("OMS db should open");
        let row: (String, i64, String) = conn
            .query_row(
                "SELECT status, sent_ts_ms, last_error FROM oms_intents WHERE intent_id = ?",
                params![intent.intent_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("intent should exist");
        assert_eq!(row.0, "UNKNOWN");
        assert!(row.1 > 0);
        assert_eq!(row.2, "timeout");

        let order_row: (String, Option<String>) = conn
            .query_row(
                "SELECT status, exchange_order_id FROM oms_orders WHERE intent_id = ? ORDER BY id DESC LIMIT 1",
                params![intent.intent_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("order row should exist");
        assert_eq!(order_row.0, "UNKNOWN");
        assert_eq!(order_row.1, None);

        let matched = oms
            .match_intent_for_fill(
                &json!({
                    "coin": "BTC",
                    "time": row.1,
                }),
                "BTC",
                "ADD",
                "BUY",
                row.1,
            )
            .expect("matching should succeed")
            .expect("time proximity should match");
        assert_eq!(matched.intent_id, intent.intent_id);
        assert_eq!(matched.matched_via, "time_proximity");
    }

    #[test]
    fn mark_sent_extracts_exchange_order_id_and_fill_helpers_dedupe() {
        let (_dir, db_path, oms) = new_oms(DEFAULT_MATCH_TTL_MS);
        let sent_intent = oms
            .create_intent(CreateIntentRequest {
                symbol: "BTC",
                action: "OPEN",
                side: "BUY",
                requested_size: Some(1.0),
                requested_notional: Some(100.0),
                leverage: Some(3.0),
                decision_ts_ms: Some(1_700_000_000_000),
                reason: Some("sent"),
                confidence: Some("high"),
                entry_atr: None,
                meta: None,
                dedupe_open: false,
                strategy_version: None,
                strategy_sha1: None,
            })
            .expect("intent should insert");

        oms.mark_sent(
            &sent_intent,
            SentOrderRequest {
                symbol: "BTC",
                side: "BUY",
                order_type: "market_open",
                reduce_only: false,
                requested_size: Some(1.0),
                result: Some(&json!({
                    "status": "ok",
                    "response": {
                        "type": "order",
                        "data": {
                            "statuses": [
                                {"resting": {"oid": 9001}}
                            ]
                        }
                    }
                })),
                exchange_order_id: None,
            },
        )
        .expect("mark_sent should succeed");

        let conn = Connection::open(&db_path).expect("OMS db should open");
        let sent_row: (String, String, String) = conn
            .query_row(
                "SELECT status, exchange_order_id, last_error FROM oms_intents WHERE intent_id = ?",
                params![sent_intent.intent_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("sent row should exist");
        assert_eq!(sent_row.0, "SENT");
        assert_eq!(sent_row.1, "9001");
        assert!(sent_row.2.is_empty());

        let exchange_match = oms
            .match_intent_for_fill(
                &json!({
                    "oid": "9001",
                    "hash": "0xfill_a",
                    "time": 1_700_000_001_000i64,
                }),
                "BTC",
                "OPEN",
                "BUY",
                1_700_000_001_000i64,
            )
            .expect("exchange matching should succeed")
            .expect("exchange order id should match");
        assert_eq!(exchange_match.intent_id, sent_intent.intent_id);
        assert_eq!(exchange_match.matched_via, "exchange_order_id");

        let inserted = oms
            .insert_fill(InsertFillRequest {
                ts_ms: 1_700_000_001_000,
                symbol: "BTC",
                intent_id: Some(&sent_intent.intent_id),
                order_id: None,
                action: Some("OPEN"),
                side: Some("BUY"),
                pos_type: Some("LONG"),
                price: 100.0,
                size: 1.0,
                notional: 100.0,
                fee_usd: Some(0.1),
                fee_token: Some("USDC"),
                fee_rate: Some(0.001),
                pnl_usd: Some(0.0),
                fill_hash: Some("0xsibling"),
                fill_tid: Some(42),
                matched_via: Some("exchange_order_id"),
                raw: Some(&json!({"oid": "9001"})),
            })
            .expect("first fill insert should succeed");
        assert!(inserted);

        let deduped = oms
            .insert_fill(InsertFillRequest {
                ts_ms: 1_700_000_001_000,
                symbol: "BTC",
                intent_id: Some(&sent_intent.intent_id),
                order_id: None,
                action: Some("OPEN"),
                side: Some("BUY"),
                pos_type: Some("LONG"),
                price: 100.0,
                size: 1.0,
                notional: 100.0,
                fee_usd: Some(0.1),
                fee_token: Some("USDC"),
                fee_rate: Some(0.001),
                pnl_usd: Some(0.0),
                fill_hash: Some("0xsibling"),
                fill_tid: Some(42),
                matched_via: Some("exchange_order_id"),
                raw: Some(&json!({"oid": "9001"})),
            })
            .expect("duplicate fill insert should succeed");
        assert!(!deduped);

        let sibling_match = oms
            .match_intent_for_fill(
                &json!({
                    "hash": "0xsibling",
                    "time": 1_700_000_001_500i64,
                }),
                "BTC",
                "OPEN",
                "BUY",
                1_700_000_001_500i64,
            )
            .expect("fill hash matching should succeed")
            .expect("fill hash sibling should match");
        assert_eq!(sibling_match.intent_id, sent_intent.intent_id);
        assert_eq!(sibling_match.matched_via, "fill_hash_sibling");

        let by_client_order = oms
            .create_intent(CreateIntentRequest {
                symbol: "ETH",
                action: "OPEN",
                side: "SELL",
                requested_size: Some(2.0),
                requested_notional: Some(200.0),
                leverage: Some(2.0),
                decision_ts_ms: Some(1_700_000_010_000),
                reason: Some("client"),
                confidence: Some("medium"),
                entry_atr: None,
                meta: None,
                dedupe_open: false,
                strategy_version: None,
                strategy_sha1: None,
            })
            .expect("client intent should insert");
        let client_match = oms
            .match_intent_for_fill(
                &json!({
                    "client_order_id": by_client_order.client_order_id.clone().expect("cloid should exist"),
                    "time": 1_700_000_011_000i64,
                }),
                "ETH",
                "OPEN",
                "SELL",
                1_700_000_011_000i64,
            )
            .expect("client order matching should succeed")
            .expect("client order id should match");
        assert_eq!(client_match.intent_id, by_client_order.intent_id);
        assert_eq!(client_match.matched_via, "client_order_id");
    }
}
