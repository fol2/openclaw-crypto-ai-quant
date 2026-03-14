use anyhow::{bail, Context, Result};
use chrono::Utc;
use rusqlite::{backup::Backup, params, Connection, OptionalExtension};
use serde::Serialize;
use serde_json::{json, Value};
use sha3::{Digest, Sha3_256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::live_hyperliquid::{HyperliquidClient, HyperliquidFill};
use crate::live_oms::{LiveOms, LiveOmsOptions};
use crate::live_secrets::load_live_secrets;
use crate::live_state::sync_exchange_positions;

const DEFAULT_CURSOR_KEY: &str = "hyperliquid_fill_sync_v1";
const DEFAULT_DB_TIMEOUT_MS: u64 = 1_000;
const SYNC_RUN_STATUS_STARTED: &str = "started";
const SYNC_RUN_STATUS_SUCCESS: &str = "success";
const SYNC_RUN_STATUS_UNSUPPORTED_REMOTE_FILLS: &str = "unsupported_remote_fills";
const SYNC_RUN_STATUS_FAILED: &str = "failed";
const LIVE_FILL_SYNC_SOURCE: &str = "live_fill_sync";
const LIVE_FILL_SYNC_DRY_RUN_SOURCE: &str = "live_fill_sync_dry_run";
const CLEARINGHOUSE_STATE_REQUEST_TYPE: &str = "clearinghouseState";

pub struct LiveFillSyncInput<'a> {
    pub live_db: &'a Path,
    pub secrets_path: &'a Path,
    pub profile: Option<&'a str>,
    pub config_path: Option<&'a Path>,
    pub config_id: Option<&'a str>,
    pub start_ms: Option<i64>,
    pub end_ms: Option<i64>,
    pub lookback_hours: i64,
    pub overlap_minutes: i64,
    pub cursor_key: Option<&'a str>,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveFillSyncReport {
    pub sync_run_id: i64,
    pub sync_run_status: String,
    pub ok: bool,
    pub dry_run: bool,
    pub live_db: String,
    pub working_db: String,
    pub user: String,
    pub account_value_usd: f64,
    pub withdrawable_usd: f64,
    pub total_margin_used_usd: f64,
    pub exchange_position_count: usize,
    pub window: SyncWindowReport,
    pub fetched_remote_fills: usize,
    pub supported_remote_fills: usize,
    pub unsupported_remote_fills: usize,
    pub inserted_oms_fills: usize,
    pub linked_existing_oms_fills: usize,
    pub inserted_trades: usize,
    pub backfilled_existing_trades: usize,
    pub inserted_manual_intents: usize,
    pub relabelled_manual_trades: usize,
    pub matched_by: BTreeMap<String, usize>,
    pub fills_by_symbol: BTreeMap<String, usize>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SyncWindowReport {
    pub start_ms: i64,
    pub end_ms: i64,
    pub source: String,
    pub cursor_key: Option<String>,
}

#[derive(Debug, Default)]
struct SyncStats {
    fetched_remote_fills: usize,
    supported_remote_fills: usize,
    unsupported_remote_fills: usize,
    inserted_oms_fills: usize,
    linked_existing_oms_fills: usize,
    inserted_trades: usize,
    backfilled_existing_trades: usize,
    inserted_manual_intents: usize,
    relabelled_manual_trades: usize,
    matched_by: BTreeMap<String, usize>,
    fills_by_symbol: BTreeMap<String, usize>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct WorkingLiveDb {
    path: PathBuf,
    cleanup: bool,
}

#[derive(Debug, Clone)]
struct SyncWindow {
    start_ms: i64,
    end_ms: i64,
    source: String,
    cursor_key: Option<String>,
}

#[derive(Debug, Clone)]
struct SyncRunProgress {
    run_id: i64,
    wallet_address: Option<String>,
    resolved_window: Option<SyncWindow>,
    account_value_usd: Option<f64>,
    withdrawable_usd: Option<f64>,
    total_margin_used_usd: Option<f64>,
    exchange_position_count: Option<usize>,
}

#[derive(Debug, Clone)]
struct ParsedFill {
    timestamp: String,
    ts_ms: i64,
    symbol: String,
    action: String,
    side: String,
    pos_type: String,
    price: f64,
    size: f64,
    notional_usd: f64,
    pnl_usd: f64,
    fee_usd: f64,
    fee_token: Option<String>,
    fee_rate: Option<f64>,
    fill_hash: Option<String>,
    fill_tid: Option<i64>,
    exchange_order_id: Option<String>,
    cloid: Option<String>,
    dir: String,
}

#[derive(Debug, Clone)]
struct IntentSnapshot {
    intent_id: String,
    reason: Option<String>,
    reason_code: Option<String>,
    confidence: Option<String>,
    leverage: Option<f64>,
    entry_atr: Option<f64>,
    client_order_id: Option<String>,
    exchange_order_id: Option<String>,
}

#[derive(Debug, Clone)]
struct FillResolution {
    intent: Option<IntentSnapshot>,
    matched_via: String,
    is_manual_trade: bool,
}

#[derive(Debug, Clone)]
struct ExistingOmsFill {
    id: i64,
    intent_id: Option<String>,
}

#[derive(Debug, Clone)]
struct ExistingOmsFillWithoutTrade {
    id: i64,
    intent_id: Option<String>,
    matched_via: Option<String>,
    raw_json: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CloidKind {
    None,
    Manual,
    Aiq,
    Other,
}

pub fn run_sync(input: LiveFillSyncInput<'_>) -> Result<LiveFillSyncReport> {
    let working_db = prepare_working_live_db(input.live_db, input.dry_run)?;
    let source_live_db = input.live_db.to_path_buf();
    let sync_result = run_sync_inner(&working_db.path, input, &source_live_db);
    if working_db.cleanup {
        let _ = fs::remove_file(&working_db.path);
    }
    sync_result
}

fn run_sync_inner(
    working_db_path: &Path,
    input: LiveFillSyncInput<'_>,
    source_live_db: &Path,
) -> Result<LiveFillSyncReport> {
    ensure_sync_schema(working_db_path)?;
    let cursor_key = input
        .cursor_key
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_CURSOR_KEY);
    let run_id = insert_sync_run_start(
        working_db_path,
        &input,
        source_live_db,
        working_db_path,
        cursor_key,
    )?;
    let mut stats = SyncStats::default();
    let mut progress = SyncRunProgress {
        run_id,
        wallet_address: None,
        resolved_window: None,
        account_value_usd: None,
        withdrawable_usd: None,
        total_margin_used_usd: None,
        exchange_position_count: None,
    };
    let sync_result: Result<LiveFillSyncReport> = (|| {
        let secrets = load_live_secrets(input.secrets_path)?;
        progress.wallet_address = Some(secrets.main_address.clone());
        let client = HyperliquidClient::new(&secrets, None)?;
        let account_observation = client.clearinghouse_state_observation()?;
        let account_snapshot = &account_observation.account_snapshot;
        progress.account_value_usd = Some(account_snapshot.account_value_usd);
        progress.withdrawable_usd = Some(account_snapshot.withdrawable_usd);
        progress.total_margin_used_usd = Some(account_snapshot.total_margin_used_usd);
        let exchange_positions = &account_observation.positions;
        progress.exchange_position_count = Some(exchange_positions.len());
        let account_snapshot_event_id = persist_account_snapshot_event(
            working_db_path,
            &account_observation,
            run_id,
            &secrets.main_address,
            input.dry_run,
        )?;
        let window = resolve_sync_window(
            working_db_path,
            input.start_ms,
            input.end_ms,
            input.lookback_hours,
            input.overlap_minutes,
            cursor_key,
        )?;
        progress.resolved_window = Some(window.clone());

        let options = LiveOmsOptions::default();
        let aiq_cloid_prefix = options.cloid_prefix.clone();
        let oms = LiveOms::with_options(working_db_path, options)?;
        let conn = open_sync_connection(working_db_path)?;
        ensure_sync_run_columns(&conn)?;
        drop(conn);

        let remote_fills = client
            .user_fills_by_time(window.start_ms, window.end_ms)
            .context("failed to fetch Hyperliquid user fills")?;

        stats.fetched_remote_fills = remote_fills.len();
        reconcile_remote_fills(
            working_db_path,
            &oms,
            &remote_fills,
            &aiq_cloid_prefix,
            run_id,
            &mut stats,
        )?;
        backfill_existing_oms_trades(working_db_path, &oms, &aiq_cloid_prefix, run_id, &mut stats)?;
        relabel_existing_manual_trades(working_db_path, &aiq_cloid_prefix, run_id, &mut stats)?;

        let ok = stats.unsupported_remote_fills == 0;
        let sync_run_status = if ok {
            SYNC_RUN_STATUS_SUCCESS
        } else {
            SYNC_RUN_STATUS_UNSUPPORTED_REMOTE_FILLS
        };
        if ok {
            persist_exchange_snapshot_projection(
                working_db_path,
                &account_observation,
                account_snapshot_event_id,
                run_id,
                input.dry_run,
            )?;
            if !input.dry_run {
                write_sync_cursor(
                    working_db_path,
                    cursor_key,
                    window.start_ms,
                    window.end_ms,
                    run_id,
                )?;
            }
        }
        finalize_sync_run(working_db_path, &progress, &stats, sync_run_status, None)?;
        Ok(LiveFillSyncReport {
            sync_run_id: run_id,
            sync_run_status: sync_run_status.to_string(),
            ok,
            dry_run: input.dry_run,
            live_db: source_live_db.display().to_string(),
            working_db: working_db_path.display().to_string(),
            user: secrets.main_address,
            account_value_usd: account_snapshot.account_value_usd,
            withdrawable_usd: account_snapshot.withdrawable_usd,
            total_margin_used_usd: account_snapshot.total_margin_used_usd,
            exchange_position_count: exchange_positions.len(),
            window: SyncWindowReport {
                start_ms: window.start_ms,
                end_ms: window.end_ms,
                source: window.source,
                cursor_key: window.cursor_key,
            },
            fetched_remote_fills: stats.fetched_remote_fills,
            supported_remote_fills: stats.supported_remote_fills,
            unsupported_remote_fills: stats.unsupported_remote_fills,
            inserted_oms_fills: stats.inserted_oms_fills,
            linked_existing_oms_fills: stats.linked_existing_oms_fills,
            inserted_trades: stats.inserted_trades,
            backfilled_existing_trades: stats.backfilled_existing_trades,
            inserted_manual_intents: stats.inserted_manual_intents,
            relabelled_manual_trades: stats.relabelled_manual_trades,
            matched_by: stats.matched_by.clone(),
            fills_by_symbol: stats.fills_by_symbol.clone(),
            warnings: stats.warnings.clone(),
        })
    })();

    match sync_result {
        Ok(report) => Ok(report),
        Err(error) => {
            if let Err(finalize_error) = finalize_sync_run(
                working_db_path,
                &progress,
                &stats,
                SYNC_RUN_STATUS_FAILED,
                Some(&error.to_string()),
            ) {
                return Err(error.context(format!(
                    "failed to finalise exchange sync run {}: {finalize_error}",
                    progress.run_id
                )));
            }
            Err(error)
        }
    }
}

fn prepare_working_live_db(live_db: &Path, dry_run: bool) -> Result<WorkingLiveDb> {
    if !dry_run {
        return Ok(WorkingLiveDb {
            path: live_db.to_path_buf(),
            cleanup: false,
        });
    }

    let temp_path = std::env::temp_dir().join(format!(
        "aiq-runtime-live-sync-{}-{}.db",
        std::process::id(),
        Utc::now()
            .timestamp_nanos_opt()
            .unwrap_or_else(|| Utc::now().timestamp_micros() * 1_000)
    ));
    let source = Connection::open(live_db).with_context(|| {
        format!(
            "failed to open source live db for dry-run: {}",
            live_db.display()
        )
    })?;
    let mut dest = Connection::open(&temp_path).with_context(|| {
        format!(
            "failed to create temporary dry-run live db: {}",
            temp_path.display()
        )
    })?;
    {
        let backup = Backup::new(&source, &mut dest).with_context(|| {
            format!(
                "failed to initialise dry-run backup from {} to {}",
                live_db.display(),
                temp_path.display()
            )
        })?;
        backup.step(-1).with_context(|| {
            format!(
                "failed to copy live db into temporary dry-run db {}",
                temp_path.display()
            )
        })?;
    }

    Ok(WorkingLiveDb {
        path: temp_path,
        cleanup: true,
    })
}

fn ensure_sync_schema(db_path: &Path) -> Result<()> {
    let conn = open_sync_connection(db_path)?;
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            type TEXT,
            action TEXT,
            price REAL,
            size REAL,
            notional REAL,
            reason TEXT,
            reason_code TEXT,
            confidence TEXT,
            pnl REAL,
            fee_usd REAL,
            fee_token TEXT,
            fee_rate REAL,
            balance REAL,
            entry_atr REAL,
            leverage REAL,
            margin_used REAL,
            meta_json TEXT,
            run_fingerprint TEXT,
            fill_hash TEXT,
            fill_tid INTEGER,
            sync_run_id INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_trades_fill_lookup
            ON trades(fill_hash, fill_tid);
        CREATE TABLE IF NOT EXISTS runtime_sync_cursors (
            sync_key TEXT PRIMARY KEY,
            last_start_ts_ms INTEGER NOT NULL,
            last_end_ts_ms INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            last_run_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS exchange_sync_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at_ts_ms INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            finished_at_ts_ms INTEGER,
            finished_at TEXT,
            status TEXT NOT NULL,
            error_text TEXT,
            dry_run INTEGER NOT NULL,
            wallet_address TEXT,
            profile TEXT,
            config_path TEXT,
            config_id TEXT,
            source_live_db TEXT NOT NULL,
            working_db TEXT NOT NULL,
            requested_start_ms INTEGER,
            requested_end_ms INTEGER,
            resolved_start_ms INTEGER,
            resolved_end_ms INTEGER,
            window_source TEXT,
            cursor_key TEXT,
            fetched_remote_fills INTEGER NOT NULL DEFAULT 0,
            supported_remote_fills INTEGER NOT NULL DEFAULT 0,
            unsupported_remote_fills INTEGER NOT NULL DEFAULT 0,
            inserted_oms_fills INTEGER NOT NULL DEFAULT 0,
            linked_existing_oms_fills INTEGER NOT NULL DEFAULT 0,
            inserted_trades INTEGER NOT NULL DEFAULT 0,
            backfilled_existing_trades INTEGER NOT NULL DEFAULT 0,
            inserted_manual_intents INTEGER NOT NULL DEFAULT 0,
            relabelled_manual_trades INTEGER NOT NULL DEFAULT 0,
            exchange_position_count INTEGER NOT NULL DEFAULT 0,
            account_value_usd REAL,
            withdrawable_usd REAL,
            total_margin_used_usd REAL,
            warnings_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_exchange_sync_runs_started_at
            ON exchange_sync_runs(started_at_ts_ms DESC);
        CREATE INDEX IF NOT EXISTS idx_exchange_sync_runs_status_started_at
            ON exchange_sync_runs(status, started_at_ts_ms DESC);
        CREATE TABLE IF NOT EXISTS exchange_account_snapshot_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sync_run_id INTEGER NOT NULL,
            observed_at_ts_ms INTEGER NOT NULL,
            observed_at TEXT NOT NULL,
            source TEXT NOT NULL,
            request_type TEXT NOT NULL,
            wallet_address TEXT NOT NULL,
            raw_payload_json TEXT NOT NULL,
            payload_digest TEXT NOT NULL,
            account_value_usd REAL NOT NULL,
            withdrawable_usd REAL NOT NULL,
            total_margin_used_usd REAL NOT NULL,
            position_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runtime_account_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            account_value_usd REAL NOT NULL,
            withdrawable_usd REAL NOT NULL,
            total_margin_used_usd REAL NOT NULL,
            source TEXT NOT NULL,
            meta_json TEXT,
            sync_run_id INTEGER,
            account_snapshot_event_id INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_account_snapshots_ts_ms
            ON runtime_account_snapshots(ts_ms DESC);
        CREATE TABLE IF NOT EXISTS runtime_exchange_positions (
            symbol TEXT PRIMARY KEY,
            pos_type TEXT NOT NULL,
            size REAL NOT NULL,
            entry_price REAL NOT NULL,
            leverage REAL NOT NULL,
            margin_used REAL NOT NULL,
            ts_ms INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            source TEXT NOT NULL,
            sync_run_id INTEGER
        );
        ",
    )
    .context("failed to ensure live fill sync schema")?;
    ensure_sync_run_columns(&conn)?;
    conn.execute_batch(
        "
        CREATE INDEX IF NOT EXISTS idx_exchange_account_snapshot_events_sync_run_id
            ON exchange_account_snapshot_events(sync_run_id, observed_at_ts_ms DESC);
        CREATE INDEX IF NOT EXISTS idx_exchange_account_snapshot_events_payload_digest
            ON exchange_account_snapshot_events(payload_digest);
        CREATE INDEX IF NOT EXISTS idx_runtime_account_snapshots_account_snapshot_event_id
            ON runtime_account_snapshots(account_snapshot_event_id);
        ",
    )
    .context("failed to ensure live fill sync indexes")?;
    Ok(())
}

fn ensure_sync_run_columns(conn: &Connection) -> Result<()> {
    add_column_if_missing(conn, "runtime_sync_cursors", "last_run_id", "INTEGER")?;
    add_column_if_missing(conn, "runtime_account_snapshots", "sync_run_id", "INTEGER")?;
    add_column_if_missing(
        conn,
        "runtime_account_snapshots",
        "account_snapshot_event_id",
        "INTEGER",
    )?;
    add_column_if_missing(conn, "runtime_exchange_positions", "sync_run_id", "INTEGER")?;
    add_column_if_missing(conn, "trades", "sync_run_id", "INTEGER")?;
    add_column_if_missing(conn, "oms_intents", "sync_run_id", "INTEGER")?;
    add_column_if_missing(conn, "oms_fills", "sync_run_id", "INTEGER")?;
    Ok(())
}

fn add_column_if_missing(
    conn: &Connection,
    table_name: &str,
    column_name: &str,
    column_type: &str,
) -> Result<()> {
    let table_exists = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1 LIMIT 1",
            params![table_name],
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    if !table_exists || table_column_exists(conn, table_name, column_name)? {
        return Ok(());
    }
    conn.execute(
        &format!("ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"),
        [],
    )
    .with_context(|| format!("failed to add {column_name} to live fill sync table {table_name}"))?;
    Ok(())
}

fn table_column_exists(conn: &Connection, table_name: &str, column_name: &str) -> Result<bool> {
    let pragma = format!("PRAGMA table_info({table_name})");
    let mut stmt = conn.prepare(&pragma)?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for row in rows {
        if row?.eq_ignore_ascii_case(column_name) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn resolve_sync_window(
    db_path: &Path,
    start_ms: Option<i64>,
    end_ms: Option<i64>,
    lookback_hours: i64,
    overlap_minutes: i64,
    cursor_key: &str,
) -> Result<SyncWindow> {
    let end_ms = end_ms.unwrap_or_else(|| Utc::now().timestamp_millis());
    if end_ms <= 0 {
        bail!("sync end_ms must be positive");
    }

    if let Some(start_ms) = start_ms {
        if start_ms >= end_ms {
            bail!("sync start_ms must be earlier than end_ms");
        }
        return Ok(SyncWindow {
            start_ms,
            end_ms,
            source: "explicit".to_string(),
            cursor_key: Some(cursor_key.to_string()),
        });
    }

    let overlap_ms = overlap_minutes.max(0) * 60 * 1_000;
    let lookback_ms = lookback_hours.max(1) * 60 * 60 * 1_000;
    let conn = open_sync_connection(db_path)?;
    let cursor_end_ms = conn
        .query_row(
            "SELECT last_end_ts_ms FROM runtime_sync_cursors WHERE sync_key = ?1",
            params![cursor_key],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .context("failed to read sync cursor")?;

    let (start_ms, source) = if let Some(cursor_end_ms) = cursor_end_ms.filter(|value| *value > 0) {
        ((cursor_end_ms - overlap_ms).max(0), "cursor".to_string())
    } else {
        ((end_ms - lookback_ms).max(0), "lookback".to_string())
    };

    if start_ms >= end_ms {
        bail!("resolved sync start_ms must be earlier than end_ms");
    }

    Ok(SyncWindow {
        start_ms,
        end_ms,
        source,
        cursor_key: Some(cursor_key.to_string()),
    })
}

fn timestamp_from_ms(ts_ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ts_ms)
        .map(|value| value.to_rfc3339())
        .unwrap_or_else(|| Utc::now().to_rfc3339())
}

fn insert_sync_run_start(
    db_path: &Path,
    input: &LiveFillSyncInput<'_>,
    source_live_db: &Path,
    working_db_path: &Path,
    cursor_key: &str,
) -> Result<i64> {
    let started_at_ts_ms = Utc::now().timestamp_millis();
    let started_at = timestamp_from_ms(started_at_ts_ms);
    let conn = open_sync_connection(db_path)?;
    conn.execute(
        "
        INSERT INTO exchange_sync_runs (
            started_at_ts_ms, started_at, status, dry_run, profile, config_path, config_id,
            source_live_db, working_db, requested_start_ms, requested_end_ms, cursor_key
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7,
            ?8, ?9, ?10, ?11, ?12
        )
        ",
        params![
            started_at_ts_ms,
            started_at,
            SYNC_RUN_STATUS_STARTED,
            i64::from(input.dry_run),
            input.profile,
            input.config_path.map(|path| path.display().to_string()),
            input.config_id,
            source_live_db.display().to_string(),
            working_db_path.display().to_string(),
            input.start_ms,
            input.end_ms,
            cursor_key,
        ],
    )
    .context("failed to insert exchange sync run header")?;
    Ok(conn.last_insert_rowid())
}

fn finalize_sync_run(
    db_path: &Path,
    progress: &SyncRunProgress,
    stats: &SyncStats,
    status: &str,
    error_text: Option<&str>,
) -> Result<()> {
    let finished_at_ts_ms = Utc::now().timestamp_millis();
    let finished_at = timestamp_from_ms(finished_at_ts_ms);
    let conn = open_sync_connection(db_path)?;
    conn.execute(
        "
        UPDATE exchange_sync_runs
        SET finished_at_ts_ms = ?2,
            finished_at = ?3,
            status = ?4,
            error_text = ?5,
            wallet_address = ?6,
            resolved_start_ms = ?7,
            resolved_end_ms = ?8,
            window_source = ?9,
            cursor_key = COALESCE(?10, cursor_key),
            fetched_remote_fills = ?11,
            supported_remote_fills = ?12,
            unsupported_remote_fills = ?13,
            inserted_oms_fills = ?14,
            linked_existing_oms_fills = ?15,
            inserted_trades = ?16,
            backfilled_existing_trades = ?17,
            inserted_manual_intents = ?18,
            relabelled_manual_trades = ?19,
            exchange_position_count = ?20,
            account_value_usd = ?21,
            withdrawable_usd = ?22,
            total_margin_used_usd = ?23,
            warnings_json = ?24
        WHERE id = ?1
        ",
        params![
            progress.run_id,
            finished_at_ts_ms,
            finished_at,
            status,
            error_text,
            progress.wallet_address.as_deref(),
            progress
                .resolved_window
                .as_ref()
                .map(|window| window.start_ms),
            progress
                .resolved_window
                .as_ref()
                .map(|window| window.end_ms),
            progress
                .resolved_window
                .as_ref()
                .map(|window| window.source.as_str()),
            progress
                .resolved_window
                .as_ref()
                .and_then(|window| window.cursor_key.as_deref()),
            stats.fetched_remote_fills as i64,
            stats.supported_remote_fills as i64,
            stats.unsupported_remote_fills as i64,
            stats.inserted_oms_fills as i64,
            stats.linked_existing_oms_fills as i64,
            stats.inserted_trades as i64,
            stats.backfilled_existing_trades as i64,
            stats.inserted_manual_intents as i64,
            stats.relabelled_manual_trades as i64,
            progress.exchange_position_count.unwrap_or_default() as i64,
            progress.account_value_usd,
            progress.withdrawable_usd,
            progress.total_margin_used_usd,
            serde_json::to_string(&stats.warnings)
                .context("failed to serialise sync run warnings")?,
        ],
    )
    .with_context(|| format!("failed to finalise exchange sync run {}", progress.run_id))?;
    Ok(())
}

fn write_sync_cursor(
    db_path: &Path,
    cursor_key: &str,
    start_ms: i64,
    end_ms: i64,
    run_id: i64,
) -> Result<()> {
    let conn = open_sync_connection(db_path)?;
    conn.execute(
        "
        INSERT INTO runtime_sync_cursors (
            sync_key, last_start_ts_ms, last_end_ts_ms, updated_at, last_run_id
        ) VALUES (?1, ?2, ?3, ?4, ?5)
        ON CONFLICT(sync_key) DO UPDATE SET
            last_start_ts_ms = excluded.last_start_ts_ms,
            last_end_ts_ms = excluded.last_end_ts_ms,
            updated_at = excluded.updated_at,
            last_run_id = excluded.last_run_id
        ",
        params![
            cursor_key,
            start_ms,
            end_ms,
            Utc::now().to_rfc3339(),
            run_id
        ],
    )
    .context("failed to persist sync cursor")?;
    Ok(())
}

fn exchange_snapshot_source(dry_run: bool) -> &'static str {
    if dry_run {
        LIVE_FILL_SYNC_DRY_RUN_SOURCE
    } else {
        LIVE_FILL_SYNC_SOURCE
    }
}

fn persist_account_snapshot_event(
    db_path: &Path,
    account_observation: &crate::live_hyperliquid::HyperliquidClearinghouseStateObservation,
    sync_run_id: i64,
    wallet_address: &str,
    dry_run: bool,
) -> Result<i64> {
    let ts_ms = account_observation.observed_at_ts_ms;
    let timestamp = timestamp_from_ms(ts_ms);
    let source = exchange_snapshot_source(dry_run);
    let conn = open_sync_connection(db_path)?;
    conn.execute(
        "
        INSERT INTO exchange_account_snapshot_events (
            sync_run_id, observed_at_ts_ms, observed_at, source, request_type, wallet_address,
            raw_payload_json, payload_digest, account_value_usd, withdrawable_usd,
            total_margin_used_usd, position_count
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
        ",
        params![
            sync_run_id,
            ts_ms,
            &timestamp,
            source,
            CLEARINGHOUSE_STATE_REQUEST_TYPE,
            wallet_address,
            &account_observation.payload_json,
            &account_observation.payload_digest,
            account_observation.account_snapshot.account_value_usd,
            account_observation.account_snapshot.withdrawable_usd,
            account_observation.account_snapshot.total_margin_used_usd,
            account_observation.positions.len() as i64,
        ],
    )
    .context("failed to insert exchange account snapshot evidence")?;
    Ok(conn.last_insert_rowid())
}

fn persist_exchange_snapshot_projection(
    db_path: &Path,
    account_observation: &crate::live_hyperliquid::HyperliquidClearinghouseStateObservation,
    account_snapshot_event_id: i64,
    sync_run_id: i64,
    dry_run: bool,
) -> Result<()> {
    let ts_ms = account_observation.observed_at_ts_ms;
    let timestamp = timestamp_from_ms(ts_ms);
    let source = exchange_snapshot_source(dry_run);
    let mut conn = open_sync_connection(db_path)?;
    let tx = conn
        .transaction()
        .context("failed to start exchange snapshot transaction")?;
    tx.execute(
        "
        INSERT INTO runtime_account_snapshots (
            ts_ms, timestamp, account_value_usd, withdrawable_usd, total_margin_used_usd,
            source, meta_json, sync_run_id, account_snapshot_event_id
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        ",
        params![
            ts_ms,
            &timestamp,
            account_observation.account_snapshot.account_value_usd,
            account_observation.account_snapshot.withdrawable_usd,
            account_observation.account_snapshot.total_margin_used_usd,
            source,
            json!({
                "position_count": account_observation.positions.len(),
            })
            .to_string(),
            sync_run_id,
            account_snapshot_event_id,
        ],
    )
    .context("failed to insert runtime account snapshot")?;
    tx.execute("DELETE FROM runtime_exchange_positions", [])
        .context("failed to clear runtime exchange positions")?;
    for position in &account_observation.positions {
        tx.execute(
            "
            INSERT INTO runtime_exchange_positions (
                symbol, pos_type, size, entry_price, leverage, margin_used, ts_ms, updated_at,
                source, sync_run_id
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ",
            params![
                position.symbol.trim().to_ascii_uppercase(),
                position.pos_type.trim().to_ascii_uppercase(),
                position.size,
                position.entry_price,
                position.leverage,
                position.margin_used,
                ts_ms,
                &timestamp,
                source,
                sync_run_id,
            ],
        )
        .with_context(|| {
            format!(
                "failed to persist runtime exchange position for {}",
                position.symbol
            )
        })?;
    }
    tx.commit()
        .context("failed to commit exchange snapshot transaction")?;

    sync_exchange_positions(db_path, &account_observation.positions, ts_ms)
        .context("failed to sync position_state from exchange positions")?;
    Ok(())
}

fn reconcile_remote_fills(
    db_path: &Path,
    oms: &LiveOms,
    fills: &[HyperliquidFill],
    aiq_cloid_prefix: &str,
    sync_run_id: i64,
    stats: &mut SyncStats,
) -> Result<()> {
    let conn = open_sync_connection(db_path)?;
    for fill in fills {
        let Some(parsed) = parse_fill(&fill.raw) else {
            let fill_hash = fill.raw.get("hash").and_then(json_to_text);
            let fill_tid = fill.raw.get("tid").and_then(parse_json_integer);
            if fill_fully_present_locally(&conn, fill_hash, fill_tid)? {
                stats.warnings.push(format!(
                    "unsupported remote fill already present locally: {}",
                    compact_fill_identity(&fill.raw)
                ));
                continue;
            }
            stats.unsupported_remote_fills += 1;
            stats.warnings.push(format!(
                "unsupported remote fill skipped: {}",
                compact_fill_identity(&fill.raw)
            ));
            continue;
        };

        stats.supported_remote_fills += 1;
        *stats
            .fills_by_symbol
            .entry(parsed.symbol.clone())
            .or_default() += 1;
        let resolution = resolve_fill(
            oms,
            &conn,
            &parsed,
            &fill.raw,
            aiq_cloid_prefix,
            sync_run_id,
            stats,
        )?;
        *stats
            .matched_by
            .entry(resolution.matched_via.clone())
            .or_default() += 1;
        let fill_link = upsert_oms_fill(&conn, &parsed, &fill.raw, &resolution, sync_run_id)?;
        if fill_link.inserted {
            stats.inserted_oms_fills += 1;
        }
        if fill_link.linked_existing {
            stats.linked_existing_oms_fills += 1;
        }
        let trade_result = ensure_trade_row(
            &conn,
            &parsed,
            &fill.raw,
            resolution.intent.as_ref(),
            &resolution.matched_via,
            sync_run_id,
            resolution.is_manual_trade,
        )?;
        if trade_result.inserted {
            stats.inserted_trades += 1;
        }
    }
    Ok(())
}

fn backfill_existing_oms_trades(
    db_path: &Path,
    oms: &LiveOms,
    aiq_cloid_prefix: &str,
    sync_run_id: i64,
    stats: &mut SyncStats,
) -> Result<()> {
    let conn = open_sync_connection(db_path)?;
    let missing_rows = load_oms_fills_without_trade(&conn)?;
    for row in missing_rows {
        let raw_fill: Value = serde_json::from_str(&row.raw_json)
            .with_context(|| format!("failed to parse oms_fills.raw_json for row {}", row.id))?;
        let Some(parsed) = parse_fill(&raw_fill) else {
            stats.warnings.push(format!(
                "unsupported oms fill without trade skipped: row_id={} {}",
                row.id,
                compact_fill_identity(&raw_fill)
            ));
            continue;
        };

        let resolution = if let Some(intent_id) = row.intent_id.as_deref() {
            let intent = load_intent_snapshot(&conn, intent_id)?;
            FillResolution {
                is_manual_trade: intent_id.starts_with("manual_")
                    || is_manual_fill_cloid(parsed.cloid.as_deref(), aiq_cloid_prefix),
                intent,
                matched_via: row
                    .matched_via
                    .clone()
                    .unwrap_or_else(|| "oms_fill_backfill".to_string()),
            }
        } else {
            resolve_fill(
                oms,
                &conn,
                &parsed,
                &raw_fill,
                aiq_cloid_prefix,
                sync_run_id,
                stats,
            )?
        };

        if row.intent_id.is_none() {
            if let Some(intent) = resolution.intent.as_ref() {
                link_existing_oms_fill(
                    &conn,
                    row.id,
                    &intent.intent_id,
                    &resolution.matched_via,
                    sync_run_id,
                )?;
                stats.linked_existing_oms_fills += 1;
            }
        }

        let trade_result = ensure_trade_row(
            &conn,
            &parsed,
            &raw_fill,
            resolution.intent.as_ref(),
            &resolution.matched_via,
            sync_run_id,
            resolution.is_manual_trade,
        )?;
        if trade_result.inserted {
            stats.backfilled_existing_trades += 1;
        }
    }
    Ok(())
}

fn relabel_existing_manual_trades(
    db_path: &Path,
    aiq_cloid_prefix: &str,
    sync_run_id: i64,
    stats: &mut SyncStats,
) -> Result<()> {
    let conn = open_sync_connection(db_path)?;
    let mut stmt = conn.prepare(
        "
        SELECT t.id, t.fill_hash, t.fill_tid, f.raw_json
        FROM trades t
        JOIN oms_fills f
          ON IFNULL(t.fill_hash, '') = IFNULL(f.fill_hash, '')
         AND IFNULL(t.fill_tid, -1) = IFNULL(f.fill_tid, -1)
        WHERE COALESCE(t.reason, '') != 'manual_trade'
           OR COALESCE(t.reason_code, '') != 'manual_trade'
           OR COALESCE(t.confidence, '') != 'MANUAL'
        ",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<i64>>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;

    for row in rows {
        let (trade_id, fill_hash, fill_tid, raw_json) = row?;
        let raw_fill: Value = serde_json::from_str(&raw_json).with_context(|| {
            format!("failed to parse oms_fills.raw_json for trade {}", trade_id)
        })?;
        let Some(parsed) = parse_fill(&raw_fill) else {
            continue;
        };
        if !is_manual_fill_cloid(parsed.cloid.as_deref(), aiq_cloid_prefix) {
            continue;
        }
        if relabel_manual_trade(
            &conn,
            trade_id,
            &parsed,
            &raw_fill,
            fill_hash.as_deref(),
            fill_tid,
            sync_run_id,
        )? {
            stats.relabelled_manual_trades += 1;
        }
    }
    Ok(())
}

fn resolve_fill(
    oms: &LiveOms,
    conn: &Connection,
    parsed: &ParsedFill,
    raw_fill: &Value,
    aiq_cloid_prefix: &str,
    sync_run_id: i64,
    stats: &mut SyncStats,
) -> Result<FillResolution> {
    if let Some(matched) = oms.match_intent_for_fill(
        raw_fill,
        &parsed.symbol,
        &parsed.action,
        &parsed.side,
        parsed.ts_ms,
    )? {
        let intent = load_intent_snapshot(conn, &matched.intent_id)?;
        let is_manual_trade = matched.intent_id.starts_with("manual_")
            || is_manual_fill_cloid(parsed.cloid.as_deref(), aiq_cloid_prefix);
        return Ok(FillResolution {
            intent,
            matched_via: matched.matched_via.to_string(),
            is_manual_trade,
        });
    }

    let cloid_kind = classify_cloid(parsed.cloid.as_deref(), aiq_cloid_prefix);
    if cloid_kind == CloidKind::Aiq {
        return Ok(FillResolution {
            intent: None,
            matched_via: "unmatched_aiq_fill".to_string(),
            is_manual_trade: false,
        });
    }

    let intent = ensure_manual_intent(conn, parsed, raw_fill, sync_run_id)?;
    stats.inserted_manual_intents += usize::from(intent.1);
    Ok(FillResolution {
        intent: Some(intent.0),
        matched_via: "manual_orphan".to_string(),
        is_manual_trade: true,
    })
}

fn ensure_manual_intent(
    conn: &Connection,
    parsed: &ParsedFill,
    raw_fill: &Value,
    sync_run_id: i64,
) -> Result<(IntentSnapshot, bool)> {
    let intent_id = manual_intent_id(parsed, raw_fill);
    if let Some(intent) = load_intent_snapshot(conn, &intent_id)? {
        return Ok((intent, false));
    }

    let raw_cloid = parsed.cloid.as_deref();
    let meta_json = json!({
        "manual": true,
        "source": "live_fill_sync",
        "fill": raw_fill,
    })
    .to_string();
    conn.execute(
        "
        INSERT OR IGNORE INTO oms_intents (
            intent_id, created_ts_ms, sent_ts_ms, symbol, action, side, requested_size,
            requested_notional, entry_atr, leverage, decision_ts_ms, strategy_version,
            strategy_sha1, reason, confidence, status, dedupe_key, client_order_id,
            exchange_order_id, last_error, meta_json, sync_run_id
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7,
            ?8, ?9, ?10, ?11, ?12,
            ?13, ?14, ?15, ?16, ?17, ?18,
            ?19, ?20, ?21, ?22
        )
        ",
        params![
            intent_id,
            parsed.ts_ms,
            parsed.ts_ms,
            parsed.symbol,
            parsed.action,
            parsed.side,
            parsed.size,
            parsed.notional_usd,
            Option::<f64>::None,
            Option::<f64>::None,
            Option::<i64>::None,
            Option::<String>::None,
            Option::<String>::None,
            "MANUAL_FILL",
            "n/a",
            "FILLED",
            Option::<String>::None,
            raw_cloid,
            parsed.exchange_order_id.as_deref(),
            Option::<String>::None,
            meta_json,
            sync_run_id,
        ],
    )
    .context("failed to insert synthetic manual OMS intent")?;
    let intent = load_intent_snapshot(conn, &intent_id)?
        .with_context(|| format!("failed to reload synthetic manual intent {}", intent_id))?;
    Ok((intent, true))
}

fn manual_intent_id(parsed: &ParsedFill, raw_fill: &Value) -> String {
    if let Some(fill_hash) = parsed.fill_hash.as_deref() {
        let stripped_hash = fill_hash.trim_start_matches("0x");
        let suffix = parsed
            .fill_tid
            .map(|value| value.to_string())
            .unwrap_or_else(|| "na".to_string());
        return format!(
            "manual_{}_{}",
            &stripped_hash[..12.min(stripped_hash.len())],
            suffix
        );
    }

    let digest = Sha3_256::digest(raw_fill.to_string().as_bytes());
    format!("manual_{}", hex::encode(&digest[..8]))
}

fn load_intent_snapshot(conn: &Connection, intent_id: &str) -> Result<Option<IntentSnapshot>> {
    conn.query_row(
        "
        SELECT intent_id, reason, confidence, leverage, entry_atr, client_order_id,
               exchange_order_id, meta_json
        FROM oms_intents
        WHERE intent_id = ?1
        LIMIT 1
        ",
        params![intent_id],
        |row| {
            let intent_id: String = row.get(0)?;
            let reason: Option<String> = row.get(1)?;
            let confidence: Option<String> = row.get(2)?;
            let leverage: Option<f64> = row.get(3)?;
            let entry_atr: Option<f64> = row.get(4)?;
            let client_order_id: Option<String> = row.get(5)?;
            let exchange_order_id: Option<String> = row.get(6)?;
            let meta_json: Option<String> = row.get(7)?;
            let parsed_meta = meta_json
                .as_deref()
                .and_then(|payload| serde_json::from_str::<Value>(payload).ok());
            let reason_code = parsed_meta
                .as_ref()
                .and_then(|value| value.get("reason_code"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            Ok(IntentSnapshot {
                intent_id,
                reason,
                reason_code,
                confidence,
                leverage,
                entry_atr,
                client_order_id,
                exchange_order_id,
            })
        },
    )
    .optional()
    .context("failed to load OMS intent snapshot")
}

struct OmsFillLinkResult {
    inserted: bool,
    linked_existing: bool,
}

fn upsert_oms_fill(
    conn: &Connection,
    parsed: &ParsedFill,
    raw_fill: &Value,
    resolution: &FillResolution,
    sync_run_id: i64,
) -> Result<OmsFillLinkResult> {
    let existing = load_existing_oms_fill(conn, parsed.fill_hash.as_deref(), parsed.fill_tid)?;
    if let Some(existing) = existing {
        if existing.intent_id.is_none() {
            if let Some(intent) = resolution.intent.as_ref() {
                link_existing_oms_fill(
                    conn,
                    existing.id,
                    &intent.intent_id,
                    &resolution.matched_via,
                    sync_run_id,
                )?;
                return Ok(OmsFillLinkResult {
                    inserted: false,
                    linked_existing: true,
                });
            }
        }
        return Ok(OmsFillLinkResult {
            inserted: false,
            linked_existing: false,
        });
    }

    conn.execute(
        "
        INSERT OR IGNORE INTO oms_fills (
            ts_ms, symbol, intent_id, order_id, action, side, pos_type, price, size, notional,
            fee_usd, fee_token, fee_rate, pnl_usd, fill_hash, fill_tid, matched_via, raw_json,
            sync_run_id
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
            ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19
        )
        ",
        params![
            parsed.ts_ms,
            parsed.symbol,
            resolution
                .intent
                .as_ref()
                .map(|intent| intent.intent_id.as_str()),
            parsed
                .exchange_order_id
                .as_deref()
                .and_then(|value| value.parse::<i64>().ok()),
            parsed.action,
            parsed.side,
            parsed.pos_type,
            parsed.price,
            parsed.size,
            parsed.notional_usd,
            parsed.fee_usd,
            parsed.fee_token.as_deref(),
            parsed.fee_rate,
            parsed.pnl_usd,
            parsed.fill_hash.as_deref(),
            parsed.fill_tid,
            resolution.matched_via.as_str(),
            raw_fill.to_string(),
            sync_run_id,
        ],
    )
    .context("failed to insert reconciled OMS fill")?;
    Ok(OmsFillLinkResult {
        inserted: true,
        linked_existing: false,
    })
}

fn load_existing_oms_fill(
    conn: &Connection,
    fill_hash: Option<&str>,
    fill_tid: Option<i64>,
) -> Result<Option<ExistingOmsFill>> {
    conn.query_row(
        "
        SELECT id, intent_id, matched_via
        FROM oms_fills
        WHERE IFNULL(fill_hash, '') = IFNULL(?1, '')
          AND IFNULL(fill_tid, -1) = IFNULL(?2, -1)
        ORDER BY id DESC
        LIMIT 1
        ",
        params![fill_hash, fill_tid],
        |row| {
            Ok(ExistingOmsFill {
                id: row.get(0)?,
                intent_id: row.get(1)?,
            })
        },
    )
    .optional()
    .context("failed to query existing oms fill")
}

fn link_existing_oms_fill(
    conn: &Connection,
    oms_fill_id: i64,
    intent_id: &str,
    matched_via: &str,
    sync_run_id: i64,
) -> Result<()> {
    conn.execute(
        "
        UPDATE oms_fills
        SET intent_id = ?1,
            sync_run_id = ?2,
            matched_via = CASE
                WHEN matched_via IS NULL OR TRIM(matched_via) = '' THEN ?3
                ELSE matched_via
            END
        WHERE id = ?4
        ",
        params![intent_id, sync_run_id, matched_via, oms_fill_id],
    )
    .with_context(|| format!("failed to link existing oms fill {}", oms_fill_id))?;
    Ok(())
}

struct TradeUpsertResult {
    inserted: bool,
}

fn ensure_trade_row(
    conn: &Connection,
    parsed: &ParsedFill,
    raw_fill: &Value,
    intent: Option<&IntentSnapshot>,
    matched_via: &str,
    sync_run_id: i64,
    is_manual_trade: bool,
) -> Result<TradeUpsertResult> {
    let existing_trade_id =
        find_trade_id_by_fill(conn, parsed.fill_hash.as_deref(), parsed.fill_tid)?;
    if let Some(trade_id) = existing_trade_id {
        if is_manual_trade {
            let _ = relabel_manual_trade(
                conn,
                trade_id,
                parsed,
                raw_fill,
                parsed.fill_hash.as_deref(),
                parsed.fill_tid,
                sync_run_id,
            )?;
        }
        return Ok(TradeUpsertResult { inserted: false });
    }

    let meta_json = build_trade_meta_json(parsed, raw_fill, intent, matched_via, is_manual_trade);
    let leverage = intent.and_then(|intent| intent.leverage);
    let entry_atr = intent.and_then(|intent| intent.entry_atr);
    let reason = if is_manual_trade {
        "manual_trade".to_string()
    } else {
        intent
            .and_then(|intent| intent.reason.clone())
            .unwrap_or_else(|| format!("LIVE_FILL {}", parsed.dir))
    };
    let reason_code = if is_manual_trade {
        Some("manual_trade".to_string())
    } else {
        intent.and_then(|intent| intent.reason_code.clone())
    };
    let confidence = if is_manual_trade {
        "MANUAL".to_string()
    } else {
        intent
            .and_then(|intent| intent.confidence.clone())
            .unwrap_or_else(|| "N/A".to_string())
    };
    let margin_used = leverage
        .filter(|value| *value > 0.0)
        .map(|value| parsed.notional_usd / value);

    conn.execute(
        "
        INSERT INTO trades (
            timestamp, symbol, type, action, price, size, notional, reason, reason_code,
            confidence, pnl, fee_usd, fee_token, fee_rate, balance, entry_atr, leverage,
            margin_used, meta_json, fill_hash, fill_tid, sync_run_id
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9,
            ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17,
            ?18, ?19, ?20, ?21, ?22
        )
        ",
        params![
            parsed.timestamp,
            parsed.symbol,
            parsed.pos_type,
            parsed.action,
            parsed.price,
            parsed.size,
            parsed.notional_usd,
            reason,
            reason_code,
            confidence,
            parsed.pnl_usd,
            parsed.fee_usd,
            parsed.fee_token.as_deref(),
            parsed.fee_rate,
            Option::<f64>::None,
            entry_atr,
            leverage,
            margin_used,
            meta_json,
            parsed.fill_hash.as_deref(),
            parsed.fill_tid,
            sync_run_id,
        ],
    )
    .context("failed to insert reconciled trade row")?;

    Ok(TradeUpsertResult { inserted: true })
}

fn relabel_manual_trade(
    conn: &Connection,
    trade_id: i64,
    parsed: &ParsedFill,
    raw_fill: &Value,
    fill_hash: Option<&str>,
    fill_tid: Option<i64>,
    sync_run_id: i64,
) -> Result<bool> {
    let meta_json = build_trade_meta_json(parsed, raw_fill, None, "manual_trade", true);
    let changed = conn.execute(
        "
        UPDATE trades
        SET reason = 'manual_trade',
            reason_code = 'manual_trade',
            confidence = 'MANUAL',
            meta_json = ?1,
            fill_hash = COALESCE(fill_hash, ?2),
            fill_tid = COALESCE(fill_tid, ?3),
            sync_run_id = ?4
        WHERE id = ?5
          AND (
                COALESCE(reason, '') != 'manual_trade'
             OR COALESCE(reason_code, '') != 'manual_trade'
             OR COALESCE(confidence, '') != 'MANUAL'
          )
        ",
        params![meta_json, fill_hash, fill_tid, sync_run_id, trade_id],
    )?;
    Ok(changed > 0)
}

fn build_trade_meta_json(
    parsed: &ParsedFill,
    raw_fill: &Value,
    intent: Option<&IntentSnapshot>,
    matched_via: &str,
    is_manual_trade: bool,
) -> String {
    let source = if is_manual_trade {
        "manual_trade"
    } else {
        "live_fill_sync"
    };
    json!({
        "source": source,
        "fill": raw_fill,
        "oms": {
            "intent_id": intent.map(|intent| intent.intent_id.as_str()),
            "client_order_id": intent
                .and_then(|intent| intent.client_order_id.as_deref())
                .or(parsed.cloid.as_deref()),
            "exchange_order_id": intent
                .and_then(|intent| intent.exchange_order_id.as_deref())
                .or(parsed.exchange_order_id.as_deref()),
            "matched_via": matched_via,
        }
    })
    .to_string()
}

fn find_trade_id_by_fill(
    conn: &Connection,
    fill_hash: Option<&str>,
    fill_tid: Option<i64>,
) -> Result<Option<i64>> {
    conn.query_row(
        "
        SELECT id
        FROM trades
        WHERE IFNULL(fill_hash, '') = IFNULL(?1, '')
          AND IFNULL(fill_tid, -1) = IFNULL(?2, -1)
        ORDER BY id DESC
        LIMIT 1
        ",
        params![fill_hash, fill_tid],
        |row| row.get(0),
    )
    .optional()
    .context("failed to query trade row by fill")
}

fn fill_fully_present_locally(
    conn: &Connection,
    fill_hash: Option<&str>,
    fill_tid: Option<i64>,
) -> Result<bool> {
    if fill_hash.is_none() && fill_tid.is_none() {
        return Ok(false);
    }
    let oms_fill_id = load_existing_oms_fill(conn, fill_hash, fill_tid)?.map(|row| row.id);
    let trade_id = find_trade_id_by_fill(conn, fill_hash, fill_tid)?;
    Ok(oms_fill_id.is_some() && trade_id.is_some())
}

fn load_oms_fills_without_trade(conn: &Connection) -> Result<Vec<ExistingOmsFillWithoutTrade>> {
    let mut stmt = conn.prepare(
        "
        SELECT f.id, f.intent_id, f.matched_via, f.raw_json
        FROM oms_fills f
        LEFT JOIN trades t
          ON IFNULL(t.fill_hash, '') = IFNULL(f.fill_hash, '')
         AND IFNULL(t.fill_tid, -1) = IFNULL(f.fill_tid, -1)
        WHERE t.id IS NULL
        ORDER BY f.ts_ms ASC, f.id ASC
        ",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(ExistingOmsFillWithoutTrade {
            id: row.get(0)?,
            intent_id: row.get(1)?,
            matched_via: row.get(2)?,
            raw_json: row.get(3)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .context("failed to read oms fills without trade rows")
}

fn parse_fill(raw: &Value) -> Option<ParsedFill> {
    let symbol = raw
        .get("coin")
        .or_else(|| raw.get("symbol"))
        .and_then(json_to_text)?
        .trim()
        .to_ascii_uppercase();
    if symbol.is_empty() {
        return None;
    }
    let price = raw.get("px").and_then(parse_json_number)?;
    let size = raw.get("sz").and_then(parse_json_number)?;
    if price <= 0.0 || size <= 0.0 {
        return None;
    }

    let ts_ms = raw
        .get("time")
        .or_else(|| raw.get("timestamp"))
        .and_then(parse_json_integer)
        .unwrap_or_else(|| Utc::now().timestamp_millis());
    let timestamp = chrono::DateTime::from_timestamp_millis(ts_ms)
        .map(|value| value.to_rfc3339())
        .unwrap_or_else(|| Utc::now().to_rfc3339());
    let dir = raw
        .get("dir")
        .and_then(json_to_text)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let start_position = raw
        .get("startPosition")
        .and_then(parse_json_number)
        .unwrap_or(0.0);
    let (action, pos_type) = classify_fill_direction(&dir, start_position, size)?;
    let side = raw
        .get("side")
        .and_then(json_to_text)
        .map(|value| {
            if value.eq_ignore_ascii_case("B") {
                "BUY".to_string()
            } else {
                "SELL".to_string()
            }
        })
        .unwrap_or_else(|| {
            if pos_type == "LONG" {
                if action == "OPEN" || action == "ADD" {
                    "BUY".to_string()
                } else {
                    "SELL".to_string()
                }
            } else if action == "OPEN" || action == "ADD" {
                "SELL".to_string()
            } else {
                "BUY".to_string()
            }
        });
    let fee_usd = raw.get("fee").and_then(parse_json_number).unwrap_or(0.0);
    let notional_usd = price * size;
    Some(ParsedFill {
        timestamp,
        ts_ms,
        symbol,
        action: action.to_string(),
        side,
        pos_type: pos_type.to_string(),
        price,
        size,
        notional_usd,
        pnl_usd: raw
            .get("closedPnl")
            .and_then(parse_json_number)
            .unwrap_or(0.0),
        fee_usd,
        fee_token: raw
            .get("feeToken")
            .and_then(json_to_text)
            .map(|value| value.to_string()),
        fee_rate: if notional_usd > 0.0 {
            Some(fee_usd / notional_usd)
        } else {
            None
        },
        fill_hash: raw.get("hash").and_then(json_to_text).map(str::to_string),
        fill_tid: raw.get("tid").and_then(parse_json_integer),
        exchange_order_id: raw.get("oid").and_then(json_scalar_to_string),
        cloid: raw.get("cloid").and_then(json_scalar_to_string),
        dir,
    })
}

fn classify_fill_direction(
    dir: &str,
    start_position: f64,
    fill_size: f64,
) -> Option<(&'static str, &'static str)> {
    let pos_type = if dir.contains("long") {
        "LONG"
    } else if dir.contains("short") {
        "SHORT"
    } else {
        return None;
    };
    if dir.starts_with("open") {
        let action = if start_position.abs() < 1e-9 {
            "OPEN"
        } else {
            "ADD"
        };
        return Some((action, pos_type));
    }
    if dir.starts_with("close") {
        let ending_position = if pos_type == "LONG" {
            start_position - fill_size
        } else {
            start_position + fill_size
        };
        let action = if ending_position.abs() < 1e-9 {
            "CLOSE"
        } else {
            "REDUCE"
        };
        return Some((action, pos_type));
    }
    None
}

fn compact_fill_identity(raw: &Value) -> String {
    let symbol = raw
        .get("coin")
        .or_else(|| raw.get("symbol"))
        .and_then(json_to_text)
        .unwrap_or("unknown");
    let dir = raw.get("dir").and_then(json_to_text).unwrap_or("unknown");
    let fill_hash = raw.get("hash").and_then(json_to_text).unwrap_or("no-hash");
    let fill_tid = raw
        .get("tid")
        .and_then(parse_json_integer)
        .map(|value| value.to_string())
        .unwrap_or_else(|| "no-tid".to_string());
    format!("{symbol}:{dir}:{fill_hash}:{fill_tid}")
}

fn json_to_text(value: &Value) -> Option<&str> {
    match value {
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        }
        _ => None,
    }
}

fn json_scalar_to_string(value: &Value) -> Option<String> {
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

fn parse_json_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(raw) => raw.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn parse_json_integer(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => number.as_i64(),
        Value::String(raw) => raw.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn classify_cloid(cloid: Option<&str>, aiq_cloid_prefix: &str) -> CloidKind {
    let Some(cloid) = cloid.map(str::trim).filter(|value| !value.is_empty()) else {
        return CloidKind::None;
    };
    let Some(bytes) = decode_hex_cloid(cloid) else {
        return CloidKind::Other;
    };
    if bytes.starts_with(b"man_") {
        return CloidKind::Manual;
    }
    let aiq_prefix_bytes = decode_prefix_bytes(aiq_cloid_prefix);
    if !aiq_prefix_bytes.is_empty() && bytes.starts_with(&aiq_prefix_bytes) {
        return CloidKind::Aiq;
    }
    CloidKind::Other
}

fn is_manual_fill_cloid(cloid: Option<&str>, aiq_cloid_prefix: &str) -> bool {
    matches!(
        classify_cloid(cloid, aiq_cloid_prefix),
        CloidKind::None | CloidKind::Manual
    )
}

fn decode_hex_cloid(cloid: &str) -> Option<Vec<u8>> {
    let stripped = cloid.trim().strip_prefix("0x")?;
    hex::decode(stripped).ok()
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

fn open_sync_connection(db_path: &Path) -> Result<Connection> {
    let conn = Connection::open(db_path)
        .with_context(|| format!("failed to open live db at {}", db_path.display()))?;
    conn.busy_timeout(Duration::from_millis(DEFAULT_DB_TIMEOUT_MS))
        .context("failed to configure live fill sync SQLite busy timeout")?;
    conn.execute_batch(
        "
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        ",
    )
    .context("failed to configure live fill sync SQLite pragmas")?;
    Ok(conn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_oms::CreateIntentRequest;
    use tempfile::NamedTempFile;

    fn temp_db() -> NamedTempFile {
        let file = NamedTempFile::new().unwrap();
        ensure_sync_schema(file.path()).unwrap();
        let _ = LiveOms::new(file.path()).unwrap();
        let conn = Connection::open(file.path()).unwrap();
        ensure_sync_run_columns(&conn).unwrap();
        file
    }

    fn manual_fill() -> Value {
        json!({
            "coin": "ETH",
            "px": "2300",
            "sz": "0.5",
            "time": 1_772_000_000_000_i64,
            "dir": "Open Long",
            "startPosition": "0",
            "fee": "0.4025",
            "closedPnl": "0",
            "hash": "0xmanualfill0001",
            "tid": 101,
            "oid": "9001"
        })
    }

    #[test]
    fn sync_run_headers_persist_status_counts_and_cursor_provenance() {
        let db = temp_db();
        let input = LiveFillSyncInput {
            live_db: db.path(),
            secrets_path: Path::new("/tmp/secrets.json"),
            profile: Some("production"),
            config_path: Some(Path::new("/tmp/live.yaml")),
            config_id: Some("cfg-123"),
            start_ms: Some(100),
            end_ms: Some(200),
            lookback_hours: 24,
            overlap_minutes: 10,
            cursor_key: Some("test_cursor"),
            dry_run: false,
        };
        let run_id =
            insert_sync_run_start(db.path(), &input, db.path(), db.path(), "test_cursor").unwrap();
        let progress = SyncRunProgress {
            run_id,
            wallet_address: Some("0xabc".to_string()),
            resolved_window: Some(SyncWindow {
                start_ms: 120,
                end_ms: 220,
                source: "cursor".to_string(),
                cursor_key: Some("test_cursor".to_string()),
            }),
            account_value_usd: Some(500.0),
            withdrawable_usd: Some(300.0),
            total_margin_used_usd: Some(200.0),
            exchange_position_count: Some(2),
        };
        let stats = SyncStats {
            fetched_remote_fills: 4,
            supported_remote_fills: 3,
            unsupported_remote_fills: 1,
            inserted_oms_fills: 2,
            linked_existing_oms_fills: 1,
            inserted_trades: 2,
            backfilled_existing_trades: 1,
            inserted_manual_intents: 1,
            relabelled_manual_trades: 0,
            matched_by: BTreeMap::new(),
            fills_by_symbol: BTreeMap::new(),
            warnings: vec!["unsupported remote fill skipped".to_string()],
        };

        finalize_sync_run(
            db.path(),
            &progress,
            &stats,
            SYNC_RUN_STATUS_UNSUPPORTED_REMOTE_FILLS,
            None,
        )
        .unwrap();
        write_sync_cursor(db.path(), "test_cursor", 120, 220, run_id).unwrap();

        let conn = Connection::open(db.path()).unwrap();
        let row: (String, i64, i64, i64, i64) = conn
            .query_row(
                "SELECT status, fetched_remote_fills, unsupported_remote_fills, exchange_position_count, resolved_start_ms
                 FROM exchange_sync_runs WHERE id = ?1",
                params![run_id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .unwrap();
        let cursor_last_run_id: i64 = conn
            .query_row(
                "SELECT last_run_id FROM runtime_sync_cursors WHERE sync_key = 'test_cursor'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(row.0, SYNC_RUN_STATUS_UNSUPPORTED_REMOTE_FILLS);
        assert_eq!(row.1, 4);
        assert_eq!(row.2, 1);
        assert_eq!(row.3, 2);
        assert_eq!(row.4, 120);
        assert_eq!(cursor_last_run_id, run_id);
    }

    #[test]
    fn failed_sync_run_keeps_requested_cursor_key_before_window_resolution() {
        let db = temp_db();
        let input = LiveFillSyncInput {
            live_db: db.path(),
            secrets_path: Path::new("/tmp/secrets.json"),
            profile: Some("production"),
            config_path: Some(Path::new("/tmp/live.yaml")),
            config_id: Some("cfg-123"),
            start_ms: None,
            end_ms: None,
            lookback_hours: 24,
            overlap_minutes: 10,
            cursor_key: Some("failed_cursor"),
            dry_run: false,
        };
        let run_id =
            insert_sync_run_start(db.path(), &input, db.path(), db.path(), "failed_cursor")
                .unwrap();
        let progress = SyncRunProgress {
            run_id,
            wallet_address: None,
            resolved_window: None,
            account_value_usd: None,
            withdrawable_usd: None,
            total_margin_used_usd: None,
            exchange_position_count: None,
        };

        finalize_sync_run(
            db.path(),
            &progress,
            &SyncStats::default(),
            SYNC_RUN_STATUS_FAILED,
            Some("client init failed"),
        )
        .unwrap();

        let conn = Connection::open(db.path()).unwrap();
        let row: (String, String) = conn
            .query_row(
                "SELECT status, cursor_key FROM exchange_sync_runs WHERE id = ?1",
                params![run_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();

        assert_eq!(row.0, SYNC_RUN_STATUS_FAILED);
        assert_eq!(row.1, "failed_cursor");
    }

    #[test]
    fn remote_orphan_fill_creates_manual_intent_and_trade() {
        let db = temp_db();
        let oms = LiveOms::new(db.path()).unwrap();
        let fill = HyperliquidFill { raw: manual_fill() };
        let mut stats = SyncStats::default();
        reconcile_remote_fills(db.path(), &oms, &[fill], "aiq_", 101, &mut stats).unwrap();

        let conn = Connection::open(db.path()).unwrap();
        let intent_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM oms_intents", [], |row| row.get(0))
            .unwrap();
        let trade: (String, String, String, i64) = conn
            .query_row(
                "SELECT reason, reason_code, confidence, sync_run_id FROM trades LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        let oms_fill_sync_run_id: i64 = conn
            .query_row("SELECT sync_run_id FROM oms_fills LIMIT 1", [], |row| {
                row.get(0)
            })
            .unwrap();
        let intent_sync_run_id: i64 = conn
            .query_row("SELECT sync_run_id FROM oms_intents LIMIT 1", [], |row| {
                row.get(0)
            })
            .unwrap();

        assert_eq!(intent_count, 1);
        assert_eq!(stats.inserted_manual_intents, 1);
        assert_eq!(trade.0, "manual_trade");
        assert_eq!(trade.1, "manual_trade");
        assert_eq!(trade.2, "MANUAL");
        assert_eq!(trade.3, 101);
        assert_eq!(oms_fill_sync_run_id, 101);
        assert_eq!(intent_sync_run_id, 101);
    }

    #[test]
    fn remote_fill_matches_existing_intent_by_client_order_id() {
        let db = temp_db();
        let oms = LiveOms::new(db.path()).unwrap();
        let reason_meta = json!({
            "phase": "entry",
            "reason_code": "entry_signal"
        });
        let intent = oms
            .create_intent(CreateIntentRequest {
                symbol: "ETH",
                action: "OPEN",
                side: "BUY",
                requested_size: Some(0.5),
                requested_notional: Some(1150.0),
                leverage: Some(2.0),
                decision_ts_ms: Some(1_772_000_000_000_i64),
                reason: Some("Signal Trigger"),
                confidence: Some("high"),
                entry_atr: Some(12.0),
                meta: Some(&reason_meta),
                dedupe_open: false,
                strategy_version: None,
                strategy_sha1: None,
            })
            .unwrap();

        let fill = HyperliquidFill {
            raw: json!({
                "coin": "ETH",
                "px": "2300",
                "sz": "0.5",
                "time": 1_772_000_001_000_i64,
                "dir": "Open Long",
                "startPosition": "0",
                "fee": "0.4025",
                "closedPnl": "0",
                "hash": "0xaiqfill0001",
                "tid": 102,
                "oid": "9002",
                "cloid": intent.client_order_id.as_deref().unwrap()
            }),
        };
        let mut stats = SyncStats::default();
        reconcile_remote_fills(db.path(), &oms, &[fill], "aiq_", 102, &mut stats).unwrap();

        let conn = Connection::open(db.path()).unwrap();
        let trade: (String, String, String, f64) = conn
            .query_row(
                "SELECT reason, reason_code, confidence, leverage FROM trades LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();

        assert_eq!(trade.0, "Signal Trigger");
        assert_eq!(trade.1, "entry_signal");
        assert_eq!(trade.2, "high");
        assert_eq!(trade.3, 2.0);
        assert_eq!(stats.inserted_trades, 1);
    }

    #[test]
    fn backfill_existing_oms_fill_without_trade_inserts_trade() {
        let db = temp_db();
        let conn = Connection::open(db.path()).unwrap();
        let raw_fill = manual_fill();
        conn.execute(
            "
            INSERT INTO oms_fills (
                ts_ms, symbol, intent_id, order_id, action, side, pos_type, price, size, notional,
                fee_usd, fee_token, fee_rate, pnl_usd, fill_hash, fill_tid, matched_via, raw_json
            ) VALUES (
                ?1, ?2, NULL, ?3, ?4, ?5, ?6, ?7, ?8, ?9,
                ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17
            )
            ",
            params![
                1_772_000_000_000_i64,
                "ETH",
                9001_i64,
                "OPEN",
                "BUY",
                "LONG",
                2300.0_f64,
                0.5_f64,
                1150.0_f64,
                0.4025_f64,
                Option::<String>::None,
                0.00035_f64,
                0.0_f64,
                "0xmanualfill0001",
                101_i64,
                "manual_orphan",
                raw_fill.to_string()
            ],
        )
        .unwrap();
        drop(conn);

        let oms = LiveOms::new(db.path()).unwrap();
        let mut stats = SyncStats::default();
        backfill_existing_oms_trades(db.path(), &oms, "aiq_", 103, &mut stats).unwrap();

        let conn = Connection::open(db.path()).unwrap();
        let trade_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM trades", [], |row| row.get(0))
            .unwrap();
        assert_eq!(trade_count, 1);
        assert_eq!(stats.backfilled_existing_trades, 1);
    }

    #[test]
    fn unsupported_remote_fill_does_not_fail_when_already_present_locally() {
        let db = temp_db();
        let conn = Connection::open(db.path()).unwrap();
        let fill_hash = "0x2836bb61531ec60e29b00436ce797102143c0046ee11e4e0cbff66b412129ff8";
        let fill_tid = 855885986782764_i64;
        let raw_fill = json!({
            "coin": "DOGE",
            "px": "0.20",
            "sz": "1000",
            "time": 1_773_162_768_225_i64,
            "dir": "Long > Short",
            "startPosition": "500",
            "fee": "0.07",
            "closedPnl": "3.5",
            "hash": fill_hash,
            "tid": fill_tid,
            "oid": "7001"
        });
        conn.execute(
            "
            INSERT INTO oms_fills (
                ts_ms, symbol, intent_id, order_id, action, side, pos_type, price, size, notional,
                fee_usd, fee_token, fee_rate, pnl_usd, fill_hash, fill_tid, matched_via, raw_json
            ) VALUES (
                ?1, ?2, NULL, ?3, ?4, ?5, ?6, ?7, ?8, ?9,
                ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17
            )
            ",
            params![
                1_773_162_768_225_i64,
                "DOGE",
                7001_i64,
                "UNKNOWN",
                "BUY",
                "UNKNOWN",
                0.20_f64,
                1000.0_f64,
                200.0_f64,
                0.07_f64,
                Option::<String>::None,
                Option::<f64>::None,
                3.5_f64,
                fill_hash,
                fill_tid,
                "exchange_order_id",
                raw_fill.to_string()
            ],
        )
        .unwrap();
        conn.execute(
            "
            INSERT INTO trades (
                timestamp, symbol, type, action, price, size, notional, reason, reason_code,
                confidence, pnl, fee_usd, fee_token, fee_rate, balance, entry_atr, leverage,
                margin_used, meta_json, fill_hash, fill_tid
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9,
                ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17,
                ?18, ?19, ?20, ?21
            )
            ",
            params![
                "2026-03-10T17:12:48.225000+00:00",
                "DOGE",
                "LONG",
                "CLOSE",
                0.20_f64,
                500.0_f64,
                100.0_f64,
                "manual_trade",
                "manual_trade",
                "MANUAL",
                3.5_f64,
                0.07_f64,
                Option::<String>::None,
                Option::<f64>::None,
                Option::<f64>::None,
                Option::<f64>::None,
                Option::<f64>::None,
                Option::<f64>::None,
                "{}",
                fill_hash,
                fill_tid
            ],
        )
        .unwrap();
        drop(conn);

        let oms = LiveOms::new(db.path()).unwrap();
        let mut stats = SyncStats::default();
        reconcile_remote_fills(
            db.path(),
            &oms,
            &[HyperliquidFill { raw: raw_fill }],
            "aiq_",
            104,
            &mut stats,
        )
        .unwrap();

        assert_eq!(stats.unsupported_remote_fills, 0);
        assert!(stats
            .warnings
            .iter()
            .any(|warning| warning.contains("already present locally")));
    }

    #[test]
    fn ensure_sync_schema_backfills_account_snapshot_event_reference_on_legacy_db() {
        let file = NamedTempFile::new().unwrap();
        let conn = Connection::open(file.path()).unwrap();
        conn.execute_batch(
            "
            CREATE TABLE runtime_account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                account_value_usd REAL NOT NULL,
                withdrawable_usd REAL NOT NULL,
                total_margin_used_usd REAL NOT NULL,
                source TEXT NOT NULL,
                meta_json TEXT,
                sync_run_id INTEGER
            );
            ",
        )
        .unwrap();
        drop(conn);

        ensure_sync_schema(file.path()).unwrap();

        let conn = Connection::open(file.path()).unwrap();
        assert!(table_column_exists(
            &conn,
            "runtime_account_snapshots",
            "account_snapshot_event_id",
        )
        .unwrap());
        assert!(conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='exchange_account_snapshot_events'",
                [],
                |_| Ok(()),
            )
            .optional()
            .unwrap()
            .is_some());
    }

    #[test]
    fn persist_exchange_snapshot_writes_account_and_position_tables() {
        let db = temp_db();
        let observation = crate::live_hyperliquid::HyperliquidClearinghouseStateObservation {
            observed_at_ts_ms: 1_773_000_000_111_i64,
            account_snapshot: crate::live_hyperliquid::HyperliquidAccountSnapshot {
                account_value_usd: 200.5,
                withdrawable_usd: 52.25,
                total_margin_used_usd: 148.25,
            },
            positions: vec![
                crate::live_hyperliquid::HyperliquidPosition {
                    symbol: "DOGE".to_string(),
                    pos_type: "LONG".to_string(),
                    size: 1692.0,
                    entry_price: 0.094602,
                    leverage: 4.0,
                    margin_used: 40.0,
                },
                crate::live_hyperliquid::HyperliquidPosition {
                    symbol: "HYPE".to_string(),
                    pos_type: "SHORT".to_string(),
                    size: 5.34,
                    entry_price: 37.2767,
                    leverage: 4.0,
                    margin_used: 50.6,
                },
            ],
            payload_json: json!({
                "marginSummary": {
                    "accountValue": "200.5",
                    "totalMarginUsed": "148.25"
                },
                "withdrawable": "52.25",
                "assetPositions": [
                    {
                        "position": {
                            "coin": "DOGE",
                            "entryPx": "0.094602",
                            "marginUsed": "40.0",
                            "leverage": { "value": "4" },
                            "szi": "1692.0"
                        }
                    },
                    {
                        "position": {
                            "coin": "HYPE",
                            "entryPx": "37.2767",
                            "marginUsed": "50.6",
                            "leverage": { "value": "4" },
                            "szi": "-5.34"
                        }
                    }
                ]
            })
            .to_string(),
            payload_digest: "digest-123".to_string(),
        };

        let account_snapshot_event_id =
            persist_account_snapshot_event(db.path(), &observation, 105, "0xabc", false).unwrap();

        let conn = Connection::open(db.path()).unwrap();
        let snapshot_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM runtime_account_snapshots", [], |row| {
                row.get(0)
            })
            .unwrap();
        let position_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM runtime_exchange_positions", [], |row| {
                row.get(0)
            })
            .unwrap();
        drop(conn);

        assert_eq!(snapshot_count, 0);
        assert_eq!(position_count, 0);

        persist_exchange_snapshot_projection(
            db.path(),
            &observation,
            account_snapshot_event_id,
            105,
            false,
        )
        .unwrap();

        let conn = Connection::open(db.path()).unwrap();
        let account_row: (i64, String, f64, f64, f64, i64, i64) = conn
            .query_row(
                "SELECT ts_ms, timestamp, account_value_usd, withdrawable_usd, total_margin_used_usd, sync_run_id, account_snapshot_event_id
                 FROM runtime_account_snapshots ORDER BY ts_ms DESC LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                    ))
                },
            )
            .unwrap();
        let evidence_row: (i64, i64, i64, String, String, String, String, String, i64) = conn
            .query_row(
                "SELECT id, sync_run_id, observed_at_ts_ms, source, wallet_address, request_type, raw_payload_json, payload_digest, position_count
                 FROM exchange_account_snapshot_events ORDER BY observed_at_ts_ms DESC LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                        row.get(7)?,
                        row.get(8)?,
                    ))
                },
            )
            .unwrap();
        let position_row: (i64, i64) = conn
            .query_row(
                "SELECT COUNT(*), MIN(sync_run_id) FROM runtime_exchange_positions",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();

        assert_eq!(
            account_row,
            (
                observation.observed_at_ts_ms,
                timestamp_from_ms(observation.observed_at_ts_ms),
                200.5,
                52.25,
                148.25,
                105,
                account_snapshot_event_id,
            )
        );
        assert_eq!(
            evidence_row,
            (
                account_row.6,
                105,
                observation.observed_at_ts_ms,
                LIVE_FILL_SYNC_SOURCE.to_string(),
                "0xabc".to_string(),
                CLEARINGHOUSE_STATE_REQUEST_TYPE.to_string(),
                observation.payload_json.clone(),
                "digest-123".to_string(),
                2,
            )
        );
        assert_eq!(position_row, (2, 105));
    }
}
