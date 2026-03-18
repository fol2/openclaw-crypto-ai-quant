use anyhow::{Context, Result};
use chrono::Utc;
use fs2::FileExt;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use serde_json::json;
use signal_hook::{consts::signal::SIGINT, consts::signal::SIGTERM, flag, SigId};
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::live_cycle::{self, LiveCycleInput, LiveCycleReport};
use crate::live_hyperliquid::HyperliquidInfoClient;
use crate::live_state::{build_strategy_state, ensure_live_runtime_tables, sync_exchange_positions};
use crate::paper_config::PaperEffectiveConfig;

pub struct AssistDaemonInput<'a> {
    pub effective_config: PaperEffectiveConfig,
    pub runtime_bootstrap: aiq_runtime_core::runtime::RuntimeBootstrap,
    pub live_db: &'a Path,
    pub candles_db: &'a Path,
    pub explicit_symbols: &'a [String],
    pub symbols_file: Option<&'a Path>,
    pub btc_symbol: &'a str,
    pub lookback_bars: usize,
    pub wallet_address: &'a str,
    pub lock_path: Option<&'a Path>,
    pub status_path: Option<&'a Path>,
    pub idle_sleep_ms: u64,
    pub max_idle_polls: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AssistDaemonReport {
    pub ok: bool,
    pub pid: u32,
    pub lock_path: String,
    pub status_path: String,
    pub started_at_ms: i64,
    pub stopped_at_ms: i64,
    pub stop_requested: bool,
    pub runtime_bootstrap: aiq_runtime_core::runtime::RuntimeBootstrap,
    pub last_cycle: Option<LiveCycleReport>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistDaemonStatus {
    pub ok: bool,
    pub running: bool,
    pub pid: u32,
    pub config_path: String,
    pub config_id: String,
    pub live_db: String,
    pub candles_db: String,
    pub lock_path: String,
    pub status_path: String,
    pub started_at_ms: i64,
    pub updated_at_ms: i64,
    pub stopped_at_ms: Option<i64>,
    pub stop_requested: bool,
    pub btc_symbol: String,
    pub lookback_bars: usize,
    pub explicit_symbols: Vec<String>,
    pub symbols_file: Option<String>,
    pub latest_common_close_ts_ms: Option<i64>,
    pub last_step_close_ts_ms: Option<i64>,
    pub last_plans_count: usize,
    pub last_signals_count: usize,
    pub last_tunnel_count: usize,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

pub fn run_daemon(input: AssistDaemonInput<'_>) -> Result<AssistDaemonReport> {
    if !input.live_db.exists() {
        anyhow::bail!("live db not found: {}", input.live_db.display());
    }
    if !input.candles_db.exists() {
        anyhow::bail!("candles db not found: {}", input.candles_db.display());
    }

    let client = HyperliquidInfoClient::new(input.wallet_address, Some(4.0))?;
    let lock_path = resolve_lock_path(input.lock_path);
    let status_path = resolve_status_path(input.status_path, &lock_path);
    let _lock_file = acquire_lock(&lock_path)?;
    let started_at_ms = Utc::now().timestamp_millis();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let _signal_guard = install_signal_handlers(&stop_flag)?;

    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    let mut stop_requested = false;
    let mut idle_polls = 0usize;
    let mut last_step_close_ts_ms = None;
    let mut last_cycle = None;
    let mut last_signals_count: usize = 0;

    ensure_live_runtime_tables(&rusqlite::Connection::open(input.live_db)?)?;
    write_status(
        &status_path,
        &build_status(
            &input,
            &lock_path,
            started_at_ms,
            StatusContext {
                latest_common_close_ts_ms: None,
                last_step_close_ts_ms,
                last_signals_count: 0,
                warnings: &[],
                errors: &[],
                last_cycle: None,
                stopped: false,
            },
        ),
    )?;

    eprintln!(
        "[assist] started pid={} lock={} wallet={}…{}",
        std::process::id(),
        lock_path.display(),
        &input.wallet_address[..6],
        &input.wallet_address[input.wallet_address.len() - 4..],
    );

    loop {
        if stop_flag.load(Ordering::Relaxed) {
            stop_requested = true;
            warnings.push("assist daemon stop requested".to_string());
            break;
        }

        let symbols = load_symbols(input.explicit_symbols, input.symbols_file)?;

        // Read-only exchange sync — positions + account snapshot in one call
        let (account_snapshot, exchange_positions) = match client.account_and_positions() {
            Ok(result) => result,
            Err(err) => {
                warnings.push(format!("assist exchange sync failed: {err:#}"));
                sleep_with_stop_flag(input.idle_sleep_ms, &stop_flag);
                continue;
            }
        };
        let observed_at_ms = Utc::now().timestamp_millis();

        // Sync position_state for strategy state building
        sync_exchange_positions(input.live_db, &exchange_positions, observed_at_ms)?;
        // Persist runtime_exchange_positions + account snapshot for Hub display
        persist_exchange_snapshot(
            input.live_db,
            &account_snapshot,
            &exchange_positions,
            observed_at_ms,
        )?;

        let state = build_strategy_state(
            input.live_db,
            &account_snapshot,
            &exchange_positions,
            observed_at_ms,
        )?;
        let active_symbols = active_symbols(&symbols, &state);
        if active_symbols.is_empty() {
            warnings.push("assist daemon idle: no configured symbols or open positions".to_string());
            idle_polls = idle_polls.saturating_add(1);
            if input.max_idle_polls > 0 && idle_polls >= input.max_idle_polls {
                break;
            }
            sleep_with_stop_flag(input.idle_sleep_ms, &stop_flag);
            continue;
        }

        let interval = input
            .effective_config
            .resolve_shared_interval(&active_symbols, true)?;
        if last_step_close_ts_ms.is_none() {
            last_step_close_ts_ms = load_last_assist_step_close_ts_ms(
                input.live_db,
                &input.runtime_bootstrap.config_fingerprint,
                &interval,
            )?;
        }
        let latest_common_close_ts_ms = latest_common_close_ts_ms(
            input.candles_db,
            &interval,
            &active_symbols,
            input.btc_symbol,
        )?;
        let interval_ms = interval_to_ms(&interval)?;
        let next_due_step_close_ts_ms = last_step_close_ts_ms
            .and_then(|value| value.checked_add(interval_ms))
            .unwrap_or(latest_common_close_ts_ms);

        if next_due_step_close_ts_ms > latest_common_close_ts_ms {
            idle_polls = idle_polls.saturating_add(1);
            write_status(
                &status_path,
                &build_status(
                    &input,
                    &lock_path,
                    started_at_ms,
                    StatusContext {
                        latest_common_close_ts_ms: Some(latest_common_close_ts_ms),
                        last_step_close_ts_ms,
                        last_signals_count,
                        warnings: &warnings,
                        errors: &errors,
                        last_cycle: last_cycle.as_ref(),
                        stopped: false,
                    },
                ),
            )?;
            if input.max_idle_polls > 0 && idle_polls >= input.max_idle_polls {
                warnings.push(format!(
                    "assist daemon follow exhausted after {} idle poll(s)",
                    input.max_idle_polls
                ));
                break;
            }
            sleep_with_stop_flag(input.idle_sleep_ms, &stop_flag);
            continue;
        }

        // Run the strategy cycle — signals + exit tunnel computation
        let cycle_report = live_cycle::run_cycle(LiveCycleInput {
            effective_config: input.effective_config.clone(),
            runtime_bootstrap: input.runtime_bootstrap.clone(),
            state,
            explicit_symbols: &symbols,
            candles_db: input.candles_db,
            btc_symbol: input.btc_symbol,
            lookback_bars: input.lookback_bars,
            step_close_ts_ms: next_due_step_close_ts_ms,
        })?;
        warnings.extend(cycle_report.warnings.iter().cloned());
        errors.extend(cycle_report.errors.iter().cloned());

        // SAFETY: NO submit_cycle_plans() — assist mode never places orders
        let plans_count = cycle_report.plans.len();
        if plans_count > 0 {
            eprintln!(
                "[assist] step={} would-be plans: {}",
                next_due_step_close_ts_ms,
                cycle_report
                    .plans
                    .iter()
                    .map(|p| format!("{} {} {} @{:.2}", p.symbol, p.action, p.side, p.reference_price))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        // Persist exit tunnel rows
        persist_exit_tunnel_rows(input.live_db, &cycle_report.tunnel_rows)?;

        // Persist signal rows from the cycle
        let signals_written = persist_signal_rows(
            input.live_db,
            &cycle_report,
            next_due_step_close_ts_ms,
        )?;

        // Record runtime cycle step
        record_runtime_cycle_step(
            input.live_db,
            &input.runtime_bootstrap.config_fingerprint,
            &interval,
            next_due_step_close_ts_ms,
            &cycle_report.active_symbols,
            plans_count,
        )?;

        eprintln!(
            "[assist] step={} symbols={} tunnel={} signals={} plans={} (discarded)",
            next_due_step_close_ts_ms,
            cycle_report.active_symbols.len(),
            cycle_report.tunnel_rows.len(),
            signals_written,
            plans_count,
        );

        last_step_close_ts_ms = Some(next_due_step_close_ts_ms);
        last_cycle = Some(cycle_report);
        last_signals_count = signals_written;
        idle_polls = 0;

        write_status(
            &status_path,
            &build_status(
                &input,
                &lock_path,
                started_at_ms,
                StatusContext {
                    latest_common_close_ts_ms: Some(latest_common_close_ts_ms),
                    last_step_close_ts_ms,
                    last_signals_count,
                    warnings: &warnings,
                    errors: &errors,
                    last_cycle: last_cycle.as_ref(),
                    stopped: false,
                },
            ),
        )?;
        sleep_with_stop_flag(input.idle_sleep_ms, &stop_flag);
    }

    let stopped_at_ms = Utc::now().timestamp_millis();
    write_status(
        &status_path,
        &build_status(
            &input,
            &lock_path,
            started_at_ms,
            StatusContext {
                latest_common_close_ts_ms: None,
                last_step_close_ts_ms,
                last_signals_count,
                warnings: &warnings,
                errors: &errors,
                last_cycle: last_cycle.as_ref(),
                stopped: true,
            },
        )
        .with_stopped_at(stopped_at_ms, stop_requested),
    )?;

    Ok(AssistDaemonReport {
        ok: errors.is_empty(),
        pid: std::process::id(),
        lock_path: lock_path.display().to_string(),
        status_path: status_path.display().to_string(),
        started_at_ms,
        stopped_at_ms,
        stop_requested,
        runtime_bootstrap: input.runtime_bootstrap,
        last_cycle,
        warnings,
        errors,
    })
}

// ---------------------------------------------------------------------------
// Signal rows persistence
// ---------------------------------------------------------------------------

fn ensure_signals_table(conn: &rusqlite::Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            signal TEXT,
            confidence TEXT,
            price REAL,
            rsi REAL,
            ema_fast REAL,
            ema_slow REAL,
            meta_json TEXT,
            run_fingerprint TEXT
        );",
    )?;
    Ok(())
}

fn persist_signal_rows(
    live_db: &Path,
    cycle_report: &LiveCycleReport,
    step_close_ts_ms: i64,
) -> Result<usize> {
    let conn = rusqlite::Connection::open(live_db)?;
    ensure_signals_table(&conn)?;

    let run_fingerprint = format!("assist_{}", step_close_ts_ms);
    let timestamp = chrono::DateTime::from_timestamp_millis(step_close_ts_ms)
        .map(|value| value.to_rfc3339())
        .unwrap_or_else(|| Utc::now().to_rfc3339());

    let mut count = 0usize;
    for plan in &cycle_report.plans {
        let signal = if plan.side == "BUY" {
            "BUY"
        } else if plan.side == "SELL" {
            "SELL"
        } else {
            continue;
        };

        let meta = json!({
            "source": "assist",
            "phase": plan.phase,
            "action": plan.action,
            "reason": plan.reason,
            "reason_code": plan.reason_code,
            "score": plan.score,
            "behaviour_trace": plan.behaviour_trace.iter().map(|t| format!("{:?}", t)).collect::<Vec<_>>(),
        });

        conn.execute(
            "INSERT INTO signals (timestamp, symbol, signal, confidence, price, rsi, ema_fast, ema_slow, meta_json, run_fingerprint)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                timestamp,
                plan.symbol,
                signal,
                plan.confidence,
                plan.reference_price,
                Option::<f64>::None,
                Option::<f64>::None,
                Option::<f64>::None,
                meta.to_string(),
                run_fingerprint,
            ],
        )?;
        count += 1;
    }
    Ok(count)
}

// ---------------------------------------------------------------------------
// Exchange snapshot persistence (for Hub display)
// ---------------------------------------------------------------------------

fn persist_exchange_snapshot(
    live_db: &Path,
    account_snapshot: &crate::live_hyperliquid::HyperliquidAccountSnapshot,
    exchange_positions: &[crate::live_hyperliquid::HyperliquidPosition],
    ts_ms: i64,
) -> Result<()> {
    let timestamp = chrono::DateTime::from_timestamp_millis(ts_ms)
        .map(|value| value.to_rfc3339())
        .unwrap_or_else(|| Utc::now().to_rfc3339());
    let conn = rusqlite::Connection::open(live_db)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS runtime_account_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            account_value_usd REAL NOT NULL,
            withdrawable_usd REAL NOT NULL,
            total_margin_used_usd REAL NOT NULL,
            source TEXT NOT NULL,
            meta_json TEXT
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
            source TEXT NOT NULL
        );",
    )?;
    conn.execute(
        "INSERT INTO runtime_account_snapshots (
            ts_ms, timestamp, account_value_usd, withdrawable_usd, total_margin_used_usd, source, meta_json
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            ts_ms,
            timestamp,
            account_snapshot.account_value_usd,
            account_snapshot.withdrawable_usd,
            account_snapshot.total_margin_used_usd,
            "assist",
            json!({
                "position_count": exchange_positions.len(),
            })
            .to_string(),
        ],
    )?;
    conn.execute("DELETE FROM runtime_exchange_positions", [])?;
    for position in exchange_positions {
        conn.execute(
            "INSERT INTO runtime_exchange_positions (
                symbol, pos_type, size, entry_price, leverage, margin_used, ts_ms, updated_at, source
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                position.symbol.trim().to_ascii_uppercase(),
                position.pos_type.trim().to_ascii_uppercase(),
                position.size,
                position.entry_price,
                position.leverage,
                position.margin_used,
                ts_ms,
                timestamp,
                "assist",
            ],
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Exit tunnel persistence (duplicated from live_daemon for module isolation)
// ---------------------------------------------------------------------------

fn persist_exit_tunnel_rows(
    live_db: &Path,
    rows: &[crate::paper_cycle::ExitTunnelRow],
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let conn = rusqlite::Connection::open(live_db)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS exit_tunnel (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            open_time_ms INTEGER NOT NULL DEFAULT 0,
            upper_full REAL NOT NULL,
            has_upper_full INTEGER NOT NULL DEFAULT 1,
            upper_partial REAL,
            lower_full REAL NOT NULL,
            has_lower_full INTEGER NOT NULL DEFAULT 1,
            entry_price REAL NOT NULL,
            pos_type TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_exit_tunnel_sym_ts ON exit_tunnel(symbol, ts_ms);",
    )?;
    ensure_exit_tunnel_columns(&conn)?;
    for row in rows {
        conn.execute(
            "INSERT INTO exit_tunnel (ts_ms, symbol, open_time_ms, upper_full, has_upper_full, upper_partial, lower_full, has_lower_full, entry_price, pos_type) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                row.ts_ms,
                row.symbol,
                row.open_time_ms,
                row.upper_full,
                row.has_upper_full,
                row.upper_partial,
                row.lower_full,
                row.has_lower_full,
                row.entry_price,
                row.pos_type,
            ],
        )?;
    }
    Ok(())
}

fn ensure_exit_tunnel_columns(conn: &rusqlite::Connection) -> Result<()> {
    add_column_if_missing(
        conn,
        "ALTER TABLE exit_tunnel ADD COLUMN has_upper_full INTEGER NOT NULL DEFAULT 1",
    )?;
    add_column_if_missing(
        conn,
        "ALTER TABLE exit_tunnel ADD COLUMN has_lower_full INTEGER NOT NULL DEFAULT 1",
    )?;
    add_column_if_missing(
        conn,
        "ALTER TABLE exit_tunnel ADD COLUMN open_time_ms INTEGER NOT NULL DEFAULT 0",
    )?;
    Ok(())
}

fn add_column_if_missing(conn: &rusqlite::Connection, sql: &str) -> Result<()> {
    match conn.execute(sql, []) {
        Ok(_) => Ok(()),
        Err(rusqlite::Error::SqliteFailure(_, Some(message)))
            if message.contains("duplicate column name") =>
        {
            Ok(())
        }
        Err(err) => Err(err.into()),
    }
}

// ---------------------------------------------------------------------------
// Helpers (adapted from live_daemon)
// ---------------------------------------------------------------------------

fn active_symbols(
    symbols: &[String],
    state: &bt_core::decision_kernel::StrategyState,
) -> Vec<String> {
    let mut merged = BTreeSet::new();
    merged.extend(
        symbols
            .iter()
            .map(|symbol| symbol.trim().to_ascii_uppercase())
            .filter(|symbol| !symbol.is_empty()),
    );
    merged.extend(state.positions.keys().cloned());
    merged.into_iter().collect()
}

fn load_symbols(explicit_symbols: &[String], symbols_file: Option<&Path>) -> Result<Vec<String>> {
    let mut merged = explicit_symbols.to_vec();
    if merged.is_empty() {
        if let Ok(raw) = std::env::var("AI_QUANT_SYMBOLS") {
            merged.extend(
                raw.split(',')
                    .map(str::trim)
                    .filter(|symbol| !symbol.is_empty())
                    .map(ToOwned::to_owned),
            );
        }
    }
    if let Some(symbols_file) = symbols_file {
        let file_symbols = std::fs::read_to_string(symbols_file)
            .with_context(|| format!("failed to read symbols file: {}", symbols_file.display()))?
            .lines()
            .map(str::trim)
            .filter(|symbol| !symbol.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        merged.extend(file_symbols);
    }
    let mut symbols = merged
        .into_iter()
        .map(|symbol| symbol.trim().to_ascii_uppercase())
        .filter(|symbol| !symbol.is_empty())
        .collect::<Vec<_>>();
    symbols.sort();
    symbols.dedup();
    Ok(symbols)
}

fn latest_common_close_ts_ms(
    candles_db: &Path,
    interval: &str,
    active_symbols: &[String],
    btc_symbol: &str,
) -> Result<i64> {
    let conn = rusqlite::Connection::open(candles_db)?;
    let mut symbols = BTreeSet::new();
    symbols.extend(active_symbols.iter().cloned());
    let btc_symbol = btc_symbol.trim().to_ascii_uppercase();
    if !btc_symbol.is_empty() {
        symbols.insert(btc_symbol);
    }
    let mut latest_common: Option<i64> = None;
    for symbol in symbols {
        let latest_symbol_close = conn
            .query_row(
                "SELECT MAX(COALESCE(t_close, t)) FROM candles WHERE symbol = ?1 AND interval = ?2",
                params![symbol, interval],
                |row| row.get::<_, Option<i64>>(0),
            )?
            .with_context(|| {
                format!(
                    "assist daemon requires candle coverage for {} at {}",
                    symbol, interval
                )
            })?;
        latest_common = Some(match latest_common {
            Some(current) => current.min(latest_symbol_close),
            None => latest_symbol_close,
        });
    }
    latest_common.context("assist daemon requires at least one available candle close")
}

fn interval_to_ms(interval: &str) -> Result<i64> {
    let interval = interval.trim();
    if let Some(minutes) = interval.strip_suffix('m') {
        let minutes = minutes.parse::<i64>().context("invalid minute interval")?;
        return Ok(minutes * 60_000);
    }
    if let Some(hours) = interval.strip_suffix('h') {
        let hours = hours.parse::<i64>().context("invalid hour interval")?;
        return Ok(hours * 60 * 60_000);
    }
    anyhow::bail!("unsupported interval: {interval}")
}

fn load_last_assist_step_close_ts_ms(
    live_db: &Path,
    config_fingerprint: &str,
    interval: &str,
) -> Result<Option<i64>> {
    let conn = rusqlite::Connection::open(live_db)?;
    let exists: i64 = conn.query_row(
        "SELECT COUNT(1) FROM sqlite_master WHERE type = 'table' AND name = 'runtime_cycle_steps'",
        [],
        |row| row.get(0),
    )?;
    if exists == 0 {
        return Ok(None);
    }
    let mut stmt = conn.prepare(
        "SELECT step_id, step_close_ts_ms
         FROM runtime_cycle_steps
         WHERE interval = ?1
         ORDER BY step_close_ts_ms DESC",
    )?;
    let rows = stmt.query_map([interval], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })?;
    for row in rows {
        let (step_id, step_close_ts_ms) = row?;
        // Accept steps from both live and assist modes
        let expected_live = crate::paper_cycle::derive_step_id(
            config_fingerprint,
            interval,
            step_close_ts_ms,
            true,
        );
        let expected_assist = format!("assist_{}", expected_live);
        if step_id == expected_live || step_id == expected_assist {
            return Ok(Some(step_close_ts_ms));
        }
    }
    Ok(None)
}

fn record_runtime_cycle_step(
    live_db: &Path,
    config_fingerprint: &str,
    interval: &str,
    step_close_ts_ms: i64,
    active_symbols: &[String],
    execution_count: usize,
) -> Result<()> {
    let conn = rusqlite::Connection::open(live_db)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS runtime_cycle_steps (
            step_id TEXT PRIMARY KEY,
            step_close_ts_ms INTEGER NOT NULL,
            interval TEXT NOT NULL,
            symbols_json TEXT NOT NULL,
            snapshot_exported_at_ms INTEGER NOT NULL,
            execution_count INTEGER NOT NULL,
            trades_written INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )",
    )?;
    // Use the live step_id so that switching between assist and live is seamless
    let step_id =
        crate::paper_cycle::derive_step_id(config_fingerprint, interval, step_close_ts_ms, true);
    conn.execute(
        "INSERT OR REPLACE INTO runtime_cycle_steps (
            step_id, step_close_ts_ms, interval, symbols_json, snapshot_exported_at_ms,
            execution_count, trades_written, created_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            step_id,
            step_close_ts_ms,
            interval,
            serde_json::to_string(active_symbols)?,
            step_close_ts_ms,
            execution_count as i64,
            0_i64,
            Utc::now().to_rfc3339()
        ],
    )?;
    Ok(())
}

fn resolve_lock_path(lock_path: Option<&Path>) -> PathBuf {
    lock_path.map(Path::to_path_buf).unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("ai_quant_assist.lock")
    })
}

fn resolve_status_path(status_path: Option<&Path>, lock_path: &Path) -> PathBuf {
    status_path.map(Path::to_path_buf).unwrap_or_else(|| {
        let name = lock_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("ai_quant_assist.lock");
        let status_name = name
            .strip_suffix(".lock")
            .map(|name| format!("{name}.status.json"))
            .unwrap_or_else(|| format!("{name}.status.json"));
        lock_path
            .parent()
            .map(|parent| parent.join(&status_name))
            .unwrap_or_else(|| PathBuf::from(status_name))
    })
}

fn acquire_lock(lock_path: &Path) -> Result<File> {
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut lock_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)
        .with_context(|| format!("failed to open assist lock file: {}", lock_path.display()))?;
    lock_file
        .try_lock_exclusive()
        .with_context(|| format!("failed to acquire assist lock {}", lock_path.display()))?;
    lock_file.set_len(0)?;
    lock_file.seek(SeekFrom::Start(0))?;
    write!(lock_file, "{}", std::process::id())?;
    lock_file.flush()?;
    Ok(lock_file)
}

fn install_signal_handlers(stop_flag: &Arc<AtomicBool>) -> Result<Vec<SigId>> {
    Ok(vec![
        flag::register(SIGINT, Arc::clone(stop_flag))?,
        flag::register(SIGTERM, Arc::clone(stop_flag))?,
    ])
}

fn sleep_with_stop_flag(idle_sleep_ms: u64, stop_flag: &AtomicBool) {
    let sleep = Duration::from_millis(idle_sleep_ms.max(100));
    let step = Duration::from_millis(100);
    let mut elapsed = Duration::ZERO;
    while elapsed < sleep {
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }
        let current = std::cmp::min(step, sleep.saturating_sub(elapsed));
        std::thread::sleep(current);
        elapsed += current;
    }
}

struct StatusContext<'a> {
    latest_common_close_ts_ms: Option<i64>,
    last_step_close_ts_ms: Option<i64>,
    last_signals_count: usize,
    warnings: &'a [String],
    errors: &'a [String],
    last_cycle: Option<&'a LiveCycleReport>,
    stopped: bool,
}

fn build_status(
    input: &AssistDaemonInput<'_>,
    lock_path: &Path,
    started_at_ms: i64,
    context: StatusContext<'_>,
) -> AssistDaemonStatus {
    AssistDaemonStatus {
        ok: context.errors.is_empty(),
        running: !context.stopped,
        pid: std::process::id(),
        config_path: input.effective_config.config_path().display().to_string(),
        config_id: input.effective_config.config_id().to_string(),
        live_db: input.live_db.display().to_string(),
        candles_db: input.candles_db.display().to_string(),
        lock_path: lock_path.display().to_string(),
        status_path: resolve_status_path(input.status_path, lock_path)
            .display()
            .to_string(),
        started_at_ms,
        updated_at_ms: Utc::now().timestamp_millis(),
        stopped_at_ms: None,
        stop_requested: false,
        btc_symbol: input.btc_symbol.trim().to_ascii_uppercase(),
        lookback_bars: input.lookback_bars,
        explicit_symbols: input.explicit_symbols.to_vec(),
        symbols_file: input.symbols_file.map(|path| path.display().to_string()),
        latest_common_close_ts_ms: context.latest_common_close_ts_ms,
        last_step_close_ts_ms: context.last_step_close_ts_ms,
        last_plans_count: context
            .last_cycle
            .map(|cycle| cycle.plans.len())
            .unwrap_or(0),
        last_signals_count: context.last_signals_count,
        last_tunnel_count: context
            .last_cycle
            .map(|cycle| cycle.tunnel_rows.len())
            .unwrap_or(0),
        warnings: context.warnings.to_vec(),
        errors: context.errors.to_vec(),
    }
}

impl AssistDaemonStatus {
    fn with_stopped_at(mut self, stopped_at_ms: i64, stop_requested: bool) -> Self {
        self.running = false;
        self.stopped_at_ms = Some(stopped_at_ms);
        self.stop_requested = stop_requested;
        self.updated_at_ms = stopped_at_ms;
        self
    }
}

fn write_status(status_path: &Path, status: &AssistDaemonStatus) -> Result<()> {
    if let Some(parent) = status_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp_path = status_path.with_extension("status.json.tmp");
    let payload = serde_json::to_vec_pretty(status)?;
    std::fs::write(&tmp_path, payload)?;
    std::fs::rename(&tmp_path, status_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn resolve_status_path_derives_default_from_lock_path() {
        let lock_path = PathBuf::from("/tmp/ai_quant_assist.lock");
        let status_path = resolve_status_path(None, &lock_path);
        assert_eq!(
            status_path,
            PathBuf::from("/tmp/ai_quant_assist.status.json")
        );
    }

    #[test]
    fn ensure_signals_table_is_idempotent() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        ensure_signals_table(&conn).unwrap();
        ensure_signals_table(&conn).unwrap();
    }
}
