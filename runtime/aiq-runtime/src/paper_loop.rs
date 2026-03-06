use aiq_runtime_core::paper::restore_paper_state;
use aiq_runtime_core::runtime::RuntimeBootstrap;
use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{params, Connection};
use serde::Serialize;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::paper_cycle::{self, PaperCycleInput};
use crate::paper_export;

pub struct PaperLoopInput<'a> {
    pub runtime_bootstrap: RuntimeBootstrap,
    pub config_path: &'a Path,
    pub live: bool,
    pub paper_db: &'a Path,
    pub candles_db: &'a Path,
    pub explicit_symbols: &'a [String],
    pub btc_symbol: &'a str,
    pub lookback_bars: usize,
    pub start_step_close_ts_ms: Option<i64>,
    pub max_steps: usize,
    pub exported_at_ms: Option<i64>,
    pub dry_run: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperLoopStepReport {
    pub step_id: String,
    pub step_close_ts_ms: i64,
    pub snapshot_exported_at_ms: i64,
    pub active_symbols: Vec<String>,
    pub candidate_count: usize,
    pub executed_entry_count: usize,
    pub trades_written: usize,
    pub runtime_step_recorded: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperLoopReport {
    pub ok: bool,
    pub dry_run: bool,
    pub interval: Option<String>,
    pub explicit_symbols: Vec<String>,
    pub initial_last_applied_step_close_ts_ms: Option<i64>,
    pub latest_common_close_ts_ms: Option<i64>,
    pub next_due_step_close_ts_ms: Option<i64>,
    pub executed_steps: usize,
    pub runtime_bootstrap: RuntimeBootstrap,
    pub steps: Vec<PaperLoopStepReport>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
struct LoopContext {
    active_symbols: Vec<String>,
    interval: String,
    latest_common_close_ts_ms: i64,
    last_applied_step_close_ts_ms: Option<i64>,
}

struct WorkingPaperDb {
    path: PathBuf,
    cleanup: bool,
}

pub fn run_loop(input: PaperLoopInput<'_>) -> Result<PaperLoopReport> {
    if !input.paper_db.exists() {
        anyhow::bail!("paper db not found: {}", input.paper_db.display());
    }
    if !input.candles_db.exists() {
        anyhow::bail!("candles db not found: {}", input.candles_db.display());
    }
    if input.max_steps == 0 {
        anyhow::bail!("max_steps must be positive");
    }

    let explicit_symbols = normalise_symbols(input.explicit_symbols);
    let working_paper_db = prepare_working_paper_db(input.paper_db, input.dry_run)?;
    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    let mut steps = Vec::new();
    let mut initial_last_applied_step_close_ts_ms = None;
    let mut interval = None;
    let mut latest_common_close_ts_ms = None;
    let mut next_due_step_close_ts_ms = None;

    loop {
        if steps.len() >= input.max_steps {
            break;
        }

        let maybe_context = inspect_loop_context(
            &input.runtime_bootstrap,
            input.config_path,
            input.live,
            working_paper_db.path(),
            input.candles_db,
            &explicit_symbols,
            input.btc_symbol,
        )?;
        let Some(context) = maybe_context else {
            if steps.is_empty() {
                anyhow::bail!("paper loop requires explicit symbols or open paper positions");
            }
            warnings.push("paper loop stopped: no active symbols remain".to_string());
            break;
        };

        if initial_last_applied_step_close_ts_ms.is_none() {
            initial_last_applied_step_close_ts_ms = context.last_applied_step_close_ts_ms;
        }
        interval = Some(context.interval.clone());
        latest_common_close_ts_ms = Some(context.latest_common_close_ts_ms);

        let interval_ms = interval_to_ms(&context.interval).with_context(|| {
            format!("unsupported interval for paper loop: {}", context.interval)
        })?;
        let candidate_next_due = match context.last_applied_step_close_ts_ms {
            Some(last_applied_step_close_ts_ms) => {
                let expected_next = last_applied_step_close_ts_ms
                    .checked_add(interval_ms)
                    .context("paper loop interval overflow")?;
                if steps.is_empty() {
                    if let Some(start_step_close_ts_ms) = input.start_step_close_ts_ms {
                        if start_step_close_ts_ms != expected_next {
                            anyhow::bail!(
                                "paper loop start_step_close_ts_ms {} does not match next unapplied step {}",
                                start_step_close_ts_ms,
                                expected_next
                            );
                        }
                    }
                }
                expected_next
            }
            None => input.start_step_close_ts_ms.context(
                "paper loop requires --start-step-close-ts-ms when no prior runtime_cycle_steps exist",
            )?,
        };
        next_due_step_close_ts_ms = Some(candidate_next_due);
        if candidate_next_due > context.latest_common_close_ts_ms {
            if steps.is_empty() {
                warnings.push(format!(
                    "paper loop idle: next due step {} is newer than latest common close {}",
                    candidate_next_due, context.latest_common_close_ts_ms
                ));
            }
            break;
        }
        ensure_exact_step_candle_coverage(
            input.candles_db,
            &context.interval,
            &context.active_symbols,
            input.btc_symbol,
            candidate_next_due,
        )?;

        let cycle_report = paper_cycle::run_cycle(PaperCycleInput {
            runtime_bootstrap: input.runtime_bootstrap.clone(),
            config_path: input.config_path,
            live: input.live,
            paper_db: working_paper_db.path(),
            candles_db: input.candles_db,
            explicit_symbols: &explicit_symbols,
            btc_symbol: input.btc_symbol,
            lookback_bars: input.lookback_bars,
            step_close_ts_ms: candidate_next_due,
            exported_at_ms: Some(input.exported_at_ms.unwrap_or(candidate_next_due)),
            dry_run: false,
        })?;
        warnings.extend(cycle_report.warnings.iter().cloned());
        errors.extend(cycle_report.errors.iter().cloned());
        steps.push(PaperLoopStepReport::from_cycle_report(
            cycle_report,
            input.dry_run,
        ));
    }

    let final_context = inspect_loop_context(
        &input.runtime_bootstrap,
        input.config_path,
        input.live,
        working_paper_db.path(),
        input.candles_db,
        &explicit_symbols,
        input.btc_symbol,
    )?;
    if let Some(context) = final_context.as_ref() {
        interval = Some(context.interval.clone());
        latest_common_close_ts_ms = Some(context.latest_common_close_ts_ms);
        let interval_ms = interval_to_ms(&context.interval).with_context(|| {
            format!("unsupported interval for paper loop: {}", context.interval)
        })?;
        next_due_step_close_ts_ms = context
            .last_applied_step_close_ts_ms
            .and_then(|last_applied_step_close_ts_ms| {
                last_applied_step_close_ts_ms.checked_add(interval_ms)
            })
            .or(input.start_step_close_ts_ms);
    }

    Ok(PaperLoopReport {
        ok: errors.is_empty(),
        dry_run: input.dry_run,
        interval,
        explicit_symbols,
        initial_last_applied_step_close_ts_ms,
        latest_common_close_ts_ms,
        next_due_step_close_ts_ms,
        executed_steps: steps.len(),
        runtime_bootstrap: input.runtime_bootstrap,
        steps,
        warnings,
        errors,
    })
}

impl PaperLoopStepReport {
    fn from_cycle_report(value: paper_cycle::PaperCycleReport, dry_run: bool) -> Self {
        Self {
            step_id: value.step_id,
            step_close_ts_ms: value.step_close_ts_ms,
            snapshot_exported_at_ms: value.snapshot_exported_at_ms,
            active_symbols: value.active_symbols,
            candidate_count: value.candidate_count,
            executed_entry_count: value.executed_entry_count,
            trades_written: if dry_run { 0 } else { value.trades_written },
            runtime_step_recorded: if dry_run {
                false
            } else {
                value.runtime_step_recorded
            },
            warnings: value.warnings,
            errors: value.errors,
        }
    }
}

impl WorkingPaperDb {
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for WorkingPaperDb {
    fn drop(&mut self) {
        if !self.cleanup {
            return;
        }
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_file(format!("{}-wal", self.path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", self.path.display()));
    }
}

fn inspect_loop_context(
    runtime_bootstrap: &RuntimeBootstrap,
    config_path: &Path,
    live: bool,
    paper_db: &Path,
    candles_db: &Path,
    explicit_symbols: &[String],
    btc_symbol: &str,
) -> Result<Option<LoopContext>> {
    let snapshot = paper_export::export_paper_snapshot(paper_db, Utc::now().timestamp_millis())?;
    let (paper_state, _) = restore_paper_state(&snapshot).map_err(anyhow::Error::msg)?;

    let mut active_symbols = BTreeSet::new();
    active_symbols.extend(explicit_symbols.iter().cloned());
    active_symbols.extend(paper_state.positions.keys().cloned());
    let active_symbols = active_symbols.into_iter().collect::<Vec<_>>();
    if active_symbols.is_empty() {
        return Ok(None);
    }

    let interval = resolve_shared_interval(config_path, &active_symbols, live)?;
    let latest_common_close_ts_ms =
        latest_common_close_ts_ms(candles_db, &interval, &active_symbols, btc_symbol)?;
    let last_applied_step_close_ts_ms = load_last_applied_step_close_ts_ms(
        paper_db,
        &runtime_bootstrap.config_fingerprint,
        &interval,
        live,
    )?;

    Ok(Some(LoopContext {
        active_symbols,
        interval,
        latest_common_close_ts_ms,
        last_applied_step_close_ts_ms,
    }))
}

fn resolve_shared_interval(
    config_path: &Path,
    active_symbols: &[String],
    live: bool,
) -> Result<String> {
    let mut interval = None;
    for symbol in active_symbols {
        let config = bt_core::config::load_config_checked(
            config_path
                .to_str()
                .context("config path must be valid UTF-8")?,
            Some(symbol),
            live,
        )
        .map_err(anyhow::Error::msg)?;
        match interval.as_deref() {
            Some(current_interval) if current_interval != config.engine.interval => {
                anyhow::bail!(
                    "paper loop requires a shared interval; {} resolved to {} but prior symbols use {}",
                    symbol,
                    config.engine.interval,
                    current_interval
                );
            }
            None => interval = Some(config.engine.interval.clone()),
            _ => {}
        }
    }
    interval.context("paper loop requires at least one active symbol interval")
}

fn latest_common_close_ts_ms(
    candles_db: &Path,
    interval: &str,
    active_symbols: &[String],
    btc_symbol: &str,
) -> Result<i64> {
    let conn = Connection::open(candles_db)?;
    let mut symbols = BTreeSet::new();
    symbols.extend(active_symbols.iter().cloned());
    let btc_symbol = btc_symbol.trim().to_ascii_uppercase();
    if !btc_symbol.is_empty() {
        symbols.insert(btc_symbol);
    }

    let mut latest_common: Option<i64> = None;
    for symbol in symbols {
        let latest_symbol_close_ts_ms: Option<i64> = conn.query_row(
            "SELECT MAX(COALESCE(t_close, t)) FROM candles WHERE symbol = ?1 AND interval = ?2",
            params![symbol, interval],
            |row| row.get(0),
        )?;
        let latest_symbol_close_ts_ms = latest_symbol_close_ts_ms.with_context(|| {
            format!(
                "paper loop requires candle coverage for {} at interval {}",
                symbol, interval
            )
        })?;
        latest_common = Some(match latest_common {
            Some(current) => current.min(latest_symbol_close_ts_ms),
            None => latest_symbol_close_ts_ms,
        });
    }

    latest_common.context("paper loop requires at least one available candle close")
}

fn load_last_applied_step_close_ts_ms(
    paper_db: &Path,
    config_fingerprint: &str,
    interval: &str,
    live: bool,
) -> Result<Option<i64>> {
    let conn = Connection::open(paper_db)?;
    if !table_exists(&conn, "runtime_cycle_steps")? {
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
        let expected_step_id =
            paper_cycle::derive_step_id(config_fingerprint, interval, step_close_ts_ms, live);
        if step_id == expected_step_id {
            return Ok(Some(step_close_ts_ms));
        }
    }

    Ok(None)
}

fn table_exists(conn: &Connection, table_name: &str) -> Result<bool> {
    let exists: i64 = conn.query_row(
        "SELECT COUNT(1) FROM sqlite_master WHERE type = 'table' AND name = ?1",
        [table_name],
        |row| row.get(0),
    )?;
    Ok(exists > 0)
}

fn prepare_working_paper_db(paper_db: &Path, dry_run: bool) -> Result<WorkingPaperDb> {
    if !dry_run {
        return Ok(WorkingPaperDb {
            path: paper_db.to_path_buf(),
            cleanup: false,
        });
    }

    let temp_name = format!(
        "aiq-runtime-paper-loop-{}-{}.db",
        std::process::id(),
        Utc::now()
            .timestamp_nanos_opt()
            .unwrap_or_else(|| Utc::now().timestamp_micros() * 1_000)
    );
    let temp_path = std::env::temp_dir().join(temp_name);
    snapshot_sqlite_db(paper_db, &temp_path)?;
    Ok(WorkingPaperDb {
        path: temp_path,
        cleanup: true,
    })
}

fn ensure_exact_step_candle_coverage(
    candles_db: &Path,
    interval: &str,
    active_symbols: &[String],
    btc_symbol: &str,
    step_close_ts_ms: i64,
) -> Result<()> {
    let conn = Connection::open(candles_db)?;
    let mut symbols = BTreeSet::new();
    symbols.extend(active_symbols.iter().cloned());
    let btc_symbol = btc_symbol.trim().to_ascii_uppercase();
    if !btc_symbol.is_empty() {
        symbols.insert(btc_symbol);
    }

    for symbol in symbols {
        let step_exists: i64 = conn.query_row(
            "SELECT COUNT(1)
             FROM candles
             WHERE symbol = ?1 AND interval = ?2 AND COALESCE(t_close, t) = ?3",
            params![symbol, interval, step_close_ts_ms],
            |row| row.get(0),
        )?;
        if step_exists == 0 {
            anyhow::bail!(
                "paper loop requires an exact candle close at {} for {} on {}",
                step_close_ts_ms,
                symbol,
                interval
            );
        }
    }

    Ok(())
}

fn snapshot_sqlite_db(source_path: &Path, target_path: &Path) -> Result<()> {
    if target_path.exists() {
        std::fs::remove_file(target_path).with_context(|| {
            format!(
                "failed to remove existing temporary sqlite snapshot {}",
                target_path.display()
            )
        })?;
    }

    let conn = Connection::open(source_path)?;
    let quoted_target_path = target_path
        .to_string_lossy()
        .replace('\'', "''");
    conn.execute_batch(&format!("VACUUM main INTO '{}';", quoted_target_path))
        .with_context(|| {
            format!(
                "failed to snapshot sqlite db {} into {}",
                source_path.display(),
                target_path.display()
            )
        })?;
    Ok(())
}

fn interval_to_ms(interval: &str) -> Result<i64> {
    let trimmed = interval.trim();
    if trimmed.len() < 2 {
        anyhow::bail!("invalid interval: {}", interval);
    }
    let (number, unit) = trimmed.split_at(trimmed.len() - 1);
    let value = number
        .parse::<i64>()
        .with_context(|| format!("invalid interval value: {}", interval))?;
    let ms = match unit {
        "m" => value * 60_000,
        "h" => value * 3_600_000,
        _ => anyhow::bail!("unsupported interval unit: {}", interval),
    };
    Ok(ms)
}

fn normalise_symbols(raw: &[String]) -> Vec<String> {
    let mut symbols = raw
        .iter()
        .map(|symbol| symbol.trim().to_ascii_uppercase())
        .filter(|symbol| !symbol.is_empty())
        .collect::<Vec<_>>();
    symbols.sort();
    symbols.dedup();
    symbols
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiq_runtime_core::runtime::{build_bootstrap, RuntimeMode};
    use rusqlite::Connection;
    use tempfile::tempdir;

    const START_STEP_CLOSE_TS_MS: i64 = 1_773_422_400_000;
    const NEXT_STEP_CLOSE_TS_MS: i64 = 1_773_424_200_000;
    const LAST_STEP_CLOSE_TS_MS: i64 = 1_773_426_000_000;

    fn seed_paper_db(path: &Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE trades (
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
                fill_tid INTEGER
            );
            CREATE TABLE position_state (
                symbol TEXT PRIMARY KEY,
                open_trade_id INTEGER,
                trailing_sl REAL,
                last_funding_time INTEGER,
                adds_count INTEGER,
                tp1_taken INTEGER,
                last_add_time INTEGER,
                entry_adx_threshold REAL,
                updated_at TEXT
            );
            CREATE TABLE runtime_cooldowns (
                symbol TEXT PRIMARY KEY,
                last_entry_attempt_s REAL,
                last_exit_attempt_s REAL,
                updated_at TEXT
            );
            CREATE TABLE runtime_last_closes (
                symbol TEXT PRIMARY KEY,
                close_ts_ms INTEGER NOT NULL,
                side TEXT NOT NULL,
                reason TEXT,
                updated_at TEXT NOT NULL
            );
            "#,
        )
        .unwrap();
        conn.execute(
            "INSERT INTO trades (timestamp,symbol,action,type,price,size,notional,reason,confidence,balance,pnl,fee_usd,fee_rate,entry_atr,leverage,margin_used,meta_json)
             VALUES ('2026-03-05T10:00:00+00:00','ETH','OPEN','LONG',100.0,1.0,100.0,'seed','medium',1000.0,0.0,0.0,0.0,5.0,3.0,33.3,'{}')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO position_state VALUES ('ETH',1,95.0,1772676500000,0,0,1772676600000,22.0,'2026-03-05T10:08:20+00:00')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO runtime_cooldowns VALUES ('ETH',1772676500.0,1772676550.0,'2026-03-05T10:15:00+00:00')",
            [],
        )
        .unwrap();
    }

    fn seed_candles_db(path: &Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE candles (
                symbol TEXT,
                interval TEXT,
                t INTEGER,
                t_close INTEGER,
                o REAL,
                h REAL,
                l REAL,
                c REAL,
                v REAL,
                n INTEGER
            );
            "#,
        )
        .unwrap();

        let base = 1_772_670_000_000_i64;
        for (symbol, start, drift) in [("ETH", 100.0, 0.25), ("BTC", 50_000.0, 20.0)] {
            let mut price: f64 = start;
            for idx in 0..420_i64 {
                let t = base + (idx * 1_800_000);
                let open: f64 = price;
                let close: f64 = price + drift;
                let high = open.max(close) + 0.5;
                let low = open.min(close) - 0.5;
                let volume = 1000.0 + idx as f64;
                conn.execute(
                    "INSERT INTO candles VALUES (?1, '30m', ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1)",
                    (symbol, t, t + 1_800_000, open, high, low, close, volume),
                )
                .unwrap();
                price = close;
            }
        }
    }

    fn seed_gapped_candles_db(path: &Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE candles (
                symbol TEXT,
                interval TEXT,
                t INTEGER,
                t_close INTEGER,
                o REAL,
                h REAL,
                l REAL,
                c REAL,
                v REAL,
                n INTEGER
            );
            "#,
        )
        .unwrap();

        let base = 1_772_670_000_000_i64;
        for (symbol, start, drift) in [("ETH", 100.0, 0.25), ("BTC", 50_000.0, 20.0)] {
            let mut price: f64 = start;
            for idx in 0..100_i64 {
                if idx == 90 {
                    price += drift;
                    continue;
                }
                let t = base + (idx * 1_800_000);
                let open: f64 = price;
                let close: f64 = price + drift;
                let high = open.max(close) + 0.5;
                let low = open.min(close) - 0.5;
                let volume = 1000.0 + idx as f64;
                conn.execute(
                    "INSERT INTO candles VALUES (?1, '30m', ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1)",
                    (symbol, t, t + 1_800_000, open, high, low, close, volume),
                )
                .unwrap();
                price = close;
            }
        }
    }

    fn base_cfg_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("config/strategy_overrides.yaml.example")
    }

    fn runtime_bootstrap() -> RuntimeBootstrap {
        let cfg =
            bt_core::config::load_config_checked(base_cfg_path().to_str().unwrap(), None, false)
                .unwrap();
        build_bootstrap(&cfg, RuntimeMode::Paper, None).unwrap()
    }

    #[test]
    fn loop_requires_bootstrap_start_when_no_prior_step_exists() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_paper_db(&paper_db);
        seed_candles_db(&candles_db);

        let err = run_loop(PaperLoopInput {
            runtime_bootstrap: runtime_bootstrap(),
            config_path: &base_cfg_path(),
            live: false,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["ETH".to_string()],
            btc_symbol: "BTC",
            lookback_bars: 400,
            start_step_close_ts_ms: None,
            max_steps: 1,
            exported_at_ms: None,
            dry_run: false,
        })
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("requires --start-step-close-ts-ms"));
    }

    #[test]
    fn loop_catches_up_multiple_steps_and_then_goes_idle() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_paper_db(&paper_db);
        seed_candles_db(&candles_db);

        let first = run_loop(PaperLoopInput {
            runtime_bootstrap: runtime_bootstrap(),
            config_path: &base_cfg_path(),
            live: false,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["ETH".to_string()],
            btc_symbol: "BTC",
            lookback_bars: 400,
            start_step_close_ts_ms: Some(START_STEP_CLOSE_TS_MS),
            max_steps: 2,
            exported_at_ms: None,
            dry_run: false,
        })
        .unwrap();

        assert_eq!(first.executed_steps, 2);
        assert_eq!(
            first
                .steps
                .iter()
                .map(|step| step.step_close_ts_ms)
                .collect::<Vec<_>>(),
            vec![START_STEP_CLOSE_TS_MS, NEXT_STEP_CLOSE_TS_MS]
        );
        let conn = Connection::open(&paper_db).unwrap();
        let recorded_steps: i64 = conn
            .query_row("SELECT COUNT(*) FROM runtime_cycle_steps", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(recorded_steps, 2);
        conn.close().unwrap();

        let second = run_loop(PaperLoopInput {
            runtime_bootstrap: runtime_bootstrap(),
            config_path: &base_cfg_path(),
            live: false,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["ETH".to_string()],
            btc_symbol: "BTC",
            lookback_bars: 400,
            start_step_close_ts_ms: None,
            max_steps: 2,
            exported_at_ms: None,
            dry_run: false,
        })
        .unwrap();

        assert_eq!(second.executed_steps, 1);
        assert_eq!(
            second
                .steps
                .iter()
                .map(|step| step.step_close_ts_ms)
                .collect::<Vec<_>>(),
            vec![LAST_STEP_CLOSE_TS_MS]
        );

        let third = run_loop(PaperLoopInput {
            runtime_bootstrap: runtime_bootstrap(),
            config_path: &base_cfg_path(),
            live: false,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["ETH".to_string()],
            btc_symbol: "BTC",
            lookback_bars: 400,
            start_step_close_ts_ms: None,
            max_steps: 1,
            exported_at_ms: None,
            dry_run: false,
        })
        .unwrap();

        assert_eq!(third.executed_steps, 0);
        assert_eq!(third.latest_common_close_ts_ms, Some(LAST_STEP_CLOSE_TS_MS));
        assert_eq!(
            third.next_due_step_close_ts_ms,
            Some(LAST_STEP_CLOSE_TS_MS + 1_800_000)
        );
    }

    #[test]
    fn loop_dry_run_advances_planned_steps_without_writing_runtime_cycle_steps() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_paper_db(&paper_db);
        seed_candles_db(&candles_db);

        let report = run_loop(PaperLoopInput {
            runtime_bootstrap: runtime_bootstrap(),
            config_path: &base_cfg_path(),
            live: false,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["ETH".to_string()],
            btc_symbol: "BTC",
            lookback_bars: 400,
            start_step_close_ts_ms: Some(START_STEP_CLOSE_TS_MS),
            max_steps: 2,
            exported_at_ms: None,
            dry_run: true,
        })
        .unwrap();

        assert!(report.dry_run);
        assert_eq!(report.executed_steps, 2);
        assert_eq!(
            report
                .steps
                .iter()
                .map(|step| step.step_close_ts_ms)
                .collect::<Vec<_>>(),
            vec![START_STEP_CLOSE_TS_MS, NEXT_STEP_CLOSE_TS_MS]
        );
        assert_eq!(report.steps[0].trades_written, 0);
        assert_eq!(report.steps[1].trades_written, 0);
        assert!(!report.steps[0].runtime_step_recorded);
        assert!(!report.steps[1].runtime_step_recorded);
        assert!(
            report.steps[1]
                .warnings
                .iter()
                .any(|warning| warning.contains("tp1_taken blocks same-direction pyramiding"))
        );

        let conn = Connection::open(&paper_db).unwrap();
        let step_table_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'runtime_cycle_steps'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(step_table_count, 0);
    }

    #[test]
    fn loop_rejects_missing_exact_step_close() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_paper_db(&paper_db);
        seed_gapped_candles_db(&candles_db);

        let err = run_loop(PaperLoopInput {
            runtime_bootstrap: runtime_bootstrap(),
            config_path: &base_cfg_path(),
            live: false,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["ETH".to_string()],
            btc_symbol: "BTC",
            lookback_bars: 400,
            start_step_close_ts_ms: Some(1_772_832_000_000),
            max_steps: 2,
            exported_at_ms: None,
            dry_run: false,
        })
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("requires an exact candle close at 1772833800000")
        );
    }

    #[test]
    fn prepare_working_paper_db_snapshots_wal_state_for_dry_run() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");

        let conn = Connection::open(&paper_db).unwrap();
        conn.execute_batch(
            r#"
            PRAGMA journal_mode=WAL;
            CREATE TABLE position_state (
                symbol TEXT PRIMARY KEY,
                open_trade_id INTEGER,
                trailing_sl REAL,
                last_funding_time INTEGER,
                adds_count INTEGER,
                tp1_taken INTEGER,
                last_add_time INTEGER,
                entry_adx_threshold REAL,
                updated_at TEXT
            );
            INSERT INTO position_state VALUES ('ETH',1,95.0,1772676500000,0,0,1772676600000,22.0,'2026-03-05T10:08:20+00:00');
            "#,
        )
        .unwrap();
        conn.close().unwrap();

        let working = prepare_working_paper_db(&paper_db, true).unwrap();
        let cloned = Connection::open(working.path()).unwrap();
        let row_count: i64 = cloned
            .query_row("SELECT COUNT(*) FROM position_state", [], |row| row.get(0))
            .unwrap();
        assert_eq!(row_count, 1);
    }
}
