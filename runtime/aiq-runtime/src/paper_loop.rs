use aiq_runtime_core::runtime::{build_bootstrap, RuntimeBootstrap, RuntimeMode};
use anyhow::{Context, Result};
use bt_core::config::{self, StrategyConfig};
use serde::Serialize;
use std::path::Path;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::paper_cycle::{self, PaperCycleInput, PaperCycleReport};

pub struct PaperLoopInput<'a> {
    pub config_path: &'a Path,
    pub live: bool,
    pub profile_override: Option<&'a str>,
    pub paper_db: &'a Path,
    pub candles_db: &'a Path,
    pub explicit_symbols: &'a [String],
    pub symbols_file: Option<&'a Path>,
    pub btc_symbol: &'a str,
    pub lookback_bars: usize,
    pub start_step_close_ts_ms: Option<i64>,
    pub settle_delay_ms: u64,
    pub max_cycles: Option<usize>,
    pub exported_at_ms: Option<i64>,
    pub dry_run: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperLoopStepReport {
    pub status: String,
    pub step_close_ts_ms: i64,
    pub trades_written: usize,
    pub active_symbols: Vec<String>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperLoopReport {
    pub ok: bool,
    pub dry_run: bool,
    pub interval: String,
    pub interval_ms: i64,
    pub settle_delay_ms: u64,
    pub start_step_close_ts_ms: i64,
    pub final_step_close_ts_ms: i64,
    pub requested_cycles: Option<usize>,
    pub cycles_attempted: usize,
    pub cycles_executed: usize,
    pub cycles_duplicate_skipped: usize,
    pub runtime_bootstrap: RuntimeBootstrap,
    pub explicit_symbols: Vec<String>,
    pub steps: Vec<PaperLoopStepReport>,
}

pub fn run_loop(input: PaperLoopInput<'_>) -> Result<PaperLoopReport> {
    if !input.paper_db.exists() {
        anyhow::bail!("paper db not found: {}", input.paper_db.display());
    }
    if !input.candles_db.exists() {
        anyhow::bail!("candles db not found: {}", input.candles_db.display());
    }
    if matches!(input.max_cycles, Some(0)) {
        anyhow::bail!("max_cycles must be at least 1 when provided");
    }

    let mut runtime_bootstrap: Option<RuntimeBootstrap> = None;
    let mut interval: Option<String> = None;
    let mut interval_ms: Option<i64> = None;
    let mut current_step_close_ts_ms = input.start_step_close_ts_ms;
    let explicit_symbols = normalise_symbols(input.explicit_symbols);
    let mut steps = Vec::new();
    let mut cycles_attempted = 0usize;
    let mut cycles_executed = 0usize;
    let mut cycles_duplicate_skipped = 0usize;

    loop {
        if input
            .max_cycles
            .is_some_and(|limit| cycles_attempted >= limit)
        {
            break;
        }

        let base_cfg = load_base_config(input.config_path, input.live)?;
        let reloaded_interval = base_cfg.engine.interval.trim().to_string();
        let reloaded_interval_ms = interval_to_ms(&reloaded_interval);
        if reloaded_interval_ms <= 0 {
            anyhow::bail!(
                "unsupported engine.interval for paper loop: {}",
                reloaded_interval
            );
        }

        match (&interval, interval_ms) {
            (Some(existing), Some(existing_ms)) => {
                if existing != &reloaded_interval || existing_ms != reloaded_interval_ms {
                    anyhow::bail!(
                        "paper loop requires a stable engine.interval; restart after changing {} -> {}",
                        existing,
                        reloaded_interval
                    );
                }
            }
            _ => {
                interval = Some(reloaded_interval.clone());
                interval_ms = Some(reloaded_interval_ms);
            }
        }

        let bootstrap = build_bootstrap(&base_cfg, RuntimeMode::Paper, input.profile_override)
            .map_err(anyhow::Error::msg)?;
        runtime_bootstrap = Some(bootstrap.clone());

        let interval_ms = interval_ms.expect("interval_ms initialised");
        let current_now_ms = now_ms()?;
        let settle_delay_ms = input.settle_delay_ms.min(i64::MAX as u64) as i64;
        if current_step_close_ts_ms.is_none() {
            current_step_close_ts_ms = Some(latest_ready_step_close_ts(
                current_now_ms,
                interval_ms,
                settle_delay_ms,
            )?);
        }
        let step_close_ts_ms = current_step_close_ts_ms.expect("current step initialised");
        let eligible_at_ms = step_close_ts_ms.saturating_add(settle_delay_ms);

        if current_now_ms < eligible_at_ms {
            thread::sleep(Duration::from_millis(
                eligible_at_ms.saturating_sub(current_now_ms) as u64,
            ));
            continue;
        }

        let mut symbols = explicit_symbols.clone();
        if let Some(symbols_file) = input.symbols_file {
            let file_symbols = std::fs::read_to_string(symbols_file)
                .with_context(|| format!("failed to read symbols file {}", symbols_file.display()))?
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>();
            symbols.extend(file_symbols);
        }

        let result = paper_cycle::run_cycle(PaperCycleInput {
            runtime_bootstrap: bootstrap,
            config_path: input.config_path,
            live: input.live,
            paper_db: input.paper_db,
            candles_db: input.candles_db,
            explicit_symbols: &symbols,
            btc_symbol: input.btc_symbol,
            lookback_bars: input.lookback_bars,
            step_close_ts_ms,
            exported_at_ms: input.exported_at_ms,
            dry_run: input.dry_run,
        });

        match result {
            Ok(report) => {
                cycles_attempted += 1;
                cycles_executed += 1;
                steps.push(step_report_from_cycle("executed", &report));
            }
            Err(err) if is_duplicate_step_error(&err) => {
                cycles_attempted += 1;
                cycles_duplicate_skipped += 1;
                steps.push(PaperLoopStepReport {
                    status: "duplicate_skipped".to_string(),
                    step_close_ts_ms,
                    trades_written: 0,
                    active_symbols: Vec::new(),
                    warnings: vec![err.to_string()],
                    errors: Vec::new(),
                });
            }
            Err(err) => return Err(err),
        }

        current_step_close_ts_ms = Some(step_close_ts_ms.saturating_add(interval_ms));
    }

    let runtime_bootstrap =
        runtime_bootstrap.context("paper loop could not build runtime bootstrap")?;
    let interval = interval.context("paper loop could not resolve engine.interval")?;
    let interval_ms = interval_ms.context("paper loop could not resolve interval ms")?;
    let start_step_close_ts_ms = steps
        .first()
        .map(|step| step.step_close_ts_ms)
        .or(current_step_close_ts_ms)
        .context("paper loop did not resolve any step close timestamp")?;
    let final_step_close_ts_ms = steps
        .last()
        .map(|step| step.step_close_ts_ms)
        .unwrap_or(start_step_close_ts_ms);

    Ok(PaperLoopReport {
        ok: true,
        dry_run: input.dry_run,
        interval,
        interval_ms,
        settle_delay_ms: input.settle_delay_ms,
        start_step_close_ts_ms,
        final_step_close_ts_ms,
        requested_cycles: input.max_cycles,
        cycles_attempted,
        cycles_executed,
        cycles_duplicate_skipped,
        runtime_bootstrap,
        explicit_symbols,
        steps,
    })
}

fn load_base_config(path: &Path, live: bool) -> Result<StrategyConfig> {
    config::load_config_checked(
        path.to_str().context("config path must be valid UTF-8")?,
        None,
        live,
    )
    .map_err(anyhow::Error::msg)
}

fn normalise_symbols(raw: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for symbol in raw {
        let symbol = symbol.trim().to_ascii_uppercase();
        if symbol.is_empty() || out.iter().any(|existing| existing == &symbol) {
            continue;
        }
        out.push(symbol);
    }
    out
}

fn interval_to_ms(interval: &str) -> i64 {
    let trimmed = interval.trim();
    if trimmed.is_empty() {
        return 0;
    }
    let (num_part, unit) = trimmed.split_at(trimmed.len().saturating_sub(1));
    let count = num_part.parse::<i64>().unwrap_or(1);
    match unit {
        "m" => count.saturating_mul(60_000),
        "h" => count.saturating_mul(3_600_000),
        "d" => count.saturating_mul(86_400_000),
        _ => 0,
    }
}

fn latest_ready_step_close_ts(now_ms: i64, interval_ms: i64, settle_delay_ms: i64) -> Result<i64> {
    if interval_ms <= 0 {
        anyhow::bail!("interval_ms must be positive");
    }
    let effective_now_ms = now_ms.saturating_sub(settle_delay_ms.max(0));
    Ok((effective_now_ms / interval_ms) * interval_ms)
}

fn is_duplicate_step_error(err: &anyhow::Error) -> bool {
    err.to_string().contains("paper cycle step already applied")
}

fn now_ms() -> Result<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before UNIX_EPOCH")?;
    Ok(duration.as_millis().min(i64::MAX as u128) as i64)
}

fn step_report_from_cycle(status: &str, report: &PaperCycleReport) -> PaperLoopStepReport {
    let mut warnings = report.warnings.clone();
    if !report.ok {
        warnings.push("paper cycle reported non-empty diagnostics".to_string());
    }
    PaperLoopStepReport {
        status: status.to_string(),
        step_close_ts_ms: report.step_close_ts_ms,
        trades_written: report.trades_written,
        active_symbols: report.active_symbols.clone(),
        warnings,
        errors: report.errors.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use tempfile::tempdir;

    const FIXED_STEP_CLOSE_TS_MS: i64 = 1_772_676_000_000;
    const FIXED_EXPORTED_AT_MS: i64 = 1_772_676_900_000;

    fn seed_paper_db(path: &Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                action TEXT,
                type TEXT,
                price REAL,
                size REAL,
                notional REAL,
                reason TEXT,
                confidence TEXT,
                entry_atr REAL,
                leverage REAL,
                margin_used REAL,
                balance REAL,
                pnl REAL,
                fee_usd REAL,
                fee_rate REAL,
                meta_json TEXT,
                run_fingerprint TEXT
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
                close_ts_ms INTEGER,
                side TEXT,
                reason TEXT,
                updated_at TEXT
            );
            "#,
        )
        .unwrap();
        conn.execute(
            "INSERT INTO trades (timestamp,symbol,action,type,price,size,notional,reason,confidence,entry_atr,leverage,margin_used,balance,pnl,fee_usd,fee_rate,meta_json,run_fingerprint)
             VALUES ('2026-03-05T10:00:00+00:00','BTC','OPEN','LONG',100.0,2.0,200.0,'Signal Trigger','high',5.0,4.0,50.0,1000.0,0.0,0.0,0.0,'{}','seed')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO position_state VALUES ('BTC',1,95.0,1772676500000,0,0,1772676600000,22.0,'2026-03-05T10:08:20+00:00')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO runtime_cooldowns VALUES ('BTC',1772676500.0,1772676550.0,'2026-03-05T10:15:00+00:00')",
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

    fn config_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("config/strategy_overrides.yaml.example")
    }

    #[test]
    fn latest_ready_step_respects_settle_delay() {
        let step = latest_ready_step_close_ts(1_800_000, 1_800_000, 5_000).unwrap();
        assert_eq!(step, 0);
        let step = latest_ready_step_close_ts(1_805_000, 1_800_000, 5_000).unwrap();
        assert_eq!(step, 1_800_000);
    }

    #[test]
    fn paper_loop_runs_one_bounded_cycle() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_paper_db(&paper_db);
        seed_candles_db(&candles_db);

        let report = run_loop(PaperLoopInput {
            config_path: &config_path(),
            live: false,
            profile_override: None,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["ETH".to_string()],
            symbols_file: None,
            btc_symbol: "BTC",
            lookback_bars: 400,
            start_step_close_ts_ms: Some(FIXED_STEP_CLOSE_TS_MS),
            settle_delay_ms: 0,
            max_cycles: Some(1),
            exported_at_ms: Some(FIXED_EXPORTED_AT_MS),
            dry_run: false,
        })
        .unwrap();

        assert_eq!(report.cycles_attempted, 1);
        assert_eq!(report.cycles_executed, 1);
        assert_eq!(report.cycles_duplicate_skipped, 0);
        assert_eq!(report.steps[0].status, "executed");
        assert_eq!(report.steps[0].step_close_ts_ms, FIXED_STEP_CLOSE_TS_MS);
    }

    #[test]
    fn paper_loop_treats_preexisting_step_as_duplicate_skip() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_paper_db(&paper_db);
        seed_candles_db(&candles_db);

        let first = run_loop(PaperLoopInput {
            config_path: &config_path(),
            live: false,
            profile_override: None,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["ETH".to_string()],
            symbols_file: None,
            btc_symbol: "BTC",
            lookback_bars: 400,
            start_step_close_ts_ms: Some(FIXED_STEP_CLOSE_TS_MS),
            settle_delay_ms: 0,
            max_cycles: Some(1),
            exported_at_ms: Some(FIXED_EXPORTED_AT_MS),
            dry_run: false,
        })
        .unwrap();
        assert_eq!(first.cycles_executed, 1);

        let second = run_loop(PaperLoopInput {
            config_path: &config_path(),
            live: false,
            profile_override: None,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["ETH".to_string()],
            symbols_file: None,
            btc_symbol: "BTC",
            lookback_bars: 400,
            start_step_close_ts_ms: Some(FIXED_STEP_CLOSE_TS_MS),
            settle_delay_ms: 0,
            max_cycles: Some(1),
            exported_at_ms: Some(FIXED_EXPORTED_AT_MS),
            dry_run: false,
        })
        .unwrap();

        assert_eq!(second.cycles_attempted, 1);
        assert_eq!(second.cycles_executed, 0);
        assert_eq!(second.cycles_duplicate_skipped, 1);
        assert_eq!(second.steps[0].status, "duplicate_skipped");
        assert!(
            second.steps[0].warnings[0].contains("already applied"),
            "expected duplicate-step warning"
        );
    }
}
