use anyhow::Result;
use chrono::Utc;
use serde::Serialize;
use std::env;
use std::path::Path;

use crate::paper_daemon::{self, PaperDaemonStatusSnapshot};
use crate::paper_lane::PaperLane;
use crate::paper_manifest::{
    self, PaperManifestInput, PaperManifestLaunchState, PaperManifestReport,
};

pub struct PaperStatusInput<'a> {
    pub config: Option<&'a Path>,
    pub lane: Option<PaperLane>,
    pub project_dir: Option<&'a Path>,
    pub live: bool,
    pub profile: Option<&'a str>,
    pub db: Option<&'a Path>,
    pub candles_db: Option<&'a Path>,
    pub symbols: &'a [String],
    pub symbols_file: Option<&'a Path>,
    pub watch_symbols_file: bool,
    pub btc_symbol: &'a str,
    pub lookback_bars: Option<usize>,
    pub start_step_close_ts_ms: Option<i64>,
    pub lock_path: Option<&'a Path>,
    pub status_path: Option<&'a Path>,
    pub stale_after_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PaperServiceState {
    Blocked,
    IdleNoSymbols,
    BootstrapRequired,
    BootstrapReady,
    ResumeReady,
    CaughtUpIdle,
    Running,
    RestartRequired,
    StatusStale,
    Stopped,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperStatusReport {
    pub ok: bool,
    pub service_state: PaperServiceState,
    pub status_file_present: bool,
    pub status_age_ms: Option<i64>,
    pub stale_after_ms: Option<i64>,
    pub contract_matches_status: bool,
    pub mismatch_reasons: Vec<String>,
    pub manifest: PaperManifestReport,
    pub daemon_status: Option<PaperDaemonStatusSnapshot>,
    pub warnings: Vec<String>,
}

pub fn build_status(input: PaperStatusInput<'_>) -> Result<PaperStatusReport> {
    let manifest = paper_manifest::build_manifest(PaperManifestInput {
        config: input.config,
        lane: input.lane,
        project_dir: input.project_dir,
        live: input.live,
        profile: input.profile,
        db: input.db,
        candles_db: input.candles_db,
        symbols: input.symbols,
        symbols_file: input.symbols_file,
        watch_symbols_file: input.watch_symbols_file,
        btc_symbol: input.btc_symbol,
        lookback_bars: input.lookback_bars,
        start_step_close_ts_ms: input.start_step_close_ts_ms,
        lock_path: input.lock_path,
        status_path: input.status_path,
    })?;
    let stale_after_ms = input
        .stale_after_ms
        .or_else(|| env_i64("AI_QUANT_STATUS_STALE_AFTER_MS"));
    let mut warnings = manifest.warnings.clone();
    let daemon_status = paper_daemon::load_status_file(Path::new(&manifest.status_path))?;
    let status_file_present = daemon_status.is_some();
    let mut status_age_ms = None;
    let mut mismatch_reasons = Vec::new();
    let mut contract_matches_status = true;
    let mut service_state = launch_state_to_service_state(&manifest.resume.launch_state);
    let mut daemon_health_ok = true;

    if let Some(status) = daemon_status.as_ref() {
        status_age_ms = Some((Utc::now().timestamp_millis() - status.updated_at_ms).max(0));
        mismatch_reasons = compare_manifest_and_status(&manifest, status);
        daemon_health_ok = status.ok && status.errors.is_empty();
        if !daemon_health_ok {
            mismatch_reasons.push(match status.errors.as_slice() {
                [] => {
                    "daemon reported an unhealthy status without explicit error entries".to_string()
                }
                [single] => format!("daemon reported an unhealthy status: {single}"),
                many => format!(
                    "daemon reported unhealthy status errors: {}",
                    many.join("; ")
                ),
            });
        }
        contract_matches_status = mismatch_reasons.is_empty();

        if status.running {
            if stale_after_ms
                .zip(status_age_ms)
                .is_some_and(|(stale_after_ms, status_age_ms)| status_age_ms > stale_after_ms)
            {
                service_state = PaperServiceState::StatusStale;
                warnings.push(format!(
                    "paper status marked the daemon lane stale because the status age {}ms exceeds {}ms",
                    status_age_ms.unwrap_or_default(),
                    stale_after_ms.unwrap_or_default()
                ));
            } else if !contract_matches_status {
                service_state = PaperServiceState::RestartRequired;
                warnings.push(format!(
                    "paper status detected a running daemon that no longer matches the current launch contract or health gate: {}",
                    mismatch_reasons.join("; ")
                ));
            } else {
                service_state = PaperServiceState::Running;
            }
        } else {
            service_state = PaperServiceState::Stopped;
        }
    } else if manifest.resume.launch_ready {
        warnings.push(format!(
            "paper status found no daemon status file at {}; the lane is not currently supervised by a Rust status surface",
            manifest.status_path
        ));
    }

    Ok(PaperStatusReport {
        ok: manifest.ok && daemon_health_ok,
        service_state,
        status_file_present,
        status_age_ms,
        stale_after_ms,
        contract_matches_status,
        mismatch_reasons,
        manifest,
        daemon_status,
        warnings,
    })
}

fn launch_state_to_service_state(value: &PaperManifestLaunchState) -> PaperServiceState {
    match value {
        PaperManifestLaunchState::Blocked => PaperServiceState::Blocked,
        PaperManifestLaunchState::IdleNoSymbols => PaperServiceState::IdleNoSymbols,
        PaperManifestLaunchState::BootstrapRequired => PaperServiceState::BootstrapRequired,
        PaperManifestLaunchState::BootstrapReady => PaperServiceState::BootstrapReady,
        PaperManifestLaunchState::ResumeReady => PaperServiceState::ResumeReady,
        PaperManifestLaunchState::CaughtUpIdle => PaperServiceState::CaughtUpIdle,
    }
}

fn compare_manifest_and_status(
    manifest: &PaperManifestReport,
    status: &PaperDaemonStatusSnapshot,
) -> Vec<String> {
    let mut mismatches = Vec::new();

    if status.config_path != manifest.config_path {
        mismatches.push(format!(
            "config path mismatch (status={} current={})",
            status.config_path, manifest.config_path
        ));
    }
    if status.runtime_bootstrap.config_fingerprint != manifest.runtime_bootstrap.config_fingerprint
    {
        mismatches.push(format!(
            "config fingerprint mismatch (status={} current={})",
            status.runtime_bootstrap.config_fingerprint,
            manifest.runtime_bootstrap.config_fingerprint
        ));
    }
    if status.runtime_bootstrap.pipeline.profile != manifest.runtime_bootstrap.pipeline.profile {
        mismatches.push(format!(
            "runtime profile mismatch (status={} current={})",
            status.runtime_bootstrap.pipeline.profile, manifest.runtime_bootstrap.pipeline.profile
        ));
    }
    if status.paper_db != manifest.paper_db {
        mismatches.push(format!(
            "paper db mismatch (status={} current={})",
            status.paper_db, manifest.paper_db
        ));
    }
    if status.candles_db != manifest.candles_db {
        mismatches.push(format!(
            "candles db mismatch (status={} current={})",
            status.candles_db, manifest.candles_db
        ));
    }
    if canonical_btc_symbol(&status.btc_symbol) != canonical_btc_symbol(&manifest.btc_symbol) {
        mismatches.push(format!(
            "btc symbol mismatch (status={} current={})",
            status.btc_symbol, manifest.btc_symbol
        ));
    }
    if status.lookback_bars != manifest.lookback_bars {
        mismatches.push(format!(
            "lookback bars mismatch (status={} current={})",
            status.lookback_bars, manifest.lookback_bars
        ));
    }
    if status.explicit_symbols != manifest.symbols {
        mismatches.push(format!(
            "explicit symbols mismatch (status={} current={})",
            status.explicit_symbols.join(","),
            manifest.symbols.join(",")
        ));
    }
    if status.lock_path != manifest.lock_path {
        mismatches.push(format!(
            "lock path mismatch (status={} current={})",
            status.lock_path, manifest.lock_path
        ));
    }
    if status.status_path != manifest.status_path {
        mismatches.push(format!(
            "status path mismatch (status={} current={})",
            status.status_path, manifest.status_path
        ));
    }
    if status.watch_symbols_file != manifest.watch_symbols_file {
        mismatches.push(format!(
            "watch-symbols-file mismatch (status={} current={})",
            status.watch_symbols_file, manifest.watch_symbols_file
        ));
    }
    if status.symbols_file != manifest.symbols_file {
        mismatches.push(format!(
            "symbols-file mismatch (status={:?} current={:?})",
            status.symbols_file, manifest.symbols_file
        ));
    }
    if bootstrap_step_is_contract(manifest, status)
        && status.start_step_close_ts_ms != manifest.start_step_close_ts_ms
    {
        mismatches.push(format!(
            "start-step-close-ts-ms mismatch (status={:?} current={:?})",
            status.start_step_close_ts_ms, manifest.start_step_close_ts_ms
        ));
    }

    mismatches
}

fn env_i64(name: &str) -> Option<i64> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .and_then(|value| value.parse::<i64>().ok())
}

fn canonical_btc_symbol(value: &str) -> String {
    value.trim().to_ascii_uppercase()
}

fn bootstrap_step_is_contract(
    manifest: &PaperManifestReport,
    status: &PaperDaemonStatusSnapshot,
) -> bool {
    manifest.start_step_close_ts_ms.is_some()
        || status.start_step_close_ts_ms.is_some()
            && status.initial_last_applied_step_close_ts_ms.is_none()
            && status.executed_steps == 0
            && status
                .next_due_step_close_ts_ms
                .map(|next_due_step_close_ts_ms| {
                    Some(next_due_step_close_ts_ms) == status.start_step_close_ts_ms
                })
                .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiq_runtime_core::runtime::{build_bootstrap, RuntimeMode};
    use chrono::Utc;
    use rusqlite::{params, Connection};
    use std::fs;
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;

    use crate::paper_daemon::{PaperDaemonStatus, PaperDaemonStatusSnapshot};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn env_lock() -> &'static Mutex<()> {
        ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    fn write_config(path: &Path, interval: &str) {
        fs::write(
            path,
            format!("global:\n  engine:\n    interval: {}\n", interval),
        )
        .unwrap();
    }

    fn init_paper_db(path: &Path) {
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
            CREATE TABLE runtime_cycle_steps (
                step_id TEXT PRIMARY KEY,
                step_close_ts_ms INTEGER NOT NULL,
                interval TEXT NOT NULL,
                symbols_json TEXT NOT NULL,
                snapshot_exported_at_ms INTEGER NOT NULL,
                execution_count INTEGER NOT NULL,
                trades_written INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            "#,
        )
        .unwrap();
    }

    fn init_candles_db(path: &Path, symbols: &[&str], interval: &str, closes: &[i64]) {
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
        for symbol in symbols {
            let mut price = 100.0;
            for close in closes {
                conn.execute(
                    "INSERT INTO candles VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1)",
                    params![
                        symbol,
                        interval,
                        close - 1_800_000_i64,
                        close,
                        price,
                        price + 1.0,
                        price - 1.0,
                        price + 0.5,
                        1_000.0,
                    ],
                )
                .unwrap();
                price += 1.0;
            }
        }
    }

    fn status_fixture(
        interval: &str,
        with_runtime_step: bool,
    ) -> (
        tempfile::TempDir,
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
        PaperManifestReport,
    ) {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        let lock_path = dir.path().join("paper.lock");
        let status_path = dir.path().join("paper.status.json");
        write_config(&config_path, interval);
        init_paper_db(&paper_db);
        init_candles_db(
            &candles_db,
            &["BTC", "ETH"],
            interval,
            &[1_773_422_400_000, 1_773_424_200_000, 1_773_426_000_000],
        );

        if with_runtime_step {
            let cfg =
                bt_core::config::load_config_checked(config_path.to_str().unwrap(), None, false)
                    .unwrap();
            let bootstrap = build_bootstrap(&cfg, RuntimeMode::Paper, Some("production")).unwrap();
            let conn = Connection::open(&paper_db).unwrap();
            conn.execute(
                "INSERT INTO runtime_cycle_steps (step_id, step_close_ts_ms, interval, symbols_json, snapshot_exported_at_ms, execution_count, trades_written, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    format!("{}:{}", bootstrap.config_fingerprint, 1_773_424_200_000_i64),
                    1_773_424_200_000_i64,
                    interval,
                    r#"[\"ETH\"]"#,
                    1_773_424_200_000_i64,
                    1_i64,
                    0_i64,
                    "2026-03-06T00:00:00+00:00",
                ],
            )
            .unwrap();
        }

        let manifest = paper_manifest::build_manifest(PaperManifestInput {
            config: Some(&config_path),
            lane: None,
            project_dir: None,
            live: false,
            profile: Some("production"),
            db: Some(&paper_db),
            candles_db: Some(&candles_db),
            symbols: &["ETH".to_string()],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: Some(200),
            start_step_close_ts_ms: None,
            lock_path: Some(&lock_path),
            status_path: Some(&status_path),
        })
        .unwrap();

        (
            dir,
            config_path,
            paper_db,
            candles_db,
            lock_path,
            status_path,
            manifest,
        )
    }

    fn write_status(
        path: &Path,
        manifest: &PaperManifestReport,
        running: bool,
        updated_at_ms: i64,
    ) {
        let status = PaperDaemonStatus {
            ok: true,
            running,
            pid: 1234,
            config_path: manifest.config_path.clone(),
            paper_db: manifest.paper_db.clone(),
            candles_db: manifest.candles_db.clone(),
            lock_path: manifest.lock_path.clone(),
            status_path: manifest.status_path.clone(),
            started_at_ms: updated_at_ms - 500,
            updated_at_ms,
            stopped_at_ms: (!running).then_some(updated_at_ms),
            stop_requested: !running,
            dry_run: false,
            runtime_bootstrap: manifest.runtime_bootstrap.clone(),
            btc_symbol: manifest.btc_symbol.clone(),
            lookback_bars: manifest.lookback_bars,
            explicit_symbols: manifest.symbols.clone(),
            watch_symbols_file: manifest.watch_symbols_file,
            symbols_file: manifest.symbols_file.clone(),
            start_step_close_ts_ms: manifest.start_step_close_ts_ms,
            manifest_symbols: manifest.resume.active_symbols.clone(),
            last_active_symbols: manifest.resume.active_symbols.clone(),
            manifest_reload_count: 0,
            manifest_reload_failure_count: 0,
            initial_last_applied_step_close_ts_ms: manifest.resume.last_applied_step_close_ts_ms,
            latest_common_close_ts_ms: manifest.resume.latest_common_close_ts_ms,
            next_due_step_close_ts_ms: manifest.resume.next_due_step_close_ts_ms,
            executed_steps: 0,
            idle_polls: 0,
            warnings: Vec::new(),
            errors: Vec::new(),
        };
        fs::write(path, serde_json::to_vec_pretty(&status).unwrap()).unwrap();
    }

    #[test]
    fn status_reports_bootstrap_required_without_status_file() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let (_dir, config_path, paper_db, candles_db, lock_path, status_path, manifest) =
            status_fixture("30m", false);

        let report = build_status(PaperStatusInput {
            config: Some(&config_path),
            lane: None,
            project_dir: None,
            live: false,
            profile: Some("production"),
            db: Some(&paper_db),
            candles_db: Some(&candles_db),
            symbols: &["ETH".to_string()],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: Some(200),
            start_step_close_ts_ms: None,
            lock_path: Some(&lock_path),
            status_path: Some(&status_path),
            stale_after_ms: None,
        })
        .unwrap();

        assert_eq!(
            manifest.resume.launch_state,
            PaperManifestLaunchState::BootstrapRequired
        );
        assert_eq!(report.service_state, PaperServiceState::BootstrapRequired);
        assert!(!report.status_file_present);
        assert!(report.daemon_status.is_none());
    }

    #[test]
    fn status_reports_running_when_status_matches_manifest() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let (_dir, config_path, paper_db, candles_db, lock_path, status_path, manifest) =
            status_fixture("30m", true);
        write_status(
            Path::new(&manifest.status_path),
            &manifest,
            true,
            Utc::now().timestamp_millis(),
        );

        let report = build_status(PaperStatusInput {
            config: Some(&config_path),
            lane: None,
            project_dir: None,
            live: false,
            profile: Some("production"),
            db: Some(&paper_db),
            candles_db: Some(&candles_db),
            symbols: &["ETH".to_string()],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: Some(200),
            start_step_close_ts_ms: None,
            lock_path: Some(&lock_path),
            status_path: Some(&status_path),
            stale_after_ms: Some(60_000),
        })
        .unwrap();

        assert_eq!(report.service_state, PaperServiceState::Running);
        assert!(report.status_file_present);
        assert!(report.contract_matches_status);
        assert!(report.mismatch_reasons.is_empty());
    }

    #[test]
    fn status_reports_restart_required_for_contract_mismatch() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let (_dir, config_path, paper_db, candles_db, lock_path, status_path, manifest) =
            status_fixture("30m", true);
        let mut mismatched = manifest.clone();
        mismatched.runtime_bootstrap.config_fingerprint = "stale-fingerprint".to_string();
        write_status(
            Path::new(&manifest.status_path),
            &mismatched,
            true,
            Utc::now().timestamp_millis(),
        );

        let report = build_status(PaperStatusInput {
            config: Some(&config_path),
            lane: None,
            project_dir: None,
            live: false,
            profile: Some("production"),
            db: Some(&paper_db),
            candles_db: Some(&candles_db),
            symbols: &["ETH".to_string()],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: Some(200),
            start_step_close_ts_ms: None,
            lock_path: Some(&lock_path),
            status_path: Some(&status_path),
            stale_after_ms: Some(60_000),
        })
        .unwrap();

        assert_eq!(report.service_state, PaperServiceState::RestartRequired);
        assert!(!report.contract_matches_status);
        assert!(report
            .mismatch_reasons
            .iter()
            .any(|reason| reason.contains("config fingerprint mismatch")));
    }

    #[test]
    fn status_reports_restart_required_for_profile_mismatch() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let (_dir, config_path, paper_db, candles_db, lock_path, status_path, manifest) =
            status_fixture("30m", true);
        let mut mismatched = manifest.clone();
        mismatched.runtime_bootstrap.pipeline.profile = "parity_baseline".to_string();
        write_status(
            Path::new(&manifest.status_path),
            &mismatched,
            true,
            Utc::now().timestamp_millis(),
        );

        let report = build_status(PaperStatusInput {
            config: Some(&config_path),
            lane: None,
            project_dir: None,
            live: false,
            profile: Some("production"),
            db: Some(&paper_db),
            candles_db: Some(&candles_db),
            symbols: &["ETH".to_string()],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: Some(200),
            start_step_close_ts_ms: None,
            lock_path: Some(&lock_path),
            status_path: Some(&status_path),
            stale_after_ms: Some(60_000),
        })
        .unwrap();

        assert_eq!(report.service_state, PaperServiceState::RestartRequired);
        assert!(!report.contract_matches_status);
        assert!(report
            .mismatch_reasons
            .iter()
            .any(|reason| reason.contains("runtime profile mismatch")));
    }

    #[test]
    fn status_reports_restart_required_for_launch_input_mismatch() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let (_dir, config_path, paper_db, candles_db, lock_path, status_path, manifest) =
            status_fixture("30m", true);
        let mut mismatched = manifest.clone();
        mismatched.paper_db = "/tmp/other-paper.db".to_string();
        mismatched.candles_db = "/tmp/other-candles.db".to_string();
        mismatched.btc_symbol = "ETH".to_string();
        mismatched.lookback_bars = 123;
        mismatched.symbols = vec!["BTC".to_string()];
        mismatched.start_step_close_ts_ms = Some(1_773_426_000_000);
        write_status(
            Path::new(&manifest.status_path),
            &mismatched,
            true,
            Utc::now().timestamp_millis(),
        );

        let report = build_status(PaperStatusInput {
            config: Some(&config_path),
            lane: None,
            project_dir: None,
            live: false,
            profile: Some("production"),
            db: Some(&paper_db),
            candles_db: Some(&candles_db),
            symbols: &["ETH".to_string()],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: Some(200),
            start_step_close_ts_ms: Some(1_773_422_400_000),
            lock_path: Some(&lock_path),
            status_path: Some(&status_path),
            stale_after_ms: Some(60_000),
        })
        .unwrap();

        assert_eq!(report.service_state, PaperServiceState::RestartRequired);
        assert!(report
            .mismatch_reasons
            .iter()
            .any(|reason| reason.contains("paper db mismatch")));
        assert!(report
            .mismatch_reasons
            .iter()
            .any(|reason| reason.contains("candles db mismatch")));
        assert!(report
            .mismatch_reasons
            .iter()
            .any(|reason| reason.contains("btc symbol mismatch")));
        assert!(report
            .mismatch_reasons
            .iter()
            .any(|reason| reason.contains("lookback bars mismatch")));
        assert!(report
            .mismatch_reasons
            .iter()
            .any(|reason| reason.contains("explicit symbols mismatch")));
        assert!(report
            .mismatch_reasons
            .iter()
            .any(|reason| reason.contains("start-step-close-ts-ms mismatch")));
    }

    #[test]
    fn status_reports_restart_required_for_unhealthy_running_status() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let (_dir, config_path, paper_db, candles_db, lock_path, status_path, manifest) =
            status_fixture("30m", true);
        write_status(
            Path::new(&manifest.status_path),
            &manifest,
            true,
            Utc::now().timestamp_millis(),
        );

        let mut status: PaperDaemonStatusSnapshot =
            serde_json::from_slice(&fs::read(&status_path).unwrap()).unwrap();
        status.ok = false;
        status.errors = vec!["daemon step failed".to_string()];
        fs::write(&status_path, serde_json::to_vec_pretty(&status).unwrap()).unwrap();

        let report = build_status(PaperStatusInput {
            config: Some(&config_path),
            lane: None,
            project_dir: None,
            live: false,
            profile: Some("production"),
            db: Some(&paper_db),
            candles_db: Some(&candles_db),
            symbols: &["ETH".to_string()],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: Some(200),
            start_step_close_ts_ms: None,
            lock_path: Some(&lock_path),
            status_path: Some(&status_path),
            stale_after_ms: Some(60_000),
        })
        .unwrap();

        assert_eq!(report.service_state, PaperServiceState::RestartRequired);
        assert!(!report.ok);
        assert!(!report.contract_matches_status);
        assert!(!report.mismatch_reasons.is_empty());
    }

    #[test]
    fn status_ignores_bootstrap_step_after_resume_handoff() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let (_dir, config_path, paper_db, candles_db, lock_path, status_path, manifest) =
            status_fixture("30m", true);
        write_status(
            Path::new(&manifest.status_path),
            &manifest,
            true,
            Utc::now().timestamp_millis(),
        );

        let mut status: PaperDaemonStatusSnapshot =
            serde_json::from_slice(&fs::read(&status_path).unwrap()).unwrap();
        status.start_step_close_ts_ms = Some(1_773_426_000_000);
        status.next_due_step_close_ts_ms = Some(1_773_426_000_000);
        status.initial_last_applied_step_close_ts_ms = Some(1_773_424_200_000);
        status.executed_steps = 0;
        fs::write(&status_path, serde_json::to_vec_pretty(&status).unwrap()).unwrap();

        let report = build_status(PaperStatusInput {
            config: Some(&config_path),
            lane: None,
            project_dir: None,
            live: false,
            profile: Some("production"),
            db: Some(&paper_db),
            candles_db: Some(&candles_db),
            symbols: &["ETH".to_string()],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: Some(200),
            start_step_close_ts_ms: None,
            lock_path: Some(&lock_path),
            status_path: Some(&status_path),
            stale_after_ms: Some(60_000),
        })
        .unwrap();

        assert_eq!(report.service_state, PaperServiceState::Running);
        assert!(report.contract_matches_status);
        assert!(report
            .mismatch_reasons
            .iter()
            .all(|reason| !reason.contains("start-step-close-ts-ms mismatch")));
    }

    #[test]
    fn status_requires_bootstrap_step_during_fresh_lane_handoff() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let (_dir, config_path, paper_db, candles_db, lock_path, status_path, manifest) =
            status_fixture("30m", false);
        write_status(
            Path::new(&manifest.status_path),
            &manifest,
            true,
            Utc::now().timestamp_millis(),
        );

        let mut status: PaperDaemonStatusSnapshot =
            serde_json::from_slice(&fs::read(&status_path).unwrap()).unwrap();
        status.start_step_close_ts_ms = Some(1_773_422_400_000);
        status.next_due_step_close_ts_ms = Some(1_773_422_400_000);
        status.executed_steps = 0;
        fs::write(&status_path, serde_json::to_vec_pretty(&status).unwrap()).unwrap();

        let report = build_status(PaperStatusInput {
            config: Some(&config_path),
            lane: None,
            project_dir: None,
            live: false,
            profile: Some("production"),
            db: Some(&paper_db),
            candles_db: Some(&candles_db),
            symbols: &["ETH".to_string()],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: Some(200),
            start_step_close_ts_ms: None,
            lock_path: Some(&lock_path),
            status_path: Some(&status_path),
            stale_after_ms: Some(60_000),
        })
        .unwrap();

        assert_eq!(report.service_state, PaperServiceState::RestartRequired);
        assert!(!report.contract_matches_status);
        assert!(report
            .mismatch_reasons
            .iter()
            .any(|reason| reason.contains("start-step-close-ts-ms mismatch")));
    }

    #[test]
    fn status_treats_btc_symbol_case_insensitively() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let (_dir, config_path, paper_db, candles_db, lock_path, status_path, manifest) =
            status_fixture("30m", true);
        write_status(
            Path::new(&manifest.status_path),
            &manifest,
            true,
            Utc::now().timestamp_millis(),
        );

        let mut status: PaperDaemonStatusSnapshot =
            serde_json::from_slice(&fs::read(&status_path).unwrap()).unwrap();
        status.btc_symbol = "btc".to_string();
        fs::write(&status_path, serde_json::to_vec_pretty(&status).unwrap()).unwrap();

        let report = build_status(PaperStatusInput {
            config: Some(&config_path),
            lane: None,
            project_dir: None,
            live: false,
            profile: Some("production"),
            db: Some(&paper_db),
            candles_db: Some(&candles_db),
            symbols: &["ETH".to_string()],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: Some(200),
            start_step_close_ts_ms: None,
            lock_path: Some(&lock_path),
            status_path: Some(&status_path),
            stale_after_ms: Some(60_000),
        })
        .unwrap();

        assert_eq!(report.service_state, PaperServiceState::Running);
        assert!(report.contract_matches_status);
        assert!(report
            .mismatch_reasons
            .iter()
            .all(|reason| !reason.contains("btc symbol mismatch")));
    }

    #[test]
    fn status_reports_stale_for_old_running_status() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let (_dir, config_path, paper_db, candles_db, lock_path, status_path, manifest) =
            status_fixture("30m", true);
        write_status(Path::new(&manifest.status_path), &manifest, true, 1_000);

        let report = build_status(PaperStatusInput {
            config: Some(&config_path),
            lane: None,
            project_dir: None,
            live: false,
            profile: Some("production"),
            db: Some(&paper_db),
            candles_db: Some(&candles_db),
            symbols: &["ETH".to_string()],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: Some(200),
            start_step_close_ts_ms: None,
            lock_path: Some(&lock_path),
            status_path: Some(&status_path),
            stale_after_ms: Some(10),
        })
        .unwrap();

        assert_eq!(report.service_state, PaperServiceState::StatusStale);
        assert!(report.status_age_ms.is_some_and(|age| age > 10));
    }
}
