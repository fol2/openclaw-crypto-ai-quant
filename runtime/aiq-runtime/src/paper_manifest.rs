use aiq_runtime_core::runtime::RuntimeBootstrap;
use anyhow::{Context, Result};
use serde::Serialize;
use std::env;
use std::path::{Path, PathBuf};

use crate::paper_config::PaperEffectiveConfig;
use crate::paper_daemon;
use crate::paper_lane::PaperLane;
use crate::paper_loop;

const DEFAULT_PAPER_DB_PATH: &str = "trading_engine.db";
const DEFAULT_LOOKBACK_BARS: usize = 400;

pub struct PaperManifestInput<'a> {
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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PaperManifestLaunchState {
    Blocked,
    IdleNoSymbols,
    BootstrapRequired,
    BootstrapReady,
    ResumeReady,
    CaughtUpIdle,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PaperManifestResumeState {
    pub launch_ready: bool,
    pub launch_state: PaperManifestLaunchState,
    pub active_symbols: Vec<String>,
    pub last_applied_step_close_ts_ms: Option<i64>,
    pub latest_common_close_ts_ms: Option<i64>,
    pub next_due_step_close_ts_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperManifestReport {
    pub ok: bool,
    pub runtime_bootstrap: RuntimeBootstrap,
    pub base_config_path: String,
    pub config_path: String,
    pub active_yaml_path: String,
    pub effective_yaml_path: String,
    pub paper_db: String,
    pub paper_db_exists: bool,
    pub candles_db: String,
    pub candles_db_exists: bool,
    pub interval: String,
    pub lookback_bars: usize,
    pub symbols: Vec<String>,
    pub symbols_file: Option<String>,
    pub watch_symbols_file: bool,
    pub btc_symbol: String,
    pub start_step_close_ts_ms: Option<i64>,
    pub lock_path: String,
    pub status_path: String,
    pub instance_tag: Option<String>,
    pub lane: Option<String>,
    pub service_name: Option<String>,
    pub promoted_role: Option<String>,
    pub promoted_config_path: Option<String>,
    pub strategy_mode: Option<String>,
    pub strategy_mode_source: Option<String>,
    pub strategy_overrides_sha1: String,
    pub config_id: String,
    pub resume: PaperManifestResumeState,
    pub warnings: Vec<String>,
    pub daemon_command: Vec<String>,
}

pub fn build_manifest(input: PaperManifestInput<'_>) -> Result<PaperManifestReport> {
    let lane_defaults = match input.lane {
        Some(lane) => Some(lane.defaults(input.project_dir)?),
        None => None,
    };
    let effective_config =
        PaperEffectiveConfig::resolve(input.config, input.lane, input.project_dir)?;
    let mut warnings = effective_config.warnings().to_vec();
    let symbols = resolve_symbols(input.symbols);
    let symbols_file = input
        .symbols_file
        .map(Path::to_path_buf)
        .or_else(|| {
            if symbols.is_empty() {
                lane_defaults
                    .as_ref()
                    .map(|defaults| defaults.symbols_file.clone())
            } else {
                None
            }
        })
        .or_else(|| env_path("AI_QUANT_SYMBOLS_FILE"));
    let watch_symbols_file = input.watch_symbols_file
        || (input.symbols.is_empty() && input.symbols_file.is_none() && lane_defaults.is_some())
        || env_bool("AI_QUANT_WATCH_SYMBOLS_FILE");
    let interval_symbols = resolve_interval_symbols(&symbols, symbols_file.as_deref())?;
    let runtime_bootstrap = effective_config.build_runtime_bootstrap(
        bootstrap_symbol_hint(&interval_symbols),
        input.live,
        input.profile,
    )?;
    let config =
        effective_config.load_config(bootstrap_symbol_hint(&interval_symbols), input.live)?;
    let interval = if interval_symbols.is_empty() {
        config.engine.interval.trim().to_string()
    } else {
        effective_config.resolve_shared_interval(&interval_symbols, input.live)?
    };
    if let Some(env_interval) = env_string("AI_QUANT_INTERVAL") {
        let env_interval = env_interval.trim().to_string();
        if !env_interval.is_empty() && env_interval != interval {
            warnings.push(format!(
                "AI_QUANT_INTERVAL={} differs from the resolved config interval={}",
                env_interval, interval
            ));
        }
    }

    let paper_db = input
        .db
        .map(Path::to_path_buf)
        .or_else(|| {
            lane_defaults
                .as_ref()
                .map(|defaults| defaults.paper_db.clone())
        })
        .or_else(|| env_path("AI_QUANT_DB_PATH"))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PAPER_DB_PATH));

    let candles_db = resolve_candles_db(input.candles_db, lane_defaults.as_ref(), &interval)?;
    let lookback_bars = input
        .lookback_bars
        .or_else(|| {
            lane_defaults
                .as_ref()
                .map(|defaults| defaults.lookback_bars)
        })
        .or_else(|| env_usize("AI_QUANT_LOOKBACK_BARS"))
        .unwrap_or(DEFAULT_LOOKBACK_BARS);
    let btc_symbol = normalise_symbol(input.btc_symbol).unwrap_or_else(|| "BTC".to_string());
    let start_step_close_ts_ms = input
        .start_step_close_ts_ms
        .or_else(|| env_i64("AI_QUANT_PAPER_START_STEP_CLOSE_TS_MS"));
    let lock_path = input
        .lock_path
        .map(Path::to_path_buf)
        .or_else(|| {
            lane_defaults
                .as_ref()
                .map(|defaults| defaults.lock_path.clone())
        })
        .unwrap_or_else(|| paper_daemon::resolve_lock_path(None, input.live));
    let status_path = input
        .status_path
        .map(Path::to_path_buf)
        .or_else(|| {
            lane_defaults
                .as_ref()
                .map(|defaults| defaults.status_path.clone())
        })
        .unwrap_or_else(|| paper_daemon::resolve_status_path(None, &lock_path));

    if symbols.is_empty() && symbols_file.is_none() {
        warnings.push(
            "no explicit symbol source resolved; daemon will rely on open paper positions only"
                .to_string(),
        );
    }
    let watch_symbols_file_configured = !(watch_symbols_file && symbols_file.is_none());
    if !watch_symbols_file_configured {
        warnings.push(
            "watch-symbols-file is enabled but no symbols-file source resolved; the daemon would fail closed until a symbols file is configured"
                .to_string(),
        );
    }
    if let Some(symbols_file) = symbols_file.as_ref() {
        if !symbols_file.exists() {
            warnings.push(format!(
                "symbols-file does not exist yet: {}",
                symbols_file.display()
            ));
        }
    }
    if !paper_db.exists() {
        warnings.push(format!(
            "paper db does not exist yet: {}",
            paper_db.display()
        ));
    }
    if !candles_db.exists() {
        warnings.push(format!(
            "candles db does not exist yet: {}",
            candles_db.display()
        ));
    }

    let resume = resolve_resume_state(
        &mut warnings,
        &runtime_bootstrap,
        &effective_config,
        input.live,
        &paper_db,
        &candles_db,
        &interval_symbols,
        watch_symbols_file_configured,
        watch_symbols_file,
        &btc_symbol,
        start_step_close_ts_ms,
    );

    let mut daemon_command = vec![
        "aiq-runtime".to_string(),
        "paper".to_string(),
        "daemon".to_string(),
        "--config".to_string(),
        effective_config.config_path().display().to_string(),
        "--db".to_string(),
        paper_db.display().to_string(),
        "--candles-db".to_string(),
        candles_db.display().to_string(),
        "--btc-symbol".to_string(),
        btc_symbol.clone(),
        "--lookback-bars".to_string(),
        lookback_bars.to_string(),
        "--lock-path".to_string(),
        lock_path.display().to_string(),
        "--status-path".to_string(),
        status_path.display().to_string(),
    ];
    if input.live {
        daemon_command.push("--live".to_string());
    }
    if let Some(profile) = input
        .profile
        .map(str::trim)
        .filter(|profile| !profile.is_empty())
    {
        daemon_command.push("--profile".to_string());
        daemon_command.push(profile.to_string());
    }
    if !symbols.is_empty() {
        daemon_command.push("--symbols".to_string());
        daemon_command.push(symbols.join(","));
    }
    if let Some(symbols_file) = symbols_file.as_ref() {
        daemon_command.push("--symbols-file".to_string());
        daemon_command.push(symbols_file.display().to_string());
    }
    if watch_symbols_file {
        daemon_command.push("--watch-symbols-file".to_string());
    }
    if let Some(start_step_close_ts_ms) = start_step_close_ts_ms {
        daemon_command.push("--start-step-close-ts-ms".to_string());
        daemon_command.push(start_step_close_ts_ms.to_string());
    }

    Ok(PaperManifestReport {
        ok: true,
        runtime_bootstrap,
        base_config_path: effective_config.base_config_path().display().to_string(),
        config_path: effective_config.config_path().display().to_string(),
        active_yaml_path: effective_config.active_yaml_path().display().to_string(),
        effective_yaml_path: effective_config.effective_yaml_path().display().to_string(),
        paper_db: paper_db.display().to_string(),
        paper_db_exists: paper_db.exists(),
        candles_db: candles_db.display().to_string(),
        candles_db_exists: candles_db.exists(),
        interval,
        lookback_bars,
        symbols,
        symbols_file: symbols_file.map(|path| path.display().to_string()),
        watch_symbols_file,
        btc_symbol,
        start_step_close_ts_ms,
        lock_path: lock_path.display().to_string(),
        status_path: status_path.display().to_string(),
        instance_tag: lane_defaults
            .as_ref()
            .map(|defaults| defaults.instance_tag.clone())
            .or_else(|| env_string("AI_QUANT_INSTANCE_TAG")),
        lane: input.lane.map(|lane| lane.as_str().to_string()),
        service_name: lane_defaults
            .as_ref()
            .map(|defaults| defaults.service_name.clone()),
        promoted_role: effective_config.promoted_role().map(ToOwned::to_owned),
        promoted_config_path: effective_config
            .promoted_config_path()
            .map(|path| path.display().to_string()),
        strategy_mode: effective_config.strategy_mode().map(ToOwned::to_owned),
        strategy_mode_source: effective_config
            .strategy_mode_source()
            .map(ToOwned::to_owned),
        strategy_overrides_sha1: effective_config.strategy_overrides_sha1().to_string(),
        config_id: effective_config.config_id().to_string(),
        resume,
        warnings,
        daemon_command,
    })
}

fn resolve_resume_state(
    warnings: &mut Vec<String>,
    runtime_bootstrap: &RuntimeBootstrap,
    effective_config: &PaperEffectiveConfig,
    live: bool,
    paper_db: &Path,
    candles_db: &Path,
    explicit_symbols: &[String],
    watch_symbols_file_configured: bool,
    watch_symbols_file: bool,
    btc_symbol: &str,
    start_step_close_ts_ms: Option<i64>,
) -> PaperManifestResumeState {
    let mut resume = PaperManifestResumeState {
        launch_ready: false,
        launch_state: PaperManifestLaunchState::Blocked,
        active_symbols: Vec::new(),
        last_applied_step_close_ts_ms: None,
        latest_common_close_ts_ms: None,
        next_due_step_close_ts_ms: None,
    };

    if !paper_db.exists() || !candles_db.exists() {
        return resume;
    }
    if !watch_symbols_file_configured {
        return resume;
    }

    let Some(context) = (match paper_loop::inspect_loop_context(
        runtime_bootstrap,
        effective_config,
        live,
        paper_db,
        candles_db,
        explicit_symbols,
        btc_symbol,
    ) {
        Ok(context) => context,
        Err(err) => {
            warnings.push(format!(
                "paper manifest could not inspect daemon resume state: {err:#}"
            ));
            return resume;
        }
    }) else {
        resume.launch_ready = watch_symbols_file;
        resume.launch_state = PaperManifestLaunchState::IdleNoSymbols;
        if !watch_symbols_file {
            warnings.push(
                "paper manifest resolved no active symbols or open paper positions; launch would fail closed without watch-symbols-file"
                    .to_string(),
            );
        }
        return resume;
    };

    resume.active_symbols = context.active_symbols.clone();
    resume.last_applied_step_close_ts_ms = context.last_applied_step_close_ts_ms;
    resume.latest_common_close_ts_ms = Some(context.latest_common_close_ts_ms);

    let interval_ms = match paper_loop::interval_to_ms(&context.interval) {
        Ok(interval_ms) => interval_ms,
        Err(err) => {
            warnings.push(format!(
                "paper manifest could not derive resume interval width: {err:#}"
            ));
            return resume;
        }
    };

    match context.last_applied_step_close_ts_ms {
        Some(last_applied_step_close_ts_ms) => {
            let Some(next_due_step_close_ts_ms) =
                last_applied_step_close_ts_ms.checked_add(interval_ms)
            else {
                warnings.push(
                    "paper manifest detected interval overflow while deriving the next due step"
                        .to_string(),
                );
                return resume;
            };
            resume.next_due_step_close_ts_ms = Some(next_due_step_close_ts_ms);
            if let Some(start_step_close_ts_ms) = start_step_close_ts_ms {
                if start_step_close_ts_ms != next_due_step_close_ts_ms {
                    warnings.push(format!(
                        "paper manifest start-step-close-ts-ms {} does not match the next resumable step {}",
                        start_step_close_ts_ms, next_due_step_close_ts_ms
                    ));
                    return resume;
                }
            }
            resume.launch_ready = true;
            resume.launch_state = if next_due_step_close_ts_ms > context.latest_common_close_ts_ms {
                PaperManifestLaunchState::CaughtUpIdle
            } else {
                PaperManifestLaunchState::ResumeReady
            };
        }
        None => {
            resume.next_due_step_close_ts_ms = start_step_close_ts_ms;
            if start_step_close_ts_ms.is_some() {
                resume.launch_ready = true;
                resume.launch_state = PaperManifestLaunchState::BootstrapReady;
            } else {
                resume.launch_state = PaperManifestLaunchState::BootstrapRequired;
                warnings.push(
                    "paper manifest found no prior runtime_cycle_steps for this config fingerprint and interval; provide --start-step-close-ts-ms or AI_QUANT_PAPER_START_STEP_CLOSE_TS_MS for the first launch"
                        .to_string(),
                );
            }
        }
    }

    resume
}

fn bootstrap_symbol_hint(symbols: &[String]) -> Option<&str> {
    (symbols.len() == 1).then(|| symbols[0].as_str())
}

fn resolve_candles_db(
    candles_db: Option<&Path>,
    lane_defaults: Option<&crate::paper_lane::PaperLaneDefaults>,
    interval: &str,
) -> Result<PathBuf> {
    if let Some(candles_db) = candles_db {
        return Ok(candles_db.to_path_buf());
    }
    if let Some(candles_db) = env_path("AI_QUANT_CANDLES_DB_PATH") {
        return Ok(candles_db);
    }
    if let Some(candles_db_dir) = lane_defaults.map(|defaults| defaults.candles_db_dir.clone()) {
        return Ok(candles_db_dir.join(format!("candles_{}.db", interval)));
    }
    if let Some(candles_db_dir) = env_path("AI_QUANT_CANDLES_DB_DIR") {
        return Ok(candles_db_dir.join(format!("candles_{}.db", interval)));
    }
    anyhow::bail!(
        "paper manifest requires --candles-db, AI_QUANT_CANDLES_DB_PATH, AI_QUANT_CANDLES_DB_DIR, or a lane/project-root default"
    );
}

fn resolve_symbols(cli_symbols: &[String]) -> Vec<String> {
    let mut raw = cli_symbols.to_vec();
    if raw.is_empty() {
        raw.extend(parse_symbol_list(
            &env_string("AI_QUANT_SYMBOLS").unwrap_or_default(),
        ));
    }
    let mut symbols = raw
        .into_iter()
        .filter_map(|symbol| normalise_symbol(symbol))
        .collect::<Vec<_>>();
    symbols.sort();
    symbols.dedup();
    symbols
}

fn resolve_interval_symbols(
    symbols: &[String],
    symbols_file: Option<&Path>,
) -> Result<Vec<String>> {
    let mut merged = symbols.to_vec();
    if let Some(symbols_file) = symbols_file {
        match std::fs::read_to_string(symbols_file) {
            Ok(file_symbols) => merged.extend(parse_symbol_list(&file_symbols)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(err).with_context(|| {
                    format!(
                        "failed to read symbols file while resolving manifest: {}",
                        symbols_file.display()
                    )
                })
            }
        }
    }
    let mut symbols = merged
        .into_iter()
        .filter_map(normalise_symbol)
        .collect::<Vec<_>>();
    symbols.sort();
    symbols.dedup();
    Ok(symbols)
}

fn parse_symbol_list(raw: &str) -> Vec<String> {
    raw.replace('\n', ",")
        .split(',')
        .map(str::trim)
        .filter(|symbol| !symbol.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn normalise_symbol(raw: impl AsRef<str>) -> Option<String> {
    let symbol = raw.as_ref().trim().to_ascii_uppercase();
    (!symbol.is_empty()).then_some(symbol)
}

fn env_string(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_path(name: &str) -> Option<PathBuf> {
    env_string(name).map(PathBuf::from)
}

fn env_usize(name: &str) -> Option<usize> {
    env_string(name).and_then(|value| value.parse::<usize>().ok())
}

fn env_i64(name: &str) -> Option<i64> {
    env_string(name).and_then(|value| value.parse::<i64>().ok())
}

fn env_bool(name: &str) -> bool {
    env_string(name)
        .map(|value| value.to_ascii_lowercase())
        .map(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
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

    use crate::paper_lane::PaperLane;

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    struct EnvGuard {
        saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
    }

    impl EnvGuard {
        fn set(vars: &[(&'static str, Option<&str>)]) -> Self {
            let mut saved = Vec::new();
            for (name, value) in vars {
                saved.push((*name, env::var_os(name)));
                match value {
                    Some(value) => env::set_var(name, value),
                    None => env::remove_var(name),
                }
            }
            Self { saved }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in self.saved.drain(..) {
                match value {
                    Some(value) => env::set_var(name, value),
                    None => env::remove_var(name),
                }
            }
        }
    }

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

    #[test]
    fn manifest_uses_env_defaults_and_derives_candles_db() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let paper_db = dir.path().join("paper.db");
        let candles_dir = dir.path().join("candles");
        let lock_path = dir.path().join("paper.lock");
        let status_path = dir.path().join("paper.status.json");
        fs::create_dir_all(&candles_dir).unwrap();
        fs::write(candles_dir.join("candles_30m.db"), b"").unwrap();
        write_config(&config_path, "30m");

        let _env = EnvGuard::set(&[
            (
                "AI_QUANT_STRATEGY_YAML",
                Some(config_path.to_str().unwrap()),
            ),
            ("AI_QUANT_DB_PATH", Some(paper_db.to_str().unwrap())),
            (
                "AI_QUANT_CANDLES_DB_DIR",
                Some(candles_dir.to_str().unwrap()),
            ),
            ("AI_QUANT_SYMBOLS", Some("btc,eth")),
            ("AI_QUANT_LOOKBACK_BARS", Some("200")),
            ("AI_QUANT_LOCK_PATH", Some(lock_path.to_str().unwrap())),
            ("AI_QUANT_STATUS_PATH", Some(status_path.to_str().unwrap())),
            ("AI_QUANT_INSTANCE_TAG", Some("v8-paper1")),
            ("AI_QUANT_PROMOTED_ROLE", Some("primary")),
            ("AI_QUANT_STRATEGY_MODE", Some("primary")),
            ("AI_QUANT_INTERVAL", None),
            ("AI_QUANT_SYMBOLS_FILE", None),
            ("AI_QUANT_CANDLES_DB_PATH", None),
        ]);

        let report = build_manifest(PaperManifestInput {
            config: None,
            lane: None,
            project_dir: None,
            live: false,
            profile: None,
            db: None,
            candles_db: None,
            symbols: &[],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "btc",
            lookback_bars: None,
            start_step_close_ts_ms: None,
            lock_path: None,
            status_path: None,
        })
        .unwrap();

        assert_eq!(report.interval, "30m");
        assert_eq!(report.lookback_bars, 200);
        assert_eq!(report.symbols, vec!["BTC", "ETH"]);
        assert_eq!(report.config_path, config_path.display().to_string());
        assert_eq!(report.paper_db, paper_db.display().to_string());
        assert_eq!(
            report.candles_db,
            candles_dir.join("candles_30m.db").display().to_string()
        );
        assert_eq!(report.lock_path, lock_path.display().to_string());
        assert_eq!(report.status_path, status_path.display().to_string());
        assert_eq!(report.instance_tag.as_deref(), Some("v8-paper1"));
        assert_eq!(report.promoted_role.as_deref(), Some("primary"));
        assert_eq!(report.strategy_mode.as_deref(), Some("primary"));
        assert!(report
            .daemon_command
            .windows(2)
            .any(|window| window == ["--symbols", "BTC,ETH"]));
        assert!(report
            .daemon_command
            .windows(2)
            .any(|window| window == ["--status-path", status_path.to_str().unwrap()]));
    }

    #[test]
    fn manifest_warns_when_interval_env_disagrees_with_config() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let candles_dir = dir.path().join("candles");
        fs::create_dir_all(&candles_dir).unwrap();
        write_config(&config_path, "30m");

        let _env = EnvGuard::set(&[
            (
                "AI_QUANT_STRATEGY_YAML",
                Some(config_path.to_str().unwrap()),
            ),
            (
                "AI_QUANT_CANDLES_DB_DIR",
                Some(candles_dir.to_str().unwrap()),
            ),
            ("AI_QUANT_INTERVAL", Some("1h")),
            ("AI_QUANT_DB_PATH", None),
            ("AI_QUANT_SYMBOLS", None),
            ("AI_QUANT_SYMBOLS_FILE", None),
            ("AI_QUANT_LOOKBACK_BARS", None),
            ("AI_QUANT_LOCK_PATH", None),
            ("AI_QUANT_STATUS_PATH", None),
            ("AI_QUANT_CANDLES_DB_PATH", None),
        ]);

        let report = build_manifest(PaperManifestInput {
            config: None,
            lane: None,
            project_dir: None,
            live: false,
            profile: Some("production"),
            db: None,
            candles_db: None,
            symbols: &[],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: None,
            start_step_close_ts_ms: None,
            lock_path: None,
            status_path: None,
        })
        .unwrap();

        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("AI_QUANT_INTERVAL=1h differs")));
        assert_eq!(
            report.candles_db,
            candles_dir.join("candles_30m.db").display().to_string()
        );
        assert_eq!(report.runtime_bootstrap.pipeline.profile, "production");
    }

    #[test]
    fn manifest_uses_symbol_specific_shared_interval_for_candles_db() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let candles_dir = dir.path().join("candles");
        fs::create_dir_all(&candles_dir).unwrap();
        write_config(&config_path, "30m");
        fs::write(
            &config_path,
            "global:\n  engine:\n    interval: 30m\nsymbols:\n  BTC:\n    engine:\n      interval: 1h\n",
        )
        .unwrap();

        let _env = EnvGuard::set(&[
            (
                "AI_QUANT_STRATEGY_YAML",
                Some(config_path.to_str().unwrap()),
            ),
            (
                "AI_QUANT_CANDLES_DB_DIR",
                Some(candles_dir.to_str().unwrap()),
            ),
            ("AI_QUANT_SYMBOLS", Some("BTC")),
            ("AI_QUANT_DB_PATH", None),
            ("AI_QUANT_SYMBOLS_FILE", None),
            ("AI_QUANT_LOOKBACK_BARS", None),
            ("AI_QUANT_LOCK_PATH", None),
            ("AI_QUANT_STATUS_PATH", None),
            ("AI_QUANT_CANDLES_DB_PATH", None),
            ("AI_QUANT_INTERVAL", None),
        ]);

        let report = build_manifest(PaperManifestInput {
            config: None,
            lane: None,
            project_dir: None,
            live: false,
            profile: None,
            db: None,
            candles_db: None,
            symbols: &[],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: None,
            start_step_close_ts_ms: None,
            lock_path: None,
            status_path: None,
        })
        .unwrap();

        assert_eq!(report.interval, "1h");
        assert_eq!(
            report.candles_db,
            candles_dir.join("candles_1h.db").display().to_string()
        );
    }

    #[test]
    fn env_path_treats_empty_values_as_unset() {
        let _guard = env_lock().lock().unwrap();
        let _env = EnvGuard::set(&[("AI_QUANT_DB_PATH", Some(""))]);

        assert_eq!(env_path("AI_QUANT_DB_PATH"), None);
    }

    #[test]
    fn manifest_uses_lane_defaults_without_env() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_dir = dir.path().join("config");
        let config_path = config_dir.join("strategy_overrides.paper3.yaml");
        let candles_dir = dir.path().join("candles_dbs");
        fs::create_dir_all(&config_dir).unwrap();
        fs::create_dir_all(&candles_dir).unwrap();
        write_config(&config_path, "30m");

        let _env = EnvGuard::set(&[
            (
                "AI_QUANT_CANDLES_DB_DIR",
                Some(candles_dir.to_str().unwrap()),
            ),
            ("AI_QUANT_STRATEGY_YAML", None),
            ("AI_QUANT_DB_PATH", None),
            ("AI_QUANT_SYMBOLS", None),
            ("AI_QUANT_SYMBOLS_FILE", None),
            ("AI_QUANT_LOOKBACK_BARS", None),
            ("AI_QUANT_LOCK_PATH", None),
            ("AI_QUANT_STATUS_PATH", None),
            ("AI_QUANT_PROMOTED_ROLE", None),
            ("AI_QUANT_STRATEGY_MODE", None),
            ("AI_QUANT_STRATEGY_MODE_FILE", None),
        ]);

        let report = build_manifest(PaperManifestInput {
            config: None,
            lane: Some(PaperLane::Paper3),
            project_dir: Some(dir.path()),
            live: false,
            profile: None,
            db: None,
            candles_db: None,
            symbols: &[],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: None,
            start_step_close_ts_ms: None,
            lock_path: None,
            status_path: None,
        })
        .unwrap();

        assert_eq!(report.lane.as_deref(), Some("paper3"));
        assert_eq!(
            report.service_name.as_deref(),
            Some("openclaw-ai-quant-trader-v8-paper3")
        );
        assert_eq!(report.instance_tag.as_deref(), Some("v8-paper3"));
        assert_eq!(report.promoted_role.as_deref(), Some("conservative"));
        assert_eq!(report.strategy_mode.as_deref(), Some("conservative"));
        assert_eq!(report.strategy_mode_source.as_deref(), Some("lane_default"));
        assert_eq!(report.base_config_path, config_path.display().to_string());
        assert!(report.paper_db.ends_with("trading_engine_v8_paper3.db"));
        assert!(report.lock_path.ends_with("ai_quant_paper_v8_paper3.lock"));
        assert!(report
            .status_path
            .ends_with("ai_quant_paper_v8_paper3.status.json"));
        assert!(report
            .symbols_file
            .as_deref()
            .is_some_and(|path| path.ends_with("artifacts/state/paper_watchlist_paper3.txt")));
        assert!(report.watch_symbols_file);
        assert_eq!(report.lookback_bars, 200);
        assert_eq!(
            report.resume.launch_state,
            PaperManifestLaunchState::Blocked
        );
        assert!(!report.resume.launch_ready);
    }

    #[test]
    fn manifest_reports_bootstrap_required_without_prior_runtime_steps() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles_30m.db");
        write_config(&config_path, "30m");
        init_paper_db(&paper_db);
        init_candles_db(
            &candles_db,
            &["BTC", "ETH"],
            "30m",
            &[1_773_422_400_000, 1_773_424_200_000],
        );

        let report = build_manifest(PaperManifestInput {
            config: Some(&config_path),
            lane: None,
            project_dir: None,
            live: false,
            profile: Some("production"),
            db: Some(&paper_db),
            candles_db: Some(&candles_db),
            symbols: &["eth".to_string()],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: Some(200),
            start_step_close_ts_ms: None,
            lock_path: None,
            status_path: None,
        })
        .unwrap();

        assert_eq!(
            report.resume.launch_state,
            PaperManifestLaunchState::BootstrapRequired
        );
        assert!(!report.resume.launch_ready);
        assert_eq!(report.resume.active_symbols, vec!["ETH"]);
        assert_eq!(report.resume.last_applied_step_close_ts_ms, None);
        assert_eq!(
            report.resume.latest_common_close_ts_ms,
            Some(1_773_424_200_000)
        );
        assert_eq!(report.resume.next_due_step_close_ts_ms, None);
        assert!(report
            .warnings
            .iter()
            .any(|warning| { warning.contains("provide --start-step-close-ts-ms") }));
    }

    #[test]
    fn manifest_reports_bootstrap_ready_and_emits_start_step() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles_30m.db");
        write_config(&config_path, "30m");
        init_paper_db(&paper_db);
        init_candles_db(
            &candles_db,
            &["BTC", "ETH"],
            "30m",
            &[1_773_422_400_000, 1_773_424_200_000],
        );

        let report = build_manifest(PaperManifestInput {
            config: Some(&config_path),
            lane: None,
            project_dir: None,
            live: false,
            profile: None,
            db: Some(&paper_db),
            candles_db: Some(&candles_db),
            symbols: &["ETH".to_string()],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: None,
            start_step_close_ts_ms: Some(1_773_422_400_000),
            lock_path: None,
            status_path: None,
        })
        .unwrap();

        assert_eq!(report.start_step_close_ts_ms, Some(1_773_422_400_000));
        assert_eq!(
            report.resume.launch_state,
            PaperManifestLaunchState::BootstrapReady
        );
        assert!(report.resume.launch_ready);
        assert!(report
            .daemon_command
            .windows(2)
            .any(|window| { window == ["--start-step-close-ts-ms", "1773422400000"] }));
    }

    #[test]
    fn manifest_reports_resume_ready_from_runtime_cycle_steps() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles_30m.db");
        write_config(&config_path, "30m");
        init_paper_db(&paper_db);
        init_candles_db(
            &candles_db,
            &["BTC", "ETH"],
            "30m",
            &[1_773_422_400_000, 1_773_424_200_000, 1_773_426_000_000],
        );

        let config =
            bt_core::config::load_config_checked(config_path.to_str().unwrap(), None, false)
                .unwrap();
        let runtime_bootstrap =
            build_bootstrap(&config, RuntimeMode::Paper, Some("production")).unwrap();
        let step_id = crate::paper_cycle::derive_step_id(
            &runtime_bootstrap.config_fingerprint,
            "30m",
            1_773_422_400_000,
            false,
        );
        let conn = Connection::open(&paper_db).unwrap();
        conn.execute(
            "INSERT INTO runtime_cycle_steps (step_id, step_close_ts_ms, interval, symbols_json, snapshot_exported_at_ms, execution_count, trades_written, created_at)
             VALUES (?1, ?2, '30m', '[\"ETH\"]', ?3, 1, 0, ?4)",
            params![
                step_id,
                1_773_422_400_000_i64,
                1_773_422_400_000_i64,
                Utc::now().to_rfc3339(),
            ],
        )
        .unwrap();

        let report = build_manifest(PaperManifestInput {
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
            lookback_bars: None,
            start_step_close_ts_ms: None,
            lock_path: None,
            status_path: None,
        })
        .unwrap();

        assert_eq!(
            report.resume.launch_state,
            PaperManifestLaunchState::ResumeReady
        );
        assert!(report.resume.launch_ready);
        assert_eq!(
            report.resume.last_applied_step_close_ts_ms,
            Some(1_773_422_400_000)
        );
        assert_eq!(
            report.resume.next_due_step_close_ts_ms,
            Some(1_773_424_200_000)
        );
        assert_eq!(report.resume.active_symbols, vec!["ETH"]);
    }

    #[test]
    fn manifest_blocks_watch_mode_without_symbols_file_source() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles_30m.db");
        write_config(&config_path, "30m");
        init_paper_db(&paper_db);
        init_candles_db(
            &candles_db,
            &["BTC", "ETH"],
            "30m",
            &[1_773_422_400_000, 1_773_424_200_000],
        );

        let report = build_manifest(PaperManifestInput {
            config: Some(&config_path),
            lane: None,
            project_dir: None,
            live: false,
            profile: None,
            db: Some(&paper_db),
            candles_db: Some(&candles_db),
            symbols: &["ETH".to_string()],
            symbols_file: None,
            watch_symbols_file: true,
            btc_symbol: "BTC",
            lookback_bars: None,
            start_step_close_ts_ms: None,
            lock_path: None,
            status_path: None,
        })
        .unwrap();

        assert_eq!(
            report.resume.launch_state,
            PaperManifestLaunchState::Blocked
        );
        assert!(!report.resume.launch_ready);
        assert!(report
            .warnings
            .iter()
            .any(|warning| { warning.contains("watch-symbols-file is enabled") }));
        assert!(report
            .daemon_command
            .iter()
            .any(|arg| arg == "--watch-symbols-file"));
    }

    #[test]
    fn manifest_derives_default_status_path_from_lock_path() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let candles_dir = dir.path().join("candles");
        let lock_path = dir.path().join("paper-daemon.lock");
        fs::create_dir_all(&candles_dir).unwrap();
        fs::write(candles_dir.join("candles_30m.db"), b"").unwrap();
        write_config(&config_path, "30m");

        let _env = EnvGuard::set(&[
            (
                "AI_QUANT_STRATEGY_YAML",
                Some(config_path.to_str().unwrap()),
            ),
            (
                "AI_QUANT_CANDLES_DB_DIR",
                Some(candles_dir.to_str().unwrap()),
            ),
            ("AI_QUANT_LOCK_PATH", Some(lock_path.to_str().unwrap())),
            ("AI_QUANT_DB_PATH", None),
            ("AI_QUANT_SYMBOLS", None),
            ("AI_QUANT_SYMBOLS_FILE", None),
            ("AI_QUANT_LOOKBACK_BARS", None),
            ("AI_QUANT_STATUS_PATH", None),
            ("AI_QUANT_CANDLES_DB_PATH", None),
            ("AI_QUANT_INTERVAL", None),
        ]);

        let report = build_manifest(PaperManifestInput {
            config: None,
            lane: None,
            project_dir: None,
            live: false,
            profile: None,
            db: None,
            candles_db: None,
            symbols: &[],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: None,
            start_step_close_ts_ms: None,
            lock_path: None,
            status_path: None,
        })
        .unwrap();

        assert_eq!(report.lock_path, lock_path.display().to_string());
        assert_eq!(
            report.status_path,
            dir.path()
                .join("paper-daemon.status.json")
                .display()
                .to_string()
        );
    }
}
