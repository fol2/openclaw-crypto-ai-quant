use aiq_runtime_core::runtime::{build_bootstrap, RuntimeBootstrap, RuntimeMode};
use anyhow::{Context, Result};
use serde::Serialize;
use std::env;
use std::path::{Path, PathBuf};

use crate::paper_daemon;

const DEFAULT_CONFIG_PATH: &str = "config/strategy_overrides.yaml";
const DEFAULT_PAPER_DB_PATH: &str = "trading_engine.db";
const DEFAULT_LOOKBACK_BARS: usize = 400;

pub struct PaperManifestInput<'a> {
    pub config: Option<&'a Path>,
    pub live: bool,
    pub profile: Option<&'a str>,
    pub db: Option<&'a Path>,
    pub candles_db: Option<&'a Path>,
    pub symbols: &'a [String],
    pub symbols_file: Option<&'a Path>,
    pub btc_symbol: &'a str,
    pub lookback_bars: Option<usize>,
    pub lock_path: Option<&'a Path>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperManifestReport {
    pub ok: bool,
    pub runtime_bootstrap: RuntimeBootstrap,
    pub config_path: String,
    pub paper_db: String,
    pub paper_db_exists: bool,
    pub candles_db: String,
    pub candles_db_exists: bool,
    pub interval: String,
    pub lookback_bars: usize,
    pub symbols: Vec<String>,
    pub symbols_file: Option<String>,
    pub btc_symbol: String,
    pub lock_path: String,
    pub instance_tag: Option<String>,
    pub promoted_role: Option<String>,
    pub strategy_mode: Option<String>,
    pub warnings: Vec<String>,
    pub daemon_command: Vec<String>,
}

pub fn build_manifest(input: PaperManifestInput<'_>) -> Result<PaperManifestReport> {
    let mut warnings = Vec::new();
    let config_path = resolve_config_path(input.config);
    let config = bt_core::config::load_config_checked(
        config_path
            .to_str()
            .context("config path must be valid UTF-8")?,
        None,
        input.live,
    )
    .map_err(anyhow::Error::msg)?;
    let runtime_bootstrap =
        build_bootstrap(&config, RuntimeMode::Paper, input.profile).map_err(anyhow::Error::msg)?;

    let interval = config.engine.interval.trim().to_string();
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
        .or_else(|| env_path("AI_QUANT_DB_PATH"))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PAPER_DB_PATH));

    let candles_db = resolve_candles_db(input.candles_db, &interval)?;
    let symbols = resolve_symbols(input.symbols);
    let symbols_file = input
        .symbols_file
        .map(Path::to_path_buf)
        .or_else(|| env_path("AI_QUANT_SYMBOLS_FILE"));
    let lookback_bars = input
        .lookback_bars
        .or_else(|| env_usize("AI_QUANT_LOOKBACK_BARS"))
        .unwrap_or(DEFAULT_LOOKBACK_BARS);
    let btc_symbol = normalise_symbol(input.btc_symbol).unwrap_or_else(|| "BTC".to_string());
    let lock_path = paper_daemon::resolve_lock_path(input.lock_path, input.live);

    if symbols.is_empty() && symbols_file.is_none() {
        warnings.push(
            "no explicit symbol source resolved; daemon will rely on open paper positions only"
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

    let mut daemon_command = vec![
        "aiq-runtime".to_string(),
        "paper".to_string(),
        "daemon".to_string(),
        "--config".to_string(),
        config_path.display().to_string(),
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

    Ok(PaperManifestReport {
        ok: true,
        runtime_bootstrap,
        config_path: config_path.display().to_string(),
        paper_db: paper_db.display().to_string(),
        paper_db_exists: paper_db.exists(),
        candles_db: candles_db.display().to_string(),
        candles_db_exists: candles_db.exists(),
        interval,
        lookback_bars,
        symbols,
        symbols_file: symbols_file.map(|path| path.display().to_string()),
        btc_symbol,
        lock_path: lock_path.display().to_string(),
        instance_tag: env_string("AI_QUANT_INSTANCE_TAG"),
        promoted_role: env_string("AI_QUANT_PROMOTED_ROLE"),
        strategy_mode: env_string("AI_QUANT_STRATEGY_MODE"),
        warnings,
        daemon_command,
    })
}

fn resolve_config_path(config: Option<&Path>) -> PathBuf {
    let configured = config
        .map(Path::to_path_buf)
        .or_else(|| env_path("AI_QUANT_STRATEGY_YAML"))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG_PATH));

    if configured.exists() {
        return configured;
    }

    let fallback = PathBuf::from(format!("{}.example", configured.display()));
    if fallback.exists() {
        return fallback;
    }

    configured
}

fn resolve_candles_db(candles_db: Option<&Path>, interval: &str) -> Result<PathBuf> {
    if let Some(candles_db) = candles_db {
        return Ok(candles_db.to_path_buf());
    }
    if let Some(candles_db) = env_path("AI_QUANT_CANDLES_DB_PATH") {
        return Ok(candles_db);
    }
    if let Some(candles_db_dir) = env_path("AI_QUANT_CANDLES_DB_DIR") {
        return Ok(candles_db_dir.join(format!("candles_{}.db", interval)));
    }
    anyhow::bail!(
        "paper manifest requires --candles-db, AI_QUANT_CANDLES_DB_PATH, or AI_QUANT_CANDLES_DB_DIR"
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
    env::var_os(name).map(PathBuf::from)
}

fn env_usize(name: &str) -> Option<usize> {
    env_string(name).and_then(|value| value.parse::<usize>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;

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

    #[test]
    fn manifest_uses_env_defaults_and_derives_candles_db() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let paper_db = dir.path().join("paper.db");
        let candles_dir = dir.path().join("candles");
        let lock_path = dir.path().join("paper.lock");
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
            ("AI_QUANT_INSTANCE_TAG", Some("v8-paper1")),
            ("AI_QUANT_PROMOTED_ROLE", Some("primary")),
            ("AI_QUANT_STRATEGY_MODE", Some("primary")),
            ("AI_QUANT_INTERVAL", None),
            ("AI_QUANT_SYMBOLS_FILE", None),
            ("AI_QUANT_CANDLES_DB_PATH", None),
        ]);

        let report = build_manifest(PaperManifestInput {
            config: None,
            live: false,
            profile: None,
            db: None,
            candles_db: None,
            symbols: &[],
            symbols_file: None,
            btc_symbol: "btc",
            lookback_bars: None,
            lock_path: None,
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
        assert_eq!(report.instance_tag.as_deref(), Some("v8-paper1"));
        assert_eq!(report.promoted_role.as_deref(), Some("primary"));
        assert_eq!(report.strategy_mode.as_deref(), Some("primary"));
        assert!(report
            .daemon_command
            .windows(2)
            .any(|window| window == ["--symbols", "BTC,ETH"]));
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
            ("AI_QUANT_CANDLES_DB_PATH", None),
        ]);

        let report = build_manifest(PaperManifestInput {
            config: None,
            live: false,
            profile: Some("production"),
            db: None,
            candles_db: None,
            symbols: &[],
            symbols_file: None,
            btc_symbol: "BTC",
            lookback_bars: None,
            lock_path: None,
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
}
