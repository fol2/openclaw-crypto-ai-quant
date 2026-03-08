use aiq_runtime_core::runtime::{build_bootstrap, RuntimeBootstrap, RuntimeMode};
use anyhow::Result;
use serde::Serialize;
use std::env;
use std::path::{Path, PathBuf};

use crate::live_lane;
use crate::paper_config::PaperEffectiveConfig;

const DEFAULT_LOOKBACK_BARS: usize = 200;
const LIVE_CONFIRM_VALUE: &str = "I_UNDERSTAND_THIS_CAN_LOSE_MONEY";

pub struct LiveManifestInput<'a> {
    pub config: Option<&'a Path>,
    pub project_dir: Option<&'a Path>,
    pub profile: Option<&'a str>,
    pub db: Option<&'a Path>,
    pub market_db: Option<&'a Path>,
    pub lock_path: Option<&'a Path>,
    pub status_path: Option<&'a Path>,
    pub lookback_bars: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveManifestLaunchState {
    Blocked,
    Ready,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LiveManifestReport {
    pub ok: bool,
    pub runtime_owner: String,
    pub launch_state: LiveManifestLaunchState,
    pub runtime_bootstrap: RuntimeBootstrap,
    pub base_config_path: String,
    pub config_path: String,
    pub active_yaml_path: String,
    pub effective_yaml_path: String,
    pub live_db: String,
    pub live_db_exists: bool,
    pub market_db: String,
    pub market_db_exists: bool,
    pub candles_db_dir: String,
    pub candles_db_dir_exists: bool,
    pub interval: String,
    pub lookback_bars: usize,
    pub lock_path: String,
    pub status_path: String,
    pub service_name: String,
    pub instance_tag: String,
    pub strategy_mode_file: String,
    pub event_log_dir: String,
    pub live_enable: bool,
    pub live_confirmed: bool,
    pub safety_gate_ready: bool,
    pub promoted_role: Option<String>,
    pub promoted_config_path: Option<String>,
    pub strategy_mode: Option<String>,
    pub strategy_mode_source: Option<String>,
    pub strategy_overrides_sha1: String,
    pub config_id: String,
    pub daemon_command: Vec<String>,
    pub warnings: Vec<String>,
}

pub fn build_manifest(input: LiveManifestInput<'_>) -> Result<LiveManifestReport> {
    let defaults = live_lane::defaults(input.project_dir)?;
    let base_config_path = input
        .config
        .map(Path::to_path_buf)
        .or_else(|| env_path("AI_QUANT_BASE_STRATEGY_YAML"))
        .or_else(|| env_path("AI_QUANT_STRATEGY_YAML"))
        .unwrap_or_else(|| defaults.config_path.clone());
    let effective_config = PaperEffectiveConfig::resolve(
        Some(base_config_path.as_path()),
        None,
        Some(defaults.project_dir.as_path()),
    )?;
    let mut warnings = effective_config.warnings().to_vec();
    let config = effective_config.load_config(None, true)?;
    let runtime_bootstrap =
        build_bootstrap(&config, RuntimeMode::Live, input.profile).map_err(anyhow::Error::msg)?;
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

    let live_db = input
        .db
        .map(Path::to_path_buf)
        .or_else(|| env_path("AI_QUANT_DB_PATH"))
        .unwrap_or_else(|| defaults.live_db.clone());
    let market_db = input
        .market_db
        .map(Path::to_path_buf)
        .or_else(|| env_path("AI_QUANT_MARKET_DB_PATH"))
        .unwrap_or_else(|| defaults.market_db.clone());
    let lock_path = input
        .lock_path
        .map(Path::to_path_buf)
        .or_else(|| env_path("AI_QUANT_LOCK_PATH"))
        .unwrap_or_else(|| defaults.lock_path.clone());
    let status_path = input
        .status_path
        .map(Path::to_path_buf)
        .or_else(|| env_path("AI_QUANT_STATUS_PATH"))
        .unwrap_or_else(|| defaults.status_path.clone());
    let lookback_bars = input
        .lookback_bars
        .or_else(|| env_usize("AI_QUANT_LOOKBACK_BARS"))
        .unwrap_or(DEFAULT_LOOKBACK_BARS);
    let candles_db_dir =
        env_path("AI_QUANT_CANDLES_DB_DIR").unwrap_or_else(|| defaults.candles_db_dir.clone());
    let strategy_mode_file = env_path("AI_QUANT_STRATEGY_MODE_FILE")
        .unwrap_or_else(|| defaults.strategy_mode_file.clone());
    let event_log_dir =
        env_path("AI_QUANT_EVENT_LOG_DIR").unwrap_or_else(|| defaults.event_log_dir.clone());
    let service_name =
        env_string("AI_QUANT_LIVE_SERVICE_NAME").unwrap_or_else(|| defaults.service_name.clone());
    let instance_tag =
        env_string("AI_QUANT_INSTANCE_TAG").unwrap_or_else(|| defaults.instance_tag.clone());

    let live_enable = env_bool("AI_QUANT_LIVE_ENABLE");
    let live_confirmed = env::var("AI_QUANT_LIVE_CONFIRM")
        .map(|value| value.trim() == LIVE_CONFIRM_VALUE)
        .unwrap_or(false);
    let safety_gate_ready = live_enable && live_confirmed;
    let launch_state = if safety_gate_ready {
        LiveManifestLaunchState::Ready
    } else {
        warnings.push(
            "live safety gates are not fully satisfied; set AI_QUANT_LIVE_ENABLE=1 and AI_QUANT_LIVE_CONFIRM=I_UNDERSTAND_THIS_CAN_LOSE_MONEY before cutover"
                .to_string(),
        );
        LiveManifestLaunchState::Blocked
    };

    if !live_db.exists() {
        warnings.push(format!("live db does not exist yet: {}", live_db.display()));
    }
    if !market_db.exists() {
        warnings.push(format!(
            "market db does not exist yet: {}",
            market_db.display()
        ));
    }
    if !candles_db_dir.exists() {
        warnings.push(format!(
            "candles db dir does not exist yet: {}",
            candles_db_dir.display()
        ));
    }
    if !strategy_mode_file.exists() {
        warnings.push(format!(
            "strategy-mode file does not exist yet: {}",
            strategy_mode_file.display()
        ));
    }

    let runtime_bin = env_path("AI_QUANT_RUNTIME_BIN")
        .unwrap_or_else(|| PathBuf::from("aiq-runtime"))
        .display()
        .to_string();
    let mut daemon_command = vec![
        runtime_bin,
        "live".to_string(),
        "daemon".to_string(),
        "--config".to_string(),
        effective_config.base_config_path().display().to_string(),
        "--db".to_string(),
        live_db.display().to_string(),
        "--lock-path".to_string(),
        lock_path.display().to_string(),
        "--status-path".to_string(),
        status_path.display().to_string(),
        "--btc-symbol".to_string(),
        "BTC".to_string(),
    ];
    if let Some(lookback_bars) = input
        .lookback_bars
        .or_else(|| env_usize("AI_QUANT_LOOKBACK_BARS"))
    {
        daemon_command.push("--lookback-bars".to_string());
        daemon_command.push(lookback_bars.to_string());
    }

    Ok(LiveManifestReport {
        ok: safety_gate_ready,
        runtime_owner: "rust".to_string(),
        launch_state,
        runtime_bootstrap,
        base_config_path: effective_config.base_config_path().display().to_string(),
        config_path: effective_config.config_path().display().to_string(),
        active_yaml_path: effective_config.active_yaml_path().display().to_string(),
        effective_yaml_path: effective_config.effective_yaml_path().display().to_string(),
        live_db: live_db.display().to_string(),
        live_db_exists: live_db.exists(),
        market_db: market_db.display().to_string(),
        market_db_exists: market_db.exists(),
        candles_db_dir: candles_db_dir.display().to_string(),
        candles_db_dir_exists: candles_db_dir.exists(),
        interval,
        lookback_bars,
        lock_path: lock_path.display().to_string(),
        status_path: status_path.display().to_string(),
        service_name,
        instance_tag,
        strategy_mode_file: strategy_mode_file.display().to_string(),
        event_log_dir: event_log_dir.display().to_string(),
        live_enable,
        live_confirmed,
        safety_gate_ready,
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
        daemon_command,
        warnings,
    })
}

fn env_bool(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
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
    env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{env_lock, EnvGuard};
    use std::fs;
    use tempfile::tempdir;

    fn write_config(path: &Path, interval: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            path,
            format!("global:\n  engine:\n    interval: {}\n", interval),
        )
        .unwrap();
    }

    #[test]
    fn manifest_uses_live_defaults_without_env() {
        let _guard = env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = tempdir().unwrap();
        let config_path = dir
            .path()
            .join("config")
            .join("strategy_overrides.live.yaml");
        write_config(&config_path, "30m");

        let _env = EnvGuard::set(&[
            ("AI_QUANT_STRATEGY_YAML", None),
            ("AI_QUANT_DB_PATH", None),
            ("AI_QUANT_MARKET_DB_PATH", None),
            ("AI_QUANT_LOCK_PATH", None),
            ("AI_QUANT_STATUS_PATH", None),
            ("AI_QUANT_LOOKBACK_BARS", None),
            ("AI_QUANT_LIVE_ENABLE", None),
            ("AI_QUANT_LIVE_CONFIRM", None),
            ("AI_QUANT_CANDLES_DB_DIR", None),
            ("AI_QUANT_STRATEGY_MODE_FILE", None),
            ("AI_QUANT_EVENT_LOG_DIR", None),
            ("AI_QUANT_INSTANCE_TAG", None),
            ("AI_QUANT_LIVE_SERVICE_NAME", None),
        ]);

        let report = build_manifest(LiveManifestInput {
            config: None,
            project_dir: Some(dir.path()),
            profile: None,
            db: None,
            market_db: None,
            lock_path: None,
            status_path: None,
            lookback_bars: None,
        })
        .unwrap();

        assert_eq!(report.runtime_owner, "rust");
        assert_eq!(report.launch_state, LiveManifestLaunchState::Blocked);
        assert_eq!(report.service_name, "openclaw-ai-quant-live-v8");
        assert_eq!(report.instance_tag, "v8-LIVE");
        assert!(report.config_path.contains("artifacts/_runtime_configs/"));
        assert!(report.config_path.ends_with(".runtime.yaml"));
        assert!(report.live_db.ends_with("trading_engine_v8_live.db"));
        assert!(report.market_db.ends_with("market_data_v8_live.db"));
        assert!(report.lock_path.ends_with("ai_quant_v8_live.lock"));
        assert!(report.status_path.ends_with("ai_quant_v8_live.status.json"));
        assert!(report.daemon_command.starts_with(&[
            "aiq-runtime".to_string(),
            "live".to_string(),
            "daemon".to_string()
        ]));
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("live safety gates are not fully satisfied")));
    }

    #[test]
    fn manifest_reports_ready_when_live_safety_gates_are_set() {
        let _guard = env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = tempdir().unwrap();
        let config_path = dir
            .path()
            .join("config")
            .join("strategy_overrides.live.yaml");
        write_config(&config_path, "30m");

        let _env = EnvGuard::set(&[
            ("AI_QUANT_LIVE_ENABLE", Some("1")),
            (
                "AI_QUANT_LIVE_CONFIRM",
                Some("I_UNDERSTAND_THIS_CAN_LOSE_MONEY"),
            ),
            ("AI_QUANT_STRATEGY_YAML", None),
            ("AI_QUANT_DB_PATH", None),
            ("AI_QUANT_MARKET_DB_PATH", None),
            ("AI_QUANT_LOCK_PATH", None),
            ("AI_QUANT_STATUS_PATH", None),
            ("AI_QUANT_CANDLES_DB_DIR", None),
            ("AI_QUANT_STRATEGY_MODE_FILE", None),
            ("AI_QUANT_EVENT_LOG_DIR", None),
            ("AI_QUANT_INSTANCE_TAG", None),
            ("AI_QUANT_LIVE_SERVICE_NAME", None),
        ]);

        let report = build_manifest(LiveManifestInput {
            config: None,
            project_dir: Some(dir.path()),
            profile: Some("production"),
            db: None,
            market_db: None,
            lock_path: None,
            status_path: None,
            lookback_bars: Some(200),
        })
        .unwrap();

        assert_eq!(report.launch_state, LiveManifestLaunchState::Ready);
        assert!(report.safety_gate_ready);
        assert!(report.live_enable);
        assert!(report.live_confirmed);
        assert_eq!(
            report.runtime_bootstrap.mode,
            aiq_runtime_core::runtime::RuntimeMode::Live
        );
    }

    #[test]
    fn manifest_prefers_env_strategy_yaml_over_live_default() {
        let _guard = env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let dir = tempdir().unwrap();
        let default_config_path = dir
            .path()
            .join("config")
            .join("strategy_overrides.live.yaml");
        let env_config_path = dir.path().join("config").join("custom-live.yaml");
        write_config(&default_config_path, "30m");
        write_config(&env_config_path, "15m");

        let _env = EnvGuard::set(&[
            (
                "AI_QUANT_STRATEGY_YAML",
                Some(env_config_path.to_str().unwrap()),
            ),
            ("AI_QUANT_BASE_STRATEGY_YAML", None),
            ("AI_QUANT_LIVE_ENABLE", Some("1")),
            (
                "AI_QUANT_LIVE_CONFIRM",
                Some("I_UNDERSTAND_THIS_CAN_LOSE_MONEY"),
            ),
            ("AI_QUANT_DB_PATH", None),
            ("AI_QUANT_MARKET_DB_PATH", None),
            ("AI_QUANT_LOCK_PATH", None),
            ("AI_QUANT_STATUS_PATH", None),
            ("AI_QUANT_CANDLES_DB_DIR", None),
            ("AI_QUANT_STRATEGY_MODE_FILE", None),
            ("AI_QUANT_EVENT_LOG_DIR", None),
            ("AI_QUANT_INSTANCE_TAG", None),
            ("AI_QUANT_LIVE_SERVICE_NAME", None),
        ]);

        let report = build_manifest(LiveManifestInput {
            config: None,
            project_dir: Some(dir.path()),
            profile: None,
            db: None,
            market_db: None,
            lock_path: None,
            status_path: None,
            lookback_bars: None,
        })
        .unwrap();

        assert_eq!(
            report.base_config_path,
            env_config_path.display().to_string()
        );
        assert_eq!(report.interval, "15m");
    }
}
