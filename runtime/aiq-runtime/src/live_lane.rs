use anyhow::{Context, Result};
use std::env;
use std::path::{Path, PathBuf};

use crate::paper_lane::default_status_path;

const DEFAULT_LIVE_LOOKBACK_BARS: usize = 200;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveLaneDefaults {
    pub project_dir: PathBuf,
    pub service_name: String,
    pub instance_tag: String,
    pub config_path: PathBuf,
    pub live_db: PathBuf,
    pub market_db: PathBuf,
    pub lock_path: PathBuf,
    pub status_path: PathBuf,
    pub candles_db_dir: PathBuf,
    pub strategy_mode_file: PathBuf,
    pub event_log_dir: PathBuf,
    pub default_strategy_mode: String,
    pub lookback_bars: usize,
}

pub fn defaults(project_dir: Option<&Path>) -> Result<LiveLaneDefaults> {
    let root = resolve_project_dir(project_dir)?;
    let lock_path = root.join("ai_quant_v8_live.lock");
    let status_path = default_status_path(&lock_path);
    Ok(LiveLaneDefaults {
        project_dir: root.clone(),
        service_name: "openclaw-ai-quant-live-v8".to_string(),
        instance_tag: "v8-LIVE".to_string(),
        config_path: root.join("config").join("strategy_overrides.live.yaml"),
        live_db: root.join("trading_engine_v8_live.db"),
        market_db: root.join("market_data_v8_live.db"),
        lock_path,
        status_path,
        candles_db_dir: root.join("candles_dbs"),
        strategy_mode_file: root
            .join("artifacts")
            .join("state")
            .join("strategy_mode_v8_live.txt"),
        event_log_dir: root.join("artifacts").join("events").join("v8_live"),
        default_strategy_mode: "primary".to_string(),
        lookback_bars: DEFAULT_LIVE_LOOKBACK_BARS,
    })
}

fn resolve_project_dir(project_dir: Option<&Path>) -> Result<PathBuf> {
    let path = project_dir
        .map(Path::to_path_buf)
        .unwrap_or(env::current_dir().context("failed to resolve the current directory")?);
    Ok(path.canonicalize().unwrap_or(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_v8_live_conventions() {
        let defaults = defaults(None).unwrap();
        assert_eq!(defaults.service_name, "openclaw-ai-quant-live-v8");
        assert_eq!(defaults.instance_tag, "v8-LIVE");
        assert!(defaults
            .config_path
            .ends_with("config/strategy_overrides.live.yaml"));
        assert!(defaults.live_db.ends_with("trading_engine_v8_live.db"));
        assert!(defaults.market_db.ends_with("market_data_v8_live.db"));
        assert!(defaults.lock_path.ends_with("ai_quant_v8_live.lock"));
        assert!(defaults
            .status_path
            .ends_with("ai_quant_v8_live.status.json"));
        assert!(defaults
            .strategy_mode_file
            .ends_with("artifacts/state/strategy_mode_v8_live.txt"));
        assert!(defaults.event_log_dir.ends_with("artifacts/events/v8_live"));
        assert_eq!(defaults.default_strategy_mode, "primary");
        assert_eq!(defaults.lookback_bars, 200);
    }
}
