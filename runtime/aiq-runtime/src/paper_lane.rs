use anyhow::{Context, Result};
use serde::Serialize;
use std::env;
use std::path::{Path, PathBuf};

const DEFAULT_LANE_LOOKBACK_BARS: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PaperLane {
    Paper1,
    Paper2,
    Paper3,
    Livepaper,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaperLaneDefaults {
    pub lane: PaperLane,
    pub project_dir: PathBuf,
    pub service_name: String,
    pub instance_tag: String,
    pub config_path: PathBuf,
    pub paper_db: PathBuf,
    pub lock_path: PathBuf,
    pub status_path: PathBuf,
    pub candles_db_dir: PathBuf,
    pub symbols_file: PathBuf,
    pub strategy_mode_file: PathBuf,
    pub event_log_dir: PathBuf,
    pub promoted_role: Option<String>,
    pub default_strategy_mode: Option<String>,
    pub lookback_bars: usize,
}

impl PaperLane {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Paper1 => "paper1",
            Self::Paper2 => "paper2",
            Self::Paper3 => "paper3",
            Self::Livepaper => "livepaper",
        }
    }

    pub fn service_name(self) -> &'static str {
        match self {
            Self::Paper1 => "openclaw-ai-quant-trader-v8-paper1",
            Self::Paper2 => "openclaw-ai-quant-trader-v8-paper2",
            Self::Paper3 => "openclaw-ai-quant-trader-v8-paper3",
            Self::Livepaper => "openclaw-ai-quant-trader-v8-livepaper",
        }
    }

    pub fn instance_tag(self) -> &'static str {
        match self {
            Self::Paper1 => "v8-paper1",
            Self::Paper2 => "v8-paper2",
            Self::Paper3 => "v8-paper3",
            Self::Livepaper => "v8-livepaper",
        }
    }

    pub fn default_config_basename(self) -> &'static str {
        match self {
            Self::Paper1 => "strategy_overrides.paper1.yaml",
            Self::Paper2 => "strategy_overrides.paper2.yaml",
            Self::Paper3 => "strategy_overrides.paper3.yaml",
            Self::Livepaper => "strategy_overrides.livepaper.yaml",
        }
    }

    pub fn promoted_role(self) -> Option<&'static str> {
        match self {
            Self::Paper1 => Some("primary"),
            Self::Paper2 => Some("fallback"),
            Self::Paper3 => Some("conservative"),
            Self::Livepaper => None,
        }
    }

    pub fn default_strategy_mode(self) -> Option<&'static str> {
        match self {
            Self::Paper1 => Some("primary"),
            Self::Paper2 => Some("fallback"),
            Self::Paper3 => Some("conservative"),
            Self::Livepaper => None,
        }
    }

    pub fn defaults(self, project_dir: Option<&Path>) -> Result<PaperLaneDefaults> {
        let root = resolve_project_dir(project_dir)?;
        let lock_path = root.join(format!(
            "ai_quant_paper_{}.lock",
            self.instance_tag().replace('-', "_")
        ));
        let status_path = default_status_path(&lock_path);
        Ok(PaperLaneDefaults {
            lane: self,
            project_dir: root.clone(),
            service_name: self.service_name().to_string(),
            instance_tag: self.instance_tag().to_string(),
            config_path: root.join("config").join(self.default_config_basename()),
            paper_db: root.join(format!(
                "trading_engine_{}.db",
                self.instance_tag().replace('-', "_")
            )),
            lock_path,
            status_path,
            candles_db_dir: root.join("candles_dbs"),
            symbols_file: root
                .join("artifacts")
                .join("state")
                .join(format!("paper_watchlist_{}.txt", self.as_str())),
            strategy_mode_file: root.join("artifacts").join("state").join(format!(
                "strategy_mode_{}.txt",
                self.instance_tag().replace('-', "_")
            )),
            event_log_dir: root
                .join("artifacts")
                .join("events")
                .join(self.instance_tag().replace('-', "_")),
            promoted_role: self.promoted_role().map(ToOwned::to_owned),
            default_strategy_mode: self.default_strategy_mode().map(ToOwned::to_owned),
            lookback_bars: DEFAULT_LANE_LOOKBACK_BARS,
        })
    }
}

pub(crate) fn default_status_path(lock_path: &Path) -> PathBuf {
    let file_name = lock_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("ai_quant_paper.lock");
    let status_name = file_name
        .strip_suffix(".lock")
        .map(|stem| format!("{stem}.status.json"))
        .unwrap_or_else(|| format!("{file_name}.status.json"));
    match lock_path.parent() {
        Some(parent) => parent.join(status_name),
        None => PathBuf::from(status_name),
    }
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
    fn paper2_defaults_match_v8_conventions() {
        let defaults = PaperLane::Paper2.defaults(None).unwrap();
        assert_eq!(defaults.service_name, "openclaw-ai-quant-trader-v8-paper2");
        assert_eq!(defaults.instance_tag, "v8-paper2");
        assert_eq!(defaults.promoted_role.as_deref(), Some("fallback"));
        assert_eq!(defaults.default_strategy_mode.as_deref(), Some("fallback"));
        assert!(defaults
            .config_path
            .ends_with("config/strategy_overrides.paper2.yaml"));
        assert!(defaults.paper_db.ends_with("trading_engine_v8_paper2.db"));
        assert!(defaults
            .lock_path
            .ends_with("ai_quant_paper_v8_paper2.lock"));
        assert!(defaults
            .status_path
            .ends_with("ai_quant_paper_v8_paper2.status.json"));
        assert!(defaults
            .symbols_file
            .ends_with("artifacts/state/paper_watchlist_paper2.txt"));
        assert!(defaults
            .strategy_mode_file
            .ends_with("artifacts/state/strategy_mode_v8_paper2.txt"));
        assert_eq!(defaults.lookback_bars, 200);
    }

    #[test]
    fn livepaper_defaults_keep_promoted_role_unset() {
        let defaults = PaperLane::Livepaper.defaults(None).unwrap();
        assert_eq!(defaults.promoted_role, None);
        assert_eq!(defaults.default_strategy_mode, None);
        assert!(defaults
            .config_path
            .ends_with("config/strategy_overrides.livepaper.yaml"));
    }
}
