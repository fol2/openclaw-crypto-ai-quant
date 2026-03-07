use anyhow::{Context, Result};
use serde::Serialize;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::paper_config::PaperEffectiveConfig;
use crate::paper_daemon::{self, PaperDaemonInput, PaperDaemonReport};
use crate::paper_manifest::{self, PaperManifestInput, PaperManifestReport};
use crate::paper_service::{
    self, PaperServiceApplyInput, PaperServiceApplyReport, PaperServiceApplyRequestedAction,
    PaperServiceInput, PaperServiceReport,
};
use crate::paper_status::{self, PaperStatusInput, PaperStatusReport};

const DEFAULT_LOOKBACK_BARS: usize = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PaperLaneName {
    Paper1,
    Paper2,
    Paper3,
    Livepaper,
}

impl PaperLaneName {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Paper1 => "paper1",
            Self::Paper2 => "paper2",
            Self::Paper3 => "paper3",
            Self::Livepaper => "livepaper",
        }
    }

    fn service_name(self) -> String {
        format!("openclaw-ai-quant-trader-v8-{}", self.as_str())
    }

    fn instance_tag(self) -> String {
        format!("v8-{}", self.as_str())
    }

    fn strategy_yaml_name(self) -> String {
        format!("strategy_overrides.{}.yaml", self.as_str())
    }

    fn db_name(self) -> String {
        format!("trading_engine_v8_{}.db", self.as_str())
    }

    fn lock_name(self) -> String {
        format!("ai_quant_paper_v8_{}.lock", self.as_str())
    }

    fn status_name(self) -> String {
        format!("ai_quant_paper_v8_{}.status.json", self.as_str())
    }

    fn strategy_mode_state_name(self) -> String {
        format!("strategy_mode_v8_{}.txt", self.as_str())
    }

    fn promoted_role(self) -> Option<&'static str> {
        match self {
            Self::Paper1 => Some("primary"),
            Self::Paper2 => Some("fallback"),
            Self::Paper3 => Some("conservative"),
            Self::Livepaper => None,
        }
    }

    fn strategy_mode(self) -> Option<&'static str> {
        match self {
            Self::Paper1 => Some("primary"),
            Self::Paper2 => Some("fallback"),
            Self::Paper3 => Some("conservative"),
            Self::Livepaper => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PaperLaneInput<'a> {
    pub lane: PaperLaneName,
    pub project_dir: Option<&'a Path>,
    pub profile: Option<&'a str>,
    pub symbols: &'a [String],
    pub symbols_file: Option<&'a Path>,
    pub watch_symbols_file: bool,
    pub btc_symbol: &'a str,
    pub lookback_bars: Option<usize>,
    pub start_step_close_ts_ms: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct PaperLaneStatusInput<'a> {
    pub lane: PaperLaneInput<'a>,
    pub stale_after_ms: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct PaperLaneApplyInput<'a> {
    pub lane: PaperLaneStatusInput<'a>,
    pub requested_action: PaperServiceApplyRequestedAction,
    pub start_wait_ms: u64,
    pub stop_wait_ms: u64,
    pub poll_ms: u64,
}

#[derive(Debug, Clone)]
pub struct PaperLaneDaemonInput<'a> {
    pub lane: PaperLaneInput<'a>,
    pub idle_sleep_ms: u64,
    pub max_idle_polls: usize,
    pub exported_at_ms: Option<i64>,
    pub dry_run: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PaperLaneSpecReport {
    pub lane: PaperLaneName,
    pub project_dir: String,
    pub service_name: String,
    pub instance_tag: String,
    pub config_path: String,
    pub paper_db: String,
    pub lock_path: String,
    pub status_path: String,
    pub candles_db_dir: String,
    pub strategy_mode_file: String,
    pub promoted_role: Option<String>,
    pub strategy_mode: Option<String>,
    pub lookback_bars: usize,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperLaneManifestReport {
    pub lane: PaperLaneSpecReport,
    pub manifest: PaperManifestReport,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperLaneStatusReport {
    pub lane: PaperLaneSpecReport,
    pub status: PaperStatusReport,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperLaneServiceReport {
    pub lane: PaperLaneSpecReport,
    pub service: PaperServiceReport,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperLaneApplyReport {
    pub lane: PaperLaneSpecReport,
    pub service_apply: PaperServiceApplyReport,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperLaneDaemonRunReport {
    pub lane: PaperLaneSpecReport,
    pub daemon: PaperDaemonReport,
}

#[derive(Debug, Clone)]
struct PaperLaneSpec {
    lane: PaperLaneName,
    project_dir: PathBuf,
    service_name: String,
    instance_tag: String,
    config_path: PathBuf,
    paper_db: PathBuf,
    lock_path: PathBuf,
    status_path: PathBuf,
    candles_db_dir: PathBuf,
    strategy_mode_file: PathBuf,
    effective_config_output_root: PathBuf,
    artifacts_dir: PathBuf,
    promoted_role: Option<String>,
    strategy_mode: Option<String>,
    lookback_bars: usize,
    warnings: Vec<String>,
}

impl PaperLaneSpec {
    fn resolve(input: &PaperLaneInput<'_>) -> Result<Self> {
        let lane = input.lane;
        let project_dir = input
            .project_dir
            .map(Path::to_path_buf)
            .unwrap_or(std::env::current_dir().context("failed to resolve the current directory")?);
        let project_dir = project_dir.canonicalize().unwrap_or(project_dir);
        let config_path = project_dir.join("config").join(lane.strategy_yaml_name());
        let paper_db = project_dir.join(lane.db_name());
        let lock_path = project_dir.join(lane.lock_name());
        let status_path = project_dir.join(lane.status_name());
        let candles_db_dir = project_dir.join("candles_dbs");
        let strategy_mode_file = project_dir
            .join("artifacts")
            .join("state")
            .join(lane.strategy_mode_state_name());
        let effective_config_output_root = project_dir.clone();
        let artifacts_dir = project_dir.join("artifacts");
        let mut warnings = Vec::new();
        if !config_path.exists() {
            warnings.push(format!(
                "conventional lane config path does not exist yet: {}",
                config_path.display()
            ));
        }
        if !candles_db_dir.exists() {
            warnings.push(format!(
                "conventional candles DB directory does not exist yet: {}",
                candles_db_dir.display()
            ));
        }

        Ok(Self {
            lane,
            project_dir,
            service_name: lane.service_name(),
            instance_tag: lane.instance_tag(),
            config_path,
            paper_db,
            lock_path,
            status_path,
            candles_db_dir,
            strategy_mode_file,
            effective_config_output_root,
            artifacts_dir,
            promoted_role: lane.promoted_role().map(ToOwned::to_owned),
            strategy_mode: lane.strategy_mode().map(ToOwned::to_owned),
            lookback_bars: input.lookback_bars.unwrap_or(DEFAULT_LOOKBACK_BARS),
            warnings,
        })
    }

    fn report(&self) -> PaperLaneSpecReport {
        PaperLaneSpecReport {
            lane: self.lane,
            project_dir: self.project_dir.display().to_string(),
            service_name: self.service_name.clone(),
            instance_tag: self.instance_tag.clone(),
            config_path: self.config_path.display().to_string(),
            paper_db: self.paper_db.display().to_string(),
            lock_path: self.lock_path.display().to_string(),
            status_path: self.status_path.display().to_string(),
            candles_db_dir: self.candles_db_dir.display().to_string(),
            strategy_mode_file: self.strategy_mode_file.display().to_string(),
            promoted_role: self.promoted_role.clone(),
            strategy_mode: self.strategy_mode.clone(),
            lookback_bars: self.lookback_bars,
            warnings: self.warnings.clone(),
        }
    }

    fn env_bindings(&self) -> Vec<(&'static str, Option<OsString>)> {
        vec![
            ("AI_QUANT_MODE", Some(OsString::from("paper"))),
            (
                "AI_QUANT_INSTANCE_TAG",
                Some(OsString::from(&self.instance_tag)),
            ),
            (
                "AI_QUANT_STRATEGY_YAML",
                Some(self.config_path.as_os_str().to_os_string()),
            ),
            (
                "AI_QUANT_DB_PATH",
                Some(self.paper_db.as_os_str().to_os_string()),
            ),
            (
                "AI_QUANT_LOCK_PATH",
                Some(self.lock_path.as_os_str().to_os_string()),
            ),
            (
                "AI_QUANT_STATUS_PATH",
                Some(self.status_path.as_os_str().to_os_string()),
            ),
            (
                "AI_QUANT_CANDLES_DB_DIR",
                Some(self.candles_db_dir.as_os_str().to_os_string()),
            ),
            (
                "AI_QUANT_ARTIFACTS_DIR",
                Some(self.artifacts_dir.as_os_str().to_os_string()),
            ),
            (
                "AI_QUANT_EFFECTIVE_CONFIG_OUTPUT_ROOT",
                Some(self.effective_config_output_root.as_os_str().to_os_string()),
            ),
            (
                "AI_QUANT_STRATEGY_MODE_FILE",
                Some(self.strategy_mode_file.as_os_str().to_os_string()),
            ),
            (
                "AI_QUANT_LOOKBACK_BARS",
                Some(OsString::from(self.lookback_bars.to_string())),
            ),
            (
                "AI_QUANT_PROMOTED_ROLE",
                self.promoted_role.as_ref().map(OsString::from),
            ),
            (
                "AI_QUANT_STRATEGY_MODE",
                self.strategy_mode.as_ref().map(OsString::from),
            ),
        ]
    }
}

struct LaneEnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl LaneEnvGuard {
    fn new(spec: &PaperLaneSpec) -> Self {
        let lock = env_lock().lock().unwrap();
        let mut saved = Vec::new();
        for (name, value) in spec.env_bindings() {
            saved.push((name, std::env::var_os(name)));
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
        Self { _lock: lock, saved }
    }
}

impl Drop for LaneEnvGuard {
    fn drop(&mut self) {
        for (name, value) in self.saved.drain(..) {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn env_lock() -> &'static Mutex<()> {
    ENV_LOCK.get_or_init(|| Mutex::new(()))
}

pub fn build_manifest(input: PaperLaneInput<'_>) -> Result<PaperLaneManifestReport> {
    let spec = PaperLaneSpec::resolve(&input)?;
    let manifest = {
        let _env = LaneEnvGuard::new(&spec);
        paper_manifest::build_manifest(PaperManifestInput {
            config: Some(&spec.config_path),
            live: false,
            profile: input.profile,
            db: Some(&spec.paper_db),
            candles_db: None,
            symbols: input.symbols,
            symbols_file: input.symbols_file,
            watch_symbols_file: input.watch_symbols_file,
            btc_symbol: input.btc_symbol,
            lookback_bars: Some(spec.lookback_bars),
            start_step_close_ts_ms: input.start_step_close_ts_ms,
            lock_path: Some(&spec.lock_path),
            status_path: Some(&spec.status_path),
        })?
    };

    Ok(PaperLaneManifestReport {
        lane: spec.report(),
        manifest,
    })
}

pub fn build_status(input: PaperLaneStatusInput<'_>) -> Result<PaperLaneStatusReport> {
    let spec = PaperLaneSpec::resolve(&input.lane)?;
    let status = {
        let _env = LaneEnvGuard::new(&spec);
        paper_status::build_status(PaperStatusInput {
            config: Some(&spec.config_path),
            live: false,
            profile: input.lane.profile,
            db: Some(&spec.paper_db),
            candles_db: None,
            symbols: input.lane.symbols,
            symbols_file: input.lane.symbols_file,
            watch_symbols_file: input.lane.watch_symbols_file,
            btc_symbol: input.lane.btc_symbol,
            lookback_bars: Some(spec.lookback_bars),
            start_step_close_ts_ms: input.lane.start_step_close_ts_ms,
            lock_path: Some(&spec.lock_path),
            status_path: Some(&spec.status_path),
            stale_after_ms: input.stale_after_ms,
        })?
    };

    Ok(PaperLaneStatusReport {
        lane: spec.report(),
        status,
    })
}

pub fn build_service(input: PaperLaneStatusInput<'_>) -> Result<PaperLaneServiceReport> {
    let spec = PaperLaneSpec::resolve(&input.lane)?;
    let service = {
        let _env = LaneEnvGuard::new(&spec);
        paper_service::build_service(PaperServiceInput {
            config: Some(&spec.config_path),
            live: false,
            profile: input.lane.profile,
            db: Some(&spec.paper_db),
            candles_db: None,
            symbols: input.lane.symbols,
            symbols_file: input.lane.symbols_file,
            watch_symbols_file: input.lane.watch_symbols_file,
            btc_symbol: input.lane.btc_symbol,
            lookback_bars: Some(spec.lookback_bars),
            start_step_close_ts_ms: input.lane.start_step_close_ts_ms,
            lock_path: Some(&spec.lock_path),
            status_path: Some(&spec.status_path),
            stale_after_ms: input.stale_after_ms,
        })?
    };

    Ok(PaperLaneServiceReport {
        lane: spec.report(),
        service,
    })
}

pub fn apply_service(input: PaperLaneApplyInput<'_>) -> Result<PaperLaneApplyReport> {
    let spec = PaperLaneSpec::resolve(&input.lane.lane)?;
    let service_apply = {
        let _env = LaneEnvGuard::new(&spec);
        paper_service::apply_service(PaperServiceApplyInput {
            service: PaperServiceInput {
                config: Some(&spec.config_path),
                live: false,
                profile: input.lane.lane.profile,
                db: Some(&spec.paper_db),
                candles_db: None,
                symbols: input.lane.lane.symbols,
                symbols_file: input.lane.lane.symbols_file,
                watch_symbols_file: input.lane.lane.watch_symbols_file,
                btc_symbol: input.lane.lane.btc_symbol,
                lookback_bars: Some(spec.lookback_bars),
                start_step_close_ts_ms: input.lane.lane.start_step_close_ts_ms,
                lock_path: Some(&spec.lock_path),
                status_path: Some(&spec.status_path),
                stale_after_ms: input.lane.stale_after_ms,
            },
            requested_action: input.requested_action,
            start_wait_ms: input.start_wait_ms,
            stop_wait_ms: input.stop_wait_ms,
            poll_ms: input.poll_ms,
        })?
    };

    Ok(PaperLaneApplyReport {
        lane: spec.report(),
        service_apply,
    })
}

pub fn run_daemon(input: PaperLaneDaemonInput<'_>) -> Result<PaperLaneDaemonRunReport> {
    let spec = PaperLaneSpec::resolve(&input.lane)?;
    let daemon = {
        let _env = LaneEnvGuard::new(&spec);
        let manifest = paper_manifest::build_manifest(PaperManifestInput {
            config: Some(&spec.config_path),
            live: false,
            profile: input.lane.profile,
            db: Some(&spec.paper_db),
            candles_db: None,
            symbols: input.lane.symbols,
            symbols_file: input.lane.symbols_file,
            watch_symbols_file: input.lane.watch_symbols_file,
            btc_symbol: input.lane.btc_symbol,
            lookback_bars: Some(spec.lookback_bars),
            start_step_close_ts_ms: input.lane.start_step_close_ts_ms,
            lock_path: Some(&spec.lock_path),
            status_path: Some(&spec.status_path),
        })?;
        let effective_config = PaperEffectiveConfig::resolve(Some(&spec.config_path))?;
        let candles_db = PathBuf::from(&manifest.candles_db);
        paper_daemon::run_daemon(PaperDaemonInput {
            effective_config,
            runtime_bootstrap: manifest.runtime_bootstrap.clone(),
            profile_override: input.lane.profile,
            live: false,
            paper_db: &spec.paper_db,
            candles_db: &candles_db,
            explicit_symbols: input.lane.symbols,
            symbols_file: input.lane.symbols_file,
            btc_symbol: input.lane.btc_symbol,
            lookback_bars: manifest.lookback_bars,
            start_step_close_ts_ms: input.lane.start_step_close_ts_ms,
            idle_sleep_ms: input.idle_sleep_ms,
            max_idle_polls: input.max_idle_polls,
            exported_at_ms: input.exported_at_ms,
            dry_run: input.dry_run,
            lock_path: Some(&spec.lock_path),
            status_path: Some(&spec.status_path),
            watch_symbols_file: input.lane.watch_symbols_file,
            emit_progress: true,
        })?
    };

    Ok(PaperLaneDaemonRunReport {
        lane: spec.report(),
        daemon,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn paper_lane_spec_maps_candidate_defaults() {
        let dir = tempdir().unwrap();
        let spec = PaperLaneSpec::resolve(&PaperLaneInput {
            lane: PaperLaneName::Paper2,
            project_dir: Some(dir.path()),
            profile: None,
            symbols: &[],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: None,
            start_step_close_ts_ms: None,
        })
        .unwrap();

        assert_eq!(spec.service_name, "openclaw-ai-quant-trader-v8-paper2");
        assert_eq!(spec.instance_tag, "v8-paper2");
        assert_eq!(spec.promoted_role.as_deref(), Some("fallback"));
        assert_eq!(spec.strategy_mode.as_deref(), Some("fallback"));
        assert!(
            spec.config_path
                .ends_with("config/strategy_overrides.paper2.yaml"),
            "unexpected config path: {}",
            spec.config_path.display()
        );
        assert!(
            spec.paper_db.ends_with("trading_engine_v8_paper2.db"),
            "unexpected db path: {}",
            spec.paper_db.display()
        );
    }

    #[test]
    fn livepaper_lane_omits_promoted_role_defaults() {
        let dir = tempdir().unwrap();
        let spec = PaperLaneSpec::resolve(&PaperLaneInput {
            lane: PaperLaneName::Livepaper,
            project_dir: Some(dir.path()),
            profile: None,
            symbols: &[],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC",
            lookback_bars: Some(300),
            start_step_close_ts_ms: None,
        })
        .unwrap();

        assert_eq!(spec.service_name, "openclaw-ai-quant-trader-v8-livepaper");
        assert_eq!(spec.instance_tag, "v8-livepaper");
        assert_eq!(spec.promoted_role, None);
        assert_eq!(spec.strategy_mode, None);
        assert_eq!(spec.lookback_bars, 300);
    }
}
