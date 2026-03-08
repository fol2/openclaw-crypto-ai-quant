use anyhow::Result;
use chrono::Utc;
use serde::Serialize;
use std::env;
use std::path::Path;

use crate::live_daemon::{self, LiveDaemonStatusSnapshot};
use crate::live_manifest::{self, LiveManifestInput, LiveManifestLaunchState, LiveManifestReport};

pub struct LiveStatusInput<'a> {
    pub config: Option<&'a Path>,
    pub project_dir: Option<&'a Path>,
    pub profile: Option<&'a str>,
    pub db: Option<&'a Path>,
    pub market_db: Option<&'a Path>,
    pub candles_db: Option<&'a Path>,
    pub symbols: &'a [String],
    pub symbols_file: Option<&'a Path>,
    pub btc_symbol: &'a str,
    pub secrets_path: Option<&'a Path>,
    pub lock_path: Option<&'a Path>,
    pub status_path: Option<&'a Path>,
    pub lookback_bars: Option<usize>,
    pub stale_after_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveServiceState {
    Blocked,
    Ready,
    Running,
    RestartRequired,
    StatusStale,
    Stopped,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LiveStatusReport {
    pub ok: bool,
    pub service_state: LiveServiceState,
    pub status_file_present: bool,
    pub status_age_ms: Option<i64>,
    pub stale_after_ms: Option<i64>,
    pub contract_matches_status: bool,
    pub mismatch_reasons: Vec<String>,
    pub manifest: LiveManifestReport,
    pub daemon_status: Option<LiveDaemonStatusSnapshot>,
    pub warnings: Vec<String>,
}

pub fn build_status(input: LiveStatusInput<'_>) -> Result<LiveStatusReport> {
    let manifest = live_manifest::build_manifest(LiveManifestInput {
        config: input.config,
        project_dir: input.project_dir,
        profile: input.profile,
        db: input.db,
        market_db: input.market_db,
        candles_db: input.candles_db,
        symbols: input.symbols,
        symbols_file: input.symbols_file,
        btc_symbol: input.btc_symbol,
        secrets_path: input.secrets_path,
        lock_path: input.lock_path,
        status_path: input.status_path,
        lookback_bars: input.lookback_bars,
    })?;
    let stale_after_ms = input
        .stale_after_ms
        .or_else(|| env_i64("AI_QUANT_STATUS_STALE_AFTER_MS"));
    let mut warnings = manifest.warnings.clone();
    let daemon_status = live_daemon::load_status_file(Path::new(&manifest.status_path))?;
    let status_file_present = daemon_status.is_some();
    let mut status_age_ms = None;
    let mut mismatch_reasons = Vec::new();
    let mut contract_matches_status = true;
    let mut service_state = launch_state_to_service_state(&manifest.launch_state);
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
                service_state = LiveServiceState::StatusStale;
                warnings.push(format!(
                    "live status marked the daemon lane stale because the status age {}ms exceeds {}ms",
                    status_age_ms.unwrap_or_default(),
                    stale_after_ms.unwrap_or_default()
                ));
            } else if !contract_matches_status {
                service_state = LiveServiceState::RestartRequired;
                warnings.push(format!(
                    "live status detected a running daemon that no longer matches the current launch contract or health gate: {}",
                    mismatch_reasons.join("; ")
                ));
            } else {
                service_state = LiveServiceState::Running;
            }
        } else {
            service_state = LiveServiceState::Stopped;
        }
    } else if manifest.launch_state == LiveManifestLaunchState::Ready {
        warnings.push(format!(
            "live status found no daemon status file at {}; the lane is not currently supervised by a Rust status surface",
            manifest.status_path
        ));
    }

    Ok(LiveStatusReport {
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

fn launch_state_to_service_state(value: &LiveManifestLaunchState) -> LiveServiceState {
    match value {
        LiveManifestLaunchState::Blocked => LiveServiceState::Blocked,
        LiveManifestLaunchState::Ready => LiveServiceState::Ready,
    }
}

fn compare_manifest_and_status(
    manifest: &LiveManifestReport,
    status: &LiveDaemonStatusSnapshot,
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
    if status.live_db != manifest.live_db {
        mismatches.push(format!(
            "live db mismatch (status={} current={})",
            status.live_db, manifest.live_db
        ));
    }
    if status.candles_db != manifest.candles_db {
        mismatches.push(format!(
            "candles db mismatch (status={} current={})",
            status.candles_db, manifest.candles_db
        ));
    }
    if status.btc_symbol != manifest.btc_symbol {
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
    if manifest.symbols_file.is_none() && status.explicit_symbols != manifest.explicit_symbols {
        mismatches.push(format!(
            "explicit symbols mismatch (status={} current={})",
            status.explicit_symbols.join(","),
            manifest.explicit_symbols.join(",")
        ));
    }
    if status.symbols_file != manifest.symbols_file {
        mismatches.push(format!(
            "symbols-file mismatch (status={:?} current={:?})",
            status.symbols_file, manifest.symbols_file
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

    mismatches
}

fn env_i64(name: &str) -> Option<i64> {
    env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<i64>().ok())
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
    fn status_reports_ready_when_launch_contract_is_ready_and_unsupervised() {
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
            ("AI_QUANT_CANDLES_DB_DIR", None),
            ("AI_QUANT_SYMBOLS_FILE", None),
            ("AI_QUANT_STRATEGY_MODE_FILE", None),
            ("AI_QUANT_EVENT_LOG_DIR", None),
            ("AI_QUANT_INSTANCE_TAG", None),
            ("AI_QUANT_LIVE_SERVICE_NAME", None),
            ("AI_QUANT_LOCK_PATH", None),
            ("AI_QUANT_STATUS_PATH", None),
            ("AI_QUANT_SECRETS_PATH", None),
            ("AI_QUANT_LIVE_ENABLE", Some("1")),
            (
                "AI_QUANT_LIVE_CONFIRM",
                Some("I_UNDERSTAND_THIS_CAN_LOSE_MONEY"),
            ),
        ]);

        let report = build_status(LiveStatusInput {
            config: None,
            project_dir: Some(dir.path()),
            profile: None,
            db: None,
            market_db: None,
            candles_db: None,
            symbols: &[],
            symbols_file: None,
            btc_symbol: "BTC",
            secrets_path: None,
            lock_path: None,
            status_path: None,
            lookback_bars: Some(200),
            stale_after_ms: Some(30_000),
        })
        .unwrap();

        assert_eq!(report.service_state, LiveServiceState::Ready);
        assert!(!report.status_file_present);
        assert!(report.manifest.safety_gate_ready);
    }
}
