use anyhow::Result;
use serde::Serialize;
use std::path::Path;

use crate::paper_status::{self, PaperServiceState, PaperStatusInput, PaperStatusReport};

pub struct PaperServiceInput<'a> {
    pub config: Option<&'a Path>,
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
pub enum PaperSupervisorAction {
    Hold,
    Start,
    Restart,
    Monitor,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperServiceReport {
    pub ok: bool,
    pub desired_action: PaperSupervisorAction,
    pub action_reason: String,
    pub daemon_command: Vec<String>,
    pub lock_path: String,
    pub status_path: String,
    pub status: PaperStatusReport,
    pub warnings: Vec<String>,
}

pub fn build_service(input: PaperServiceInput<'_>) -> Result<PaperServiceReport> {
    let status = paper_status::build_status(PaperStatusInput {
        config: input.config,
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
        stale_after_ms: input.stale_after_ms,
    })?;
    let (desired_action, action_reason) = derive_action(&status);
    let warnings = status.warnings.clone();

    Ok(PaperServiceReport {
        ok: status.ok,
        desired_action,
        action_reason,
        daemon_command: status.manifest.daemon_command.clone(),
        lock_path: status.manifest.lock_path.clone(),
        status_path: status.manifest.status_path.clone(),
        status,
        warnings,
    })
}

fn derive_action(status: &PaperStatusReport) -> (PaperSupervisorAction, String) {
    match status.service_state {
        PaperServiceState::Blocked => (
            PaperSupervisorAction::Hold,
            "current launch contract is blocked and should not be supervised yet".to_string(),
        ),
        PaperServiceState::IdleNoSymbols => {
            if status.manifest.resume.launch_ready {
                (
                    PaperSupervisorAction::Start,
                    "launch contract is launch-ready but currently has no active symbols; the Rust daemon may start and wait for a watched symbols file or future open positions"
                        .to_string(),
                )
            } else {
                (
                    PaperSupervisorAction::Hold,
                    "no active symbols or open paper positions are currently available for this lane"
                        .to_string(),
                )
            }
        }
        PaperServiceState::BootstrapRequired => (
            PaperSupervisorAction::Hold,
            "first launch still requires --start-step-close-ts-ms or AI_QUANT_PAPER_START_STEP_CLOSE_TS_MS"
                .to_string(),
        ),
        PaperServiceState::BootstrapReady => (
            PaperSupervisorAction::Start,
            "launch contract is bootstrap-ready and no Rust daemon is currently supervising the lane"
                .to_string(),
        ),
        PaperServiceState::ResumeReady => (
            PaperSupervisorAction::Start,
            "launch contract is resumable and the Rust daemon should be started with the current lane contract"
                .to_string(),
        ),
        PaperServiceState::CaughtUpIdle => (
            PaperSupervisorAction::Start,
            "launch contract is caught up and the Rust daemon may start in idle follow mode"
                .to_string(),
        ),
        PaperServiceState::Running => (
            PaperSupervisorAction::Monitor,
            "daemon is running and matches the current launch contract".to_string(),
        ),
        PaperServiceState::RestartRequired => (
            PaperSupervisorAction::Restart,
            format!(
                "running daemon no longer matches the current launch contract: {}",
                status.mismatch_reasons.join("; ")
            ),
        ),
        PaperServiceState::StatusStale => (
            PaperSupervisorAction::Restart,
            match status.status_age_ms.zip(status.stale_after_ms) {
                Some((status_age_ms, stale_after_ms)) => format!(
                    "daemon status is stale: age {}ms exceeds the configured threshold {}ms",
                    status_age_ms, stale_after_ms
                ),
                None => "daemon status is stale and should be refreshed via a supervised restart"
                    .to_string(),
            },
        ),
        PaperServiceState::Stopped => {
            if status.manifest.resume.launch_ready {
                (
                    PaperSupervisorAction::Start,
                    "persisted daemon status is stopped; start the lane with the current launch contract"
                        .to_string(),
                )
            } else {
                (
                    PaperSupervisorAction::Hold,
                    "persisted daemon status is stopped and the current launch contract is not ready to restart"
                        .to_string(),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiq_runtime_core::runtime::{build_bootstrap, RuntimeMode};
    use tempfile::tempdir;

    use crate::paper_manifest::{
        PaperManifestLaunchState, PaperManifestReport, PaperManifestResumeState,
    };

    fn build_manifest_report(
        service_state: PaperServiceState,
        launch_ready: bool,
    ) -> PaperStatusReport {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        std::fs::write(&config_path, "global:\n  engine:\n    interval: 30m\n").unwrap();
        let config =
            bt_core::config::load_config_checked(config_path.to_str().unwrap(), None, false)
                .unwrap();
        let runtime_bootstrap =
            build_bootstrap(&config, RuntimeMode::Paper, Some("production")).unwrap();

        let launch_state = match service_state {
            PaperServiceState::IdleNoSymbols => PaperManifestLaunchState::IdleNoSymbols,
            _ if launch_ready => PaperManifestLaunchState::ResumeReady,
            _ => PaperManifestLaunchState::BootstrapRequired,
        };

        let manifest = PaperManifestReport {
            ok: true,
            runtime_bootstrap,
            base_config_path: config_path.display().to_string(),
            config_path: config_path.display().to_string(),
            active_yaml_path: config_path.display().to_string(),
            effective_yaml_path: config_path.display().to_string(),
            paper_db: dir.path().join("paper.db").display().to_string(),
            paper_db_exists: true,
            candles_db: dir.path().join("candles.db").display().to_string(),
            candles_db_exists: true,
            interval: "30m".to_string(),
            lookback_bars: 200,
            symbols: vec!["ETH".to_string()],
            symbols_file: None,
            watch_symbols_file: false,
            btc_symbol: "BTC".to_string(),
            start_step_close_ts_ms: None,
            lock_path: dir.path().join("paper.lock").display().to_string(),
            status_path: dir.path().join("paper.status.json").display().to_string(),
            instance_tag: None,
            promoted_role: None,
            strategy_mode: None,
            promoted_config_path: None,
            strategy_mode_source: None,
            strategy_overrides_sha1: "a".repeat(64),
            config_id: "a".repeat(64),
            resume: PaperManifestResumeState {
                launch_ready,
                launch_state,
                active_symbols: vec!["ETH".to_string()],
                last_applied_step_close_ts_ms: Some(1_773_424_200_000),
                latest_common_close_ts_ms: Some(1_773_426_000_000),
                next_due_step_close_ts_ms: Some(1_773_426_000_000),
            },
            warnings: Vec::new(),
            daemon_command: vec![
                "aiq-runtime".to_string(),
                "paper".to_string(),
                "daemon".to_string(),
            ],
        };

        PaperStatusReport {
            ok: true,
            service_state,
            status_file_present: true,
            status_age_ms: Some(120_000),
            stale_after_ms: Some(60_000),
            contract_matches_status: true,
            mismatch_reasons: Vec::new(),
            manifest,
            daemon_status: None,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn service_reports_monitor_for_running_lane() {
        let report = build_manifest_report(PaperServiceState::Running, true);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Monitor);
        assert!(reason.contains("matches the current launch contract"));
    }

    #[test]
    fn service_reports_restart_for_restart_required_lane() {
        let mut report = build_manifest_report(PaperServiceState::RestartRequired, true);
        report.mismatch_reasons = vec!["config fingerprint mismatch".to_string()];
        report.contract_matches_status = false;

        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Restart);
        assert!(reason.contains("config fingerprint mismatch"));
    }

    #[test]
    fn service_reports_restart_for_stale_lane() {
        let report = build_manifest_report(PaperServiceState::StatusStale, true);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Restart);
        assert!(reason.contains("stale"));
    }

    #[test]
    fn service_reports_hold_for_bootstrap_required_lane() {
        let report = build_manifest_report(PaperServiceState::BootstrapRequired, false);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Hold);
        assert!(reason.contains("start-step-close-ts-ms"));
    }

    #[test]
    fn service_reports_start_for_launch_ready_idle_watch_lane() {
        let report = build_manifest_report(PaperServiceState::IdleNoSymbols, true);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Start);
        assert!(reason.contains("wait for a watched symbols file"));
    }

    #[test]
    fn service_reports_hold_for_non_ready_idle_lane() {
        let report = build_manifest_report(PaperServiceState::IdleNoSymbols, false);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Hold);
        assert!(reason.contains("no active symbols"));
    }

    #[test]
    fn service_reports_start_for_stopped_launch_ready_lane() {
        let report = build_manifest_report(PaperServiceState::Stopped, true);
        let (action, reason) = derive_action(&report);

        assert_eq!(action, PaperSupervisorAction::Start);
        assert!(reason.contains("stopped"));
    }
}
