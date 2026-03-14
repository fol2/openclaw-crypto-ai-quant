use anyhow::{anyhow, bail, Context, Result};
use bt_core::config::{
    load_config_document_checked, materialise_runtime_document, strategy_config_fingerprint_sha256,
    strategy_config_to_yaml_value, StrategyConfig,
};
use bt_core::sweep::apply_one_pub;
use bt_data::sqlite_loader::query_time_range_multi;
use chrono::{SecondsFormat, Utc};
use clap::{Args, Parser, Subcommand, ValueEnum};
use fs2::FileExt;
use reqwest::blocking::Client;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::Duration;

#[path = "../live_secrets.rs"]
mod live_secrets;
#[cfg(test)]
#[allow(dead_code)]
#[path = "../paper_config.rs"]
mod paper_config;
#[cfg(test)]
#[allow(dead_code)]
#[path = "../paper_lane.rs"]
mod paper_lane;
#[cfg(test)]
#[path = "../test_support.rs"]
mod test_support;

const DEFAULT_FACTORY_SETTINGS: &str = "config/factory_defaults.yaml";
const DEFAULT_CONFIG_PATH: &str = "config/strategy_overrides.yaml";
const DEFAULT_ARTIFACTS_DIR: &str = "artifacts";
const DEFAULT_CANDLES_DB_DIR: &str = "candles_dbs";
const DEFAULT_FUNDING_DB: &str = "candles_dbs/funding_rates.db";
const DEFAULT_FACTORY_VERSION: &str = "factory_cycle_rust_v1";
const DEFAULT_SELECTION_POLICY: &str = "promotion_roles_v2";
const DEFAULT_LIVE_YAML_PATH: &str = "config/strategy_overrides.live.yaml";
const DEFAULT_LIVE_SERVICE: &str = "openclaw-ai-quant-live-v8";
const DEFAULT_PRIMARY_PAPER_DB_PATH: &str = "trading_engine_v8_paper1.db";
const DEFAULT_LIVE_SECRETS_PATH: &str = "~/.config/openclaw/ai-quant-secrets.json";

#[derive(Debug, Parser)]
#[command(
    name = "aiq-factory",
    version,
    about = "Rust-owned strategy factory cycle for AI Quant"
)]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Debug, Subcommand)]
enum CommandKind {
    /// Run the factory cycle and emit the resulting artefact bundle.
    Run(RunArgs),
}

#[derive(Debug, Clone, Args)]
struct RunArgs {
    /// Project root. Defaults to the current working directory.
    #[arg(long)]
    project_dir: Option<PathBuf>,
    /// Base strategy config.
    #[arg(long, default_value = DEFAULT_CONFIG_PATH)]
    config: PathBuf,
    /// Factory defaults/settings YAML.
    #[arg(long, default_value = DEFAULT_FACTORY_SETTINGS)]
    settings: PathBuf,
    /// Factory profile to execute.
    #[arg(long, value_enum, default_value = "daily")]
    profile: FactoryProfile,
    /// Emit JSON summary to stdout.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, ValueEnum)]
enum FactoryProfile {
    Daily,
    Deep,
}

impl FactoryProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::Daily => "daily",
            Self::Deep => "deep",
        }
    }

    fn run_prefix(self) -> &'static str {
        match self {
            Self::Daily => "nightly",
            Self::Deep => "deep",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct FactoryDefaults {
    version: u32,
    backtester_bin: Option<PathBuf>,
    artifacts_dir: PathBuf,
    candles_db_dir: PathBuf,
    funding_db: PathBuf,
    balance: BalanceSettings,
    comparison: ComparisonSettings,
    profiles: BTreeMap<String, FactoryProfileSettings>,
    validation: ValidationSettings,
    selection: SelectionSettings,
    deployment: DeploymentSettings,
}

impl Default for FactoryDefaults {
    fn default() -> Self {
        let mut profiles = BTreeMap::new();
        profiles.insert("daily".to_string(), FactoryProfileSettings::daily());
        profiles.insert("deep".to_string(), FactoryProfileSettings::deep());
        Self {
            version: 1,
            backtester_bin: None,
            artifacts_dir: PathBuf::from(DEFAULT_ARTIFACTS_DIR),
            candles_db_dir: PathBuf::from(DEFAULT_CANDLES_DB_DIR),
            funding_db: PathBuf::from(DEFAULT_FUNDING_DB),
            balance: BalanceSettings::default(),
            comparison: ComparisonSettings::default(),
            profiles,
            validation: ValidationSettings::default(),
            selection: SelectionSettings::default(),
            deployment: DeploymentSettings::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct FactoryProfileSettings {
    sweep_spec: PathBuf,
    initial_balance: f64,
    gpu: bool,
    tpe: bool,
    tpe_trials: usize,
    tpe_batch: usize,
    tpe_seed: u64,
    sweep_top_k: usize,
    shortlist_per_mode: usize,
    allow_unsafe_gpu_sweep: bool,
}

impl FactoryProfileSettings {
    fn daily() -> Self {
        Self {
            sweep_spec: PathBuf::from("backtester/sweeps/full_144v.yaml"),
            initial_balance: 10_000.0,
            gpu: true,
            tpe: true,
            tpe_trials: 1_000_000,
            tpe_batch: 256,
            tpe_seed: 42,
            sweep_top_k: 50_000,
            shortlist_per_mode: 5,
            allow_unsafe_gpu_sweep: false,
        }
    }

    fn deep() -> Self {
        Self {
            tpe_trials: 10_000_000,
            shortlist_per_mode: 8,
            ..Self::daily()
        }
    }
}

impl Default for FactoryProfileSettings {
    fn default() -> Self {
        Self::daily()
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BalanceSourceMode {
    Fixed,
    LiveEquity,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct BalanceSettings {
    mode: BalanceSourceMode,
    live_secrets_path: Option<PathBuf>,
}

impl Default for BalanceSettings {
    fn default() -> Self {
        Self {
            mode: BalanceSourceMode::LiveEquity,
            live_secrets_path: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct ComparisonSettings {
    require_challenger_win: bool,
}

impl Default for ComparisonSettings {
    fn default() -> Self {
        Self {
            require_challenger_win: true,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct BalanceResolution {
    source: &'static str,
    amount_usd: f64,
    secrets_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct ValidationSettings {
    min_trades: u32,
    slippage_bps: f64,
    max_top1_pnl_pct: f64,
    walk_forward_splits: usize,
    parity_abs_eps: f64,
    parity_rel_eps: f64,
    parity_trade_delta_max: u32,
    parity_enforce: bool,
}

impl Default for ValidationSettings {
    fn default() -> Self {
        Self {
            min_trades: 30,
            slippage_bps: 20.0,
            max_top1_pnl_pct: 0.50,
            walk_forward_splits: 3,
            parity_abs_eps: 0.001,
            parity_rel_eps: 0.000_001,
            parity_trade_delta_max: 0,
            parity_enforce: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct SelectionSettings {
    selected_roles: Vec<String>,
    paper_targets: Vec<PaperTarget>,
}

impl Default for SelectionSettings {
    fn default() -> Self {
        Self {
            selected_roles: vec![
                "primary".to_string(),
                "fallback".to_string(),
                "conservative".to_string(),
            ],
            paper_targets: vec![
                PaperTarget {
                    role: "primary".to_string(),
                    slot: 1,
                    service: "openclaw-ai-quant-trader-v8-paper1".to_string(),
                    yaml_path: PathBuf::from("config/strategy_overrides.paper1.yaml"),
                },
                PaperTarget {
                    role: "fallback".to_string(),
                    slot: 2,
                    service: "openclaw-ai-quant-trader-v8-paper2".to_string(),
                    yaml_path: PathBuf::from("config/strategy_overrides.paper2.yaml"),
                },
                PaperTarget {
                    role: "conservative".to_string(),
                    slot: 3,
                    service: "openclaw-ai-quant-trader-v8-paper3".to_string(),
                    yaml_path: PathBuf::from("config/strategy_overrides.paper3.yaml"),
                },
            ],
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct DeploymentSettings {
    apply_to_paper: bool,
    restart_services: bool,
    apply_to_live: bool,
    restart_live_service: bool,
    live_yaml_path: PathBuf,
    live_service: String,
}

impl Default for DeploymentSettings {
    fn default() -> Self {
        Self {
            apply_to_paper: true,
            restart_services: true,
            apply_to_live: false,
            restart_live_service: true,
            live_yaml_path: PathBuf::from(DEFAULT_LIVE_YAML_PATH),
            live_service: DEFAULT_LIVE_SERVICE.to_string(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct PaperTarget {
    role: String,
    slot: usize,
    service: String,
    yaml_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
struct SweepCandidateRow {
    candidate_mode: bool,
    #[allow(dead_code)]
    config_id: String,
    max_drawdown_pct: f64,
    overrides: BTreeMap<String, f64>,
    profit_factor: f64,
    total_pnl: f64,
    total_trades: u32,
}

#[derive(Debug, Clone)]
struct ShortlistedCandidate {
    shortlist_mode: ShortlistMode,
    rank: usize,
    sweep: SweepCandidateRow,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
enum ShortlistMode {
    Efficient,
    Growth,
    Conservative,
}

impl ShortlistMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Efficient => "efficient",
            Self::Growth => "growth",
            Self::Conservative => "conservative",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReplaySummary {
    initial_balance: f64,
    final_balance: f64,
    total_pnl: f64,
    total_trades: u32,
    profit_factor: f64,
    max_drawdown_pct: f64,
    total_fees: f64,
    by_symbol: Vec<SymbolBucket>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SymbolBucket {
    symbol: String,
    trades: u32,
    pnl: f64,
    win_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ParityCheck {
    status: String,
    cpu_total_trades: u32,
    gpu_total_trades: u32,
    trade_delta: u32,
    cpu_final_balance: f64,
    gpu_final_balance: f64,
    balance_delta_abs: f64,
    balance_delta_rel: f64,
    thresholds: ParityThresholds,
}

#[derive(Debug, Clone, Serialize)]
struct ParityThresholds {
    abs_eps: f64,
    rel_eps: f64,
    trade_delta_max: u32,
}

#[derive(Debug, Clone, Serialize)]
struct SplitReturn {
    split: usize,
    start_ts_ms: i64,
    end_ts_ms: i64,
    days: f64,
    final_balance: f64,
    total_pnl: f64,
    daily_return: f64,
}

#[derive(Debug, Clone, Serialize)]
struct WalkForwardSummary {
    split_count: usize,
    median_oos_daily_return: f64,
    splits: Vec<SplitReturn>,
}

#[derive(Debug, Clone, Serialize)]
struct ValidationItem {
    candidate_mode: bool,
    canonical_cpu_verified: bool,
    config_id: String,
    config_path: String,
    config_sha256: String,
    final_balance: f64,
    initial_balance: f64,
    max_drawdown_pct: f64,
    pipeline_stage: String,
    profit_factor: f64,
    rank: usize,
    replay_report_path: String,
    replay_stage: String,
    schema_version: u32,
    shortlist_mode: String,
    slippage_pnl_at_reject_bps: f64,
    slippage_reject_bps: f64,
    sort_by: String,
    step4_parity: ParityCheck,
    sweep_stage: String,
    top1_pnl_pct: f64,
    total_fees: f64,
    total_pnl: f64,
    total_trades: u32,
    validation_gate: String,
    walk_forward_summary_path: String,
    wf_median_oos_daily_return: f64,
    blocked_reasons: Vec<String>,
    rejected: bool,
    reject_reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct SlotBinding {
    role: String,
    slot: usize,
    service: String,
    yaml_path: String,
    selected: bool,
    config_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct GateReport {
    schema_version: u32,
    run_id: String,
    generated_at_ms: i64,
    selection_policy: &'static str,
    require_ssot_evidence: bool,
    candidate_count: usize,
    deployable_count: usize,
    selected_count: usize,
    blocked: bool,
    blocked_reason: String,
    blocked_reasons_count: usize,
    selection_warnings: Vec<String>,
    candidates: Vec<ValidationItem>,
    slot_bindings: Vec<SlotBinding>,
}

#[derive(Debug, Clone, Serialize)]
struct SelectionCandidate {
    role: String,
    slot: usize,
    service: String,
    source: String,
    selected: bool,
    config_id: String,
}

#[derive(Debug, Clone, Serialize)]
struct SelectionReport {
    version: &'static str,
    run_id: String,
    run_dir: String,
    interval: String,
    deploy_stage: String,
    deployed: bool,
    deployments: Vec<DeploymentEvent>,
    challenges: Vec<DeploymentChallenge>,
    promotion_stage: String,
    selection_policy: &'static str,
    selection_stage: String,
    selection_warnings: Vec<String>,
    step5_gate_status: String,
    step5_gate_block_reason: String,
    paper_promotion_gate: Option<PaperPromotionGate>,
    live_promotion: Option<LivePromotionEvent>,
    effective_config_id: String,
    effective_config_path: String,
    promotion_reference_epoch_s: f64,
    evidence_bundle_paths: SelectionEvidencePaths,
    selected: ValidationItem,
    selected_candidates: Vec<ValidationItem>,
    selected_candidates_by_role: Vec<SelectionCandidate>,
    selected_targets: Vec<PaperTargetSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct SelectionEvidencePaths {
    run_dir: String,
    run_metadata_json: String,
    report_json: String,
    report_md: String,
    selection_json: String,
    selection_md: String,
    step5_gate_report_json: String,
    step5_gate_report_md: String,
    configs_dir: String,
    replays_dir: String,
}

#[derive(Debug, Clone, Serialize)]
struct PaperTargetSummary {
    service: String,
    slot: usize,
    yaml_path: String,
}

#[derive(Debug, Clone, Serialize)]
struct DeploymentEvent {
    role: String,
    slot: usize,
    source_config_path: String,
    config_id: String,
    config_sha256: String,
    promoted_config_path: String,
    target_yaml_path: String,
    service: String,
    soak_marker_path: Option<String>,
    restarted_service: bool,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct PerformanceSummary {
    config_id: String,
    config_path: String,
    config_sha256: String,
    final_balance: f64,
    initial_balance: f64,
    max_drawdown_pct: f64,
    profit_factor: f64,
    total_pnl: f64,
    total_trades: u32,
    total_fees: f64,
    top1_pnl_pct: f64,
    slippage_pnl_at_reject_bps: f64,
    slippage_reject_bps: f64,
    wf_median_oos_daily_return: f64,
    replay_report_path: String,
    walk_forward_summary_path: String,
    rejected: bool,
    reject_reason: String,
    blocked_reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct DeploymentChallenge {
    role: String,
    slot: usize,
    service: String,
    target_yaml_path: String,
    source_config_path: String,
    decision: String,
    reason: String,
    incumbent: Option<PerformanceSummary>,
    challenger: PerformanceSummary,
}

#[derive(Debug, Clone)]
struct PreparedDeployment {
    role: String,
    slot: usize,
    service: String,
    config_id: String,
    config_sha256: String,
    source_config_path: PathBuf,
    promoted_config_path: PathBuf,
    target_yaml_path: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
struct LivePromotionEvent {
    role: String,
    config_id: String,
    source_config_path: String,
    promoted_config_path: String,
    live_yaml_path: String,
    live_service: String,
    deployment_dir: String,
    service_was_active_before: bool,
    restarted_service: bool,
    restart_required: bool,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct PaperPromotionGate {
    status: String,
    role: String,
    paper_db_path: String,
    config_fingerprint: String,
    first_trade_ts: Option<String>,
    last_trade_ts: Option<String>,
    runtime_hours: f64,
    close_trades: u32,
    profit_factor: f64,
    max_drawdown_pct: f64,
    slippage_pnl_at_reject_bps: f64,
    kill_switch_events: u32,
    thresholds: PaperPromotionThresholds,
    blocked_reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PaperPromotionThresholds {
    min_runtime_hours: f64,
    min_close_trades: u32,
    min_profit_factor: f64,
    max_drawdown_pct: f64,
    require_positive_slippage_pnl: bool,
    require_zero_kill_switch_events: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PaperSoakMarker {
    version: String,
    role: String,
    slot: usize,
    service: String,
    paper_db_path: String,
    config_id: String,
    config_sha256: String,
    deployed_at_ms: i64,
    deployed_by_run_id: String,
    source_config_path: String,
    target_yaml_path: String,
}

#[derive(Debug, Clone)]
struct TargetSnapshot {
    target_yaml_path: PathBuf,
    original_bytes: Option<Vec<u8>>,
}

#[derive(Debug)]
struct FactoryRunLock {
    file: File,
}

impl Drop for FactoryRunLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

#[derive(Debug, Clone, Serialize)]
struct RunMetadata {
    version: &'static str,
    generated_at_ms: i64,
    run_id: String,
    git_head: String,
    args: RunMetadataArgs,
    items: Vec<ValidationItem>,
    selected_roles: Vec<SelectionCandidate>,
}

#[derive(Debug, Clone, Serialize)]
struct RunMetadataArgs {
    profile: String,
    config: String,
    settings: String,
    sweep_spec: String,
    run_id: String,
    shortlist_per_mode: usize,
    min_trades: u32,
    slippage_reject_bps: f64,
    max_top1_pnl_pct: f64,
    walk_forward_splits: usize,
    gpu: bool,
    tpe: bool,
    tpe_trials: usize,
    tpe_batch: usize,
    tpe_seed: u64,
    sweep_top_k: usize,
    initial_balance: f64,
    initial_balance_source: String,
    initial_balance_secrets_path: Option<String>,
    candles_db_dir: String,
    funding_db: String,
    start_ts_ms: i64,
    end_ts_ms: i64,
    apply_to_paper: bool,
    restart_paper_services: bool,
    apply_to_live: bool,
    restart_live_service: bool,
    live_yaml_path: String,
    live_service: String,
}

#[derive(Debug, Clone, Serialize)]
struct FactoryRunSummary {
    job: String,
    version: &'static str,
    run_id: String,
    run_dir: String,
    report_json: String,
    selection_json: String,
    initial_balance_source: String,
    initial_balance_usd: f64,
    blocked: bool,
    blocked_reason: String,
    promotion_stage: String,
    paper_promotion_gate: Option<PaperPromotionGate>,
    live_promotion: Option<LivePromotionEvent>,
    challenges: Vec<DeploymentChallenge>,
    selected_roles: Vec<SelectionCandidate>,
}

struct ReplayCommandInput<'a> {
    paths: &'a ResolvedPaths,
    config_path: &'a Path,
    initial_balance: f64,
    db_window: &'a ResolvedDbWindow,
    output_path: &'a Path,
    stdout_path: &'a Path,
    stderr_path: &'a Path,
    slippage_bps: Option<f64>,
}

struct SweepCommandInput<'a> {
    paths: &'a ResolvedPaths,
    profile: &'a FactoryProfileSettings,
    config_path: &'a Path,
    initial_balance: f64,
    db_window: &'a ResolvedDbWindow,
    output_path: &'a Path,
    stderr_path: &'a Path,
    stdout_path: &'a Path,
}

struct SelectionReportInput<'a> {
    run_id: &'a str,
    run_dir: &'a Path,
    effective_config_id: &'a str,
    effective_config_path: &'a Path,
    validated: &'a [ValidationItem],
    selected: &'a [SelectionCandidate],
    selection: &'a SelectionSettings,
    base_cfg: &'a StrategyConfig,
    blocked: bool,
    blocked_reason: String,
    challenges: Vec<DeploymentChallenge>,
    deployments: Vec<DeploymentEvent>,
    paper_promotion_gate: Option<PaperPromotionGate>,
    live_promotion: Option<LivePromotionEvent>,
    promotion_requested: bool,
    selection_warnings: Vec<String>,
}

struct RunMetadataInput<'a> {
    run_id: &'a str,
    paths: &'a ResolvedPaths,
    settings: &'a FactoryDefaults,
    balance: &'a BalanceResolution,
    profile: FactoryProfile,
    profile_settings: &'a FactoryProfileSettings,
    db_window: &'a ResolvedDbWindow,
    validated: &'a [ValidationItem],
    selected: &'a [SelectionCandidate],
}

struct ChallengeContext<'a> {
    paths: &'a ResolvedPaths,
    validation: &'a ValidationSettings,
    selection: &'a SelectionSettings,
    comparison: &'a ComparisonSettings,
    db_window: &'a ResolvedDbWindow,
    selected: &'a [SelectionCandidate],
    validated: &'a [ValidationItem],
    initial_balance: f64,
    run_dir: &'a Path,
}

#[derive(Debug, Clone)]
struct ResolvedPaths {
    project_dir: PathBuf,
    config_path: PathBuf,
    settings_path: PathBuf,
    artifacts_root: PathBuf,
    candles_db_dir: PathBuf,
    funding_db_path: PathBuf,
    backtester_bin: PathBuf,
}

#[derive(Debug, Clone)]
struct ResolvedDbWindow {
    main_interval: String,
    entry_interval: String,
    exit_interval: String,
    main_db: PathBuf,
    entry_db: PathBuf,
    exit_db: PathBuf,
    start_ts_ms: i64,
    end_ts_ms: i64,
}

#[derive(Debug, Clone)]
struct MaterialisedRoleConfig {
    document: serde_yaml::Value,
    config_id: String,
    mode_applied: bool,
    warnings: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CommandKind::Run(args) => {
            let summary = run_factory_cycle(args)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&summary).context("serialise factory summary")?
            );
        }
    }
    Ok(())
}

fn run_factory_cycle(args: RunArgs) -> Result<FactoryRunSummary> {
    let project_dir = args
        .project_dir
        .unwrap_or(std::env::current_dir().context("resolve current directory")?)
        .canonicalize()
        .context("canonicalise project_dir")?;
    let settings_path = resolve_under_project(&project_dir, &args.settings);
    let config_path = resolve_under_project(&project_dir, &args.config);
    let settings = load_factory_defaults(&settings_path)?;
    validate_selection_settings(&settings.selection)?;
    validate_factory_defaults(&settings)?;
    let profile_settings = settings
        .profiles
        .get(args.profile.as_str())
        .cloned()
        .ok_or_else(|| anyhow!("missing profile settings for {}", args.profile.as_str()))?;
    let backtester_bin = resolve_backtester_bin(&project_dir, settings.backtester_bin.clone())?;
    let paths = ResolvedPaths {
        project_dir: project_dir.clone(),
        config_path: config_path.clone(),
        settings_path: settings_path.clone(),
        artifacts_root: resolve_under_project(&project_dir, &settings.artifacts_dir),
        candles_db_dir: resolve_under_project(&project_dir, &settings.candles_db_dir),
        funding_db_path: resolve_under_project(&project_dir, &settings.funding_db),
        backtester_bin,
    };
    let resolved_balance =
        resolve_initial_balance(&paths.project_dir, &settings.balance, &profile_settings)?;

    fs::create_dir_all(&paths.artifacts_root).with_context(|| {
        format!(
            "create artifacts root {}",
            paths.artifacts_root.as_path().display()
        )
    })?;
    let _factory_lock = acquire_factory_lock(&paths.project_dir)?;

    let now = Utc::now();
    let run_stamp = now.format("%Y%m%dT%H%M%SZ").to_string();
    let unique_stamp = format!("{run_stamp}_{:03}", now.timestamp_subsec_millis());
    let run_id = format!("{}_{}", args.profile.run_prefix(), unique_stamp);
    let run_dir = paths
        .artifacts_root
        .join(now.format("%Y-%m-%d").to_string())
        .join(format!("run_{run_id}"));
    prepare_run_dirs(&run_dir)?;

    let raw_document: serde_yaml::Value = serde_yaml::from_str(
        &fs::read_to_string(&paths.config_path)
            .with_context(|| format!("read base config {}", paths.config_path.display()))?,
    )
    .with_context(|| format!("parse base config {}", paths.config_path.display()))?;
    let base_cfg = load_config_document_checked(&raw_document, None, false, None)
        .map_err(|err| anyhow!(err))
        .context("load base factory config")?;
    let effective_config_id = strategy_config_fingerprint_sha256(&base_cfg);
    let effective_config_path = write_effective_config(
        &paths.artifacts_root,
        args.profile,
        &unique_stamp,
        &paths.config_path,
        &raw_document,
    )?;

    let db_window = compute_common_time_range(&base_cfg, &paths.candles_db_dir)?;
    let sweep_candidates_path = run_dir.join("sweeps/sweep_candidates.jsonl");
    let sweep_results_path = run_dir.join("sweeps/sweep_results.jsonl");
    run_sweep(SweepCommandInput {
        paths: &paths,
        profile: &profile_settings,
        config_path: &effective_config_path,
        initial_balance: resolved_balance.amount_usd,
        db_window: &db_window,
        output_path: &sweep_candidates_path,
        stderr_path: &run_dir.join("sweeps/sweep.stderr.txt"),
        stdout_path: &run_dir.join("sweeps/sweep.stdout.txt"),
    })?;
    fs::copy(&sweep_candidates_path, &sweep_results_path).with_context(|| {
        format!(
            "copy {} to {}",
            sweep_candidates_path.display(),
            sweep_results_path.display()
        )
    })?;

    let sweep_rows = parse_sweep_candidates(&sweep_candidates_path)?;
    if sweep_rows.is_empty() {
        bail!("factory sweep produced no candidate rows");
    }
    let shortlist = build_shortlist(&sweep_rows, profile_settings.shortlist_per_mode);
    let mut validated = Vec::new();
    for candidate in shortlist {
        validated.push(validate_candidate(
            &paths,
            &settings.validation,
            &raw_document,
            &base_cfg,
            &run_dir,
            &db_window,
            candidate,
            resolved_balance.amount_usd,
        )?);
    }
    if validated.is_empty() {
        bail!("factory validation produced no candidate evidence");
    }

    let selected = select_roles(&validated, &settings.selection);
    let challenges = build_deployment_challenges(ChallengeContext {
        paths: &paths,
        validation: &settings.validation,
        selection: &settings.selection,
        comparison: &settings.comparison,
        db_window: &db_window,
        selected: &selected,
        validated: &validated,
        initial_balance: resolved_balance.amount_usd,
        run_dir: &run_dir,
    })?;
    let gate_report = build_gate_report(&run_id, &validated, &selected, &settings.selection);
    let (paper_deployed, deployments, paper_promotion_gate, live_promotion, selection_warnings) =
        if gate_report.blocked {
            (false, Vec::new(), None, None, Vec::new())
        } else {
            match deploy_selected_configs(
                &paths.project_dir,
                &run_id,
                &run_dir,
                &challenges,
                &settings.deployment,
            ) {
                Ok(events) => {
                    let mut warnings = Vec::new();
                    let (paper_promotion_gate, live_promotion) =
                        if settings.deployment.apply_to_live {
                            match assess_primary_paper_promotion_gate(
                                &paths.project_dir,
                                &settings.selection,
                                &validated,
                                &selected,
                            ) {
                                Ok(gate) if gate.status == "pass" => {
                                    match promote_primary_live(
                                        &paths.project_dir,
                                        &paths.artifacts_root,
                                        &run_dir,
                                        &validated,
                                        &selected,
                                        &events,
                                        &settings.deployment,
                                    ) {
                                        Ok(event) => (Some(gate), Some(event)),
                                        Err(err) => {
                                            warnings.push(format!("live_promotion_failed: {err}"));
                                            (Some(gate), None)
                                        }
                                    }
                                }
                                Ok(gate) => {
                                    warnings.push(format!(
                                        "live_promotion_blocked: {}",
                                        gate.blocked_reasons.join("; ")
                                    ));
                                    (Some(gate), None)
                                }
                                Err(err) => {
                                    warnings.push(format!("live_promotion_failed: {err}"));
                                    (None, None)
                                }
                            }
                        } else {
                            (None, None)
                        };
                    (true, events, paper_promotion_gate, live_promotion, warnings)
                }
                Err(err) => (
                    false,
                    Vec::new(),
                    None,
                    None,
                    vec![format!("paper_deploy_failed: {err}")],
                ),
            }
        };
    let selection_report = build_selection_report(SelectionReportInput {
        run_id: &run_id,
        run_dir: &run_dir,
        effective_config_id: &effective_config_id,
        effective_config_path: &effective_config_path,
        validated: &validated,
        selected: &selected,
        selection: &settings.selection,
        base_cfg: &base_cfg,
        blocked: gate_report.blocked,
        blocked_reason: gate_report.blocked_reason.clone(),
        challenges: challenges.clone(),
        deployments: deployments.clone(),
        paper_promotion_gate: paper_promotion_gate.clone(),
        live_promotion: live_promotion.clone(),
        promotion_requested: settings.deployment.apply_to_live,
        selection_warnings: selection_warnings.clone(),
    })?;
    write_reports(&run_dir, &validated, &gate_report, &selection_report)?;
    let metadata = build_run_metadata(RunMetadataInput {
        run_id: &run_id,
        paths: &paths,
        settings: &settings,
        balance: &resolved_balance,
        profile: args.profile,
        profile_settings: &profile_settings,
        db_window: &db_window,
        validated: &validated,
        selected: &selected,
    })?;
    write_json(&run_dir.join("run_metadata.json"), &metadata)?;

    Ok(FactoryRunSummary {
        job: "factory".to_string(),
        version: DEFAULT_FACTORY_VERSION,
        run_id,
        run_dir: run_dir.display().to_string(),
        report_json: run_dir.join("reports/report.json").display().to_string(),
        selection_json: run_dir.join("reports/selection.json").display().to_string(),
        initial_balance_source: resolved_balance.source.to_string(),
        initial_balance_usd: resolved_balance.amount_usd,
        blocked: gate_report.blocked
            || !paper_deployed
            || (settings.deployment.apply_to_live && live_promotion.is_none()),
        blocked_reason: if !selection_warnings.is_empty() {
            selection_warnings.join("; ")
        } else {
            gate_report.blocked_reason.clone()
        },
        promotion_stage: selection_report.promotion_stage.clone(),
        paper_promotion_gate,
        live_promotion,
        challenges,
        selected_roles: selected,
    })
}

fn load_factory_defaults(settings_path: &Path) -> Result<FactoryDefaults> {
    if !settings_path.is_file() {
        return Ok(FactoryDefaults::default());
    }
    let text = fs::read_to_string(settings_path)
        .with_context(|| format!("read factory defaults {}", settings_path.display()))?;
    let mut loaded: FactoryDefaults = serde_yaml::from_str(&text)
        .with_context(|| format!("parse factory defaults {}", settings_path.display()))?;
    if loaded.profiles.is_empty() {
        loaded.profiles = FactoryDefaults::default().profiles;
    }
    Ok(loaded)
}

fn acquire_factory_lock(project_dir: &Path) -> Result<FactoryRunLock> {
    let lock_dir = project_dir.join("artifacts").join("_locks");
    fs::create_dir_all(&lock_dir).with_context(|| format!("create {}", lock_dir.display()))?;
    let lock_path = lock_dir.join("factory_cycle.lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("open {}", lock_path.display()))?;
    file.try_lock_exclusive().with_context(|| {
        format!(
            "factory cycle already running (lock: {})",
            lock_path.display()
        )
    })?;
    Ok(FactoryRunLock { file })
}

fn resolve_under_project(project_dir: &Path, raw: &Path) -> PathBuf {
    if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        project_dir.join(raw)
    }
}

fn resolve_backtester_bin(project_dir: &Path, configured: Option<PathBuf>) -> Result<PathBuf> {
    let candidates = configured
        .into_iter()
        .map(|path| resolve_under_project(project_dir, &path))
        .chain([
            project_dir.join("backtester/target/release/mei-backtester"),
            project_dir.join("target/release/mei-backtester"),
            project_dir.join("backtester/target/debug/mei-backtester"),
        ]);
    for candidate in candidates {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!(
        "unable to locate the Rust backtester binary; build `mei-backtester` or set backtester_bin in {}",
        DEFAULT_FACTORY_SETTINGS
    )
}

fn resolve_initial_balance(
    project_dir: &Path,
    balance: &BalanceSettings,
    profile: &FactoryProfileSettings,
) -> Result<BalanceResolution> {
    resolve_initial_balance_with(project_dir, balance, profile, fetch_live_equity)
}

fn resolve_initial_balance_with<F>(
    project_dir: &Path,
    balance: &BalanceSettings,
    profile: &FactoryProfileSettings,
    fetcher: F,
) -> Result<BalanceResolution>
where
    F: Fn(&Path) -> Result<f64>,
{
    match balance.mode {
        BalanceSourceMode::Fixed => Ok(BalanceResolution {
            source: "fixed",
            amount_usd: profile.initial_balance,
            secrets_path: None,
        }),
        BalanceSourceMode::LiveEquity => {
            let secrets_path =
                resolve_live_secrets_path(project_dir, balance.live_secrets_path.as_ref())?;
            let amount_usd = fetcher(&secrets_path)?;
            if amount_usd <= 0.0 {
                bail!(
                    "live equity must be > 0.0 for factory balance seeding, got {}",
                    amount_usd
                );
            }
            Ok(BalanceResolution {
                source: "live_equity",
                amount_usd,
                secrets_path: Some(secrets_path.display().to_string()),
            })
        }
    }
}

fn resolve_live_secrets_path(project_dir: &Path, configured: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(path) = configured {
        return Ok(resolve_under_project(project_dir, path));
    }
    if let Ok(path) = env::var("AI_QUANT_SECRETS_PATH") {
        let path = path.trim();
        if !path.is_empty() {
            return Ok(PathBuf::from(path));
        }
    }
    Ok(live_secrets::expand_path(Path::new(
        DEFAULT_LIVE_SECRETS_PATH,
    )))
}

fn fetch_live_equity(secrets_path: &Path) -> Result<f64> {
    let secrets = live_secrets::load_live_secrets(secrets_path)?;
    let payload = serde_json::json!({
        "type": "clearinghouseState",
        "user": secrets.main_address,
    });
    let response: serde_json::Value = Client::builder()
        .timeout(Duration::from_secs_f64(4.0))
        .build()
        .context("build Hyperliquid HTTP client")?
        .post("https://api.hyperliquid.xyz/info")
        .json(&payload)
        .send()
        .context("request live account snapshot")?
        .error_for_status()
        .context("live account snapshot returned error status")?
        .json()
        .context("parse live account snapshot JSON")?;
    let account_value = response
        .get("marginSummary")
        .and_then(|value| value.get("accountValue"))
        .ok_or_else(|| anyhow!("clearinghouseState missing marginSummary.accountValue"))?;
    match account_value {
        serde_json::Value::Number(number) => number
            .as_f64()
            .ok_or_else(|| anyhow!("accountValue is not a finite number")),
        serde_json::Value::String(text) => text
            .trim()
            .parse::<f64>()
            .with_context(|| format!("parse accountValue `{text}`")),
        other => bail!("unsupported accountValue JSON type: {other}"),
    }
}

fn prepare_run_dirs(run_dir: &Path) -> Result<()> {
    for dir in [
        run_dir.to_path_buf(),
        run_dir.join("configs"),
        run_dir.join("replays"),
        run_dir.join("sweeps"),
        run_dir.join("reports"),
        run_dir.join("promoted_configs"),
    ] {
        fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    }
    Ok(())
}

fn write_effective_config(
    artifacts_root: &Path,
    profile: FactoryProfile,
    run_stamp: &str,
    base_config_path: &Path,
    base_document: &serde_yaml::Value,
) -> Result<PathBuf> {
    let effective_dir = artifacts_root.join("_effective_configs");
    fs::create_dir_all(&effective_dir)
        .with_context(|| format!("create {}", effective_dir.display()))?;
    let path = effective_dir.join(format!("{}_{}.yaml", profile.run_prefix(), run_stamp));
    let header = format!(
        "# Generated by aiq-factory at {}\n# Source: {}\n",
        Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        base_config_path.display()
    );
    write_yaml_document(&path, header, base_document)
        .with_context(|| format!("write effective config {}", path.display()))?;
    Ok(path)
}

fn write_yaml_document(path: &Path, header: String, document: &serde_yaml::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let yaml = serde_yaml::to_string(document).context("serialise YAML document")?;
    fs::write(path, format!("{header}{yaml}"))
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn read_yaml_document(path: &Path) -> Result<serde_yaml::Value> {
    serde_yaml::from_str(
        &fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))
}

fn replace_global_preserving_root(
    base_document: &serde_yaml::Value,
    strategy_config: &StrategyConfig,
) -> Result<serde_yaml::Value> {
    let mut output = match base_document {
        serde_yaml::Value::Mapping(root) => serde_yaml::Value::Mapping(root.clone()),
        _ => bail!("factory config document must have a mapping root"),
    };
    let output_map = output
        .as_mapping_mut()
        .context("factory config document must have a mapping root")?;
    output_map.insert(
        serde_yaml::Value::String("global".to_string()),
        strategy_config_to_yaml_value(strategy_config),
    );
    Ok(output)
}

fn shortlist_mode_target_role(mode: ShortlistMode) -> &'static str {
    match mode {
        ShortlistMode::Efficient => "primary",
        ShortlistMode::Growth => "fallback",
        ShortlistMode::Conservative => "conservative",
    }
}

fn role_strategy_mode(role: &str) -> Option<&'static str> {
    match role {
        "primary" => Some("primary"),
        "fallback" => Some("fallback"),
        "conservative" => Some("conservative"),
        _ => None,
    }
}

fn materialise_role_effective_config(
    document: &serde_yaml::Value,
    role: &str,
) -> Result<MaterialisedRoleConfig> {
    let mut warnings = Vec::new();
    let (effective_document, mode_applied) =
        apply_strategy_mode_overlay_factory(document, role_strategy_mode(role), &mut warnings)?;
    let cfg = load_config_document_checked(&effective_document, None, false, None)
        .map_err(|err| anyhow!(err))
        .context("load lane-effective config")?;
    Ok(MaterialisedRoleConfig {
        document: effective_document,
        config_id: strategy_config_fingerprint_sha256(&cfg),
        mode_applied,
        warnings,
    })
}

fn write_role_effective_config(
    source_document: &serde_yaml::Value,
    role: &str,
    output_path: &Path,
    source_path: &Path,
    strict_mode: bool,
) -> Result<MaterialisedRoleConfig> {
    let materialised = materialise_role_effective_config(source_document, role)?;
    if strict_mode && role_strategy_mode(role).is_some() && !materialised.mode_applied {
        let detail = if materialised.warnings.is_empty() {
            "missing strategy-mode overlay".to_string()
        } else {
            materialised.warnings.join("; ")
        };
        bail!(
            "failed to materialise lane-effective config for role `{role}` from {}: {detail}",
            source_path.display()
        );
    }
    let header = format!(
        "# Generated by aiq-factory at {}\n# Role: {role}\n# Source: {}\n# Strategy mode: {}\n",
        Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        source_path.display(),
        role_strategy_mode(role).unwrap_or("none")
    );
    write_yaml_document(output_path, header, &materialised.document)?;
    Ok(materialised)
}

fn write_role_effective_config_from_path(
    source_path: &Path,
    role: &str,
    output_path: &Path,
    strict_mode: bool,
) -> Result<MaterialisedRoleConfig> {
    let source_document = read_yaml_document(source_path)?;
    write_role_effective_config(
        &source_document,
        role,
        output_path,
        source_path,
        strict_mode,
    )
}

fn apply_strategy_mode_overlay_factory(
    document: &serde_yaml::Value,
    strategy_mode: Option<&str>,
    warnings: &mut Vec<String>,
) -> Result<(serde_yaml::Value, bool)> {
    let Some(mode_key) = strategy_mode else {
        return Ok((document.clone(), false));
    };

    let serde_yaml::Value::Mapping(root_map) = document else {
        bail!("lane-effective config document must have a mapping root");
    };
    let modes_key = serde_yaml::Value::String("modes".to_string());
    let Some(modes_value) = root_map.get(&modes_key) else {
        warnings.push(format!(
            "strategy mode {mode_key} requested but no modes section exists in the active YAML"
        ));
        return Ok((document.clone(), false));
    };
    let serde_yaml::Value::Mapping(modes_map) = modes_value else {
        warnings.push("modes section is not a mapping; ignoring strategy mode".to_string());
        return Ok((document.clone(), false));
    };
    let Some(mode_overlay) = lookup_mapping_value(modes_map, mode_key) else {
        warnings.push(format!(
            "strategy mode {mode_key} was requested but no matching modes entry was found"
        ));
        return Ok((document.clone(), false));
    };

    let mut output = document.clone();
    let output_map = output
        .as_mapping_mut()
        .context("lane-effective config document must have a mapping root")?;
    let global_key = serde_yaml::Value::String("global".to_string());
    let symbols_key = serde_yaml::Value::String("symbols".to_string());
    output_map
        .entry(global_key.clone())
        .or_insert_with(|| serde_yaml::Value::Mapping(Default::default()));
    output_map
        .entry(symbols_key.clone())
        .or_insert_with(|| serde_yaml::Value::Mapping(Default::default()));

    if let serde_yaml::Value::Mapping(mode_map) = mode_overlay {
        let has_structured_overlay =
            mode_map.contains_key(&global_key) || mode_map.contains_key(&symbols_key);
        if has_structured_overlay {
            if let Some(global_overlay) = mode_map.get(&global_key) {
                let global = output_map
                    .get_mut(&global_key)
                    .context("global section must exist")?;
                bt_core::config::deep_merge_yaml_value_python_compat(global, global_overlay);
            }
            if let Some(symbols_overlay) = mode_map.get(&symbols_key) {
                let symbols = output_map
                    .get_mut(&symbols_key)
                    .context("symbols section must exist")?;
                bt_core::config::deep_merge_yaml_value_python_compat(symbols, symbols_overlay);
            }
            return Ok((output, true));
        }
    }

    let global = output_map
        .get_mut(&global_key)
        .context("global section must exist")?;
    bt_core::config::deep_merge_yaml_value_python_compat(global, mode_overlay);
    Ok((output, true))
}

fn lookup_mapping_value<'a>(
    map: &'a serde_yaml::Mapping,
    key: &str,
) -> Option<&'a serde_yaml::Value> {
    let exact = serde_yaml::Value::String(key.to_string());
    let upper = serde_yaml::Value::String(key.to_ascii_uppercase());
    let lower = serde_yaml::Value::String(key.to_ascii_lowercase());
    map.get(&exact)
        .or_else(|| map.get(&upper))
        .or_else(|| map.get(&lower))
}

fn compute_common_time_range(
    cfg: &StrategyConfig,
    candles_db_dir: &Path,
) -> Result<ResolvedDbWindow> {
    let main_interval = cfg.engine.interval.trim();
    let entry_interval = if cfg.engine.entry_interval.trim().is_empty() {
        main_interval
    } else {
        cfg.engine.entry_interval.trim()
    };
    let exit_interval = if cfg.engine.exit_interval.trim().is_empty() {
        main_interval
    } else {
        cfg.engine.exit_interval.trim()
    };
    let main_db = candles_db_path(candles_db_dir, main_interval);
    let entry_db = candles_db_path(candles_db_dir, entry_interval);
    let exit_db = candles_db_path(candles_db_dir, exit_interval);
    let dbs = [
        (vec![main_db.clone()], main_interval),
        (vec![entry_db.clone()], entry_interval),
        (vec![exit_db.clone()], exit_interval),
    ];
    let mut overlap_from = i64::MIN;
    let mut overlap_to = i64::MAX;
    for (paths, interval) in dbs {
        let path_strings = paths
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>();
        let (min_t, max_t) = query_time_range_multi(&path_strings, interval)
            .map_err(|err| anyhow!("query DB range for interval={interval}: {err}"))?
            .ok_or_else(|| anyhow!("no candles found for interval={interval}"))?;
        overlap_from = overlap_from.max(min_t);
        overlap_to = overlap_to.min(max_t);
    }
    if overlap_from >= overlap_to {
        bail!("no overlapping candle coverage across main/entry/exit intervals");
    }
    Ok(ResolvedDbWindow {
        main_interval: main_interval.to_string(),
        entry_interval: entry_interval.to_string(),
        exit_interval: exit_interval.to_string(),
        main_db,
        entry_db,
        exit_db,
        start_ts_ms: overlap_from,
        end_ts_ms: overlap_to,
    })
}

fn candles_db_path(candles_db_dir: &Path, interval: &str) -> PathBuf {
    candles_db_dir.join(format!("candles_{}.db", interval))
}

fn run_sweep(input: SweepCommandInput<'_>) -> Result<()> {
    let SweepCommandInput {
        paths,
        profile,
        config_path,
        initial_balance,
        db_window,
        output_path,
        stderr_path,
        stdout_path,
    } = input;
    let mut cmd = Command::new(&paths.backtester_bin);
    cmd.arg("sweep")
        .arg("--config")
        .arg(config_path)
        .arg("--sweep-spec")
        .arg(resolve_under_project(
            &paths.project_dir,
            &profile.sweep_spec,
        ))
        .arg("--output")
        .arg(output_path)
        .arg("--output-mode")
        .arg("candidate")
        .arg("--initial-balance")
        .arg(initial_balance.to_string())
        .arg("--candles-db")
        .arg(&db_window.main_db)
        .arg("--interval")
        .arg(&db_window.main_interval)
        .arg("--entry-candles-db")
        .arg(&db_window.entry_db)
        .arg("--entry-interval")
        .arg(&db_window.entry_interval)
        .arg("--exit-candles-db")
        .arg(&db_window.exit_db)
        .arg("--exit-interval")
        .arg(&db_window.exit_interval)
        .arg("--funding-db")
        .arg(&paths.funding_db_path)
        .arg("--start-ts")
        .arg(db_window.start_ts_ms.to_string())
        .arg("--end-ts")
        .arg(db_window.end_ts_ms.to_string())
        .arg("--no-auto-scope")
        .arg("--sweep-top-k")
        .arg(profile.sweep_top_k.to_string());
    if profile.gpu {
        cmd.arg("--gpu");
    }
    if profile.tpe {
        cmd.arg("--tpe")
            .arg("--tpe-trials")
            .arg(profile.tpe_trials.to_string())
            .arg("--tpe-batch")
            .arg(profile.tpe_batch.to_string())
            .arg("--tpe-seed")
            .arg(profile.tpe_seed.to_string());
    }
    if profile.allow_unsafe_gpu_sweep {
        cmd.arg("--allow-unsafe-gpu-sweep");
    }
    run_command(&mut cmd, &paths.project_dir, stdout_path, stderr_path)
        .context("run factory sweep")?;
    Ok(())
}

fn parse_sweep_candidates(path: &Path) -> Result<Vec<SweepCandidateRow>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut rows = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: SweepCandidateRow = serde_json::from_str(line)
            .with_context(|| format!("parse sweep candidate row {}", idx + 1))?;
        rows.push(row);
    }
    Ok(rows)
}

fn build_shortlist(rows: &[SweepCandidateRow], per_mode: usize) -> Vec<ShortlistedCandidate> {
    let mut shortlist = Vec::new();
    let mut seen = HashSet::new();
    for (mode, sorter) in [
        (
            ShortlistMode::Efficient,
            shortlist_order_efficient as fn(&SweepCandidateRow, &SweepCandidateRow) -> Ordering,
        ),
        (ShortlistMode::Growth, shortlist_order_growth),
        (ShortlistMode::Conservative, shortlist_order_conservative),
    ] {
        let mut ranked = rows.to_vec();
        ranked.sort_by(sorter);
        for (idx, row) in ranked.into_iter().enumerate() {
            let override_key = stable_override_key(&row.overrides);
            if !seen.insert(override_key) {
                continue;
            }
            shortlist.push(ShortlistedCandidate {
                shortlist_mode: mode,
                rank: idx + 1,
                sweep: row,
            });
            if shortlist
                .iter()
                .filter(|item| item.shortlist_mode == mode)
                .count()
                >= per_mode
            {
                break;
            }
        }
    }
    shortlist
}

fn shortlist_order_efficient(a: &SweepCandidateRow, b: &SweepCandidateRow) -> Ordering {
    b.total_pnl
        .total_cmp(&a.total_pnl)
        .then_with(|| b.profit_factor.total_cmp(&a.profit_factor))
        .then_with(|| a.max_drawdown_pct.total_cmp(&b.max_drawdown_pct))
}

fn shortlist_order_growth(a: &SweepCandidateRow, b: &SweepCandidateRow) -> Ordering {
    b.profit_factor
        .total_cmp(&a.profit_factor)
        .then_with(|| b.total_pnl.total_cmp(&a.total_pnl))
        .then_with(|| a.max_drawdown_pct.total_cmp(&b.max_drawdown_pct))
}

fn shortlist_order_conservative(a: &SweepCandidateRow, b: &SweepCandidateRow) -> Ordering {
    let a_positive = a.total_pnl > 0.0;
    let b_positive = b.total_pnl > 0.0;
    b_positive
        .cmp(&a_positive)
        .then_with(|| a.max_drawdown_pct.total_cmp(&b.max_drawdown_pct))
        .then_with(|| b.profit_factor.total_cmp(&a.profit_factor))
        .then_with(|| b.total_pnl.total_cmp(&a.total_pnl))
}

fn validate_candidate(
    paths: &ResolvedPaths,
    validation: &ValidationSettings,
    base_document: &serde_yaml::Value,
    base_cfg: &StrategyConfig,
    run_dir: &Path,
    db_window: &ResolvedDbWindow,
    shortlisted: ShortlistedCandidate,
    initial_balance: f64,
) -> Result<ValidationItem> {
    let candidate_name = format!(
        "candidate_{}_rank{}",
        shortlisted.shortlist_mode.as_str(),
        shortlisted.rank
    );
    let candidate_config_path = run_dir
        .join("configs")
        .join(format!("{candidate_name}.yaml"));
    let generated_document = generate_candidate_config(
        base_document,
        base_cfg,
        &shortlisted.sweep.overrides,
        &candidate_config_path,
        &candidate_name,
        shortlisted.shortlist_mode,
        shortlisted.rank,
        &shortlisted.sweep,
    )?;
    let role = shortlist_mode_target_role(shortlisted.shortlist_mode);
    let candidate_effective_config_path = run_dir
        .join("configs")
        .join(format!("{candidate_name}.{role}.effective.yaml"));
    let materialised = write_role_effective_config(
        &generated_document,
        role,
        &candidate_effective_config_path,
        &candidate_config_path,
        true,
    )?;
    let config_id = materialised.config_id;
    let config_sha256 = file_sha256_hex(&candidate_effective_config_path)?;
    let replay_dir = run_dir.join("replays");
    let replay_path = replay_dir.join(format!("{candidate_name}.replay.json"));
    let replay_report = run_replay(ReplayCommandInput {
        paths,
        config_path: &candidate_effective_config_path,
        initial_balance,
        db_window,
        output_path: &replay_path,
        stdout_path: &run_dir
            .join("replays")
            .join(format!("{candidate_name}.replay.stdout.txt")),
        stderr_path: &run_dir
            .join("replays")
            .join(format!("{candidate_name}.replay.stderr.txt")),
        slippage_bps: None,
    })?;
    let slippage_path = replay_dir.join(format!("{candidate_name}.slippage_20bps.json"));
    let slippage_report = run_replay(ReplayCommandInput {
        paths,
        config_path: &candidate_effective_config_path,
        initial_balance,
        db_window,
        output_path: &slippage_path,
        stdout_path: &run_dir
            .join("replays")
            .join(format!("{candidate_name}.slippage.stdout.txt")),
        stderr_path: &run_dir
            .join("replays")
            .join(format!("{candidate_name}.slippage.stderr.txt")),
        slippage_bps: Some(validation.slippage_bps),
    })?;
    let walk_forward_summary = build_walk_forward_summary(
        paths,
        &candidate_effective_config_path,
        initial_balance,
        db_window,
        validation.walk_forward_splits,
        &run_dir.join("walk_forward").join(&candidate_name),
    )?;
    let walk_forward_path = run_dir
        .join("walk_forward")
        .join(&candidate_name)
        .join("summary.json");
    write_json(&walk_forward_path, &walk_forward_summary)?;
    let top1_pnl_pct = top1_symbol_share(&replay_report);
    let parity =
        compare_cpu_replay_to_gpu_candidate(&replay_report, &shortlisted.sweep, validation);
    let mut blocked_reasons = Vec::new();
    if replay_report.total_trades < validation.min_trades {
        blocked_reasons.push(format!(
            "minimum_trades: {} < {}",
            replay_report.total_trades, validation.min_trades
        ));
    }
    if slippage_report.total_pnl <= 0.0 {
        blocked_reasons.push(format!(
            "slippage_stress: pnl_at_{}bps <= 0",
            validation.slippage_bps
        ));
    }
    if top1_pnl_pct >= validation.max_top1_pnl_pct {
        blocked_reasons.push(format!(
            "concentration: top1_pnl_pct {:.4} >= {:.4}",
            top1_pnl_pct, validation.max_top1_pnl_pct
        ));
    }
    if walk_forward_summary.median_oos_daily_return <= 0.0 {
        blocked_reasons.push(format!(
            "walk_forward: median_oos_daily_return {:.6} <= 0",
            walk_forward_summary.median_oos_daily_return
        ));
    }
    if validation.parity_enforce && parity.status != "pass" {
        blocked_reasons.push("gpu_cpu_parity".to_string());
    }
    let rejected = !blocked_reasons.is_empty();
    let reject_reason = blocked_reasons.join("; ");
    Ok(ValidationItem {
        candidate_mode: true,
        canonical_cpu_verified: true,
        config_id,
        config_path: candidate_effective_config_path.display().to_string(),
        config_sha256,
        final_balance: replay_report.final_balance,
        initial_balance: replay_report.initial_balance,
        max_drawdown_pct: replay_report.max_drawdown_pct,
        pipeline_stage: "candidate_validation".to_string(),
        profit_factor: replay_report.profit_factor,
        rank: shortlisted.rank,
        replay_report_path: replay_path.display().to_string(),
        replay_stage: "cpu_replay".to_string(),
        schema_version: 1,
        shortlist_mode: shortlisted.shortlist_mode.as_str().to_string(),
        slippage_pnl_at_reject_bps: slippage_report.total_pnl,
        slippage_reject_bps: validation.slippage_bps,
        sort_by: shortlisted.shortlist_mode.as_str().to_string(),
        step4_parity: parity,
        sweep_stage: if shortlisted.sweep.candidate_mode {
            "gpu_tpe".to_string()
        } else {
            "sweep".to_string()
        },
        top1_pnl_pct,
        total_fees: replay_report.total_fees,
        total_pnl: replay_report.total_pnl,
        total_trades: replay_report.total_trades,
        validation_gate: "candidate->validated".to_string(),
        walk_forward_summary_path: walk_forward_path.display().to_string(),
        wf_median_oos_daily_return: walk_forward_summary.median_oos_daily_return,
        blocked_reasons: blocked_reasons.clone(),
        rejected,
        reject_reason,
    })
}

fn evaluate_config_performance(
    paths: &ResolvedPaths,
    validation: &ValidationSettings,
    config_path: &Path,
    output_prefix: &Path,
    db_window: &ResolvedDbWindow,
    initial_balance: f64,
    role: Option<&str>,
) -> Result<PerformanceSummary> {
    let raw_document = read_yaml_document(config_path)?;
    let runtime_document =
        materialise_runtime_document(&raw_document, false, role.and_then(role_strategy_mode))
            .map_err(|err| anyhow!(err))
            .with_context(|| format!("materialise config {}", config_path.display()))?;
    let runtime_config_path = output_prefix.with_extension("runtime.yaml");
    let header = if let Some(role) = role {
        format!(
            "# Materialised by aiq-factory at {}\n# Source: {}\n# Role: {}\n",
            Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
            config_path.display(),
            role,
        )
    } else {
        format!(
            "# Materialised by aiq-factory at {}\n# Source: {}\n",
            Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
            config_path.display(),
        )
    };
    write_yaml_document(&runtime_config_path, header, &runtime_document)?;
    let config = load_config_document_checked(&runtime_document, None, false, None)
        .map_err(|err| anyhow!(err))
        .with_context(|| format!("load config {}", config_path.display()))?;
    let config_id = strategy_config_fingerprint_sha256(&config);
    let config_sha256 = file_sha256_hex(&runtime_config_path)?;

    let replay_report_path = output_prefix.with_extension("replay.json");
    let replay = run_replay(ReplayCommandInput {
        paths,
        config_path: &runtime_config_path,
        initial_balance,
        db_window,
        output_path: &replay_report_path,
        stdout_path: &output_prefix.with_extension("replay.stdout.txt"),
        stderr_path: &output_prefix.with_extension("replay.stderr.txt"),
        slippage_bps: None,
    })?;
    let slippage_report = run_replay(ReplayCommandInput {
        paths,
        config_path: &runtime_config_path,
        initial_balance,
        db_window,
        output_path: &output_prefix.with_extension("slippage_20bps.json"),
        stdout_path: &output_prefix.with_extension("slippage.stdout.txt"),
        stderr_path: &output_prefix.with_extension("slippage.stderr.txt"),
        slippage_bps: Some(validation.slippage_bps),
    })?;
    let walk_forward_dir = output_prefix
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(
            "{}_walk_forward",
            output_prefix
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("config")
        ));
    let walk_forward_summary = build_walk_forward_summary(
        paths,
        &runtime_config_path,
        initial_balance,
        db_window,
        validation.walk_forward_splits,
        &walk_forward_dir,
    )?;
    let walk_forward_summary_path = walk_forward_dir.join("summary.json");
    write_json(&walk_forward_summary_path, &walk_forward_summary)?;
    let top1_pnl_pct = top1_symbol_share(&replay);
    let mut blocked_reasons = Vec::new();
    if replay.total_trades < validation.min_trades {
        blocked_reasons.push(format!(
            "minimum_trades: {} < {}",
            replay.total_trades, validation.min_trades
        ));
    }
    if slippage_report.total_pnl <= 0.0 {
        blocked_reasons.push(format!(
            "slippage_stress: pnl_at_{}bps <= 0",
            validation.slippage_bps
        ));
    }
    if top1_pnl_pct >= validation.max_top1_pnl_pct {
        blocked_reasons.push(format!(
            "concentration: top1_pnl_pct {:.4} >= {:.4}",
            top1_pnl_pct, validation.max_top1_pnl_pct
        ));
    }
    if walk_forward_summary.median_oos_daily_return <= 0.0 {
        blocked_reasons.push(format!(
            "walk_forward: median_oos_daily_return {:.6} <= 0",
            walk_forward_summary.median_oos_daily_return
        ));
    }
    let rejected = !blocked_reasons.is_empty();
    let reject_reason = blocked_reasons.join("; ");
    Ok(PerformanceSummary {
        config_id,
        config_path: runtime_config_path.display().to_string(),
        config_sha256,
        final_balance: replay.final_balance,
        initial_balance: replay.initial_balance,
        max_drawdown_pct: replay.max_drawdown_pct,
        profit_factor: replay.profit_factor,
        total_pnl: replay.total_pnl,
        total_trades: replay.total_trades,
        total_fees: replay.total_fees,
        top1_pnl_pct,
        slippage_pnl_at_reject_bps: slippage_report.total_pnl,
        slippage_reject_bps: validation.slippage_bps,
        wf_median_oos_daily_return: walk_forward_summary.median_oos_daily_return,
        replay_report_path: replay_report_path.display().to_string(),
        walk_forward_summary_path: walk_forward_summary_path.display().to_string(),
        rejected,
        reject_reason,
        blocked_reasons,
    })
}

fn build_deployment_challenges(ctx: ChallengeContext<'_>) -> Result<Vec<DeploymentChallenge>> {
    let ChallengeContext {
        paths,
        validation,
        selection,
        comparison,
        db_window,
        selected,
        validated,
        initial_balance,
        run_dir,
    } = ctx;
    let mut reports = Vec::new();
    for role in selected {
        let selected_candidate = validated
            .iter()
            .find(|item| item.config_id == role.config_id)
            .ok_or_else(|| anyhow!("missing selected candidate for role={}", role.role))?;
        let target = selection
            .paper_targets
            .iter()
            .find(|item| item.role == role.role)
            .ok_or_else(|| anyhow!("missing paper target for role={}", role.role))?;
        let challenger_summary = evaluate_config_performance(
            paths,
            validation,
            Path::new(&selected_candidate.config_path),
            &run_dir
                .join("lane_effective")
                .join(format!("challenger_{}", target.role.to_ascii_lowercase())),
            db_window,
            initial_balance,
            Some(target.role.as_str()),
        )?;
        let incumbent = incumbent_performance(
            paths,
            validation,
            resolve_under_project(&paths.project_dir, &target.yaml_path).as_path(),
            db_window,
            initial_balance,
            run_dir,
            target.role.as_str(),
        )?;
        reports.push(compare_challenger_to_incumbent(
            target,
            comparison,
            selected_candidate.config_path.as_str(),
            challenger_summary,
            incumbent,
        ));
    }
    Ok(reports)
}

fn incumbent_performance(
    paths: &ResolvedPaths,
    validation: &ValidationSettings,
    target_yaml_path: &Path,
    db_window: &ResolvedDbWindow,
    initial_balance: f64,
    run_dir: &Path,
    role: &str,
) -> Result<Option<PerformanceSummary>> {
    if !target_yaml_path.is_file() {
        return Ok(None);
    }
    let output_prefix = run_dir
        .join("incumbents")
        .join(format!("incumbent_{}", role.to_ascii_lowercase()));
    let effective_config_path = output_prefix.with_extension("effective.yaml");
    write_role_effective_config_from_path(target_yaml_path, role, &effective_config_path, false)?;
    Ok(Some(evaluate_config_performance(
        paths,
        validation,
        &effective_config_path,
        &output_prefix,
        db_window,
        initial_balance,
        Some(role),
    )?))
}

fn compare_challenger_to_incumbent(
    target: &PaperTarget,
    comparison: &ComparisonSettings,
    source_config_path: &str,
    challenger: PerformanceSummary,
    incumbent: Option<PerformanceSummary>,
) -> DeploymentChallenge {
    let (decision, reason) = match incumbent.as_ref() {
        Some(_) if challenger.rejected => (
            "incumbent_holds".to_string(),
            format!(
                "lane-effective challenger fails the validation suite: {}",
                challenger.reject_reason
            ),
        ),
        None if challenger.rejected => (
            "challenger_rejected".to_string(),
            format!(
                "lane-effective challenger fails the validation suite: {}",
                challenger.reject_reason
            ),
        ),
        None => (
            "no_incumbent".to_string(),
            "no incumbent config deployed on target".to_string(),
        ),
        Some(current) if current.config_id == challenger.config_id => (
            "same_config".to_string(),
            "challenger already matches the deployed target config".to_string(),
        ),
        Some(current) if current.rejected && !challenger.rejected => (
            "challenger_wins".to_string(),
            "incumbent fails the current validation suite while challenger passes".to_string(),
        ),
        Some(current) if !comparison.require_challenger_win => (
            "challenger_wins".to_string(),
            "comparison gate disabled; challenger allowed".to_string(),
        ),
        Some(current) if role_prefers_candidate(target.role.as_str(), &challenger, current) => (
            "challenger_wins".to_string(),
            "challenger outranks current deployed config on the role comparator".to_string(),
        ),
        Some(_) => (
            "incumbent_holds".to_string(),
            "current deployed config remains stronger on the role comparator".to_string(),
        ),
    };
    DeploymentChallenge {
        role: target.role.clone(),
        slot: target.slot,
        service: target.service.clone(),
        target_yaml_path: target.yaml_path.display().to_string(),
        source_config_path: source_config_path.to_string(),
        decision,
        reason,
        incumbent,
        challenger,
    }
}

fn role_prefers_candidate(
    role: &str,
    challenger: &PerformanceSummary,
    incumbent: &PerformanceSummary,
) -> bool {
    match role {
        "conservative" => challenger
            .max_drawdown_pct
            .total_cmp(&incumbent.max_drawdown_pct)
            .then_with(|| incumbent.profit_factor.total_cmp(&challenger.profit_factor))
            .then_with(|| incumbent.total_pnl.total_cmp(&challenger.total_pnl))
            .is_lt(),
        "fallback" => challenger
            .profit_factor
            .total_cmp(&incumbent.profit_factor)
            .then_with(|| challenger.total_pnl.total_cmp(&incumbent.total_pnl))
            .then_with(|| {
                incumbent
                    .max_drawdown_pct
                    .total_cmp(&challenger.max_drawdown_pct)
            })
            .is_gt(),
        _ => challenger
            .total_pnl
            .total_cmp(&incumbent.total_pnl)
            .then_with(|| challenger.profit_factor.total_cmp(&incumbent.profit_factor))
            .then_with(|| {
                incumbent
                    .max_drawdown_pct
                    .total_cmp(&challenger.max_drawdown_pct)
            })
            .is_gt(),
    }
}

fn stable_override_key(overrides: &BTreeMap<String, f64>) -> String {
    overrides
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn generate_candidate_config(
    base_document: &serde_yaml::Value,
    base_cfg: &StrategyConfig,
    overrides: &BTreeMap<String, f64>,
    output_path: &Path,
    candidate_name: &str,
    mode: ShortlistMode,
    rank: usize,
    sweep: &SweepCandidateRow,
) -> Result<serde_yaml::Value> {
    let mut cfg = base_cfg.clone();
    for (path, value) in overrides {
        apply_one_pub(&mut cfg, path, *value);
    }
    let root = replace_global_preserving_root(base_document, &cfg)?;
    let header = format!(
        "# Generated by aiq-factory at {}\n# Candidate: {candidate_name}\n# Source: sweep shortlist rank #{rank} by {}\n# Sweep candidate: PnL {:.2} | {} trades | PF {:.2} | DD {:.2}%\n# Overrides applied: {}\n",
        Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        mode.as_str(),
        sweep.total_pnl,
        sweep.total_trades,
        sweep.profit_factor,
        sweep.max_drawdown_pct * 100.0,
        overrides.len(),
    );
    write_yaml_document(output_path, header, &root)?;
    Ok(root)
}

fn run_replay(input: ReplayCommandInput<'_>) -> Result<ReplaySummary> {
    let ReplayCommandInput {
        paths,
        config_path,
        initial_balance,
        db_window,
        output_path,
        stdout_path,
        stderr_path,
        slippage_bps,
    } = input;
    let mut cmd = Command::new(&paths.backtester_bin);
    cmd.arg("replay")
        .arg("--config")
        .arg(config_path)
        .arg("--funding-db")
        .arg(&paths.funding_db_path)
        .arg("--candles-db")
        .arg(&db_window.main_db)
        .arg("--interval")
        .arg(&db_window.main_interval)
        .arg("--entry-candles-db")
        .arg(&db_window.entry_db)
        .arg("--entry-interval")
        .arg(&db_window.entry_interval)
        .arg("--exit-candles-db")
        .arg(&db_window.exit_db)
        .arg("--exit-interval")
        .arg(&db_window.exit_interval)
        .arg("--initial-balance")
        .arg(initial_balance.to_string())
        .arg("--start-ts")
        .arg(db_window.start_ts_ms.to_string())
        .arg("--end-ts")
        .arg(db_window.end_ts_ms.to_string())
        .arg("--no-auto-scope")
        .arg("--output")
        .arg(output_path);
    if let Some(slippage) = slippage_bps {
        cmd.arg("--slippage-bps").arg(slippage.to_string());
    }
    run_command(&mut cmd, &paths.project_dir, stdout_path, stderr_path)
        .context("run CPU replay")?;
    let report: ReplaySummary = serde_json::from_str(
        &fs::read_to_string(output_path)
            .with_context(|| format!("read replay report {}", output_path.display()))?,
    )
    .with_context(|| format!("parse replay report {}", output_path.display()))?;
    Ok(report)
}

fn build_walk_forward_summary(
    paths: &ResolvedPaths,
    config_path: &Path,
    initial_balance: f64,
    db_window: &ResolvedDbWindow,
    split_count: usize,
    output_dir: &Path,
) -> Result<WalkForwardSummary> {
    fs::create_dir_all(output_dir).with_context(|| format!("create {}", output_dir.display()))?;
    let splits = split_time_range(
        (db_window.start_ts_ms, db_window.end_ts_ms),
        split_count.max(1),
    );
    let mut outputs = Vec::new();
    for (idx, (start_ts_ms, end_ts_ms)) in splits.into_iter().enumerate() {
        let replay_path = output_dir.join(format!("split{}.replay.json", idx + 1));
        let report = run_replay(ReplayCommandInput {
            paths,
            config_path,
            initial_balance,
            db_window: &ResolvedDbWindow {
                start_ts_ms,
                end_ts_ms,
                ..db_window.clone()
            },
            output_path: &replay_path,
            stdout_path: &output_dir.join(format!("split{}.stdout.txt", idx + 1)),
            stderr_path: &output_dir.join(format!("split{}.stderr.txt", idx + 1)),
            slippage_bps: None,
        })?;
        let days = ((end_ts_ms - start_ts_ms) as f64 / 86_400_000.0).max(1.0);
        let daily_return = if report.initial_balance.abs() < f64::EPSILON {
            0.0
        } else {
            ((report.final_balance / report.initial_balance) - 1.0) / days
        };
        outputs.push(SplitReturn {
            split: idx + 1,
            start_ts_ms,
            end_ts_ms,
            days,
            final_balance: report.final_balance,
            total_pnl: report.total_pnl,
            daily_return,
        });
    }
    let median = median(outputs.iter().map(|item| item.daily_return).collect());
    Ok(WalkForwardSummary {
        split_count: outputs.len(),
        median_oos_daily_return: median,
        splits: outputs,
    })
}

fn split_time_range(time_range: (i64, i64), split_count: usize) -> Vec<(i64, i64)> {
    let total = time_range.1 - time_range.0;
    let step = (total / split_count as i64).max(1);
    let mut out = Vec::new();
    let mut start = time_range.0;
    for idx in 0..split_count {
        let end = if idx + 1 == split_count {
            time_range.1
        } else {
            start + step
        };
        out.push((start, end));
        start = end;
    }
    out
}

fn top1_symbol_share(report: &ReplaySummary) -> f64 {
    if report.total_pnl <= 0.0 {
        return 1.0;
    }
    let max_symbol = report
        .by_symbol
        .iter()
        .map(|item| item.pnl.max(0.0))
        .fold(0.0_f64, f64::max);
    max_symbol / report.total_pnl
}

fn compare_cpu_replay_to_gpu_candidate(
    cpu: &ReplaySummary,
    gpu: &SweepCandidateRow,
    validation: &ValidationSettings,
) -> ParityCheck {
    let gpu_final_balance = cpu.initial_balance + gpu.total_pnl;
    let balance_delta_abs = (cpu.final_balance - gpu_final_balance).abs();
    let balance_delta_rel = if cpu.final_balance.abs() < f64::EPSILON {
        0.0
    } else {
        balance_delta_abs / cpu.final_balance.abs()
    };
    let trade_delta = cpu.total_trades.abs_diff(gpu.total_trades);
    let pass = trade_delta <= validation.parity_trade_delta_max
        && (balance_delta_abs <= validation.parity_abs_eps
            || balance_delta_rel <= validation.parity_rel_eps);
    ParityCheck {
        status: if pass { "pass" } else { "warn" }.to_string(),
        cpu_total_trades: cpu.total_trades,
        gpu_total_trades: gpu.total_trades,
        trade_delta,
        cpu_final_balance: cpu.final_balance,
        gpu_final_balance,
        balance_delta_abs,
        balance_delta_rel,
        thresholds: ParityThresholds {
            abs_eps: validation.parity_abs_eps,
            rel_eps: validation.parity_rel_eps,
            trade_delta_max: validation.parity_trade_delta_max,
        },
    }
}

fn select_roles(
    validated: &[ValidationItem],
    selection: &SelectionSettings,
) -> Vec<SelectionCandidate> {
    let deployable = validated
        .iter()
        .filter(|item| !item.rejected)
        .cloned()
        .collect::<Vec<_>>();
    if deployable.is_empty() {
        return Vec::new();
    }
    let active_targets = active_paper_targets(selection);
    if active_targets.is_empty() {
        return Vec::new();
    }
    let by_mode = deployable.iter().fold(
        HashMap::<String, Vec<ValidationItem>>::new(),
        |mut acc, item| {
            acc.entry(item.sort_by.clone())
                .or_default()
                .push(item.clone());
            acc
        },
    );
    let mut used = BTreeSet::new();
    let mut selected = Vec::new();
    for target in active_targets {
        let preferred_mode = match target.role.as_str() {
            "primary" => "efficient",
            "fallback" => "growth",
            "conservative" => "conservative",
            _ => "efficient",
        };
        let choice = by_mode
            .get(preferred_mode)
            .and_then(|items| first_unused(items, &used))
            .or_else(|| first_unused(&deployable, &used));
        let Some(choice) = choice else {
            break;
        };
        used.insert(choice.config_id.clone());
        selected.push(SelectionCandidate {
            role: target.role.clone(),
            slot: target.slot,
            service: target.service.clone(),
            source: "promotion_roles".to_string(),
            selected: true,
            config_id: choice.config_id,
        });
    }
    selected
}

fn first_unused(items: &[ValidationItem], used: &BTreeSet<String>) -> Option<ValidationItem> {
    let mut sorted = items.to_vec();
    sorted.sort_by(role_candidate_order);
    sorted
        .into_iter()
        .find(|item| !used.contains(&item.config_id))
}

fn role_candidate_order(a: &ValidationItem, b: &ValidationItem) -> Ordering {
    match a.sort_by.as_str() {
        "conservative" => a
            .max_drawdown_pct
            .total_cmp(&b.max_drawdown_pct)
            .then_with(|| b.profit_factor.total_cmp(&a.profit_factor))
            .then_with(|| b.total_pnl.total_cmp(&a.total_pnl)),
        "growth" => b
            .profit_factor
            .total_cmp(&a.profit_factor)
            .then_with(|| b.total_pnl.total_cmp(&a.total_pnl))
            .then_with(|| a.max_drawdown_pct.total_cmp(&b.max_drawdown_pct)),
        _ => b
            .total_pnl
            .total_cmp(&a.total_pnl)
            .then_with(|| b.profit_factor.total_cmp(&a.profit_factor))
            .then_with(|| a.max_drawdown_pct.total_cmp(&b.max_drawdown_pct)),
    }
}

fn build_gate_report(
    run_id: &str,
    validated: &[ValidationItem],
    selected: &[SelectionCandidate],
    selection: &SelectionSettings,
) -> GateReport {
    let active_targets = active_paper_targets(selection);
    let deployable_count = validated.iter().filter(|item| !item.rejected).count();
    let blocked = deployable_count < active_targets.len();
    let blocked_reason = if blocked {
        if deployable_count == 0 {
            "no deployable candidates passed factory validation".to_string()
        } else {
            format!(
                "only {} deployable candidates for {} paper slots",
                deployable_count,
                active_targets.len()
            )
        }
    } else {
        String::new()
    };
    let slot_bindings = active_targets
        .iter()
        .map(|target| {
            let chosen = selected.iter().find(|item| item.role == target.role);
            SlotBinding {
                role: target.role.clone(),
                slot: target.slot,
                service: target.service.clone(),
                yaml_path: target.yaml_path.display().to_string(),
                selected: chosen.is_some(),
                config_id: chosen.map(|item| item.config_id.clone()),
            }
        })
        .collect::<Vec<_>>();
    GateReport {
        schema_version: 1,
        run_id: run_id.to_string(),
        generated_at_ms: Utc::now().timestamp_millis(),
        selection_policy: DEFAULT_SELECTION_POLICY,
        require_ssot_evidence: true,
        candidate_count: validated.len(),
        deployable_count,
        selected_count: selected.len(),
        blocked,
        blocked_reason: blocked_reason.clone(),
        blocked_reasons_count: usize::from(blocked),
        selection_warnings: Vec::new(),
        candidates: validated.to_vec(),
        slot_bindings,
    }
}

fn build_selection_report(input: SelectionReportInput<'_>) -> Result<SelectionReport> {
    let SelectionReportInput {
        run_id,
        run_dir,
        effective_config_id,
        effective_config_path,
        validated,
        selected,
        selection,
        base_cfg,
        blocked,
        blocked_reason,
        challenges,
        deployments,
        paper_promotion_gate,
        live_promotion,
        promotion_requested,
        selection_warnings,
    } = input;
    let selected_items = selected
        .iter()
        .filter_map(|role| {
            validated
                .iter()
                .find(|item| item.config_id == role.config_id)
        })
        .cloned()
        .collect::<Vec<_>>();
    let primary = selected
        .iter()
        .find(|item| item.role == "primary")
        .and_then(|role| {
            validated
                .iter()
                .find(|item| item.config_id == role.config_id)
        })
        .cloned()
        .or_else(|| selected_items.first().cloned())
        .or_else(|| best_overall_candidate(validated))
        .ok_or_else(|| anyhow!("missing selected primary candidate"))?;
    let active_targets = active_paper_targets(selection);
    let applied_count = deployments
        .iter()
        .filter(|item| item.status == "applied")
        .count();
    let same_config_count = deployments
        .iter()
        .filter(|item| item.status == "already_applied")
        .count();
    let incumbent_hold_count = deployments
        .iter()
        .filter(|item| item.status == "incumbent_holds")
        .count();
    let deployment_failed = selection_warnings
        .iter()
        .any(|warning| warning.starts_with("paper_deploy_failed:"));
    let promotion_failed = selection_warnings.iter().any(|warning| {
        warning.starts_with("live_promotion_failed:")
            || warning.starts_with("live_promotion_blocked:")
    });
    let deploy_stage = if blocked {
        "blocked"
    } else if deployment_failed {
        "failed"
    } else if applied_count > 0 {
        "paper_applied"
    } else if same_config_count == active_targets.len() && !active_targets.is_empty() {
        "paper_steady"
    } else if incumbent_hold_count > 0 {
        "challenger_blocked"
    } else if deployments.is_empty() {
        "pending"
    } else {
        "paper_ready"
    };
    Ok(SelectionReport {
        version: "factory_cycle_selection_v2",
        run_id: run_id.to_string(),
        run_dir: run_dir.display().to_string(),
        interval: base_cfg.engine.interval.clone(),
        deploy_stage: deploy_stage.to_string(),
        deployed: !blocked && !deployment_failed && applied_count > 0,
        deployments,
        challenges,
        promotion_stage: if blocked || deployment_failed {
            "blocked"
        } else if !promotion_requested {
            "paper_applied"
        } else if promotion_failed {
            "failed"
        } else if live_promotion.is_some() {
            "live_applied"
        } else {
            "pending"
        }
        .to_string(),
        selection_policy: DEFAULT_SELECTION_POLICY,
        selection_stage: if blocked || deployment_failed || promotion_failed {
            "blocked"
        } else {
            "selected"
        }
        .to_string(),
        selection_warnings,
        step5_gate_status: if blocked || deployment_failed {
            "blocked"
        } else {
            "passed"
        }
        .to_string(),
        step5_gate_block_reason: blocked_reason,
        paper_promotion_gate,
        live_promotion,
        effective_config_id: effective_config_id.to_string(),
        effective_config_path: effective_config_path.display().to_string(),
        promotion_reference_epoch_s: Utc::now().timestamp_millis() as f64 / 1000.0,
        evidence_bundle_paths: SelectionEvidencePaths {
            run_dir: run_dir.display().to_string(),
            run_metadata_json: run_dir.join("run_metadata.json").display().to_string(),
            report_json: run_dir.join("reports/report.json").display().to_string(),
            report_md: run_dir.join("reports/report.md").display().to_string(),
            selection_json: run_dir.join("reports/selection.json").display().to_string(),
            selection_md: run_dir.join("reports/selection.md").display().to_string(),
            step5_gate_report_json: run_dir
                .join("reports/step5_gate_report.json")
                .display()
                .to_string(),
            step5_gate_report_md: run_dir
                .join("reports/step5_gate_report.md")
                .display()
                .to_string(),
            configs_dir: run_dir.join("configs").display().to_string(),
            replays_dir: run_dir.join("replays").display().to_string(),
        },
        selected: primary,
        selected_candidates: selected_items,
        selected_candidates_by_role: selected.to_vec(),
        selected_targets: active_targets
            .iter()
            .map(|target| PaperTargetSummary {
                service: target.service.clone(),
                slot: target.slot,
                yaml_path: target.yaml_path.display().to_string(),
            })
            .collect(),
    })
}

fn active_paper_targets(selection: &SelectionSettings) -> Vec<&PaperTarget> {
    let enabled = selection
        .selected_roles
        .iter()
        .map(|role| role.to_ascii_lowercase())
        .collect::<HashSet<_>>();
    if enabled.is_empty() {
        return selection.paper_targets.iter().collect();
    }
    selection
        .paper_targets
        .iter()
        .filter(|target| enabled.contains(&target.role.to_ascii_lowercase()))
        .collect()
}

fn validate_selection_settings(selection: &SelectionSettings) -> Result<()> {
    if selection.selected_roles.is_empty() {
        return Ok(());
    }
    let known = selection
        .paper_targets
        .iter()
        .map(|target| target.role.to_ascii_lowercase())
        .collect::<HashSet<_>>();
    let unknown = selection
        .selected_roles
        .iter()
        .map(|role| role.to_ascii_lowercase())
        .filter(|role| !known.contains(role))
        .collect::<BTreeSet<_>>();
    if !unknown.is_empty() {
        bail!(
            "unknown selection.selected_roles entries: {}",
            unknown.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
    Ok(())
}

fn validate_factory_defaults(settings: &FactoryDefaults) -> Result<()> {
    if settings.deployment.apply_to_live && !settings.deployment.apply_to_paper {
        bail!("deployment.apply_to_live requires deployment.apply_to_paper=true");
    }
    if settings.deployment.apply_to_live {
        let enabled_roles = settings
            .selection
            .selected_roles
            .iter()
            .map(|role| role.to_ascii_lowercase())
            .collect::<HashSet<_>>();
        if !enabled_roles.is_empty() && !enabled_roles.contains("primary") {
            bail!(
                "deployment.apply_to_live requires `primary` to be present in selection.selected_roles"
            );
        }
    }
    Ok(())
}

fn assess_primary_paper_promotion_gate(
    project_dir: &Path,
    selection: &SelectionSettings,
    validated: &[ValidationItem],
    selected: &[SelectionCandidate],
) -> Result<PaperPromotionGate> {
    let primary_role = selected
        .iter()
        .find(|role| role.role.eq_ignore_ascii_case("primary"))
        .ok_or_else(|| anyhow!("live promotion requested but no primary role was selected"))?;
    let primary_candidate = validated
        .iter()
        .find(|item| item.config_id == primary_role.config_id)
        .ok_or_else(|| anyhow!("missing primary validation item {}", primary_role.config_id))?;
    let target = selection
        .paper_targets
        .iter()
        .find(|target| target.role.eq_ignore_ascii_case("primary"))
        .ok_or_else(|| anyhow!("missing paper target for primary role"))?;
    let expected_config_id = current_target_config_id(
        Path::new(&primary_candidate.config_path),
        target.role.as_str(),
    )?
    .ok_or_else(|| anyhow!("missing selected primary candidate config payload"))?;
    let paper_db_path = paper_db_path_for_slot(project_dir, target.slot)?;
    let Some(marker) = load_paper_soak_marker(project_dir, target.role.as_str())? else {
        return Ok(blocked_paper_promotion_gate(
            &paper_db_path,
            primary_candidate,
            vec![format!(
                "missing current paper soak marker for role `{}`; deploy to paper before live promotion",
                target.role
            )],
        ));
    };
    let target_yaml_path = resolve_under_project(project_dir, &target.yaml_path);
    if marker.slot != target.slot
        || marker.service != target.service
        || Path::new(&marker.target_yaml_path) != target_yaml_path.as_path()
    {
        return Ok(blocked_paper_promotion_gate(
            &paper_db_path,
            primary_candidate,
            vec![format!(
                "paper soak marker for role `{}` no longer matches slot/service/yaml target",
                target.role
            )],
        ));
    }
    let current_target_config_id =
        current_target_config_id(target_yaml_path.as_path(), target.role.as_str())?;
    if current_target_config_id.as_deref() != Some(marker.config_id.as_str()) {
        return Ok(blocked_paper_promotion_gate(
            &paper_db_path,
            primary_candidate,
            vec![format!(
                "paper soak marker for role `{}` no longer matches the deployed target fingerprint",
                target.role
            )],
        ));
    }
    if marker.config_id != expected_config_id {
        return Ok(blocked_paper_promotion_gate(
            &paper_db_path,
            primary_candidate,
            vec![format!(
                "paper soak marker for role `{}` points to config {} rather than selected {}",
                target.role, marker.config_id, expected_config_id
            )],
        ));
    }
    assess_paper_promotion_gate(
        &paper_db_path,
        primary_candidate,
        Some(marker.deployed_at_ms),
    )
}

fn assess_paper_promotion_gate(
    paper_db_path: &Path,
    candidate: &ValidationItem,
    deployed_after_ms: Option<i64>,
) -> Result<PaperPromotionGate> {
    let conn = Connection::open(paper_db_path)
        .with_context(|| format!("open paper DB {}", paper_db_path.display()))?;
    let config_fingerprint = candidate.config_id.clone();

    let mut stmt = conn.prepare(
        "SELECT DISTINCT t.id, t.timestamp, t.action, COALESCE(t.pnl, 0.0), COALESCE(t.fee_usd, 0.0), COALESCE(t.balance, 0.0), COALESCE(d.run_fingerprint, '')
         FROM trades t
         JOIN decision_events d ON d.trade_id = t.id
         WHERE d.event_type = 'fill' AND d.status = 'executed' AND d.config_fingerprint = ?
         ORDER BY t.id ASC",
    )?;
    let mut rows = stmt.query([config_fingerprint.as_str()])?;
    let mut first_trade_ts: Option<String> = None;
    let mut last_trade_ts: Option<String> = None;
    let mut first_trade_ms: Option<i64> = None;
    let mut last_trade_ms: Option<i64> = None;
    let mut close_trades = 0u32;
    let mut gross_profit = 0.0f64;
    let mut gross_loss = 0.0f64;
    let mut peak_balance: Option<f64> = None;
    let mut max_drawdown_pct = 0.0f64;
    let mut evidence_runs = HashSet::new();
    while let Some(row) = rows.next()? {
        let timestamp: String = row.get(1)?;
        let action: String = row.get(2)?;
        let pnl: f64 = row.get(3)?;
        let fee_usd: f64 = row.get(4)?;
        let balance: f64 = row.get(5)?;
        let run_fingerprint: String = row.get(6)?;
        let ts_ms = parse_trade_timestamp_ms(&timestamp);
        if let Some(anchor_ms) = deployed_after_ms {
            let Some(trade_ms) = ts_ms else {
                continue;
            };
            if trade_ms < anchor_ms {
                continue;
            }
        }
        if !run_fingerprint.is_empty() {
            evidence_runs.insert(run_fingerprint);
        }
        if first_trade_ts.is_none() {
            first_trade_ts = Some(timestamp.clone());
            first_trade_ms = ts_ms;
        }
        last_trade_ts = Some(timestamp);
        last_trade_ms = ts_ms;

        if is_close_action(&action) {
            close_trades += 1;
            let net_pnl = pnl - fee_usd;
            if net_pnl > 0.0 {
                gross_profit += net_pnl;
            } else {
                gross_loss += net_pnl.abs();
            }
        }

        match peak_balance {
            Some(peak) if peak > 0.0 => {
                let next_peak = peak.max(balance);
                let drawdown_pct = ((next_peak - balance).max(0.0) / next_peak) * 100.0;
                if drawdown_pct > max_drawdown_pct {
                    max_drawdown_pct = drawdown_pct;
                }
                peak_balance = Some(next_peak);
            }
            _ => {
                peak_balance = Some(balance);
            }
        }
    }

    let runtime_hours = match (first_trade_ms, last_trade_ms) {
        (Some(first), Some(last)) if last >= first => (last - first) as f64 / 3_600_000.0,
        _ => 0.0,
    };
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };
    let kill_switch_events = count_kill_switch_events(&conn, &evidence_runs)?;
    let thresholds = PaperPromotionThresholds {
        min_runtime_hours: 24.0,
        min_close_trades: 20,
        min_profit_factor: 1.2,
        max_drawdown_pct: 10.0,
        require_positive_slippage_pnl: true,
        require_zero_kill_switch_events: true,
    };
    let mut blocked_reasons = Vec::new();
    if close_trades == 0 {
        blocked_reasons.push(format!(
            "no paper fill evidence for config_fingerprint {} in {}",
            config_fingerprint,
            paper_db_path.display()
        ));
    }
    if runtime_hours < thresholds.min_runtime_hours && close_trades < thresholds.min_close_trades {
        blocked_reasons.push(format!(
            "paper runtime below soak gate ({runtime_hours:.2}h, {close_trades} close trades)"
        ));
    }
    if profit_factor < thresholds.min_profit_factor {
        blocked_reasons.push(format!(
            "paper profit factor {:.4} below {:.2}",
            profit_factor, thresholds.min_profit_factor
        ));
    }
    if max_drawdown_pct >= thresholds.max_drawdown_pct {
        blocked_reasons.push(format!(
            "paper max drawdown {:.4}% >= {:.2}%",
            max_drawdown_pct, thresholds.max_drawdown_pct
        ));
    }
    if candidate.slippage_pnl_at_reject_bps <= 0.0 {
        blocked_reasons.push(format!(
            "validation slippage stress pnl_at_{}bps <= 0",
            candidate.slippage_reject_bps
        ));
    }
    if kill_switch_events > 0 {
        blocked_reasons.push(format!(
            "paper kill-switch evidence present ({kill_switch_events} events)"
        ));
    }

    Ok(PaperPromotionGate {
        status: if blocked_reasons.is_empty() {
            "pass".to_string()
        } else {
            "blocked".to_string()
        },
        role: "primary".to_string(),
        paper_db_path: paper_db_path.display().to_string(),
        config_fingerprint,
        first_trade_ts,
        last_trade_ts,
        runtime_hours,
        close_trades,
        profit_factor,
        max_drawdown_pct,
        slippage_pnl_at_reject_bps: candidate.slippage_pnl_at_reject_bps,
        kill_switch_events,
        thresholds,
        blocked_reasons,
    })
}

fn blocked_paper_promotion_gate(
    paper_db_path: &Path,
    candidate: &ValidationItem,
    blocked_reasons: Vec<String>,
) -> PaperPromotionGate {
    PaperPromotionGate {
        status: "blocked".to_string(),
        role: "primary".to_string(),
        paper_db_path: paper_db_path.display().to_string(),
        config_fingerprint: candidate.config_id.clone(),
        first_trade_ts: None,
        last_trade_ts: None,
        runtime_hours: 0.0,
        close_trades: 0,
        profit_factor: 0.0,
        max_drawdown_pct: 0.0,
        slippage_pnl_at_reject_bps: candidate.slippage_pnl_at_reject_bps,
        kill_switch_events: 0,
        thresholds: PaperPromotionThresholds {
            min_runtime_hours: 24.0,
            min_close_trades: 20,
            min_profit_factor: 1.2,
            max_drawdown_pct: 10.0,
            require_positive_slippage_pnl: true,
            require_zero_kill_switch_events: true,
        },
        blocked_reasons,
    }
}

fn count_kill_switch_events(conn: &Connection, evidence_runs: &HashSet<String>) -> Result<u32> {
    if evidence_runs.is_empty() {
        return Ok(0);
    }

    let mut audit_stmt = conn.prepare(
        "SELECT COALESCE(event, ''), COALESCE(run_fingerprint, '')
         FROM audit_events",
    )?;
    let mut audit_rows = audit_stmt.query([])?;
    let mut count = 0u32;
    while let Some(row) = audit_rows.next()? {
        let event: String = row.get(0)?;
        let run_fingerprint: String = row.get(1)?;
        if !evidence_runs.contains(&run_fingerprint) {
            continue;
        }
        if event.starts_with("RISK_KILL_") {
            count += 1;
        }
    }
    Ok(count)
}

fn paper_db_path_for_slot(project_dir: &Path, slot: usize) -> Result<PathBuf> {
    match slot {
        1 => Ok(project_dir.join(DEFAULT_PRIMARY_PAPER_DB_PATH)),
        2 => Ok(project_dir.join("trading_engine_v8_paper2.db")),
        3 => Ok(project_dir.join("trading_engine_v8_paper3.db")),
        _ => bail!("unsupported paper target slot for live promotion gate: {slot}"),
    }
}

fn write_paper_soak_marker(
    project_dir: &Path,
    run_id: &str,
    plan: &PreparedDeployment,
) -> Result<PathBuf> {
    let marker_path = paper_soak_marker_path(project_dir, plan.role.as_str());
    let marker = PaperSoakMarker {
        version: "factory_paper_soak_v1".to_string(),
        role: plan.role.clone(),
        slot: plan.slot,
        service: plan.service.clone(),
        paper_db_path: paper_db_path_for_slot(project_dir, plan.slot)?
            .display()
            .to_string(),
        config_id: plan.config_id.clone(),
        config_sha256: plan.config_sha256.clone(),
        deployed_at_ms: Utc::now().timestamp_millis(),
        deployed_by_run_id: run_id.to_string(),
        source_config_path: plan.source_config_path.display().to_string(),
        target_yaml_path: plan.target_yaml_path.display().to_string(),
    };
    write_json(&marker_path, &marker)?;
    Ok(marker_path)
}

fn load_paper_soak_marker(project_dir: &Path, role: &str) -> Result<Option<PaperSoakMarker>> {
    let marker_path = paper_soak_marker_path(project_dir, role);
    if !marker_path.is_file() {
        return Ok(None);
    }
    let bytes =
        fs::read(&marker_path).with_context(|| format!("read {}", marker_path.display()))?;
    let marker = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {}", marker_path.display()))?;
    Ok(Some(marker))
}

fn paper_soak_marker_path(project_dir: &Path, role: &str) -> PathBuf {
    project_dir.join("artifacts").join("state").join(format!(
        "factory_paper_soak_{}.json",
        role.to_ascii_lowercase()
    ))
}

fn current_target_config_id(path: &Path, role: &str) -> Result<Option<String>> {
    if !path.is_file() {
        return Ok(None);
    }
    let materialised = materialise_role_effective_config(&read_yaml_document(path)?, role)?;
    Ok(Some(materialised.config_id))
}

fn parse_trade_timestamp_ms(raw: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|dt| dt.timestamp_millis())
}

fn is_close_action(action: &str) -> bool {
    matches!(
        action,
        "REDUCE" | "CLOSE" | "REDUCE_LONG" | "REDUCE_SHORT" | "CLOSE_LONG" | "CLOSE_SHORT"
    )
}

fn promote_primary_live(
    project_dir: &Path,
    artifacts_root: &Path,
    run_dir: &Path,
    validated: &[ValidationItem],
    selected: &[SelectionCandidate],
    paper_deployments: &[DeploymentEvent],
    deployment: &DeploymentSettings,
) -> Result<LivePromotionEvent> {
    let primary_role = selected
        .iter()
        .find(|role| role.role.eq_ignore_ascii_case("primary"))
        .ok_or_else(|| anyhow!("live promotion requested but no primary role was selected"))?;
    let primary_candidate = validated
        .iter()
        .find(|item| item.config_id == primary_role.config_id)
        .ok_or_else(|| anyhow!("missing primary validation item {}", primary_role.config_id))?;
    let primary_paper = paper_deployments
        .iter()
        .find(|event| event.role.eq_ignore_ascii_case("primary"))
        .ok_or_else(|| anyhow!("missing primary paper deployment artefact"))?;

    let source_config_path = PathBuf::from(&primary_candidate.config_path);
    let promoted_config_path = if primary_paper.promoted_config_path.trim().is_empty() {
        PathBuf::from(&primary_paper.source_config_path)
    } else {
        PathBuf::from(&primary_paper.promoted_config_path)
    };
    let promoted_text = fs::read_to_string(&promoted_config_path).with_context(|| {
        format!(
            "read promoted live candidate {}",
            promoted_config_path.display()
        )
    })?;
    serde_yaml::from_str::<serde_yaml::Value>(&promoted_text).with_context(|| {
        format!(
            "parse promoted live candidate YAML {}",
            promoted_config_path.display()
        )
    })?;

    let live_yaml_path = resolve_under_project(project_dir, &deployment.live_yaml_path);
    let current_live_text = if live_yaml_path.is_file() {
        let text = fs::read_to_string(&live_yaml_path)
            .with_context(|| format!("read current live YAML {}", live_yaml_path.display()))?;
        serde_yaml::from_str::<serde_yaml::Value>(&text)
            .with_context(|| format!("parse current live YAML {}", live_yaml_path.display()))?;
        Some(text)
    } else {
        None
    };
    let current_interval = current_live_text
        .as_deref()
        .map(load_yaml_engine_interval)
        .unwrap_or_default();
    let promoted_interval = load_yaml_engine_interval(&promoted_text);
    let restart_required = current_interval != promoted_interval
        && !current_interval.is_empty()
        && !promoted_interval.is_empty();
    if restart_required && !deployment.restart_live_service {
        bail!(
            "live promotion changes engine.interval ({} -> {}) but restart_live_service=false",
            current_interval,
            promoted_interval
        );
    }
    let live_service_was_active = if deployment.restart_live_service {
        user_service_is_active(deployment.live_service.as_str())?
    } else {
        false
    };

    let deploy_root = artifacts_root.join("deployments").join("live");
    fs::create_dir_all(&deploy_root)
        .with_context(|| format!("create {}", deploy_root.display()))?;
    let deploy_stamp = unique_utc_stamp(Utc::now());
    let deploy_dir = deploy_root.join(format!("manual_switch_{}_primary", deploy_stamp));
    fs::create_dir_all(&deploy_dir).with_context(|| format!("create {}", deploy_dir.display()))?;

    let promoted_payload = format!("{}\n", promoted_text.trim_end());
    fs::write(deploy_dir.join("promoted_config.yaml"), &promoted_payload).with_context(|| {
        format!(
            "write {}",
            deploy_dir.join("promoted_config.yaml").display()
        )
    })?;
    fs::write(
        deploy_dir.join("prev_config.yaml"),
        current_live_text.clone().unwrap_or_default(),
    )
    .with_context(|| format!("write {}", deploy_dir.join("prev_config.yaml").display()))?;

    let live_snapshot = snapshot_target(&live_yaml_path)?;
    let promotion_result: Result<LivePromotionEvent> = (|| {
        atomic_write_text(&live_yaml_path, &promoted_payload)?;
        let restarted_service = if deployment.restart_live_service {
            restart_user_service(deployment.live_service.as_str())?
        } else {
            false
        };
        let event = serde_json::json!({
            "version": "factory_live_promotion_v1",
            "source": "aiq-factory",
            "ts_utc": iso_utc_now(),
            "run_dir": run_dir.display().to_string(),
            "selection_policy": DEFAULT_SELECTION_POLICY,
            "role": primary_role.role,
            "config_id": primary_role.config_id,
            "source_config_path": source_config_path.display().to_string(),
            "promoted_config_path": promoted_config_path.display().to_string(),
            "live_yaml_path": live_yaml_path.display().to_string(),
            "live_service": deployment.live_service,
            "service_was_active_before": live_service_was_active,
            "restart_required": restart_required,
            "restarted_service": restarted_service,
        });
        write_json(&deploy_dir.join("promotion_event.json"), &event)?;
        Ok(LivePromotionEvent {
            role: primary_role.role.clone(),
            config_id: primary_role.config_id.clone(),
            source_config_path: source_config_path.display().to_string(),
            promoted_config_path: promoted_config_path.display().to_string(),
            live_yaml_path: live_yaml_path.display().to_string(),
            live_service: deployment.live_service.clone(),
            deployment_dir: deploy_dir.display().to_string(),
            service_was_active_before: live_service_was_active,
            restarted_service,
            restart_required,
            status: "applied".to_string(),
        })
    })();

    match promotion_result {
        Ok(event) => Ok(event),
        Err(err) => {
            rollback_target_snapshots(&[live_snapshot])?;
            if should_restart_live_service_after_rollback(
                deployment.restart_live_service,
                live_service_was_active,
            ) {
                restart_user_service(deployment.live_service.as_str())?;
            }
            let _ = fs::remove_dir_all(&deploy_dir);
            bail!("live promotion rolled back after failure: {err}");
        }
    }
}

fn unique_utc_stamp(now: chrono::DateTime<Utc>) -> String {
    format!(
        "{}_{:03}",
        now.format("%Y%m%dT%H%M%SZ"),
        now.timestamp_subsec_millis()
    )
}

fn write_reports(
    run_dir: &Path,
    validated: &[ValidationItem],
    gate_report: &GateReport,
    selection_report: &SelectionReport,
) -> Result<()> {
    let report_json = serde_json::json!({ "items": validated });
    write_json(&run_dir.join("reports/report.json"), &report_json)?;
    write_json(&run_dir.join("reports/step5_gate_report.json"), gate_report)?;
    write_json(&run_dir.join("reports/selection.json"), selection_report)?;

    let report_md = build_report_markdown(validated);
    fs::write(run_dir.join("reports/report.md"), report_md)
        .context("write factory report markdown")?;
    fs::write(
        run_dir.join("reports/step5_gate_report.md"),
        build_gate_markdown(gate_report),
    )
    .context("write gate markdown")?;
    fs::write(
        run_dir.join("reports/selection.md"),
        build_selection_markdown(selection_report),
    )
    .context("write selection markdown")?;
    Ok(())
}

fn deploy_selected_configs(
    project_dir: &Path,
    run_id: &str,
    run_dir: &Path,
    challenges: &[DeploymentChallenge],
    deployment: &DeploymentSettings,
) -> Result<Vec<DeploymentEvent>> {
    let mut plans = Vec::new();
    let mut passthrough_events = Vec::new();
    for challenge in challenges {
        let config_path = PathBuf::from(&challenge.source_config_path);
        let target_yaml_path =
            resolve_under_project(project_dir, Path::new(&challenge.target_yaml_path));
        let existing_marker = load_paper_soak_marker(project_dir, challenge.role.as_str())?;
        if matches!(
            challenge.decision.as_str(),
            "same_config" | "incumbent_holds" | "challenger_rejected"
        ) {
            passthrough_events.push(DeploymentEvent {
                role: challenge.role.clone(),
                slot: challenge.slot,
                source_config_path: challenge.source_config_path.clone(),
                config_id: challenge.challenger.config_id.clone(),
                config_sha256: challenge.challenger.config_sha256.clone(),
                promoted_config_path: String::new(),
                target_yaml_path: target_yaml_path.display().to_string(),
                service: challenge.service.clone(),
                soak_marker_path: existing_marker.map(|marker| {
                    paper_soak_marker_path(project_dir, marker.role.as_str())
                        .display()
                        .to_string()
                }),
                restarted_service: false,
                status: match challenge.decision.as_str() {
                    "same_config" => "already_applied".to_string(),
                    other => other.to_string(),
                },
            });
            continue;
        }
        let promoted_config_path = run_dir
            .join("promoted_configs")
            .join(format!("{}.yaml", challenge.role));
        fs::copy(&config_path, &promoted_config_path).with_context(|| {
            format!(
                "copy promoted config {} -> {}",
                config_path.display(),
                promoted_config_path.display()
            )
        })?;
        plans.push(PreparedDeployment {
            role: challenge.role.clone(),
            slot: challenge.slot,
            service: challenge.service.clone(),
            config_id: challenge.challenger.config_id.clone(),
            config_sha256: challenge.challenger.config_sha256.clone(),
            source_config_path: config_path,
            promoted_config_path,
            target_yaml_path,
        });
    }
    if !deployment.apply_to_paper {
        let mut events = passthrough_events;
        events.extend(plans.into_iter().map(|plan| DeploymentEvent {
            role: plan.role,
            slot: plan.slot,
            source_config_path: plan.source_config_path.display().to_string(),
            config_id: plan.config_id,
            config_sha256: plan.config_sha256,
            promoted_config_path: plan.promoted_config_path.display().to_string(),
            target_yaml_path: plan.target_yaml_path.display().to_string(),
            service: plan.service,
            soak_marker_path: None,
            restarted_service: false,
            status: "staged_only".to_string(),
        }));
        return Ok(events);
    }

    let mut steady_state_events = Vec::new();
    let mut pending_plans = Vec::new();
    for plan in plans {
        let existing_marker = load_paper_soak_marker(project_dir, plan.role.as_str())?;
        let current_target_config_id =
            current_target_config_id(plan.target_yaml_path.as_path(), plan.role.as_str())?;
        let already_applied = existing_marker.as_ref().is_some_and(|marker| {
            marker.config_id == plan.config_id
                && marker.slot == plan.slot
                && marker.service == plan.service
                && Path::new(&marker.target_yaml_path) == plan.target_yaml_path.as_path()
        }) && current_target_config_id.as_deref()
            == Some(plan.config_id.as_str());
        if already_applied {
            steady_state_events.push(DeploymentEvent {
                role: plan.role.clone(),
                slot: plan.slot,
                source_config_path: plan.source_config_path.display().to_string(),
                config_id: plan.config_id.clone(),
                config_sha256: plan.config_sha256.clone(),
                promoted_config_path: plan.promoted_config_path.display().to_string(),
                target_yaml_path: plan.target_yaml_path.display().to_string(),
                service: plan.service.clone(),
                soak_marker_path: existing_marker.map(|marker| {
                    paper_soak_marker_path(project_dir, marker.role.as_str())
                        .display()
                        .to_string()
                }),
                restarted_service: false,
                status: "already_applied".to_string(),
            });
        } else {
            pending_plans.push(plan);
        }
    }

    if pending_plans.is_empty() {
        let mut events = passthrough_events;
        events.extend(steady_state_events);
        return Ok(events);
    }

    let snapshots = pending_plans
        .iter()
        .map(|plan| snapshot_target(plan.target_yaml_path.as_path()))
        .collect::<Result<Vec<_>>>()?;

    let deployment_result: Result<Vec<DeploymentEvent>> = (|| {
        for plan in &pending_plans {
            if let Some(parent) = plan.target_yaml_path.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create {}", parent.display()))?;
            }
            fs::copy(&plan.source_config_path, &plan.target_yaml_path).with_context(|| {
                format!(
                    "copy selected config {} -> {}",
                    plan.source_config_path.display(),
                    plan.target_yaml_path.display()
                )
            })?;
        }

        let mut restarted = HashSet::new();
        if deployment.restart_services {
            for plan in &pending_plans {
                restart_user_service(plan.service.as_str())?;
                restarted.insert(plan.service.clone());
            }
        }

        let mut markers = HashMap::new();
        for plan in &pending_plans {
            let marker_path = write_paper_soak_marker(project_dir, run_id, plan)?;
            markers.insert(plan.role.clone(), marker_path);
        }

        let mut events = passthrough_events.clone();
        events.extend(steady_state_events.clone());
        events.extend(pending_plans.iter().map(|plan| {
            DeploymentEvent {
                role: plan.role.clone(),
                slot: plan.slot,
                source_config_path: plan.source_config_path.display().to_string(),
                config_id: plan.config_id.clone(),
                config_sha256: plan.config_sha256.clone(),
                promoted_config_path: plan.promoted_config_path.display().to_string(),
                target_yaml_path: plan.target_yaml_path.display().to_string(),
                service: plan.service.clone(),
                soak_marker_path: markers
                    .get(&plan.role)
                    .map(|path| path.display().to_string()),
                restarted_service: restarted.contains(&plan.service),
                status: "applied".to_string(),
            }
        }));
        Ok(events)
    })();

    match deployment_result {
        Ok(events) => Ok(events),
        Err(err) => {
            rollback_target_snapshots(&snapshots)?;
            if deployment.restart_services {
                restart_user_services(
                    pending_plans
                        .iter()
                        .map(|plan| plan.service.as_str())
                        .collect(),
                )?;
            }
            bail!("paper deployment rolled back after failure: {err}");
        }
    }
}

fn build_run_metadata(input: RunMetadataInput<'_>) -> Result<RunMetadata> {
    let RunMetadataInput {
        run_id,
        paths,
        settings,
        balance,
        profile,
        profile_settings,
        db_window,
        validated,
        selected,
    } = input;
    Ok(RunMetadata {
        version: DEFAULT_FACTORY_VERSION,
        generated_at_ms: Utc::now().timestamp_millis(),
        run_id: run_id.to_string(),
        git_head: git_head(&paths.project_dir)?,
        args: RunMetadataArgs {
            profile: profile.as_str().to_string(),
            config: paths.config_path.display().to_string(),
            settings: paths.settings_path.display().to_string(),
            sweep_spec: resolve_under_project(&paths.project_dir, &profile_settings.sweep_spec)
                .display()
                .to_string(),
            run_id: run_id.to_string(),
            shortlist_per_mode: profile_settings.shortlist_per_mode,
            min_trades: settings.validation.min_trades,
            slippage_reject_bps: settings.validation.slippage_bps,
            max_top1_pnl_pct: settings.validation.max_top1_pnl_pct,
            walk_forward_splits: settings.validation.walk_forward_splits,
            gpu: profile_settings.gpu,
            tpe: profile_settings.tpe,
            tpe_trials: profile_settings.tpe_trials,
            tpe_batch: profile_settings.tpe_batch,
            tpe_seed: profile_settings.tpe_seed,
            sweep_top_k: profile_settings.sweep_top_k,
            initial_balance: balance.amount_usd,
            initial_balance_source: balance.source.to_string(),
            initial_balance_secrets_path: balance.secrets_path.clone(),
            candles_db_dir: paths.candles_db_dir.display().to_string(),
            funding_db: paths.funding_db_path.display().to_string(),
            start_ts_ms: db_window.start_ts_ms,
            end_ts_ms: db_window.end_ts_ms,
            apply_to_paper: settings.deployment.apply_to_paper,
            restart_paper_services: settings.deployment.restart_services,
            apply_to_live: settings.deployment.apply_to_live,
            restart_live_service: settings.deployment.restart_live_service,
            live_yaml_path: resolve_under_project(
                &paths.project_dir,
                &settings.deployment.live_yaml_path,
            )
            .display()
            .to_string(),
            live_service: settings.deployment.live_service.clone(),
        },
        items: validated.to_vec(),
        selected_roles: selected.to_vec(),
    })
}

fn best_overall_candidate(validated: &[ValidationItem]) -> Option<ValidationItem> {
    let mut items = validated.to_vec();
    items.sort_by(|a, b| {
        b.total_pnl
            .total_cmp(&a.total_pnl)
            .then_with(|| b.profit_factor.total_cmp(&a.profit_factor))
            .then_with(|| a.max_drawdown_pct.total_cmp(&b.max_drawdown_pct))
    });
    items.into_iter().next()
}

fn build_report_markdown(items: &[ValidationItem]) -> String {
    let mut out = String::from("# Factory Report\n\n");
    out.push_str(
        "| Candidate | Mode | Trades | PnL | PF | DD | Top-1 PnL | WF median | Status |\n",
    );
    out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for item in items {
        out.push_str(&format!(
            "| {} | {} | {} | {:.2} | {:.2} | {:.2}% | {:.2}% | {:.6} | {} |\n",
            Path::new(&item.config_path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("candidate"),
            item.sort_by,
            item.total_trades,
            item.total_pnl,
            item.profit_factor,
            item.max_drawdown_pct * 100.0,
            item.top1_pnl_pct * 100.0,
            item.wf_median_oos_daily_return,
            if item.rejected {
                "rejected"
            } else {
                "validated"
            },
        ));
    }
    out
}

fn build_gate_markdown(report: &GateReport) -> String {
    format!(
        "# Step 5 Gate Report\n\n- Blocked: {}\n- Reason: {}\n- Candidate count: {}\n- Deployable count: {}\n- Selected count: {}\n",
        report.blocked,
        if report.blocked_reason.is_empty() {
            "none"
        } else {
            report.blocked_reason.as_str()
        },
        report.candidate_count,
        report.deployable_count,
        report.selected_count
    )
}

fn build_selection_markdown(report: &SelectionReport) -> String {
    let mut out = String::from("# Factory Selection\n\n");
    out.push_str(&format!(
        "- Run: `{}`\n- Selection stage: `{}`\n- Gate status: `{}`\n- Promotion stage: `{}`\n\n",
        report.run_id, report.selection_stage, report.step5_gate_status, report.promotion_stage
    ));
    if let Some(gate) = &report.paper_promotion_gate {
        out.push_str(&format!(
            "- Paper promotion gate: `{}` ({:.1}h, {} close trades, PF {:.2}, DD {:.2}%)\n\n",
            gate.status,
            gate.runtime_hours,
            gate.close_trades,
            gate.profit_factor,
            gate.max_drawdown_pct
        ));
    }
    if let Some(live) = &report.live_promotion {
        out.push_str(&format!(
            "- Live promotion: `{}` via `{}`\n- Live YAML: `{}`\n- Deployment dir: `{}`\n\n",
            live.status, live.live_service, live.live_yaml_path, live.deployment_dir
        ));
    }
    if !report.challenges.is_empty() {
        out.push_str("| Role | Decision | Reason |\n|---|---|---|\n");
        for challenge in &report.challenges {
            out.push_str(&format!(
                "| {} | `{}` | {} |\n",
                challenge.role, challenge.decision, challenge.reason
            ));
        }
        out.push('\n');
    }
    out.push_str("| Role | Config ID | Service |\n|---|---|---|\n");
    for item in &report.selected_candidates_by_role {
        out.push_str(&format!(
            "| {} | `{}` | `{}` |\n",
            item.role, item.config_id, item.service
        ));
    }
    out
}

fn run_command(
    cmd: &mut Command,
    cwd: &Path,
    stdout_path: &Path,
    stderr_path: &Path,
) -> Result<Output> {
    cmd.current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = cmd
        .output()
        .with_context(|| format!("spawn {:?}", cmd.get_program()))?;
    fs::write(stdout_path, &output.stdout)
        .with_context(|| format!("write {}", stdout_path.display()))?;
    fs::write(stderr_path, &output.stderr)
        .with_context(|| format!("write {}", stderr_path.display()))?;
    if !output.status.success() {
        bail!(
            "command {:?} failed with status {:?}",
            cmd.get_program(),
            output.status.code()
        );
    }
    Ok(output)
}

fn restart_user_service(service: &str) -> Result<bool> {
    let unit = service_unit_name(service);
    let output = Command::new("systemctl")
        .arg("--user")
        .arg("restart")
        .arg(&unit)
        .output()
        .with_context(|| format!("restart {unit}"))?;
    if !output.status.success() {
        bail!(
            "systemctl --user restart {} failed: {}",
            unit,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(true)
}

fn user_service_is_active(service: &str) -> Result<bool> {
    let unit = service_unit_name(service);
    let output = Command::new("systemctl")
        .arg("--user")
        .arg("is-active")
        .arg(&unit)
        .output()
        .with_context(|| format!("query state for {unit}"))?;
    if output.status.success() {
        return Ok(true);
    }
    let status = String::from_utf8_lossy(&output.stdout)
        .trim()
        .to_ascii_lowercase();
    if matches!(
        status.as_str(),
        "inactive" | "failed" | "activating" | "deactivating"
    ) {
        return Ok(false);
    }
    if output.status.code() == Some(3) {
        return Ok(false);
    }
    bail!(
        "systemctl --user is-active {} failed: {}",
        unit,
        String::from_utf8_lossy(&output.stderr).trim()
    )
}

fn should_restart_live_service_after_rollback(
    restart_live_service: bool,
    live_service_was_active: bool,
) -> bool {
    restart_live_service && live_service_was_active
}

fn service_unit_name(service: &str) -> String {
    let trimmed = service.trim();
    if trimmed.ends_with(".service") {
        trimmed.to_string()
    } else {
        format!("{trimmed}.service")
    }
}

fn restart_user_services(services: Vec<&str>) -> Result<()> {
    let mut attempted = BTreeSet::new();
    let mut errors = Vec::new();
    for service in services {
        if !attempted.insert(service.to_string()) {
            continue;
        }
        if let Err(err) = restart_user_service(service) {
            errors.push(err.to_string());
        }
    }
    if !errors.is_empty() {
        bail!("rollback service restart failures: {}", errors.join(" | "));
    }
    Ok(())
}

fn snapshot_target(path: &Path) -> Result<TargetSnapshot> {
    let original_bytes = if path.is_file() {
        Some(fs::read(path).with_context(|| format!("read {}", path.display()))?)
    } else {
        None
    };
    Ok(TargetSnapshot {
        target_yaml_path: path.to_path_buf(),
        original_bytes,
    })
}

fn rollback_target_snapshots(snapshots: &[TargetSnapshot]) -> Result<()> {
    for snapshot in snapshots {
        match &snapshot.original_bytes {
            Some(bytes) => {
                if let Some(parent) = snapshot.target_yaml_path.parent() {
                    fs::create_dir_all(parent)
                        .with_context(|| format!("create {}", parent.display()))?;
                }
                fs::write(&snapshot.target_yaml_path, bytes)
                    .with_context(|| format!("restore {}", snapshot.target_yaml_path.display()))?;
            }
            None => {
                if snapshot.target_yaml_path.exists() {
                    fs::remove_file(&snapshot.target_yaml_path).with_context(|| {
                        format!("remove {}", snapshot.target_yaml_path.display())
                    })?;
                }
            }
        }
    }
    Ok(())
}

fn atomic_write_text(path: &Path, text: &str) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("invalid target path {}", path.display()))?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let stem = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("config");
    let tmp = parent.join(format!(
        ".{stem}.tmp.{}.{}",
        std::process::id(),
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    fs::write(&tmp, text).with_context(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

fn load_yaml_engine_interval(yaml_text: &str) -> String {
    let parsed: serde_yaml::Value = match serde_yaml::from_str(yaml_text) {
        Ok(value) => value,
        Err(_) => return String::new(),
    };
    parsed
        .get("global")
        .and_then(|value| value.get("engine"))
        .and_then(|value| value.get("interval"))
        .and_then(|value| value.as_str())
        .map(|value| value.trim().to_string())
        .unwrap_or_default()
}

fn iso_utc_now() -> String {
    Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(value).context("serialise JSON artefact")?,
    )
    .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn file_sha256_hex(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn git_head(project_dir: &Path) -> Result<String> {
    let output = Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .current_dir(project_dir)
        .output()
        .context("run git rev-parse HEAD")?;
    if !output.status.success() {
        bail!("git rev-parse HEAD failed");
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paper_config::PaperEffectiveConfig;
    use crate::paper_lane::PaperLane;
    use crate::test_support::{env_lock, EnvGuard};

    fn config_yaml(interval: &str) -> String {
        format!(
            "global:\n  engine:\n    interval: {interval}\n    entry_interval: 3m\n    exit_interval: 3m\n"
        )
    }

    fn strategy_fingerprint_from_yaml(yaml: &str) -> String {
        let raw_document: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let cfg = load_config_document_checked(&raw_document, None, false, None)
            .map_err(|err| anyhow!(err))
            .unwrap();
        strategy_config_fingerprint_sha256(&cfg)
    }

    fn factory_root_config_yaml() -> String {
        r#"
global:
  trade:
    leverage: 2.0
  engine:
    interval: 30m
    entry_interval: 3m
    exit_interval: 3m
modes:
  primary:
    global:
      engine:
        interval: 5m
        entry_interval: 1m
        exit_interval: 1m
  fallback:
    global:
      engine:
        interval: 1h
        entry_interval: 3m
        exit_interval: 3m
  conservative:
    global:
      engine:
        interval: 1h
        entry_interval: 15m
        exit_interval: 15m
symbols:
  ETH:
    trade:
      leverage: 4.0
"#
        .to_string()
    }

    fn paper_promotion_db(project_dir: &Path) -> PathBuf {
        project_dir.join(DEFAULT_PRIMARY_PAPER_DB_PATH)
    }

    fn create_paper_gate_schema(conn: &Connection) {
        conn.execute_batch(
            "
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                action TEXT,
                pnl REAL,
                fee_usd REAL,
                balance REAL
            );
            CREATE TABLE decision_events (
                id TEXT PRIMARY KEY,
                trade_id INTEGER,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                config_fingerprint TEXT,
                run_fingerprint TEXT
            );
            CREATE TABLE audit_events (
                id INTEGER PRIMARY KEY,
                event TEXT,
                level TEXT,
                data_json TEXT,
                run_fingerprint TEXT
            );
            ",
        )
        .unwrap();
    }

    fn insert_paper_fill(
        conn: &Connection,
        trade_id: i64,
        timestamp: &str,
        action: &str,
        pnl: f64,
        fee_usd: f64,
        balance: f64,
        config_fingerprint: &str,
        run_fingerprint: &str,
    ) {
        conn.execute(
            "INSERT INTO trades (id, timestamp, action, pnl, fee_usd, balance) VALUES (?, ?, ?, ?, ?, ?)",
            rusqlite::params![trade_id, timestamp, action, pnl, fee_usd, balance],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO decision_events (id, trade_id, event_type, status, config_fingerprint, run_fingerprint)
             VALUES (?, ?, 'fill', 'executed', ?, ?)",
            rusqlite::params![
                format!("decision-{trade_id}"),
                trade_id,
                config_fingerprint,
                run_fingerprint
            ],
        )
        .unwrap();
    }

    fn paper_target_primary() -> PaperTarget {
        PaperTarget {
            role: "primary".to_string(),
            slot: 1,
            service: "openclaw-ai-quant-trader-v8-paper1".to_string(),
            yaml_path: PathBuf::from("config/strategy_overrides.paper1.yaml"),
        }
    }

    fn validation_item(mode: &str, config_id: &str, pnl: f64, pf: f64, dd: f64) -> ValidationItem {
        ValidationItem {
            candidate_mode: true,
            canonical_cpu_verified: true,
            config_id: config_id.to_string(),
            config_path: format!("/tmp/{config_id}.yaml"),
            config_sha256: config_id.to_string(),
            final_balance: 10_000.0 + pnl,
            initial_balance: 10_000.0,
            max_drawdown_pct: dd,
            pipeline_stage: "candidate_validation".to_string(),
            profit_factor: pf,
            rank: 1,
            replay_report_path: "/tmp/replay.json".to_string(),
            replay_stage: "cpu_replay".to_string(),
            schema_version: 1,
            shortlist_mode: mode.to_string(),
            slippage_pnl_at_reject_bps: pnl,
            slippage_reject_bps: 20.0,
            sort_by: mode.to_string(),
            step4_parity: ParityCheck {
                status: "pass".to_string(),
                cpu_total_trades: 40,
                gpu_total_trades: 40,
                trade_delta: 0,
                cpu_final_balance: 10_000.0 + pnl,
                gpu_final_balance: 10_000.0 + pnl,
                balance_delta_abs: 0.0,
                balance_delta_rel: 0.0,
                thresholds: ParityThresholds {
                    abs_eps: 0.001,
                    rel_eps: 0.000001,
                    trade_delta_max: 0,
                },
            },
            sweep_stage: "gpu_tpe".to_string(),
            top1_pnl_pct: 0.2,
            total_fees: 1.0,
            total_pnl: pnl,
            total_trades: 40,
            validation_gate: "candidate->validated".to_string(),
            walk_forward_summary_path: "/tmp/wf.json".to_string(),
            wf_median_oos_daily_return: 0.01,
            blocked_reasons: Vec::new(),
            rejected: false,
            reject_reason: String::new(),
        }
    }

    fn performance_summary(config_id: &str, pnl: f64, pf: f64, dd: f64) -> PerformanceSummary {
        PerformanceSummary {
            config_id: config_id.to_string(),
            config_path: format!("/tmp/{config_id}.yaml"),
            config_sha256: format!("sha-{config_id}"),
            final_balance: 10_000.0 + pnl,
            initial_balance: 10_000.0,
            max_drawdown_pct: dd,
            profit_factor: pf,
            total_pnl: pnl,
            total_trades: 40,
            total_fees: 10.0,
            top1_pnl_pct: 0.2,
            slippage_pnl_at_reject_bps: pnl,
            slippage_reject_bps: 20.0,
            wf_median_oos_daily_return: 0.01,
            replay_report_path: "/tmp/replay.json".to_string(),
            walk_forward_summary_path: "/tmp/wf.json".to_string(),
            rejected: false,
            reject_reason: String::new(),
            blocked_reasons: Vec::new(),
        }
    }

    #[test]
    fn generated_candidate_configs_preserve_root_overlays() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("candidate.yaml");
        let raw_document: serde_yaml::Value =
            serde_yaml::from_str(&factory_root_config_yaml()).unwrap();
        let base_cfg = load_config_document_checked(&raw_document, None, false, None)
            .map_err(|err| anyhow!(err))
            .unwrap();
        let candidate = generate_candidate_config(
            &raw_document,
            &base_cfg,
            &BTreeMap::from([("trade.leverage".to_string(), 7.0)]),
            &path,
            "candidate_growth_rank1",
            ShortlistMode::Growth,
            1,
            &SweepCandidateRow {
                candidate_mode: true,
                config_id: "row".to_string(),
                max_drawdown_pct: 0.1,
                overrides: BTreeMap::from([("trade.leverage".to_string(), 7.0)]),
                profit_factor: 1.6,
                total_pnl: 42.0,
                total_trades: 64,
            },
        )
        .unwrap();

        let root = candidate.as_mapping().unwrap();
        let modes_key = serde_yaml::Value::String("modes".to_string());
        let symbols_key = serde_yaml::Value::String("symbols".to_string());
        assert!(root.contains_key(&modes_key));
        assert!(root.contains_key(&symbols_key));

        let cfg = load_config_document_checked(&candidate, None, false, None)
            .map_err(|err| anyhow!(err))
            .unwrap();
        assert!((cfg.trade.leverage - 7.0).abs() < f64::EPSILON);
        assert_eq!(read_yaml_document(&path).unwrap(), candidate);
    }

    #[test]
    fn factory_fallback_effective_config_matches_runtime_lane_resolution() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let config_dir = dir.path().join("config");
        let runtime_output = dir.path().join("runtime-output");
        let config_path = config_dir.join("strategy_overrides.paper2.yaml");
        let output_path = dir.path().join("factory").join("fallback.effective.yaml");
        fs::create_dir_all(&config_dir).unwrap();
        fs::write(&config_path, factory_root_config_yaml()).unwrap();
        let _env = EnvGuard::set(&[
            ("AI_QUANT_PROMOTED_ROLE", None),
            ("AI_QUANT_STRATEGY_MODE", None),
            ("AI_QUANT_STRATEGY_MODE_FILE", None),
            (
                "AI_QUANT_EFFECTIVE_CONFIG_OUTPUT_ROOT",
                Some(runtime_output.to_str().unwrap()),
            ),
        ]);

        let source_document = read_yaml_document(&config_path).unwrap();
        let factory = write_role_effective_config(
            &source_document,
            "fallback",
            &output_path,
            &config_path,
            true,
        )
        .unwrap();
        let runtime =
            PaperEffectiveConfig::resolve(None, Some(PaperLane::Paper2), Some(dir.path())).unwrap();
        let runtime_cfg = runtime.load_config(None, false).unwrap();

        assert_eq!(
            factory.config_id,
            strategy_config_fingerprint_sha256(&runtime_cfg)
        );
        assert_eq!(
            read_yaml_document(&output_path).unwrap(),
            read_yaml_document(runtime.effective_yaml_path()).unwrap()
        );
        assert_eq!(runtime.build_report(None, false).unwrap().interval, "1h");
    }

    #[test]
    fn factory_conservative_effective_config_matches_runtime_lane_resolution() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let config_dir = dir.path().join("config");
        let runtime_output = dir.path().join("runtime-output");
        let config_path = config_dir.join("strategy_overrides.paper3.yaml");
        let output_path = dir
            .path()
            .join("factory")
            .join("conservative.effective.yaml");
        fs::create_dir_all(&config_dir).unwrap();
        fs::write(&config_path, factory_root_config_yaml()).unwrap();
        let _env = EnvGuard::set(&[
            ("AI_QUANT_PROMOTED_ROLE", None),
            ("AI_QUANT_STRATEGY_MODE", None),
            ("AI_QUANT_STRATEGY_MODE_FILE", None),
            (
                "AI_QUANT_EFFECTIVE_CONFIG_OUTPUT_ROOT",
                Some(runtime_output.to_str().unwrap()),
            ),
        ]);

        let source_document = read_yaml_document(&config_path).unwrap();
        let factory = write_role_effective_config(
            &source_document,
            "conservative",
            &output_path,
            &config_path,
            true,
        )
        .unwrap();
        let runtime =
            PaperEffectiveConfig::resolve(None, Some(PaperLane::Paper3), Some(dir.path())).unwrap();
        let runtime_cfg = runtime.load_config(None, false).unwrap();

        assert_eq!(
            factory.config_id,
            strategy_config_fingerprint_sha256(&runtime_cfg)
        );
        assert_eq!(
            read_yaml_document(&output_path).unwrap(),
            read_yaml_document(runtime.effective_yaml_path()).unwrap()
        );
        assert_eq!(runtime.build_report(None, false).unwrap().interval, "1h");
    }

    #[test]
    fn shortlist_modes_deduplicate_candidates() {
        let row = SweepCandidateRow {
            candidate_mode: true,
            config_id: "row".to_string(),
            max_drawdown_pct: 0.1,
            overrides: BTreeMap::from([("trade.sl_atr_mult".to_string(), 2.0)]),
            profit_factor: 1.3,
            total_pnl: 20.0,
            total_trades: 40,
        };
        let rows = vec![row.clone(), row];
        let shortlist = build_shortlist(&rows, 2);
        assert_eq!(shortlist.len(), 1);
    }

    #[test]
    fn top1_symbol_share_uses_positive_symbol_pnl() {
        let report = ReplaySummary {
            initial_balance: 10_000.0,
            final_balance: 10_100.0,
            total_pnl: 100.0,
            total_trades: 10,
            profit_factor: 1.5,
            max_drawdown_pct: 0.1,
            total_fees: 1.0,
            by_symbol: vec![
                SymbolBucket {
                    symbol: "BTC".to_string(),
                    trades: 5,
                    pnl: 60.0,
                    win_rate: 0.5,
                },
                SymbolBucket {
                    symbol: "ETH".to_string(),
                    trades: 5,
                    pnl: 40.0,
                    win_rate: 0.5,
                },
            ],
        };
        assert!((top1_symbol_share(&report) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn selection_prefers_mode_specific_roles_and_avoids_duplicates() {
        let items = vec![
            validation_item("efficient", "cfg-a", 100.0, 1.2, 0.3),
            validation_item("growth", "cfg-b", 50.0, 2.0, 0.4),
            validation_item("conservative", "cfg-c", 10.0, 1.1, 0.05),
        ];
        let selected = select_roles(&items, &SelectionSettings::default());
        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0].config_id, "cfg-a");
        assert_eq!(selected[1].config_id, "cfg-b");
        assert_eq!(selected[2].config_id, "cfg-c");
    }

    #[test]
    fn selected_roles_limits_active_targets_and_gate_requirements() {
        let mut selection = SelectionSettings::default();
        selection.selected_roles = vec!["primary".to_string()];
        let targets = active_paper_targets(&selection);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].role, "primary");

        let items = vec![validation_item("efficient", "cfg-a", 100.0, 1.2, 0.3)];
        let selected = select_roles(&items, &selection);
        assert_eq!(selected.len(), 1);

        let gate = build_gate_report("run", &items, &selected, &selection);
        assert!(!gate.blocked);
        assert_eq!(gate.slot_bindings.len(), 1);
    }

    #[test]
    fn selection_validation_rejects_unknown_selected_roles() {
        let mut selection = SelectionSettings::default();
        selection.selected_roles = vec!["typo-primary".to_string()];
        let err = validate_selection_settings(&selection).unwrap_err();
        assert!(err.to_string().contains("unknown selection.selected_roles"));
    }

    #[test]
    fn rollback_target_snapshots_restores_original_yaml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("paper1.yaml");
        fs::write(&path, "before: true\n").unwrap();
        let snapshot = snapshot_target(&path).unwrap();
        fs::write(&path, "after: true\n").unwrap();
        rollback_target_snapshots(&[snapshot]).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "before: true\n");
    }

    #[test]
    fn acquire_factory_lock_rejects_parallel_holder() {
        let dir = tempfile::tempdir().unwrap();
        let _lock = acquire_factory_lock(dir.path()).unwrap();
        let err = acquire_factory_lock(dir.path()).unwrap_err();
        assert!(err.to_string().contains("factory cycle already running"));
    }

    #[test]
    fn validate_factory_defaults_requires_primary_when_live_promotion_is_on() {
        let mut settings = FactoryDefaults::default();
        settings.deployment.apply_to_live = true;
        settings.selection.selected_roles = vec!["fallback".to_string()];
        let err = validate_factory_defaults(&settings).unwrap_err();
        assert!(err.to_string().contains("requires `primary` to be present"));
    }

    #[test]
    fn resolve_initial_balance_uses_live_equity_by_default() {
        let settings = BalanceSettings::default();
        let profile = FactoryProfileSettings::daily();
        let result = resolve_initial_balance_with(Path::new("."), &settings, &profile, |_path| {
            Ok(12_345.67)
        })
        .unwrap();
        assert_eq!(result.source, "live_equity");
        assert!((result.amount_usd - 12_345.67).abs() < 1e-9);
    }

    #[test]
    fn challenger_compare_keeps_incumbent_when_role_order_prefers_it() {
        let target = paper_target_primary();
        let challenger = performance_summary("challenger", 100.0, 1.2, 0.3);
        let incumbent = performance_summary("incumbent", 120.0, 1.3, 0.25);
        let source_config_path = challenger.config_path.clone();
        let report = compare_challenger_to_incumbent(
            &target,
            &ComparisonSettings::default(),
            source_config_path.as_str(),
            challenger,
            Some(incumbent),
        );
        assert_eq!(report.decision, "incumbent_holds");
    }

    #[test]
    fn rejected_lane_effective_challenger_never_deploys_without_incumbent() {
        let target = paper_target_primary();
        let mut challenger = performance_summary("challenger", 100.0, 1.2, 0.3);
        challenger.rejected = true;
        challenger.reject_reason = "lane validation failed".to_string();
        let source_config_path = challenger.config_path.clone();
        let report = compare_challenger_to_incumbent(
            &target,
            &ComparisonSettings::default(),
            source_config_path.as_str(),
            challenger,
            None,
        );
        assert_eq!(report.decision, "challenger_rejected");
        assert!(report.reason.contains("lane-effective challenger fails"));
    }

    #[test]
    fn promote_primary_live_updates_live_yaml_and_writes_deploy_artefacts() {
        let dir = tempfile::tempdir().unwrap();
        let project_dir = dir.path();
        let artifacts_root = project_dir.join("artifacts");
        let run_dir = artifacts_root.join("2026-03-14").join("run_nightly_test");
        fs::create_dir_all(run_dir.join("promoted_configs")).unwrap();

        let live_yaml = project_dir
            .join("config")
            .join("strategy_overrides.live.yaml");
        fs::create_dir_all(live_yaml.parent().unwrap()).unwrap();
        fs::write(&live_yaml, config_yaml("30m")).unwrap();

        let source_config = run_dir.join("configs").join("candidate_primary.yaml");
        fs::create_dir_all(source_config.parent().unwrap()).unwrap();
        fs::write(&source_config, config_yaml("30m")).unwrap();

        let promoted_config = run_dir.join("promoted_configs").join("primary.yaml");
        fs::write(&promoted_config, config_yaml("30m")).unwrap();

        let validated = vec![ValidationItem {
            config_path: source_config.display().to_string(),
            ..validation_item("efficient", "cfg-primary", 100.0, 1.2, 0.3)
        }];
        let selected = vec![SelectionCandidate {
            role: "primary".to_string(),
            slot: 1,
            service: "openclaw-ai-quant-trader-v8-paper1".to_string(),
            source: "promotion_roles".to_string(),
            selected: true,
            config_id: "cfg-primary".to_string(),
        }];
        let paper_deployments = vec![DeploymentEvent {
            role: "primary".to_string(),
            slot: 1,
            source_config_path: source_config.display().to_string(),
            config_id: "cfg-primary".to_string(),
            config_sha256: "cfg-primary".to_string(),
            promoted_config_path: promoted_config.display().to_string(),
            target_yaml_path: project_dir
                .join("config")
                .join("strategy_overrides.paper1.yaml")
                .display()
                .to_string(),
            service: "openclaw-ai-quant-trader-v8-paper1".to_string(),
            soak_marker_path: None,
            restarted_service: false,
            status: "applied".to_string(),
        }];
        let deployment = DeploymentSettings {
            apply_to_paper: true,
            restart_services: false,
            apply_to_live: true,
            restart_live_service: false,
            live_yaml_path: PathBuf::from("config/strategy_overrides.live.yaml"),
            live_service: "openclaw-ai-quant-live-v8".to_string(),
        };

        let event = promote_primary_live(
            project_dir,
            &artifacts_root,
            &run_dir,
            &validated,
            &selected,
            &paper_deployments,
            &deployment,
        )
        .unwrap();

        assert_eq!(event.role, "primary");
        assert_eq!(event.status, "applied");
        assert!(!event.service_was_active_before);
        assert!(!event.restarted_service);
        assert_eq!(fs::read_to_string(&live_yaml).unwrap(), config_yaml("30m"));
        assert!(Path::new(&event.deployment_dir)
            .join("promoted_config.yaml")
            .is_file());
        assert!(Path::new(&event.deployment_dir)
            .join("prev_config.yaml")
            .is_file());
        assert!(Path::new(&event.deployment_dir)
            .join("promotion_event.json")
            .is_file());
    }

    #[test]
    fn rollback_only_restarts_live_service_when_it_was_previously_active() {
        assert!(!should_restart_live_service_after_rollback(true, false));
        assert!(!should_restart_live_service_after_rollback(false, true));
        assert!(should_restart_live_service_after_rollback(true, true));
    }

    #[test]
    fn paper_promotion_gate_blocks_without_runtime_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = paper_promotion_db(dir.path());
        fs::create_dir_all(db_path.parent().unwrap()).unwrap();
        let conn = Connection::open(&db_path).unwrap();
        create_paper_gate_schema(&conn);

        let candidate = validation_item("efficient", "cfg-primary", 100.0, 1.2, 0.3);
        let gate = assess_paper_promotion_gate(&db_path, &candidate, None).unwrap();
        assert_eq!(gate.status, "blocked");
        assert!(gate
            .blocked_reasons
            .iter()
            .any(|reason| reason.contains("no paper fill evidence")));
    }

    #[test]
    fn paper_promotion_gate_passes_with_soaked_primary_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = paper_promotion_db(dir.path());
        fs::create_dir_all(db_path.parent().unwrap()).unwrap();
        let conn = Connection::open(&db_path).unwrap();
        create_paper_gate_schema(&conn);

        let mut candidate = validation_item("efficient", "cfg-primary", 100.0, 1.2, 0.3);
        candidate.config_sha256 = "ephemeral-file-sha".to_string();
        let config_id = candidate.config_id.clone();
        insert_paper_fill(
            &conn,
            1,
            "2026-03-01T00:00:00Z",
            "CLOSE",
            15.0,
            0.5,
            10_050.0,
            &config_id,
            "run-1",
        );
        insert_paper_fill(
            &conn,
            2,
            "2026-03-02T01:30:00Z",
            "CLOSE",
            8.0,
            0.5,
            10_120.0,
            &config_id,
            "run-1",
        );

        let gate = assess_paper_promotion_gate(&db_path, &candidate, None).unwrap();
        assert_eq!(gate.status, "pass");
        assert!(gate.runtime_hours >= 24.0);
        assert_eq!(gate.close_trades, 2);
        assert_eq!(gate.kill_switch_events, 0);
    }

    #[test]
    fn paper_promotion_gate_uses_current_soak_window_after_redeploy() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = paper_promotion_db(dir.path());
        fs::create_dir_all(db_path.parent().unwrap()).unwrap();
        let conn = Connection::open(&db_path).unwrap();
        create_paper_gate_schema(&conn);

        let candidate = validation_item("efficient", "cfg-primary", 100.0, 1.2, 0.3);
        let config_fingerprint = candidate.config_id.clone();
        insert_paper_fill(
            &conn,
            1,
            "2026-03-01T00:00:00Z",
            "CLOSE",
            15.0,
            0.5,
            10_050.0,
            &config_fingerprint,
            "run-old",
        );
        insert_paper_fill(
            &conn,
            2,
            "2026-03-02T01:30:00Z",
            "CLOSE",
            8.0,
            0.5,
            10_120.0,
            &config_fingerprint,
            "run-old",
        );

        let redeploy_after_ms = parse_trade_timestamp_ms("2026-03-03T00:00:00Z").unwrap();
        let gate =
            assess_paper_promotion_gate(&db_path, &candidate, Some(redeploy_after_ms)).unwrap();
        assert_eq!(gate.status, "blocked");
        assert_eq!(gate.close_trades, 0);
        assert!(gate
            .blocked_reasons
            .iter()
            .any(|reason| reason.contains("no paper fill evidence")));
    }

    #[test]
    fn primary_soak_marker_identity_uses_stable_config_id() {
        let dir = tempfile::tempdir().unwrap();
        let project_dir = dir.path();
        let db_path = paper_promotion_db(project_dir);
        fs::create_dir_all(db_path.parent().unwrap()).unwrap();
        let conn = Connection::open(&db_path).unwrap();
        create_paper_gate_schema(&conn);

        let yaml = config_yaml("30m");
        let config_id = strategy_fingerprint_from_yaml(&yaml);
        let source_config = project_dir.join("candidate_primary.yaml");
        let target_yaml = project_dir
            .join("config")
            .join("strategy_overrides.paper1.yaml");
        fs::write(&source_config, &yaml).unwrap();
        fs::create_dir_all(target_yaml.parent().unwrap()).unwrap();
        fs::write(&target_yaml, &yaml).unwrap();
        let mut candidate = validation_item("efficient", &config_id, 100.0, 1.2, 0.3);
        candidate.config_path = source_config.display().to_string();
        candidate.config_sha256 = "new-file-sha".to_string();
        insert_paper_fill(
            &conn,
            1,
            "2026-03-03T00:00:00Z",
            "CLOSE",
            15.0,
            0.5,
            10_050.0,
            &candidate.config_id,
            "run-current",
        );
        insert_paper_fill(
            &conn,
            2,
            "2026-03-04T01:30:00Z",
            "CLOSE",
            8.0,
            0.5,
            10_120.0,
            &candidate.config_id,
            "run-current",
        );

        let marker = PaperSoakMarker {
            version: "factory_paper_soak_v1".to_string(),
            role: "primary".to_string(),
            slot: 1,
            service: "openclaw-ai-quant-trader-v8-paper1".to_string(),
            paper_db_path: db_path.display().to_string(),
            config_id: candidate.config_id.clone(),
            config_sha256: "old-file-sha".to_string(),
            deployed_at_ms: parse_trade_timestamp_ms("2026-03-03T00:00:00Z").unwrap(),
            deployed_by_run_id: "run-123".to_string(),
            source_config_path: "/tmp/source.yaml".to_string(),
            target_yaml_path: target_yaml.display().to_string(),
        };
        write_json(&paper_soak_marker_path(project_dir, "primary"), &marker).unwrap();

        let selected = vec![SelectionCandidate {
            role: "primary".to_string(),
            slot: 1,
            service: "openclaw-ai-quant-trader-v8-paper1".to_string(),
            source: "promotion_roles".to_string(),
            selected: true,
            config_id: candidate.config_id.clone(),
        }];
        let gate = assess_primary_paper_promotion_gate(
            project_dir,
            &SelectionSettings::default(),
            &[candidate],
            &selected,
        )
        .unwrap();
        assert_eq!(gate.status, "pass");
    }

    #[test]
    fn deploy_selected_configs_preserves_existing_soak_marker_for_same_config() {
        let dir = tempfile::tempdir().unwrap();
        let project_dir = dir.path();
        let run_dir = project_dir
            .join("artifacts")
            .join("2026-03-14")
            .join("run_test");
        fs::create_dir_all(run_dir.join("promoted_configs")).unwrap();

        let yaml = config_yaml("30m");
        let config_id = strategy_fingerprint_from_yaml(&yaml);
        let source_config = run_dir.join("configs").join("candidate_primary.yaml");
        fs::create_dir_all(source_config.parent().unwrap()).unwrap();
        fs::write(&source_config, &yaml).unwrap();

        let target_yaml = project_dir
            .join("config")
            .join("strategy_overrides.paper1.yaml");
        fs::create_dir_all(target_yaml.parent().unwrap()).unwrap();
        fs::write(&target_yaml, &yaml).unwrap();

        let old_marker = PaperSoakMarker {
            version: "factory_paper_soak_v1".to_string(),
            role: "primary".to_string(),
            slot: 1,
            service: "openclaw-ai-quant-trader-v8-paper1".to_string(),
            paper_db_path: paper_promotion_db(project_dir).display().to_string(),
            config_id: config_id.clone(),
            config_sha256: "old-file-sha".to_string(),
            deployed_at_ms: 111,
            deployed_by_run_id: "old-run".to_string(),
            source_config_path: "/tmp/source.yaml".to_string(),
            target_yaml_path: target_yaml.display().to_string(),
        };
        write_json(&paper_soak_marker_path(project_dir, "primary"), &old_marker).unwrap();

        let runtime_config = run_dir
            .join("lane_effective")
            .join("challenger_primary.runtime.yaml");
        fs::create_dir_all(runtime_config.parent().unwrap()).unwrap();
        fs::write(&runtime_config, &yaml).unwrap();
        let challenge = DeploymentChallenge {
            role: "primary".to_string(),
            slot: 1,
            service: "openclaw-ai-quant-trader-v8-paper1".to_string(),
            target_yaml_path: paper_target_primary().yaml_path.display().to_string(),
            source_config_path: source_config.display().to_string(),
            decision: "same_config".to_string(),
            reason: "test".to_string(),
            incumbent: Some(performance_summary(&config_id, 90.0, 1.1, 0.4)),
            challenger: PerformanceSummary {
                config_path: runtime_config.display().to_string(),
                ..performance_summary(&config_id, 100.0, 1.2, 0.3)
            },
        };
        let events = deploy_selected_configs(
            project_dir,
            "run-new",
            &run_dir,
            &[challenge],
            &DeploymentSettings {
                apply_to_paper: true,
                restart_services: false,
                apply_to_live: false,
                restart_live_service: false,
                live_yaml_path: PathBuf::from(DEFAULT_LIVE_YAML_PATH),
                live_service: DEFAULT_LIVE_SERVICE.to_string(),
            },
        )
        .unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].status, "already_applied");
        assert_eq!(
            events[0].source_config_path,
            source_config.display().to_string()
        );
        let marker = load_paper_soak_marker(project_dir, "primary")
            .unwrap()
            .unwrap();
        assert_eq!(marker.deployed_at_ms, 111);
        assert_eq!(marker.deployed_by_run_id, "old-run");
    }

    #[test]
    fn primary_soak_marker_must_match_full_target_identity() {
        let dir = tempfile::tempdir().unwrap();
        let project_dir = dir.path();
        let db_path = paper_promotion_db(project_dir);
        fs::create_dir_all(db_path.parent().unwrap()).unwrap();
        let conn = Connection::open(&db_path).unwrap();
        create_paper_gate_schema(&conn);

        let yaml = config_yaml("30m");
        let config_id = strategy_fingerprint_from_yaml(&yaml);
        let source_config = project_dir.join("candidate_primary.yaml");
        fs::write(&source_config, &yaml).unwrap();
        let mut candidate = validation_item("efficient", &config_id, 100.0, 1.2, 0.3);
        candidate.config_path = source_config.display().to_string();
        insert_paper_fill(
            &conn,
            1,
            "2026-03-03T00:00:00Z",
            "CLOSE",
            15.0,
            0.5,
            10_050.0,
            &candidate.config_id,
            "run-current",
        );
        insert_paper_fill(
            &conn,
            2,
            "2026-03-04T01:30:00Z",
            "CLOSE",
            8.0,
            0.5,
            10_120.0,
            &candidate.config_id,
            "run-current",
        );

        let marker = PaperSoakMarker {
            version: "factory_paper_soak_v1".to_string(),
            role: "primary".to_string(),
            slot: 2,
            service: "openclaw-ai-quant-trader-v8-paper2".to_string(),
            paper_db_path: db_path.display().to_string(),
            config_id: candidate.config_id.clone(),
            config_sha256: "old-file-sha".to_string(),
            deployed_at_ms: parse_trade_timestamp_ms("2026-03-03T00:00:00Z").unwrap(),
            deployed_by_run_id: "run-123".to_string(),
            source_config_path: "/tmp/source.yaml".to_string(),
            target_yaml_path: project_dir
                .join("config")
                .join("strategy_overrides.paper2.yaml")
                .display()
                .to_string(),
        };
        write_json(&paper_soak_marker_path(project_dir, "primary"), &marker).unwrap();

        let selected = vec![SelectionCandidate {
            role: "primary".to_string(),
            slot: 1,
            service: "openclaw-ai-quant-trader-v8-paper1".to_string(),
            source: "promotion_roles".to_string(),
            selected: true,
            config_id: candidate.config_id.clone(),
        }];
        let gate = assess_primary_paper_promotion_gate(
            project_dir,
            &SelectionSettings::default(),
            &[candidate],
            &selected,
        )
        .unwrap();
        assert_eq!(gate.status, "blocked");
        assert!(gate
            .blocked_reasons
            .iter()
            .any(|reason| reason.contains("no longer matches slot/service/yaml target")));
    }
}
