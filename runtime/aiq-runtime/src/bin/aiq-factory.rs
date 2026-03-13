use anyhow::{anyhow, bail, Context, Result};
use bt_core::config::{
    load_config_document_checked, strategy_config_fingerprint_sha256,
    strategy_config_to_yaml_value, StrategyConfig,
};
use bt_core::sweep::apply_one_pub;
use bt_data::sqlite_loader::query_time_range_multi;
use chrono::{SecondsFormat, Utc};
use clap::{Args, Parser, Subcommand, ValueEnum};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const DEFAULT_FACTORY_SETTINGS: &str = "config/factory_defaults.yaml";
const DEFAULT_CONFIG_PATH: &str = "config/strategy_overrides.yaml";
const DEFAULT_ARTIFACTS_DIR: &str = "artifacts";
const DEFAULT_CANDLES_DB_DIR: &str = "candles_dbs";
const DEFAULT_FUNDING_DB: &str = "candles_dbs/funding_rates.db";
const DEFAULT_FACTORY_VERSION: &str = "factory_cycle_rust_v1";
const DEFAULT_SELECTION_POLICY: &str = "promotion_roles_v2";

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
}

impl Default for DeploymentSettings {
    fn default() -> Self {
        Self {
            apply_to_paper: true,
            restart_services: true,
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
    promotion_stage: String,
    selection_policy: &'static str,
    selection_stage: String,
    selection_warnings: Vec<String>,
    step5_gate_status: String,
    step5_gate_block_reason: String,
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
    source_config_path: String,
    promoted_config_path: String,
    target_yaml_path: String,
    service: String,
    restarted_service: bool,
    status: String,
}

#[derive(Debug, Clone)]
struct PreparedDeployment {
    role: String,
    service: String,
    source_config_path: PathBuf,
    promoted_config_path: PathBuf,
    target_yaml_path: PathBuf,
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
    candles_db_dir: String,
    funding_db: String,
    start_ts_ms: i64,
    end_ts_ms: i64,
}

#[derive(Debug, Clone, Serialize)]
struct FactoryRunSummary {
    job: String,
    version: &'static str,
    run_id: String,
    run_dir: String,
    report_json: String,
    selection_json: String,
    blocked: bool,
    blocked_reason: String,
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
    deployments: Vec<DeploymentEvent>,
    selection_warnings: Vec<String>,
}

struct RunMetadataInput<'a> {
    run_id: &'a str,
    paths: &'a ResolvedPaths,
    settings: &'a FactoryDefaults,
    profile: FactoryProfile,
    profile_settings: &'a FactoryProfileSettings,
    db_window: &'a ResolvedDbWindow,
    validated: &'a [ValidationItem],
    selected: &'a [SelectionCandidate],
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

    fs::create_dir_all(&paths.artifacts_root).with_context(|| {
        format!(
            "create artifacts root {}",
            paths.artifacts_root.as_path().display()
        )
    })?;
    let _factory_lock = acquire_factory_lock(&paths.artifacts_root)?;

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
        &base_cfg,
    )?;

    let db_window = compute_common_time_range(&base_cfg, &paths.candles_db_dir)?;
    let sweep_candidates_path = run_dir.join("sweeps/sweep_candidates.jsonl");
    let sweep_results_path = run_dir.join("sweeps/sweep_results.jsonl");
    run_sweep(
        &paths,
        &profile_settings,
        &effective_config_path,
        &db_window,
        &sweep_candidates_path,
        &run_dir.join("sweeps/sweep.stderr.txt"),
        &run_dir.join("sweeps/sweep.stdout.txt"),
    )?;
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
            &base_cfg,
            &run_dir,
            &db_window,
            candidate,
            profile_settings.initial_balance,
        )?);
    }
    if validated.is_empty() {
        bail!("factory validation produced no candidate evidence");
    }

    let selected = select_roles(&validated, &settings.selection);
    let gate_report = build_gate_report(&run_id, &validated, &selected, &settings.selection);
    let (deployed, deployments, selection_warnings) = if gate_report.blocked {
        (false, Vec::new(), Vec::new())
    } else {
        match deploy_selected_configs(
            &paths.project_dir,
            &run_dir,
            &validated,
            &selected,
            &settings.selection,
            &settings.deployment,
        ) {
            Ok(events) => (true, events, Vec::new()),
            Err(err) => (
                false,
                Vec::new(),
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
        deployments: deployments.clone(),
        selection_warnings: selection_warnings.clone(),
    })?;
    write_reports(&run_dir, &validated, &gate_report, &selection_report)?;
    let metadata = build_run_metadata(RunMetadataInput {
        run_id: &run_id,
        paths: &paths,
        settings: &settings,
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
        blocked: gate_report.blocked || !deployed,
        blocked_reason: if !selection_warnings.is_empty() {
            selection_warnings.join("; ")
        } else {
            gate_report.blocked_reason.clone()
        },
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

fn acquire_factory_lock(artifacts_root: &Path) -> Result<FactoryRunLock> {
    let lock_dir = artifacts_root.join("_locks");
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
    base_cfg: &StrategyConfig,
) -> Result<PathBuf> {
    let effective_dir = artifacts_root.join("_effective_configs");
    fs::create_dir_all(&effective_dir)
        .with_context(|| format!("create {}", effective_dir.display()))?;
    let path = effective_dir.join(format!("{}_{}.yaml", profile.run_prefix(), run_stamp));
    let mut root = serde_yaml::Mapping::new();
    root.insert(
        serde_yaml::Value::String("global".to_string()),
        strategy_config_to_yaml_value(base_cfg),
    );
    let header = format!(
        "# Generated by aiq-factory at {}\n# Source: {}\n",
        Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        base_config_path.display()
    );
    let yaml = serde_yaml::to_string(&serde_yaml::Value::Mapping(root))
        .context("serialise effective factory config")?;
    fs::write(&path, format!("{header}{yaml}"))
        .with_context(|| format!("write effective config {}", path.display()))?;
    Ok(path)
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

fn run_sweep(
    paths: &ResolvedPaths,
    profile: &FactoryProfileSettings,
    config_path: &Path,
    db_window: &ResolvedDbWindow,
    output_path: &Path,
    stderr_path: &Path,
    stdout_path: &Path,
) -> Result<()> {
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
        .arg(profile.initial_balance.to_string())
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
    let generated_cfg = generate_candidate_config(
        base_cfg,
        &shortlisted.sweep.overrides,
        &candidate_config_path,
        &candidate_name,
        shortlisted.shortlist_mode,
        shortlisted.rank,
        &shortlisted.sweep,
    )?;
    let config_id = strategy_config_fingerprint_sha256(&generated_cfg);
    let config_sha256 = file_sha256_hex(&candidate_config_path)?;
    let replay_dir = run_dir.join("replays");
    let replay_path = replay_dir.join(format!("{candidate_name}.replay.json"));
    let replay_report = run_replay(ReplayCommandInput {
        paths,
        config_path: &candidate_config_path,
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
        config_path: &candidate_config_path,
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
        &candidate_config_path,
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
        config_path: candidate_config_path.display().to_string(),
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

fn stable_override_key(overrides: &BTreeMap<String, f64>) -> String {
    overrides
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn generate_candidate_config(
    base_cfg: &StrategyConfig,
    overrides: &BTreeMap<String, f64>,
    output_path: &Path,
    candidate_name: &str,
    mode: ShortlistMode,
    rank: usize,
    sweep: &SweepCandidateRow,
) -> Result<StrategyConfig> {
    let mut cfg = base_cfg.clone();
    for (path, value) in overrides {
        apply_one_pub(&mut cfg, path, *value);
    }
    let mut root = serde_yaml::Mapping::new();
    root.insert(
        serde_yaml::Value::String("global".to_string()),
        strategy_config_to_yaml_value(&cfg),
    );
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
    let body = serde_yaml::to_string(&serde_yaml::Value::Mapping(root))
        .context("serialise candidate config YAML")?;
    fs::write(output_path, format!("{header}{body}"))
        .with_context(|| format!("write {}", output_path.display()))?;
    Ok(cfg)
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
        deployments,
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
    let deployment_failed = !selection_warnings.is_empty();
    let deploy_stage = if blocked {
        "blocked"
    } else if deployment_failed {
        "failed"
    } else if deployments.is_empty() {
        "pending"
    } else {
        "paper_applied"
    };
    Ok(SelectionReport {
        version: "factory_cycle_selection_v2",
        run_id: run_id.to_string(),
        run_dir: run_dir.display().to_string(),
        interval: base_cfg.engine.interval.clone(),
        deploy_stage: deploy_stage.to_string(),
        deployed: !blocked && !deployment_failed && !deployments.is_empty(),
        deployments,
        promotion_stage: if blocked || deployment_failed {
            "blocked"
        } else {
            "paper_applied"
        }
        .to_string(),
        selection_policy: DEFAULT_SELECTION_POLICY,
        selection_stage: if blocked || deployment_failed {
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
    run_dir: &Path,
    validated: &[ValidationItem],
    selected: &[SelectionCandidate],
    selection: &SelectionSettings,
    deployment: &DeploymentSettings,
) -> Result<Vec<DeploymentEvent>> {
    let mut plans = Vec::new();
    for role in selected {
        let target = selection
            .paper_targets
            .iter()
            .find(|item| item.role == role.role)
            .ok_or_else(|| anyhow!("missing paper target for role={}", role.role))?;
        let source_item = validated
            .iter()
            .find(|item| item.config_id == role.config_id)
            .ok_or_else(|| anyhow!("missing selected candidate for role={}", role.role))?;
        let config_path = PathBuf::from(&source_item.config_path);
        let promoted_config_path = run_dir
            .join("promoted_configs")
            .join(format!("{}.yaml", target.role));
        fs::copy(&config_path, &promoted_config_path).with_context(|| {
            format!(
                "copy promoted config {} -> {}",
                config_path.display(),
                promoted_config_path.display()
            )
        })?;
        plans.push(PreparedDeployment {
            role: target.role.clone(),
            service: target.service.clone(),
            source_config_path: config_path,
            promoted_config_path,
            target_yaml_path: resolve_under_project(project_dir, &target.yaml_path),
        });
    }
    if !deployment.apply_to_paper {
        return Ok(plans
            .into_iter()
            .map(|plan| DeploymentEvent {
                role: plan.role,
                source_config_path: plan.source_config_path.display().to_string(),
                promoted_config_path: plan.promoted_config_path.display().to_string(),
                target_yaml_path: plan.target_yaml_path.display().to_string(),
                service: plan.service,
                restarted_service: false,
                status: "staged_only".to_string(),
            })
            .collect());
    }

    let snapshots = plans
        .iter()
        .map(|plan| snapshot_target(plan.target_yaml_path.as_path()))
        .collect::<Result<Vec<_>>>()?;

    let deployment_result: Result<Vec<DeploymentEvent>> = (|| {
        for plan in &plans {
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
            for plan in &plans {
                restart_user_service(plan.service.as_str())?;
                restarted.insert(plan.service.clone());
            }
        }

        Ok(plans
            .iter()
            .map(|plan| DeploymentEvent {
                role: plan.role.clone(),
                source_config_path: plan.source_config_path.display().to_string(),
                promoted_config_path: plan.promoted_config_path.display().to_string(),
                target_yaml_path: plan.target_yaml_path.display().to_string(),
                service: plan.service.clone(),
                restarted_service: restarted.contains(&plan.service),
                status: "applied".to_string(),
            })
            .collect())
    })();

    match deployment_result {
        Ok(events) => Ok(events),
        Err(err) => {
            rollback_target_snapshots(&snapshots)?;
            if deployment.restart_services {
                restart_user_services(plans.iter().map(|plan| plan.service.as_str()).collect())?;
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
            initial_balance: profile_settings.initial_balance,
            candles_db_dir: paths.candles_db_dir.display().to_string(),
            funding_db: paths.funding_db_path.display().to_string(),
            start_ts_ms: db_window.start_ts_ms,
            end_ts_ms: db_window.end_ts_ms,
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
        "- Run: `{}`\n- Selection stage: `{}`\n- Gate status: `{}`\n\n",
        report.run_id, report.selection_stage, report.step5_gate_status
    ));
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
    let unit = if service.ends_with(".service") {
        service.to_string()
    } else {
        format!("{service}.service")
    };
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

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
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
}
