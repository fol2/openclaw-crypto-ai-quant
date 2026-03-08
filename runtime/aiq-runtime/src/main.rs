use aiq_runtime_core::paper::restore_paper_state;
use aiq_runtime_core::runtime::{build_bootstrap, RuntimeMode};
use aiq_runtime_core::snapshot::{load_snapshot, snapshot_to_pretty_json};
use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use std::ffi::OsString;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
mod live_cycle;
#[allow(dead_code)]
mod live_daemon;
mod live_hyperliquid;
mod live_lane;
mod live_manifest;
#[allow(dead_code)]
mod live_oms;
#[allow(dead_code)]
mod live_risk;
mod live_safety;
mod live_secrets;
#[allow(dead_code)]
mod live_state;
mod paper_config;
mod paper_cycle;
mod paper_daemon;
mod paper_export;
mod paper_lane;
mod paper_loop;
mod paper_manifest;
mod paper_run_once;
mod paper_seed;
mod paper_service;
mod paper_status;
#[cfg(test)]
mod test_support;

#[derive(Debug, Parser)]
#[command(author, version, about = "Rust runtime foundation for AI Quant")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Resolve and print the active pipeline profile.
    Pipeline(CommonArgs),
    /// Run a non-mutating runtime bootstrap doctor check.
    Doctor(CommonArgs),
    /// Validate or export runtime continuation snapshots.
    Snapshot {
        #[command(subcommand)]
        command: SnapshotCommand,
    },
    /// Bootstrap and inspect Rust-owned paper runtime state without executing orders.
    Paper {
        #[command(subcommand)]
        command: PaperCommand,
    },
    /// Resolve Rust-owned live control-plane config surfaces without executing orders.
    Live {
        #[command(subcommand)]
        command: LiveCommand,
    },
}

#[derive(Debug, Clone, Args)]
struct CommonArgs {
    /// YAML config path. Backward-compatible with strategy_overrides.yaml.
    #[arg(long, default_value = "config/strategy_overrides.yaml")]
    config: PathBuf,
    /// Optional symbol override for per-symbol config resolution.
    #[arg(long)]
    symbol: Option<String>,
    /// Apply the live overlay when loading config.
    #[arg(long)]
    live: bool,
    /// Override the runtime pipeline profile.
    #[arg(long)]
    profile: Option<String>,
    /// Runtime surface being prepared.
    #[arg(long, value_enum, default_value = "doctor")]
    mode: ModeArg,
    /// Emit machine-readable JSON instead of a human summary.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Args)]
struct PaperCommonArgs {
    /// YAML config path. Backward-compatible with strategy_overrides.yaml.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Conventional paper lane preset. Resolves paper1/paper2/paper3/livepaper defaults in Rust.
    #[arg(long, value_enum)]
    lane: Option<PaperLaneArg>,
    /// Optional project/worktree root for lane-default paths. Falls back to the current working directory.
    #[arg(long)]
    project_dir: Option<PathBuf>,
    /// Apply the live overlay when loading config.
    #[arg(long)]
    live: bool,
    /// Override the runtime pipeline profile.
    #[arg(long)]
    profile: Option<String>,
    /// Emit machine-readable JSON instead of a human summary.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Args)]
struct PaperDoctorCommonArgs {
    #[command(flatten)]
    paper: PaperCommonArgs,
    /// Optional symbol override for per-symbol config resolution.
    #[arg(long)]
    symbol: Option<String>,
}

#[derive(Debug, Clone, Args)]
struct PaperEffectiveConfigArgs {
    /// Optional YAML config path override. Falls back to AI_QUANT_STRATEGY_YAML or strategy_overrides.yaml.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Conventional paper lane preset. Resolves paper1/paper2/paper3/livepaper defaults in Rust.
    #[arg(long, value_enum)]
    lane: Option<PaperLaneArg>,
    /// Optional project/worktree root for lane-default paths. Falls back to the current working directory.
    #[arg(long)]
    project_dir: Option<PathBuf>,
    /// Apply the live overlay when loading config.
    #[arg(long)]
    live: bool,
    /// Optional symbol override for per-symbol config resolution.
    #[arg(long)]
    symbol: Option<String>,
    /// Emit machine-readable JSON instead of a human summary.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Args)]
struct LiveEffectiveConfigArgs {
    /// Optional YAML config path override. Falls back to AI_QUANT_STRATEGY_YAML or strategy_overrides.yaml.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Optional symbol override for per-symbol config resolution.
    #[arg(long)]
    symbol: Option<String>,
    /// Emit machine-readable JSON instead of a human summary.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Args)]
struct LiveManifestArgs {
    /// Optional YAML config path override. Falls back to the conventional live config path.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Optional project/worktree root for live-default paths. Falls back to the current working directory.
    #[arg(long)]
    project_dir: Option<PathBuf>,
    /// Override the runtime pipeline profile.
    #[arg(long)]
    profile: Option<String>,
    /// Optional live DB override. Falls back to AI_QUANT_DB_PATH or the conventional live DB path.
    #[arg(long)]
    db: Option<PathBuf>,
    /// Optional market DB override. Falls back to AI_QUANT_MARKET_DB_PATH or the conventional live market DB path.
    #[arg(long)]
    market_db: Option<PathBuf>,
    /// Optional daemon lock path override. Falls back to AI_QUANT_LOCK_PATH or the conventional live lock path.
    #[arg(long)]
    lock_path: Option<PathBuf>,
    /// Optional daemon status path override. Falls back to AI_QUANT_STATUS_PATH or the lock-derived default.
    #[arg(long)]
    status_path: Option<PathBuf>,
    /// Optional lookback override. Falls back to AI_QUANT_LOOKBACK_BARS or 200.
    #[arg(long)]
    lookback_bars: Option<usize>,
    /// Emit machine-readable JSON instead of a human summary.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Args)]
struct LiveDaemonArgs {
    /// Optional YAML config path override. Falls back to the conventional live config path.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Optional project/worktree root for live-default paths. Falls back to the current working directory.
    #[arg(long)]
    project_dir: Option<PathBuf>,
    /// Override the runtime pipeline profile.
    #[arg(long)]
    profile: Option<String>,
    /// Optional live DB override. Falls back to AI_QUANT_DB_PATH or the conventional live DB path.
    #[arg(long)]
    db: Option<PathBuf>,
    /// Optional candle DB override. Falls back to AI_QUANT_CANDLES_DB_DIR + the resolved interval.
    #[arg(long)]
    candles_db: Option<PathBuf>,
    /// Explicit symbol list override (comma-delimited). Falls back to AI_QUANT_SYMBOLS when empty.
    #[arg(long, value_delimiter = ',')]
    symbols: Vec<String>,
    /// Optional file containing one symbol per line. Loaded on each daemon iteration.
    #[arg(long)]
    symbols_file: Option<PathBuf>,
    /// BTC anchor symbol for alignment context.
    #[arg(long, default_value = "BTC")]
    btc_symbol: String,
    /// Optional lookback override. Falls back to AI_QUANT_LOOKBACK_BARS or 400.
    #[arg(long)]
    lookback_bars: Option<usize>,
    /// Optional live secrets path. Falls back to AI_QUANT_SECRETS_PATH or ~/.config/openclaw/ai-quant-secrets.json.
    #[arg(long)]
    secrets_path: Option<PathBuf>,
    /// Optional daemon lock path override. Falls back to AI_QUANT_LOCK_PATH or the conventional live lock path.
    #[arg(long)]
    lock_path: Option<PathBuf>,
    /// Optional daemon status path override. Falls back to AI_QUANT_STATUS_PATH or the lock-derived default.
    #[arg(long)]
    status_path: Option<PathBuf>,
    /// Sleep duration between live daemon polls.
    #[arg(long, default_value_t = 5_000)]
    idle_sleep_ms: u64,
    /// Maximum number of idle polls before exiting. Zero means unbounded.
    #[arg(long, default_value_t = 0)]
    max_idle_polls: usize,
    /// Emit machine-readable JSON instead of a human summary.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModeArg {
    Live,
    Paper,
    Replay,
    Backtest,
    SweepCpu,
    Doctor,
    Migrate,
}

#[derive(Debug, Subcommand)]
enum SnapshotCommand {
    /// Validate an init-state snapshot and confirm bt-core compatibility.
    Validate {
        #[arg(long)]
        path: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Export a v2 paper snapshot from a trading DB.
    ExportPaper {
        #[arg(long, default_value = "trading_engine.db")]
        db: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        exported_at_ms: Option<i64>,
        #[arg(long)]
        json: bool,
    },
    /// Seed a paper DB from a v2 snapshot using the Rust continuation path.
    SeedPaper {
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long, default_value = "trading_engine.db")]
        target_db: PathBuf,
        #[arg(long)]
        strict_replace: bool,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
enum PaperCommand {
    /// Resolve the shared Rust effective-config contract for paper control-plane consumers.
    EffectiveConfig(PaperEffectiveConfigArgs),
    /// Resolve the Rust paper daemon service/env contract without executing any steps.
    Manifest(PaperManifestArgs),
    /// Resolve the current Rust paper daemon service state from the launch contract plus status file.
    Status(PaperStatusArgs),
    /// Resolve or apply the Rust paper daemon supervisor contract.
    Service {
        #[command(subcommand)]
        command: PaperServiceCommand,
    },
    /// Restore paper state from the DB through the Rust snapshot/bootstrap path.
    Doctor(PaperDoctorArgs),
    /// Execute one Rust paper step for a single symbol.
    RunOnce(PaperRunOnceArgs),
    /// Execute one repeatable Rust paper cycle across explicit symbols plus open paper positions.
    Cycle(PaperCycleArgs),
    /// Execute a bounded Rust paper catch-up loop across unapplied cycle steps.
    Loop(PaperLoopArgs),
    /// Execute an opt-in long-running Rust paper daemon with scheduler-owned watchlist orchestration.
    Daemon(PaperDaemonArgs),
}

#[derive(Debug, Subcommand)]
enum LiveCommand {
    /// Resolve the shared Rust effective-config contract for live-facing control-plane consumers.
    EffectiveConfig(LiveEffectiveConfigArgs),
    /// Resolve the current live launch contract and safety-gate state.
    Manifest(LiveManifestArgs),
    /// Execute the Rust live daemon owner for the production live lane.
    Daemon(LiveDaemonArgs),
}

#[derive(Debug, Clone, Args)]
struct PaperManifestArgs {
    /// Optional YAML config path override. Falls back to AI_QUANT_STRATEGY_YAML or strategy_overrides.yaml.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Conventional paper lane preset. Resolves paper1/paper2/paper3/livepaper defaults in Rust.
    #[arg(long, value_enum)]
    lane: Option<PaperLaneArg>,
    /// Optional project/worktree root for lane-default paths. Falls back to the current working directory.
    #[arg(long)]
    project_dir: Option<PathBuf>,
    /// Apply the live overlay when loading config.
    #[arg(long)]
    live: bool,
    /// Override the runtime pipeline profile.
    #[arg(long)]
    profile: Option<String>,
    /// Optional paper DB override. Falls back to AI_QUANT_DB_PATH or trading_engine.db.
    #[arg(long)]
    db: Option<PathBuf>,
    /// Optional candle DB override. Falls back to AI_QUANT_CANDLES_DB_PATH or AI_QUANT_CANDLES_DB_DIR + interval.
    #[arg(long)]
    candles_db: Option<PathBuf>,
    /// Explicit symbol list override (comma-delimited). Falls back to AI_QUANT_SYMBOLS when empty.
    #[arg(long, value_delimiter = ',')]
    symbols: Vec<String>,
    /// Optional file containing one symbol per line. Falls back to AI_QUANT_SYMBOLS_FILE.
    #[arg(long)]
    symbols_file: Option<PathBuf>,
    /// Reload the symbols file when its metadata changes, without restarting the daemon.
    #[arg(long, requires = "symbols_file")]
    watch_symbols_file: bool,
    /// BTC anchor symbol for alignment context.
    #[arg(long, default_value = "BTC")]
    btc_symbol: String,
    /// Optional lookback override. Falls back to AI_QUANT_LOOKBACK_BARS or 400.
    #[arg(long)]
    lookback_bars: Option<usize>,
    /// Optional bootstrap step identity override. Falls back to AI_QUANT_PAPER_START_STEP_CLOSE_TS_MS.
    #[arg(long)]
    start_step_close_ts_ms: Option<i64>,
    /// Derive the first bootstrap step from the latest common candle close when no prior Rust steps exist.
    #[arg(long)]
    bootstrap_from_latest_common_close: bool,
    /// Optional daemon lock path override. Falls back to AI_QUANT_LOCK_PATH or the default paper lock.
    #[arg(long)]
    lock_path: Option<PathBuf>,
    /// Optional daemon status path override. Falls back to AI_QUANT_STATUS_PATH or the lock-derived default.
    #[arg(long)]
    status_path: Option<PathBuf>,
    /// Emit machine-readable JSON instead of a human summary.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Args)]
struct PaperStatusArgs {
    #[command(flatten)]
    manifest: PaperManifestArgs,
    /// Optional staleness threshold for the daemon status file in milliseconds.
    #[arg(long)]
    stale_after_ms: Option<i64>,
}

#[derive(Debug, Clone, Args)]
struct PaperServiceArgs {
    #[command(flatten)]
    status: PaperStatusArgs,
}

#[derive(Debug, Clone, Args)]
struct PaperServiceApplyArgs {
    #[command(flatten)]
    service: PaperServiceArgs,
    /// Requested supervisor action. `auto` reuses the read-only paper service recommendation.
    #[arg(long, value_enum, default_value = "auto")]
    action: PaperServiceApplyActionArg,
    /// Maximum time to wait for a newly spawned daemon to publish a running status contract.
    #[arg(long, default_value_t = 5_000)]
    start_wait_ms: u64,
    /// Maximum time to wait for a supervised stop before failing closed.
    #[arg(long, default_value_t = 30_000)]
    stop_wait_ms: u64,
    /// Poll interval for status/lock checks while supervising a lane.
    #[arg(long, default_value_t = 100)]
    poll_ms: u64,
}

#[derive(Debug, Parser)]
struct PaperServiceInspectCompatCli {
    #[command(flatten)]
    args: PaperServiceArgs,
}

#[derive(Debug, Clone, Subcommand)]
enum PaperServiceCommand {
    /// Resolve the current Rust paper daemon supervisor action without mutating runtime state.
    Inspect(PaperServiceArgs),
    /// Apply the current Rust paper daemon supervisor action via an opt-in side-effecting supervisor.
    Apply(PaperServiceApplyArgs),
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PaperServiceApplyActionArg {
    Auto,
    Start,
    Restart,
    Stop,
    Resume,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PaperLaneArg {
    Paper1,
    Paper2,
    Paper3,
    Livepaper,
}

#[derive(Debug, Clone, Args)]
struct PaperDoctorArgs {
    #[command(flatten)]
    common: PaperDoctorCommonArgs,
    /// Paper DB path to inspect and restore.
    #[arg(long, default_value = "trading_engine.db")]
    db: PathBuf,
    /// Override exported_at_ms for reproducible bootstrap reports.
    #[arg(long)]
    exported_at_ms: Option<i64>,
}

#[derive(Debug, Clone, Args)]
struct PaperRunOnceArgs {
    #[command(flatten)]
    common: PaperCommonArgs,
    /// Paper DB path to restore from and project back into.
    #[arg(long, default_value = "trading_engine.db")]
    db: PathBuf,
    /// Candle SQLite DB path used for this one-shot step.
    #[arg(long)]
    candles_db: PathBuf,
    /// Target symbol for the one-shot decision/execution step.
    #[arg(long)]
    target_symbol: String,
    /// BTC anchor symbol for alignment context.
    #[arg(long, default_value = "BTC")]
    btc_symbol: String,
    /// Number of bars to load for indicator warm-up.
    #[arg(long, default_value_t = 400)]
    lookback_bars: usize,
    /// Override exported_at_ms for reproducible reports and write timestamps.
    #[arg(long)]
    exported_at_ms: Option<i64>,
    /// Resolve the step but do not write any DB projections.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Clone, Args)]
struct PaperCycleArgs {
    #[command(flatten)]
    common: PaperCommonArgs,
    /// Paper DB path to restore from and project back into.
    #[arg(long, default_value = "trading_engine.db")]
    db: PathBuf,
    /// Candle SQLite DB path used for this cycle.
    #[arg(long)]
    candles_db: PathBuf,
    /// Explicit symbol list (comma-delimited). Open paper positions are always included.
    #[arg(long, value_delimiter = ',')]
    symbols: Vec<String>,
    /// Optional file containing one symbol per line.
    #[arg(long)]
    symbols_file: Option<PathBuf>,
    /// BTC anchor symbol for alignment context.
    #[arg(long, default_value = "BTC")]
    btc_symbol: String,
    /// Number of bars to load for indicator warm-up.
    #[arg(long, default_value_t = 400)]
    lookback_bars: usize,
    /// Explicit repeatable cycle step identity (bar close timestamp in ms).
    #[arg(long)]
    step_close_ts_ms: i64,
    /// Override exported_at_ms for reproducible artefacts.
    #[arg(long)]
    exported_at_ms: Option<i64>,
    /// Resolve the cycle but do not write any DB projections.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Clone, Args)]
struct PaperLoopArgs {
    #[command(flatten)]
    common: PaperCommonArgs,
    /// Paper DB path to restore from and project back into.
    #[arg(long, default_value = "trading_engine.db")]
    db: PathBuf,
    /// Candle SQLite DB path used for this loop.
    #[arg(long)]
    candles_db: PathBuf,
    /// Explicit symbol list (comma-delimited). Open paper positions are always included.
    #[arg(long, value_delimiter = ',')]
    symbols: Vec<String>,
    /// Optional file containing one symbol per line. Loaded once at start-up.
    #[arg(long)]
    symbols_file: Option<PathBuf>,
    /// BTC anchor symbol for alignment context.
    #[arg(long, default_value = "BTC")]
    btc_symbol: String,
    /// Number of bars to load for indicator warm-up.
    #[arg(long, default_value_t = 400)]
    lookback_bars: usize,
    /// Required bootstrap step identity when no prior runtime_cycle_steps exist.
    #[arg(long)]
    start_step_close_ts_ms: Option<i64>,
    /// Maximum number of unapplied cycle steps to execute before exiting.
    #[arg(long, default_value_t = 1)]
    max_steps: usize,
    /// Keep polling for the next due step instead of exiting idle after catch-up.
    #[arg(long)]
    follow: bool,
    /// Sleep duration between follow-mode idle polls.
    #[arg(long, default_value_t = 5_000)]
    idle_sleep_ms: u64,
    /// Maximum number of follow-mode idle polls before exiting. Zero means unlimited.
    #[arg(long, default_value_t = 0)]
    max_idle_polls: usize,
    /// Override exported_at_ms for reproducible artefacts; defaults to each step close.
    #[arg(long)]
    exported_at_ms: Option<i64>,
    /// Resolve the loop but do not write any DB projections.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Clone, Args)]
struct PaperDaemonArgs {
    #[command(flatten)]
    common: PaperCommonArgs,
    /// Paper DB path to restore from and project back into.
    #[arg(long)]
    db: Option<PathBuf>,
    /// Candle SQLite DB path used for this daemon lane.
    #[arg(long)]
    candles_db: Option<PathBuf>,
    /// Explicit symbol list (comma-delimited). Open paper positions are always included.
    #[arg(long, value_delimiter = ',')]
    symbols: Vec<String>,
    /// Optional file containing one symbol per line. Loaded once at start-up unless watch mode is enabled.
    #[arg(long)]
    symbols_file: Option<PathBuf>,
    /// Reload the symbols file when its metadata changes, without restarting the daemon.
    #[arg(long, requires = "symbols_file")]
    watch_symbols_file: bool,
    /// BTC anchor symbol for alignment context.
    #[arg(long, default_value = "BTC")]
    btc_symbol: String,
    /// Number of bars to load for indicator warm-up.
    #[arg(long)]
    lookback_bars: Option<usize>,
    /// Required bootstrap step identity when no prior runtime_cycle_steps exist.
    #[arg(long)]
    start_step_close_ts_ms: Option<i64>,
    /// Derive the first bootstrap step from the latest common candle close when no prior Rust steps exist.
    #[arg(long)]
    bootstrap_from_latest_common_close: bool,
    /// Sleep duration between idle follow polls.
    #[arg(long, default_value_t = 5_000)]
    idle_sleep_ms: u64,
    /// Maximum number of idle polls before exiting. Zero means unbounded.
    #[arg(long, default_value_t = 0)]
    max_idle_polls: usize,
    /// Override exported_at_ms for reproducible artefacts; defaults to each step close.
    #[arg(long)]
    exported_at_ms: Option<i64>,
    /// Resolve the daemon lane but do not write any DB projections.
    #[arg(long)]
    dry_run: bool,
    /// Optional daemon lock path. Defaults to AI_QUANT_LOCK_PATH or the project paper/live lock file.
    #[arg(long)]
    lock_path: Option<PathBuf>,
    /// Optional daemon status path. Defaults to AI_QUANT_STATUS_PATH or the resolved lock-derived status file.
    #[arg(long)]
    status_path: Option<PathBuf>,
}

impl From<ModeArg> for RuntimeMode {
    fn from(value: ModeArg) -> Self {
        match value {
            ModeArg::Live => RuntimeMode::Live,
            ModeArg::Paper => RuntimeMode::Paper,
            ModeArg::Replay => RuntimeMode::Replay,
            ModeArg::Backtest => RuntimeMode::Backtest,
            ModeArg::SweepCpu => RuntimeMode::SweepCpu,
            ModeArg::Doctor => RuntimeMode::Doctor,
            ModeArg::Migrate => RuntimeMode::Migrate,
        }
    }
}

impl From<PaperLaneArg> for paper_lane::PaperLane {
    fn from(value: PaperLaneArg) -> Self {
        match value {
            PaperLaneArg::Paper1 => Self::Paper1,
            PaperLaneArg::Paper2 => Self::Paper2,
            PaperLaneArg::Paper3 => Self::Paper3,
            PaperLaneArg::Livepaper => Self::Livepaper,
        }
    }
}

fn main() -> Result<()> {
    let raw_args = std::env::args_os().collect::<Vec<_>>();
    if maybe_print_legacy_paper_service_help(&raw_args)? {
        return Ok(());
    }
    let cli = Cli::parse_from(preprocess_cli_args(raw_args));

    match cli.command {
        Command::Pipeline(args) | Command::Doctor(args) => run_bootstrap(args),
        Command::Snapshot { command } => run_snapshot(command),
        Command::Paper { command } => run_paper(command),
        Command::Live { command } => run_live(command),
    }
}

fn maybe_print_legacy_paper_service_help(args: &[OsString]) -> Result<bool> {
    let Some(compat_args) = legacy_paper_service_help_args(args) else {
        return Ok(false);
    };

    if let Err(err) = PaperServiceInspectCompatCli::try_parse_from(compat_args) {
        err.print()
            .context("failed to print the legacy paper service help error")?;
        std::process::exit(err.exit_code());
    }

    let mut command = PaperServiceInspectCompatCli::command();
    command = command
        .name("aiq-runtime paper service")
        .bin_name("aiq-runtime paper service")
        .about("Resolve the current Rust paper daemon supervisor action without mutating runtime state");
    command
        .print_help()
        .context("failed to print legacy paper service help")?;
    println!();
    Ok(true)
}

fn legacy_paper_service_help_args(args: &[OsString]) -> Option<Vec<OsString>> {
    if args.get(1).and_then(|value| value.to_str()) != Some("paper") {
        return None;
    }
    if args.get(2).and_then(|value| value.to_str()) != Some("service") {
        return None;
    }
    let tail = &args[3..];
    if matches!(
        tail.first().and_then(|value| value.to_str()),
        Some("inspect" | "apply")
    ) {
        return None;
    }
    if !tail
        .iter()
        .any(|value| matches!(value.to_str(), Some("-h" | "--help")))
    {
        return None;
    }

    let mut compat_args = vec![OsString::from("aiq-runtime paper service")];
    for token in tail {
        if matches!(token.to_str(), Some("-h" | "--help")) {
            continue;
        }
        compat_args.push(token.clone());
    }
    Some(compat_args)
}

fn preprocess_cli_args(mut args: Vec<OsString>) -> Vec<OsString> {
    if should_insert_paper_service_inspect(&args) {
        args.insert(3, OsString::from("inspect"));
    }
    args
}

fn should_insert_paper_service_inspect(args: &[OsString]) -> bool {
    if args.get(1).and_then(|value| value.to_str()) != Some("paper") {
        return false;
    }
    if args.get(2).and_then(|value| value.to_str()) != Some("service") {
        return false;
    }
    match args.get(3).and_then(|value| value.to_str()) {
        None => true,
        Some("inspect" | "apply") => false,
        Some(_) => true,
    }
}

fn run_bootstrap(args: CommonArgs) -> Result<()> {
    let config_path = resolve_config_path(&args.config);
    let config = bt_core::config::load_config_checked(
        config_path
            .to_str()
            .context("config path must be valid UTF-8")?,
        args.symbol.as_deref(),
        args.live,
    )
    .map_err(anyhow::Error::msg)?;

    let bootstrap = build_bootstrap(&config, args.mode.into(), args.profile.as_deref())
        .map_err(anyhow::Error::msg)?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&bootstrap)?);
    } else {
        println!("mode: {:?}", bootstrap.mode);
        println!("profile: {}", bootstrap.pipeline.profile);
        println!("ranker: {}", bootstrap.pipeline.ranker);
        println!("state_backend: {}", bootstrap.pipeline.state_backend);
        println!("audit_sink: {}", bootstrap.pipeline.audit_sink);
        println!("config_fingerprint: {}", bootstrap.config_fingerprint);
        println!("stages:");
        for stage in &bootstrap.pipeline.stages {
            println!(
                "  - {} [{}]",
                stage.id,
                if stage.enabled { "enabled" } else { "disabled" }
            );
        }
    }

    Ok(())
}

fn resolve_config_path(path: &Path) -> PathBuf {
    if path.exists() {
        return path.to_path_buf();
    }

    let fallback = PathBuf::from(format!("{}.example", path.display()));
    if fallback.exists() {
        return fallback;
    }

    path.to_path_buf()
}

fn load_symbols(symbols: Vec<String>, symbols_file: Option<&Path>) -> Result<Vec<String>> {
    let mut merged = symbols;
    if let Some(symbols_file) = symbols_file {
        let file_symbols = std::fs::read_to_string(symbols_file)?
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        merged.extend(file_symbols);
    }
    Ok(merged)
}

fn bootstrap_symbol_hint(symbols: &[String]) -> Option<&str> {
    (symbols.len() == 1).then(|| symbols[0].as_str())
}

fn default_live_secrets_path() -> PathBuf {
    std::env::var("AI_QUANT_SECRETS_PATH")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("~/.config/openclaw/ai-quant-secrets.json"))
}

fn resolve_live_candles_db(candles_db: Option<&Path>, interval: &str) -> PathBuf {
    if let Some(candles_db) = candles_db {
        return candles_db.to_path_buf();
    }
    if let Ok(path) = std::env::var("AI_QUANT_CANDLES_DB_PATH") {
        if !path.trim().is_empty() {
            return PathBuf::from(path);
        }
    }
    let candles_dir = std::env::var("AI_QUANT_CANDLES_DB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("candles_dbs"));
    candles_dir.join(format!("candles_{interval}.db"))
}

fn run_snapshot(command: SnapshotCommand) -> Result<()> {
    match command {
        SnapshotCommand::Validate { path, json } => {
            let snapshot = load_snapshot(&path)?;
            bt_core::init_state::load(path.to_str().context("snapshot path must be valid UTF-8")?)
                .map_err(anyhow::Error::msg)?;

            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "ok": true,
                        "summary": snapshot.summary(),
                        "bt_core_compatible": true,
                    }))?
                );
            } else {
                let summary = snapshot.summary();
                println!(
                    "snapshot ok: version={} source={} positions={} runtime_entry_markers={} runtime_exit_markers={}",
                    summary.version,
                    summary.source,
                    summary.position_count,
                    summary.runtime_entry_markers,
                    summary.runtime_exit_markers,
                );
            }
        }
        SnapshotCommand::ExportPaper {
            db,
            output,
            exported_at_ms,
            json,
        } => {
            let snapshot = paper_export::export_paper_snapshot(
                &db,
                exported_at_ms.unwrap_or_else(|| Utc::now().timestamp_millis()),
            )?;
            snapshot.validate()?;

            if let Some(path) = output.as_ref() {
                std::fs::write(&path, snapshot_to_pretty_json(&snapshot)?)?;
            }

            if json || output.is_none() {
                println!("{}", snapshot_to_pretty_json(&snapshot)?);
            } else {
                println!(
                    "paper snapshot exported: positions={} output={}",
                    snapshot.positions.len(),
                    output.as_ref().unwrap().display(),
                );
            }
        }
        SnapshotCommand::SeedPaper {
            snapshot,
            target_db,
            strict_replace,
            json,
        } => {
            let snapshot = load_snapshot(&snapshot)?;
            let report = paper_seed::seed_paper_db(
                &snapshot,
                &target_db,
                paper_seed::SeedOptions { strict_replace },
            )?;

            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "paper snapshot seeded: trades={} positions={} runtime_cooldowns={} target_db={} strict_replace={}",
                    report.seeded_trades,
                    report.seeded_positions,
                    report.seeded_runtime_cooldowns,
                    report.target_db,
                    report.strict_replace,
                );
            }
        }
    }

    Ok(())
}

fn run_paper(command: PaperCommand) -> Result<()> {
    match command {
        PaperCommand::EffectiveConfig(args) => {
            let effective_config = paper_config::PaperEffectiveConfig::resolve(
                args.config.as_deref(),
                args.lane.map(Into::into),
                args.project_dir.as_deref(),
            )?;
            let report = effective_config.build_report(args.symbol.as_deref(), args.live)?;

            if args.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_effective_config_report("paper", &report);
            }
        }
        PaperCommand::Manifest(args) => {
            let report = paper_manifest::build_manifest(paper_manifest::PaperManifestInput {
                config: args.config.as_deref(),
                lane: args.lane.map(Into::into),
                project_dir: args.project_dir.as_deref(),
                live: args.live,
                profile: args.profile.as_deref(),
                db: args.db.as_deref(),
                candles_db: args.candles_db.as_deref(),
                symbols: &args.symbols,
                symbols_file: args.symbols_file.as_deref(),
                watch_symbols_file: args.watch_symbols_file,
                btc_symbol: &args.btc_symbol,
                lookback_bars: args.lookback_bars,
                start_step_close_ts_ms: args.start_step_close_ts_ms,
                bootstrap_from_latest_common_close: args.bootstrap_from_latest_common_close,
                lock_path: args.lock_path.as_deref(),
                status_path: args.status_path.as_deref(),
            })?;

            if args.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "paper manifest ok: profile={}",
                    report.runtime_bootstrap.pipeline.profile
                );
                println!("base_config_path: {}", report.base_config_path);
                println!("config_path: {}", report.config_path);
                println!("active_yaml_path: {}", report.active_yaml_path);
                println!("effective_yaml_path: {}", report.effective_yaml_path);
                println!("paper_db: {}", report.paper_db);
                println!("candles_db: {}", report.candles_db);
                println!("interval: {}", report.interval);
                println!("lookback_bars: {}", report.lookback_bars);
                if let Some(lane) = report.lane.as_deref() {
                    println!("lane: {}", lane);
                }
                if let Some(service_name) = report.service_name.as_deref() {
                    println!("service_name: {}", service_name);
                }
                println!("symbols: {}", report.symbols.join(","));
                if let Some(symbols_file) = report.symbols_file.as_deref() {
                    println!("symbols_file: {}", symbols_file);
                }
                println!("watch_symbols_file: {}", report.watch_symbols_file);
                if let Some(start_step_close_ts_ms) = report.start_step_close_ts_ms {
                    println!("start_step_close_ts_ms: {}", start_step_close_ts_ms);
                }
                println!(
                    "strategy_overrides_sha1: {}",
                    report.strategy_overrides_sha1
                );
                println!("config_id: {}", report.config_id);
                if let Some(promoted_role) = report.promoted_role.as_deref() {
                    println!("promoted_role: {}", promoted_role);
                }
                if let Some(promoted_config_path) = report.promoted_config_path.as_deref() {
                    println!("promoted_config_path: {}", promoted_config_path);
                }
                if let Some(strategy_mode) = report.strategy_mode.as_deref() {
                    println!("strategy_mode: {}", strategy_mode);
                }
                if let Some(strategy_mode_source) = report.strategy_mode_source.as_deref() {
                    println!("strategy_mode_source: {}", strategy_mode_source);
                }
                println!("lock_path: {}", report.lock_path);
                println!("status_path: {}", report.status_path);
                println!("launch_state: {:?}", report.resume.launch_state);
                println!("launch_ready: {}", report.resume.launch_ready);
                println!("active_symbols: {}", report.resume.active_symbols.join(","));
                if let Some(last_applied_step_close_ts_ms) =
                    report.resume.last_applied_step_close_ts_ms
                {
                    println!(
                        "last_applied_step_close_ts_ms: {}",
                        last_applied_step_close_ts_ms
                    );
                }
                if let Some(next_due_step_close_ts_ms) = report.resume.next_due_step_close_ts_ms {
                    println!("next_due_step_close_ts_ms: {}", next_due_step_close_ts_ms);
                }
                if let Some(latest_common_close_ts_ms) = report.resume.latest_common_close_ts_ms {
                    println!("latest_common_close_ts_ms: {}", latest_common_close_ts_ms);
                }
                if !report.warnings.is_empty() {
                    println!("warnings:");
                    for warning in &report.warnings {
                        println!("  - {}", warning);
                    }
                }
            }
        }
        PaperCommand::Status(args) => {
            let report = paper_status::build_status(paper_status::PaperStatusInput {
                config: args.manifest.config.as_deref(),
                lane: args.manifest.lane.map(Into::into),
                project_dir: args.manifest.project_dir.as_deref(),
                live: args.manifest.live,
                profile: args.manifest.profile.as_deref(),
                db: args.manifest.db.as_deref(),
                candles_db: args.manifest.candles_db.as_deref(),
                symbols: &args.manifest.symbols,
                symbols_file: args.manifest.symbols_file.as_deref(),
                watch_symbols_file: args.manifest.watch_symbols_file,
                btc_symbol: &args.manifest.btc_symbol,
                lookback_bars: args.manifest.lookback_bars,
                start_step_close_ts_ms: args.manifest.start_step_close_ts_ms,
                bootstrap_from_latest_common_close: args
                    .manifest
                    .bootstrap_from_latest_common_close,
                lock_path: args.manifest.lock_path.as_deref(),
                status_path: args.manifest.status_path.as_deref(),
                stale_after_ms: args.stale_after_ms,
            })?;

            if args.manifest.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("paper status ok: state={:?}", report.service_state);
                println!("status_file_present: {}", report.status_file_present);
                println!(
                    "contract_matches_status: {}",
                    report.contract_matches_status
                );
                println!("launch_state: {:?}", report.manifest.resume.launch_state);
                println!("launch_ready: {}", report.manifest.resume.launch_ready);
                println!("status_path: {}", report.manifest.status_path);
                if let Some(status_age_ms) = report.status_age_ms {
                    println!("status_age_ms: {}", status_age_ms);
                }
                if let Some(stale_after_ms) = report.stale_after_ms {
                    println!("stale_after_ms: {}", stale_after_ms);
                }
                if !report.mismatch_reasons.is_empty() {
                    println!("mismatch_reasons:");
                    for reason in &report.mismatch_reasons {
                        println!("  - {}", reason);
                    }
                }
                if !report.warnings.is_empty() {
                    println!("warnings:");
                    for warning in &report.warnings {
                        println!("  - {}", warning);
                    }
                }
            }
        }
        PaperCommand::Service { command } => match command {
            PaperServiceCommand::Inspect(args) => {
                let report = paper_service::build_service(paper_service::PaperServiceInput {
                    config: args.status.manifest.config.as_deref(),
                    lane: args.status.manifest.lane.map(Into::into),
                    project_dir: args.status.manifest.project_dir.as_deref(),
                    live: args.status.manifest.live,
                    profile: args.status.manifest.profile.as_deref(),
                    db: args.status.manifest.db.as_deref(),
                    candles_db: args.status.manifest.candles_db.as_deref(),
                    symbols: &args.status.manifest.symbols,
                    symbols_file: args.status.manifest.symbols_file.as_deref(),
                    watch_symbols_file: args.status.manifest.watch_symbols_file,
                    btc_symbol: &args.status.manifest.btc_symbol,
                    lookback_bars: args.status.manifest.lookback_bars,
                    start_step_close_ts_ms: args.status.manifest.start_step_close_ts_ms,
                    bootstrap_from_latest_common_close: args
                        .status
                        .manifest
                        .bootstrap_from_latest_common_close,
                    lock_path: args.status.manifest.lock_path.as_deref(),
                    status_path: args.status.manifest.status_path.as_deref(),
                    stale_after_ms: args.status.stale_after_ms,
                })?;

                if args.status.manifest.json {
                    println!("{}", serde_json::to_string_pretty(&report)?);
                } else {
                    println!(
                        "paper service ok: action={:?} state={:?}",
                        report.desired_action, report.status.service_state
                    );
                    println!("action_reason: {}", report.action_reason);
                    println!("status_path: {}", report.status.manifest.status_path);
                    println!("lock_path: {}", report.status.manifest.lock_path);
                    println!(
                        "contract_matches_status: {}",
                        report.status.contract_matches_status
                    );
                    println!(
                        "launch_ready: {}",
                        report.status.manifest.resume.launch_ready
                    );
                    if !report.status.mismatch_reasons.is_empty() {
                        println!("mismatch_reasons:");
                        for reason in &report.status.mismatch_reasons {
                            println!("  - {}", reason);
                        }
                    }
                    if !report.warnings.is_empty() {
                        println!("warnings:");
                        for warning in &report.warnings {
                            println!("  - {}", warning);
                        }
                    }
                }
            }
            PaperServiceCommand::Apply(args) => {
                let requested_action = match args.action {
                    PaperServiceApplyActionArg::Auto => {
                        paper_service::PaperServiceApplyRequestedAction::Auto
                    }
                    PaperServiceApplyActionArg::Start => {
                        paper_service::PaperServiceApplyRequestedAction::Start
                    }
                    PaperServiceApplyActionArg::Restart => {
                        paper_service::PaperServiceApplyRequestedAction::Restart
                    }
                    PaperServiceApplyActionArg::Stop => {
                        paper_service::PaperServiceApplyRequestedAction::Stop
                    }
                    PaperServiceApplyActionArg::Resume => {
                        paper_service::PaperServiceApplyRequestedAction::Resume
                    }
                };
                let report = paper_service::apply_service(paper_service::PaperServiceApplyInput {
                    service: paper_service::PaperServiceInput {
                        config: args.service.status.manifest.config.as_deref(),
                        lane: args.service.status.manifest.lane.map(Into::into),
                        project_dir: args.service.status.manifest.project_dir.as_deref(),
                        live: args.service.status.manifest.live,
                        profile: args.service.status.manifest.profile.as_deref(),
                        db: args.service.status.manifest.db.as_deref(),
                        candles_db: args.service.status.manifest.candles_db.as_deref(),
                        symbols: &args.service.status.manifest.symbols,
                        symbols_file: args.service.status.manifest.symbols_file.as_deref(),
                        watch_symbols_file: args.service.status.manifest.watch_symbols_file,
                        btc_symbol: &args.service.status.manifest.btc_symbol,
                        lookback_bars: args.service.status.manifest.lookback_bars,
                        start_step_close_ts_ms: args.service.status.manifest.start_step_close_ts_ms,
                        bootstrap_from_latest_common_close: args
                            .service
                            .status
                            .manifest
                            .bootstrap_from_latest_common_close,
                        lock_path: args.service.status.manifest.lock_path.as_deref(),
                        status_path: args.service.status.manifest.status_path.as_deref(),
                        stale_after_ms: args.service.status.stale_after_ms,
                    },
                    requested_action,
                    start_wait_ms: args.start_wait_ms,
                    stop_wait_ms: args.stop_wait_ms,
                    poll_ms: args.poll_ms,
                })?;

                if args.service.status.manifest.json {
                    println!("{}", serde_json::to_string_pretty(&report)?);
                } else {
                    println!(
                        "paper service apply ok: requested={:?} applied={:?} final_action={:?} final_state={:?}",
                        report.requested_action,
                        report.applied_action,
                        report.final_service.desired_action,
                        report.final_service.status.service_state
                    );
                    println!("action_reason: {}", report.action_reason);
                    if let Some(previous_pid) = report.previous_pid {
                        println!("previous_pid: {}", previous_pid);
                    }
                    if let Some(spawned_pid) = report.spawned_pid {
                        println!("spawned_pid: {}", spawned_pid);
                    }
                    println!("status_path: {}", report.final_service.status_path);
                    println!("lock_path: {}", report.final_service.lock_path);
                    if !report.final_service.warnings.is_empty() {
                        println!("warnings:");
                        for warning in &report.final_service.warnings {
                            println!("  - {}", warning);
                        }
                    }
                }
            }
        },
        PaperCommand::Doctor(args) => {
            let effective_config = paper_config::PaperEffectiveConfig::resolve(
                args.common.paper.config.as_deref(),
                args.common.paper.lane.map(Into::into),
                args.common.paper.project_dir.as_deref(),
            )?;
            let runtime_bootstrap = effective_config.build_runtime_bootstrap(
                args.common.symbol.as_deref(),
                args.common.paper.live,
                args.common.paper.profile.as_deref(),
            )?;
            let snapshot = paper_export::export_paper_snapshot(
                &args.db,
                args.exported_at_ms
                    .unwrap_or_else(|| Utc::now().timestamp_millis()),
            )?;
            let (_state, report) = restore_paper_state(&snapshot).map_err(anyhow::Error::msg)?;

            if args.common.paper.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "ok": true,
                        "runtime_bootstrap": runtime_bootstrap,
                        "paper_bootstrap": report,
                    }))?
                );
            } else {
                println!(
                    "paper doctor ok: profile={} positions={} runtime_entry_markers={} runtime_exit_markers={}",
                    runtime_bootstrap.pipeline.profile,
                    report.position_count,
                    report.runtime_entry_markers,
                    report.runtime_exit_markers,
                );
            }
        }
        PaperCommand::RunOnce(args) => {
            let effective_config = paper_config::PaperEffectiveConfig::resolve(
                args.common.config.as_deref(),
                args.common.lane.map(Into::into),
                args.common.project_dir.as_deref(),
            )?;
            let config = effective_config
                .load_config(Some(args.target_symbol.as_str()), args.common.live)?;
            let runtime_bootstrap = effective_config.build_runtime_bootstrap(
                Some(args.target_symbol.as_str()),
                args.common.live,
                args.common.profile.as_deref(),
            )?;
            let report = paper_run_once::run_once(paper_run_once::PaperRunOnceInput {
                config: &config,
                runtime_bootstrap,
                paper_db: &args.db,
                candles_db: &args.candles_db,
                symbol: &args.target_symbol,
                btc_symbol: &args.btc_symbol,
                lookback_bars: args.lookback_bars,
                exported_at_ms: args.exported_at_ms,
                dry_run: args.dry_run,
            })?;

            if args.common.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "paper run-once ok: symbol={} intents={} fills={} trades_written={} dry_run={}",
                    report.symbol,
                    report.intent_count,
                    report.fill_count,
                    report.trades_written,
                    report.dry_run,
                );
            }
        }
        PaperCommand::Cycle(args) => {
            let effective_config = paper_config::PaperEffectiveConfig::resolve(
                args.common.config.as_deref(),
                args.common.lane.map(Into::into),
                args.common.project_dir.as_deref(),
            )?;
            let symbols = load_symbols(args.symbols, args.symbols_file.as_deref())?;
            let runtime_bootstrap = effective_config.build_runtime_bootstrap(
                bootstrap_symbol_hint(&symbols),
                args.common.live,
                args.common.profile.as_deref(),
            )?;
            let report = paper_cycle::run_cycle(paper_cycle::PaperCycleInput {
                effective_config,
                runtime_bootstrap,
                live: args.common.live,
                paper_db: &args.db,
                candles_db: &args.candles_db,
                explicit_symbols: &symbols,
                btc_symbol: &args.btc_symbol,
                lookback_bars: args.lookback_bars,
                step_close_ts_ms: args.step_close_ts_ms,
                exported_at_ms: args.exported_at_ms,
                dry_run: args.dry_run,
            })?;

            if args.common.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "paper cycle ok: step_id={} symbols={} trades_written={} dry_run={}",
                    report.step_id,
                    report.active_symbols.len(),
                    report.trades_written,
                    report.dry_run,
                );
            }
        }
        PaperCommand::Loop(args) => {
            let effective_config = paper_config::PaperEffectiveConfig::resolve(
                args.common.config.as_deref(),
                args.common.lane.map(Into::into),
                args.common.project_dir.as_deref(),
            )?;
            let bootstrap_symbols =
                load_symbols(args.symbols.clone(), args.symbols_file.as_deref())?;
            let runtime_bootstrap = effective_config.build_runtime_bootstrap(
                bootstrap_symbol_hint(&bootstrap_symbols),
                args.common.live,
                args.common.profile.as_deref(),
            )?;
            let report = paper_loop::run_loop(paper_loop::PaperLoopInput {
                effective_config,
                runtime_bootstrap,
                live: args.common.live,
                paper_db: &args.db,
                candles_db: &args.candles_db,
                explicit_symbols: &args.symbols,
                symbols_file: args.symbols_file.as_deref(),
                btc_symbol: &args.btc_symbol,
                lookback_bars: args.lookback_bars,
                start_step_close_ts_ms: args.start_step_close_ts_ms,
                max_steps: args.max_steps,
                follow: args.follow,
                idle_sleep_ms: args.idle_sleep_ms,
                max_idle_polls: args.max_idle_polls,
                exported_at_ms: args.exported_at_ms,
                dry_run: args.dry_run,
                stop_flag: None,
            })?;

            if args.common.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "paper loop ok: steps={} next_due={:?} latest_common={:?} dry_run={}",
                    report.executed_steps,
                    report.next_due_step_close_ts_ms,
                    report.latest_common_close_ts_ms,
                    report.dry_run,
                );
            }
        }
        PaperCommand::Daemon(args) => {
            let manifest = paper_manifest::build_manifest(paper_manifest::PaperManifestInput {
                config: args.common.config.as_deref(),
                lane: args.common.lane.map(Into::into),
                project_dir: args.common.project_dir.as_deref(),
                live: args.common.live,
                profile: args.common.profile.as_deref(),
                db: args.db.as_deref(),
                candles_db: args.candles_db.as_deref(),
                symbols: &args.symbols,
                symbols_file: args.symbols_file.as_deref(),
                watch_symbols_file: args.watch_symbols_file,
                btc_symbol: &args.btc_symbol,
                lookback_bars: args.lookback_bars,
                start_step_close_ts_ms: args.start_step_close_ts_ms,
                bootstrap_from_latest_common_close: args.bootstrap_from_latest_common_close,
                lock_path: args.lock_path.as_deref(),
                status_path: args.status_path.as_deref(),
            })?;
            let effective_config = paper_config::PaperEffectiveConfig::resolve(
                Some(Path::new(&manifest.base_config_path)),
                args.common.lane.map(Into::into),
                args.common.project_dir.as_deref(),
            )?;
            let bootstrap_symbols = manifest.resume.active_symbols.clone();
            let runtime_bootstrap = effective_config.build_runtime_bootstrap(
                bootstrap_symbol_hint(&bootstrap_symbols),
                args.common.live,
                args.common.profile.as_deref(),
            )?;
            let report = paper_daemon::run_daemon(paper_daemon::PaperDaemonInput {
                effective_config,
                runtime_bootstrap,
                profile_override: args.common.profile.as_deref(),
                live: args.common.live,
                paper_db: Path::new(&manifest.paper_db),
                candles_db: Path::new(&manifest.candles_db),
                explicit_symbols: &manifest.symbols,
                symbols_file: manifest.symbols_file.as_deref().map(Path::new),
                btc_symbol: &args.btc_symbol,
                lookback_bars: manifest.lookback_bars,
                start_step_close_ts_ms: manifest.start_step_close_ts_ms,
                bootstrap_from_latest_common_close: args.bootstrap_from_latest_common_close,
                idle_sleep_ms: args.idle_sleep_ms,
                max_idle_polls: args.max_idle_polls,
                exported_at_ms: args.exported_at_ms,
                dry_run: args.dry_run,
                lock_path: Some(Path::new(&manifest.lock_path)),
                status_path: Some(Path::new(&manifest.status_path)),
                watch_symbols_file: manifest.watch_symbols_file,
                emit_progress: !args.common.json,
            })?;

            if args.common.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "paper daemon ok: pid={} lock={} status={} steps={} stop_requested={} dry_run={} reloads={}",
                    report.pid,
                    report.lock_path,
                    report.status_path,
                    report.loop_report.executed_steps,
                    report.stop_requested,
                    report.dry_run,
                    report.manifest_reload_count,
                );
            }
        }
    }

    Ok(())
}

fn run_live(command: LiveCommand) -> Result<()> {
    match command {
        LiveCommand::EffectiveConfig(args) => {
            let effective_config =
                paper_config::PaperEffectiveConfig::resolve(args.config.as_deref(), None, None)?;
            let report = effective_config.build_report(args.symbol.as_deref(), true)?;

            if args.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_effective_config_report("live", &report);
            }
        }
        LiveCommand::Manifest(args) => {
            let report = live_manifest::build_manifest(live_manifest::LiveManifestInput {
                config: args.config.as_deref(),
                project_dir: args.project_dir.as_deref(),
                profile: args.profile.as_deref(),
                db: args.db.as_deref(),
                market_db: args.market_db.as_deref(),
                lock_path: args.lock_path.as_deref(),
                status_path: args.status_path.as_deref(),
                lookback_bars: args.lookback_bars,
            })?;

            if args.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("live manifest ok");
                println!("runtime_owner: {}", report.runtime_owner);
                println!("launch_state: {:?}", report.launch_state);
                println!("service_name: {}", report.service_name);
                println!("instance_tag: {}", report.instance_tag);
                println!("config_path: {}", report.config_path);
                println!("live_db: {}", report.live_db);
                println!("market_db: {}", report.market_db);
                println!("lock_path: {}", report.lock_path);
                println!("status_path: {}", report.status_path);
                println!("interval: {}", report.interval);
                println!("lookback_bars: {}", report.lookback_bars);
                println!("safety_gate_ready: {}", report.safety_gate_ready);
                println!("live_enable: {}", report.live_enable);
                println!("live_confirmed: {}", report.live_confirmed);
                if !report.warnings.is_empty() {
                    println!("warnings:");
                    for warning in &report.warnings {
                        println!("  - {}", warning);
                    }
                }
            }
        }
        LiveCommand::Daemon(args) => {
            let report = live_manifest::build_manifest(live_manifest::LiveManifestInput {
                config: args.config.as_deref(),
                project_dir: args.project_dir.as_deref(),
                profile: args.profile.as_deref(),
                db: args.db.as_deref(),
                market_db: None,
                lock_path: args.lock_path.as_deref(),
                status_path: args.status_path.as_deref(),
                lookback_bars: args.lookback_bars,
            })?;
            let effective_config = paper_config::PaperEffectiveConfig::resolve(
                Some(Path::new(&report.base_config_path)),
                None,
                args.project_dir.as_deref(),
            )?;
            let runtime_bootstrap =
                effective_config.build_runtime_bootstrap(None, true, args.profile.as_deref())?;
            let symbols = load_symbols(args.symbols.clone(), args.symbols_file.as_deref())?;
            let candles_db = resolve_live_candles_db(args.candles_db.as_deref(), &report.interval);
            let secrets_path = args
                .secrets_path
                .clone()
                .unwrap_or_else(default_live_secrets_path);
            let daemon_report = live_daemon::run_daemon(live_daemon::LiveDaemonInput {
                effective_config,
                runtime_bootstrap,
                live_db: Path::new(&report.live_db),
                candles_db: &candles_db,
                explicit_symbols: &symbols,
                symbols_file: args.symbols_file.as_deref(),
                btc_symbol: &args.btc_symbol,
                lookback_bars: args.lookback_bars.unwrap_or(400),
                secrets_path: &secrets_path,
                lock_path: Some(Path::new(&report.lock_path)),
                status_path: Some(Path::new(&report.status_path)),
                idle_sleep_ms: args.idle_sleep_ms,
                max_idle_polls: args.max_idle_polls,
            })?;

            if args.json {
                println!("{}", serde_json::to_string_pretty(&daemon_report)?);
            } else {
                println!(
                    "live daemon ok: pid={} lock={} status={} last_fill_cursor_ms={} stop_requested={} plans={}",
                    daemon_report.pid,
                    daemon_report.lock_path,
                    daemon_report.status_path,
                    daemon_report.last_fill_cursor_ms,
                    daemon_report.stop_requested,
                    daemon_report
                        .last_cycle
                        .as_ref()
                        .map(|cycle| cycle.plans.len())
                        .unwrap_or(0)
                );
            }
        }
    }

    Ok(())
}

fn print_effective_config_report(surface: &str, report: &paper_config::PaperEffectiveConfigReport) {
    println!("{surface} effective-config ok");
    println!("base_config_path: {}", report.base_config_path);
    println!("config_path: {}", report.config_path);
    println!("active_yaml_path: {}", report.active_yaml_path);
    println!("effective_yaml_path: {}", report.effective_yaml_path);
    println!("interval: {}", report.interval);
    println!(
        "strategy_overrides_sha1: {}",
        report.strategy_overrides_sha1
    );
    println!("config_id: {}", report.config_id);
    if let Some(promoted_role) = report.promoted_role.as_deref() {
        println!("promoted_role: {}", promoted_role);
    }
    if let Some(promoted_config_path) = report.promoted_config_path.as_deref() {
        println!("promoted_config_path: {}", promoted_config_path);
    }
    if let Some(strategy_mode) = report.strategy_mode.as_deref() {
        println!("strategy_mode: {}", strategy_mode);
    }
    if let Some(strategy_mode_source) = report.strategy_mode_source.as_deref() {
        println!("strategy_mode_source: {}", strategy_mode_source);
    }
    if !report.warnings.is_empty() {
        println!("warnings:");
        for warning in &report.warnings {
            println!("  - {}", warning);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_live_effective_config_surface() {
        let cli = Cli::try_parse_from(["aiq-runtime", "live", "effective-config", "--json"])
            .expect("live effective-config should parse");
        match cli.command {
            Command::Live {
                command: LiveCommand::EffectiveConfig(args),
            } => assert!(args.json),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn cli_parses_live_manifest_surface() {
        let cli = Cli::try_parse_from(["aiq-runtime", "live", "manifest", "--json"])
            .expect("live manifest should parse");
        match cli.command {
            Command::Live {
                command: LiveCommand::Manifest(args),
            } => assert!(args.json),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn cli_parses_live_daemon_surface() {
        let cli = Cli::try_parse_from(["aiq-runtime", "live", "daemon", "--json"])
            .expect("live daemon should parse");
        match cli.command {
            Command::Live {
                command: LiveCommand::Daemon(args),
            } => assert!(args.json),
            other => panic!("unexpected command: {other:?}"),
        }
    }
}
