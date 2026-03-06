use aiq_runtime_core::paper::restore_paper_state;
use aiq_runtime_core::runtime::{build_bootstrap, RuntimeMode};
use aiq_runtime_core::snapshot::{load_snapshot, snapshot_to_pretty_json};
use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};

mod paper_cycle;
mod paper_export;
mod paper_loop;
mod paper_run_once;
mod paper_seed;

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
    /// Restore paper state from the DB through the Rust snapshot/bootstrap path.
    Doctor(PaperDoctorArgs),
    /// Execute one Rust paper step for a single symbol.
    RunOnce(PaperRunOnceArgs),
    /// Execute one repeatable Rust paper cycle across explicit symbols plus open paper positions.
    Cycle(PaperCycleArgs),
    /// Execute a repeatable Rust paper loop on interval boundaries.
    Loop(PaperLoopArgs),
}

#[derive(Debug, Clone, Args)]
struct PaperDoctorArgs {
    #[command(flatten)]
    common: CommonArgs,
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
    common: CommonArgs,
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
    common: CommonArgs,
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
    common: CommonArgs,
    /// Paper DB path to restore from and project back into.
    #[arg(long, default_value = "trading_engine.db")]
    db: PathBuf,
    /// Candle SQLite DB path used for each loop cycle.
    #[arg(long)]
    candles_db: PathBuf,
    /// Explicit symbol list (comma-delimited). Open paper positions are always included.
    #[arg(long, value_delimiter = ',')]
    symbols: Vec<String>,
    /// Optional file containing one symbol per line. Re-read before each loop step.
    #[arg(long)]
    symbols_file: Option<PathBuf>,
    /// BTC anchor symbol for alignment context.
    #[arg(long, default_value = "BTC")]
    btc_symbol: String,
    /// Number of bars to load for indicator warm-up.
    #[arg(long, default_value_t = 400)]
    lookback_bars: usize,
    /// Optional explicit first step close timestamp in ms. Useful for catch-up runs and deterministic tests.
    #[arg(long)]
    start_step_close_ts_ms: Option<i64>,
    /// Additional delay after the nominal bar close before a step becomes eligible.
    #[arg(long, default_value_t = 5_000)]
    settle_delay_ms: u64,
    /// Maximum number of loop steps to attempt before exiting. Omit to run continuously.
    #[arg(long)]
    max_cycles: Option<usize>,
    /// Override exported_at_ms for reproducible artefacts.
    #[arg(long)]
    exported_at_ms: Option<i64>,
    /// Resolve the loop but do not write any DB projections.
    #[arg(long)]
    dry_run: bool,
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

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Pipeline(args) | Command::Doctor(args) => run_bootstrap(args),
        Command::Snapshot { command } => run_snapshot(command),
        Command::Paper { command } => run_paper(command),
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
        PaperCommand::Doctor(args) => {
            let config_path = resolve_config_path(&args.common.config);
            let config = bt_core::config::load_config_checked(
                config_path
                    .to_str()
                    .context("config path must be valid UTF-8")?,
                args.common.symbol.as_deref(),
                args.common.live,
            )
            .map_err(anyhow::Error::msg)?;
            let runtime_bootstrap =
                build_bootstrap(&config, RuntimeMode::Paper, args.common.profile.as_deref())
                    .map_err(anyhow::Error::msg)?;
            let snapshot = paper_export::export_paper_snapshot(
                &args.db,
                args.exported_at_ms
                    .unwrap_or_else(|| Utc::now().timestamp_millis()),
            )?;
            let (_state, report) = restore_paper_state(&snapshot).map_err(anyhow::Error::msg)?;

            if args.common.json {
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
            let config_path = resolve_config_path(&args.common.config);
            let config = bt_core::config::load_config_checked(
                config_path
                    .to_str()
                    .context("config path must be valid UTF-8")?,
                Some(args.target_symbol.as_str()),
                args.common.live,
            )
            .map_err(anyhow::Error::msg)?;
            let runtime_bootstrap =
                build_bootstrap(&config, RuntimeMode::Paper, args.common.profile.as_deref())
                    .map_err(anyhow::Error::msg)?;
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
            let config_path = resolve_config_path(&args.common.config);
            let base_cfg = bt_core::config::load_config_checked(
                config_path
                    .to_str()
                    .context("config path must be valid UTF-8")?,
                None,
                args.common.live,
            )
            .map_err(anyhow::Error::msg)?;
            let runtime_bootstrap = build_bootstrap(
                &base_cfg,
                RuntimeMode::Paper,
                args.common.profile.as_deref(),
            )
            .map_err(anyhow::Error::msg)?;
            let mut symbols = args.symbols;
            if let Some(symbols_file) = args.symbols_file.as_ref() {
                let file_symbols = std::fs::read_to_string(symbols_file)?
                    .lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>();
                symbols.extend(file_symbols);
            }
            let report = paper_cycle::run_cycle(paper_cycle::PaperCycleInput {
                runtime_bootstrap,
                config_path: &config_path,
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
            let config_path = resolve_config_path(&args.common.config);
            let report = paper_loop::run_loop(paper_loop::PaperLoopInput {
                config_path: &config_path,
                live: args.common.live,
                profile_override: args.common.profile.as_deref(),
                paper_db: &args.db,
                candles_db: &args.candles_db,
                explicit_symbols: &args.symbols,
                symbols_file: args.symbols_file.as_deref(),
                btc_symbol: &args.btc_symbol,
                lookback_bars: args.lookback_bars,
                start_step_close_ts_ms: args.start_step_close_ts_ms,
                settle_delay_ms: args.settle_delay_ms,
                max_cycles: args.max_cycles,
                exported_at_ms: args.exported_at_ms,
                dry_run: args.dry_run,
            })?;

            if args.common.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "paper loop ok: steps={} executed={} duplicate_skips={} dry_run={}",
                    report.cycles_attempted,
                    report.cycles_executed,
                    report.cycles_duplicate_skipped,
                    report.dry_run,
                );
            }
        }
    }

    Ok(())
}
