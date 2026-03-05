use aiq_runtime_core::runtime::{build_bootstrap, RuntimeMode};
use aiq_runtime_core::snapshot::{load_snapshot, snapshot_to_pretty_json};
use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};

mod paper_export;
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
