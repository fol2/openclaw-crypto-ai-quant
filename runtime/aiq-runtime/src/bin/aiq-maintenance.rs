use anyhow::{anyhow, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use reqwest::blocking::Client;
use reqwest::StatusCode;
use rusqlite::{params, Connection, OptionalExtension};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeSet;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const DEFAULT_HL_INFO_URL: &str = "https://api.hyperliquid.xyz/info";
const DEFAULT_FUNDING_DB: &str = "candles_dbs/funding_rates.db";
const DEFAULT_RUNTIME_DB: &str = "trading_engine.db";
const DEFAULT_FACTORY_SETTINGS: &str = "config/factory_defaults.yaml";
const DEFAULT_FACTORY_ARTIFACTS_DIR: &str = "artifacts";
const DEFAULT_FACTORY_LIVE_YAML_PATH: &str = "config/strategy_overrides.live.yaml";
const DEFAULT_FACTORY_LIVE_STATE_PATH: &str = "artifacts/state/factory_live_primary.json";
const DEFAULT_FETCH_TIMEOUT_S: u64 = 5;
const DEFAULT_BUSY_TIMEOUT_S: u64 = 15;
const DEFAULT_FUNDING_DAYS: i64 = 30;
const DEFAULT_KEEP_DAYS: f64 = 14.0;
const DEFAULT_SYMBOLS: &[&str] = &["BTC", "ETH", "SOL"];
const FUNDING_FETCH_MAX_RETRIES: u32 = 3;
const FUNDING_FETCH_BASE_BACKOFF_MS: u64 = 500;
const FUNDING_SYMBOL_SLEEP_MS: u64 = 200;
const LOCK_RETRY_ATTEMPTS: u32 = 3;
const LOCK_RETRY_SLEEP_MS: u64 = 250;
const EXIT_TUNNEL_KEEP_DAYS: f64 = 30.0;
const USER_UNIVERSE_ENV: &str = ".config/openclaw/ai-quant-universe.env";
const FUNDING_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS funding_rates (
    symbol TEXT NOT NULL,
    time INTEGER NOT NULL,
    funding_rate REAL NOT NULL,
    premium REAL,
    PRIMARY KEY (symbol, time)
);
"#;

#[derive(Parser, Debug)]
#[command(
    name = "aiq-maintenance",
    version,
    about = "Rust-owned maintenance and compatibility tasks for AI Quant"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    FetchFundingRates(FetchFundingRatesArgs),
    PruneFactoryArtifacts(PruneFactoryArtifactsArgs),
    PruneRuntimeLogs(PruneRuntimeLogsArgs),
    Retired(RetiredArgs),
}

#[derive(Args, Debug)]
struct FetchFundingRatesArgs {
    #[arg(long, default_value_t = DEFAULT_FUNDING_DAYS)]
    days: i64,
    #[arg(long, default_value = DEFAULT_FUNDING_DB)]
    db: PathBuf,
}

#[derive(Clone, Debug, ValueEnum)]
enum FactoryArtifactsProfile {
    Nightly,
}

impl FactoryArtifactsProfile {
    fn run_prefix(&self) -> &'static str {
        match self {
            Self::Nightly => "nightly",
        }
    }
}

#[derive(Args, Debug)]
struct PruneFactoryArtifactsArgs {
    #[arg(long)]
    project_dir: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_FACTORY_SETTINGS)]
    settings: PathBuf,
    #[arg(long, value_enum, default_value_t = FactoryArtifactsProfile::Nightly)]
    profile: FactoryArtifactsProfile,
    #[arg(long, default_value_t = false)]
    dry_run: bool,
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

#[derive(Args, Debug)]
struct PruneRuntimeLogsArgs {
    #[arg(long)]
    db: Vec<PathBuf>,
    #[arg(long)]
    keep_days: Option<f64>,
    #[arg(long, default_value_t = false)]
    dry_run: bool,
    #[arg(long, default_value_t = false)]
    vacuum: bool,
}

#[derive(Clone, Debug, ValueEnum)]
enum RetiredStatus {
    Retired,
    Disabled,
    Blocked,
}

#[derive(Args, Debug)]
struct RetiredArgs {
    #[arg(long)]
    name: String,
    #[arg(long)]
    message: String,
    #[arg(long)]
    json_path: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = RetiredStatus::Retired)]
    status: RetiredStatus,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct MaintenanceFactorySettings {
    artifacts_dir: PathBuf,
    deployment: MaintenanceDeploymentSettings,
    live_governance: MaintenanceLiveGovernanceSettings,
}

impl MaintenanceFactorySettings {
    fn live_yaml_path(&self) -> &Path {
        self.deployment
            .live_yaml_path
            .as_deref()
            .unwrap_or_else(|| Path::new(DEFAULT_FACTORY_LIVE_YAML_PATH))
    }

    fn live_state_path(&self) -> &Path {
        self.live_governance
            .state_path
            .as_deref()
            .unwrap_or_else(|| Path::new(DEFAULT_FACTORY_LIVE_STATE_PATH))
    }
}

impl Default for MaintenanceFactorySettings {
    fn default() -> Self {
        Self {
            artifacts_dir: PathBuf::from(DEFAULT_FACTORY_ARTIFACTS_DIR),
            deployment: MaintenanceDeploymentSettings::default(),
            live_governance: MaintenanceLiveGovernanceSettings::default(),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct MaintenanceDeploymentSettings {
    live_yaml_path: Option<PathBuf>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct MaintenanceLiveGovernanceSettings {
    state_path: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct FundingHistoryEntry {
    time: i64,
    #[serde(rename = "fundingRate")]
    funding_rate: String,
    premium: Option<String>,
}

#[derive(Debug)]
struct FundingFetchError {
    status: Option<StatusCode>,
    source: anyhow::Error,
}

impl FundingFetchError {
    fn retryable(&self) -> bool {
        matches!(
            self.status,
            Some(StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE)
        )
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::FetchFundingRates(args) => fetch_funding_rates(args),
        Command::PruneFactoryArtifacts(args) => prune_factory_artifacts(args),
        Command::PruneRuntimeLogs(args) => prune_runtime_logs(args),
        Command::Retired(args) => retired(args),
    }
}

fn fetch_funding_rates(args: FetchFundingRatesArgs) -> Result<()> {
    let symbols = resolve_symbols();
    let db_path = normalise_path(&args.db);
    let lookback_days = args.days.max(0);

    println!(
        "[funding] {} symbols, {} days lookback",
        symbols.len(),
        lookback_days
    );
    println!("[funding] DB: {}", db_path.display());

    ensure_parent_dir(&db_path)?;
    let conn = Connection::open(&db_path)
        .with_context(|| format!("failed to open funding DB {}", db_path.display()))?;
    configure_connection(&conn)?;
    conn.execute_batch(FUNDING_SCHEMA)
        .context("failed to ensure funding_rates schema")?;
    set_file_mode_owner_only(&db_path);

    let client = Client::builder()
        .timeout(Duration::from_secs(DEFAULT_FETCH_TIMEOUT_S))
        .build()
        .context("failed to build Hyperliquid HTTP client")?;

    let now_ms = now_ms()?;
    let default_start_ms = now_ms.saturating_sub(lookback_days.saturating_mul(86_400_000));
    let mut total_inserted = 0usize;

    for (idx, symbol) in symbols.iter().enumerate() {
        let start_ms = last_funding_time(&conn, symbol)?
            .map(|value| value.saturating_add(1))
            .unwrap_or(default_start_ms);

        if start_ms >= now_ms {
            println!("  [{}/{}] {}: up to date", idx + 1, symbols.len(), symbol);
            continue;
        }

        let entries = match fetch_funding_history_with_retry(&client, symbol, start_ms, now_ms) {
            Ok(entries) => entries,
            Err(err) => {
                eprintln!("  WARN: API error for {}: {}", symbol, err);
                continue;
            }
        };
        let inserted = insert_funding_rows(&conn, symbol, &entries)?;
        total_inserted = total_inserted.saturating_add(inserted);
        println!(
            "  [{}/{}] {}: +{} rows (from {})",
            idx + 1,
            symbols.len(),
            symbol,
            inserted,
            start_ms
        );

        if idx + 1 < symbols.len() {
            thread::sleep(Duration::from_millis(FUNDING_SYMBOL_SLEEP_MS));
        }
    }

    println!("\n[funding] Done. Total inserted: {}", total_inserted);
    Ok(())
}

fn prune_factory_artifacts(args: PruneFactoryArtifactsArgs) -> Result<()> {
    let project_dir = resolve_project_dir(args.project_dir)?;
    let settings_path = resolve_under_project(&project_dir, &args.settings);
    let settings = load_maintenance_factory_settings(&settings_path)?;
    let artifacts_root = resolve_under_project(&project_dir, &settings.artifacts_dir);
    if !artifacts_root.is_dir() {
        return Ok(());
    }

    let run_prefix = args.profile.run_prefix();
    let run_dirs = collect_factory_run_dirs(&artifacts_root, run_prefix)?;
    if run_dirs.is_empty() {
        return Ok(());
    }

    let latest_run_id = run_dirs.last().map(|entry| entry.run_id.clone());
    let mut retained_run_ids =
        collect_retained_run_ids(&project_dir, &artifacts_root, run_prefix, &settings)?;
    if let Some(run_id) = latest_run_id.as_ref() {
        retained_run_ids.insert(run_id.clone());
    }

    let deleted_run_dirs: Vec<PathBuf> = run_dirs
        .iter()
        .filter(|entry| !retained_run_ids.contains(&entry.run_id))
        .map(|entry| entry.path.clone())
        .collect();

    let deleted_effective_configs =
        collect_effective_configs_to_delete(&artifacts_root, run_prefix, &retained_run_ids)?;
    let removed_date_dirs = collect_empty_date_dirs(&artifacts_root, &deleted_run_dirs)?;

    if args.dry_run {
        println!(
            "[prune_factory_artifacts] dry_run=true profile={} keep_runs={} delete_runs={} delete_effective_configs={} remove_date_dirs={}",
            run_prefix,
            retained_run_ids.len(),
            deleted_run_dirs.len(),
            deleted_effective_configs.len(),
            removed_date_dirs.len()
        );
        for run_id in &retained_run_ids {
            println!("[prune_factory_artifacts] keep_run_id={run_id}");
        }
        if args.verbose {
            for path in &deleted_run_dirs {
                println!(
                    "[prune_factory_artifacts] would_delete_run_dir={}",
                    path.display()
                );
            }
            for path in &deleted_effective_configs {
                println!(
                    "[prune_factory_artifacts] would_delete_effective_config={}",
                    path.display()
                );
            }
            for path in &removed_date_dirs {
                println!(
                    "[prune_factory_artifacts] would_remove_date_dir={}",
                    path.display()
                );
            }
        }
        return Ok(());
    }

    for path in &deleted_run_dirs {
        fs::remove_dir_all(path).with_context(|| format!("remove {}", path.display()))?;
    }
    for path in &deleted_effective_configs {
        fs::remove_file(path).with_context(|| format!("remove {}", path.display()))?;
    }
    for path in &removed_date_dirs {
        if path.is_dir() {
            fs::remove_dir(path).with_context(|| format!("remove {}", path.display()))?;
        }
    }

    println!(
        "[prune_factory_artifacts] profile={} kept_runs={} deleted_runs={} deleted_effective_configs={} removed_date_dirs={}",
        run_prefix,
        retained_run_ids.len(),
        deleted_run_dirs.len(),
        deleted_effective_configs.len(),
        removed_date_dirs.len()
    );
    Ok(())
}

fn prune_runtime_logs(args: PruneRuntimeLogsArgs) -> Result<()> {
    let keep_days = args
        .keep_days
        .unwrap_or_else(|| env_f64("AI_QUANT_RUNTIME_LOG_KEEP_DAYS").unwrap_or(DEFAULT_KEEP_DAYS))
        .max(0.0);
    let runtime_cutoff_ts_ms = cutoff_ts_ms(keep_days)?;
    let tunnel_cutoff_ts_ms = cutoff_ts_ms(EXIT_TUNNEL_KEEP_DAYS)?;

    for db_path in resolve_runtime_dbs(args.db) {
        prune_runtime_logs_db(
            &db_path,
            runtime_cutoff_ts_ms,
            tunnel_cutoff_ts_ms,
            args.dry_run,
            args.vacuum,
        )?;
    }

    Ok(())
}

fn prune_runtime_logs_db(
    db_path: &Path,
    runtime_cutoff_ts_ms: i64,
    tunnel_cutoff_ts_ms: i64,
    dry_run: bool,
    vacuum: bool,
) -> Result<()> {
    if !db_path.exists() {
        return Ok(());
    }

    let mut attempt = 0u32;
    loop {
        let conn = Connection::open(db_path)
            .with_context(|| format!("failed to open runtime DB {}", db_path.display()))?;

        let result = (|| -> Result<()> {
            configure_connection(&conn)?;

            if !table_exists(&conn, "runtime_logs")? {
                return Ok(());
            }

            let delete_count = count_older_than(&conn, "runtime_logs", runtime_cutoff_ts_ms)?;
            if delete_count == 0 {
                return Ok(());
            }

            if dry_run {
                println!(
                    "[prune_runtime_logs] would_delete={} cutoff_ts_ms={} db={}",
                    delete_count,
                    runtime_cutoff_ts_ms,
                    db_path.display()
                );
                return Ok(());
            }

            conn.execute(
                "DELETE FROM runtime_logs WHERE ts_ms < ?1",
                params![runtime_cutoff_ts_ms],
            )
            .context("failed to delete old runtime_logs rows")?;
            println!(
                "[prune_runtime_logs] deleted={} cutoff_ts_ms={} db={}",
                delete_count,
                runtime_cutoff_ts_ms,
                db_path.display()
            );

            if table_exists(&conn, "exit_tunnel")? {
                let exit_tunnel_count =
                    count_older_than(&conn, "exit_tunnel", tunnel_cutoff_ts_ms)?;
                if exit_tunnel_count > 0 {
                    conn.execute(
                        "DELETE FROM exit_tunnel WHERE ts_ms < ?1",
                        params![tunnel_cutoff_ts_ms],
                    )
                    .context("failed to delete old exit_tunnel rows")?;
                    println!(
                        "[prune_runtime_logs] exit_tunnel deleted={}",
                        exit_tunnel_count
                    );
                }
            }

            if vacuum {
                conn.execute_batch("VACUUM")
                    .context("failed to vacuum runtime DB")?;
            }

            Ok(())
        })();

        match result {
            Ok(()) => return Ok(()),
            Err(err) if is_lock_contention_error(&err) && attempt < LOCK_RETRY_ATTEMPTS => {
                attempt += 1;
                let sleep_ms = LOCK_RETRY_SLEEP_MS.saturating_mul(u64::from(attempt));
                println!(
                    "[prune_runtime_logs] lock contention retry={}/{} sleep_s={:.2} db={}",
                    attempt,
                    LOCK_RETRY_ATTEMPTS,
                    sleep_ms as f64 / 1000.0,
                    db_path.display()
                );
                thread::sleep(Duration::from_millis(sleep_ms));
            }
            Err(err) => return Err(err),
        }
    }
}

fn retired(args: RetiredArgs) -> Result<()> {
    let payload = json!({
        "ok": false,
        "retired": true,
        "name": args.name,
        "status": retired_status_label(&args.status),
        "message": args.message,
        "timestamp_ms": now_ms()?,
    });

    eprintln!(
        "[compat:{}] {}",
        retired_status_label(&args.status),
        payload["message"].as_str().unwrap_or_default()
    );

    if let Some(path) = args.json_path {
        let target = normalise_path(&path);
        ensure_parent_dir(&target)?;
        fs::write(
            &target,
            serde_json::to_vec_pretty(&payload).context("failed to serialise retired payload")?,
        )
        .with_context(|| format!("failed to write retired payload {}", target.display()))?;
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct FactoryRunDir {
    run_id: String,
    path: PathBuf,
}

#[derive(Debug, Default, Deserialize)]
struct FactoryStateRunRefs {
    deployed_by_run_id: Option<String>,
    last_transition_by_run_id: Option<String>,
    source_config_path: Option<String>,
    manifest_path: Option<String>,
}

fn resolve_project_dir(project_dir: Option<PathBuf>) -> Result<PathBuf> {
    match project_dir {
        Some(path) => Ok(normalise_path(&path)),
        None => std::env::current_dir().context("failed to resolve current working directory"),
    }
}

fn load_maintenance_factory_settings(path: &Path) -> Result<MaintenanceFactorySettings> {
    if !path.is_file() {
        return Ok(MaintenanceFactorySettings::default());
    }
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read factory settings {}", path.display()))?;
    serde_yaml::from_str(&text)
        .with_context(|| format!("failed to parse factory settings {}", path.display()))
}

fn resolve_under_project(project_dir: &Path, raw: &Path) -> PathBuf {
    if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        project_dir.join(raw)
    }
}

fn collect_factory_run_dirs(artifacts_root: &Path, run_prefix: &str) -> Result<Vec<FactoryRunDir>> {
    let mut run_dirs = Vec::new();
    let wanted_prefix = format!("{run_prefix}_");

    for entry in fs::read_dir(artifacts_root)
        .with_context(|| format!("failed to read {}", artifacts_root.display()))?
    {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !is_date_dir_name(name) {
            continue;
        }

        for run_entry in
            fs::read_dir(&path).with_context(|| format!("failed to read {}", path.display()))?
        {
            let run_path = run_entry?.path();
            if !run_path.is_dir() {
                continue;
            }
            let Some(run_name) = run_path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            let Some(run_id) = run_name.strip_prefix("run_") else {
                continue;
            };
            if !run_id.starts_with(&wanted_prefix) {
                continue;
            }
            run_dirs.push(FactoryRunDir {
                run_id: run_id.to_string(),
                path: run_path,
            });
        }
    }

    run_dirs.sort_by(|left, right| left.run_id.cmp(&right.run_id));
    Ok(run_dirs)
}

fn collect_retained_run_ids(
    project_dir: &Path,
    artifacts_root: &Path,
    run_prefix: &str,
    settings: &MaintenanceFactorySettings,
) -> Result<BTreeSet<String>> {
    let mut retained = BTreeSet::new();
    for state_dir in paper_state_dirs(project_dir, artifacts_root) {
        if !state_dir.is_dir() {
            continue;
        }
        for entry in fs::read_dir(&state_dir)
            .with_context(|| format!("failed to read {}", state_dir.display()))?
        {
            let path = entry?.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if !name.starts_with("factory_paper_soak_") || !name.ends_with(".json") {
                continue;
            }
            extend_retained_run_ids_from_state(&mut retained, &path, run_prefix)?;
        }
    }

    let live_state_path = resolve_under_project(project_dir, settings.live_state_path());
    if live_state_path.is_file() {
        extend_retained_run_ids_from_state(&mut retained, &live_state_path, run_prefix)?;
    }

    let live_yaml_path = resolve_under_project(project_dir, settings.live_yaml_path());
    if let Some(run_id) = extract_run_id_from_live_yaml_base(&live_yaml_path, run_prefix)? {
        retained.insert(run_id);
    }

    Ok(retained)
}

fn paper_state_dirs(project_dir: &Path, artifacts_root: &Path) -> BTreeSet<PathBuf> {
    [
        project_dir.join("artifacts").join("state"),
        artifacts_root.join("state"),
    ]
    .into_iter()
    .collect()
}

fn extend_retained_run_ids_from_state(
    retained: &mut BTreeSet<String>,
    path: &Path,
    run_prefix: &str,
) -> Result<()> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let state: FactoryStateRunRefs = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;

    for maybe_run_id in [state.deployed_by_run_id, state.last_transition_by_run_id] {
        if let Some(run_id) = maybe_run_id.filter(|value| is_matching_run_id(value, run_prefix)) {
            retained.insert(run_id);
        }
    }

    for maybe_path in [state.source_config_path, state.manifest_path] {
        if let Some(run_id) = maybe_path
            .as_deref()
            .and_then(|value| extract_run_id_from_path(value, run_prefix))
        {
            retained.insert(run_id);
        }
    }

    Ok(())
}

fn extract_run_id_from_live_yaml_base(path: &Path, run_prefix: &str) -> Result<Option<String>> {
    if !path.is_file() {
        return Ok(None);
    }
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    for line in text.lines().take(20) {
        let trimmed = line.trim();
        let Some(raw) = trimmed.strip_prefix("# Base: ") else {
            continue;
        };
        let run_id = raw.trim_end_matches(".yaml").trim();
        if is_matching_run_id(run_id, run_prefix) {
            return Ok(Some(run_id.to_string()));
        }
    }
    Ok(None)
}

fn extract_run_id_from_path(raw: &str, run_prefix: &str) -> Option<String> {
    let wanted_prefix = format!("{run_prefix}_");
    let path = Path::new(raw);
    path.components().find_map(|component| {
        let name = component.as_os_str().to_str()?;
        let run_id = name.strip_prefix("run_")?;
        run_id
            .starts_with(&wanted_prefix)
            .then(|| run_id.to_string())
    })
}

fn collect_effective_configs_to_delete(
    artifacts_root: &Path,
    run_prefix: &str,
    retained_run_ids: &BTreeSet<String>,
) -> Result<Vec<PathBuf>> {
    let effective_dir = artifacts_root.join("_effective_configs");
    if !effective_dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut to_delete = Vec::new();
    for entry in fs::read_dir(&effective_dir)
        .with_context(|| format!("failed to read {}", effective_dir.display()))?
    {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
            continue;
        };
        if !is_matching_run_id(stem, run_prefix) || retained_run_ids.contains(stem) {
            continue;
        }
        to_delete.push(path);
    }
    to_delete.sort();
    Ok(to_delete)
}

fn collect_empty_date_dirs(
    artifacts_root: &Path,
    deleted_run_dirs: &[PathBuf],
) -> Result<Vec<PathBuf>> {
    let mut candidates = BTreeSet::new();
    let deleted: BTreeSet<PathBuf> = deleted_run_dirs.iter().cloned().collect();
    for path in deleted_run_dirs {
        if let Some(parent) = path.parent() {
            candidates.insert(parent.to_path_buf());
        }
    }

    let mut empty_dirs = Vec::new();
    for path in candidates {
        if path == artifacts_root {
            continue;
        }
        let has_remaining_entries = fs::read_dir(&path)
            .with_context(|| format!("failed to read {}", path.display()))?
            .filter_map(|entry| entry.ok().map(|value| value.path()))
            .any(|entry_path| !deleted.contains(&entry_path));
        if !has_remaining_entries {
            empty_dirs.push(path);
        }
    }
    Ok(empty_dirs)
}

fn is_matching_run_id(value: &str, run_prefix: &str) -> bool {
    value.starts_with(&format!("{run_prefix}_"))
}

fn is_date_dir_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(idx, byte)| matches!(idx, 4 | 7) || byte.is_ascii_digit())
}

fn resolve_symbols() -> Vec<String> {
    if let Some(symbols) = env_symbols("AI_QUANT_SYMBOLS") {
        return symbols;
    }

    if let Some(home) = std::env::var_os("HOME") {
        let universe_env = PathBuf::from(home).join(USER_UNIVERSE_ENV);
        if let Ok(contents) = fs::read_to_string(&universe_env) {
            if let Some(symbols) = parse_universe_env_symbols(&contents) {
                return symbols;
            }
        }
    }

    DEFAULT_SYMBOLS
        .iter()
        .map(|value| (*value).to_string())
        .collect()
}

fn env_symbols(name: &str) -> Option<Vec<String>> {
    std::env::var(name)
        .ok()
        .and_then(|raw| split_symbols(&raw))
        .filter(|symbols| !symbols.is_empty())
}

fn parse_universe_env_symbols(contents: &str) -> Option<Vec<String>> {
    contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .find_map(|line| {
            line.strip_prefix("AI_QUANT_SIDECAR_SYMBOLS=")
                .and_then(split_symbols)
        })
}

fn split_symbols(raw: &str) -> Option<Vec<String>> {
    let symbols: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_ascii_uppercase())
        .collect();
    if symbols.is_empty() {
        None
    } else {
        Some(symbols)
    }
}

fn fetch_funding_history_with_retry(
    client: &Client,
    symbol: &str,
    start_ms: i64,
    end_ms: i64,
) -> Result<Vec<FundingHistoryEntry>> {
    for attempt in 0..=FUNDING_FETCH_MAX_RETRIES {
        match fetch_funding_history(client, symbol, start_ms, end_ms) {
            Ok(entries) => return Ok(entries),
            Err(err) if err.retryable() && attempt < FUNDING_FETCH_MAX_RETRIES => {
                let delay_ms = FUNDING_FETCH_BASE_BACKOFF_MS.saturating_mul(1u64 << attempt);
                eprintln!(
                    "  WARN: transient API error for {}: {} (retry {}/{} after {:.2}s)",
                    symbol,
                    err.source,
                    attempt + 1,
                    FUNDING_FETCH_MAX_RETRIES,
                    delay_ms as f64 / 1000.0
                );
                thread::sleep(Duration::from_millis(delay_ms));
            }
            Err(err) => {
                return Err(err.source)
                    .with_context(|| format!("failed to fetch funding history for {}", symbol))
            }
        }
    }

    unreachable!("retry loop must return before exhaustion")
}

fn fetch_funding_history(
    client: &Client,
    symbol: &str,
    start_ms: i64,
    end_ms: i64,
) -> std::result::Result<Vec<FundingHistoryEntry>, FundingFetchError> {
    let response = client
        .post(DEFAULT_HL_INFO_URL)
        .json(&json!({
            "type": "fundingHistory",
            "coin": symbol,
            "startTime": start_ms,
            "endTime": end_ms,
        }))
        .send()
        .map_err(|err| FundingFetchError {
            status: err.status(),
            source: anyhow!(err),
        })?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        return Err(FundingFetchError {
            status: Some(status),
            source: anyhow!("HTTP {} {}", status, body.trim()),
        });
    }

    response
        .json::<Vec<FundingHistoryEntry>>()
        .map_err(|err| FundingFetchError {
            status: Some(status),
            source: anyhow!(err),
        })
}

fn last_funding_time(conn: &Connection, symbol: &str) -> Result<Option<i64>> {
    conn.query_row(
        "SELECT MAX(time) FROM funding_rates WHERE symbol = ?1",
        params![symbol],
        |row| row.get::<_, Option<i64>>(0),
    )
    .optional()
    .context("failed to query latest funding timestamp")?
    .flatten()
    .pipe(Ok)
}

fn insert_funding_rows(
    conn: &Connection,
    symbol: &str,
    entries: &[FundingHistoryEntry],
) -> Result<usize> {
    let before_changes = conn.total_changes();
    let mut statement = conn
        .prepare(
            "INSERT OR IGNORE INTO funding_rates (symbol, time, funding_rate, premium) VALUES (?1, ?2, ?3, ?4)",
        )
        .context("failed to prepare funding insert statement")?;

    for entry in entries {
        let funding_rate = match parse_required_f64(&entry.funding_rate, "fundingRate") {
            Ok(value) => value,
            Err(err) => {
                eprintln!(
                    "  WARN: bad funding entry for {} at {}: {}",
                    symbol, entry.time, err
                );
                continue;
            }
        };
        let premium = match entry
            .premium
            .as_deref()
            .map(|value| parse_required_f64(value, "premium"))
            .transpose()
        {
            Ok(value) => value,
            Err(err) => {
                eprintln!(
                    "  WARN: bad funding entry for {} at {}: {}",
                    symbol, entry.time, err
                );
                continue;
            }
        };
        statement
            .execute(params![symbol, entry.time, funding_rate, premium])
            .with_context(|| format!("failed to insert funding row for {}", symbol))?;
    }

    let inserted = conn.total_changes().saturating_sub(before_changes);
    usize::try_from(inserted).context("funding insert count overflowed usize")
}

fn resolve_runtime_dbs(values: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut resolved = Vec::new();

    let mut push_unique = |path: PathBuf| {
        let path = normalise_path(&path);
        if !resolved.contains(&path) {
            resolved.push(path);
        }
    };

    if !values.is_empty() {
        for path in values {
            push_unique(path);
        }
        return resolved;
    }

    if let Ok(path) = std::env::var("AI_QUANT_DB_PATH") {
        if !path.trim().is_empty() {
            push_unique(PathBuf::from(path));
            return resolved;
        }
    }

    push_unique(PathBuf::from(DEFAULT_RUNTIME_DB));
    resolved
}

fn configure_connection(conn: &Connection) -> Result<()> {
    conn.pragma_update(None, "journal_mode", "WAL")
        .context("failed to enable WAL mode")?;
    conn.busy_timeout(Duration::from_secs(DEFAULT_BUSY_TIMEOUT_S))
        .context("failed to apply SQLite busy_timeout")?;
    Ok(())
}

fn count_older_than(conn: &Connection, table: &str, cutoff_ts_ms: i64) -> Result<i64> {
    let sql = format!("SELECT COUNT(*) FROM {table} WHERE ts_ms < ?1");
    conn.query_row(&sql, params![cutoff_ts_ms], |row| row.get(0))
        .with_context(|| format!("failed to count rows in {}", table))
}

fn table_exists(conn: &Connection, table: &str) -> Result<bool> {
    conn.query_row(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1 LIMIT 1",
        params![table],
        |_| Ok(()),
    )
    .optional()
    .map(|row| row.is_some())
    .with_context(|| format!("failed to inspect SQLite metadata for {}", table))
}

fn is_lock_contention_error(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        let text = cause.to_string().to_ascii_lowercase();
        text.contains("locked") || text.contains("busy")
    })
}

fn cutoff_ts_ms(keep_days: f64) -> Result<i64> {
    let now_ms = now_ms()?;
    let keep_ms = (keep_days.max(0.0) * 86_400_000.0).round();
    let keep_ms_i64 = if keep_ms.is_finite() {
        keep_ms.clamp(0.0, i64::MAX as f64) as i64
    } else {
        i64::MAX
    };
    Ok(now_ms.saturating_sub(keep_ms_i64))
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
}

fn parse_required_f64(raw: &str, field: &str) -> Result<f64> {
    raw.parse::<f64>()
        .with_context(|| format!("failed to parse {} as f64", field))
}

fn retired_status_label(status: &RetiredStatus) -> &'static str {
    match status {
        RetiredStatus::Retired => "retired",
        RetiredStatus::Disabled => "disabled",
        RetiredStatus::Blocked => "blocked",
    }
}

fn normalise_path(path: &Path) -> PathBuf {
    if let Some(raw) = path.to_str() {
        if let Some(stripped) = raw.strip_prefix("~/") {
            if let Some(home) = std::env::var_os("HOME") {
                return PathBuf::from(home).join(stripped);
            }
        }
    }

    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    Ok(())
}

fn set_file_mode_owner_only(path: &Path) {
    #[cfg(unix)]
    {
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
    }
}

fn now_ms() -> Result<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is set before the Unix epoch")?;
    i64::try_from(duration.as_millis()).context("system clock exceeded i64 milliseconds")
}

trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}

impl<T> Pipe for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn parses_symbols_from_universe_env() {
        let contents = r#"
            # comment
            AI_QUANT_SIDECAR_SYMBOLS=btc, eth ,sol
        "#;
        let symbols = parse_universe_env_symbols(contents).expect("expected symbols");
        assert_eq!(symbols, vec!["BTC", "ETH", "SOL"]);
    }

    #[test]
    fn prune_runtime_logs_deletes_only_old_rows() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("runtime.db");
        let conn = Connection::open(&db_path).expect("open db");
        conn.execute_batch(
            r#"
            CREATE TABLE runtime_logs (ts_ms INTEGER NOT NULL);
            CREATE TABLE exit_tunnel (ts_ms INTEGER NOT NULL);
            "#,
        )
        .expect("schema");

        let now_ms = now_ms().expect("now");
        let stale_runtime = now_ms - 20 * 86_400_000;
        let fresh_runtime = now_ms - 1_000;
        let stale_tunnel = now_ms - 40 * 86_400_000;
        let fresh_tunnel = now_ms - 10 * 86_400_000;

        conn.execute(
            "INSERT INTO runtime_logs (ts_ms) VALUES (?1), (?2)",
            params![stale_runtime, fresh_runtime],
        )
        .expect("seed runtime_logs");
        conn.execute(
            "INSERT INTO exit_tunnel (ts_ms) VALUES (?1), (?2)",
            params![stale_tunnel, fresh_tunnel],
        )
        .expect("seed exit_tunnel");
        drop(conn);

        prune_runtime_logs(PruneRuntimeLogsArgs {
            db: vec![db_path.clone()],
            keep_days: Some(14.0),
            dry_run: false,
            vacuum: false,
        })
        .expect("prune succeeds");

        let conn = Connection::open(&db_path).expect("reopen db");
        let runtime_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM runtime_logs", [], |row| row.get(0))
            .expect("runtime_logs count");
        let tunnel_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM exit_tunnel", [], |row| row.get(0))
            .expect("exit_tunnel count");
        assert_eq!(runtime_count, 1);
        assert_eq!(tunnel_count, 1);
    }

    #[test]
    fn prune_runtime_logs_handles_multiple_databases_in_one_run() {
        let dir = tempdir().expect("tempdir");
        let db_a = dir.path().join("paper1.db");
        let db_b = dir.path().join("paper2.db");

        for db_path in [&db_a, &db_b] {
            let conn = Connection::open(db_path).expect("open db");
            conn.execute_batch(
                r#"
                CREATE TABLE runtime_logs (ts_ms INTEGER NOT NULL);
                CREATE TABLE exit_tunnel (ts_ms INTEGER NOT NULL);
                "#,
            )
            .expect("schema");

            let now_ms = now_ms().expect("now");
            let stale_runtime = now_ms - 20 * 86_400_000;
            let fresh_runtime = now_ms - 1_000;
            let stale_tunnel = now_ms - 40 * 86_400_000;
            let fresh_tunnel = now_ms - 10 * 86_400_000;

            conn.execute(
                "INSERT INTO runtime_logs (ts_ms) VALUES (?1), (?2)",
                params![stale_runtime, fresh_runtime],
            )
            .expect("seed runtime_logs");
            conn.execute(
                "INSERT INTO exit_tunnel (ts_ms) VALUES (?1), (?2)",
                params![stale_tunnel, fresh_tunnel],
            )
            .expect("seed exit_tunnel");
        }

        prune_runtime_logs(PruneRuntimeLogsArgs {
            db: vec![db_a.clone(), db_b.clone()],
            keep_days: Some(14.0),
            dry_run: false,
            vacuum: false,
        })
        .expect("prune succeeds");

        for db_path in [&db_a, &db_b] {
            let conn = Connection::open(db_path).expect("reopen db");
            let runtime_count: i64 = conn
                .query_row("SELECT COUNT(*) FROM runtime_logs", [], |row| row.get(0))
                .expect("runtime_logs count");
            let tunnel_count: i64 = conn
                .query_row("SELECT COUNT(*) FROM exit_tunnel", [], |row| row.get(0))
                .expect("exit_tunnel count");
            assert_eq!(runtime_count, 1);
            assert_eq!(tunnel_count, 1);
        }
    }

    #[test]
    fn prune_factory_artifacts_keeps_latest_and_paper_deployed_runs() {
        let dir = tempdir().expect("tempdir");
        let project_dir = dir.path();
        let artifacts_dir = project_dir.join("artifacts");

        let deployed_run_id = "nightly_20260314T004910Z_444";
        let stale_run_id = "nightly_20260313T001500Z_111";
        let latest_run_id = "nightly_20260315T010000Z_222";
        let archive_run_id = "archive_20260315T020000Z_333";

        let deployed_run_dir = create_run_dir(&artifacts_dir, "2026-03-14", deployed_run_id);
        let stale_run_dir = create_run_dir(&artifacts_dir, "2026-03-13", stale_run_id);
        let latest_run_dir = create_run_dir(&artifacts_dir, "2026-03-15", latest_run_id);
        let archive_run_dir = create_run_dir(&artifacts_dir, "2026-03-15", archive_run_id);

        write_effective_config_stub(&artifacts_dir, deployed_run_id);
        write_effective_config_stub(&artifacts_dir, stale_run_id);
        write_effective_config_stub(&artifacts_dir, latest_run_id);
        write_effective_config_stub(&artifacts_dir, archive_run_id);

        write_json_file(
            &artifacts_dir.join("state/factory_paper_soak_primary.json"),
            &json!({
                "deployed_by_run_id": deployed_run_id,
                "source_config_path": deployed_run_dir
                    .join("configs/candidate_primary.yaml")
                    .display()
                    .to_string()
            }),
        );

        prune_factory_artifacts(PruneFactoryArtifactsArgs {
            project_dir: Some(project_dir.to_path_buf()),
            settings: PathBuf::from(DEFAULT_FACTORY_SETTINGS),
            profile: FactoryArtifactsProfile::Nightly,
            dry_run: false,
            verbose: false,
        })
        .expect("prune factory artifacts succeeds");

        assert!(deployed_run_dir.is_dir(), "deployed run should remain");
        assert!(latest_run_dir.is_dir(), "latest run should remain");
        assert!(!stale_run_dir.exists(), "stale run should be removed");
        assert!(archive_run_dir.is_dir(), "other run prefixes should remain");

        assert!(
            artifacts_dir
                .join(format!("_effective_configs/{deployed_run_id}.yaml"))
                .is_file(),
            "deployed effective config should remain"
        );
        assert!(
            artifacts_dir
                .join(format!("_effective_configs/{latest_run_id}.yaml"))
                .is_file(),
            "latest effective config should remain"
        );
        assert!(
            !artifacts_dir
                .join(format!("_effective_configs/{stale_run_id}.yaml"))
                .exists(),
            "stale effective config should be removed"
        );
        assert!(
            artifacts_dir
                .join(format!("_effective_configs/{archive_run_id}.yaml"))
                .is_file(),
            "other run-prefix effective config should remain"
        );
        assert!(
            !artifacts_dir.join("2026-03-13").exists(),
            "empty nightly date directory should be removed"
        );
    }

    #[test]
    fn prune_factory_artifacts_keeps_live_yaml_base_run_when_state_is_missing() {
        let dir = tempdir().expect("tempdir");
        let project_dir = dir.path();
        let artifacts_dir = project_dir.join("artifacts");

        let live_run_id = "nightly_20260227T010108Z";
        let stale_run_id = "nightly_20260226T220000Z";
        let latest_run_id = "nightly_20260315T010000Z_222";

        let live_run_dir = create_run_dir(&artifacts_dir, "2026-02-27", live_run_id);
        let stale_run_dir = create_run_dir(&artifacts_dir, "2026-02-26", stale_run_id);
        let latest_run_dir = create_run_dir(&artifacts_dir, "2026-03-15", latest_run_id);

        write_effective_config_stub(&artifacts_dir, live_run_id);
        write_effective_config_stub(&artifacts_dir, stale_run_id);
        write_effective_config_stub(&artifacts_dir, latest_run_id);

        let live_yaml_path = project_dir.join(DEFAULT_FACTORY_LIVE_YAML_PATH);
        ensure_parent_dir(&live_yaml_path).expect("create live yaml parent");
        fs::write(
            &live_yaml_path,
            format!(
                "# Generated by generate_config.py at 2026-02-27T03:13:25Z\n# Base: {live_run_id}.yaml\nglobal: {{}}\n"
            ),
        )
        .expect("write live yaml");

        prune_factory_artifacts(PruneFactoryArtifactsArgs {
            project_dir: Some(project_dir.to_path_buf()),
            settings: PathBuf::from(DEFAULT_FACTORY_SETTINGS),
            profile: FactoryArtifactsProfile::Nightly,
            dry_run: false,
            verbose: false,
        })
        .expect("prune factory artifacts succeeds");

        assert!(live_run_dir.is_dir(), "live base run should remain");
        assert!(latest_run_dir.is_dir(), "latest run should remain");
        assert!(!stale_run_dir.exists(), "stale run should be removed");
        assert!(
            artifacts_dir
                .join(format!("_effective_configs/{live_run_id}.yaml"))
                .is_file(),
            "live base effective config should remain"
        );
        assert!(
            !artifacts_dir
                .join(format!("_effective_configs/{stale_run_id}.yaml"))
                .exists(),
            "stale effective config should be removed"
        );
    }

    #[test]
    fn prune_factory_artifacts_keeps_paper_deployed_run_with_custom_artifacts_dir() {
        let dir = tempdir().expect("tempdir");
        let project_dir = dir.path();
        let custom_artifacts_dir = project_dir.join("custom_artifacts");

        let deployed_run_id = "nightly_20260310T010000Z_111";
        let stale_run_id = "nightly_20260309T010000Z_111";
        let latest_run_id = "nightly_20260315T010000Z_222";

        let deployed_run_dir = create_run_dir(&custom_artifacts_dir, "2026-03-10", deployed_run_id);
        let stale_run_dir = create_run_dir(&custom_artifacts_dir, "2026-03-09", stale_run_id);
        let latest_run_dir = create_run_dir(&custom_artifacts_dir, "2026-03-15", latest_run_id);

        write_effective_config_stub(&custom_artifacts_dir, deployed_run_id);
        write_effective_config_stub(&custom_artifacts_dir, stale_run_id);
        write_effective_config_stub(&custom_artifacts_dir, latest_run_id);

        write_json_file(
            &project_dir.join("artifacts/state/factory_paper_soak_primary.json"),
            &json!({
                "deployed_by_run_id": deployed_run_id,
                "source_config_path": deployed_run_dir
                    .join("configs/candidate_primary.yaml")
                    .display()
                    .to_string()
            }),
        );

        let settings_path = project_dir.join("config/factory_defaults.yaml");
        ensure_parent_dir(&settings_path).expect("create settings parent");
        fs::write(&settings_path, "artifacts_dir: custom_artifacts\n").expect("write settings");

        prune_factory_artifacts(PruneFactoryArtifactsArgs {
            project_dir: Some(project_dir.to_path_buf()),
            settings: PathBuf::from(DEFAULT_FACTORY_SETTINGS),
            profile: FactoryArtifactsProfile::Nightly,
            dry_run: false,
            verbose: false,
        })
        .expect("prune factory artifacts succeeds");

        assert!(
            deployed_run_dir.is_dir(),
            "deployed run from fixed paper state path should remain"
        );
        assert!(latest_run_dir.is_dir(), "latest run should remain");
        assert!(!stale_run_dir.exists(), "stale run should be removed");
        assert!(
            custom_artifacts_dir
                .join(format!("_effective_configs/{deployed_run_id}.yaml"))
                .is_file(),
            "deployed effective config should remain"
        );
    }

    #[test]
    fn retired_writes_payload_when_requested() {
        let dir = tempdir().expect("tempdir");
        let json_path = dir.path().join("retired.json");
        retired(RetiredArgs {
            name: "factory".to_string(),
            message: "Factory automation has been retired.".to_string(),
            json_path: Some(json_path.clone()),
            status: RetiredStatus::Blocked,
        })
        .expect("retired command succeeds");

        let payload = fs::read_to_string(json_path).expect("retired payload");
        assert!(payload.contains("\"retired\": true"));
        assert!(payload.contains("\"status\": \"blocked\""));
    }

    fn create_run_dir(artifacts_dir: &Path, date: &str, run_id: &str) -> PathBuf {
        let run_dir = artifacts_dir.join(date).join(format!("run_{run_id}"));
        fs::create_dir_all(run_dir.join("promoted_configs")).expect("create run dir");
        fs::write(run_dir.join("run_metadata.json"), "{}").expect("write run metadata");
        run_dir
    }

    fn write_effective_config_stub(artifacts_dir: &Path, run_id: &str) {
        let path = artifacts_dir
            .join("_effective_configs")
            .join(format!("{run_id}.yaml"));
        ensure_parent_dir(&path).expect("create effective config parent");
        fs::write(path, "global: {}\n").expect("write effective config");
    }

    fn write_json_file(path: &Path, value: &serde_json::Value) {
        ensure_parent_dir(path).expect("create JSON parent");
        fs::write(
            path,
            serde_json::to_vec_pretty(value).expect("serialise JSON"),
        )
        .expect("write JSON");
    }
}
