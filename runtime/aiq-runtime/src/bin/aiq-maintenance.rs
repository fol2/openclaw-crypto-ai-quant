use anyhow::{anyhow, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use reqwest::blocking::Client;
use reqwest::StatusCode;
use rusqlite::{params, Connection, OptionalExtension};
use serde::Deserialize;
use serde_json::json;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const DEFAULT_HL_INFO_URL: &str = "https://api.hyperliquid.xyz/info";
const DEFAULT_FUNDING_DB: &str = "candles_dbs/funding_rates.db";
const DEFAULT_RUNTIME_DB: &str = "trading_engine.db";
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

#[derive(Args, Debug)]
struct PruneRuntimeLogsArgs {
    #[arg(long)]
    db: Option<PathBuf>,
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

fn prune_runtime_logs(args: PruneRuntimeLogsArgs) -> Result<()> {
    let db_path = resolve_runtime_db(args.db);
    if !db_path.exists() {
        return Ok(());
    }

    let keep_days = args
        .keep_days
        .unwrap_or_else(|| env_f64("AI_QUANT_RUNTIME_LOG_KEEP_DAYS").unwrap_or(DEFAULT_KEEP_DAYS))
        .max(0.0);
    let runtime_cutoff_ts_ms = cutoff_ts_ms(keep_days)?;
    let tunnel_cutoff_ts_ms = cutoff_ts_ms(EXIT_TUNNEL_KEEP_DAYS)?;

    let mut attempt = 0u32;
    loop {
        let conn = Connection::open(&db_path)
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

            if args.dry_run {
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

            if args.vacuum {
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

fn resolve_runtime_db(value: Option<PathBuf>) -> PathBuf {
    if let Some(path) = value {
        return normalise_path(&path);
    }

    if let Ok(path) = std::env::var("AI_QUANT_DB_PATH") {
        if !path.trim().is_empty() {
            return normalise_path(Path::new(&path));
        }
    }

    normalise_path(Path::new(DEFAULT_RUNTIME_DB))
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
            db: Some(db_path.clone()),
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
}
