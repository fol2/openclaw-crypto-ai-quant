#![cfg(unix)]

use libc::{flock, kill, LOCK_EX, LOCK_NB, SIGTERM};
use rusqlite::Connection;
use serde_json::Value;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::{tempdir, TempDir};

const START_STEP_CLOSE_TS_MS: i64 = 1_773_422_400_000;
const LAST_STEP_CLOSE_TS_MS: i64 = 1_773_426_000_000;

#[derive(Debug)]
struct Fixture {
    _dir: TempDir,
    paper_db: PathBuf,
    candles_db: PathBuf,
    lock_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DbSnapshot {
    db: Option<Vec<u8>>,
    wal: Option<Vec<u8>>,
    shm: Option<Vec<u8>>,
}

#[test]
fn paper_daemon_lock_contention_fails_closed_before_any_write() {
    let fixture = prepare_idle_fixture();
    let before = snapshot_db(&fixture.paper_db);
    let _lock_guard = hold_exclusive_lock(&fixture.lock_path);

    let output = daemon_command(&fixture)
        .arg("--idle-sleep-ms")
        .arg("1")
        .arg("--max-idle-polls")
        .arg("1")
        .output()
        .expect("paper daemon lock-contention smoke should spawn");

    assert!(
        !output.status.success(),
        "paper daemon should fail closed when the lane lock is already held; output:\n{}",
        combined_output(&output)
    );
    assert!(
        combined_output(&output).to_ascii_lowercase().contains("lock"),
        "paper daemon lock-contention error should mention the lock path or lock state; output:\n{}",
        combined_output(&output)
    );
    assert_eq!(
        before,
        snapshot_db(&fixture.paper_db),
        "lock contention must not mutate the paper DB",
    );
}

#[test]
fn paper_daemon_idle_follow_poll_does_not_write_when_caught_up() {
    let fixture = prepare_idle_fixture();
    let before = snapshot_db(&fixture.paper_db);

    let output = daemon_command(&fixture)
        .arg("--idle-sleep-ms")
        .arg("1")
        .arg("--max-idle-polls")
        .arg("1")
        .output()
        .expect("paper daemon idle smoke should spawn");

    assert!(
        output.status.success(),
        "paper daemon should exit cleanly after one idle poll when caught up; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    assert_eq!(
        report.pointer("/lock_path").and_then(Value::as_str),
        Some(fixture.lock_path.to_string_lossy().as_ref()),
        "paper daemon should report the acquired lock path",
    );
    assert_eq!(
        report
            .pointer("/loop_report/executed_steps")
            .and_then(Value::as_u64),
        Some(0),
        "caught-up daemon follow mode should not execute any new step",
    );
    assert_eq!(
        report
            .pointer("/loop_report/idle_polls")
            .and_then(Value::as_u64),
        Some(1),
        "max_idle_polls=1 should stop on the first idle poll",
    );
    assert_eq!(
        report.pointer("/stop_requested").and_then(Value::as_bool),
        Some(false),
        "idle poll budget exit should not be marked as an operator stop request",
    );
    assert!(
        report
            .pointer("/loop_report/warnings")
            .and_then(Value::as_array)
            .is_some_and(|warnings| warnings.iter().any(|warning| {
                warning
                    .as_str()
                    .is_some_and(|text| text.contains("paper loop idle:"))
            })),
        "idle exit should carry forward the paper loop idle warning",
    );
    assert_eq!(
        before,
        snapshot_db(&fixture.paper_db),
        "idle follow polls must not mutate the paper DB",
    );
    assert!(
        try_exclusive_lock(&fixture.lock_path).is_some(),
        "paper daemon should release its lock after an idle-budget exit",
    );
}

#[test]
fn paper_daemon_sigterm_while_idle_stops_gracefully_without_writes() {
    let fixture = prepare_idle_fixture();
    let before = snapshot_db(&fixture.paper_db);
    let child = daemon_command(&fixture)
        .arg("--idle-sleep-ms")
        .arg("250")
        .arg("--max-idle-polls")
        .arg("0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon SIGTERM smoke should spawn");

    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    thread::sleep(Duration::from_millis(150));
    send_sigterm(&child);

    let output = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        output.status.success(),
        "paper daemon should exit cleanly after SIGTERM during idle follow mode; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    assert_eq!(
        report.pointer("/lock_path").and_then(Value::as_str),
        Some(fixture.lock_path.to_string_lossy().as_ref()),
        "paper daemon should report the acquired lock path",
    );
    assert_eq!(
        report
            .pointer("/loop_report/executed_steps")
            .and_then(Value::as_u64),
        Some(0),
        "caught-up daemon should remain write-idle before SIGTERM lands",
    );
    assert!(
        report.pointer("/stop_requested").and_then(Value::as_bool) == Some(true)
            || report
                .pointer("/loop_report/stop_requested")
                .and_then(Value::as_bool)
                == Some(true),
        "SIGTERM exit should report a graceful stop request in the JSON report",
    );
    assert_eq!(
        before,
        snapshot_db(&fixture.paper_db),
        "SIGTERM during idle follow mode must not mutate the paper DB",
    );
    assert!(
        try_exclusive_lock(&fixture.lock_path).is_some(),
        "paper daemon should release its lock after a graceful SIGTERM exit",
    );
}

#[test]
fn paper_daemon_reloads_symbols_file_between_idle_polls() {
    let fixture = prepare_idle_fixture();
    let symbols_file = fixture._dir.path().join("symbols.txt");
    fs::write(&symbols_file, "ETH\n").expect("initial symbols file should be written");

    let child = daemon_command_with_symbols_file(&fixture, &symbols_file)
        .arg("--idle-sleep-ms")
        .arg("50")
        .arg("--max-idle-polls")
        .arg("2")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon symbols-file reload smoke should spawn");

    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    thread::sleep(Duration::from_millis(25));
    fs::write(&symbols_file, "ETH\nBTC\n").expect("updated symbols file should be written");

    let output = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        output.status.success(),
        "paper daemon should exit cleanly after reloading a symbols file between idle polls; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    assert_eq!(
        report
            .pointer("/loop_report/symbols_file_reload_count")
            .and_then(Value::as_u64),
        Some(1),
        "paper daemon should report exactly one symbols-file reload",
    );
    assert_eq!(
        report
            .pointer("/loop_report/latest_explicit_symbols")
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_str).collect::<Vec<_>>()),
        Some(vec!["BTC", "ETH"]),
        "paper daemon should surface the refreshed explicit symbol set",
    );
    assert!(
        report
            .pointer("/loop_report/warnings")
            .and_then(Value::as_array)
            .is_some_and(|warnings| warnings.iter().any(|warning| {
                warning
                    .as_str()
                    .is_some_and(|text| text.contains("reloaded symbols file"))
            })),
        "paper daemon should surface the symbols-file reload warning",
    );
}

#[test]
fn paper_daemon_can_idle_on_empty_symbols_file_until_a_due_step_becomes_runnable() {
    let dir = tempdir().expect("fixture tempdir should be created");
    let paper_db = dir.path().join("paper.db");
    let candles_db = dir.path().join("candles.db");
    let lock_path = dir.path().join("paper-daemon.lock");
    seed_empty_paper_db(&paper_db);
    seed_candles_db(&candles_db);
    let fixture = Fixture {
        _dir: dir,
        paper_db,
        candles_db,
        lock_path,
    };
    let symbols_file = fixture._dir.path().join("symbols-empty-then-eth.txt");
    fs::write(&symbols_file, "").expect("empty symbols file should be written");
    let before = snapshot_db(&fixture.paper_db);

    let child = daemon_command_with_symbols_file(&fixture, &symbols_file)
        .arg("--start-step-close-ts-ms")
        .arg(LAST_STEP_CLOSE_TS_MS.to_string())
        .arg("--idle-sleep-ms")
        .arg("25")
        .arg("--max-idle-polls")
        .arg("4")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon empty-watchlist follow smoke should spawn");

    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    thread::sleep(Duration::from_millis(50));
    fs::write(&symbols_file, "ETH\n").expect("symbols file should update to ETH");

    let output = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        output.status.success(),
        "paper daemon should pick up a later symbols-file update and execute a due step; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    assert_eq!(
        report
            .pointer("/loop_report/executed_steps")
            .and_then(Value::as_u64),
        Some(1),
        "a later symbols-file update should let the daemon execute one due step without restart",
    );
    assert!(
        report
            .pointer("/loop_report/warnings")
            .and_then(Value::as_array)
            .is_some_and(|warnings| warnings.iter().any(|warning| {
                warning
                    .as_str()
                    .is_some_and(|text| text.contains("no active symbols available yet"))
            })),
        "daemon should report the initial empty-watchlist idle state before the symbols-file update arrives",
    );
    assert!(
        report
            .pointer("/loop_report/steps/0/active_symbols")
            .and_then(Value::as_array)
            .is_some_and(|symbols| {
                symbols
                    .iter()
                    .filter_map(Value::as_str)
                    .any(|symbol| symbol == "ETH")
            }),
        "the executed step should reflect the symbol loaded from the updated symbols file",
    );
    assert_ne!(
        before,
        snapshot_db(&fixture.paper_db),
        "once the symbols file becomes runnable, the daemon should advance the paper DB",
    );
    assert!(
        try_exclusive_lock(&fixture.lock_path).is_some(),
        "paper daemon should release its lock after the empty-watchlist follow run completes",
    );
}

fn prepare_idle_fixture() -> Fixture {
    let fixture = seed_fixture();
    let output = loop_command(&fixture)
        .arg("--start-step-close-ts-ms")
        .arg(START_STEP_CLOSE_TS_MS.to_string())
        .arg("--max-steps")
        .arg("3")
        .output()
        .expect("paper loop bootstrap should spawn");

    assert!(
        output.status.success(),
        "paper loop bootstrap should catch the fixture up before daemon smoke tests; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    assert_eq!(
        report.pointer("/executed_steps").and_then(Value::as_u64),
        Some(3),
        "fixture bootstrap should apply the three due cycle steps",
    );
    assert_eq!(
        report
            .pointer("/latest_common_close_ts_ms")
            .and_then(Value::as_i64),
        Some(LAST_STEP_CLOSE_TS_MS),
        "fixture bootstrap should stop at the latest common candle close",
    );

    fixture
}

fn seed_fixture() -> Fixture {
    let dir = tempdir().expect("fixture tempdir should be created");
    let paper_db = dir.path().join("paper.db");
    let candles_db = dir.path().join("candles.db");
    let lock_path = dir.path().join("paper-daemon.lock");
    seed_paper_db(&paper_db);
    seed_candles_db(&candles_db);
    Fixture {
        _dir: dir,
        paper_db,
        candles_db,
        lock_path,
    }
}

fn runtime_command() -> Command {
    let mut command = Command::new(runtime_bin());
    command.current_dir(workspace_root());
    command
}

fn loop_command(fixture: &Fixture) -> Command {
    let mut command = runtime_command();
    command
        .arg("paper")
        .arg("loop")
        .arg("--config")
        .arg(config_path())
        .arg("--db")
        .arg(&fixture.paper_db)
        .arg("--candles-db")
        .arg(&fixture.candles_db)
        .arg("--symbols")
        .arg("ETH")
        .arg("--json");
    command
}

fn daemon_command(fixture: &Fixture) -> Command {
    let mut command = runtime_command();
    command
        .arg("paper")
        .arg("daemon")
        .arg("--config")
        .arg(config_path())
        .arg("--db")
        .arg(&fixture.paper_db)
        .arg("--candles-db")
        .arg(&fixture.candles_db)
        .arg("--symbols")
        .arg("ETH")
        .arg("--lock-path")
        .arg(&fixture.lock_path)
        .arg("--json");
    command
}

fn daemon_command_with_symbols_file(fixture: &Fixture, symbols_file: &Path) -> Command {
    let mut command = runtime_command();
    command
        .arg("paper")
        .arg("daemon")
        .arg("--config")
        .arg(config_path())
        .arg("--db")
        .arg(&fixture.paper_db)
        .arg("--candles-db")
        .arg(&fixture.candles_db)
        .arg("--symbols-file")
        .arg(symbols_file)
        .arg("--lock-path")
        .arg(&fixture.lock_path)
        .arg("--json");
    command
}

fn runtime_bin() -> PathBuf {
    std::env::var_os("CARGO_BIN_EXE_aiq-runtime")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("target/debug/aiq-runtime"))
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root should resolve")
}

fn config_path() -> PathBuf {
    workspace_root().join("config/strategy_overrides.yaml.example")
}

fn parse_json_output(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|err| {
        panic!(
            "command should emit JSON on stdout: {err}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
    })
}

fn combined_output(output: &Output) -> String {
    format!(
        "{}{}{}",
        String::from_utf8_lossy(&output.stdout),
        if output.stdout.is_empty() || output.stderr.is_empty() {
            ""
        } else {
            "\n"
        },
        String::from_utf8_lossy(&output.stderr),
    )
}

fn snapshot_db(paper_db: &Path) -> DbSnapshot {
    DbSnapshot {
        db: read_optional_bytes(paper_db),
        wal: read_optional_bytes(&wal_path(paper_db)),
        shm: read_optional_bytes(&shm_path(paper_db)),
    }
}

fn read_optional_bytes(path: &Path) -> Option<Vec<u8>> {
    match fs::read(path) {
        Ok(bytes) => Some(bytes),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => panic!("failed to read {}: {err}", path.display()),
    }
}

fn wal_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}-wal", path.display()))
}

fn shm_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}-shm", path.display()))
}

fn hold_exclusive_lock(path: &Path) -> File {
    let mut file = open_lock_file(path);
    write!(&mut file, "{}", std::process::id()).expect("lock holder should write its pid");
    file.flush().expect("lock holder should flush its pid");
    let rc = unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) };
    assert_eq!(
        rc,
        0,
        "test fixture should acquire the exclusive daemon lock: {}",
        std::io::Error::last_os_error()
    );
    file
}

fn try_exclusive_lock(path: &Path) -> Option<File> {
    let file = open_lock_file(path);
    let rc = unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) };
    if rc == 0 {
        Some(file)
    } else {
        None
    }
}

fn open_lock_file(path: &Path) -> File {
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .unwrap_or_else(|err| panic!("failed to open lock file {}: {err}", path.display()))
}

fn wait_for_lock_file(mut child: Child, lock_path: &Path, timeout: Duration) -> Child {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if lock_path.exists() {
            return child;
        }
        if let Some(status) = child
            .try_wait()
            .expect("daemon wait status should be readable")
        {
            let output = child
                .wait_with_output()
                .expect("daemon output should be readable after an early exit");
            panic!(
                "paper daemon exited before taking the lane lock: {status}\n{}",
                combined_output(&output)
            );
        }
        thread::sleep(Duration::from_millis(25));
    }

    let output = wait_with_timeout(child, Duration::from_secs(1));
    panic!(
        "paper daemon did not materialise the lock file {} before timeout\n{}",
        lock_path.display(),
        combined_output(&output)
    );
}

fn send_sigterm(child: &Child) {
    let rc = unsafe { kill(child.id() as i32, SIGTERM) };
    assert_eq!(
        rc,
        0,
        "SIGTERM should be delivered to the paper daemon: {}",
        std::io::Error::last_os_error()
    );
}

fn wait_with_timeout(mut child: Child, timeout: Duration) -> Output {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if child
            .try_wait()
            .expect("child wait status should be readable")
            .is_some()
        {
            return child
                .wait_with_output()
                .expect("child output should be readable after exit");
        }
        thread::sleep(Duration::from_millis(25));
    }

    let _ = child.kill();
    let output = child
        .wait_with_output()
        .expect("child output should be readable after timeout kill");
    panic!(
        "command exceeded the {} ms timeout\n{}",
        timeout.as_millis(),
        combined_output(&output)
    );
}

fn create_paper_db_schema(path: &Path) {
    let conn = Connection::open(path).expect("paper db should open");
    conn.execute_batch(
        r#"
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            type TEXT,
            action TEXT,
            price REAL,
            size REAL,
            notional REAL,
            reason TEXT,
            reason_code TEXT,
            confidence TEXT,
            pnl REAL,
            fee_usd REAL,
            fee_token TEXT,
            fee_rate REAL,
            balance REAL,
            entry_atr REAL,
            leverage REAL,
            margin_used REAL,
            meta_json TEXT,
            run_fingerprint TEXT,
            fill_hash TEXT,
            fill_tid INTEGER
        );
        CREATE TABLE position_state (
            symbol TEXT PRIMARY KEY,
            open_trade_id INTEGER,
            trailing_sl REAL,
            last_funding_time INTEGER,
            adds_count INTEGER,
            tp1_taken INTEGER,
            last_add_time INTEGER,
            entry_adx_threshold REAL,
            updated_at TEXT
        );
        CREATE TABLE runtime_cooldowns (
            symbol TEXT PRIMARY KEY,
            last_entry_attempt_s REAL,
            last_exit_attempt_s REAL,
            updated_at TEXT
        );
        CREATE TABLE runtime_last_closes (
            symbol TEXT PRIMARY KEY,
            close_ts_ms INTEGER NOT NULL,
            side TEXT NOT NULL,
            reason TEXT,
            updated_at TEXT NOT NULL
        );
        "#,
    )
    .expect("paper schema should be created");
}

fn seed_empty_paper_db(path: &Path) {
    create_paper_db_schema(path);
}

fn seed_paper_db(path: &Path) {
    create_paper_db_schema(path);
    let conn = Connection::open(path).expect("paper db should open");
    conn.execute(
        "INSERT INTO trades (timestamp,symbol,action,type,price,size,notional,reason,confidence,balance,pnl,fee_usd,fee_rate,entry_atr,leverage,margin_used,meta_json)
         VALUES ('2026-03-05T10:00:00+00:00','ETH','OPEN','LONG',100.0,1.0,100.0,'seed','medium',1000.0,0.0,0.0,0.0,5.0,3.0,33.3,'{}')",
        [],
    )
    .expect("seed trade should be inserted");
    conn.execute(
        "INSERT INTO position_state VALUES ('ETH',1,95.0,1772676500000,0,0,1772676600000,22.0,'2026-03-05T10:08:20+00:00')",
        [],
    )
    .expect("seed position should be inserted");
    conn.execute(
        "INSERT INTO runtime_cooldowns VALUES ('ETH',1772676500.0,1772676550.0,'2026-03-05T10:15:00+00:00')",
        [],
    )
    .expect("seed cooldown should be inserted");
}

fn seed_candles_db(path: &Path) {
    let conn = Connection::open(path).expect("candles db should open");
    conn.execute_batch(
        r#"
        CREATE TABLE candles (
            symbol TEXT,
            interval TEXT,
            t INTEGER,
            t_close INTEGER,
            o REAL,
            h REAL,
            l REAL,
            c REAL,
            v REAL,
            n INTEGER
        );
        "#,
    )
    .expect("candles schema should be created");

    let base = 1_772_670_000_000_i64;
    for (symbol, start, drift) in [
        ("ETH", 100.0_f64, 0.25_f64),
        ("BTC", 50_000.0_f64, 20.0_f64),
    ] {
        let mut price = start;
        for idx in 0..420_i64 {
            let t = base + (idx * 1_800_000);
            let open = price;
            let close = price + drift;
            let high = open.max(close) + 0.5;
            let low = open.min(close) - 0.5;
            let volume = 1000.0 + idx as f64;
            conn.execute(
                "INSERT INTO candles VALUES (?1, '30m', ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1)",
                (symbol, t, t + 1_800_000, open, high, low, close, volume),
            )
            .expect("seed candle should be inserted");
            price = close;
        }
    }
}
