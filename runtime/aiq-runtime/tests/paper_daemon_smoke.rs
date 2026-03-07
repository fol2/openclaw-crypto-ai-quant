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
const NEXT_STEP_CLOSE_TS_MS: i64 = LAST_STEP_CLOSE_TS_MS + 1_800_000;

#[derive(Debug)]
struct Fixture {
    _dir: TempDir,
    paper_db: PathBuf,
    candles_db: PathBuf,
    lock_path: PathBuf,
    status_path: PathBuf,
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
        report.pointer("/status_path").and_then(Value::as_str),
        Some(fixture.status_path.to_string_lossy().as_ref()),
        "paper daemon should report the configured status path",
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
                    .is_some_and(|text| text.contains("paper daemon idle:"))
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
fn paper_daemon_writes_status_file_for_running_and_stopped_lifecycle() {
    let fixture = prepare_idle_fixture();
    let child = daemon_command(&fixture)
        .arg("--idle-sleep-ms")
        .arg("250")
        .arg("--max-idle-polls")
        .arg("0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon status-path smoke should spawn");

    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    let running_status = wait_for_status_file(&fixture.status_path, Duration::from_secs(5));
    assert_eq!(
        running_status.pointer("/running").and_then(Value::as_bool),
        Some(true),
        "daemon should mark the status file as running while follow mode is active",
    );
    assert_eq!(
        running_status
            .pointer("/status_path")
            .and_then(Value::as_str),
        Some(fixture.status_path.to_string_lossy().as_ref()),
        "daemon should report the configured status path in the live status file",
    );
    assert_eq!(
        running_status.pointer("/lock_path").and_then(Value::as_str),
        Some(fixture.lock_path.to_string_lossy().as_ref()),
        "daemon should report the active lock path in the live status file",
    );

    send_sigterm(&child);
    let output = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        output.status.success(),
        "paper daemon should exit cleanly after SIGTERM in the status lifecycle smoke; output:\n{}",
        combined_output(&output)
    );

    let stopped_status = read_json_file(&fixture.status_path);
    assert_eq!(
        stopped_status.pointer("/running").and_then(Value::as_bool),
        Some(false),
        "daemon should mark the persisted status file as stopped after exit",
    );
    assert_eq!(
        stopped_status
            .pointer("/stop_requested")
            .and_then(Value::as_bool),
        Some(true),
        "SIGTERM exit should be reflected in the persisted status file",
    );
    assert!(
        stopped_status
            .pointer("/stopped_at_ms")
            .and_then(Value::as_i64)
            .is_some(),
        "persisted status should include a stopped timestamp",
    );
}

#[test]
fn paper_daemon_resumed_start_step_does_not_trip_status_restart_required() {
    let fixture = prepare_idle_fixture();
    let child = daemon_command(&fixture)
        .arg("--start-step-close-ts-ms")
        .arg(NEXT_STEP_CLOSE_TS_MS.to_string())
        .arg("--idle-sleep-ms")
        .arg("250")
        .arg("--max-idle-polls")
        .arg("0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon resumed start-step smoke should spawn");

    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    let _running_status = wait_for_status_file(&fixture.status_path, Duration::from_secs(5));

    let output = status_command(&fixture)
        .arg("--stale-after-ms")
        .arg("60000")
        .output()
        .expect("paper status smoke should spawn");
    assert!(
        output.status.success(),
        "paper status smoke should exit successfully for a resumed daemon lane; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    assert_eq!(
        report.pointer("/service_state").and_then(Value::as_str),
        Some("running"),
        "resumed lanes should not be marked restart-required just because the daemon was started with a redundant start-step flag",
    );
    assert_eq!(
        report
            .pointer("/contract_matches_status")
            .and_then(Value::as_bool),
        Some(true),
        "resumed lanes should still match the launch contract when the supervisor omits a redundant start-step flag",
    );

    send_sigterm(&child);
    let stopped = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        stopped.status.success(),
        "paper daemon should exit cleanly after the resumed-lane smoke; output:\n{}",
        combined_output(&stopped)
    );
}

#[test]
fn paper_daemon_symbols_file_single_symbol_keeps_status_contract_aligned() {
    let fixture = seed_fixture();
    let config_path = fixture._dir.path().join("strategy.yaml");
    let symbols_file = fixture._dir.path().join("symbols.txt");
    fs::write(
        &config_path,
        r#"
global:
  engine:
    interval: 30m
  runtime:
    profile: production
symbols:
  ETH:
    runtime:
      profile: parity_baseline
"#,
    )
    .unwrap();
    fs::write(&symbols_file, "ETH\n").unwrap();

    let bootstrap = runtime_command()
        .arg("paper")
        .arg("loop")
        .arg("--config")
        .arg(&config_path)
        .arg("--db")
        .arg(&fixture.paper_db)
        .arg("--candles-db")
        .arg(&fixture.candles_db)
        .arg("--symbols-file")
        .arg(&symbols_file)
        .arg("--start-step-close-ts-ms")
        .arg(START_STEP_CLOSE_TS_MS.to_string())
        .arg("--max-steps")
        .arg("3")
        .arg("--json")
        .output()
        .expect("paper loop bootstrap for symbols-file lane should spawn");
    assert!(
        bootstrap.status.success(),
        "paper loop bootstrap should succeed for symbols-file lane; output:\n{}",
        combined_output(&bootstrap)
    );

    let child = runtime_command()
        .arg("paper")
        .arg("daemon")
        .arg("--config")
        .arg(&config_path)
        .arg("--db")
        .arg(&fixture.paper_db)
        .arg("--candles-db")
        .arg(&fixture.candles_db)
        .arg("--symbols-file")
        .arg(&symbols_file)
        .arg("--lock-path")
        .arg(&fixture.lock_path)
        .arg("--status-path")
        .arg(&fixture.status_path)
        .arg("--idle-sleep-ms")
        .arg("250")
        .arg("--max-idle-polls")
        .arg("0")
        .arg("--json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon symbols-file smoke should spawn");

    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    let _running_status = wait_for_status_file(&fixture.status_path, Duration::from_secs(5));

    let output = runtime_command()
        .arg("paper")
        .arg("status")
        .arg("--config")
        .arg(&config_path)
        .arg("--db")
        .arg(&fixture.paper_db)
        .arg("--candles-db")
        .arg(&fixture.candles_db)
        .arg("--symbols-file")
        .arg(&symbols_file)
        .arg("--lock-path")
        .arg(&fixture.lock_path)
        .arg("--status-path")
        .arg(&fixture.status_path)
        .arg("--stale-after-ms")
        .arg("60000")
        .arg("--json")
        .output()
        .expect("paper status symbols-file smoke should spawn");
    assert!(
        output.status.success(),
        "paper status smoke should exit successfully for a symbols-file lane; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    assert_eq!(
        report.pointer("/service_state").and_then(Value::as_str),
        Some("running"),
        "symbols-file bootstrap lanes should remain running once the daemon and manifest resolve the same effective config",
    );
    assert_eq!(
        report
            .pointer("/contract_matches_status")
            .and_then(Value::as_bool),
        Some(true),
        "symbols-file bootstrap lanes should not drift on profile or fingerprint",
    );
    assert_eq!(
        report
            .pointer("/manifest/runtime_bootstrap/pipeline/profile")
            .and_then(Value::as_str),
        Some("parity_baseline"),
        "single-symbol symbols-file lanes should use the symbol-effective runtime profile",
    );

    send_sigterm(&child);
    let stopped = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        stopped.status.success(),
        "paper daemon should exit cleanly after the symbols-file smoke; output:\n{}",
        combined_output(&stopped)
    );
}

#[test]
fn paper_daemon_reloaded_single_symbol_updates_status_contract() {
    let fixture = seed_empty_fixture();
    let config_path = fixture._dir.path().join("strategy.yaml");
    let symbols_file = fixture._dir.path().join("symbols.txt");
    fs::write(
        &config_path,
        r#"
global:
  engine:
    interval: 30m
  runtime:
    profile: production
symbols:
  ETH:
    runtime:
      profile: parity_baseline
"#,
    )
    .unwrap();
    fs::write(&symbols_file, "").unwrap();

    let child = runtime_command()
        .arg("paper")
        .arg("daemon")
        .arg("--config")
        .arg(&config_path)
        .arg("--db")
        .arg(&fixture.paper_db)
        .arg("--candles-db")
        .arg(&fixture.candles_db)
        .arg("--symbols-file")
        .arg(&symbols_file)
        .arg("--watch-symbols-file")
        .arg("--start-step-close-ts-ms")
        .arg(START_STEP_CLOSE_TS_MS.to_string())
        .arg("--lock-path")
        .arg(&fixture.lock_path)
        .arg("--status-path")
        .arg(&fixture.status_path)
        .arg("--idle-sleep-ms")
        .arg("20")
        .arg("--max-idle-polls")
        .arg("20")
        .arg("--json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon empty-watchlist reload contract smoke should spawn");

    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    let _running_status = wait_for_status_file(&fixture.status_path, Duration::from_secs(5));
    thread::sleep(Duration::from_millis(100));
    fs::write(&symbols_file, "ETH\n").unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let status = read_json_file(&fixture.status_path);
        let active_symbols = status
            .pointer("/last_active_symbols")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if active_symbols == vec![Value::String("ETH".to_string())] {
            break;
        }
        assert!(
            Instant::now() <= deadline,
            "paper daemon did not refresh to ETH before timeout; status={status}",
        );
        thread::sleep(Duration::from_millis(20));
    }

    let output = runtime_command()
        .arg("paper")
        .arg("status")
        .arg("--config")
        .arg(&config_path)
        .arg("--db")
        .arg(&fixture.paper_db)
        .arg("--candles-db")
        .arg(&fixture.candles_db)
        .arg("--symbols-file")
        .arg(&symbols_file)
        .arg("--watch-symbols-file")
        .arg("--start-step-close-ts-ms")
        .arg(START_STEP_CLOSE_TS_MS.to_string())
        .arg("--lock-path")
        .arg(&fixture.lock_path)
        .arg("--status-path")
        .arg(&fixture.status_path)
        .arg("--stale-after-ms")
        .arg("60000")
        .arg("--json")
        .output()
        .expect("paper status empty-watchlist reload smoke should spawn");
    assert!(
        output.status.success(),
        "paper status smoke should exit successfully after a symbols-file reload; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    assert_eq!(
        report.pointer("/service_state").and_then(Value::as_str),
        Some("running"),
        "reloaded single-symbol lanes should remain running once the daemon refreshes its bootstrap contract",
    );
    assert_eq!(
        report
            .pointer("/contract_matches_status")
            .and_then(Value::as_bool),
        Some(true),
        "reloaded single-symbol lanes should not be marked drifted after the daemon recomputes the bootstrap contract",
    );
    assert_eq!(
        report
            .pointer("/manifest/runtime_bootstrap/pipeline/profile")
            .and_then(Value::as_str),
        Some("parity_baseline"),
        "reloaded single-symbol lanes should expose the symbol-effective runtime profile",
    );

    send_sigterm(&child);
    let stopped = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        stopped.status.success(),
        "paper daemon should exit cleanly after the empty-watchlist reload smoke; output:\n{}",
        combined_output(&stopped)
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

#[test]
fn paper_service_apply_starts_bootstrap_ready_lane() {
    let fixture = seed_empty_fixture();

    let output = service_apply_command(&fixture)
        .arg("--start-step-close-ts-ms")
        .arg(START_STEP_CLOSE_TS_MS.to_string())
        .arg("--action")
        .arg("auto")
        .output()
        .expect("paper service apply bootstrap smoke should spawn");

    assert!(
        output.status.success(),
        "paper service apply should start a bootstrap-ready lane; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    let spawned_pid = report
        .pointer("/spawned_pid")
        .and_then(Value::as_u64)
        .expect("apply should report the spawned pid") as i32;
    assert_eq!(
        report.pointer("/applied_action").and_then(Value::as_str),
        Some("start"),
        "bootstrap-ready apply should execute a start action",
    );
    let running_status = wait_for_status_pid(
        &fixture.status_path,
        spawned_pid as u32,
        true,
        Duration::from_secs(5),
    );
    assert_eq!(
        running_status.pointer("/running").and_then(Value::as_bool),
        Some(true),
        "the supervised daemon should publish a running status contract",
    );

    let service = parse_json_output(
        &service_command(&fixture)
            .arg("--start-step-close-ts-ms")
            .arg(START_STEP_CLOSE_TS_MS.to_string())
            .output()
            .expect("paper service follow-up should spawn"),
    );
    assert_eq!(
        service.pointer("/desired_action").and_then(Value::as_str),
        Some("monitor"),
        "once the daemon is running, the read-only service view should become monitor",
    );

    send_sigterm_pid(spawned_pid);
    let stopped_status = wait_for_status_pid(
        &fixture.status_path,
        spawned_pid as u32,
        false,
        Duration::from_secs(5),
    );
    assert_eq!(
        stopped_status.pointer("/running").and_then(Value::as_bool),
        Some(false),
        "the bootstrap smoke daemon should stop cleanly during test cleanup",
    );
}

#[test]
fn paper_service_apply_resumes_stopped_lane() {
    let fixture = prepare_idle_fixture();
    let child = daemon_command(&fixture)
        .arg("--idle-sleep-ms")
        .arg("250")
        .arg("--max-idle-polls")
        .arg("0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon resume fixture should spawn");
    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    let running_status = wait_for_status_file(&fixture.status_path, Duration::from_secs(5));
    let previous_pid = running_status
        .pointer("/pid")
        .and_then(Value::as_u64)
        .expect("running daemon pid should be present") as i32;
    send_sigterm(&child);
    let stopped = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        stopped.status.success(),
        "the resume fixture daemon should stop cleanly; output:\n{}",
        combined_output(&stopped)
    );

    let output = service_apply_command(&fixture)
        .arg("--action")
        .arg("resume")
        .output()
        .expect("paper service resume smoke should spawn");
    assert!(
        output.status.success(),
        "paper service apply should resume a stopped launch-ready lane; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    let resumed_pid = report
        .pointer("/spawned_pid")
        .and_then(Value::as_u64)
        .expect("resume should report a spawned pid") as i32;
    assert_ne!(
        resumed_pid, previous_pid,
        "resume should relaunch the lane under a new daemon pid",
    );
    assert_eq!(
        report.pointer("/applied_action").and_then(Value::as_str),
        Some("start"),
        "resume is modelled as a supervised start of the stopped lane",
    );

    send_sigterm_pid(resumed_pid);
    let stopped_status = wait_for_status_pid(
        &fixture.status_path,
        resumed_pid as u32,
        false,
        Duration::from_secs(5),
    );
    assert_eq!(
        stopped_status.pointer("/running").and_then(Value::as_bool),
        Some(false),
        "the resumed daemon should stop cleanly during test cleanup",
    );
}

#[test]
fn paper_service_apply_is_noop_for_healthy_running_lane() {
    let fixture = prepare_idle_fixture();
    let child = daemon_command(&fixture)
        .arg("--idle-sleep-ms")
        .arg("250")
        .arg("--max-idle-polls")
        .arg("0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon noop fixture should spawn");
    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    let running_status = wait_for_status_file(&fixture.status_path, Duration::from_secs(5));
    let running_pid = running_status
        .pointer("/pid")
        .and_then(Value::as_u64)
        .expect("running daemon pid should be present") as i32;

    let output = service_apply_command(&fixture)
        .arg("--action")
        .arg("auto")
        .output()
        .expect("paper service noop smoke should spawn");
    assert!(
        output.status.success(),
        "paper service apply should no-op for a healthy running lane; output:\n{}",
        combined_output(&output)
    );
    let report = parse_json_output(&output);
    assert_eq!(
        report.pointer("/applied_action").and_then(Value::as_str),
        Some("noop"),
        "healthy running lanes should stay on the read-only monitor action",
    );
    assert_eq!(
        report.pointer("/previous_pid").and_then(Value::as_u64),
        Some(running_pid as u64),
        "no-op supervision should keep the same running pid",
    );
    assert!(
        report
            .pointer("/spawned_pid")
            .and_then(Value::as_u64)
            .is_none(),
        "no-op supervision must not spawn a replacement daemon",
    );

    send_sigterm(&child);
    let stopped = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        stopped.status.success(),
        "the noop fixture daemon should stop cleanly; output:\n{}",
        combined_output(&stopped)
    );
}

#[test]
fn paper_service_apply_restarts_stale_running_lane() {
    let fixture = prepare_idle_fixture();
    let child = daemon_command(&fixture)
        .arg("--idle-sleep-ms")
        .arg("250")
        .arg("--max-idle-polls")
        .arg("0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon restart fixture should spawn");
    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    let mut running_status = wait_for_status_file(&fixture.status_path, Duration::from_secs(5));
    let previous_pid = running_status
        .pointer("/pid")
        .and_then(Value::as_u64)
        .expect("running daemon pid should be present") as i32;
    running_status["updated_at_ms"] = Value::from(1);
    fs::write(
        &fixture.status_path,
        serde_json::to_vec_pretty(&running_status).expect("stale status should serialise"),
    )
    .expect("stale status should be written");

    let output = service_apply_command(&fixture)
        .arg("--action")
        .arg("auto")
        .arg("--stale-after-ms")
        .arg("1000")
        .output()
        .expect("paper service restart smoke should spawn");
    assert!(
        output.status.success(),
        "paper service apply should restart a stale running lane; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    let restarted_pid = report
        .pointer("/spawned_pid")
        .and_then(Value::as_u64)
        .expect("restart should report a spawned pid") as i32;
    assert_ne!(
        restarted_pid, previous_pid,
        "restart should replace the stale daemon owner with a new pid",
    );
    assert_eq!(
        report.pointer("/applied_action").and_then(Value::as_str),
        Some("restart"),
        "stale supervision should execute a restart action",
    );
    let stopped = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        stopped.status.success(),
        "the stale daemon should exit cleanly after supervision; output:\n{}",
        combined_output(&stopped)
    );

    let follow_up = parse_json_output(
        &service_command(&fixture)
            .arg("--stale-after-ms")
            .arg("60000")
            .output()
            .expect("paper service follow-up should spawn"),
    );
    assert_eq!(
        follow_up.pointer("/desired_action").and_then(Value::as_str),
        Some("monitor"),
        "the replacement daemon should settle into monitor state",
    );

    send_sigterm_pid(restarted_pid);
    let stopped_status = wait_for_status_pid(
        &fixture.status_path,
        restarted_pid as u32,
        false,
        Duration::from_secs(5),
    );
    assert_eq!(
        stopped_status.pointer("/running").and_then(Value::as_bool),
        Some(false),
        "the restarted daemon should stop cleanly during test cleanup",
    );
}

#[test]
fn paper_service_apply_stops_running_lane_when_requested() {
    let fixture = prepare_idle_fixture();
    let child = daemon_command(&fixture)
        .arg("--idle-sleep-ms")
        .arg("250")
        .arg("--max-idle-polls")
        .arg("0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon stop fixture should spawn");
    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    let running_status = wait_for_status_file(&fixture.status_path, Duration::from_secs(5));
    let previous_pid = running_status
        .pointer("/pid")
        .and_then(Value::as_u64)
        .expect("running daemon pid should be present") as i32;

    let output = service_apply_command(&fixture)
        .arg("--action")
        .arg("stop")
        .output()
        .expect("paper service stop smoke should spawn");
    assert!(
        output.status.success(),
        "paper service apply should stop a healthy running lane when requested; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    assert_eq!(
        report.pointer("/applied_action").and_then(Value::as_str),
        Some("stop"),
        "explicit stop should execute a stop action",
    );
    assert_eq!(
        report.pointer("/previous_pid").and_then(Value::as_u64),
        Some(previous_pid as u64),
        "explicit stop should target the current daemon owner",
    );
    let stopped_status = wait_for_status_pid(
        &fixture.status_path,
        previous_pid as u32,
        false,
        Duration::from_secs(5),
    );
    assert_eq!(
        stopped_status
            .pointer("/stop_requested")
            .and_then(Value::as_bool),
        Some(true),
        "explicit stop should persist a graceful stop request in the status contract",
    );
    assert!(
        try_exclusive_lock(&fixture.lock_path).is_some(),
        "explicit stop should release the daemon lock",
    );

    let stopped = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        stopped.status.success(),
        "the stop fixture daemon should exit cleanly after supervision; output:\n{}",
        combined_output(&stopped)
    );
}

#[test]
fn paper_service_apply_fails_closed_on_corrupt_status_file() {
    let fixture = prepare_idle_fixture();
    fs::write(&fixture.status_path, b"{not-json").expect("corrupt status should be written");

    let output = service_apply_command(&fixture)
        .arg("--action")
        .arg("auto")
        .output()
        .expect("paper service corrupt-status smoke should spawn");

    assert!(
        !output.status.success(),
        "paper service apply must fail closed on a corrupt status file; output:\n{}",
        combined_output(&output)
    );
    assert!(
        combined_output(&output)
            .to_ascii_lowercase()
            .contains("status"),
        "the corrupt-status failure should mention the status contract",
    );
}

#[test]
fn paper_daemon_reloads_symbols_file_after_empty_watchlist() {
    let fixture = seed_empty_fixture();
    let before = snapshot_db(&fixture.paper_db);
    let symbols_file = fixture._dir.path().join("symbols.txt");
    fs::write(&symbols_file, "").expect("empty symbols file should be created");

    let child = watchlist_daemon_command(&fixture, &symbols_file)
        .arg("--start-step-close-ts-ms")
        .arg(START_STEP_CLOSE_TS_MS.to_string())
        .arg("--idle-sleep-ms")
        .arg("20")
        .arg("--max-idle-polls")
        .arg("20")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon watchlist reload smoke should spawn");

    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    thread::sleep(Duration::from_millis(80));
    fs::write(&symbols_file, "ETH\n").expect("symbols file should update");

    let output = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        output.status.success(),
        "paper daemon should reload an empty symbols file and catch up cleanly; output:\n{}",
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
        Some(3),
        "daemon should execute all due steps once the watchlist file becomes active",
    );
    assert!(
        report
            .pointer("/loop_report/idle_polls")
            .and_then(Value::as_u64)
            .is_some_and(|idle_polls| idle_polls >= 1),
        "watchlist bootstrap should include at least one idle poll before symbols appear",
    );
    assert!(
        report
            .pointer("/watch_symbols_file")
            .and_then(Value::as_bool)
            == Some(true),
        "watchlist daemon smoke should enable daemon-owned file watching",
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
        "daemon should surface the empty-watchlist idle warning",
    );
    assert!(
        report
            .pointer("/loop_report/warnings")
            .and_then(Value::as_array)
            .is_some_and(|warnings| warnings.iter().any(|warning| {
                warning
                    .as_str()
                    .is_some_and(|text| text.contains("paper daemon reloaded symbols: ETH"))
            })),
        "daemon should surface the watchlist reload warning",
    );
    assert_eq!(
        report
            .pointer("/manifest_reload_count")
            .and_then(Value::as_u64),
        Some(1),
        "daemon should record one accepted manifest reload",
    );
    assert_eq!(
        report
            .pointer("/loop_report/steps/0/active_symbols/0")
            .and_then(Value::as_str),
        Some("ETH"),
        "the first executed step should use the symbols loaded from the watchlist file",
    );
    assert_ne!(
        before,
        snapshot_db(&fixture.paper_db),
        "watchlist bootstrap should mutate the paper DB once due steps execute",
    );
    assert!(
        try_exclusive_lock(&fixture.lock_path).is_some(),
        "paper daemon should release its lock after a watchlist-driven exit",
    );
}

#[test]
fn paper_daemon_retains_last_good_manifest_on_invalid_reload() {
    let fixture = prepare_idle_fixture();
    let symbols_file = fixture._dir.path().join("symbols.txt");
    fs::write(&symbols_file, "BTC\n").expect("initial symbols file should be created");

    let child = watchlist_daemon_command(&fixture, &symbols_file)
        .arg("--idle-sleep-ms")
        .arg("20")
        .arg("--max-idle-polls")
        .arg("0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon invalid-reload smoke should spawn");

    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    thread::sleep(Duration::from_millis(80));
    fs::write(&symbols_file, [0xff, 0xfe, 0x00]).expect("symbols file should become invalid");
    thread::sleep(Duration::from_millis(80));
    send_sigterm(&child);

    let output = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        output.status.success(),
        "paper daemon should stop cleanly after an invalid watchlist reload; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    assert_eq!(
        report
            .pointer("/manifest_reload_failure_count")
            .and_then(Value::as_u64),
        Some(1),
        "daemon should count the failed manifest reload",
    );
    assert_eq!(
        report
            .pointer("/manifest_symbols")
            .and_then(Value::as_array),
        Some(&vec![Value::String("BTC".to_string())]),
        "daemon should retain the last good manifest after an invalid reload",
    );
    assert!(
        report
            .pointer("/loop_report/warnings")
            .and_then(Value::as_array)
            .is_some_and(|warnings| warnings.iter().any(|warning| {
                warning.as_str().is_some_and(|text| {
                    text.contains("ignored symbols file reload")
                        && text.contains("retaining last good manifest")
                })
            })),
        "daemon should surface the invalid-reload retention warning",
    );
}

#[test]
fn paper_daemon_retains_last_good_manifest_on_semantically_torn_reload() {
    let fixture = prepare_idle_fixture();
    let symbols_file = fixture._dir.path().join("symbols-semantic-tear.txt");
    fs::write(&symbols_file, "BTC\n").expect("initial symbols file should be created");

    let child = watchlist_daemon_command(&fixture, &symbols_file)
        .arg("--idle-sleep-ms")
        .arg("20")
        .arg("--max-idle-polls")
        .arg("0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon semantically-torn reload smoke should spawn");

    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    thread::sleep(Duration::from_millis(80));
    fs::write(&symbols_file, "ET\n").expect("symbols file should become semantically torn");
    thread::sleep(Duration::from_millis(80));
    send_sigterm(&child);

    let output = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        output.status.success(),
        "paper daemon should stop cleanly after a semantically torn watchlist reload; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    assert_eq!(
        report
            .pointer("/manifest_reload_failure_count")
            .and_then(Value::as_u64),
        Some(1),
        "daemon should count a semantically torn reload as a manifest reload failure",
    );
    assert_eq!(
        report
            .pointer("/manifest_reload_count")
            .and_then(Value::as_u64),
        Some(0),
        "daemon should not count a rejected semantically torn reload as a successful manifest refresh",
    );
    assert_eq!(
        report
            .pointer("/manifest_symbols")
            .and_then(Value::as_array),
        Some(&vec![Value::String("BTC".to_string())]),
        "daemon should retain the last good manifest after a semantically torn reload",
    );
    assert!(
        report
            .pointer("/loop_report/warnings")
            .and_then(Value::as_array)
            .is_some_and(|warnings| warnings.iter().any(|warning| {
                warning.as_str().is_some_and(|text| {
                    text.contains("ignored symbols file reload")
                        && text.contains("retaining last good manifest")
                })
            })),
        "daemon should surface the semantically torn reload retention warning",
    );
}

#[test]
fn paper_daemon_retains_last_good_manifest_on_runtime_invalid_reload() {
    let fixture = prepare_idle_fixture();
    let config_path = fixture._dir.path().join("strategy-invalid-reload.yaml");
    let symbols_file = fixture._dir.path().join("symbols-runtime-invalid.txt");
    fs::write(
        &config_path,
        r#"
global:
  engine:
    interval: 30m
  runtime:
    profile: production
symbols:
  ETH:
    runtime:
      profile: production
    pipeline:
      profiles:
        production:
          enabled_stages: [ranking]
          disabled_stages: [ranking]
"#,
    )
    .unwrap();
    fs::write(&symbols_file, "BTC\n").expect("initial symbols file should be created");

    let child = runtime_command()
        .arg("paper")
        .arg("daemon")
        .arg("--config")
        .arg(&config_path)
        .arg("--db")
        .arg(&fixture.paper_db)
        .arg("--candles-db")
        .arg(&fixture.candles_db)
        .arg("--symbols-file")
        .arg(&symbols_file)
        .arg("--watch-symbols-file")
        .arg("--start-step-close-ts-ms")
        .arg(START_STEP_CLOSE_TS_MS.to_string())
        .arg("--lock-path")
        .arg(&fixture.lock_path)
        .arg("--status-path")
        .arg(&fixture.status_path)
        .arg("--idle-sleep-ms")
        .arg("20")
        .arg("--max-idle-polls")
        .arg("0")
        .arg("--json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon runtime-invalid reload smoke should spawn");

    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    thread::sleep(Duration::from_millis(80));
    fs::write(&symbols_file, "ETH\n").expect("symbols file should become runtime-invalid");
    thread::sleep(Duration::from_millis(80));
    send_sigterm(&child);

    let output = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        output.status.success(),
        "paper daemon should stop cleanly after a runtime-invalid watchlist reload; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    assert_eq!(
        report
            .pointer("/manifest_reload_failure_count")
            .and_then(Value::as_u64),
        Some(1),
        "daemon should count the runtime-invalid manifest reload",
    );
    assert_eq!(
        report
            .pointer("/manifest_symbols")
            .and_then(Value::as_array),
        Some(&vec![Value::String("BTC".to_string())]),
        "daemon should retain the last good manifest after a runtime-invalid reload",
    );
    assert!(
        report
            .pointer("/loop_report/warnings")
            .and_then(Value::as_array)
            .is_some_and(|warnings| warnings.iter().any(|warning| {
                warning.as_str().is_some_and(|text| {
                    text.contains("ignored symbols file reload")
                        && text.contains("retaining last good manifest")
                })
            })),
        "daemon should surface the runtime-invalid reload retention warning",
    );
}

#[test]
fn paper_daemon_noop_symbols_file_rewrite_does_not_increment_reload_count() {
    let fixture = prepare_idle_fixture();
    let before = snapshot_db(&fixture.paper_db);
    let symbols_file = fixture._dir.path().join("symbols-noop-rewrite.txt");
    fs::write(&symbols_file, "BTC\n").expect("initial symbols file should be created");

    let child = watchlist_daemon_command(&fixture, &symbols_file)
        .arg("--idle-sleep-ms")
        .arg("20")
        .arg("--max-idle-polls")
        .arg("2")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon noop-rewrite smoke should spawn");

    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    thread::sleep(Duration::from_millis(80));
    fs::write(&symbols_file, "BTC\n")
        .expect("symbols file should be rewritten with identical contents");

    let output = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        output.status.success(),
        "paper daemon should stay clean across a no-op symbols-file rewrite; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    assert_eq!(
        report
            .pointer("/manifest_reload_count")
            .and_then(Value::as_u64),
        Some(0),
        "a no-op rewrite should not count as a successful manifest reload",
    );
    assert!(
        report
            .pointer("/loop_report/warnings")
            .and_then(Value::as_array)
            .is_some_and(|warnings| warnings.iter().all(|warning| {
                warning
                    .as_str()
                    .is_some_and(|text| !text.contains("paper daemon reloaded symbols:"))
            })),
        "a no-op rewrite should not emit a reload warning",
    );
    assert_eq!(
        before,
        snapshot_db(&fixture.paper_db),
        "a no-op rewrite must not mutate the paper DB",
    );
}

#[test]
fn paper_daemon_merged_manifest_noop_does_not_increment_reload_count() {
    let fixture = prepare_idle_fixture();
    let before = snapshot_db(&fixture.paper_db);
    let symbols_file = fixture._dir.path().join("symbols-merged-noop.txt");
    fs::write(&symbols_file, "").expect("initial symbols file should be created");

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
        .arg("BTC")
        .arg("--symbols-file")
        .arg(&symbols_file)
        .arg("--watch-symbols-file")
        .arg("--lock-path")
        .arg(&fixture.lock_path)
        .arg("--idle-sleep-ms")
        .arg("20")
        .arg("--max-idle-polls")
        .arg("2")
        .arg("--json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let child = command
        .spawn()
        .expect("paper daemon merged-manifest noop smoke should spawn");
    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    thread::sleep(Duration::from_millis(80));
    fs::write(&symbols_file, "BTC\n").expect("symbols file should rewrite to the explicit symbol");

    let output = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        output.status.success(),
        "paper daemon should stay clean across a merged-manifest no-op rewrite; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    assert_eq!(
        report
            .pointer("/manifest_reload_count")
            .and_then(Value::as_u64),
        Some(0),
        "a merged-manifest no-op rewrite should not count as a successful manifest reload",
    );
    assert_eq!(
        report
            .pointer("/manifest_symbols")
            .and_then(Value::as_array),
        Some(&vec![Value::String("BTC".to_string())]),
        "effective manifest should remain unchanged across a merged-manifest no-op rewrite",
    );
    assert_eq!(
        before,
        snapshot_db(&fixture.paper_db),
        "a merged-manifest no-op rewrite must not mutate the paper DB",
    );
}

#[test]
fn paper_daemon_keeps_open_positions_in_active_symbols_after_manifest_reload() {
    let fixture = prepare_idle_fixture();
    let before = snapshot_db(&fixture.paper_db);
    let symbols_file = fixture._dir.path().join("symbols.txt");
    fs::write(&symbols_file, "ETH\n").expect("initial symbols file should be created");

    let child = watchlist_daemon_command(&fixture, &symbols_file)
        .arg("--idle-sleep-ms")
        .arg("20")
        .arg("--max-idle-polls")
        .arg("10")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("paper daemon open-position union smoke should spawn");

    let child = wait_for_lock_file(child, &fixture.lock_path, Duration::from_secs(5));
    thread::sleep(Duration::from_millis(80));
    fs::write(&symbols_file, "BTC\n").expect("symbols file should reload to BTC");

    let output = wait_with_timeout(child, Duration::from_secs(5));
    assert!(
        output.status.success(),
        "paper daemon should exit cleanly after a manifest reload while caught up; output:\n{}",
        combined_output(&output)
    );

    let report = parse_json_output(&output);
    assert_eq!(
        report
            .pointer("/manifest_reload_count")
            .and_then(Value::as_u64),
        Some(1),
        "daemon should accept the BTC reload",
    );
    assert_eq!(
        report
            .pointer("/manifest_symbols")
            .and_then(Value::as_array),
        Some(&vec![Value::String("BTC".to_string())]),
        "final manifest should reflect the reloaded file contents",
    );
    assert_eq!(
        report
            .pointer("/last_active_symbols")
            .and_then(Value::as_array),
        Some(&vec![
            Value::String("BTC".to_string()),
            Value::String("ETH".to_string()),
        ]),
        "active symbols should union the reloaded manifest with open paper positions",
    );
    assert_eq!(
        before,
        snapshot_db(&fixture.paper_db),
        "idle manifest reload must not mutate the paper DB",
    );
}

fn seed_fixture() -> Fixture {
    let dir = tempdir().expect("fixture tempdir should be created");
    let paper_db = dir.path().join("paper.db");
    let candles_db = dir.path().join("candles.db");
    let lock_path = dir.path().join("paper-daemon.lock");
    let status_path = dir.path().join("paper-daemon.status.json");
    seed_paper_db(&paper_db);
    seed_candles_db(&candles_db);
    Fixture {
        _dir: dir,
        paper_db,
        candles_db,
        lock_path,
        status_path,
    }
}

fn seed_empty_fixture() -> Fixture {
    let dir = tempdir().expect("fixture tempdir should be created");
    let paper_db = dir.path().join("paper.db");
    let candles_db = dir.path().join("candles.db");
    let lock_path = dir.path().join("paper-daemon.lock");
    let status_path = dir.path().join("paper-daemon.status.json");
    seed_empty_paper_db(&paper_db);
    seed_candles_db(&candles_db);
    Fixture {
        _dir: dir,
        paper_db,
        candles_db,
        lock_path,
        status_path,
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
        .arg("--status-path")
        .arg(&fixture.status_path)
        .arg("--json");
    command
}

fn status_command(fixture: &Fixture) -> Command {
    let mut command = runtime_command();
    command
        .arg("paper")
        .arg("status")
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
        .arg("--status-path")
        .arg(&fixture.status_path)
        .arg("--json");
    command
}

fn service_command(fixture: &Fixture) -> Command {
    let mut command = runtime_command();
    command
        .arg("paper")
        .arg("service")
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
        .arg("--status-path")
        .arg(&fixture.status_path)
        .arg("--json");
    command
}

fn service_apply_command(fixture: &Fixture) -> Command {
    let mut command = runtime_command();
    command
        .arg("paper")
        .arg("service")
        .arg("apply")
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
        .arg("--status-path")
        .arg(&fixture.status_path)
        .arg("--start-wait-ms")
        .arg("5000")
        .arg("--stop-wait-ms")
        .arg("5000")
        .arg("--poll-ms")
        .arg("50")
        .arg("--json");
    command
}

fn watchlist_daemon_command(fixture: &Fixture, symbols_file: &Path) -> Command {
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
        .arg("--watch-symbols-file")
        .arg("--lock-path")
        .arg(&fixture.lock_path)
        .arg("--status-path")
        .arg(&fixture.status_path)
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

fn read_json_file(path: &Path) -> Value {
    serde_json::from_slice(
        &fs::read(path)
            .unwrap_or_else(|err| panic!("failed to read JSON file {}: {err}", path.display())),
    )
    .unwrap_or_else(|err| panic!("failed to parse JSON file {}: {err}", path.display()))
}

fn wait_for_status_file(path: &Path, timeout: Duration) -> Value {
    let deadline = Instant::now() + timeout;
    loop {
        if path.exists() {
            let value = read_json_file(path);
            if value.pointer("/running").is_some() {
                return value;
            }
        }
        assert!(
            Instant::now() <= deadline,
            "paper daemon did not materialise the status file {} before timeout",
            path.display()
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn wait_for_status_pid(path: &Path, pid: u32, running: bool, timeout: Duration) -> Value {
    let deadline = Instant::now() + timeout;
    loop {
        if path.exists() {
            let value = read_json_file(path);
            if value.pointer("/running").and_then(Value::as_bool) == Some(running)
                && value.pointer("/pid").and_then(Value::as_u64) == Some(pid as u64)
            {
                return value;
            }
        }
        assert!(
            Instant::now() <= deadline,
            "paper daemon status {} did not reach pid={} running={} before timeout",
            path.display(),
            pid,
            running
        );
        thread::sleep(Duration::from_millis(20));
    }
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

fn send_sigterm_pid(pid: i32) {
    let rc = unsafe { kill(pid, SIGTERM) };
    assert_eq!(
        rc,
        0,
        "SIGTERM should be delivered to the paper daemon pid {}: {}",
        pid,
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

fn seed_paper_db(path: &Path) {
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

fn seed_empty_paper_db(path: &Path) {
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
    .expect("empty paper schema should be created");
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
