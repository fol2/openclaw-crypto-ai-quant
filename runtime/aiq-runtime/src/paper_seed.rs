use aiq_runtime_core::snapshot::{SnapshotFile, SNAPSHOT_V2};
use chrono::{DateTime, Utc};
use rusqlite::Connection;
use serde::Serialize;
use serde_json::json;
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeedOptions {
    pub strict_replace: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SeedReport {
    pub ok: bool,
    pub snapshot_source: String,
    pub snapshot_version: u32,
    pub target_db: String,
    pub strict_replace: bool,
    pub seeded_trades: usize,
    pub seeded_positions: usize,
    pub seeded_runtime_cooldowns: usize,
    pub seeded_runtime_last_closes: usize,
    pub seed_timestamp_ms: i64,
}

pub fn seed_paper_db(
    snapshot: &SnapshotFile,
    target_db: &Path,
    options: SeedOptions,
) -> anyhow::Result<SeedReport> {
    snapshot.validate()?;
    if snapshot.version != SNAPSHOT_V2 {
        anyhow::bail!("snapshot seed-paper requires init-state v2");
    }
    if !target_db.exists() {
        anyhow::bail!("target paper db not found: {}", target_db.display());
    }

    let seed_ts_ms = snapshot.exported_at_ms;
    let seed_iso = iso_from_ms(seed_ts_ms);
    let snapshot_symbols: BTreeSet<String> = snapshot
        .positions
        .iter()
        .map(|position| position.symbol.trim().to_ascii_uppercase())
        .filter(|symbol| !symbol.is_empty())
        .collect();
    let mut seed_symbols = snapshot_symbols.clone();
    if let Some(runtime) = snapshot.runtime.as_ref() {
        seed_symbols.extend(
            runtime
                .entry_attempt_ms_by_symbol
                .keys()
                .chain(runtime.exit_attempt_ms_by_symbol.keys())
                .chain(runtime.last_close_info_by_symbol.keys())
                .map(|symbol| symbol.trim().to_ascii_uppercase())
                .filter(|symbol| !symbol.is_empty()),
        );
    }

    let conn = Connection::open(target_db)?;
    ensure_seed_tables(&conn)?;
    conn.execute("BEGIN IMMEDIATE", [])?;
    let result = (|| {
        if options.strict_replace {
            reset_seed_targets(&conn)?;
        } else {
            reject_untracked_open_positions(&conn, &snapshot_symbols)?;
            clear_previous_seed_rows(&conn, &seed_symbols)?;
        }

        let seeded_trades = seed_trades_and_positions(&conn, snapshot, &seed_iso)?;
        let (seeded_runtime_cooldowns, seeded_runtime_last_closes) =
            seed_runtime_state(&conn, snapshot, &seed_iso)?;

        Ok(SeedReport {
            ok: true,
            snapshot_source: snapshot.source.clone(),
            snapshot_version: snapshot.version,
            target_db: target_db.display().to_string(),
            strict_replace: options.strict_replace,
            seeded_trades,
            seeded_positions: snapshot.positions.len(),
            seeded_runtime_cooldowns,
            seeded_runtime_last_closes,
            seed_timestamp_ms: seed_ts_ms,
        })
    })();

    match result {
        Ok(report) => {
            conn.execute("COMMIT", [])?;
            Ok(report)
        }
        Err(err) => {
            let _ = conn.execute("ROLLBACK", []);
            Err(err)
        }
    }
}

fn ensure_seed_tables(conn: &Connection) -> anyhow::Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS trades (
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
        CREATE TABLE IF NOT EXISTS position_state (
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
        CREATE TABLE IF NOT EXISTS position_state_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_ts_ms INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            open_trade_id INTEGER,
            trailing_sl REAL,
            last_funding_time INTEGER,
            adds_count INTEGER,
            tp1_taken INTEGER,
            last_add_time INTEGER,
            entry_adx_threshold REAL,
            event_type TEXT NOT NULL,
            run_fingerprint TEXT
        );
        CREATE TABLE IF NOT EXISTS runtime_cooldowns (
            symbol TEXT PRIMARY KEY,
            last_entry_attempt_s REAL,
            last_exit_attempt_s REAL,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS runtime_last_closes (
            symbol TEXT PRIMARY KEY,
            close_ts_ms INTEGER NOT NULL,
            side TEXT NOT NULL,
            reason TEXT,
            updated_at TEXT NOT NULL
        );
        "#,
    )?;
    Ok(())
}

fn reset_seed_targets(conn: &Connection) -> anyhow::Result<()> {
    conn.execute("DELETE FROM trades", [])?;
    conn.execute("DELETE FROM position_state", [])?;
    conn.execute("DELETE FROM position_state_history", [])?;
    conn.execute("DELETE FROM runtime_cooldowns", [])?;
    conn.execute("DELETE FROM runtime_last_closes", [])?;
    Ok(())
}

fn reject_untracked_open_positions(
    conn: &Connection,
    snapshot_symbols: &BTreeSet<String>,
) -> anyhow::Result<()> {
    let mut stmt = conn.prepare(
        r#"
        SELECT t.symbol
        FROM trades t
        INNER JOIN (
            SELECT symbol, MAX(id) AS open_id
            FROM trades WHERE action = 'OPEN' GROUP BY symbol
        ) lo ON t.id = lo.open_id
        LEFT JOIN (
            SELECT symbol, MAX(id) AS close_id
            FROM trades WHERE action = 'CLOSE' GROUP BY symbol
        ) lc ON t.symbol = lc.symbol
        WHERE lc.close_id IS NULL OR lo.open_id > lc.close_id
        "#,
    )?;

    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    let mut stale_symbols = Vec::new();
    for row in rows {
        let symbol = row?.trim().to_ascii_uppercase();
        if !symbol.is_empty() && !snapshot_symbols.contains(&symbol) {
            stale_symbols.push(symbol);
        }
    }

    if !stale_symbols.is_empty() {
        stale_symbols.sort();
        anyhow::bail!(
            "non-strict seed would leave existing open paper positions outside snapshot: {}. use --strict-replace",
            stale_symbols.join(", ")
        );
    }

    Ok(())
}

fn clear_previous_seed_rows(
    conn: &Connection,
    seed_symbols: &BTreeSet<String>,
) -> anyhow::Result<()> {
    conn.execute(
        "DELETE FROM trades WHERE reason IN ('state_sync_seed', 'state_sync_balance_seed')",
        [],
    )?;

    for symbol in seed_symbols {
        conn.execute("DELETE FROM position_state WHERE symbol = ?", [symbol])?;
        conn.execute("DELETE FROM runtime_cooldowns WHERE symbol = ?", [symbol])?;
        conn.execute("DELETE FROM runtime_last_closes WHERE symbol = ?", [symbol])?;
        conn.execute(
            "DELETE FROM position_state_history WHERE symbol = ? AND event_type = 'seed'",
            [symbol],
        )?;
    }
    Ok(())
}

fn seed_trades_and_positions(
    conn: &Connection,
    snapshot: &SnapshotFile,
    seed_iso: &str,
) -> anyhow::Result<usize> {
    let mut seeded_trades = 0usize;
    conn.execute(
        "INSERT INTO trades (timestamp, symbol, action, type, price, size, notional, reason, confidence, balance, pnl, fee_usd, fee_rate, entry_atr, leverage, margin_used, meta_json)
         VALUES (?1, 'SYSTEM', 'SYSTEM', 'SYSTEM', 0.0, 0.0, 0.0, 'state_sync_balance_seed', 'medium', ?2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ?3)",
        (
            seed_iso,
            snapshot.balance,
            json!({"source": "snapshot_seed"}).to_string(),
        ),
    )?;
    seeded_trades += 1;

    for (idx, position) in snapshot.positions.iter().enumerate() {
        let position_ts_ms = if position.open_time_ms > 0 {
            position.open_time_ms
        } else {
            snapshot.exported_at_ms
        };
        let position_iso = iso_from_ms(position_ts_ms);
        let pos_type = if position.side.eq_ignore_ascii_case("long") {
            "LONG"
        } else {
            "SHORT"
        };
        let notional = position.size * position.entry_price;
        conn.execute(
            "INSERT INTO trades (timestamp, symbol, action, type, price, size, notional, reason, confidence, balance, pnl, fee_usd, fee_rate, entry_atr, leverage, margin_used, meta_json)
             VALUES (?1, ?2, 'OPEN', ?3, ?4, ?5, ?6, 'state_sync_seed', ?7, ?8, 0.0, 0.0, 0.0, ?9, ?10, ?11, ?12)",
            (
                position_iso,
                position.symbol.as_str(),
                pos_type,
                position.entry_price,
                position.size,
                notional,
                position.confidence.as_str(),
                snapshot.balance,
                position.entry_atr,
                position.leverage,
                position.margin_used,
                json!({"source": "snapshot_seed", "seed_idx": idx}).to_string(),
            ),
        )?;
        let open_trade_id = conn.last_insert_rowid();
        let last_funding_time_ms = if position.last_funding_time_ms > 0 {
            position.last_funding_time_ms
        } else {
            position_ts_ms
        };
        conn.execute(
            "INSERT OR REPLACE INTO position_state
             (symbol, open_trade_id, trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            (
                position.symbol.as_str(),
                open_trade_id,
                position.trailing_sl,
                last_funding_time_ms,
                i64::from(position.adds_count),
                if position.tp1_taken { 1_i64 } else { 0_i64 },
                position.last_add_time_ms,
                position.entry_adx_threshold,
                seed_iso,
            ),
        )?;
        conn.execute(
            "INSERT INTO position_state_history
             (event_ts_ms, updated_at, symbol, open_trade_id, trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold, event_type, run_fingerprint)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 'seed', 'snapshot_seed')",
            (
                position_ts_ms,
                seed_iso,
                position.symbol.as_str(),
                open_trade_id,
                position.trailing_sl,
                last_funding_time_ms,
                i64::from(position.adds_count),
                if position.tp1_taken { 1_i64 } else { 0_i64 },
                position.last_add_time_ms,
                position.entry_adx_threshold,
            ),
        )?;
        seeded_trades += 1;
    }

    Ok(seeded_trades)
}

fn seed_runtime_state(
    conn: &Connection,
    snapshot: &SnapshotFile,
    seed_iso: &str,
) -> anyhow::Result<(usize, usize)> {
    let Some(runtime) = snapshot.runtime.as_ref() else {
        return Ok((0, 0));
    };

    let mut symbols = BTreeSet::new();
    symbols.extend(
        runtime
            .entry_attempt_ms_by_symbol
            .keys()
            .map(|symbol| symbol.trim().to_ascii_uppercase())
            .filter(|symbol| !symbol.is_empty()),
    );
    symbols.extend(
        runtime
            .exit_attempt_ms_by_symbol
            .keys()
            .map(|symbol| symbol.trim().to_ascii_uppercase())
            .filter(|symbol| !symbol.is_empty()),
    );

    let mut inserted = 0usize;
    for symbol in symbols {
        let entry_ms = runtime.entry_attempt_ms_by_symbol.get(&symbol).copied();
        let exit_ms = runtime.exit_attempt_ms_by_symbol.get(&symbol).copied();
        conn.execute(
            "INSERT OR REPLACE INTO runtime_cooldowns (symbol, last_entry_attempt_s, last_exit_attempt_s, updated_at)
             VALUES (?1, ?2, ?3, ?4)",
            (
                symbol.as_str(),
                entry_ms.map(ms_to_secs),
                exit_ms.map(ms_to_secs),
                seed_iso,
            ),
        )?;
        inserted += 1;
    }
    let mut inserted_last_closes = 0usize;
    for (symbol, close) in &runtime.last_close_info_by_symbol {
        let symbol = symbol.trim().to_ascii_uppercase();
        let side = close.side.trim().to_ascii_lowercase();
        if symbol.is_empty() || close.timestamp_ms <= 0 || (side != "long" && side != "short") {
            continue;
        }
        conn.execute(
            "INSERT OR REPLACE INTO runtime_last_closes (symbol, close_ts_ms, side, reason, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            (
                symbol.as_str(),
                close.timestamp_ms,
                side.as_str(),
                close.reason.as_str(),
                seed_iso,
            ),
        )?;
        inserted_last_closes += 1;
    }
    Ok((inserted, inserted_last_closes))
}

fn iso_from_ms(ms: i64) -> String {
    DateTime::from_timestamp_millis(ms)
        .map(|dt| dt.with_timezone(&Utc).to_rfc3339())
        .unwrap_or_else(|| Utc::now().to_rfc3339())
}

fn ms_to_secs(value: i64) -> f64 {
    (value as f64) / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiq_runtime_core::snapshot::{SnapshotPosition, SnapshotRuntimeState};
    use tempfile::tempdir;

    fn sample_snapshot() -> SnapshotFile {
        SnapshotFile {
            version: SNAPSHOT_V2,
            source: "paper".to_string(),
            exported_at_ms: 1_772_676_900_000,
            balance: 1000.0,
            positions: vec![SnapshotPosition {
                symbol: "BTC".to_string(),
                side: "long".to_string(),
                size: 2.0,
                entry_price: 100.0,
                entry_atr: 5.0,
                trailing_sl: Some(95.0),
                confidence: "high".to_string(),
                leverage: 4.0,
                margin_used: 50.0,
                adds_count: 1,
                tp1_taken: false,
                open_time_ms: 1_772_676_500_000,
                last_funding_time_ms: 1_772_676_580_000,
                last_add_time_ms: 1_772_676_600_000,
                entry_adx_threshold: 23.5,
            }],
            runtime: Some(SnapshotRuntimeState {
                entry_attempt_ms_by_symbol: [("BTC".to_string(), 1_772_676_500_000)].into(),
                exit_attempt_ms_by_symbol: [("BTC".to_string(), 1_772_676_550_000)].into(),
                last_close_info_by_symbol: [(
                    "ETH".to_string(),
                    aiq_runtime_core::snapshot::SnapshotLastCloseInfo {
                        timestamp_ms: 1_772_676_400_000,
                        side: "short".to_string(),
                        reason: "Signal Trigger".to_string(),
                    },
                )]
                .into(),
            }),
        }
    }

    #[test]
    fn seed_paper_db_rejects_v1_snapshot() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("paper.db");
        let conn = Connection::open(&db_path).unwrap();
        seed_schema(&conn);
        conn.close().unwrap();

        let mut snapshot = sample_snapshot();
        snapshot.version = 1;
        snapshot.runtime = None;
        let err = seed_paper_db(
            &snapshot,
            &db_path,
            SeedOptions {
                strict_replace: true,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("requires init-state v2"));
    }

    fn seed_schema(conn: &Connection) {
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
            CREATE TABLE position_state_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_ts_ms INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open_trade_id INTEGER,
                trailing_sl REAL,
                last_funding_time INTEGER,
                adds_count INTEGER,
                tp1_taken INTEGER,
                last_add_time INTEGER,
                entry_adx_threshold REAL,
                event_type TEXT NOT NULL,
                run_fingerprint TEXT
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
        .unwrap();
    }

    #[test]
    fn seed_paper_db_strict_replace_writes_projection_tables() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("paper.db");
        let conn = Connection::open(&db_path).unwrap();
        seed_schema(&conn);
        conn.close().unwrap();

        let report = seed_paper_db(
            &sample_snapshot(),
            &db_path,
            SeedOptions {
                strict_replace: true,
            },
        )
        .unwrap();

        let conn = Connection::open(&db_path).unwrap();
        let trade_count: i64 = conn
            .query_row("SELECT COUNT(1) FROM trades", [], |row| row.get(0))
            .unwrap();
        let state_count: i64 = conn
            .query_row("SELECT COUNT(1) FROM position_state", [], |row| row.get(0))
            .unwrap();
        let cooldown_count: i64 = conn
            .query_row("SELECT COUNT(1) FROM runtime_cooldowns", [], |row| {
                row.get(0)
            })
            .unwrap();
        let close_count: i64 = conn
            .query_row("SELECT COUNT(1) FROM runtime_last_closes", [], |row| {
                row.get(0)
            })
            .unwrap();
        let open_ts: String = conn
            .query_row(
                "SELECT timestamp FROM trades WHERE action='OPEN' AND symbol='BTC' LIMIT 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let last_funding_time: i64 = conn
            .query_row(
                "SELECT last_funding_time FROM position_state WHERE symbol='BTC' LIMIT 1",
                [],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(report.seeded_trades, 2);
        assert_eq!(report.seeded_positions, 1);
        assert_eq!(report.seeded_runtime_cooldowns, 1);
        assert_eq!(report.seeded_runtime_last_closes, 1);
        assert_eq!(trade_count, 2);
        assert_eq!(state_count, 1);
        assert_eq!(cooldown_count, 1);
        assert_eq!(close_count, 1);
        assert_eq!(open_ts, iso_from_ms(1_772_676_500_000));
        assert_eq!(last_funding_time, 1_772_676_580_000);
        let seeded_snapshot =
            crate::paper_export::export_paper_snapshot(&db_path, sample_snapshot().exported_at_ms)
                .unwrap();
        assert_eq!(
            seeded_snapshot
                .runtime
                .as_ref()
                .unwrap()
                .last_close_info_by_symbol
                .get("ETH")
                .unwrap()
                .timestamp_ms,
            1_772_676_400_000
        );
    }

    #[test]
    fn seed_paper_db_non_strict_rejects_open_positions_outside_snapshot() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("paper.db");
        let conn = Connection::open(&db_path).unwrap();
        seed_schema(&conn);
        conn.execute(
            "INSERT INTO trades (timestamp, symbol, action, type, price, size, notional, reason, confidence, balance, pnl, fee_usd, fee_rate, entry_atr, leverage, margin_used, meta_json)
             VALUES ('2026-03-05T10:00:00+00:00', 'ETH', 'OPEN', 'LONG', 10.0, 1.0, 10.0, 'manual', 'medium', 1000.0, 0.0, 0.0, 0.0, 1.0, 1.0, 10.0, '{}')",
            [],
        )
        .unwrap();
        conn.close().unwrap();

        let err = seed_paper_db(
            &sample_snapshot(),
            &db_path,
            SeedOptions {
                strict_replace: false,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains(
            "non-strict seed would leave existing open paper positions outside snapshot"
        ));
    }

    #[test]
    fn seed_paper_db_requires_existing_target_db() {
        let dir = tempdir().unwrap();
        let missing_db = dir.path().join("missing-paper.db");
        let err = seed_paper_db(
            &sample_snapshot(),
            &missing_db,
            SeedOptions {
                strict_replace: true,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("target paper db not found"));
    }
}
