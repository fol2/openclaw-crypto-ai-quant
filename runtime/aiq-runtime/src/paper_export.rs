use aiq_runtime_core::snapshot::{
    SnapshotFile, SnapshotLastCloseInfo, SnapshotPosition, SnapshotRuntimeState, SNAPSHOT_V2,
};
use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
struct OpenPositionSeed {
    open_id: i64,
    open_ts: Option<String>,
    symbol: String,
    pos_type: String,
    avg_entry: f64,
    net_size: f64,
    entry_atr: f64,
    confidence: String,
    leverage: f64,
    margin_used: f64,
    last_add_time_ms: i64,
}

pub fn export_paper_snapshot(db_path: &Path, exported_at_ms: i64) -> anyhow::Result<SnapshotFile> {
    let conn = Connection::open(db_path)?;
    let balance = conn
        .query_row(
            "SELECT balance FROM trades ORDER BY id DESC LIMIT 1",
            [],
            |row| row.get::<_, f64>(0),
        )
        .optional()?
        .unwrap_or(0.0);

    let mut positions = reconstruct_open_positions(&conn)?;
    positions.sort_by(|left, right| left.symbol.cmp(&right.symbol));
    let runtime = load_runtime_markers(&conn)?;

    Ok(SnapshotFile {
        version: SNAPSHOT_V2,
        source: "paper".to_string(),
        exported_at_ms,
        balance,
        positions,
        runtime: Some(runtime),
    })
}

fn reconstruct_open_positions(conn: &Connection) -> anyhow::Result<Vec<SnapshotPosition>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT t.id AS open_id, t.timestamp AS open_ts, t.symbol, t.type AS pos_type,
               t.price AS open_px, t.size AS open_sz, t.confidence,
               t.entry_atr, t.leverage, t.margin_used
        FROM trades t
        INNER JOIN (
            SELECT symbol, MAX(id) AS open_id
            FROM trades WHERE action = 'OPEN' GROUP BY symbol
        ) lo ON t.id = lo.open_id
        LEFT JOIN (
            SELECT symbol, MAX(id) AS close_id
            FROM trades WHERE action = 'CLOSE' GROUP BY symbol
        ) lc ON t.symbol = lc.symbol
        WHERE lc.close_id IS NULL OR t.id > lc.close_id
        "#,
    )?;

    let open_rows = stmt.query_map([], |row| {
        Ok(OpenPositionSeed {
            open_id: row.get(0)?,
            open_ts: row.get(1)?,
            symbol: row.get(2)?,
            pos_type: row.get(3)?,
            avg_entry: row.get(4)?,
            net_size: row.get(5)?,
            confidence: row
                .get::<_, Option<String>>(6)?
                .unwrap_or_else(|| "medium".to_string()),
            entry_atr: row.get::<_, Option<f64>>(7)?.unwrap_or(0.0),
            leverage: row.get::<_, Option<f64>>(8)?.unwrap_or(1.0),
            margin_used: row.get::<_, Option<f64>>(9)?.unwrap_or(0.0),
            last_add_time_ms: 0,
        })
    })?;

    let has_position_state = table_exists(conn, "position_state")?;
    let has_last_funding_time =
        has_position_state && table_column_exists(conn, "position_state", "last_funding_time")?;
    let mut positions = Vec::new();

    for seed in open_rows {
        let mut seed = seed?;
        replay_add_reduce_fills(conn, &mut seed)?;
        if seed.net_size <= 0.0 {
            continue;
        }
        if seed.leverage > 0.0 {
            seed.margin_used = (seed.net_size * seed.avg_entry) / seed.leverage;
        }

        let mut trailing_sl = None;
        let mut last_funding_time_ms = parse_timestamp_ms(seed.open_ts.as_deref());
        let mut adds_count = 0_u32;
        let mut tp1_taken = false;
        let mut last_add_time_ms = seed.last_add_time_ms;
        let mut entry_adx_threshold = 0.0;

        if has_position_state {
            if has_last_funding_time {
                let state_row = conn
                    .query_row(
                        r#"
                        SELECT open_trade_id, trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold
                        FROM position_state
                        WHERE symbol = ?
                        "#,
                        [&seed.symbol],
                        |row| {
                            Ok((
                                row.get::<_, Option<i64>>(0)?,
                                row.get::<_, Option<f64>>(1)?,
                                row.get::<_, Option<i64>>(2)?,
                                row.get::<_, Option<i64>>(3)?,
                                row.get::<_, Option<i64>>(4)?,
                                row.get::<_, Option<i64>>(5)?,
                                row.get::<_, Option<f64>>(6)?,
                            ))
                        },
                    )
                    .optional()?;

                if let Some((
                    Some(open_trade_id),
                    db_trailing_sl,
                    db_last_funding,
                    db_adds,
                    db_tp1,
                    db_last_add,
                    db_adx,
                )) = state_row
                {
                    if open_trade_id == seed.open_id {
                        trailing_sl = db_trailing_sl;
                        last_funding_time_ms = db_last_funding.unwrap_or(last_funding_time_ms);
                        adds_count = db_adds.unwrap_or(0).max(0) as u32;
                        tp1_taken = db_tp1.unwrap_or(0) != 0;
                        last_add_time_ms = db_last_add.unwrap_or(0);
                        entry_adx_threshold = db_adx.unwrap_or(0.0);
                    }
                }
            } else {
                let state_row = conn
                    .query_row(
                        r#"
                        SELECT open_trade_id, trailing_sl, adds_count, tp1_taken, last_add_time, entry_adx_threshold
                        FROM position_state
                        WHERE symbol = ?
                        "#,
                        [&seed.symbol],
                        |row| {
                            Ok((
                                row.get::<_, Option<i64>>(0)?,
                                row.get::<_, Option<f64>>(1)?,
                                row.get::<_, Option<i64>>(2)?,
                                row.get::<_, Option<i64>>(3)?,
                                row.get::<_, Option<i64>>(4)?,
                                row.get::<_, Option<f64>>(5)?,
                            ))
                        },
                    )
                    .optional()?;

                if let Some((
                    Some(open_trade_id),
                    db_trailing_sl,
                    db_adds,
                    db_tp1,
                    db_last_add,
                    db_adx,
                )) = state_row
                {
                    if open_trade_id == seed.open_id {
                        trailing_sl = db_trailing_sl;
                        adds_count = db_adds.unwrap_or(0).max(0) as u32;
                        tp1_taken = db_tp1.unwrap_or(0) != 0;
                        last_add_time_ms = db_last_add.unwrap_or(0);
                        entry_adx_threshold = db_adx.unwrap_or(0.0);
                    }
                }
            }
        }

        positions.push(SnapshotPosition {
            symbol: seed.symbol.clone(),
            side: if seed.pos_type.eq_ignore_ascii_case("LONG") {
                "long".to_string()
            } else {
                "short".to_string()
            },
            size: seed.net_size,
            entry_price: seed.avg_entry,
            entry_atr: seed.entry_atr,
            trailing_sl,
            confidence: seed.confidence.to_ascii_lowercase(),
            leverage: seed.leverage,
            margin_used: seed.margin_used,
            adds_count,
            tp1_taken,
            open_time_ms: parse_timestamp_ms(seed.open_ts.as_deref()),
            last_funding_time_ms,
            last_add_time_ms,
            entry_adx_threshold,
        });
    }

    Ok(positions)
}

fn replay_add_reduce_fills(conn: &Connection, seed: &mut OpenPositionSeed) -> anyhow::Result<()> {
    let fallback_atr = seed.entry_atr;
    let mut stmt = conn.prepare(
        r#"
        SELECT action, price, size, entry_atr, timestamp
        FROM trades
        WHERE symbol = ? AND id > ? AND action IN ('ADD', 'REDUCE')
        ORDER BY id ASC
        "#,
    )?;

    let rows = stmt.query_map((&seed.symbol, seed.open_id), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, f64>(1)?,
            row.get::<_, f64>(2)?,
            row.get::<_, Option<f64>>(3)?.unwrap_or(fallback_atr),
            row.get::<_, Option<String>>(4)?,
        ))
    })?;

    for row in rows {
        let (action, price, size, fill_atr, timestamp) = row?;
        if action == "ADD" {
            let new_total = seed.net_size + size;
            if new_total > 0.0 {
                seed.avg_entry = ((seed.avg_entry * seed.net_size) + (price * size)) / new_total;
                seed.entry_atr = ((seed.entry_atr * seed.net_size) + (fill_atr * size)) / new_total;
            }
            seed.net_size = new_total;
            let add_ts_ms = parse_timestamp_ms(timestamp.as_deref());
            if add_ts_ms > 0 {
                seed.last_add_time_ms = add_ts_ms;
            }
        } else if action == "REDUCE" {
            seed.net_size -= size;
            if seed.net_size <= 0.0 {
                break;
            }
        }
    }

    Ok(())
}

fn load_runtime_markers(conn: &Connection) -> anyhow::Result<SnapshotRuntimeState> {
    if !table_exists(conn, "runtime_cooldowns")? {
        return Ok(SnapshotRuntimeState::default());
    }

    let mut entry_attempt_ms_by_symbol = BTreeMap::new();
    let mut exit_attempt_ms_by_symbol = BTreeMap::new();
    let mut stmt = conn.prepare(
        "SELECT symbol, last_entry_attempt_s, last_exit_attempt_s FROM runtime_cooldowns",
    )?;
    let now_s = Utc::now().timestamp() as f64;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<f64>>(1)?,
            row.get::<_, Option<f64>>(2)?,
        ))
    })?;

    for row in rows {
        let (symbol, entry_s, exit_s) = row?;
        if let Some(entry_ms) = normalise_runtime_marker_ms(entry_s, now_s) {
            entry_attempt_ms_by_symbol.insert(symbol.clone(), entry_ms);
        }
        if let Some(exit_ms) = normalise_runtime_marker_ms(exit_s, now_s) {
            exit_attempt_ms_by_symbol.insert(symbol.clone(), exit_ms);
        }
    }

    let last_close_info_by_symbol = load_last_close_info(conn)?;

    Ok(SnapshotRuntimeState {
        entry_attempt_ms_by_symbol,
        exit_attempt_ms_by_symbol,
        last_close_info_by_symbol,
    })
}

fn load_last_close_info(
    conn: &Connection,
) -> anyhow::Result<BTreeMap<String, SnapshotLastCloseInfo>> {
    let mut last_close_info_by_symbol = BTreeMap::new();
    if table_exists(conn, "runtime_last_closes")? {
        let mut stmt = conn.prepare(
            "SELECT symbol, close_ts_ms, side, reason FROM runtime_last_closes ORDER BY close_ts_ms DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })?;

        for row in rows {
            let (symbol, timestamp_ms, side, reason) = row?;
            let symbol = symbol.trim().to_ascii_uppercase();
            let side = side.trim().to_ascii_lowercase();
            if symbol.is_empty() || last_close_info_by_symbol.contains_key(&symbol) {
                continue;
            }
            if timestamp_ms <= 0 || (side != "long" && side != "short") {
                continue;
            }
            last_close_info_by_symbol.insert(
                symbol,
                SnapshotLastCloseInfo {
                    timestamp_ms,
                    side,
                    reason: reason.unwrap_or_default(),
                },
            );
        }
    }

    let mut stmt = conn.prepare(
        "SELECT symbol, timestamp, type, reason FROM trades WHERE action = 'CLOSE' ORDER BY id DESC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, Option<String>>(3)?,
        ))
    })?;

    for row in rows {
        let (symbol, timestamp, side, reason) = row?;
        let symbol = symbol.trim().to_ascii_uppercase();
        if symbol.is_empty() || last_close_info_by_symbol.contains_key(&symbol) {
            continue;
        }
        let timestamp_ms = parse_timestamp_ms(timestamp.as_deref());
        let side = side.unwrap_or_default().trim().to_ascii_lowercase();
        if timestamp_ms <= 0 || (side != "long" && side != "short") {
            continue;
        }
        last_close_info_by_symbol.insert(
            symbol,
            SnapshotLastCloseInfo {
                timestamp_ms,
                side,
                reason: reason.unwrap_or_default(),
            },
        );
    }

    Ok(last_close_info_by_symbol)
}

fn normalise_runtime_marker_ms(raw: Option<f64>, now_s: f64) -> Option<i64> {
    let mut value = raw?;
    if value <= 0.0 {
        return None;
    }
    if value > 10_000_000_000.0 {
        value /= 1000.0;
    }
    if value > (now_s + 86_400.0) {
        return None;
    }
    Some((value * 1000.0).round() as i64)
}

fn parse_timestamp_ms(raw: Option<&str>) -> i64 {
    raw.and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        .map(|dt| dt.with_timezone(&Utc).timestamp_millis())
        .unwrap_or(0)
}

fn table_exists(conn: &Connection, table_name: &str) -> anyhow::Result<bool> {
    let row = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            [table_name],
            |_| Ok(()),
        )
        .optional()?;
    Ok(row.is_some())
}

fn table_column_exists(
    conn: &Connection,
    table_name: &str,
    column_name: &str,
) -> anyhow::Result<bool> {
    let pragma = format!("PRAGMA table_info({table_name})");
    let mut stmt = conn.prepare(&pragma)?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for row in rows {
        if row?.eq_ignore_ascii_case(column_name) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn export_paper_snapshot_reconstructs_positions_and_runtime_markers() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("paper.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                action TEXT,
                type TEXT,
                price REAL,
                size REAL,
                reason TEXT,
                confidence TEXT,
                entry_atr REAL,
                leverage REAL,
                margin_used REAL,
                balance REAL
            );
            CREATE TABLE position_state (
                symbol TEXT PRIMARY KEY,
                open_trade_id INTEGER,
                trailing_sl REAL,
                last_funding_time INTEGER,
                adds_count INTEGER,
                tp1_taken INTEGER,
                last_add_time INTEGER,
                entry_adx_threshold REAL
            );
            CREATE TABLE runtime_cooldowns (
                symbol TEXT PRIMARY KEY,
                last_entry_attempt_s REAL,
                last_exit_attempt_s REAL,
                updated_at TEXT
            );
            "#,
        )
        .unwrap();
        conn.execute(
            "INSERT INTO trades VALUES (1, '2026-03-05T10:00:00+00:00', 'BTC', 'OPEN', 'LONG', 100.0, 2.0, 'Signal Trigger', 'high', 5.0, 4.0, 50.0, 1000.0)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO trades VALUES (2, '2026-03-05T10:10:00+00:00', 'BTC', 'ADD', 'LONG', 110.0, 1.0, 'Pyramid Add', 'high', 6.0, 4.0, 25.0, 1010.0)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO trades VALUES (3, '2026-03-05T09:55:00+00:00', 'ETH', 'CLOSE', 'SHORT', 200.0, 1.0, 'Signal Trigger', 'medium', 4.0, 3.0, 0.0, 995.0)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO position_state VALUES ('BTC', 1, 95.0, 1772676555000, 1, 0, 1772676600000, 23.5)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO runtime_cooldowns VALUES ('BTC', 1772676500.0, 1772676550.0, '2026-03-05T10:15:00+00:00')",
            [],
        )
        .unwrap();
        conn.close().unwrap();

        let snapshot = export_paper_snapshot(&PathBuf::from(&db_path), 1_772_676_900_000).unwrap();

        assert_eq!(snapshot.version, SNAPSHOT_V2);
        assert_eq!(snapshot.positions.len(), 1);
        assert_eq!(snapshot.positions[0].symbol, "BTC");
        assert!((snapshot.positions[0].size - 3.0).abs() < 1e-9);
        assert!((snapshot.positions[0].entry_price - 103.33333333333333).abs() < 1e-9);
        assert!((snapshot.positions[0].margin_used - 77.5).abs() < 1e-9);
        assert_eq!(
            snapshot.positions[0].last_funding_time_ms,
            1_772_676_555_000
        );
        assert_eq!(
            snapshot
                .runtime
                .as_ref()
                .unwrap()
                .entry_attempt_ms_by_symbol
                .get("BTC")
                .copied(),
            Some(1_772_676_500_000)
        );
        assert_eq!(
            snapshot
                .runtime
                .as_ref()
                .unwrap()
                .last_close_info_by_symbol
                .get("ETH")
                .unwrap()
                .timestamp_ms,
            1_772_704_500_000
        );
    }
}
