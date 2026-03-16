use anyhow::Result;
use bt_core::decision_kernel::{Position, PositionSide, StrategyState};
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::BTreeMap;
use std::path::Path;

use crate::live_hyperliquid::{HyperliquidAccountSnapshot, HyperliquidPosition};

#[derive(Debug, Clone, PartialEq)]
pub struct LiveStateSyncReport {
    pub position_count: usize,
    pub removed_symbols: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmittedOrderAction {
    Open,
    Add,
    Close,
    Reduce,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubmittedOrderProjection<'a> {
    pub symbol: &'a str,
    pub action: SubmittedOrderAction,
    pub pos_type: Option<&'a str>,
    pub ts_ms: i64,
    pub entry_adx_threshold: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct PositionStateRow {
    pos_type: String,
    open_time_ms: i64,
    trailing_sl: Option<f64>,
    last_funding_time_ms: i64,
    adds_count: u32,
    tp1_taken: bool,
    last_add_time_ms: i64,
    entry_adx_threshold: f64,
    updated_at: String,
}

pub fn ensure_live_runtime_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS position_state (
            symbol TEXT PRIMARY KEY,
            open_trade_id INTEGER,
            pos_type TEXT,
            open_time_ms INTEGER,
            trailing_sl REAL,
            last_funding_time INTEGER,
            adds_count INTEGER,
            tp1_taken INTEGER,
            last_add_time INTEGER,
            entry_adx_threshold REAL,
            updated_at TEXT
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
    add_position_state_column_if_missing(
        conn,
        "ALTER TABLE position_state ADD COLUMN pos_type TEXT",
    )?;
    add_position_state_column_if_missing(
        conn,
        "ALTER TABLE position_state ADD COLUMN open_time_ms INTEGER",
    )?;
    Ok(())
}

fn add_position_state_column_if_missing(conn: &Connection, sql: &str) -> Result<()> {
    match conn.execute(sql, []) {
        Ok(_) => Ok(()),
        Err(rusqlite::Error::SqliteFailure(_, Some(message)))
            if message.contains("duplicate column name") =>
        {
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

pub fn sync_exchange_positions(
    db_path: &Path,
    exchange_positions: &[HyperliquidPosition],
    observed_at_ms: i64,
) -> Result<LiveStateSyncReport> {
    let mut conn = Connection::open(db_path)?;
    ensure_live_runtime_tables(&conn)?;
    let tx = conn.transaction()?;
    let updated_at = iso_from_ms(observed_at_ms);

    let mut seen = BTreeMap::new();
    for position in exchange_positions {
        let symbol = position.symbol.trim().to_ascii_uppercase();
        if symbol.is_empty() {
            continue;
        }
        let prior = load_position_state_row(&tx, &symbol)?;
        let current_pos_type = position.pos_type.trim().to_ascii_uppercase();
        let current_open_trade = current_open_trade_marker(&tx, &symbol)?;
        let side_changed = prior
            .as_ref()
            .map(|row| !row.pos_type.is_empty() && !row.pos_type.eq_ignore_ascii_case(&current_pos_type))
            .unwrap_or(false);
        tx.execute(
            "INSERT OR REPLACE INTO position_state (symbol, open_trade_id, pos_type, open_time_ms, trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold, updated_at)
             VALUES (?1, NULL, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                symbol,
                current_pos_type,
                if side_changed {
                    observed_at_ms.max(0)
                } else if current_open_trade
                    .as_ref()
                    .is_some_and(|(_, side)| side.eq_ignore_ascii_case(&current_pos_type))
                {
                    current_open_trade
                        .as_ref()
                        .map(|(ts_ms, _)| *ts_ms)
                        .unwrap_or(observed_at_ms.max(0))
                } else {
                    prior
                        .as_ref()
                        .map(|row| row.open_time_ms)
                        .filter(|value| *value > 0)
                        .unwrap_or(observed_at_ms.max(0))
                },
                if side_changed {
                    None
                } else {
                    prior.as_ref().and_then(|row| row.trailing_sl)
                },
                if side_changed {
                    observed_at_ms.max(0)
                } else {
                    prior
                        .as_ref()
                        .map(|row| row.last_funding_time_ms)
                        .unwrap_or(observed_at_ms.max(0))
                },
                if side_changed {
                    0_i64
                } else {
                    prior.as_ref().map(|row| row.adds_count as i64).unwrap_or(0_i64)
                },
                if !side_changed && prior.as_ref().map(|row| row.tp1_taken).unwrap_or(false) {
                    1_i64
                } else {
                    0_i64
                },
                if side_changed {
                    0_i64
                } else {
                    prior.as_ref().map(|row| row.last_add_time_ms).unwrap_or(0_i64)
                },
                if side_changed {
                    0.0_f64
                } else {
                    prior
                        .as_ref()
                        .map(|row| row.entry_adx_threshold)
                        .unwrap_or(0.0_f64)
                },
                updated_at,
            ],
        )?;
        seen.insert(symbol, ());
    }

    let existing_symbols = {
        let mut stmt = tx.prepare("SELECT symbol FROM position_state ORDER BY symbol ASC")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    let removed_symbols = existing_symbols
        .into_iter()
        .filter(|symbol| !seen.contains_key(symbol))
        .collect::<Vec<_>>();
    for symbol in &removed_symbols {
        tx.execute("DELETE FROM position_state WHERE symbol = ?1", [symbol])?;
    }
    tx.commit()?;

    Ok(LiveStateSyncReport {
        position_count: exchange_positions.len(),
        removed_symbols,
    })
}

fn current_open_trade_marker(
    conn: &Connection,
    symbol: &str,
) -> Result<Option<(i64, String)>> {
    let has_trades = conn
        .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='trades' LIMIT 1")?
        .exists([])?;
    if !has_trades {
        return Ok(None);
    }

    let mut stmt = conn.prepare(
        "SELECT id, timestamp, type, action
         FROM trades
         WHERE symbol = ?1 AND action IN ('OPEN', 'CLOSE')
         ORDER BY id ASC",
    )?;
    let mut rows = stmt
        .query_map([symbol], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                row.get::<_, Option<String>>(3)?.unwrap_or_default(),
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    rows.sort_by(|left, right| {
        let left_ts = parse_trade_timestamp_ms(&left.1);
        let right_ts = parse_trade_timestamp_ms(&right.1);
        left_ts.cmp(&right_ts).then(left.0.cmp(&right.0))
    });

    let mut current = None;
    for (_, timestamp, side, action) in rows {
        let action = action.trim().to_ascii_uppercase();
        match action.as_str() {
            "OPEN" => {
                let timestamp_ms = parse_trade_timestamp_ms(&timestamp);
                let side = side.trim().to_ascii_uppercase();
                if timestamp_ms > 0 && (side == "LONG" || side == "SHORT") {
                    current = Some((timestamp_ms, side));
                }
            }
            "CLOSE" => current = None,
            _ => {}
        }
    }

    Ok(current)
}

pub fn build_strategy_state(
    db_path: &Path,
    account_snapshot: &HyperliquidAccountSnapshot,
    exchange_positions: &[HyperliquidPosition],
    observed_at_ms: i64,
) -> Result<StrategyState> {
    let conn = Connection::open(db_path)?;
    ensure_live_runtime_tables(&conn)?;
    let mut state = StrategyState::new(account_snapshot.withdrawable_usd.max(0.0), observed_at_ms);

    let mut last_entry_ms = BTreeMap::new();
    let mut last_exit_ms = BTreeMap::new();
    let mut stmt = conn.prepare(
        "SELECT symbol, last_entry_attempt_s, last_exit_attempt_s FROM runtime_cooldowns ORDER BY symbol ASC",
    )?;
    let cooldown_rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<f64>>(1)?,
            row.get::<_, Option<f64>>(2)?,
        ))
    })?;
    for row in cooldown_rows {
        let (symbol, entry_s, exit_s) = row?;
        if let Some(entry_s) = entry_s {
            last_entry_ms.insert(symbol.clone(), secs_to_ms(entry_s));
        }
        if let Some(exit_s) = exit_s {
            last_exit_ms.insert(symbol, secs_to_ms(exit_s));
        }
    }
    state.last_entry_ms = last_entry_ms;
    state.last_exit_ms = last_exit_ms;

    let mut close_stmt = conn.prepare(
        "SELECT symbol, close_ts_ms, side, reason FROM runtime_last_closes ORDER BY symbol ASC",
    )?;
    let close_rows = close_stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;
    for row in close_rows {
        let (symbol, ts_ms, side, reason) = row?;
        state.last_close_info.insert(symbol, (ts_ms, side, reason));
    }

    for position in exchange_positions {
        let symbol = position.symbol.trim().to_ascii_uppercase();
        if symbol.is_empty() || position.size <= 0.0 {
            continue;
        }
        let extras = load_position_state_row(&conn, &symbol)?;
        let side = if position.pos_type.eq_ignore_ascii_case("LONG") {
            PositionSide::Long
        } else {
            PositionSide::Short
        };
        state.positions.insert(
            symbol.clone(),
            Position {
                symbol,
                side,
                quantity: position.size,
                avg_entry_price: position.entry_price.max(0.0),
                opened_at_ms: extras
                    .as_ref()
                    .map(|row| row.open_time_ms)
                    .filter(|value| *value > 0)
                    .unwrap_or(observed_at_ms),
                updated_at_ms: observed_at_ms,
                notional_usd: position.size * position.entry_price.max(0.0),
                margin_usd: position.margin_used.max(0.0),
                confidence: None,
                entry_atr: None,
                entry_adx_threshold: extras
                    .as_ref()
                    .map(|row| row.entry_adx_threshold)
                    .filter(|value| *value > 0.0),
                adds_count: extras.as_ref().map(|row| row.adds_count).unwrap_or(0),
                tp1_taken: extras.as_ref().map(|row| row.tp1_taken).unwrap_or(false),
                trailing_sl: extras.as_ref().and_then(|row| row.trailing_sl),
                mae_usd: 0.0,
                mfe_usd: 0.0,
                last_funding_ms: Some(
                    extras
                        .as_ref()
                        .map(|row| row.last_funding_time_ms)
                        .unwrap_or(observed_at_ms),
                ),
            },
        );
    }

    Ok(state)
}

pub fn note_submitted_order(
    db_path: &Path,
    projection: SubmittedOrderProjection<'_>,
) -> Result<()> {
    let symbol = projection.symbol.trim().to_ascii_uppercase();
    if symbol.is_empty() {
        return Ok(());
    }

    let mut conn = Connection::open(db_path)?;
    ensure_live_runtime_tables(&conn)?;
    let tx = conn.transaction()?;
    let updated_at = iso_from_ms(projection.ts_ms);
    let prior = load_position_state_row(&tx, &symbol)?;
    let prior_entry_s = tx
        .query_row(
            "SELECT last_entry_attempt_s FROM runtime_cooldowns WHERE symbol = ?1 LIMIT 1",
            [symbol.as_str()],
            |row| row.get::<_, Option<f64>>(0),
        )
        .optional()?
        .flatten();
    let prior_exit_s = tx
        .query_row(
            "SELECT last_exit_attempt_s FROM runtime_cooldowns WHERE symbol = ?1 LIMIT 1",
            [symbol.as_str()],
            |row| row.get::<_, Option<f64>>(0),
        )
        .optional()?
        .flatten();

    let (last_entry_attempt_s, last_exit_attempt_s) = match projection.action {
        SubmittedOrderAction::Open | SubmittedOrderAction::Add => {
            (Some(ms_to_secs(projection.ts_ms)), prior_exit_s)
        }
        SubmittedOrderAction::Close | SubmittedOrderAction::Reduce => {
            (prior_entry_s, Some(ms_to_secs(projection.ts_ms)))
        }
    };
    tx.execute(
        "INSERT OR REPLACE INTO runtime_cooldowns (symbol, last_entry_attempt_s, last_exit_attempt_s, updated_at)
         VALUES (?1, ?2, ?3, ?4)",
        params![
            symbol,
            last_entry_attempt_s,
            last_exit_attempt_s,
            updated_at,
        ],
    )?;

    if matches!(
        projection.action,
        SubmittedOrderAction::Open | SubmittedOrderAction::Add
    ) {
        let projected_pos_type = projection
            .pos_type
            .map(|value| value.trim().to_ascii_uppercase())
            .filter(|value| !value.is_empty());
        let (pos_type, open_time_ms, last_funding_time_ms, adds_count, tp1_taken, last_add_time_ms, entry_adx_threshold) =
            match projection.action {
                SubmittedOrderAction::Open => (
                    projected_pos_type
                        .or_else(|| {
                            prior
                                .as_ref()
                                .map(|row| row.pos_type.clone())
                                .filter(|value| !value.is_empty())
                        })
                        .unwrap_or_default(),
                    projection.ts_ms.max(0),
                    projection.ts_ms.max(0),
                    0_i64,
                    0_i64,
                    0_i64,
                    projection
                        .entry_adx_threshold
                        .filter(|value| *value > 0.0)
                        .or_else(|| {
                            prior
                                .as_ref()
                                .map(|row| row.entry_adx_threshold)
                                .filter(|value| *value > 0.0)
                        })
                        .unwrap_or(0.0),
                ),
                SubmittedOrderAction::Add => (
                    projected_pos_type
                        .or_else(|| {
                            prior
                                .as_ref()
                                .map(|row| row.pos_type.clone())
                                .filter(|value| !value.is_empty())
                        })
                        .unwrap_or_default(),
                    prior
                        .as_ref()
                        .map(|row| row.open_time_ms)
                        .filter(|value| *value > 0)
                        .unwrap_or(projection.ts_ms.max(0)),
                    prior
                        .as_ref()
                        .map(|row| row.last_funding_time_ms)
                        .filter(|value| *value > 0)
                        .unwrap_or(projection.ts_ms.max(0)),
                    prior
                        .as_ref()
                        .map(|row| i64::from(row.adds_count))
                        .unwrap_or(0_i64)
                        + 1_i64,
                    if prior.as_ref().is_some_and(|row| row.tp1_taken) {
                        1_i64
                    } else {
                        0_i64
                    },
                    projection.ts_ms.max(0),
                    prior
                        .as_ref()
                        .map(|row| row.entry_adx_threshold)
                        .filter(|value| *value > 0.0)
                        .or_else(|| projection.entry_adx_threshold.filter(|value| *value > 0.0))
                        .unwrap_or(0.0),
                ),
                SubmittedOrderAction::Close | SubmittedOrderAction::Reduce => unreachable!(),
            };

        tx.execute(
            "INSERT OR REPLACE INTO position_state (symbol, open_trade_id, pos_type, open_time_ms, trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold, updated_at)
             VALUES (?1, NULL, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                symbol,
                pos_type,
                open_time_ms,
                prior.as_ref().and_then(|row| row.trailing_sl),
                last_funding_time_ms,
                adds_count,
                tp1_taken,
                last_add_time_ms,
                entry_adx_threshold,
                updated_at,
            ],
        )?;
    }

    tx.commit()?;
    Ok(())
}

pub fn record_full_close(
    db_path: &Path,
    symbol: &str,
    close_ts_ms: i64,
    side: &str,
    reason: &str,
) -> Result<()> {
    let symbol = symbol.trim().to_ascii_uppercase();
    if symbol.is_empty() {
        return Ok(());
    }
    let conn = Connection::open(db_path)?;
    ensure_live_runtime_tables(&conn)?;
    conn.execute(
        "INSERT OR REPLACE INTO runtime_last_closes (symbol, close_ts_ms, side, reason, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            symbol,
            close_ts_ms.max(0),
            side.trim().to_ascii_uppercase(),
            reason.trim(),
            iso_from_ms(close_ts_ms),
        ],
    )?;
    Ok(())
}

fn load_position_state_row(conn: &Connection, symbol: &str) -> Result<Option<PositionStateRow>> {
    conn.query_row(
        "SELECT pos_type, open_time_ms, trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold, updated_at
         FROM position_state WHERE symbol = ?1 LIMIT 1",
        [symbol],
        |row| {
            Ok(PositionStateRow {
                pos_type: row.get::<_, Option<String>>(0)?.unwrap_or_default(),
                open_time_ms: row.get::<_, Option<i64>>(1)?.unwrap_or(0),
                trailing_sl: row.get::<_, Option<f64>>(2)?,
                last_funding_time_ms: row.get::<_, Option<i64>>(3)?.unwrap_or(0),
                adds_count: row.get::<_, Option<i64>>(4)?.unwrap_or(0).max(0) as u32,
                tp1_taken: row.get::<_, Option<i64>>(5)?.unwrap_or(0) != 0,
                last_add_time_ms: row.get::<_, Option<i64>>(6)?.unwrap_or(0),
                entry_adx_threshold: row.get::<_, Option<f64>>(7)?.unwrap_or(0.0),
                updated_at: row.get::<_, Option<String>>(8)?.unwrap_or_default(),
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

fn iso_from_ms(ts_ms: i64) -> String {
    chrono::DateTime::<Utc>::from_timestamp_millis(ts_ms)
        .unwrap_or_else(Utc::now)
        .to_rfc3339()
}

fn parse_trade_timestamp_ms(raw: &str) -> i64 {
    let normalised = if raw.contains('T') {
        raw.to_string()
    } else {
        raw.replace(' ', "T")
    };
    chrono::DateTime::parse_from_rfc3339(&normalised)
        .map(|timestamp| timestamp.with_timezone(&Utc).timestamp_millis())
        .unwrap_or(0)
}

fn secs_to_ms(value: f64) -> i64 {
    (value * 1000.0).round() as i64
}

fn ms_to_secs(value: i64) -> f64 {
    (value as f64) / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample_snapshot() -> HyperliquidAccountSnapshot {
        HyperliquidAccountSnapshot {
            account_value_usd: 1_250.0,
            withdrawable_usd: 900.0,
            total_margin_used_usd: 350.0,
        }
    }

    #[test]
    fn submitted_open_projects_runtime_state_and_cooldowns() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("live.db");
        note_submitted_order(
            &db_path,
            SubmittedOrderProjection {
                symbol: "eth",
                action: SubmittedOrderAction::Open,
                pos_type: Some("LONG"),
                ts_ms: 1_772_700_000_000,
                entry_adx_threshold: Some(22.5),
            },
        )
        .unwrap();

        let conn = Connection::open(&db_path).unwrap();
        let (last_entry_s, last_exit_s): (Option<f64>, Option<f64>) = conn
            .query_row(
                "SELECT last_entry_attempt_s, last_exit_attempt_s FROM runtime_cooldowns WHERE symbol = 'ETH'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(last_entry_s, Some(1_772_700_000.0));
        assert_eq!(last_exit_s, None);

        let row: (i64, i64, i64, i64, f64) = conn
            .query_row(
                "SELECT open_time_ms, last_funding_time, adds_count, last_add_time, entry_adx_threshold FROM position_state WHERE symbol = 'ETH'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
            )
            .unwrap();
        assert_eq!(row.0, 1_772_700_000_000);
        assert_eq!(row.1, 1_772_700_000_000);
        assert_eq!(row.2, 0);
        assert_eq!(row.3, 0);
        assert_eq!(row.4, 22.5);
    }

    #[test]
    fn submitted_add_increments_adds_count_and_updates_entry_cooldown() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("live.db");
        let conn = Connection::open(&db_path).unwrap();
        ensure_live_runtime_tables(&conn).unwrap();
        conn.execute(
            "INSERT INTO position_state (symbol, open_trade_id, pos_type, open_time_ms, trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold, updated_at)
             VALUES ('ETH', NULL, 'LONG', 1772699900000, NULL, 1772700000000, 1, 0, 1772700100000, 21.0, '2026-03-08T00:00:00+00:00')",
            [],
        )
        .unwrap();

        note_submitted_order(
            &db_path,
            SubmittedOrderProjection {
                symbol: "ETH",
                action: SubmittedOrderAction::Add,
                pos_type: Some("LONG"),
                ts_ms: 1_772_700_200_000,
                entry_adx_threshold: Some(30.0),
            },
        )
        .unwrap();

        let conn = Connection::open(&db_path).unwrap();
        let row: (i64, i64, i64, i64, f64) = conn
            .query_row(
                "SELECT open_time_ms, last_funding_time, adds_count, last_add_time, entry_adx_threshold FROM position_state WHERE symbol = 'ETH'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
            )
            .unwrap();
        assert_eq!(row.0, 1_772_699_900_000);
        assert_eq!(row.1, 1_772_700_000_000);
        assert_eq!(row.2, 2);
        assert_eq!(row.3, 1_772_700_200_000);
        assert_eq!(row.4, 21.0);
    }

    #[test]
    fn build_strategy_state_prefers_position_open_time_for_live_identity() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("live.db");
        let conn = Connection::open(&db_path).unwrap();
        ensure_live_runtime_tables(&conn).unwrap();
        conn.execute(
            "INSERT INTO position_state (symbol, open_trade_id, pos_type, open_time_ms, trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold, updated_at)
             VALUES ('ETH', NULL, 'LONG', 1772699900000, NULL, 1772700000000, 0, 0, 0, 0.0, '2026-03-08T00:00:00+00:00')",
            [],
        )
        .unwrap();
        drop(conn);

        let state = build_strategy_state(
            &db_path,
            &sample_snapshot(),
            &[HyperliquidPosition {
                symbol: "ETH".to_string(),
                pos_type: "LONG".to_string(),
                size: 1.5,
                entry_price: 2000.0,
                leverage: 3.0,
                margin_used: 1000.0,
            }],
            1_772_700_300_000,
        )
        .unwrap();

        assert_eq!(state.positions["ETH"].opened_at_ms, 1_772_699_900_000);
    }

    #[test]
    fn sync_exchange_positions_resets_open_time_when_side_changes() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("live.db");
        let conn = Connection::open(&db_path).unwrap();
        ensure_live_runtime_tables(&conn).unwrap();
        conn.execute(
            "INSERT INTO position_state (symbol, open_trade_id, pos_type, open_time_ms, trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold, updated_at)
             VALUES ('ETH', NULL, 'LONG', 1772699900000, 1900.0, 1772700000000, 2, 1, 1772700100000, 21.0, '2026-03-08T00:00:00+00:00')",
            [],
        )
        .unwrap();
        drop(conn);

        sync_exchange_positions(
            &db_path,
            &[HyperliquidPosition {
                symbol: "ETH".to_string(),
                pos_type: "SHORT".to_string(),
                size: 1.0,
                entry_price: 1900.0,
                leverage: 3.0,
                margin_used: 633.0,
            }],
            1_772_700_300_000,
        )
        .unwrap();

        let conn = Connection::open(&db_path).unwrap();
        let row: (String, i64, Option<f64>, i64, i64, i64) = conn
            .query_row(
                "SELECT pos_type, open_time_ms, trailing_sl, adds_count, tp1_taken, last_add_time FROM position_state WHERE symbol = 'ETH'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?)),
            )
            .unwrap();
        assert_eq!(row.0, "SHORT");
        assert_eq!(row.1, 1_772_700_300_000);
        assert!(row.2.is_none());
        assert_eq!(row.3, 0);
        assert_eq!(row.4, 0);
        assert_eq!(row.5, 0);
    }

    #[test]
    fn sync_exchange_positions_uses_latest_open_trade_marker_for_same_side_reopen() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("live.db");
        let conn = Connection::open(&db_path).unwrap();
        ensure_live_runtime_tables(&conn).unwrap();
        conn.execute_batch(
            "
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                type TEXT,
                action TEXT
            );
            ",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO position_state (symbol, open_trade_id, pos_type, open_time_ms, trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold, updated_at)
             VALUES ('ETH', NULL, 'LONG', 1772699900000, NULL, 1772700000000, 0, 0, 0, 0.0, '2026-03-08T00:00:00+00:00')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO trades (timestamp, symbol, type, action) VALUES ('2026-03-08T00:03:00+00:00', 'ETH', 'LONG', 'CLOSE')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO trades (timestamp, symbol, type, action) VALUES ('2026-03-08T00:04:00+00:00', 'ETH', 'LONG', 'OPEN')",
            [],
        )
        .unwrap();
        drop(conn);

        sync_exchange_positions(
            &db_path,
            &[HyperliquidPosition {
                symbol: "ETH".to_string(),
                pos_type: "LONG".to_string(),
                size: 1.0,
                entry_price: 1900.0,
                leverage: 3.0,
                margin_used: 633.0,
            }],
            1_772_700_300_000,
        )
        .unwrap();

        let conn = Connection::open(&db_path).unwrap();
        let open_time_ms: i64 = conn
            .query_row(
                "SELECT open_time_ms FROM position_state WHERE symbol = 'ETH'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(open_time_ms, 1_772_928_240_000);
    }

    #[test]
    fn record_full_close_surfaces_last_close_info_in_strategy_state() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("live.db");
        record_full_close(&db_path, "btc", 1_772_700_300_000, "SELL", "stop_loss").unwrap();

        let state =
            build_strategy_state(&db_path, &sample_snapshot(), &[], 1_772_700_300_000).unwrap();
        assert_eq!(
            state.last_close_info.get("BTC"),
            Some(&(
                1_772_700_300_000,
                "SELL".to_string(),
                "stop_loss".to_string()
            ))
        );
    }
}
