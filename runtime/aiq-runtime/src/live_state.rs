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
    pub ts_ms: i64,
    pub entry_adx_threshold: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct PositionStateRow {
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
    Ok(())
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
        tx.execute(
            "INSERT OR REPLACE INTO position_state (symbol, open_trade_id, trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold, updated_at)
             VALUES (?1, NULL, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                symbol,
                prior.as_ref().and_then(|row| row.trailing_sl),
                prior
                    .as_ref()
                    .map(|row| row.last_funding_time_ms)
                    .unwrap_or(observed_at_ms.max(0)),
                prior.as_ref().map(|row| row.adds_count as i64).unwrap_or(0_i64),
                if prior.as_ref().map(|row| row.tp1_taken).unwrap_or(false) {
                    1_i64
                } else {
                    0_i64
                },
                prior.as_ref().map(|row| row.last_add_time_ms).unwrap_or(0_i64),
                prior
                    .as_ref()
                    .map(|row| row.entry_adx_threshold)
                    .unwrap_or(0.0_f64),
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
                    .map(|row| row.last_funding_time_ms)
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
        let (last_funding_time_ms, adds_count, tp1_taken, last_add_time_ms, entry_adx_threshold) =
            match projection.action {
                SubmittedOrderAction::Open => (
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
            "INSERT OR REPLACE INTO position_state (symbol, open_trade_id, trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold, updated_at)
             VALUES (?1, NULL, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                symbol,
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
        "SELECT trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold, updated_at
         FROM position_state WHERE symbol = ?1 LIMIT 1",
        [symbol],
        |row| {
            Ok(PositionStateRow {
                trailing_sl: row.get::<_, Option<f64>>(0)?,
                last_funding_time_ms: row.get::<_, Option<i64>>(1)?.unwrap_or(0),
                adds_count: row.get::<_, Option<i64>>(2)?.unwrap_or(0).max(0) as u32,
                tp1_taken: row.get::<_, Option<i64>>(3)?.unwrap_or(0) != 0,
                last_add_time_ms: row.get::<_, Option<i64>>(4)?.unwrap_or(0),
                entry_adx_threshold: row.get::<_, Option<f64>>(5)?.unwrap_or(0.0),
                updated_at: row.get::<_, Option<String>>(6)?.unwrap_or_default(),
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

        let row: (i64, i64, i64, f64) = conn
            .query_row(
                "SELECT last_funding_time, adds_count, last_add_time, entry_adx_threshold FROM position_state WHERE symbol = 'ETH'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(row.0, 1_772_700_000_000);
        assert_eq!(row.1, 0);
        assert_eq!(row.2, 0);
        assert_eq!(row.3, 22.5);
    }

    #[test]
    fn submitted_add_increments_adds_count_and_updates_entry_cooldown() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("live.db");
        let conn = Connection::open(&db_path).unwrap();
        ensure_live_runtime_tables(&conn).unwrap();
        conn.execute(
            "INSERT INTO position_state (symbol, open_trade_id, trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold, updated_at)
             VALUES ('ETH', NULL, NULL, 1772700000000, 1, 0, 1772700100000, 21.0, '2026-03-08T00:00:00+00:00')",
            [],
        )
        .unwrap();

        note_submitted_order(
            &db_path,
            SubmittedOrderProjection {
                symbol: "ETH",
                action: SubmittedOrderAction::Add,
                ts_ms: 1_772_700_200_000,
                entry_adx_threshold: Some(30.0),
            },
        )
        .unwrap();

        let conn = Connection::open(&db_path).unwrap();
        let row: (i64, i64, i64, f64) = conn
            .query_row(
                "SELECT last_funding_time, adds_count, last_add_time, entry_adx_threshold FROM position_state WHERE symbol = 'ETH'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(row.0, 1_772_700_000_000);
        assert_eq!(row.1, 2);
        assert_eq!(row.2, 1_772_700_200_000);
        assert_eq!(row.3, 21.0);
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
