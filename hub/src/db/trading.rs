use rusqlite::{params, Connection};
use serde::Serialize;
use crate::error::HubError;

// ── Types ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct OpenPosition {
    pub symbol: String,
    #[serde(rename = "type")]
    pub pos_type: String,
    pub open_trade_id: i64,
    pub open_timestamp: Option<String>,
    pub entry_price: f64,
    pub size: f64,
    pub confidence: Option<String>,
    pub entry_atr: f64,
    pub leverage: f64,
    pub margin_used: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecentTrade {
    pub timestamp: Option<String>,
    pub symbol: String,
    #[serde(rename = "type")]
    pub trade_type: Option<String>,
    pub action: Option<String>,
    pub price: Option<f64>,
    pub size: Option<f64>,
    pub notional: Option<f64>,
    pub pnl: Option<f64>,
    pub fee_usd: Option<f64>,
    pub reason: Option<String>,
    pub confidence: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecentSignal {
    pub timestamp: Option<String>,
    pub symbol: String,
    pub signal: Option<String>,
    pub confidence: Option<String>,
    pub price: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LastSignal {
    pub timestamp: Option<String>,
    pub signal: Option<String>,
    pub confidence: Option<String>,
    pub price: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LastTrade {
    pub timestamp: Option<String>,
    #[serde(rename = "type")]
    pub trade_type: Option<String>,
    pub action: Option<String>,
    pub price: Option<f64>,
    pub size: Option<f64>,
    pub pnl: Option<f64>,
    pub reason: Option<String>,
    pub confidence: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BalanceInfo {
    pub balance_source: String,
    pub realised_usd: Option<f64>,
    pub equity_est_usd: Option<f64>,
    pub unreal_pnl_est_usd: f64,
    pub est_close_fees_usd: f64,
    pub fee_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DailyMetrics {
    pub utc_day: String,
    pub trades: i64,
    pub start_balance: Option<f64>,
    pub end_balance: Option<f64>,
    pub pnl_usd: f64,
    pub fees_usd: f64,
    pub net_pnl_usd: f64,
    pub peak_realised_balance: Option<f64>,
    pub drawdown_pct: f64,
}

// ── Queries ──────────────────────────────────────────────────────────────

/// Compute all open positions from the trades table.
pub fn compute_open_positions(conn: &Connection) -> Result<Vec<OpenPosition>, HubError> {
    // Find the latest OPEN trade per symbol that is not closed.
    let mut stmt = conn.prepare(
        "SELECT t.id, t.timestamp, t.symbol, t.type, t.price, t.size, t.confidence,
                t.entry_atr, t.leverage, t.margin_used
         FROM trades t
         JOIN (
             SELECT symbol, MAX(id) AS open_id
             FROM trades WHERE action = 'OPEN' GROUP BY symbol
         ) lo ON lo.symbol = t.symbol AND lo.open_id = t.id
         LEFT JOIN (
             SELECT symbol, MAX(id) AS close_id
             FROM trades WHERE action = 'CLOSE' GROUP BY symbol
         ) lc ON lc.symbol = t.symbol
         WHERE lc.close_id IS NULL OR t.id > lc.close_id",
    )?;

    let rows: Vec<_> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,       // id
                row.get::<_, Option<String>>(1)?, // timestamp
                row.get::<_, String>(2)?,     // symbol
                row.get::<_, String>(3)?,     // type
                row.get::<_, f64>(4)?,        // price
                row.get::<_, f64>(5)?,        // size
                row.get::<_, Option<String>>(6)?, // confidence
                row.get::<_, Option<f64>>(7)?, // entry_atr
                row.get::<_, Option<f64>>(8)?, // leverage
                row.get::<_, Option<f64>>(9)?, // margin_used
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    let mut positions = Vec::new();

    for (open_id, ts, symbol, pos_type, mut avg_entry, mut net_size, confidence, entry_atr_opt, leverage_opt, margin_used_opt) in rows {
        let pos_type_upper = pos_type.to_uppercase();
        if pos_type_upper != "LONG" && pos_type_upper != "SHORT" {
            continue;
        }
        if avg_entry <= 0.0 || net_size <= 0.0 {
            continue;
        }

        let mut entry_atr = entry_atr_opt.unwrap_or(0.0);

        // Replay ADD/REDUCE to rebuild position
        let mut add_reduce_stmt = conn.prepare(
            "SELECT action, price, size, entry_atr FROM trades
             WHERE symbol = ? AND id > ? AND action IN ('ADD', 'REDUCE')
             ORDER BY id ASC",
        )?;

        let adjustments: Vec<_> = add_reduce_stmt
            .query_map(params![symbol, open_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, f64>(1)?,
                    row.get::<_, f64>(2)?,
                    row.get::<_, Option<f64>>(3)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let mut closed = false;
        for (action, px, sz, fill_atr) in &adjustments {
            if *px <= 0.0 || *sz <= 0.0 {
                continue;
            }
            if action == "ADD" {
                let new_total = net_size + sz;
                if new_total > 0.0 {
                    avg_entry = (avg_entry * net_size + px * sz) / new_total;
                    if let Some(fa) = fill_atr {
                        if *fa > 0.0 {
                            entry_atr = if entry_atr > 0.0 {
                                (entry_atr * net_size + fa * sz) / new_total
                            } else {
                                *fa
                            };
                        }
                    }
                    net_size = new_total;
                }
            } else if action == "REDUCE" {
                net_size -= sz;
                if net_size <= 0.0 {
                    closed = true;
                    break;
                }
            }
        }

        if closed || net_size <= 0.0 {
            continue;
        }

        let leverage = leverage_opt.unwrap_or(1.0).max(0.01);
        let margin_used = margin_used_opt.unwrap_or_else(|| net_size.abs() * avg_entry / leverage);

        positions.push(OpenPosition {
            symbol: symbol.to_uppercase(),
            pos_type: pos_type_upper,
            open_trade_id: open_id,
            open_timestamp: ts,
            entry_price: avg_entry,
            size: net_size,
            confidence,
            entry_atr,
            leverage,
            margin_used,
        });
    }

    Ok(positions)
}

/// Fetch recent trades (most recent first).
pub fn recent_trades(conn: &Connection, limit: u32) -> Result<Vec<RecentTrade>, HubError> {
    let mut stmt = conn.prepare(
        "SELECT timestamp, symbol, type, action, price, size, notional, pnl, fee_usd, reason, confidence
         FROM trades ORDER BY id DESC LIMIT ?",
    )?;
    let rows = stmt
        .query_map(params![limit], |row| {
            Ok(RecentTrade {
                timestamp: row.get(0)?,
                symbol: row.get::<_, String>(1).unwrap_or_default(),
                trade_type: row.get(2)?,
                action: row.get(3)?,
                price: row.get(4)?,
                size: row.get(5)?,
                notional: row.get(6)?,
                pnl: row.get(7)?,
                fee_usd: row.get(8)?,
                reason: row.get(9)?,
                confidence: row.get(10)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Fetch recent signals.
pub fn recent_signals(conn: &Connection, limit: u32) -> Result<Vec<RecentSignal>, HubError> {
    let mut stmt = conn.prepare(
        "SELECT timestamp, symbol, signal, confidence, price
         FROM signals ORDER BY id DESC LIMIT ?",
    )?;
    let rows = stmt
        .query_map(params![limit], |row| {
            Ok(RecentSignal {
                timestamp: row.get(0)?,
                symbol: row.get::<_, String>(1).unwrap_or_default(),
                signal: row.get(2)?,
                confidence: row.get(3)?,
                price: row.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Fetch the last signal per symbol.
pub fn last_signals_by_symbol(
    conn: &Connection,
    symbols: &[String],
) -> Result<std::collections::HashMap<String, LastSignal>, HubError> {
    if symbols.is_empty() {
        return Ok(Default::default());
    }
    let placeholders = symbols.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT t.symbol, t.timestamp, t.signal, t.confidence, t.price
         FROM signals t
         JOIN (SELECT symbol, MAX(id) AS max_id FROM signals WHERE symbol IN ({placeholders}) GROUP BY symbol) m
           ON m.symbol = t.symbol AND m.max_id = t.id"
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::types::ToSql> = symbols.iter().map(|s| s as &dyn rusqlite::types::ToSql).collect();
    let rows = stmt.query_map(params.as_slice(), |row| {
        Ok((
            row.get::<_, String>(0)?,
            LastSignal {
                timestamp: row.get(1)?,
                signal: row.get(2)?,
                confidence: row.get(3)?,
                price: row.get(4)?,
            },
        ))
    })?.collect::<Result<Vec<_>, _>>()?;

    Ok(rows.into_iter().collect())
}

/// Fetch the last trade per symbol.
pub fn last_trades_by_symbol(
    conn: &Connection,
    symbols: &[String],
) -> Result<std::collections::HashMap<String, LastTrade>, HubError> {
    if symbols.is_empty() {
        return Ok(Default::default());
    }
    let placeholders = symbols.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT t.symbol, t.timestamp, t.type, t.action, t.price, t.size, t.pnl, t.reason, t.confidence
         FROM trades t
         JOIN (SELECT symbol, MAX(id) AS max_id FROM trades WHERE symbol IN ({placeholders}) GROUP BY symbol) m
           ON m.symbol = t.symbol AND m.max_id = t.id"
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::types::ToSql> = symbols.iter().map(|s| s as &dyn rusqlite::types::ToSql).collect();
    let rows = stmt.query_map(params.as_slice(), |row| {
        Ok((
            row.get::<_, String>(0)?,
            LastTrade {
                timestamp: row.get(1)?,
                trade_type: row.get(2)?,
                action: row.get(3)?,
                price: row.get(4)?,
                size: row.get(5)?,
                pnl: row.get(6)?,
                reason: row.get(7)?,
                confidence: row.get(8)?,
            },
        ))
    })?.collect::<Result<Vec<_>, _>>()?;

    Ok(rows.into_iter().collect())
}

/// Fetch the latest balance from trades table.
pub fn latest_balance(conn: &Connection) -> Result<Option<(f64, Option<String>)>, HubError> {
    let mut stmt = conn.prepare(
        "SELECT balance, timestamp FROM trades WHERE balance IS NOT NULL ORDER BY id DESC LIMIT 1",
    )?;
    let result = stmt.query_row([], |row| {
        Ok((row.get::<_, f64>(0)?, row.get::<_, Option<String>>(1)?))
    });
    match result {
        Ok(v) => Ok(Some(v)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Check if a table exists in the database.
pub fn has_table(conn: &Connection, table: &str) -> bool {
    conn.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name=?")
        .and_then(|mut s| s.query_row(params![table], |_| Ok(())))
        .is_ok()
}

/// List recent symbols from candles + trades.
pub fn list_recent_symbols(
    conn: &Connection,
    candles_conn: Option<&Connection>,
    interval: &str,
    now_ms: i64,
    limit: usize,
) -> Result<Vec<String>, HubError> {
    let mut symbols: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // From candles
    let cconn = candles_conn.unwrap_or(conn);
    let candle_cutoff = now_ms - 6 * 60 * 60 * 1000;
    if let Ok(mut stmt) = cconn.prepare(
        "SELECT symbol FROM candles WHERE interval = ? AND COALESCE(t_close, t) >= ?
         GROUP BY symbol ORDER BY MAX(COALESCE(t_close, t)) DESC LIMIT ?"
    ) {
        if let Ok(rows) = stmt.query_map(params![interval, candle_cutoff, limit as u32], |row| {
            row.get::<_, String>(0)
        }) {
            for row in rows.flatten() {
                let s = row.to_uppercase();
                if seen.insert(s.clone()) {
                    symbols.push(s);
                }
            }
        }
    }

    // From trades
    if let Ok(mut stmt) = conn.prepare(
        "SELECT DISTINCT symbol FROM trades WHERE timestamp >= datetime('now', '-24 hours') ORDER BY symbol"
    ) {
        if let Ok(rows) = stmt.query_map([], |row| row.get::<_, String>(0)) {
            for row in rows.flatten() {
                let s = row.to_uppercase();
                if seen.insert(s.clone()) {
                    symbols.push(s);
                }
            }
        }
    }

    symbols.truncate(200);
    Ok(symbols)
}
