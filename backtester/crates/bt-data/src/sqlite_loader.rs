use bt_core::candle::{CandleData, FundingRateData, OhlcvBar};
use rusqlite::{Connection, OpenFlags};
use std::time::Instant;

/// Query the min/max timestamp for a given interval in the candle database.
/// Returns `(min_t, max_t)` in milliseconds, or `None` if the table is empty.
pub fn query_time_range(
    db_path: &str,
    interval: &str,
) -> Result<Option<(i64, i64)>, Box<dyn std::error::Error>> {
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut stmt = conn.prepare(
        "SELECT MIN(t), MAX(t) FROM candles WHERE interval = ?",
    )?;
    let result = stmt.query_row([interval], |row| {
        let min_t: Option<i64> = row.get(0)?;
        let max_t: Option<i64> = row.get(1)?;
        Ok(min_t.zip(max_t))
    })?;
    Ok(result)
}

/// Load all candle rows for a given interval from the SQLite database,
/// returning them grouped by symbol with bars sorted by open-time ascending.
///
/// Optional `from_ts` / `to_ts` (milliseconds) restrict the loaded range.
pub fn load_candles(
    db_path: &str,
    interval: &str,
) -> Result<CandleData, Box<dyn std::error::Error>> {
    load_candles_filtered(db_path, interval, None, None)
}

/// Like [`load_candles`] but with optional timestamp filters.
pub fn load_candles_filtered(
    db_path: &str,
    interval: &str,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
) -> Result<CandleData, Box<dyn std::error::Error>> {
    let start = Instant::now();

    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;

    // Tune for bulk read performance
    conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = OFF;")?;

    // Build dynamic WHERE clause
    let mut where_parts = vec!["interval = ?1".to_string()];
    if from_ts.is_some() {
        where_parts.push("t >= ?2".to_string());
    }
    if to_ts.is_some() {
        where_parts.push(format!("t <= ?{}", if from_ts.is_some() { 3 } else { 2 }));
    }
    let query = format!(
        "SELECT symbol, t, t_close, o, h, l, c, v, COALESCE(n, 0) \
         FROM candles WHERE {} ORDER BY symbol, t ASC",
        where_parts.join(" AND "),
    );

    let mut stmt = conn.prepare(&query)?;

    let mut data: CandleData = CandleData::default();
    let mut total_bars: u64 = 0;
    let mut current_symbol = String::new();
    let mut current_vec: Vec<OhlcvBar> = Vec::new();

    // Bind parameters dynamically
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(interval.to_string())];
    if let Some(ft) = from_ts {
        params.push(Box::new(ft));
    }
    if let Some(tt) = to_ts {
        params.push(Box::new(tt));
    }
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let rows = stmt.query_map(param_refs.as_slice(), |row| {
        Ok((
            row.get::<_, String>(0)?,
            OhlcvBar {
                t: row.get(1)?,
                t_close: row.get(2)?,
                o: row.get(3)?,
                h: row.get(4)?,
                l: row.get(5)?,
                c: row.get(6)?,
                v: row.get(7)?,
                n: row.get(8)?,
            },
        ))
    })?;

    for row_result in rows {
        let (symbol, bar) = row_result?;
        total_bars += 1;

        if symbol != current_symbol {
            // Flush the previous symbol's bars
            if !current_symbol.is_empty() {
                data.insert(std::mem::take(&mut current_symbol), std::mem::take(&mut current_vec));
            }
            current_symbol = symbol;
            current_vec = Vec::with_capacity(512);
        }
        current_vec.push(bar);
    }

    // Flush the last symbol
    if !current_symbol.is_empty() {
        data.insert(current_symbol, current_vec);
    }

    let elapsed = start.elapsed();
    let filter_info = match (from_ts, to_ts) {
        (Some(f), Some(t)) => format!(", range={f}..{t}"),
        (Some(f), None) => format!(", from={f}"),
        (None, Some(t)) => format!(", to={t}"),
        (None, None) => String::new(),
    };
    println!(
        "[bt-data] Loaded {} symbols, {} bars in {:.2}s from {:?} (interval={}{})",
        data.len(),
        total_bars,
        elapsed.as_secs_f64(),
        db_path,
        interval,
        filter_info,
    );

    Ok(data)
}

/// Load the distinct list of symbols available for a given interval.
pub fn load_symbols(
    db_path: &str,
    interval: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;

    let mut stmt =
        conn.prepare("SELECT DISTINCT symbol FROM candles WHERE interval = ? ORDER BY symbol")?;

    let symbols: Vec<String> = stmt
        .query_map([interval], |row| row.get(0))?
        .collect::<Result<Vec<_>, _>>()?;

    println!(
        "[bt-data] Found {} symbols for interval={} in {:?}",
        symbols.len(),
        interval,
        db_path,
    );

    Ok(symbols)
}

/// Load symbols considered "active" during the given time range using a universe
/// history SQLite database (see `tools/sync_universe_history.py`).
///
/// A symbol is considered active when its observed listing interval overlaps with
/// the backtest window:
///   first_seen_ms <= to_ts AND last_seen_ms >= from_ts
pub fn load_universe_active_symbols(
    db_path: &str,
    from_ts: i64,
    to_ts: i64,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;

    let mut stmt = conn.prepare(
        "SELECT symbol FROM universe_listings \
         WHERE first_seen_ms <= ?2 AND last_seen_ms >= ?1 \
         ORDER BY symbol",
    )?;

    let symbols: Vec<String> = stmt
        .query_map([from_ts, to_ts], |row| row.get(0))?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(symbols)
}

/// Load funding rates from a SQLite database, returning them grouped by
/// symbol as sorted Vec<(timestamp_ms, rate)>.
///
/// Expected table schema:
/// ```sql
/// CREATE TABLE funding_rates (
///     symbol TEXT NOT NULL,
///     time INTEGER NOT NULL,
///     funding_rate REAL NOT NULL,
///     PRIMARY KEY (symbol, time)
/// );
/// ```
pub fn load_funding_rates(
    db_path: &str,
) -> Result<FundingRateData, Box<dyn std::error::Error>> {
    load_funding_rates_filtered(db_path, None, None)
}

pub fn load_funding_rates_filtered(
    db_path: &str,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
) -> Result<FundingRateData, Box<dyn std::error::Error>> {
    let start = Instant::now();

    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA synchronous = OFF;")?;

    let mut where_parts: Vec<String> = Vec::new();
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    let mut idx = 1;
    if let Some(ft) = from_ts {
        where_parts.push(format!("time >= ?{idx}"));
        params.push(Box::new(ft));
        idx += 1;
    }
    if let Some(tt) = to_ts {
        where_parts.push(format!("time <= ?{idx}"));
        params.push(Box::new(tt));
    }
    let where_clause = if where_parts.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", where_parts.join(" AND "))
    };
    let query = format!(
        "SELECT symbol, time, funding_rate FROM funding_rates{} ORDER BY symbol, time ASC",
        where_clause,
    );

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
    let mut stmt = conn.prepare(&query)?;

    let mut data: FundingRateData = FundingRateData::default();
    let mut total: u64 = 0;
    let mut current_symbol = String::new();
    let mut current_vec: Vec<(i64, f64)> = Vec::new();

    let rows = stmt.query_map(param_refs.as_slice(), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, f64>(2)?,
        ))
    })?;

    for row_result in rows {
        let (symbol, time_ms, rate) = row_result?;
        total += 1;

        if symbol != current_symbol {
            if !current_symbol.is_empty() {
                data.insert(
                    std::mem::take(&mut current_symbol),
                    std::mem::take(&mut current_vec),
                );
            }
            current_symbol = symbol;
            current_vec = Vec::with_capacity(256);
        }
        current_vec.push((time_ms, rate));
    }

    if !current_symbol.is_empty() {
        data.insert(current_symbol, current_vec);
    }

    let elapsed = start.elapsed();
    println!(
        "[bt-data] Loaded {} funding rate entries across {} symbols in {:.2}s from {:?}",
        total,
        data.len(),
        elapsed.as_secs_f64(),
        db_path,
    );

    Ok(data)
}
