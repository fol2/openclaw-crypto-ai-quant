use rusqlite::{params, Connection};
use serde::Serialize;

use crate::error::HubError;

#[derive(Debug, Clone, Serialize)]
pub struct Candle {
    pub t: i64,
    pub t_close: i64,
    pub o: f64,
    pub h: f64,
    pub l: f64,
    pub c: f64,
    pub v: f64,
    pub n: i64,
}

/// Fetch OHLCV candles for a symbol/interval from a candle DB connection.
pub fn fetch_candles(
    conn: &Connection,
    symbol: &str,
    interval: &str,
    limit: u32,
) -> Result<Vec<Candle>, HubError> {
    let mut stmt = conn.prepare(
        "SELECT t, t_close, o, h, l, c, v, n
         FROM candles
         WHERE symbol = ? AND interval = ?
         ORDER BY COALESCE(t_close, t) DESC
         LIMIT ?",
    )?;

    let rows: Vec<Candle> = stmt
        .query_map(params![symbol, interval, limit], |row| {
            Ok(Candle {
                t: row.get(0)?,
                t_close: row.get::<_, i64>(1).unwrap_or(0),
                o: row.get(2)?,
                h: row.get(3)?,
                l: row.get(4)?,
                c: row.get(5)?,
                v: row.get(6)?,
                n: row.get::<_, i64>(7).unwrap_or(0),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    // Reverse to chronological order.
    let mut candles = rows;
    candles.reverse();
    Ok(candles)
}

/// List available candle intervals by scanning DB files in the candles directory.
pub fn list_available_intervals(candles_dir: &std::path::Path) -> Vec<String> {
    let mut intervals: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(candles_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(iv) = name
                .strip_prefix("candles_")
                .and_then(|s| s.strip_suffix(".db"))
            {
                if !iv.is_empty() {
                    intervals.push(iv.to_string());
                }
            }
        }
    }
    intervals.sort_by(|a, b| interval_sort_key(a).cmp(&interval_sort_key(b)));
    intervals
}

fn interval_sort_key(iv: &str) -> (u64, String) {
    let s = iv.trim().to_lowercase();
    let (num_str, unit) = s.split_at(s.len().saturating_sub(1));
    let n: u64 = num_str.parse().unwrap_or(u64::MAX);
    let mult: u64 = match unit {
        "m" => 1,
        "h" => 60,
        "d" => 60 * 24,
        _ => 1_000_000,
    };
    (n * mult, s)
}
