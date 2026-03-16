use crate::error::HubError;
use rusqlite::Connection;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct TunnelPoint {
    pub ts_ms: i64,
    pub upper_full: f64,
    pub has_upper_full: bool,
    pub upper_partial: Option<f64>,
    pub lower_full: f64,
    pub has_lower_full: bool,
    pub entry_price: f64,
    pub pos_type: String,
}

/// Fetch exit tunnel data for a symbol within a time range.
///
/// Returns an empty Vec if the `exit_tunnel` table doesn't exist (older DBs).
pub fn fetch_tunnel(
    conn: &Connection,
    symbol: &str,
    from_ts: Option<i64>,
    to_ts: Option<i64>,
    limit: u32,
) -> Result<Vec<TunnelPoint>, HubError> {
    // Gracefully handle missing table (older DBs that predate exit_tunnel).
    let table_exists: bool = conn
        .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='exit_tunnel' LIMIT 1")?
        .exists([])?;
    if !table_exists {
        return Ok(Vec::new());
    }

    let has_upper_full_column = exit_tunnel_has_column(conn, "has_upper_full")?;
    let has_lower_full_column = exit_tunnel_has_column(conn, "has_lower_full")?;

    let mut sql = format!(
        "SELECT ts_ms, upper_full, {}, upper_partial, lower_full, {}, entry_price, pos_type \
         FROM exit_tunnel WHERE symbol = ?",
        if has_upper_full_column {
            "COALESCE(has_upper_full, 1)"
        } else {
            "1"
        },
        if has_lower_full_column {
            "COALESCE(has_lower_full, 1)"
        } else {
            "1"
        },
    );
    let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(symbol.to_string())];

    if let Some(from) = from_ts {
        sql.push_str(" AND ts_ms >= ?");
        param_values.push(Box::new(from));
    }
    if let Some(to) = to_ts {
        sql.push_str(" AND ts_ms <= ?");
        param_values.push(Box::new(to));
    }
    sql.push_str(" ORDER BY ts_ms ASC LIMIT ?");
    param_values.push(Box::new(limit));

    let params_ref: Vec<&dyn rusqlite::types::ToSql> =
        param_values.iter().map(|b| b.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt
        .query_map(params_ref.as_slice(), |row| {
            Ok(TunnelPoint {
                ts_ms: row.get(0)?,
                upper_full: row.get(1)?,
                has_upper_full: row.get::<_, i64>(2)? != 0,
                upper_partial: row.get(3)?,
                lower_full: row.get(4)?,
                has_lower_full: row.get::<_, i64>(5)? != 0,
                entry_price: row.get(6)?,
                pos_type: row.get(7)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

fn exit_tunnel_has_column(conn: &Connection, column: &str) -> Result<bool, HubError> {
    let mut stmt = conn.prepare("PRAGMA table_info(exit_tunnel)")?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(columns.iter().any(|name| name == column))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_tunnel_defaults_flags_for_legacy_schema() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE exit_tunnel (
                ts_ms INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                upper_full REAL NOT NULL,
                upper_partial REAL,
                lower_full REAL NOT NULL,
                entry_price REAL NOT NULL,
                pos_type TEXT NOT NULL
            );
            INSERT INTO exit_tunnel (ts_ms, symbol, upper_full, upper_partial, lower_full, entry_price, pos_type)
            VALUES (1000, 'ETH', 105.0, NULL, 99.0, 100.0, 'LONG');
            "#,
        )
        .unwrap();

        let rows = fetch_tunnel(&conn, "ETH", None, None, 10).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].has_upper_full);
        assert!(rows[0].has_lower_full);
    }

    #[test]
    fn fetch_tunnel_reads_explicit_presence_flags() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE exit_tunnel (
                ts_ms INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                upper_full REAL NOT NULL,
                has_upper_full INTEGER NOT NULL DEFAULT 1,
                upper_partial REAL,
                lower_full REAL NOT NULL,
                has_lower_full INTEGER NOT NULL DEFAULT 1,
                entry_price REAL NOT NULL,
                pos_type TEXT NOT NULL
            );
            INSERT INTO exit_tunnel (ts_ms, symbol, upper_full, has_upper_full, upper_partial, lower_full, has_lower_full, entry_price, pos_type)
            VALUES (1000, 'ETH', 0.0, 0, NULL, 99.0, 1, 100.0, 'LONG');
            "#,
        )
        .unwrap();

        let rows = fetch_tunnel(&conn, "ETH", None, None, 10).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].has_upper_full);
        assert!(rows[0].has_lower_full);
    }
}
