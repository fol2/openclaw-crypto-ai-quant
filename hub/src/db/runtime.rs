use rusqlite::Connection;

use crate::error::HubError;
use crate::heartbeat::{parse_heartbeat_line, Heartbeat};

/// Fetch and parse the latest heartbeat from the `runtime_logs` table.
pub fn fetch_heartbeat_from_db(conn: &Connection) -> Result<Option<Heartbeat>, HubError> {
    let mut stmt = conn.prepare(
        "SELECT ts_ms, message FROM runtime_logs
         WHERE message LIKE '%engine ok%'
         ORDER BY ts_ms DESC LIMIT 1",
    )?;

    let result = stmt.query_row([], |row| {
        let ts_ms: i64 = row.get(0)?;
        let message: String = row.get(1)?;
        Ok((ts_ms, message))
    });

    match result {
        Ok((ts_ms, message)) => {
            let hb = parse_heartbeat_line(&message, Some(ts_ms), "sqlite");
            Ok(Some(hb))
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Find the earliest runtime log timestamp for a specific deployed config identity.
pub fn first_seen_config_id_ts_ms(
    conn: &Connection,
    config_id: &str,
) -> Result<Option<i64>, HubError> {
    let like_pattern = format!("%config_id={config_id}%");
    let mut stmt = conn.prepare(
        "SELECT MIN(ts_ms) FROM runtime_logs
         WHERE message LIKE ?",
    )?;
    let ts_ms = stmt.query_row([like_pattern], |row| row.get::<_, Option<i64>>(0))?;
    Ok(ts_ms)
}
