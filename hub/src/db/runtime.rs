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
