use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use std::collections::BTreeSet;
use std::path::Path;

const TABLE_NAME: &str = "decision_events";
const LEGACY_TABLE_NAME: &str = "decision_events_legacy_contract_backup";
const REQUIRED_COLUMNS: &[&str] = &[
    "id",
    "timestamp_ms",
    "symbol",
    "event_type",
    "status",
    "decision_phase",
];
const OPTIONAL_COLUMNS: &[(&str, &str)] = &[
    ("parent_decision_id", "TEXT"),
    ("trade_id", "INTEGER"),
    ("triggered_by", "TEXT"),
    ("action_taken", "TEXT"),
    ("rejection_reason", "TEXT"),
    ("context_json", "TEXT"),
    ("config_fingerprint", "TEXT"),
    ("run_fingerprint", "TEXT"),
    ("reason_code", "TEXT"),
];

pub(crate) struct FillDecisionEvent<'a> {
    pub(crate) id: &'a str,
    pub(crate) timestamp_ms: i64,
    pub(crate) symbol: &'a str,
    pub(crate) trade_id: i64,
    pub(crate) config_fingerprint: &'a str,
    pub(crate) run_fingerprint: &'a str,
    pub(crate) reason_code: Option<&'a str>,
}

pub(crate) fn reconcile_db_schema(db_path: &Path) -> Result<()> {
    let mut conn = Connection::open(db_path)
        .with_context(|| format!("failed to open decision event db: {}", db_path.display()))?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    ensure_schema_tx(&tx)?;
    tx.commit()?;
    Ok(())
}

pub(crate) fn ensure_schema_tx(tx: &Transaction<'_>) -> Result<()> {
    if !table_exists_tx(tx, TABLE_NAME)? {
        create_table_tx(tx, TABLE_NAME)?;
        create_indexes_tx(tx, TABLE_NAME)?;
        return Ok(());
    }

    let columns = table_columns_tx(tx, TABLE_NAME)?;
    if has_required_columns(&columns) {
        add_missing_optional_columns_tx(tx, TABLE_NAME, &columns)?;
        create_indexes_tx(tx, TABLE_NAME)?;
        return Ok(());
    }

    migrate_to_canonical_tx(tx, &columns)
}

pub(crate) fn insert_fill_tx(tx: &Transaction<'_>, record: &FillDecisionEvent<'_>) -> Result<()> {
    ensure_schema_tx(tx)?;
    tx.execute(
        "INSERT INTO decision_events (
             id, timestamp_ms, symbol, event_type, status, decision_phase,
             trade_id, config_fingerprint, run_fingerprint, reason_code
         ) VALUES (?1, ?2, ?3, 'fill', 'executed', 'execution', ?4, ?5, ?6, ?7)",
        params![
            record.id,
            record.timestamp_ms,
            record.symbol,
            record.trade_id,
            record.config_fingerprint,
            record.run_fingerprint,
            record.reason_code,
        ],
    )?;
    Ok(())
}

fn migrate_to_canonical_tx(tx: &Transaction<'_>, columns: &BTreeSet<String>) -> Result<()> {
    let row_count: i64 =
        tx.query_row("SELECT COUNT(*) FROM decision_events", [], |row| row.get(0))?;

    if row_count > 0 {
        if !columns.contains("trade_id") {
            anyhow::bail!(
                "decision_events schema reconcile cannot backfill {TABLE_NAME}.timestamp_ms/symbol because trade_id is missing"
            );
        }
        if !columns.contains("timestamp_ms") {
            let missing_timestamps: i64 = tx.query_row(
                "SELECT COUNT(*)
                 FROM decision_events d
                 LEFT JOIN trades t ON t.id = d.trade_id
                 WHERE d.trade_id IS NULL OR t.timestamp IS NULL",
                [],
                |row| row.get(0),
            )?;
            if missing_timestamps > 0 {
                anyhow::bail!(
                    "decision_events schema reconcile refused to backfill timestamp_ms for {missing_timestamps} row(s)"
                );
            }
        }
        if !columns.contains("symbol") {
            let missing_symbols: i64 = tx.query_row(
                "SELECT COUNT(*)
                 FROM decision_events d
                 LEFT JOIN trades t ON t.id = d.trade_id
                 WHERE d.trade_id IS NULL OR t.symbol IS NULL OR trim(t.symbol) = ''",
                [],
                |row| row.get(0),
            )?;
            if missing_symbols > 0 {
                anyhow::bail!(
                    "decision_events schema reconcile refused to backfill symbol for {missing_symbols} row(s)"
                );
            }
        }
        if !columns.contains("decision_phase") {
            let non_fill_rows: i64 = tx.query_row(
                "SELECT COUNT(*) FROM decision_events WHERE event_type <> 'fill'",
                [],
                |row| row.get(0),
            )?;
            if non_fill_rows > 0 {
                anyhow::bail!(
                    "decision_events schema reconcile refused to infer decision_phase for {non_fill_rows} non-fill row(s)"
                );
            }
        }
    }

    tx.execute_batch(&format!(
        "ALTER TABLE {TABLE_NAME} RENAME TO {LEGACY_TABLE_NAME};"
    ))?;
    create_table_tx(tx, TABLE_NAME)?;

    // Backfill reduced rows from trades when the old table omitted the legacy required columns.
    tx.execute_batch(&format!(
        "INSERT INTO {TABLE_NAME} (
             id, timestamp_ms, symbol, event_type, status, decision_phase,
             parent_decision_id, trade_id, triggered_by, action_taken,
             rejection_reason, context_json, config_fingerprint, run_fingerprint, reason_code
         )
         SELECT
             legacy.id,
             {timestamp_ms_expr},
             {symbol_expr},
             legacy.event_type,
             legacy.status,
             {decision_phase_expr},
             {parent_decision_id_expr},
             {trade_id_expr},
             {triggered_by_expr},
             {action_taken_expr},
             {rejection_reason_expr},
             {context_json_expr},
             {config_fingerprint_expr},
             {run_fingerprint_expr},
             {reason_code_expr}
         FROM {LEGACY_TABLE_NAME} AS legacy
         LEFT JOIN trades AS trade_row
           ON trade_row.id = legacy.trade_id;",
        timestamp_ms_expr = if columns.contains("timestamp_ms") {
            "legacy.timestamp_ms"
        } else {
            "CAST((julianday(trade_row.timestamp) - 2440587.5) * 86400000.0 AS INTEGER)"
        },
        symbol_expr = if columns.contains("symbol") {
            "legacy.symbol"
        } else {
            "trade_row.symbol"
        },
        decision_phase_expr = if columns.contains("decision_phase") {
            "legacy.decision_phase"
        } else {
            "'execution'"
        },
        parent_decision_id_expr = existing_or_null(columns, "parent_decision_id"),
        trade_id_expr = existing_or_null(columns, "trade_id"),
        triggered_by_expr = existing_or_null(columns, "triggered_by"),
        action_taken_expr = existing_or_null(columns, "action_taken"),
        rejection_reason_expr = existing_or_null(columns, "rejection_reason"),
        context_json_expr = existing_or_null(columns, "context_json"),
        config_fingerprint_expr = existing_or_null(columns, "config_fingerprint"),
        run_fingerprint_expr = existing_or_null(columns, "run_fingerprint"),
        reason_code_expr = existing_or_null(columns, "reason_code"),
    ))?;

    tx.execute_batch(&format!("DROP TABLE {LEGACY_TABLE_NAME};"))?;
    create_indexes_tx(tx, TABLE_NAME)?;
    Ok(())
}

fn add_missing_optional_columns_tx(
    tx: &Transaction<'_>,
    table_name: &str,
    columns: &BTreeSet<String>,
) -> Result<()> {
    for (column, column_type) in OPTIONAL_COLUMNS {
        if columns.contains(*column) {
            continue;
        }
        tx.execute_batch(&format!(
            "ALTER TABLE {table_name} ADD COLUMN {column} {column_type};"
        ))?;
    }
    Ok(())
}

fn create_table_tx(tx: &Transaction<'_>, table_name: &str) -> Result<()> {
    tx.execute_batch(&format!(
        r#"
        CREATE TABLE {table_name} (
            id TEXT PRIMARY KEY,
            timestamp_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            event_type TEXT NOT NULL,
            status TEXT NOT NULL,
            decision_phase TEXT NOT NULL,
            parent_decision_id TEXT,
            trade_id INTEGER,
            triggered_by TEXT,
            action_taken TEXT,
            rejection_reason TEXT,
            context_json TEXT,
            config_fingerprint TEXT,
            run_fingerprint TEXT,
            reason_code TEXT
        );
        "#
    ))?;
    Ok(())
}

fn create_indexes_tx(tx: &Transaction<'_>, table_name: &str) -> Result<()> {
    tx.execute_batch(&format!(
        r#"
        CREATE INDEX IF NOT EXISTS idx_de_symbol_ts ON {table_name}(symbol, timestamp_ms);
        CREATE INDEX IF NOT EXISTS idx_de_event_type ON {table_name}(event_type);
        CREATE INDEX IF NOT EXISTS idx_de_trade_id ON {table_name}(trade_id);
        CREATE INDEX IF NOT EXISTS idx_de_config_fingerprint ON {table_name}(config_fingerprint);
        CREATE INDEX IF NOT EXISTS idx_de_run_fingerprint ON {table_name}(run_fingerprint);
        CREATE INDEX IF NOT EXISTS idx_de_reason_code ON {table_name}(reason_code);
        "#
    ))?;
    Ok(())
}

fn has_required_columns(columns: &BTreeSet<String>) -> bool {
    REQUIRED_COLUMNS
        .iter()
        .all(|column| columns.contains(*column))
}

fn existing_or_null(columns: &BTreeSet<String>, column: &str) -> &'static str {
    if columns.contains(column) {
        match column {
            "parent_decision_id" => "legacy.parent_decision_id",
            "trade_id" => "legacy.trade_id",
            "triggered_by" => "legacy.triggered_by",
            "action_taken" => "legacy.action_taken",
            "rejection_reason" => "legacy.rejection_reason",
            "context_json" => "legacy.context_json",
            "config_fingerprint" => "legacy.config_fingerprint",
            "run_fingerprint" => "legacy.run_fingerprint",
            "reason_code" => "legacy.reason_code",
            _ => "NULL",
        }
    } else {
        "NULL"
    }
}

fn table_columns_tx(tx: &Transaction<'_>, table_name: &str) -> Result<BTreeSet<String>> {
    let mut stmt = tx.prepare(&format!("PRAGMA table_info({table_name})"))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows.into_iter().collect())
}

fn table_exists_tx(tx: &Transaction<'_>, table_name: &str) -> Result<bool> {
    Ok(tx
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1 LIMIT 1",
            [table_name],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_reduced_decision_events(conn: &Connection) {
        conn.execute_batch(
            r#"
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT
            );
            CREATE TABLE decision_events (
                id TEXT PRIMARY KEY,
                trade_id INTEGER,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                config_fingerprint TEXT,
                run_fingerprint TEXT
            );
            INSERT INTO trades (id, timestamp, symbol)
            VALUES (7, '2026-03-14T12:34:56.789+00:00', 'ETH');
            INSERT INTO decision_events (id, trade_id, event_type, status, config_fingerprint, run_fingerprint)
            VALUES ('evt-1', 7, 'fill', 'executed', 'cfg', 'run');
            "#,
        )
        .unwrap();
    }

    #[test]
    fn ensure_schema_tx_migrates_reduced_fill_rows() {
        let mut conn = Connection::open_in_memory().unwrap();
        create_reduced_decision_events(&conn);

        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        ensure_schema_tx(&tx).unwrap();
        tx.commit().unwrap();

        let columns = conn
            .prepare("PRAGMA table_info(decision_events)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        for column in REQUIRED_COLUMNS {
            assert!(columns.iter().any(|current| current == column));
        }

        let row = conn
            .query_row(
                "SELECT timestamp_ms, symbol, decision_phase, config_fingerprint, run_fingerprint
                 FROM decision_events WHERE id = 'evt-1'",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<String>>(4)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(row.1, "ETH");
        assert_eq!(row.2, "execution");
        assert_eq!(row.3.as_deref(), Some("cfg"));
        assert_eq!(row.4.as_deref(), Some("run"));
        assert!(row.0 > 0);

        let indexes = conn
            .prepare(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='decision_events'",
            )
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert!(indexes.iter().any(|name| name == "idx_de_symbol_ts"));
        assert!(indexes.iter().any(|name| name == "idx_de_trade_id"));
    }
}
