use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::config_audit::ConfigAuditActor;
use crate::error::HubError;

const DIAGNOSTIC_AUDIT_DIR: &str = "diagnostic_audit";
const DIAGNOSTIC_AUDIT_LEDGER: &str = "read_events.jsonl";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReadEvent {
    pub version: String,
    pub ts_ms: i64,
    pub ts_utc: String,
    pub route: String,
    pub target: String,
    pub actor: ConfigAuditActor,
}

pub fn append_read_event(
    artifacts_dir: &Path,
    event: &DiagnosticReadEvent,
) -> Result<PathBuf, HubError> {
    let ledger_path = ledger_path(artifacts_dir);
    if let Some(parent) = ledger_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = serde_json::to_string(event)?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .read(true)
        .open(&ledger_path)?;
    file.lock_exclusive()?;
    writeln!(file, "{payload}")?;
    file.sync_data()?;
    file.unlock()?;
    Ok(ledger_path)
}

fn ledger_path(artifacts_dir: &Path) -> PathBuf {
    artifacts_dir
        .join(DIAGNOSTIC_AUDIT_DIR)
        .join(DIAGNOSTIC_AUDIT_LEDGER)
}
