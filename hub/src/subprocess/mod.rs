pub mod backtester;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::process::Child;
use tokio::sync::Mutex;

/// Unique job identifier.
pub type JobId = String;

/// Job status enum.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Running,
    Done,
    Failed,
    Cancelled,
}

/// Metadata for a running or completed job.
#[derive(Debug, Clone, Serialize)]
pub struct JobInfo {
    pub id: JobId,
    pub kind: String, // "backtest" or "sweep"
    pub status: JobStatus,
    pub created_at: String,
    pub finished_at: Option<String>,
    pub stderr_tail: Vec<String>,
    pub result_json: Option<serde_json::Value>,
    pub error: Option<String>,
}

/// Thread-safe job store.
pub struct JobStore {
    pub jobs: Mutex<HashMap<JobId, JobInfo>>,
    pub handles: Mutex<HashMap<JobId, Child>>,
}

impl JobStore {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            jobs: Mutex::new(HashMap::new()),
            handles: Mutex::new(HashMap::new()),
        })
    }
}
