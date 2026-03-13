use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command;

use crate::ws::broadcast::BroadcastHub;

use super::{run_subprocess, JobId, JobStore};

pub struct FactoryArgs {
    pub executor_bin: String,
    pub config_path: String,
    pub settings_path: String,
    pub profile: String,
}

pub async fn spawn_factory(
    job_id: JobId,
    args: FactoryArgs,
    aiq_root: String,
    store: Arc<JobStore>,
    broadcast: BroadcastHub,
) {
    let mut cmd = Command::new(&args.executor_bin);
    cmd.arg("run")
        .arg("--config")
        .arg(&args.config_path)
        .arg("--settings")
        .arg(&args.settings_path)
        .arg("--profile")
        .arg(&args.profile);

    let _ = Path::new(&args.executor_bin);

    cmd.current_dir(aiq_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let jid = job_id.clone();
    tokio::spawn(async move {
        run_subprocess(jid, "factory", cmd, store, broadcast).await;
    });
}
