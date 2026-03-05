//! Rust runtime foundation contracts.
//!
//! This crate exposes the stable pipeline and runtime bootstrap types used by
//! the first Rust-only migration slice.

pub mod pipeline;
pub mod paper;
pub mod runtime;
pub mod snapshot;

pub use pipeline::{
    resolve_pipeline, PipelinePlan, PipelineResolveError, StageDescriptor, StageId, StagePlan,
    StageRegistry, DEFAULT_PROFILE, DEFAULT_RANKER,
};
pub use paper::{
    restore_paper_state, PaperBootstrapError, PaperBootstrapReport, PaperBootstrapState,
    PaperPositionState,
};
pub use runtime::{build_bootstrap, RuntimeBootstrap, RuntimeMode};
pub use snapshot::{
    load_snapshot, snapshot_to_pretty_json, SnapshotError, SnapshotFile, SnapshotPosition,
    SnapshotRuntimeState, SnapshotSummary, SNAPSHOT_V1, SNAPSHOT_V2,
};
