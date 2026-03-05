use crate::pipeline::{resolve_pipeline, PipelinePlan, PipelineResolveError, StageRegistry};
use bt_core::config::{strategy_config_fingerprint_sha256, StrategyConfig};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMode {
    Live,
    Paper,
    Replay,
    Backtest,
    SweepCpu,
    Doctor,
    Migrate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeBootstrap {
    pub mode: RuntimeMode,
    pub config_fingerprint: String,
    pub pipeline: PipelinePlan,
}

pub fn build_bootstrap(
    config: &StrategyConfig,
    mode: RuntimeMode,
    profile_override: Option<&str>,
) -> Result<RuntimeBootstrap, PipelineResolveError> {
    let registry = StageRegistry::default();
    let pipeline = resolve_pipeline(config, profile_override, &registry)?;

    Ok(RuntimeBootstrap {
        config_fingerprint: strategy_config_fingerprint_sha256(config),
        mode,
        pipeline,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_includes_pipeline_and_fingerprint() {
        let config = StrategyConfig::default();
        let bootstrap = build_bootstrap(&config, RuntimeMode::Doctor, None).unwrap();

        assert_eq!(bootstrap.mode, RuntimeMode::Doctor);
        assert_eq!(bootstrap.pipeline.profile, "production");
        assert_eq!(bootstrap.config_fingerprint.len(), 64);
    }
}
