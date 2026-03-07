use bt_core::config::{PipelineProfileConfig, StrategyConfig};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;
use thiserror::Error;

pub const DEFAULT_PROFILE: &str = "production";
pub const DEFAULT_RANKER: &str = "confidence_adx";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageId {
    MarketDataNormalisation,
    IndicatorBuild,
    GateEvaluation,
    SignalGeneration,
    Ranking,
    RiskChecks,
    IntentGeneration,
    OmsTransition,
    BrokerExecution,
    FillReconciliation,
    PersistenceAudit,
}

impl StageId {
    pub const ALL: [StageId; 11] = [
        StageId::MarketDataNormalisation,
        StageId::IndicatorBuild,
        StageId::GateEvaluation,
        StageId::SignalGeneration,
        StageId::Ranking,
        StageId::RiskChecks,
        StageId::IntentGeneration,
        StageId::OmsTransition,
        StageId::BrokerExecution,
        StageId::FillReconciliation,
        StageId::PersistenceAudit,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            StageId::MarketDataNormalisation => "market_data_normalisation",
            StageId::IndicatorBuild => "indicator_build",
            StageId::GateEvaluation => "gate_evaluation",
            StageId::SignalGeneration => "signal_generation",
            StageId::Ranking => "ranking",
            StageId::RiskChecks => "risk_checks",
            StageId::IntentGeneration => "intent_generation",
            StageId::OmsTransition => "oms_transition",
            StageId::BrokerExecution => "broker_execution",
            StageId::FillReconciliation => "fill_reconciliation",
            StageId::PersistenceAudit => "persistence_audit",
        }
    }
}

impl fmt::Display for StageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str((*self).as_str())
    }
}

impl FromStr for StageId {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "market_data_normalisation" => Ok(StageId::MarketDataNormalisation),
            "indicator_build" => Ok(StageId::IndicatorBuild),
            "gate_evaluation" => Ok(StageId::GateEvaluation),
            "signal_generation" => Ok(StageId::SignalGeneration),
            "ranking" => Ok(StageId::Ranking),
            "risk_checks" => Ok(StageId::RiskChecks),
            "intent_generation" => Ok(StageId::IntentGeneration),
            "oms_transition" => Ok(StageId::OmsTransition),
            "broker_execution" => Ok(StageId::BrokerExecution),
            "fill_reconciliation" => Ok(StageId::FillReconciliation),
            "persistence_audit" => Ok(StageId::PersistenceAudit),
            other => Err(other.to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageDescriptor {
    pub id: StageId,
    pub description: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankerDescriptor {
    pub id: &'static str,
    pub description: &'static str,
}

#[derive(Debug, Clone)]
pub struct StageRegistry {
    descriptors: BTreeMap<StageId, StageDescriptor>,
}

impl Default for StageRegistry {
    fn default() -> Self {
        let descriptors = [
            (
                StageId::MarketDataNormalisation,
                "Normalise market data snapshots before any downstream processing.",
            ),
            (
                StageId::IndicatorBuild,
                "Build indicator state shared by live, paper, and replay surfaces.",
            ),
            (
                StageId::GateEvaluation,
                "Apply pre-signal gates and stage-level feature toggles.",
            ),
            (
                StageId::SignalGeneration,
                "Generate raw strategy directions and confidence output.",
            ),
            (
                StageId::Ranking,
                "Apply the configured ranker to competing entry candidates.",
            ),
            (
                StageId::RiskChecks,
                "Run exposure, cooldown, and runtime safety guards.",
            ),
            (
                StageId::IntentGeneration,
                "Build canonical order intents from approved opportunities.",
            ),
            (
                StageId::OmsTransition,
                "Advance the persistent OMS state machine for intents and fills.",
            ),
            (
                StageId::BrokerExecution,
                "Submit exchange-side side effects or paper fills.",
            ),
            (
                StageId::FillReconciliation,
                "Reconcile broker/exchange fills back into runtime state.",
            ),
            (
                StageId::PersistenceAudit,
                "Persist runtime snapshots, decision traces, and audit artefacts.",
            ),
        ]
        .into_iter()
        .map(|(id, description)| (id, StageDescriptor { id, description }))
        .collect();

        Self { descriptors }
    }
}

impl StageRegistry {
    pub fn core() -> Self {
        Self::default()
    }

    pub fn default_order(&self) -> Vec<StageId> {
        StageId::ALL.to_vec()
    }

    pub fn descriptors(&self) -> impl Iterator<Item = &StageDescriptor> {
        self.descriptors.values()
    }

    pub fn descriptor(&self, stage_id: StageId) -> Option<&StageDescriptor> {
        self.descriptors.get(&stage_id)
    }

    fn contains(&self, stage_id: &StageId) -> bool {
        self.descriptors.contains_key(stage_id)
    }
}

#[derive(Debug, Clone)]
pub struct RankerRegistry {
    descriptors: BTreeMap<&'static str, RankerDescriptor>,
}

impl Default for RankerRegistry {
    fn default() -> Self {
        let descriptors = [
            (
                "confidence_adx",
                "Rank entry candidates by confidence tier, then ADX, then symbol.",
            ),
            (
                "raw_notional_desc",
                "Rank entry candidates by requested notional descending, then symbol.",
            ),
        ]
        .into_iter()
        .map(|(id, description)| (id, RankerDescriptor { id, description }))
        .collect();

        Self { descriptors }
    }
}

impl RankerRegistry {
    pub fn core() -> Self {
        Self::default()
    }

    pub fn descriptors(&self) -> impl Iterator<Item = &RankerDescriptor> {
        self.descriptors.values()
    }

    pub fn descriptor(&self, ranker_id: &str) -> Option<&RankerDescriptor> {
        self.descriptors.get(ranker_id.trim())
    }

    pub fn contains(&self, ranker_id: &str) -> bool {
        self.descriptor(ranker_id).is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StagePlan {
    pub id: StageId,
    pub enabled: bool,
    pub description: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PipelinePlan {
    pub profile: String,
    pub ranker: String,
    pub state_backend: String,
    pub audit_sink: String,
    pub stages: Vec<StagePlan>,
}

impl PipelinePlan {
    pub fn ordered_stage_ids(&self) -> Vec<StageId> {
        self.stages.iter().map(|stage| stage.id).collect()
    }

    pub fn enabled_stage_ids(&self) -> Vec<StageId> {
        self.stages
            .iter()
            .filter(|stage| stage.enabled)
            .map(|stage| stage.id)
            .collect()
    }

    pub fn is_enabled(&self, stage_id: StageId) -> bool {
        self.stages
            .iter()
            .find(|stage| stage.id == stage_id)
            .map(|stage| stage.enabled)
            .unwrap_or(false)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PipelineResolveError {
    #[error("unknown pipeline stage `{stage}` in profile `{profile}`")]
    UnknownStage { profile: String, stage: String },
    #[error("unknown pipeline ranker `{ranker}` in profile `{profile}`")]
    UnknownRanker { profile: String, ranker: String },
    #[error("profile `{profile}` lists conflicting enabled/disabled stage `{stage}`")]
    ConflictingStage { profile: String, stage: String },
    #[error("profile `{profile}` stage_order must be a complete permutation of all known stages")]
    IncompleteStageOrder { profile: String },
}

pub fn resolve_pipeline(
    config: &StrategyConfig,
    profile_override: Option<&str>,
    registry: &StageRegistry,
) -> Result<PipelinePlan, PipelineResolveError> {
    let profile = profile_override
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            let configured = config.runtime.profile.trim();
            (!configured.is_empty()).then(|| configured.to_string())
        })
        .unwrap_or_else(|| {
            let configured = config.pipeline.default_profile.trim();
            if configured.is_empty() {
                DEFAULT_PROFILE.to_string()
            } else {
                configured.to_string()
            }
        });

    let effective_profile = build_effective_profile(config, &profile);
    let ranker_registry = RankerRegistry::core();
    let ranker = if effective_profile.ranker.trim().is_empty() {
        DEFAULT_RANKER.to_string()
    } else {
        effective_profile.ranker.trim().to_string()
    };
    if !ranker_registry.contains(&ranker) {
        return Err(PipelineResolveError::UnknownRanker { profile, ranker });
    }
    let stage_order = resolve_stage_order(&profile, &effective_profile, registry)?;
    let disabled = parse_stage_set(&profile, &effective_profile.disabled_stages, registry)?;
    let enabled = parse_stage_set(&profile, &effective_profile.enabled_stages, registry)?;

    if let Some(conflict) = enabled.intersection(&disabled).next() {
        return Err(PipelineResolveError::ConflictingStage {
            profile,
            stage: conflict.as_str().to_string(),
        });
    }

    let stages = stage_order
        .into_iter()
        .map(|id| StagePlan {
            enabled: if disabled.contains(&id) {
                false
            } else if enabled.is_empty() {
                true
            } else {
                enabled.contains(&id)
            },
            id,
            description: registry
                .descriptor(id)
                .map(|descriptor| descriptor.description)
                .unwrap_or("Unknown stage"),
        })
        .collect();

    Ok(PipelinePlan {
        profile,
        ranker,
        state_backend: config.runtime.state_backend.trim().to_string(),
        audit_sink: config.runtime.audit_sink.trim().to_string(),
        stages,
    })
}

fn build_effective_profile(config: &StrategyConfig, profile: &str) -> PipelineProfileConfig {
    let mut base = builtin_profile(profile).unwrap_or_else(|| PipelineProfileConfig {
        ranker: DEFAULT_RANKER.to_string(),
        ..PipelineProfileConfig::default()
    });

    if let Some(override_profile) = config.pipeline.profiles.get(profile) {
        if !override_profile.ranker.trim().is_empty() {
            base.ranker = override_profile.ranker.trim().to_string();
        }
        if !override_profile.stage_order.is_empty() {
            base.stage_order = override_profile.stage_order.clone();
        }
        if !override_profile.enabled_stages.is_empty() {
            base.enabled_stages = override_profile.enabled_stages.clone();
        }
        if !override_profile.disabled_stages.is_empty() {
            base.disabled_stages = override_profile.disabled_stages.clone();
        }
    }

    base
}

fn builtin_profile(profile: &str) -> Option<PipelineProfileConfig> {
    let profile = profile.trim().to_ascii_lowercase();
    let mut cfg = PipelineProfileConfig {
        ranker: DEFAULT_RANKER.to_string(),
        ..PipelineProfileConfig::default()
    };

    match profile.as_str() {
        "production" => Some(cfg),
        "parity_baseline" => {
            cfg.disabled_stages = vec![
                StageId::BrokerExecution.as_str().to_string(),
                StageId::FillReconciliation.as_str().to_string(),
            ];
            Some(cfg)
        }
        "stage_debug" => {
            cfg.disabled_stages = vec![
                StageId::Ranking.as_str().to_string(),
                StageId::RiskChecks.as_str().to_string(),
                StageId::OmsTransition.as_str().to_string(),
                StageId::BrokerExecution.as_str().to_string(),
                StageId::FillReconciliation.as_str().to_string(),
                StageId::PersistenceAudit.as_str().to_string(),
            ];
            Some(cfg)
        }
        _ => None,
    }
}

fn resolve_stage_order(
    profile: &str,
    config: &PipelineProfileConfig,
    registry: &StageRegistry,
) -> Result<Vec<StageId>, PipelineResolveError> {
    if config.stage_order.is_empty() {
        return Ok(registry.default_order());
    }

    let parsed = parse_stage_list(profile, &config.stage_order, registry)?;
    let expected: BTreeSet<_> = registry.default_order().into_iter().collect();
    let actual: BTreeSet<_> = parsed.iter().copied().collect();
    if actual != expected || parsed.len() != expected.len() {
        return Err(PipelineResolveError::IncompleteStageOrder {
            profile: profile.to_string(),
        });
    }

    Ok(parsed)
}

fn parse_stage_set(
    profile: &str,
    values: &[String],
    registry: &StageRegistry,
) -> Result<BTreeSet<StageId>, PipelineResolveError> {
    Ok(parse_stage_list(profile, values, registry)?
        .into_iter()
        .collect())
}

fn parse_stage_list(
    profile: &str,
    values: &[String],
    registry: &StageRegistry,
) -> Result<Vec<StageId>, PipelineResolveError> {
    values
        .iter()
        .map(|value| {
            StageId::from_str(value).map_err(|stage| PipelineResolveError::UnknownStage {
                profile: profile.to_string(),
                stage,
            })
        })
        .map(|result| {
            result.and_then(|stage_id| {
                if registry.contains(&stage_id) {
                    Ok(stage_id)
                } else {
                    Err(PipelineResolveError::UnknownStage {
                        profile: profile.to_string(),
                        stage: stage_id.as_str().to_string(),
                    })
                }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bt_core::config::{PipelineConfig, RuntimeConfig};
    use std::collections::BTreeMap;

    fn config_with_runtime() -> StrategyConfig {
        StrategyConfig {
            pipeline: PipelineConfig::default(),
            runtime: RuntimeConfig::default(),
            ..StrategyConfig::default()
        }
    }

    #[test]
    fn resolves_default_production_pipeline() {
        let registry = StageRegistry::default();
        let plan = resolve_pipeline(&config_with_runtime(), None, &registry).unwrap();

        assert_eq!(plan.profile, DEFAULT_PROFILE);
        assert_eq!(plan.ranker, DEFAULT_RANKER);
        assert!(plan.stages.iter().all(|stage| stage.enabled));
        assert_eq!(
            plan.stages.first().unwrap().id,
            StageId::MarketDataNormalisation
        );
        assert_eq!(plan.stages.last().unwrap().id, StageId::PersistenceAudit);
    }

    #[test]
    fn resolves_custom_profile_overrides() {
        let registry = StageRegistry::default();
        let mut config = config_with_runtime();
        config.runtime.profile = "nightly_probe".to_string();
        config.pipeline.profiles = BTreeMap::from([(
            "nightly_probe".to_string(),
            PipelineProfileConfig {
                ranker: "raw_notional_desc".to_string(),
                stage_order: vec![
                    "market_data_normalisation".to_string(),
                    "indicator_build".to_string(),
                    "gate_evaluation".to_string(),
                    "signal_generation".to_string(),
                    "risk_checks".to_string(),
                    "intent_generation".to_string(),
                    "ranking".to_string(),
                    "oms_transition".to_string(),
                    "broker_execution".to_string(),
                    "fill_reconciliation".to_string(),
                    "persistence_audit".to_string(),
                ],
                enabled_stages: Vec::new(),
                disabled_stages: vec!["broker_execution".to_string()],
            },
        )]);

        let plan = resolve_pipeline(&config, None, &registry).unwrap();

        assert_eq!(plan.profile, "nightly_probe");
        assert_eq!(plan.ranker, "raw_notional_desc");
        assert_eq!(plan.stages[4].id, StageId::RiskChecks);
        assert!(
            !plan
                .stages
                .iter()
                .find(|stage| stage.id == StageId::BrokerExecution)
                .unwrap()
                .enabled
        );
    }

    #[test]
    fn rejects_incomplete_stage_order() {
        let registry = StageRegistry::default();
        let mut config = config_with_runtime();
        config.runtime.profile = "broken".to_string();
        config.pipeline.profiles = BTreeMap::from([(
            "broken".to_string(),
            PipelineProfileConfig {
                stage_order: vec![
                    "market_data_normalisation".to_string(),
                    "indicator_build".to_string(),
                ],
                ..PipelineProfileConfig::default()
            },
        )]);

        let err = resolve_pipeline(&config, None, &registry).unwrap_err();
        assert_eq!(
            err,
            PipelineResolveError::IncompleteStageOrder {
                profile: "broken".to_string(),
            }
        );
    }

    #[test]
    fn rejects_unknown_ranker() {
        let registry = StageRegistry::default();
        let mut config = config_with_runtime();
        config.runtime.profile = "broken".to_string();
        config.pipeline.profiles = BTreeMap::from([(
            "broken".to_string(),
            PipelineProfileConfig {
                ranker: "not_real".to_string(),
                ..PipelineProfileConfig::default()
            },
        )]);

        let err = resolve_pipeline(&config, None, &registry).unwrap_err();
        assert_eq!(
            err,
            PipelineResolveError::UnknownRanker {
                profile: "broken".to_string(),
                ranker: "not_real".to_string(),
            }
        );
    }

    #[test]
    fn pipeline_plan_reports_enabled_and_ordered_stage_ids() {
        let registry = StageRegistry::default();
        let mut config = config_with_runtime();
        config.runtime.profile = "parity_baseline".to_string();
        let plan = resolve_pipeline(&config, None, &registry).unwrap();

        assert_eq!(plan.ordered_stage_ids().len(), StageId::ALL.len());
        assert!(plan.is_enabled(StageId::Ranking));
        assert!(!plan.is_enabled(StageId::BrokerExecution));
        assert!(plan.enabled_stage_ids().contains(&StageId::Ranking));
        assert!(!plan.enabled_stage_ids().contains(&StageId::BrokerExecution));
    }
}
