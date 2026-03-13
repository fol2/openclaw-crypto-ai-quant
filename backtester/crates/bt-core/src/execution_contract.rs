use crate::behaviour::ResolvedBehaviourPlan;
use crate::behaviour::{
    DEFAULT_ENGINE_BEHAVIOURS, DEFAULT_ENTRY_PROGRESSION_BEHAVIOURS,
    DEFAULT_ENTRY_SIZING_BEHAVIOURS, DEFAULT_EXIT_BEHAVIOURS,
    DEFAULT_EXIT_SMART_EXTENDED_BEHAVIOURS, DEFAULT_RISK_BEHAVIOURS,
};
use crate::config::StrategyConfig;
use bt_signals::behaviour::{
    BehaviourDescriptor, BehaviourGroupPlan, DEFAULT_GATE_BEHAVIOURS,
    DEFAULT_SIGNAL_CONFIDENCE_BEHAVIOURS,
};

const DISABLED_SENTINEL: f64 = 1.0e12;

#[derive(Debug, Clone)]
pub struct ResolvedExecutionConfig {
    pub active_profile: String,
    pub effective_cfg: StrategyConfig,
    pub behaviour_plan: ResolvedBehaviourPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuExecutionContractError {
    ResolvePlan(String),
    UnsupportedOrder {
        group: &'static str,
        expected: Vec<String>,
        got: Vec<String>,
    },
    UnsupportedDisable {
        group: &'static str,
        id: String,
    },
}

impl std::fmt::Display for GpuExecutionContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ResolvePlan(err) => write!(f, "{err}"),
            Self::UnsupportedOrder {
                group,
                expected,
                got,
            } => write!(
                f,
                "GPU path only supports canonical order for `{group}` (expected {:?}, got {:?})",
                expected, got
            ),
            Self::UnsupportedDisable { group, id } => write!(
                f,
                "GPU path does not yet support disabling `{id}` in behaviour group `{group}`"
            ),
        }
    }
}

impl std::error::Error for GpuExecutionContractError {}

pub fn resolve_execution_config(
    cfg: &StrategyConfig,
    profile_override: Option<&str>,
) -> Result<ResolvedExecutionConfig, bt_signals::behaviour::BehaviourResolveError> {
    let active_profile = cfg.resolve_active_profile(profile_override);
    let behaviour_plan = cfg.resolve_behaviour_plan(profile_override)?;
    let effective_cfg = apply_behaviour_overrides(cfg, &behaviour_plan);
    Ok(ResolvedExecutionConfig {
        active_profile,
        effective_cfg,
        behaviour_plan,
    })
}

pub fn resolve_gpu_execution_config(
    cfg: &StrategyConfig,
    profile_override: Option<&str>,
) -> Result<ResolvedExecutionConfig, GpuExecutionContractError> {
    let resolved = resolve_execution_config(cfg, profile_override)
        .map_err(|err| GpuExecutionContractError::ResolvePlan(err.to_string()))?;
    validate_gpu_behaviour_contract(&resolved.behaviour_plan)?;
    Ok(resolved)
}

pub fn apply_behaviour_overrides(
    cfg: &StrategyConfig,
    behaviour_plan: &ResolvedBehaviourPlan,
) -> StrategyConfig {
    let mut cfg = cfg.clone();
    if !behaviour_plan.gates.is_enabled("gate.ranging_vote") {
        cfg.filters.enable_ranging_filter = false;
    }
    if !behaviour_plan.gates.is_enabled("gate.anomaly_filter") {
        cfg.filters.enable_anomaly_filter = false;
    }
    if !behaviour_plan.gates.is_enabled("gate.extension_filter") {
        cfg.filters.enable_extension_filter = false;
    }
    if !behaviour_plan.gates.is_enabled("gate.volume_confirmation") {
        cfg.filters.require_volume_confirmation = false;
    }
    if !behaviour_plan.gates.is_enabled("gate.adx_rising") {
        cfg.filters.require_adx_rising = false;
    }
    if !behaviour_plan.gates.is_enabled("gate.adx_floor") {
        cfg.thresholds.entry.min_adx = 0.0;
    }
    if !behaviour_plan
        .gates
        .is_enabled("gate.adx_floor.ave_multiplier")
    {
        cfg.thresholds.entry.ave_enabled = false;
    }
    if !behaviour_plan.gates.is_enabled("gate.alignment.macro") {
        cfg.filters.require_macro_alignment = false;
    }
    if !behaviour_plan.gates.is_enabled("gate.alignment.btc") {
        cfg.filters.require_btc_alignment = false;
    }
    if !behaviour_plan
        .signal_modes
        .is_enabled("signal.mode.pullback")
    {
        cfg.thresholds.entry.enable_pullback_entries = false;
    }
    if !behaviour_plan
        .signal_modes
        .is_enabled("signal.mode.slow_drift")
    {
        cfg.thresholds.entry.enable_slow_drift_entries = false;
    }
    if !behaviour_plan
        .signal_confidence
        .is_enabled("signal.confidence.high_volume_upgrade")
    {
        cfg.thresholds.entry.high_conf_volume_mult = DISABLED_SENTINEL;
    }
    if !behaviour_plan.engine.is_enabled("engine.atr_floor") {
        cfg.trade.min_atr_pct = 0.0;
    }
    if !behaviour_plan.engine.is_enabled("engine.reverse_signal") {
        cfg.trade.reverse_entry_signal = false;
        cfg.market_regime.enable_auto_reverse = false;
    }
    if !behaviour_plan.engine.is_enabled("engine.regime_filter") {
        cfg.market_regime.enable_regime_filter = false;
    }
    if !behaviour_plan
        .entry_progression
        .is_enabled("entry.progression.pyramiding")
    {
        cfg.trade.enable_pyramiding = false;
    }
    if !behaviour_plan
        .entry_progression
        .is_enabled("entry.progression.add_cooldown")
    {
        cfg.trade.add_cooldown_minutes = 0;
    }
    if !behaviour_plan.risk.is_enabled("risk.entry_cooldown") {
        cfg.trade.entry_cooldown_s = 0;
    }
    if !behaviour_plan.risk.is_enabled("risk.exit_cooldown") {
        cfg.trade.exit_cooldown_s = 0;
    }
    if !behaviour_plan.risk.is_enabled("risk.pesc") {
        cfg.trade.reentry_cooldown_minutes = 0;
        cfg.trade.reentry_cooldown_min_mins = 0;
        cfg.trade.reentry_cooldown_max_mins = 0;
    }
    if !behaviour_plan
        .entry_sizing
        .is_enabled("entry.sizing.dynamic")
    {
        cfg.trade.enable_dynamic_sizing = false;
    }
    if !behaviour_plan
        .entry_sizing
        .is_enabled("entry.sizing.confidence_multiplier")
    {
        cfg.trade.confidence_mult_high = 1.0;
        cfg.trade.confidence_mult_medium = 1.0;
        cfg.trade.confidence_mult_low = 1.0;
    }
    if !behaviour_plan
        .entry_sizing
        .is_enabled("entry.sizing.adx_multiplier")
    {
        cfg.trade.adx_sizing_min_mult = 1.0;
        cfg.trade.adx_sizing_full_adx = 0.0;
    }
    if !behaviour_plan
        .entry_sizing
        .is_enabled("entry.sizing.volatility_scalar")
    {
        cfg.trade.vol_baseline_pct = 0.0;
        cfg.trade.vol_scalar_min = 1.0;
        cfg.trade.vol_scalar_max = 1.0;
    }
    if !behaviour_plan
        .entry_sizing
        .is_enabled("entry.sizing.min_notional_bump")
    {
        cfg.trade.bump_to_min_notional = false;
    }
    if !behaviour_plan.exits.is_enabled("exit.stop_loss.breakeven") {
        cfg.trade.enable_breakeven_stop = false;
        cfg.trade.breakeven_start_atr = 0.0;
        cfg.trade.breakeven_buffer_atr = 0.0;
    }
    if !behaviour_plan
        .exits
        .is_enabled("exit.trailing.low_conf_override")
    {
        cfg.trade.trailing_start_atr_low_conf = 0.0;
        cfg.trade.trailing_distance_atr_low_conf = 0.0;
    }
    if !behaviour_plan.exits.is_enabled("exit.trailing.vol_buffer") {
        cfg.trade.enable_vol_buffered_trailing = false;
    }
    if !behaviour_plan.exits.is_enabled("exit.take_profit.partial") {
        cfg.trade.enable_partial_tp = false;
        cfg.trade.tp_partial_pct = 0.0;
        cfg.trade.tp_partial_atr_mult = 0.0;
    }
    if !behaviour_plan
        .exits
        .is_enabled("exit.smart.trend_exhaustion")
    {
        cfg.trade.smart_exit_adx_exhaustion_lt = 0.0;
        cfg.trade.smart_exit_adx_exhaustion_lt_low_conf = 0.0;
    }
    if !behaviour_plan.exits.is_enabled("exit.smart.tsme") {
        cfg.trade.tsme_min_profit_atr = DISABLED_SENTINEL;
        cfg.trade.tsme_require_adx_slope_negative = false;
    }
    if !behaviour_plan
        .exits
        .is_enabled("exit.smart.rsi_overextension")
    {
        cfg.trade.enable_rsi_overextension_exit = false;
    }
    cfg
}

fn validate_gpu_behaviour_contract(
    behaviour_plan: &ResolvedBehaviourPlan,
) -> Result<(), GpuExecutionContractError> {
    validate_canonical_order(
        "gates",
        ordered_ids(&DEFAULT_GATE_BEHAVIOURS),
        &behaviour_plan.gates,
    )?;
    validate_canonical_order(
        "signal_confidence",
        ordered_ids(&DEFAULT_SIGNAL_CONFIDENCE_BEHAVIOURS),
        &behaviour_plan.signal_confidence,
    )?;
    validate_canonical_order(
        "engine",
        ordered_ids(&DEFAULT_ENGINE_BEHAVIOURS),
        &behaviour_plan.engine,
    )?;
    validate_canonical_order(
        "entry_sizing",
        ordered_ids(&DEFAULT_ENTRY_SIZING_BEHAVIOURS),
        &behaviour_plan.entry_sizing,
    )?;
    validate_canonical_order(
        "entry_progression",
        ordered_ids(&DEFAULT_ENTRY_PROGRESSION_BEHAVIOURS),
        &behaviour_plan.entry_progression,
    )?;
    validate_canonical_order(
        "risk",
        ordered_ids(&DEFAULT_RISK_BEHAVIOURS),
        &behaviour_plan.risk,
    )?;
    let mut default_exit_ids = ordered_ids(&DEFAULT_EXIT_BEHAVIOURS);
    default_exit_ids.extend(ordered_ids(&DEFAULT_EXIT_SMART_EXTENDED_BEHAVIOURS));
    validate_canonical_order("exits", default_exit_ids, &behaviour_plan.exits)?;

    validate_supported_disable(
        "gates",
        &behaviour_plan.gates,
        &[
            "gate.ranging_vote",
            "gate.anomaly_filter",
            "gate.extension_filter",
            "gate.volume_confirmation",
            "gate.adx_rising",
            "gate.adx_floor",
            "gate.adx_floor.ave_multiplier",
            "gate.alignment.macro",
            "gate.alignment.btc",
        ],
    )?;
    validate_supported_disable(
        "signal_modes",
        &behaviour_plan.signal_modes,
        &[
            "signal.mode.standard_trend",
            "signal.mode.pullback",
            "signal.mode.slow_drift",
        ],
    )?;
    validate_supported_disable(
        "signal_confidence",
        &behaviour_plan.signal_confidence,
        &["signal.confidence.high_volume_upgrade"],
    )?;
    validate_supported_disable(
        "engine",
        &behaviour_plan.engine,
        &[
            "engine.atr_floor",
            "engine.reverse_signal",
            "engine.regime_filter",
        ],
    )?;
    validate_supported_disable(
        "entry_sizing",
        &behaviour_plan.entry_sizing,
        &[
            "entry.sizing.dynamic",
            "entry.sizing.confidence_multiplier",
            "entry.sizing.adx_multiplier",
            "entry.sizing.volatility_scalar",
            "entry.sizing.min_notional_bump",
        ],
    )?;
    validate_supported_disable(
        "entry_progression",
        &behaviour_plan.entry_progression,
        &[
            "entry.progression.pyramiding",
            "entry.progression.add_cooldown",
        ],
    )?;
    validate_supported_disable(
        "risk",
        &behaviour_plan.risk,
        &["risk.entry_cooldown", "risk.exit_cooldown", "risk.pesc"],
    )?;
    validate_supported_disable(
        "exits",
        &behaviour_plan.exits,
        &[
            "exit.stop_loss.ase",
            "exit.stop_loss.dase",
            "exit.stop_loss.slb",
            "exit.stop_loss.base",
            "exit.stop_loss.breakeven",
            "exit.trailing.low_conf_override",
            "exit.trailing.vol_buffer",
            "exit.trailing.base",
            "exit.take_profit.partial",
            "exit.take_profit.full",
            "exit.smart.trend_breakdown",
            "exit.smart.trend_exhaustion",
            "exit.smart.ema_macro_breakdown",
            "exit.smart.stagnation",
            "exit.smart.funding_headwind",
            "exit.smart.tsme",
            "exit.smart.mmde",
            "exit.smart.rsi_overextension",
        ],
    )?;

    Ok(())
}

fn ordered_ids(descriptors: &[BehaviourDescriptor]) -> Vec<String> {
    descriptors
        .iter()
        .map(|descriptor| descriptor.id.to_string())
        .collect()
}

fn validate_canonical_order(
    group: &'static str,
    expected: Vec<String>,
    plan: &BehaviourGroupPlan,
) -> Result<(), GpuExecutionContractError> {
    let got = plan.ordered_ids().map(str::to_string).collect::<Vec<_>>();
    if got != expected {
        return Err(GpuExecutionContractError::UnsupportedOrder {
            group,
            expected,
            got,
        });
    }
    Ok(())
}

fn validate_supported_disable(
    group: &'static str,
    plan: &BehaviourGroupPlan,
    supported: &[&str],
) -> Result<(), GpuExecutionContractError> {
    for item in &plan.items {
        if !item.enabled && !supported.iter().any(|id| *id == item.id) {
            return Err(GpuExecutionContractError::UnsupportedDisable {
                group,
                id: item.id.clone(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        resolve_execution_config, resolve_gpu_execution_config, GpuExecutionContractError,
    };
    use crate::config::{
        BehaviourGroupConfig, BehaviourProfileConfig, PipelineConfig, PipelineProfileConfig,
        RuntimeConfig, StrategyConfig,
    };
    use std::collections::BTreeMap;

    #[test]
    fn resolve_execution_config_lowers_modular_profile_into_effective_cfg() {
        let profile = "gpu_modular";
        let mut cfg = StrategyConfig::default();
        cfg.runtime = RuntimeConfig {
            profile: profile.to_string(),
            ..RuntimeConfig::default()
        };
        cfg.pipeline = PipelineConfig {
            default_profile: "production".to_string(),
            profiles: BTreeMap::from([(
                profile.to_string(),
                PipelineProfileConfig {
                    behaviours: BehaviourProfileConfig {
                        entry_sizing: BehaviourGroupConfig {
                            disabled: vec![
                                "entry.sizing.dynamic".to_string(),
                                "entry.sizing.confidence_multiplier".to_string(),
                                "entry.sizing.adx_multiplier".to_string(),
                                "entry.sizing.volatility_scalar".to_string(),
                                "entry.sizing.min_notional_bump".to_string(),
                            ],
                            ..BehaviourGroupConfig::default()
                        },
                        entry_progression: BehaviourGroupConfig {
                            disabled: vec![
                                "entry.progression.pyramiding".to_string(),
                                "entry.progression.add_cooldown".to_string(),
                            ],
                            ..BehaviourGroupConfig::default()
                        },
                        risk: BehaviourGroupConfig {
                            disabled: vec![
                                "risk.entry_cooldown".to_string(),
                                "risk.exit_cooldown".to_string(),
                                "risk.pesc".to_string(),
                            ],
                            ..BehaviourGroupConfig::default()
                        },
                        ..BehaviourProfileConfig::default()
                    },
                    ..PipelineProfileConfig::default()
                },
            )]),
        };

        let resolved = resolve_execution_config(&cfg, None).expect("profile must resolve");
        assert_eq!(resolved.active_profile, profile);
        assert!(!resolved.effective_cfg.trade.enable_dynamic_sizing);
        assert_eq!(resolved.effective_cfg.trade.confidence_mult_high, 1.0);
        assert_eq!(resolved.effective_cfg.trade.confidence_mult_medium, 1.0);
        assert_eq!(resolved.effective_cfg.trade.confidence_mult_low, 1.0);
        assert_eq!(resolved.effective_cfg.trade.adx_sizing_min_mult, 1.0);
        assert_eq!(resolved.effective_cfg.trade.adx_sizing_full_adx, 0.0);
        assert_eq!(resolved.effective_cfg.trade.vol_baseline_pct, 0.0);
        assert_eq!(resolved.effective_cfg.trade.vol_scalar_min, 1.0);
        assert_eq!(resolved.effective_cfg.trade.vol_scalar_max, 1.0);
        assert!(!resolved.effective_cfg.trade.bump_to_min_notional);
        assert!(!resolved.effective_cfg.trade.enable_pyramiding);
        assert_eq!(resolved.effective_cfg.trade.add_cooldown_minutes, 0);
        assert_eq!(resolved.effective_cfg.trade.entry_cooldown_s, 0);
        assert_eq!(resolved.effective_cfg.trade.exit_cooldown_s, 0);
        assert_eq!(resolved.effective_cfg.trade.reentry_cooldown_minutes, 0);
        assert_eq!(resolved.effective_cfg.trade.reentry_cooldown_min_mins, 0);
        assert_eq!(resolved.effective_cfg.trade.reentry_cooldown_max_mins, 0);
    }

    #[test]
    fn resolve_execution_config_materialises_gpu_parity_profile_scalars() {
        let mut cfg = StrategyConfig::default();
        cfg.runtime.profile = "parity_exit_isolation".to_string();

        let resolved = resolve_execution_config(&cfg, None).expect("profile must resolve");
        assert!(!resolved.effective_cfg.trade.enable_breakeven_stop);
        assert!(!resolved.effective_cfg.trade.enable_partial_tp);
        assert!(!resolved.effective_cfg.trade.enable_vol_buffered_trailing);
        assert_eq!(
            resolved.effective_cfg.trade.trailing_start_atr_low_conf,
            0.0
        );
        assert_eq!(
            resolved.effective_cfg.trade.trailing_distance_atr_low_conf,
            0.0
        );
        assert_eq!(
            resolved.effective_cfg.trade.smart_exit_adx_exhaustion_lt,
            0.0
        );
        assert_eq!(
            resolved
                .effective_cfg
                .trade
                .smart_exit_adx_exhaustion_lt_low_conf,
            0.0
        );
        assert!(!resolved.effective_cfg.trade.enable_rsi_overextension_exit);
        assert_eq!(resolved.active_profile, "parity_exit_isolation");
        assert!(!resolved
            .behaviour_plan
            .exits
            .is_enabled("exit.take_profit.partial"));
        assert!(!resolved.behaviour_plan.exits.is_enabled("exit.smart.tsme"));
    }

    #[test]
    fn resolve_execution_config_materialises_gpu_supported_gate_and_mode_scalars() {
        let profile = "gpu_gate_debug";
        let mut cfg = StrategyConfig::default();
        cfg.runtime.profile = profile.to_string();
        cfg.pipeline.profiles.insert(
            profile.to_string(),
            PipelineProfileConfig {
                behaviours: BehaviourProfileConfig {
                    gates: BehaviourGroupConfig {
                        disabled: vec![
                            "gate.anomaly_filter".to_string(),
                            "gate.alignment.btc".to_string(),
                        ],
                        ..BehaviourGroupConfig::default()
                    },
                    signal_modes: BehaviourGroupConfig {
                        disabled: vec!["signal.mode.pullback".to_string()],
                        ..BehaviourGroupConfig::default()
                    },
                    signal_confidence: BehaviourGroupConfig {
                        disabled: vec!["signal.confidence.high_volume_upgrade".to_string()],
                        ..BehaviourGroupConfig::default()
                    },
                    engine: BehaviourGroupConfig {
                        disabled: vec!["engine.reverse_signal".to_string()],
                        ..BehaviourGroupConfig::default()
                    },
                    ..BehaviourProfileConfig::default()
                },
                ..PipelineProfileConfig::default()
            },
        );

        let resolved = resolve_execution_config(&cfg, None).expect("profile must resolve");
        assert!(!resolved.effective_cfg.filters.enable_anomaly_filter);
        assert!(!resolved.effective_cfg.filters.require_btc_alignment);
        assert!(
            !resolved
                .effective_cfg
                .thresholds
                .entry
                .enable_pullback_entries
        );
        assert!(
            resolved
                .effective_cfg
                .thresholds
                .entry
                .high_conf_volume_mult
                > 1.0e9
        );
        assert!(!resolved.effective_cfg.trade.reverse_entry_signal);
        assert!(!resolved.effective_cfg.market_regime.enable_auto_reverse);
    }

    #[test]
    fn resolve_gpu_execution_config_accepts_parity_exit_isolation() {
        let mut cfg = StrategyConfig::default();
        cfg.runtime.profile = "parity_exit_isolation".to_string();

        let resolved = resolve_gpu_execution_config(&cfg, None).expect("gpu profile must resolve");
        assert_eq!(resolved.active_profile, "parity_exit_isolation");
        assert!(!resolved.effective_cfg.trade.enable_breakeven_stop);
        assert_eq!(
            resolved.effective_cfg.trade.smart_exit_adx_exhaustion_lt,
            0.0
        );
    }

    #[test]
    fn resolve_gpu_execution_config_rejects_non_canonical_exit_order() {
        let profile = "gpu_reorder";
        let mut cfg = StrategyConfig::default();
        cfg.runtime.profile = profile.to_string();
        cfg.pipeline.profiles.insert(
            profile.to_string(),
            PipelineProfileConfig {
                behaviours: BehaviourProfileConfig {
                    exits: BehaviourGroupConfig {
                        order: vec![
                            "exit.take_profit.full".to_string(),
                            "exit.stop_loss.base".to_string(),
                        ],
                        ..BehaviourGroupConfig::default()
                    },
                    ..BehaviourProfileConfig::default()
                },
                ..PipelineProfileConfig::default()
            },
        );

        let err = resolve_gpu_execution_config(&cfg, None).unwrap_err();
        assert!(matches!(
            err,
            GpuExecutionContractError::UnsupportedOrder { group: "exits", .. }
        ));
    }

    #[test]
    fn resolve_gpu_execution_config_accepts_standard_mode_disable() {
        let profile = "gpu_no_standard";
        let mut cfg = StrategyConfig::default();
        cfg.runtime.profile = profile.to_string();
        cfg.pipeline.profiles.insert(
            profile.to_string(),
            PipelineProfileConfig {
                behaviours: BehaviourProfileConfig {
                    signal_modes: BehaviourGroupConfig {
                        disabled: vec!["signal.mode.standard_trend".to_string()],
                        ..BehaviourGroupConfig::default()
                    },
                    ..BehaviourProfileConfig::default()
                },
                ..PipelineProfileConfig::default()
            },
        );

        let resolved =
            resolve_gpu_execution_config(&cfg, None).expect("signal mode disable should resolve");
        assert_eq!(resolved.active_profile, profile);
    }

    #[test]
    fn resolve_gpu_execution_config_accepts_signal_mode_reorder() {
        let profile = "gpu_signal_reorder";
        let mut cfg = StrategyConfig::default();
        cfg.runtime.profile = profile.to_string();
        cfg.pipeline.profiles.insert(
            profile.to_string(),
            PipelineProfileConfig {
                behaviours: BehaviourProfileConfig {
                    signal_modes: BehaviourGroupConfig {
                        order: vec![
                            "signal.mode.slow_drift".to_string(),
                            "signal.mode.standard_trend".to_string(),
                            "signal.mode.pullback".to_string(),
                        ],
                        ..BehaviourGroupConfig::default()
                    },
                    ..BehaviourProfileConfig::default()
                },
                ..PipelineProfileConfig::default()
            },
        );

        let resolved =
            resolve_gpu_execution_config(&cfg, None).expect("signal mode reorder should resolve");
        assert_eq!(resolved.active_profile, profile);
    }
}
