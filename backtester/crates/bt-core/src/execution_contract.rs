use crate::behaviour::ResolvedBehaviourPlan;
use crate::config::StrategyConfig;

const DISABLED_SENTINEL: f64 = 1.0e12;

#[derive(Debug, Clone)]
pub struct ResolvedExecutionConfig {
    pub active_profile: String,
    pub effective_cfg: StrategyConfig,
    pub behaviour_plan: ResolvedBehaviourPlan,
}

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

#[cfg(test)]
mod tests {
    use super::resolve_execution_config;
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
        assert_eq!(resolved.effective_cfg.trade.trailing_start_atr_low_conf, 0.0);
        assert_eq!(resolved.effective_cfg.trade.trailing_distance_atr_low_conf, 0.0);
        assert_eq!(resolved.effective_cfg.trade.smart_exit_adx_exhaustion_lt, 0.0);
        assert_eq!(
            resolved.effective_cfg.trade.smart_exit_adx_exhaustion_lt_low_conf,
            0.0
        );
        assert!(!resolved.effective_cfg.trade.enable_rsi_overextension_exit);
        assert_eq!(resolved.active_profile, "parity_exit_isolation");
        assert!(!resolved.behaviour_plan.exits.is_enabled("exit.take_profit.partial"));
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
        assert!(!resolved.effective_cfg.thresholds.entry.enable_pullback_entries);
        assert!(resolved.effective_cfg.thresholds.entry.high_conf_volume_mult > 1.0e9);
        assert!(!resolved.effective_cfg.trade.reverse_entry_signal);
        assert!(!resolved.effective_cfg.market_regime.enable_auto_reverse);
    }
}
