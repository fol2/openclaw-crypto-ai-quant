use crate::config::{BehaviourGroupConfig, BehaviourProfileConfig};
use bt_signals::behaviour::{
    resolve_group_plan, BehaviourDescriptor, BehaviourGroupPlan, BehaviourResolveError,
    DEFAULT_GATE_BEHAVIOURS, DEFAULT_SIGNAL_CONFIDENCE_BEHAVIOURS, DEFAULT_SIGNAL_MODE_BEHAVIOURS,
};
use serde::{Deserialize, Serialize};

pub const DEFAULT_ENGINE_BEHAVIOURS: [BehaviourDescriptor; 4] = [
    BehaviourDescriptor {
        id: "engine.atr_floor",
        description: "ATR floor normalisation before downstream entry logic.",
    },
    BehaviourDescriptor {
        id: "engine.reverse_signal",
        description: "Manual or automatic signal reversal before regime filtering.",
    },
    BehaviourDescriptor {
        id: "engine.regime_filter",
        description: "Market regime filter after reverse processing.",
    },
    BehaviourDescriptor {
        id: "engine.entry_budget",
        description: "Per-cycle entry budget limiting and candidate truncation.",
    },
];

pub const DEFAULT_ENTRY_SIZING_BEHAVIOURS: [BehaviourDescriptor; 5] = [
    BehaviourDescriptor {
        id: "entry.sizing.dynamic",
        description: "Dynamic sizing switch between static and adaptive sizing.",
    },
    BehaviourDescriptor {
        id: "entry.sizing.confidence_multiplier",
        description: "Confidence-tier sizing multiplier.",
    },
    BehaviourDescriptor {
        id: "entry.sizing.adx_multiplier",
        description: "ADX-based sizing multiplier.",
    },
    BehaviourDescriptor {
        id: "entry.sizing.volatility_scalar",
        description: "Volatility scalar adjustment on size.",
    },
    BehaviourDescriptor {
        id: "entry.sizing.min_notional_bump",
        description: "Minimum notional bump-to-floor behaviour.",
    },
];

pub const DEFAULT_ENTRY_PROGRESSION_BEHAVIOURS: [BehaviourDescriptor; 3] = [
    BehaviourDescriptor {
        id: "entry.progression.pyramiding",
        description: "Same-side pyramiding progression.",
    },
    BehaviourDescriptor {
        id: "entry.progression.add_cooldown",
        description: "Add cooldown gate for pyramids.",
    },
    BehaviourDescriptor {
        id: "entry.progression.tp1_block_add",
        description: "Block adds after TP1 has been taken.",
    },
];

pub const DEFAULT_RISK_BEHAVIOURS: [BehaviourDescriptor; 4] = [
    BehaviourDescriptor {
        id: "risk.exposure_guard",
        description: "Exposure and margin headroom guard.",
    },
    BehaviourDescriptor {
        id: "risk.entry_cooldown",
        description: "Entry cooldown enforcement.",
    },
    BehaviourDescriptor {
        id: "risk.exit_cooldown",
        description: "Exit cooldown enforcement.",
    },
    BehaviourDescriptor {
        id: "risk.pesc",
        description: "Post-exit same-direction cooldown enforcement.",
    },
];

pub const DEFAULT_EXIT_BEHAVIOURS: [BehaviourDescriptor; 13] = [
    BehaviourDescriptor {
        id: "exit.stop_loss.ase",
        description: "ADX slope tightening when underwater.",
    },
    BehaviourDescriptor {
        id: "exit.stop_loss.dase",
        description: "ADX-strength widening when profitable.",
    },
    BehaviourDescriptor {
        id: "exit.stop_loss.slb",
        description: "ADX saturation widening buffer.",
    },
    BehaviourDescriptor {
        id: "exit.stop_loss.base",
        description: "Base ATR stop-loss.",
    },
    BehaviourDescriptor {
        id: "exit.stop_loss.breakeven",
        description: "Breakeven stop uplift.",
    },
    BehaviourDescriptor {
        id: "exit.trailing.low_conf_override",
        description: "Low-confidence trailing overrides.",
    },
    BehaviourDescriptor {
        id: "exit.trailing.vol_buffer",
        description: "Volatility-buffered trailing adjustment.",
    },
    BehaviourDescriptor {
        id: "exit.trailing.base",
        description: "Base trailing-stop activation and distance.",
    },
    BehaviourDescriptor {
        id: "exit.take_profit.partial",
        description: "Partial take-profit exit.",
    },
    BehaviourDescriptor {
        id: "exit.take_profit.full",
        description: "Full take-profit exit.",
    },
    BehaviourDescriptor {
        id: "exit.smart.trend_breakdown",
        description: "Smart exit on EMA trend breakdown.",
    },
    BehaviourDescriptor {
        id: "exit.smart.trend_exhaustion",
        description: "Smart exit on ADX trend exhaustion.",
    },
    BehaviourDescriptor {
        id: "exit.smart.ema_macro_breakdown",
        description: "Smart exit on EMA macro breakdown.",
    },
];

pub const DEFAULT_EXIT_SMART_EXTENDED_BEHAVIOURS: [BehaviourDescriptor; 5] = [
    BehaviourDescriptor {
        id: "exit.smart.stagnation",
        description: "Smart exit on low-volatility stagnation.",
    },
    BehaviourDescriptor {
        id: "exit.smart.funding_headwind",
        description: "Smart exit on funding headwind.",
    },
    BehaviourDescriptor {
        id: "exit.smart.tsme",
        description: "Trend Saturation Momentum Exhaustion.",
    },
    BehaviourDescriptor {
        id: "exit.smart.mmde",
        description: "MACD Persistent Divergence Exit.",
    },
    BehaviourDescriptor {
        id: "exit.smart.rsi_overextension",
        description: "RSI overextension smart exit.",
    },
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ResolvedBehaviourPlan {
    pub gates: BehaviourGroupPlan,
    pub signal_modes: BehaviourGroupPlan,
    pub signal_confidence: BehaviourGroupPlan,
    pub exits: BehaviourGroupPlan,
    pub engine: BehaviourGroupPlan,
    pub entry_sizing: BehaviourGroupPlan,
    pub entry_progression: BehaviourGroupPlan,
    pub risk: BehaviourGroupPlan,
}

impl ResolvedBehaviourPlan {
    pub fn production() -> Self {
        resolve_behaviour_plan("production", &BehaviourProfileConfig::default())
            .expect("default behaviour plan must resolve")
    }
}

fn ordered_group(descriptors: &[BehaviourDescriptor]) -> BehaviourGroupConfig {
    BehaviourGroupConfig {
        order: descriptors
            .iter()
            .map(|descriptor| descriptor.id.to_string())
            .collect(),
        enabled: Vec::new(),
        disabled: Vec::new(),
    }
}

fn ordered_exit_group() -> BehaviourGroupConfig {
    BehaviourGroupConfig {
        order: DEFAULT_EXIT_BEHAVIOURS
            .iter()
            .chain(DEFAULT_EXIT_SMART_EXTENDED_BEHAVIOURS.iter())
            .map(|descriptor| descriptor.id.to_string())
            .collect(),
        enabled: Vec::new(),
        disabled: Vec::new(),
    }
}

fn parity_baseline_profile() -> BehaviourProfileConfig {
    BehaviourProfileConfig {
        gates: ordered_group(&DEFAULT_GATE_BEHAVIOURS),
        signal_modes: ordered_group(&DEFAULT_SIGNAL_MODE_BEHAVIOURS),
        signal_confidence: ordered_group(&DEFAULT_SIGNAL_CONFIDENCE_BEHAVIOURS),
        exits: ordered_exit_group(),
        engine: ordered_group(&DEFAULT_ENGINE_BEHAVIOURS),
        entry_sizing: ordered_group(&DEFAULT_ENTRY_SIZING_BEHAVIOURS),
        entry_progression: ordered_group(&DEFAULT_ENTRY_PROGRESSION_BEHAVIOURS),
        risk: ordered_group(&DEFAULT_RISK_BEHAVIOURS),
    }
}

pub fn builtin_behaviour_profile(profile: &str) -> Option<BehaviourProfileConfig> {
    match profile.trim().to_ascii_lowercase().as_str() {
        "parity_baseline" => Some(parity_baseline_profile()),
        "parity_exit_isolation" => {
            let mut profile = parity_baseline_profile();
            profile.exits.disabled = vec![
                "exit.stop_loss.breakeven".to_string(),
                "exit.trailing.low_conf_override".to_string(),
                "exit.trailing.vol_buffer".to_string(),
                "exit.take_profit.partial".to_string(),
                "exit.smart.trend_breakdown".to_string(),
                "exit.smart.trend_exhaustion".to_string(),
                "exit.smart.ema_macro_breakdown".to_string(),
                "exit.smart.stagnation".to_string(),
                "exit.smart.funding_headwind".to_string(),
                "exit.smart.tsme".to_string(),
                "exit.smart.mmde".to_string(),
                "exit.smart.rsi_overextension".to_string(),
            ];
            Some(profile)
        }
        _ => None,
    }
}

pub fn resolve_behaviour_plan(
    profile: &str,
    config: &BehaviourProfileConfig,
) -> Result<ResolvedBehaviourPlan, BehaviourResolveError> {
    Ok(ResolvedBehaviourPlan {
        gates: resolve_group_plan(
            &format!("{profile}.behaviours.gates"),
            &DEFAULT_GATE_BEHAVIOURS,
            &config.gates.order,
            &config.gates.enabled,
            &config.gates.disabled,
        )?,
        signal_modes: resolve_group_plan(
            &format!("{profile}.behaviours.signal_modes"),
            &DEFAULT_SIGNAL_MODE_BEHAVIOURS,
            &config.signal_modes.order,
            &config.signal_modes.enabled,
            &config.signal_modes.disabled,
        )?,
        signal_confidence: resolve_group_plan(
            &format!("{profile}.behaviours.signal_confidence"),
            &DEFAULT_SIGNAL_CONFIDENCE_BEHAVIOURS,
            &config.signal_confidence.order,
            &config.signal_confidence.enabled,
            &config.signal_confidence.disabled,
        )?,
        exits: resolve_group_plan(
            &format!("{profile}.behaviours.exits"),
            &[
                DEFAULT_EXIT_BEHAVIOURS.as_slice(),
                DEFAULT_EXIT_SMART_EXTENDED_BEHAVIOURS.as_slice(),
            ]
            .concat(),
            &config.exits.order,
            &config.exits.enabled,
            &config.exits.disabled,
        )?,
        engine: resolve_group_plan(
            &format!("{profile}.behaviours.engine"),
            &DEFAULT_ENGINE_BEHAVIOURS,
            &config.engine.order,
            &config.engine.enabled,
            &config.engine.disabled,
        )?,
        entry_sizing: resolve_group_plan(
            &format!("{profile}.behaviours.entry_sizing"),
            &DEFAULT_ENTRY_SIZING_BEHAVIOURS,
            &config.entry_sizing.order,
            &config.entry_sizing.enabled,
            &config.entry_sizing.disabled,
        )?,
        entry_progression: resolve_group_plan(
            &format!("{profile}.behaviours.entry_progression"),
            &DEFAULT_ENTRY_PROGRESSION_BEHAVIOURS,
            &config.entry_progression.order,
            &config.entry_progression.enabled,
            &config.entry_progression.disabled,
        )?,
        risk: resolve_group_plan(
            &format!("{profile}.behaviours.risk"),
            &DEFAULT_RISK_BEHAVIOURS,
            &config.risk.order,
            &config.risk.enabled,
            &config.risk.disabled,
        )?,
    })
}

impl BehaviourGroupConfig {
    pub fn is_empty(&self) -> bool {
        self.order.is_empty() && self.enabled.is_empty() && self.disabled.is_empty()
    }
}
