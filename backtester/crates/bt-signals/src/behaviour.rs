use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BehaviourDescriptor {
    pub id: &'static str,
    pub description: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BehaviourPlanItem {
    pub id: String,
    pub enabled: bool,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BehaviourGroupPlan {
    pub items: Vec<BehaviourPlanItem>,
}

impl BehaviourGroupPlan {
    pub fn ordered_ids(&self) -> impl Iterator<Item = &str> {
        self.items.iter().map(|item| item.id.as_str())
    }

    pub fn is_enabled(&self, id: &str) -> bool {
        self.items
            .iter()
            .find(|item| item.id == id)
            .map(|item| item.enabled)
            .unwrap_or(false)
    }

    pub fn item(&self, id: &str) -> Option<&BehaviourPlanItem> {
        self.items.iter().find(|item| item.id == id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BehaviourTrace {
    pub group: String,
    pub id: String,
    pub enabled: bool,
    pub status: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BehaviourResolveError {
    UnknownId { group: String, id: String },
    DuplicateId { group: String, id: String },
}

impl std::fmt::Display for BehaviourResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownId { group, id } => {
                write!(f, "unknown behaviour `{id}` in group `{group}`")
            }
            Self::DuplicateId { group, id } => {
                write!(f, "duplicate behaviour `{id}` in group `{group}`")
            }
        }
    }
}

impl std::error::Error for BehaviourResolveError {}

pub fn resolve_group_plan(
    group: &str,
    descriptors: &[BehaviourDescriptor],
    order: &[String],
    enabled: &[String],
    disabled: &[String],
) -> Result<BehaviourGroupPlan, BehaviourResolveError> {
    let known_ids = descriptors
        .iter()
        .map(|item| item.id)
        .collect::<BTreeSet<_>>();

    let mut seen_order = BTreeSet::new();
    for id in order {
        let id = id.trim();
        if id.is_empty() {
            continue;
        }
        if !known_ids.contains(id) {
            return Err(BehaviourResolveError::UnknownId {
                group: group.to_string(),
                id: id.to_string(),
            });
        }
        if !seen_order.insert(id.to_string()) {
            return Err(BehaviourResolveError::DuplicateId {
                group: group.to_string(),
                id: id.to_string(),
            });
        }
    }

    let enabled = parse_id_set(group, enabled, &known_ids)?;
    let disabled = parse_id_set(group, disabled, &known_ids)?;

    let mut resolved_order = order
        .iter()
        .map(|id| id.trim())
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    for descriptor in descriptors {
        if !resolved_order.iter().any(|id| id == descriptor.id) {
            resolved_order.push(descriptor.id.to_string());
        }
    }

    let items = resolved_order
        .into_iter()
        .map(|id| {
            let descriptor = descriptors
                .iter()
                .find(|descriptor| descriptor.id == id)
                .expect("resolved order only contains known descriptors");
            let enabled_flag = if disabled.contains(id.as_str()) {
                false
            } else if enabled.is_empty() {
                true
            } else {
                enabled.contains(id.as_str())
            };
            BehaviourPlanItem {
                id: id.clone(),
                enabled: enabled_flag,
                description: descriptor.description.to_string(),
            }
        })
        .collect();

    Ok(BehaviourGroupPlan { items })
}

fn parse_id_set<'a>(
    group: &str,
    values: &'a [String],
    known_ids: &BTreeSet<&'a str>,
) -> Result<BTreeSet<&'a str>, BehaviourResolveError> {
    let mut resolved = BTreeSet::new();
    for id in values {
        let id = id.trim();
        if id.is_empty() {
            continue;
        }
        if !known_ids.contains(id) {
            return Err(BehaviourResolveError::UnknownId {
                group: group.to_string(),
                id: id.to_string(),
            });
        }
        resolved.insert(id);
    }
    Ok(resolved)
}

pub const DEFAULT_GATE_BEHAVIOURS: [BehaviourDescriptor; 11] = [
    BehaviourDescriptor {
        id: "gate.ranging_vote",
        description: "Vote-based ranging detection.",
    },
    BehaviourDescriptor {
        id: "gate.anomaly_filter",
        description: "Anomaly filter on price change and EMA-fast deviation.",
    },
    BehaviourDescriptor {
        id: "gate.extension_filter",
        description: "Entry extension guard against EMA-fast distance.",
    },
    BehaviourDescriptor {
        id: "gate.volume_confirmation",
        description: "Volume confirmation gate.",
    },
    BehaviourDescriptor {
        id: "gate.adx_rising",
        description: "ADX-rising confirmation gate.",
    },
    BehaviourDescriptor {
        id: "gate.adx_floor",
        description: "Base ADX minimum threshold gate.",
    },
    BehaviourDescriptor {
        id: "gate.adx_floor.tmc_cap",
        description: "TMC cap on effective ADX threshold.",
    },
    BehaviourDescriptor {
        id: "gate.adx_floor.ave_multiplier",
        description: "AVE multiplier on effective ADX threshold.",
    },
    BehaviourDescriptor {
        id: "gate.alignment.macro",
        description: "Macro EMA alignment check.",
    },
    BehaviourDescriptor {
        id: "gate.alignment.btc",
        description: "BTC alignment check.",
    },
    BehaviourDescriptor {
        id: "gate.override.slow_drift_unrange",
        description: "Slow-drift override that clears ranging.",
    },
];

pub const DEFAULT_SIGNAL_MODE_BEHAVIOURS: [BehaviourDescriptor; 3] = [
    BehaviourDescriptor {
        id: "signal.mode.standard_trend",
        description: "Standard trend entry mode.",
    },
    BehaviourDescriptor {
        id: "signal.mode.pullback",
        description: "Pullback continuation entry mode.",
    },
    BehaviourDescriptor {
        id: "signal.mode.slow_drift",
        description: "Slow-drift continuation entry mode.",
    },
];

pub const DEFAULT_SIGNAL_CONFIDENCE_BEHAVIOURS: [BehaviourDescriptor; 1] = [BehaviourDescriptor {
    id: "signal.confidence.high_volume_upgrade",
    description: "High-volume confidence upgrade inside standard trend mode.",
}];
