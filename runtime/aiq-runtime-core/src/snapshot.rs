use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use thiserror::Error;

pub const SNAPSHOT_V1: u32 = 1;
pub const SNAPSHOT_V2: u32 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SnapshotRuntimeState {
    #[serde(default)]
    pub entry_attempt_ms_by_symbol: BTreeMap<String, i64>,
    #[serde(default)]
    pub exit_attempt_ms_by_symbol: BTreeMap<String, i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotPosition {
    pub symbol: String,
    pub side: String,
    pub size: f64,
    pub entry_price: f64,
    pub entry_atr: f64,
    #[serde(default)]
    pub trailing_sl: Option<f64>,
    pub confidence: String,
    pub leverage: f64,
    pub margin_used: f64,
    #[serde(default)]
    pub adds_count: u32,
    #[serde(default)]
    pub tp1_taken: bool,
    #[serde(default)]
    pub open_time_ms: i64,
    #[serde(default)]
    pub last_add_time_ms: i64,
    #[serde(default)]
    pub entry_adx_threshold: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotFile {
    pub version: u32,
    pub source: String,
    pub exported_at_ms: i64,
    pub balance: f64,
    #[serde(default)]
    pub positions: Vec<SnapshotPosition>,
    #[serde(default)]
    pub runtime: Option<SnapshotRuntimeState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SnapshotSummary {
    pub version: u32,
    pub source: String,
    pub position_count: usize,
    pub runtime_entry_markers: usize,
    pub runtime_exit_markers: usize,
}

#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("failed to read snapshot `{path}`: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse snapshot JSON: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("unsupported snapshot version `{0}`")]
    UnsupportedVersion(u32),
    #[error("snapshot balance must be >= 0")]
    NegativeBalance,
    #[error("position `{symbol}` has invalid side `{side}`")]
    InvalidSide { symbol: String, side: String },
    #[error("position `{symbol}` must have positive size")]
    InvalidSize { symbol: String },
    #[error("position `{symbol}` must have positive leverage")]
    InvalidLeverage { symbol: String },
    #[error("position `{symbol}` has invalid confidence `{confidence}`")]
    InvalidConfidence { symbol: String, confidence: String },
    #[error("runtime markers require snapshot version 2")]
    RuntimeRequiresV2,
}

impl SnapshotFile {
    pub fn validate(&self) -> Result<(), SnapshotError> {
        if self.version != SNAPSHOT_V1 && self.version != SNAPSHOT_V2 {
            return Err(SnapshotError::UnsupportedVersion(self.version));
        }
        if self.balance < 0.0 {
            return Err(SnapshotError::NegativeBalance);
        }
        if self.runtime.is_some() && self.version < SNAPSHOT_V2 {
            return Err(SnapshotError::RuntimeRequiresV2);
        }

        for position in &self.positions {
            let side = position.side.trim().to_ascii_lowercase();
            if side != "long" && side != "short" {
                return Err(SnapshotError::InvalidSide {
                    symbol: position.symbol.clone(),
                    side: position.side.clone(),
                });
            }
            if position.size <= 0.0 {
                return Err(SnapshotError::InvalidSize {
                    symbol: position.symbol.clone(),
                });
            }
            if position.leverage <= 0.0 {
                return Err(SnapshotError::InvalidLeverage {
                    symbol: position.symbol.clone(),
                });
            }
            let confidence = position.confidence.trim().to_ascii_lowercase();
            let valid_conf =
                confidence.is_empty() || matches!(confidence.as_str(), "low" | "medium" | "high");
            if !valid_conf {
                return Err(SnapshotError::InvalidConfidence {
                    symbol: position.symbol.clone(),
                    confidence: position.confidence.clone(),
                });
            }
        }

        Ok(())
    }

    pub fn summary(&self) -> SnapshotSummary {
        let runtime = self.runtime.as_ref();
        SnapshotSummary {
            version: self.version,
            source: self.source.clone(),
            position_count: self.positions.len(),
            runtime_entry_markers: runtime
                .map(|runtime| runtime.entry_attempt_ms_by_symbol.len())
                .unwrap_or(0),
            runtime_exit_markers: runtime
                .map(|runtime| runtime.exit_attempt_ms_by_symbol.len())
                .unwrap_or(0),
        }
    }
}

pub fn load_snapshot(path: &Path) -> Result<SnapshotFile, SnapshotError> {
    let raw = fs::read_to_string(path).map_err(|source| SnapshotError::Read {
        path: path.display().to_string(),
        source,
    })?;
    let snapshot: SnapshotFile = serde_json::from_str(&raw)?;
    snapshot.validate()?;
    Ok(snapshot)
}

pub fn snapshot_to_pretty_json(snapshot: &SnapshotFile) -> Result<String, SnapshotError> {
    serde_json::to_string_pretty(snapshot).map_err(SnapshotError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot(version: u32) -> SnapshotFile {
        SnapshotFile {
            version,
            source: "paper".to_string(),
            exported_at_ms: 1_770_562_800_000,
            balance: 1000.0,
            positions: vec![SnapshotPosition {
                symbol: "BTC".to_string(),
                side: "long".to_string(),
                size: 0.003,
                entry_price: 71_000.0,
                entry_atr: 1_200.0,
                trailing_sl: Some(69_500.0),
                confidence: "medium".to_string(),
                leverage: 5.0,
                margin_used: 42.6,
                adds_count: 0,
                tp1_taken: false,
                open_time_ms: 1_770_500_000_000,
                last_add_time_ms: 0,
                entry_adx_threshold: 22.0,
            }],
            runtime: if version >= SNAPSHOT_V2 {
                Some(SnapshotRuntimeState {
                    entry_attempt_ms_by_symbol: BTreeMap::from([("BTC".to_string(), 123)]),
                    exit_attempt_ms_by_symbol: BTreeMap::from([("ETH".to_string(), 456)]),
                })
            } else {
                None
            },
        }
    }

    #[test]
    fn validates_v2_runtime_snapshot() {
        let snapshot = sample_snapshot(SNAPSHOT_V2);
        snapshot.validate().unwrap();
        let summary = snapshot.summary();
        assert_eq!(summary.position_count, 1);
        assert_eq!(summary.runtime_entry_markers, 1);
        assert_eq!(summary.runtime_exit_markers, 1);
    }

    #[test]
    fn rejects_runtime_on_v1_snapshot() {
        let mut snapshot = sample_snapshot(SNAPSHOT_V1);
        snapshot.runtime = Some(SnapshotRuntimeState::default());
        let err = snapshot.validate().unwrap_err();
        assert!(matches!(err, SnapshotError::RuntimeRequiresV2));
    }

    #[test]
    fn rejects_invalid_confidence_label() {
        let mut snapshot = sample_snapshot(SNAPSHOT_V2);
        snapshot.positions[0].confidence = "urgent".to_string();
        let err = snapshot.validate().unwrap_err();
        assert!(matches!(err, SnapshotError::InvalidConfidence { .. }));
    }
}
