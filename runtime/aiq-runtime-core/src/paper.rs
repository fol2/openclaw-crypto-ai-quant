use crate::snapshot::{SnapshotFile, SnapshotRuntimeState};
use serde::Serialize;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperPositionState {
    pub symbol: String,
    pub side: String,
    pub size: f64,
    pub entry_price: f64,
    pub entry_atr: f64,
    pub trailing_sl: Option<f64>,
    pub confidence: String,
    pub leverage: f64,
    pub margin_used: f64,
    pub adds_count: u32,
    pub tp1_taken: bool,
    pub open_time_ms: i64,
    pub last_add_time_ms: i64,
    pub entry_adx_threshold: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperBootstrapState {
    pub snapshot_source: String,
    pub snapshot_exported_at_ms: i64,
    pub balance: f64,
    pub positions: BTreeMap<String, PaperPositionState>,
    pub runtime: SnapshotRuntimeState,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperBootstrapReport {
    pub ok: bool,
    pub snapshot_source: String,
    pub snapshot_exported_at_ms: i64,
    pub balance: f64,
    pub position_count: usize,
    pub runtime_entry_markers: usize,
    pub runtime_exit_markers: usize,
    pub restored_symbols: Vec<String>,
}

#[derive(Debug, Error)]
pub enum PaperBootstrapError {
    #[error("invalid snapshot: {0}")]
    Snapshot(String),
    #[error("duplicate paper position symbol `{0}` in snapshot")]
    DuplicateSymbol(String),
}

pub fn restore_paper_state(
    snapshot: &SnapshotFile,
) -> Result<(PaperBootstrapState, PaperBootstrapReport), PaperBootstrapError> {
    snapshot
        .validate()
        .map_err(|err| PaperBootstrapError::Snapshot(err.to_string()))?;

    let runtime = snapshot.runtime.clone().unwrap_or_default();
    let mut positions = BTreeMap::new();

    for position in &snapshot.positions {
        let symbol = position.symbol.trim().to_ascii_uppercase();
        if positions.contains_key(&symbol) {
            return Err(PaperBootstrapError::DuplicateSymbol(symbol));
        }

        positions.insert(
            symbol.clone(),
            PaperPositionState {
                symbol,
                side: position.side.clone(),
                size: position.size,
                entry_price: position.entry_price,
                entry_atr: position.entry_atr,
                trailing_sl: position.trailing_sl,
                confidence: position.confidence.clone(),
                leverage: position.leverage,
                margin_used: position.margin_used,
                adds_count: position.adds_count,
                tp1_taken: position.tp1_taken,
                open_time_ms: position.open_time_ms,
                last_add_time_ms: position.last_add_time_ms,
                entry_adx_threshold: position.entry_adx_threshold,
            },
        );
    }

    let restored_symbols = positions.keys().cloned().collect::<Vec<_>>();
    let state = PaperBootstrapState {
        snapshot_source: snapshot.source.clone(),
        snapshot_exported_at_ms: snapshot.exported_at_ms,
        balance: snapshot.balance,
        positions,
        runtime: runtime.clone(),
    };
    let report = PaperBootstrapReport {
        ok: true,
        snapshot_source: snapshot.source.clone(),
        snapshot_exported_at_ms: snapshot.exported_at_ms,
        balance: snapshot.balance,
        position_count: state.positions.len(),
        runtime_entry_markers: runtime.entry_attempt_ms_by_symbol.len(),
        runtime_exit_markers: runtime.exit_attempt_ms_by_symbol.len(),
        restored_symbols,
    };

    Ok((state, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{SnapshotFile, SnapshotPosition, SnapshotRuntimeState, SNAPSHOT_V2};
    use std::collections::BTreeMap;

    fn sample_snapshot() -> SnapshotFile {
        SnapshotFile {
            version: SNAPSHOT_V2,
            source: "paper".to_string(),
            exported_at_ms: 1_772_676_900_000,
            balance: 1000.0,
            positions: vec![SnapshotPosition {
                symbol: "BTC".to_string(),
                side: "long".to_string(),
                size: 2.0,
                entry_price: 100.0,
                entry_atr: 5.0,
                trailing_sl: Some(95.0),
                confidence: "high".to_string(),
                leverage: 4.0,
                margin_used: 50.0,
                adds_count: 1,
                tp1_taken: false,
                open_time_ms: 1_772_676_500_000,
                last_add_time_ms: 1_772_676_600_000,
                entry_adx_threshold: 23.5,
            }],
            runtime: Some(SnapshotRuntimeState {
                entry_attempt_ms_by_symbol: BTreeMap::from([("BTC".to_string(), 1_772_676_500_000)]),
                exit_attempt_ms_by_symbol: BTreeMap::from([("BTC".to_string(), 1_772_676_550_000)]),
            }),
        }
    }

    #[test]
    fn restores_paper_state_from_snapshot() {
        let (state, report) = restore_paper_state(&sample_snapshot()).unwrap();
        assert_eq!(state.positions.len(), 1);
        assert_eq!(report.position_count, 1);
        assert_eq!(report.restored_symbols, vec!["BTC".to_string()]);
        assert_eq!(state.runtime.entry_attempt_ms_by_symbol["BTC"], 1_772_676_500_000);
        assert!((report.balance - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_duplicate_symbols_after_normalisation() {
        let mut snapshot = sample_snapshot();
        snapshot.positions.push(SnapshotPosition {
            symbol: " btc ".to_string(),
            ..snapshot.positions[0].clone()
        });

        let err = restore_paper_state(&snapshot).unwrap_err();
        assert!(matches!(err, PaperBootstrapError::DuplicateSymbol(_)));
    }

    #[test]
    fn preserves_balance_precision_in_report() {
        let mut snapshot = sample_snapshot();
        snapshot.balance = 1000.42;

        let (_state, report) = restore_paper_state(&snapshot).unwrap();
        assert!((report.balance - 1000.42).abs() < 1e-9);
    }
}
