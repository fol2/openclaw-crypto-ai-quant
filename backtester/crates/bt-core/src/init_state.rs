//! Initial state injection for the backtester.
//!
//! Allows loading a pre-existing portfolio snapshot (exported from live or paper
//! trading) so that the backtester can continue from current positions instead
//! of starting from a blank slate.

use crate::config::Confidence;
use crate::position::{Position, PositionType};
use rustc_hash::FxHashMap;
use serde::Deserialize;

/// JSON schema version supported by this loader.
const SUPPORTED_VERSION: u32 = 1;

/// Top-level init-state file (matches `export_state.py` output).
#[derive(Debug, Deserialize)]
pub struct InitStateFile {
    pub version: u32,
    pub source: String,
    pub exported_at_ms: i64,
    pub balance: f64,
    pub positions: Vec<InitPosition>,
}

/// A single position entry in the init-state file.
#[derive(Debug, Deserialize)]
pub struct InitPosition {
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

/// Load and validate an init-state JSON file from disk.
pub fn load(path: &str) -> Result<InitStateFile, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read init-state file {:?}: {}", path, e))?;
    let state: InitStateFile = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse init-state JSON: {}", e))?;

    if state.version != SUPPORTED_VERSION {
        return Err(format!(
            "Unsupported init-state version {} (expected {})",
            state.version, SUPPORTED_VERSION,
        ));
    }
    if state.balance < 0.0 {
        return Err(format!("Invalid init-state balance: {}", state.balance));
    }

    Ok(state)
}

/// Convert the loaded init-state into simulator-compatible state.
///
/// Returns `(balance, positions)`.  Positions whose symbol is absent from
/// `valid_symbols` (when provided) are dropped with a warning on stderr.
pub fn into_sim_state(
    state: InitStateFile,
    valid_symbols: Option<&[&str]>,
) -> (f64, FxHashMap<String, Position>) {
    let mut positions = FxHashMap::default();

    for ip in state.positions {
        if let Some(syms) = valid_symbols {
            if !syms.contains(&ip.symbol.as_str()) {
                eprintln!(
                    "[init-state] WARNING: position for {:?} skipped — not in candle data",
                    ip.symbol,
                );
                continue;
            }
        }

        let pos_type = match ip.side.to_lowercase().as_str() {
            "long" => PositionType::Long,
            "short" => PositionType::Short,
            other => {
                eprintln!(
                    "[init-state] WARNING: position for {:?} skipped — unknown side {:?}",
                    ip.symbol, other,
                );
                continue;
            }
        };

        let confidence = match ip.confidence.to_lowercase().as_str() {
            "high" => Confidence::High,
            "medium" => Confidence::Medium,
            "low" => Confidence::Low,
            _ => Confidence::Medium,
        };

        let pos = Position {
            symbol: ip.symbol.clone(),
            pos_type,
            entry_price: ip.entry_price,
            size: ip.size,
            confidence,
            entry_atr: ip.entry_atr,
            entry_adx_threshold: ip.entry_adx_threshold,
            trailing_sl: ip.trailing_sl,
            leverage: ip.leverage,
            margin_used: ip.margin_used,
            adds_count: ip.adds_count,
            tp1_taken: ip.tp1_taken,
            open_time_ms: ip.open_time_ms,
            last_add_time_ms: ip.last_add_time_ms,
            mae_usd: 0.0,
            mfe_usd: 0.0,
        };
        positions.insert(ip.symbol, pos);
    }

    eprintln!(
        "[init-state] Loaded balance=${:.2}, {} position(s) from {:?} (exported at {})",
        state.balance,
        positions.len(),
        state.source,
        state.exported_at_ms,
    );

    (state.balance, positions)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_json() {
        let json = r#"{
            "version": 1,
            "source": "paper",
            "exported_at_ms": 1770562800000,
            "balance": 230.19,
            "positions": [
                {
                    "symbol": "BTC",
                    "side": "long",
                    "size": 0.003,
                    "entry_price": 71000.0,
                    "entry_atr": 1200.0,
                    "trailing_sl": 69500.0,
                    "confidence": "medium",
                    "leverage": 5.0,
                    "margin_used": 42.6,
                    "adds_count": 0,
                    "tp1_taken": false,
                    "open_time_ms": 1770500000000,
                    "last_add_time_ms": 0
                }
            ]
        }"#;

        let state: InitStateFile = serde_json::from_str(json).unwrap();
        assert_eq!(state.version, 1);
        assert_eq!(state.source, "paper");
        assert!((state.balance - 230.19).abs() < 0.01);
        assert_eq!(state.positions.len(), 1);
        assert_eq!(state.positions[0].symbol, "BTC");
        assert_eq!(state.positions[0].side, "long");
    }

    #[test]
    fn test_into_sim_state_with_valid_symbols() {
        let state = InitStateFile {
            version: 1,
            source: "paper".to_string(),
            exported_at_ms: 1770562800000,
            balance: 500.0,
            positions: vec![
                InitPosition {
                    symbol: "BTC".to_string(),
                    side: "long".to_string(),
                    size: 0.003,
                    entry_price: 71000.0,
                    entry_atr: 1200.0,
                    trailing_sl: Some(69500.0),
                    confidence: "high".to_string(),
                    leverage: 5.0,
                    margin_used: 42.6,
                    adds_count: 1,
                    tp1_taken: false,
                    open_time_ms: 1770500000000,
                    last_add_time_ms: 1770510000000,
                    entry_adx_threshold: 22.0,
                },
                InitPosition {
                    symbol: "DOGE".to_string(),
                    side: "short".to_string(),
                    size: 100.0,
                    entry_price: 0.15,
                    entry_atr: 0.005,
                    trailing_sl: None,
                    confidence: "low".to_string(),
                    leverage: 3.0,
                    margin_used: 5.0,
                    adds_count: 0,
                    tp1_taken: false,
                    open_time_ms: 1770550000000,
                    last_add_time_ms: 0,
                    entry_adx_threshold: 10.0,
                },
            ],
        };

        let valid_syms = vec!["BTC", "ETH", "DOGE"];
        let (balance, positions) = into_sim_state(state, Some(&valid_syms));
        assert!((balance - 500.0).abs() < 0.01);
        assert_eq!(positions.len(), 2);

        let btc = positions.get("BTC").unwrap();
        assert_eq!(btc.pos_type, PositionType::Long);
        assert!((btc.entry_price - 71000.0).abs() < 0.01);
        assert_eq!(btc.trailing_sl, Some(69500.0));
        assert_eq!(btc.adds_count, 1);
        assert!((btc.entry_adx_threshold - 22.0).abs() < 0.01);

        let doge = positions.get("DOGE").unwrap();
        assert_eq!(doge.pos_type, PositionType::Short);
        assert!((doge.size - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_into_sim_state_filters_unknown_symbols() {
        let state = InitStateFile {
            version: 1,
            source: "live".to_string(),
            exported_at_ms: 0,
            balance: 100.0,
            positions: vec![InitPosition {
                symbol: "UNKNOWN".to_string(),
                side: "long".to_string(),
                size: 1.0,
                entry_price: 50.0,
                entry_atr: 2.0,
                trailing_sl: None,
                confidence: "medium".to_string(),
                leverage: 3.0,
                margin_used: 16.67,
                adds_count: 0,
                tp1_taken: false,
                open_time_ms: 0,
                last_add_time_ms: 0,
                entry_adx_threshold: 0.0,
            }],
        };

        let valid_syms = vec!["BTC", "ETH"];
        let (_, positions) = into_sim_state(state, Some(&valid_syms));
        assert!(positions.is_empty());
    }

    #[test]
    fn test_unsupported_version() {
        let json = r#"{
            "version": 99,
            "source": "paper",
            "exported_at_ms": 0,
            "balance": 100.0,
            "positions": []
        }"#;

        let tmpdir = std::env::temp_dir();
        let path = tmpdir.join("test_init_state_v99.json");
        std::fs::write(&path, json).unwrap();
        let result = load(path.to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unsupported init-state version"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_no_valid_symbols_filter() {
        let state = InitStateFile {
            version: 1,
            source: "paper".to_string(),
            exported_at_ms: 0,
            balance: 1000.0,
            positions: vec![InitPosition {
                symbol: "ETH".to_string(),
                side: "long".to_string(),
                size: 0.5,
                entry_price: 3000.0,
                entry_atr: 100.0,
                trailing_sl: None,
                confidence: "high".to_string(),
                leverage: 5.0,
                margin_used: 300.0,
                adds_count: 0,
                tp1_taken: false,
                open_time_ms: 0,
                last_add_time_ms: 0,
                entry_adx_threshold: 22.0,
            }],
        };

        // None means accept all symbols
        let (balance, positions) = into_sim_state(state, None);
        assert!((balance - 1000.0).abs() < 0.01);
        assert_eq!(positions.len(), 1);
        assert!(positions.contains_key("ETH"));
    }
}
