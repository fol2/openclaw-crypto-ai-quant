use serde::Deserialize;
use std::fmt;

pub mod confidence;
pub mod entry;
pub mod gates;
mod views;

pub use views::{
    AnomalyThresholdsView, EntryThresholdsView, FiltersView, IndicatorSnapshotLike,
    RangingThresholdsView, SignalConfigLike, SignalConfigView, SnapshotView,
    StochRsiThresholdsView, ThresholdsView, TpAndMomentumThresholdsView, TradeView,
};

/// Signal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    Buy,
    Sell,
    Neutral,
}

/// Confidence tier for entry and add filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Confidence {
    #[default]
    Low = 0,
    Medium = 1,
    High = 2,
}

impl Confidence {
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "low" => Confidence::Low,
            "medium" => Confidence::Medium,
            "high" => Confidence::High,
            _ => Confidence::High,
        }
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Confidence::Low => write!(f, "low"),
            Confidence::Medium => write!(f, "medium"),
            Confidence::High => write!(f, "high"),
        }
    }
}

impl<'de> Deserialize<'de> for Confidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Confidence::from_str(&s))
    }
}

/// MACD histogram gating mode for trend entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacdMode {
    /// Require MACD_hist > prev_MACD_hist (acceleration).
    Accel,
    /// Require MACD_hist > 0 for BUY, < 0 for SELL.
    Sign,
    /// Ignore MACD_hist gate.
    None,
}

impl MacdMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "accel" => MacdMode::Accel,
            "sign" => MacdMode::Sign,
            "none" => MacdMode::None,
            _ => MacdMode::Accel,
        }
    }
}

impl fmt::Display for MacdMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MacdMode::Accel => write!(f, "accel"),
            MacdMode::Sign => write!(f, "sign"),
            MacdMode::None => write!(f, "none"),
        }
    }
}

impl<'de> Deserialize<'de> for MacdMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(MacdMode::from_str(&s))
    }
}
