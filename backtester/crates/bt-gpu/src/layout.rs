//! Combo generation, VRAM batching, and result collection.

/// High-level result for one sweep combo.
#[derive(Debug, Clone)]
pub struct GpuSweepResult {
    pub config_id: String,
    pub total_pnl: f64,
    pub final_balance: f64,
    pub total_trades: u32,
    pub total_wins: u32,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub max_drawdown_pct: f64,
    pub overrides: Vec<(String, f64)>,
}
