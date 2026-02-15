//! Combo generation, VRAM batching, and result collection.

/// High-level result for one sweep combo.
#[derive(Debug, Clone)]
pub struct GpuSweepResult {
    pub config_id: String,
    pub output_mode: String,
    pub total_pnl: f64,
    pub final_balance: f64,
    pub total_trades: u32,
    pub total_wins: u32,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub max_drawdown_pct: f64,
    pub overrides: Vec<(String, f64)>,
}

/// Wrapper for BinaryHeap min-heap ordering by PnL.
/// BinaryHeap is a max-heap; reversed comparison puts lowest PnL at root
/// so we can evict the worst result when capacity is exceeded.
#[derive(Debug, Clone)]
pub struct MinPnlHeapEntry(pub GpuSweepResult);

impl Eq for MinPnlHeapEntry {}

impl PartialEq for MinPnlHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_pnl == other.0.total_pnl
    }
}

impl PartialOrd for MinPnlHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinPnlHeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed: lowest PnL = greatest in heap ordering (evicted first by pop)
        // NaN treated as NEG_INFINITY (worst possible)
        let a = if self.0.total_pnl.is_nan() {
            f64::NEG_INFINITY
        } else {
            self.0.total_pnl
        };
        let b = if other.0.total_pnl.is_nan() {
            f64::NEG_INFINITY
        } else {
            other.0.total_pnl
        };
        a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
    }
}
