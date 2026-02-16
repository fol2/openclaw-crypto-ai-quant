// =============================================================================
// GPU Sweep Engine -- CUDA Compute Kernel
// =============================================================================
//
// Originally ported from sweep_engine.wgsl; now the sole production kernel.
// The WGSL shader is deprecated (AQC-1241) and no longer maintained.
// Mirrors the Rust backtester trade logic for parameter sweep.
// Indicators are precomputed on CPU; this kernel only runs trade decisions.
//
// Buffer layout (kernel parameters):
//   params     -- const GpuParams*          (uniform equivalent)
//   snapshots  -- const GpuSnapshot*        [num_bars * num_symbols]
//   breadth    -- const float*              [num_bars]
//   btc_bull   -- const unsigned int*       [num_bars]
//   configs    -- const GpuComboConfig*     [num_combos]
//   states     -- GpuComboState*            [num_combos]  (read-write)
//   results    -- GpuResult*                [num_combos]  (read-write)

#include <cstdint>

#define MAX_SYMBOLS      52u
// Candidate buffer must cover the full project symbol universe to keep
// ranking parity with CPU (no early truncation while scanning symbols).
#define MAX_CANDIDATES   64u

// Position type constants
#define POS_EMPTY   0u
#define POS_LONG    1u
#define POS_SHORT   2u

// Signal constants
#define SIG_NEUTRAL 0u
#define SIG_BUY     1u
#define SIG_SELL    2u

// Confidence constants
#define CONF_LOW    0u
#define CONF_MEDIUM 1u
#define CONF_HIGH   2u

// MACD mode
#define MACD_ACCEL  0u
#define MACD_SIGN   1u
#define MACD_NONE   2u

// BTC alignment state (from breadth kernel)
#define BTC_BULL_BEARISH 0u
#define BTC_BULL_BULLISH 1u
#define BTC_BULL_UNKNOWN 2u

// PESC close reasons
#define PESC_NONE        0u
#define PESC_SIGNAL_FLIP 1u
#define PESC_OTHER       2u

// -- Structs (match Rust #[repr(C)] exactly) ----------------------------------

struct GpuRawCandle {
    float open;
    float high;
    float low;
    float close;
    float volume;
    unsigned int t_sec;
    unsigned int _pad[2];
};

struct GpuParams {
    unsigned int num_combos;
    unsigned int num_symbols;
    unsigned int num_bars;
    unsigned int btc_sym_idx;
    unsigned int chunk_start;
    unsigned int chunk_end;
    unsigned int initial_balance_bits;
    unsigned int maker_fee_rate_bits;
    unsigned int taker_fee_rate_bits;
    unsigned int max_sub_per_bar;
    unsigned int trade_end_bar;
};

struct __align__(16) GpuSnapshot {
    float close;  float high;  float low;  float open;             // 16
    float volume;  unsigned int t_sec;                              // 24
    float ema_fast;  float ema_slow;  float ema_macro;             // 36
    float adx;  float adx_slope;  float adx_pos;  float adx_neg;  // 52
    float atr;  float atr_slope;  float avg_atr;                   // 64
    float bb_upper;  float bb_lower;  float bb_width;  float bb_width_ratio; // 80
    float rsi;  float stoch_k;  float stoch_d;                    // 92
    float macd_hist;  float prev_macd_hist;  float prev2_macd_hist;  float prev3_macd_hist; // 108
    float vol_sma;  unsigned int vol_trend;                        // 116
    float prev_close;  float prev_ema_fast;  float prev_ema_slow;  // 128
    float ema_slow_slope_pct;                                      // 132
    unsigned int bar_count;  unsigned int valid;                   // 140
    float funding_rate;                                            // 144
    unsigned int _pad[4];                                          // 160
};

struct __align__(16) GpuPosition {
    unsigned int active;                                           // replaces pos_type
    float entry_price;  float size;  unsigned int confidence;      // 16
    float entry_atr;  float entry_adx_threshold;                   // 24
    float trailing_sl;  float leverage;                            // 32
    float margin_used;  unsigned int adds_count;  unsigned int tp1_taken; // 44
    unsigned int open_time_sec;  unsigned int last_add_time_sec;   // 52
    unsigned int _pad[3];                                          // 64
};

struct __align__(16) GpuComboConfig {
    float allocation_pct;  float sl_atr_mult;  float tp_atr_mult;  float leverage;
    unsigned int enable_reef_filter;  float reef_long_rsi_block_gt;  float reef_short_rsi_block_lt;
    float reef_adx_threshold;  float reef_long_rsi_extreme_gt;  float reef_short_rsi_extreme_lt;
    unsigned int enable_dynamic_leverage;  float leverage_low;  float leverage_medium;
    float leverage_high;  float leverage_max_cap;  float trailing_rsi_floor_default;
    float slippage_bps;  float min_notional_usd;  unsigned int bump_to_min_notional;
    float max_total_margin_pct;  float trailing_rsi_floor_trending;  float trailing_vbts_bb_threshold;
    unsigned int enable_dynamic_sizing;  float confidence_mult_high;  float confidence_mult_medium;
    float confidence_mult_low;  float adx_sizing_min_mult;  float adx_sizing_full_adx;
    float vol_baseline_pct;  float vol_scalar_min;
    float vol_scalar_max;  float trailing_vbts_mult;
    unsigned int enable_pyramiding;  unsigned int max_adds_per_symbol;  float add_fraction_of_base_margin;
    unsigned int add_cooldown_minutes;  float add_min_profit_atr;  unsigned int add_min_confidence;
    unsigned int entry_min_confidence;  float trailing_high_profit_atr;
    unsigned int enable_partial_tp;  float tp_partial_pct;  float tp_partial_min_notional_usd;
    float trailing_start_atr;  float trailing_distance_atr;  float tp_partial_atr_mult;
    unsigned int enable_ssf_filter;  unsigned int enable_breakeven_stop;  float breakeven_start_atr;
    float breakeven_buffer_atr;
    float trailing_start_atr_low_conf;  float trailing_distance_atr_low_conf;
    float smart_exit_adx_exhaustion_lt;  float smart_exit_adx_exhaustion_lt_low_conf;
    unsigned int enable_rsi_overextension_exit;  float rsi_exit_profit_atr_switch;
    float rsi_exit_ub_lo_profit;  float rsi_exit_ub_hi_profit;
    float rsi_exit_lb_lo_profit;  float rsi_exit_lb_hi_profit;
    float rsi_exit_ub_lo_profit_low_conf;  float rsi_exit_ub_hi_profit_low_conf;
    float rsi_exit_lb_lo_profit_low_conf;  float rsi_exit_lb_hi_profit_low_conf;
    unsigned int reentry_cooldown_minutes;  unsigned int reentry_cooldown_min_mins;
    unsigned int reentry_cooldown_max_mins;  float trailing_tighten_default;
    unsigned int enable_vol_buffered_trailing;  float tsme_min_profit_atr;
    unsigned int tsme_require_adx_slope_negative;  float trailing_tighten_tspv;
    float min_atr_pct;  unsigned int reverse_entry_signal;  unsigned int block_exits_on_extreme_dev;
    float glitch_price_dev_pct;  float glitch_atr_mult;  float trailing_weak_trend_mult;
    unsigned int max_open_positions;  unsigned int max_entry_orders_per_loop;
    unsigned int enable_slow_drift_entries;  unsigned int slow_drift_require_macd_sign;
    unsigned int enable_ranging_filter;  unsigned int enable_anomaly_filter;  unsigned int enable_extension_filter;
    unsigned int require_adx_rising;  float adx_rising_saturation;  unsigned int require_volume_confirmation;
    unsigned int vol_confirm_include_prev;  unsigned int use_stoch_rsi_filter;
    unsigned int require_btc_alignment;  unsigned int require_macro_alignment;
    unsigned int enable_regime_filter;  unsigned int enable_auto_reverse;
    float auto_reverse_breadth_low;  float auto_reverse_breadth_high;
    float breadth_block_short_above;  float breadth_block_long_below;
    float min_adx;  float high_conf_volume_mult;  float btc_adx_override;
    float max_dist_ema_fast;  float ave_atr_ratio_gt;  float ave_adx_mult;
    float dre_min_adx;  float dre_max_adx;
    float dre_long_rsi_limit_low;  float dre_long_rsi_limit_high;
    float dre_short_rsi_limit_low;  float dre_short_rsi_limit_high;
    unsigned int macd_mode;  float pullback_min_adx;  float pullback_rsi_long_min;
    float pullback_rsi_short_max;  unsigned int pullback_require_macd_sign;  unsigned int pullback_confidence;
    float slow_drift_min_slope_pct;  float slow_drift_min_adx;
    float slow_drift_rsi_long_min;  float slow_drift_rsi_short_max;
    float ranging_adx_lt;  float ranging_bb_width_ratio_lt;
    float anomaly_bb_width_ratio_gt;  float slow_drift_ranging_slope_override;
    unsigned int snapshot_offset;  unsigned int breadth_offset;
    float tp_strong_adx_gt;  float tp_weak_adx_lt;
    // === Decision codegen fields (AQC-1250) ===
    unsigned int enable_pullback_entries;
    float anomaly_price_change_pct;  float anomaly_ema_dev_pct;
    float ranging_rsi_low;  float ranging_rsi_high;  unsigned int ranging_min_signals;
    float stoch_rsi_block_long_gt;  float stoch_rsi_block_short_lt;
    unsigned int ave_enabled;
    float tp_mult_strong;  float tp_mult_weak;
    unsigned int _codegen_pad;
};

struct GpuComboState {
    double balance;  unsigned int num_open;  unsigned int entries_this_bar;
    GpuPosition positions[52];
    unsigned int pesc_close_time_sec[52];
    unsigned int pesc_close_type[52];
    unsigned int pesc_close_reason[52];
    double total_pnl;  double total_fees;  unsigned int total_trades;  unsigned int total_wins;
    double gross_profit;  double gross_loss;  double max_drawdown;  double peak_equity;
    unsigned int _acc_pad[2];
};

struct __align__(16) GpuResult {
    float final_balance;  float total_pnl;  float total_fees;
    unsigned int total_trades;  unsigned int total_wins;  unsigned int total_losses;
    float gross_profit;  float gross_loss;  float max_drawdown_pct;
    unsigned int _pad[3];
};

struct EntryCandidate {
    unsigned int sym_idx;
    unsigned int signal;
    unsigned int confidence;
    float score;
    float adx;
    float atr;
    float entry_adx_threshold;
};

// -- Codegen'd decision functions (AQC-1213) ----------------------------------
// Structs defined here: GateResultD, SignalResult, TpResult, SmartExitResult,
// SizingResultD.  Device functions: check_gates_codegen, generate_signal_codegen,
// compute_sl_price_codegen, compute_trailing_codegen, check_tp_codegen,
// check_smart_exits_codegen, compute_entry_size_codegen, is_pesc_blocked_codegen.
#include "generated_decision.cu"

// -- Helper functions ---------------------------------------------------------

__device__ float get_taker_fee_rate(const GpuParams* params) {
    return __uint_as_float(params->taker_fee_rate_bits);
}

__device__ float profit_atr(const GpuPosition& pos, float price) {
    float atr = (pos.entry_atr > 0.0f) ? pos.entry_atr : (pos.entry_price * 0.005f);
    if (atr <= 0.0f) { return 0.0f; }
    if (pos.active == POS_LONG) {
        return (price - pos.entry_price) / atr;
    } else {
        return (pos.entry_price - price) / atr;
    }
}

__device__ float profit_usd(const GpuPosition& pos, float price) {
    if (pos.active == POS_LONG) {
        return (price - pos.entry_price) * pos.size;
    } else {
        return (pos.entry_price - price) * pos.size;
    }
}

__device__ float apply_atr_floor(float atr, float price, float min_atr_pct) {
    if (min_atr_pct > 0.0f) {
        float floor_val = price * min_atr_pct;
        if (atr < floor_val) { return floor_val; }
    }
    return atr;
}

__device__ bool conf_meets_min(unsigned int conf, unsigned int min_conf) {
    return conf >= min_conf;
}

// -- Signal Reversal & Regime Filter ------------------------------------------

__device__ unsigned int apply_reverse(unsigned int signal, const GpuComboConfig* cfg, float breadth_pct) {
    bool should_reverse = (cfg->reverse_entry_signal != 0u);

    if (cfg->enable_auto_reverse != 0u) {
        float low = cfg->auto_reverse_breadth_low;
        float high = cfg->auto_reverse_breadth_high;
        if (breadth_pct >= low && breadth_pct <= high) {
            should_reverse = true;
        } else {
            should_reverse = false;
        }
    }

    if (should_reverse) {
        if (signal == SIG_BUY) { return SIG_SELL; }
        if (signal == SIG_SELL) { return SIG_BUY; }
    }
    return signal;
}

__device__ unsigned int apply_regime_filter(unsigned int signal, const GpuComboConfig* cfg, float breadth_pct) {
    if (cfg->enable_regime_filter == 0u) { return signal; }
    if (signal == SIG_SELL && breadth_pct > cfg->breadth_block_short_above) { return SIG_NEUTRAL; }
    if (signal == SIG_BUY && breadth_pct < cfg->breadth_block_long_below) { return SIG_NEUTRAL; }
    return signal;
}

// -- Gate Checks (AQC-1213: delegated to check_gates_codegen) -----------------
//
// The GateResult struct is a lightweight facade used by the sweep kernel.
// The actual gate evaluation is performed by check_gates_codegen() from
// generated_decision.cu which operates in double precision (AQC-734).

struct GateResult {
    bool all_gates_pass;
    bool is_ranging;
    bool is_anomaly;
    bool is_extended;
    bool vol_confirm;
    bool bullish_alignment;
    bool bearish_alignment;
};

__device__ GateResult check_gates(const GpuSnapshot& snap, const GpuComboConfig* cfg,
                                   float ema_slope, unsigned int btc_bull,
                                   bool is_btc_symbol) {
    // AQC-1213: replaced by codegen — delegates to check_gates_codegen()
    GateResultD gd = check_gates_codegen(
        *cfg,
        (double)snap.rsi,
        (double)snap.adx,
        (double)snap.adx_slope,
        (double)snap.bb_width_ratio,
        (double)snap.ema_fast,
        (double)snap.ema_slow,
        (double)snap.ema_macro,
        (double)snap.close,
        (double)snap.prev_close,
        (double)snap.volume,
        (double)snap.vol_sma,
        snap.vol_trend,
        (double)snap.atr,
        (double)snap.avg_atr,
        (double)snap.stoch_k,
        (double)ema_slope,
        btc_bull,
        is_btc_symbol ? 1u : 0u
    );

    // Convert GateResultD -> GateResult (facade for downstream kernel code)
    GateResult result;
    result.all_gates_pass   = gd.all_gates_pass;
    result.is_ranging       = gd.is_ranging;
    result.is_anomaly       = gd.is_anomaly;
    result.is_extended      = gd.is_extended;
    result.vol_confirm      = gd.vol_confirm;
    result.bullish_alignment = gd.bullish_alignment;
    result.bearish_alignment = gd.bearish_alignment;
    return result;
}

// Backwards-compatible overload (legacy call sites that don't pass btc_bull).
__device__ GateResult check_gates(const GpuSnapshot& snap, const GpuComboConfig* cfg,
                                   float ema_slope) {
    return check_gates(snap, cfg, ema_slope, /*btc_bull=*/BTC_BULL_UNKNOWN, /*is_btc_symbol=*/false);
}

// -- Signal Generation (AQC-1213: delegated to generate_signal_codegen) -------
//
// The codegen SignalResult (from generated_decision.cu) uses:
//   int signal, int confidence, double effective_min_adx
// The sweep kernel expects unsigned int signal/confidence and float entry_adx_threshold.
// This wrapper struct + function bridges the two.

struct SignalResultLegacy {
    unsigned int signal;
    unsigned int confidence;
    float entry_adx_threshold;
};

__device__ SignalResultLegacy generate_signal(const GpuSnapshot& snap, const GpuComboConfig* cfg,
                                              const GateResult& gates, unsigned int btc_bull,
                                              bool is_btc_symbol, float ema_slope) {
    // AQC-1213: replaced by codegen — delegates to generate_signal_codegen()
    SignalResult sr = generate_signal_codegen(
        *cfg,
        (double)snap.close,
        (double)snap.ema_fast,
        (double)snap.ema_slow,
        (double)snap.adx,
        (double)snap.rsi,
        (double)snap.macd_hist,
        (double)snap.prev_macd_hist,
        (double)snap.volume,
        (double)snap.vol_sma,
        (double)snap.atr,
        (double)snap.avg_atr,
        (double)snap.stoch_k,
        (double)snap.prev_close,
        (double)snap.prev_ema_fast,
        (double)ema_slope,
        gates.all_gates_pass,
        gates.bullish_alignment,
        gates.bearish_alignment,
        gates.is_anomaly,
        gates.is_extended,
        gates.is_ranging,
        gates.vol_confirm,
        btc_bull,
        is_btc_symbol
    );

    // Convert SignalResult (codegen) -> SignalResultLegacy (kernel facade)
    SignalResultLegacy result;
    result.signal = (unsigned int)sr.signal;
    result.confidence = (unsigned int)sr.confidence;
    result.entry_adx_threshold = (float)sr.effective_min_adx;
    return result;
}

// -- Stop Loss ----------------------------------------------------------------
// AQC-1226: replaced by codegen — delegates to compute_sl_price_codegen()

__device__ float compute_sl_price(const GpuPosition& pos, const GpuSnapshot& snap, const GpuComboConfig* cfg) {
    // AQC-1226: delegate to double-precision codegen
    double sl = compute_sl_price_codegen(
        *cfg,
        (int)pos.active,
        (double)pos.entry_price,
        (double)pos.entry_atr,
        (double)snap.close,
        (double)snap.adx,
        (double)snap.adx_slope
    );
    return (float)sl;
}

__device__ bool stop_loss_hit(unsigned int pos_type, float price, float sl_price) {
    // Mirrors bt-core::exits::stop_loss::check_stop_loss:
    // LONG exits when price <= SL, SHORT exits when price >= SL.
    if (pos_type == POS_LONG) { return price <= sl_price; }
    return price >= sl_price;
}

__device__ bool check_stop_loss(const GpuPosition& pos, const GpuSnapshot& snap, const GpuComboConfig* cfg) {
    float sl = compute_sl_price(pos, snap, cfg);
    return stop_loss_hit(pos.active, snap.close, sl);
}

// -- Trailing Stop ------------------------------------------------------------
// AQC-1226: replaced by codegen — delegates to compute_trailing_codegen()

__device__ float compute_trailing(const GpuPosition& pos, const GpuSnapshot& snap,
                                  const GpuComboConfig* cfg, float p_atr) {
    // AQC-1226: delegate to double-precision codegen
    double tsl = compute_trailing_codegen(
        *cfg,
        (int)pos.active,
        (double)pos.entry_price,
        (double)snap.close,
        (double)pos.entry_atr,
        (double)pos.trailing_sl,
        (int)pos.confidence,
        (double)snap.rsi,
        (double)snap.adx,
        (double)snap.adx_slope,
        (double)snap.atr_slope,
        (double)snap.bb_width_ratio,
        (double)p_atr
    );
    return (float)tsl;
}

__device__ bool check_trailing_exit(const GpuPosition& pos, const GpuSnapshot& snap) {
    if (pos.trailing_sl <= 0.0f) { return false; }
    if (pos.active == POS_LONG) { return snap.close <= pos.trailing_sl; }
    return snap.close >= pos.trailing_sl;
}

// -- Take Profit --------------------------------------------------------------
// AQC-1226: replaced by codegen — delegates to check_tp_codegen()

// Returns: 0 = hold, 1 = partial, 2 = full close
__device__ unsigned int check_tp(const GpuPosition& pos, const GpuSnapshot& snap,
                                 const GpuComboConfig* cfg, float tp_mult) {
    // AQC-1226: delegate to double-precision codegen
    TpResult tp = check_tp_codegen(
        *cfg,
        (int)pos.active,
        (double)pos.entry_price,
        (double)pos.entry_atr,
        (double)snap.close,
        (double)pos.size,
        pos.tp1_taken,
        (double)tp_mult
    );
    return (unsigned int)tp.action;
}

// -- Smart Exits --------------------------------------------------------------
// AQC-1226: replaced by codegen — delegates to check_smart_exits_codegen()

__device__ bool check_smart_exits(const GpuPosition& pos, const GpuSnapshot& snap,
                                  const GpuComboConfig* cfg, float p_atr) {
    // AQC-1226: delegate to double-precision codegen
    SmartExitResult se = check_smart_exits_codegen(
        *cfg,
        (int)pos.active,
        (double)pos.entry_price,
        (double)pos.entry_atr,
        (double)snap.close,
        (double)snap.ema_fast,
        (double)snap.ema_slow,
        (double)snap.ema_macro,
        (double)snap.adx,
        (double)snap.adx_slope,
        (double)snap.atr,
        (double)snap.avg_atr,
        (double)snap.rsi,
        (double)snap.macd_hist,
        (double)snap.prev_macd_hist,
        (double)snap.prev2_macd_hist,
        (double)snap.prev3_macd_hist,
        (double)p_atr,
        (int)pos.confidence,
        (double)pos.entry_adx_threshold
    );
    return se.should_exit;
}

// -- Entry Sizing (AQC-1233: delegated to compute_entry_size_codegen) ---------
// The codegen SizingResultD operates in double precision (AQC-734).
// This wrapper bridges to the float-based SizingResult used by the kernel.

struct SizingResult {
    float size;
    float margin;
    float leverage;
};

__device__ SizingResult compute_entry_size(float equity, float price, unsigned int confidence,
                                           float atr, const GpuSnapshot& snap,
                                           const GpuComboConfig* cfg) {
    // AQC-1233: delegate to double-precision codegen
    SizingResultD sd = compute_entry_size_codegen(
        (double)equity, (double)price, confidence,
        (double)atr, (double)snap.adx, *cfg
    );

    SizingResult result;
    result.size = (float)sd.size;
    result.margin = (float)sd.margin;
    result.leverage = (float)sd.leverage;
    return result;
}

// -- Exit Application ---------------------------------------------------------

__device__ void clear_position(GpuPosition* pos) {
    pos->active = POS_EMPTY;
    pos->entry_price = 0.0f;
    pos->size = 0.0f;
    pos->confidence = 0u;
    pos->entry_atr = 0.0f;
    pos->entry_adx_threshold = 0.0f;
    pos->trailing_sl = 0.0f;
    pos->leverage = 0.0f;
    pos->margin_used = 0.0f;
    pos->adds_count = 0u;
    pos->tp1_taken = 0u;
    pos->open_time_sec = 0u;
    pos->last_add_time_sec = 0u;
    pos->_pad[0] = 0u;
    pos->_pad[1] = 0u;
    pos->_pad[2] = 0u;
}

__device__ void apply_close(GpuComboState* state, unsigned int sym, const GpuSnapshot& snap,
                            bool reason_is_signal_flip, float fee_rate, float slippage_bps) {
    const GpuPosition& pos = state->positions[sym];
    if (pos.active == POS_EMPTY) { return; }

    float slip = (pos.active == POS_LONG) ? slippage_bps : -slippage_bps;
    float fill_price = snap.close * (1.0f + slip / 10000.0f);

    // Use double for PnL/fee to prevent accumulation drift over 10K+ trades
    double pnl;
    if (pos.active == POS_LONG) {
        pnl = ((double)fill_price - (double)pos.entry_price) * (double)pos.size;
    } else {
        pnl = ((double)pos.entry_price - (double)fill_price) * (double)pos.size;
    }
    double notional = (double)pos.size * (double)fill_price;
    double fee = notional * (double)fee_rate;

    state->balance += pnl - fee;
    state->total_pnl += pnl;
    state->total_fees += fee;
    state->total_trades += 1u;
    if (pnl > 0.0) {
        state->total_wins += 1u;
        state->gross_profit += pnl;
    } else {
        state->gross_loss += fabs(pnl);
    }
    state->num_open -= 1u;

    // PESC tracking
    state->pesc_close_time_sec[sym] = snap.t_sec;
    state->pesc_close_type[sym] = pos.active;
    if (reason_is_signal_flip) {
        state->pesc_close_reason[sym] = PESC_SIGNAL_FLIP;
    } else {
        state->pesc_close_reason[sym] = PESC_OTHER;
    }

    // Clear position
    clear_position(&state->positions[sym]);
}

__device__ void apply_partial_close(GpuComboState* state, unsigned int sym, const GpuSnapshot& snap,
                                    float pct, float fee_rate, float slippage_bps) {
    const GpuPosition& pos = state->positions[sym];
    if (pos.active == POS_EMPTY) { return; }

    float exit_size = pos.size * pct;
    float slip = (pos.active == POS_LONG) ? slippage_bps : -slippage_bps;
    float fill_price = snap.close * (1.0f + slip / 10000.0f);

    // Use double for PnL/fee to prevent accumulation drift over 10K+ trades
    double pnl;
    if (pos.active == POS_LONG) {
        pnl = ((double)fill_price - (double)pos.entry_price) * (double)exit_size;
    } else {
        pnl = ((double)pos.entry_price - (double)fill_price) * (double)exit_size;
    }
    double notional = (double)exit_size * (double)fill_price;
    double fee = notional * (double)fee_rate;

    state->balance += pnl - fee;
    state->total_pnl += pnl;
    state->total_fees += fee;
    state->total_trades += 1u;
    if (pnl > 0.0) {
        state->total_wins += 1u;
        state->gross_profit += pnl;
    } else {
        state->gross_loss += fabs(pnl);
    }

    // Reduce position
    state->positions[sym].size -= exit_size;
    state->positions[sym].margin_used *= (1.0f - pct);
    state->positions[sym].tp1_taken = 1u;
    // CPU semantics: trailing SL is NOT modified on partial close.
    // compute_trailing() continues ratcheting on subsequent bars.
}

// -- PESC Check (AQC-1233: delegated to is_pesc_blocked_codegen) ---------------
// Wrapper preserves the existing call-site signature while delegating to the
// double-precision codegen implementation from generated_decision.cu.

__device__ bool is_pesc_blocked(const GpuComboState* state, unsigned int sym,
                                unsigned int desired_type, unsigned int current_sec,
                                float adx, const GpuComboConfig* cfg) {
    // AQC-1233: delegate to double-precision codegen
    return is_pesc_blocked_codegen(
        *cfg,
        0,  // bars_since_exit — unused by timestamp-based cooldown
        (double)adx,
        state->pesc_close_time_sec[sym],
        state->pesc_close_reason[sym],
        state->pesc_close_type[sym],
        desired_type,
        current_sec
    );
}

// -- TP Multiplier (must mirror bt-core fixed tp_atr_mult semantics) ---------

__device__ float get_tp_mult(const GpuSnapshot& snap, const GpuComboConfig* cfg) {
    // Parity with CPU: the CPU backtester always uses the configured TP ATR multiplier.
    // Dynamic TP scaling based on ADX is intentionally disabled on GPU.
    (void)snap;
    return cfg->tp_atr_mult;
}

// =============================================================================
// Main Compute Kernel
// =============================================================================

extern "C" __global__ void sweep_engine_kernel(
    const GpuParams*      __restrict__ params,
    const GpuSnapshot*    __restrict__ snapshots,
    const float*          __restrict__ breadth,
    const unsigned int*   __restrict__ btc_bullish,
    const GpuComboConfig* __restrict__ configs,
    GpuComboState*                     states,
    GpuResult*                         results,
    const GpuRawCandle*   __restrict__ sub_candles,
    const unsigned int*   __restrict__ sub_counts
) {
    unsigned int combo_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo_id >= params->num_combos) return;

    // Load state into local memory
    GpuComboState state = states[combo_id];
    GpuComboConfig cfg  = configs[combo_id];
    float fee_rate = get_taker_fee_rate(params);
    unsigned int ns = params->num_symbols;

    unsigned int max_sub = params->max_sub_per_bar;

    for (unsigned int bar = params->chunk_start; bar < params->chunk_end; bar++) {
        state.entries_this_bar = 0u;
        float breadth_pct = breadth[cfg.breadth_offset + bar];
        unsigned int btc_bull = btc_bullish[cfg.breadth_offset + bar];

        if (max_sub > 0u) {
            // ================================================================
            // SUB-BAR PATH: use sub-bar candles for exit + entry price checks
            // Indicators come from the main bar snapshot (1h resolution)
            // ================================================================

            // ── Sub-bar exits (per symbol, chronological) ───────────────────
            for (unsigned int sym = 0u; sym < ns; sym++) {
                if (state.positions[sym].active == POS_EMPTY) { continue; }

                const GpuSnapshot& ind_snap = snapshots[cfg.snapshot_offset + bar * ns + sym];
                if (ind_snap.valid == 0u) { continue; }

                // CPU semantics: always evaluate exits once on the indicator-bar snapshot at `ts`
                // (using the main bar OHLCV), then scan sub-bars in (ts, next_ts].
                //
                // Note: if glitch guard blocks exits on the indicator bar, we skip ALL exit
                // processing including trailing SL update (matching CPU Hold semantics).
                {
                    const GpuPosition& pos = state.positions[sym];
                    if (pos.active != POS_EMPTY) {
                        float p_atr = profit_atr(pos, ind_snap.close);

                        // Glitch guard (CPU semantics): block ALL exit processing including trailing update.
                        bool block_exits = false;
                        if (cfg.block_exits_on_extreme_dev != 0u && ind_snap.prev_close > 0.0f) {
                            float price_change_pct = fabsf(ind_snap.close - ind_snap.prev_close) / ind_snap.prev_close;
                            block_exits = (price_change_pct > cfg.glitch_price_dev_pct)
                                || (ind_snap.atr > 0.0f
                                    && fabsf(ind_snap.close - ind_snap.prev_close) > ind_snap.atr * cfg.glitch_atr_mult);
                        }

                        if (block_exits) {
                            // CPU returns Hold immediately — skip trailing update too.
                        } else {
                            // Stop loss
                            if (check_stop_loss(pos, ind_snap, &cfg)) {
                                apply_close(&state, sym, ind_snap, false, fee_rate, cfg.slippage_bps);
                            } else {
                                // Update trailing stop
                                float new_tsl = compute_trailing(pos, ind_snap, &cfg, p_atr);
                                if (new_tsl > 0.0f) {
                                    state.positions[sym].trailing_sl = new_tsl;
                                }

                                // Trailing stop exit
                                if (state.positions[sym].active != POS_EMPTY
                                    && check_trailing_exit(state.positions[sym], ind_snap)) {
                                    apply_close(&state, sym, ind_snap, false, fee_rate, cfg.slippage_bps);
                                } else if (state.positions[sym].active != POS_EMPTY) {
                                    // Take profit
                                    float tp_mult = get_tp_mult(ind_snap, &cfg);
                                    unsigned int tp_result = check_tp(state.positions[sym], ind_snap, &cfg, tp_mult);
                                    if (tp_result == 1u) {
                                        apply_partial_close(&state, sym, ind_snap, cfg.tp_partial_pct, fee_rate, cfg.slippage_bps);
                                    } else if (tp_result == 2u) {
                                        apply_close(&state, sym, ind_snap, false, fee_rate, cfg.slippage_bps);
                                    } else {
                                        // Smart exits
                                        if (check_smart_exits(state.positions[sym], ind_snap, &cfg, p_atr)) {
                                            apply_close(&state, sym, ind_snap, false, fee_rate, cfg.slippage_bps);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (state.positions[sym].active == POS_EMPTY) { continue; }

                unsigned int n_sub = sub_counts[bar * ns + sym];
                for (unsigned int sub_i = 0u; sub_i < n_sub; sub_i++) {
                    const GpuRawCandle& sc = sub_candles[(bar * max_sub + sub_i) * ns + sym];
                    if (sc.close <= 0.0f) { continue; }

                    // Build hybrid snapshot: indicator values from main bar, OHLCV from sub-bar
                    GpuSnapshot hybrid = ind_snap;
                    hybrid.close = sc.close;
                    hybrid.high = sc.high;
                    hybrid.low = sc.low;
                    hybrid.open = sc.open;
                    hybrid.t_sec = sc.t_sec;

                    const GpuPosition& pos = state.positions[sym];
                    if (pos.active == POS_EMPTY) { break; } // exited in earlier sub-bar
                    float p_atr = profit_atr(pos, hybrid.close);

                    // Glitch guard (CPU semantics): block ALL exit processing including trailing update.
                    bool block_exits = false;
                    if (cfg.block_exits_on_extreme_dev != 0u && hybrid.prev_close > 0.0f) {
                        float price_change_pct = fabsf(hybrid.close - hybrid.prev_close) / hybrid.prev_close;
                        block_exits = (price_change_pct > cfg.glitch_price_dev_pct)
                            || (hybrid.atr > 0.0f
                                && fabsf(hybrid.close - hybrid.prev_close) > hybrid.atr * cfg.glitch_atr_mult);
                    }
                    if (block_exits) {
                        continue;  // CPU returns Hold — skip trailing update too
                    }

                    // Stop loss
                    if (check_stop_loss(pos, hybrid, &cfg)) {
                        apply_close(&state, sym, hybrid, false, fee_rate, cfg.slippage_bps);
                        break;
                    }

                    // Update trailing stop
                    float new_tsl = compute_trailing(pos, hybrid, &cfg, p_atr);
                    if (new_tsl > 0.0f) {
                        state.positions[sym].trailing_sl = new_tsl;
                    }

                    // Trailing stop exit
                    if (check_trailing_exit(state.positions[sym], hybrid)) {
                        apply_close(&state, sym, hybrid, false, fee_rate, cfg.slippage_bps);
                        break;
                    }

                    // Take profit
                    float tp_mult = get_tp_mult(hybrid, &cfg);
                    unsigned int tp_result = check_tp(pos, hybrid, &cfg, tp_mult);
                    if (tp_result == 1u) {
                        apply_partial_close(&state, sym, hybrid, cfg.tp_partial_pct, fee_rate, cfg.slippage_bps);
                        // CPU sub-bar semantics keep scanning later sub-bars after a partial TP.
                        // Remaining size can still hit SL/TS/other exits within the same bar window.
                        continue;
                    }
                    if (tp_result == 2u) {
                        apply_close(&state, sym, hybrid, false, fee_rate, cfg.slippage_bps);
                        break;
                    }

                    // Smart exits
                    if (check_smart_exits(pos, hybrid, &cfg, p_atr)) {
                        apply_close(&state, sym, hybrid, false, fee_rate, cfg.slippage_bps);
                        break;
                    }
                }
            }

            // ── Sub-bar entries (per-tick collection + ranking) ───────────────
            // Find max sub-bar count across all symbols for this bar
            unsigned int bar_max_sub = 0u;
            for (unsigned int sym = 0u; sym < ns; sym++) {
                unsigned int c = sub_counts[bar * ns + sym];
                if (c > bar_max_sub) { bar_max_sub = c; }
            }

            for (unsigned int sub_i = 0u; sub_i < bar_max_sub; sub_i++) {
                EntryCandidate candidates[MAX_CANDIDATES];
                unsigned int num_cands = 0u;

                for (unsigned int sym = 0u; sym < ns; sym++) {
                    if (num_cands >= MAX_CANDIDATES) { break; }
                    if (state.entries_this_bar >= cfg.max_entry_orders_per_loop) { break; }

                    // Check if this symbol has a sub-bar at this slot
                    if (sub_i >= sub_counts[bar * ns + sym]) { continue; }

                    const GpuRawCandle& sc = sub_candles[(bar * max_sub + sub_i) * ns + sym];
                    if (sc.close <= 0.0f) { continue; }

                    const GpuSnapshot& ind_snap = snapshots[cfg.snapshot_offset + bar * ns + sym];
                    if (ind_snap.valid == 0u) { continue; }

                    // Build hybrid: indicators from main bar, price from sub-bar
                    GpuSnapshot hybrid = ind_snap;
                    hybrid.close = sc.close;
                    hybrid.high = sc.high;
                    hybrid.low = sc.low;
                    hybrid.open = sc.open;
                    hybrid.t_sec = sc.t_sec;

                    const GpuPosition& pos = state.positions[sym];

                    // Pyramiding (immediate, not ranked)
                    if (pos.active != POS_EMPTY && cfg.enable_pyramiding != 0u) {
                        if (pos.adds_count < cfg.max_adds_per_symbol) {
                            // AQC-1252: confidence gate for pyramid adds
                            if (cfg.add_min_confidence > 0u) {
                                bool is_btc_sym_pyr = (sym == params->btc_sym_idx);
                                GateResult gates_pyr = check_gates(hybrid, &cfg, hybrid.ema_slow_slope_pct,
                                                                    btc_bull, is_btc_sym_pyr);
                                SignalResultLegacy sig_pyr = generate_signal(
                                    hybrid, &cfg, gates_pyr, btc_bull, is_btc_sym_pyr,
                                    hybrid.ema_slow_slope_pct);
                                if (sig_pyr.confidence < cfg.add_min_confidence) {
                                    continue;
                                }
                            }
                            float p_atr_pyr = profit_atr(pos, hybrid.close);
                            if (p_atr_pyr >= cfg.add_min_profit_atr) {
                                unsigned int elapsed_sec = hybrid.t_sec - pos.last_add_time_sec;
                                if (elapsed_sec >= cfg.add_cooldown_minutes * 60u) {
                                    // Equity = balance + unrealized PnL (mirrors CPU)
                                    float pyr_equity = (float)state.balance;
                                    for (unsigned int eq_s2 = 0u; eq_s2 < ns; eq_s2++) {
                                        if (state.positions[eq_s2].active != POS_EMPTY) {
                                            const GpuSnapshot& eq_snap2 = snapshots[cfg.snapshot_offset + bar * ns + eq_s2];
                                            if (eq_snap2.valid != 0u) {
                                                pyr_equity += profit_usd(state.positions[eq_s2], eq_snap2.close);
                                            }
                                        }
                                    }
                                    if (pyr_equity < 0.0f) { pyr_equity = 0.0f; }
                                    float base_margin = pyr_equity * cfg.allocation_pct;
                                    float add_margin = base_margin * cfg.add_fraction_of_base_margin;
                                    float lev = pos.leverage;
                                    float add_notional = add_margin * lev;
                                    float add_size = add_notional / hybrid.close;

                                    if (add_notional < cfg.min_notional_usd) {
                                        if (cfg.bump_to_min_notional != 0u && hybrid.close > 0.0f) {
                                            add_notional = cfg.min_notional_usd;
                                            add_size = add_notional / hybrid.close;
                                        } else {
                                            continue;
                                        }
                                    }

                                    float fee = add_notional * fee_rate;
                                    state.balance -= fee;
                                    state.total_fees += fee;

                                    float old_size = pos.size;
                                    float new_size = old_size + add_size;
                                    float new_entry = (pos.entry_price * old_size + hybrid.close * add_size) / new_size;
                                    state.positions[sym].entry_price = new_entry;
                                    state.positions[sym].size = new_size;
                                    state.positions[sym].margin_used += add_margin;
                                    state.positions[sym].adds_count += 1u;
                                    state.positions[sym].last_add_time_sec = hybrid.t_sec;
                                }
                            }
                        }
                        continue;
                    }

                    if (pos.active != POS_EMPTY) { continue; }

                    // Gates, signal generation, filters (using hybrid snapshot with indicator values)
                    // AQC-1213: gates and signal now use codegen (double precision)
                    bool is_btc_symbol = (sym == params->btc_sym_idx);
                    GateResult gates = check_gates(hybrid, &cfg, hybrid.ema_slow_slope_pct,
                                                    btc_bull, is_btc_symbol);
                    SignalResultLegacy sig_result = generate_signal(
                        hybrid,
                        &cfg,
                        gates,
                        btc_bull,
                        is_btc_symbol,
                        hybrid.ema_slow_slope_pct
                    );
                    unsigned int signal = sig_result.signal;
                    unsigned int confidence = sig_result.confidence;
                    float entry_adx_thresh = sig_result.entry_adx_threshold;

                    if (signal == SIG_NEUTRAL) { continue; }

                    float atr = apply_atr_floor(hybrid.atr, hybrid.close, cfg.min_atr_pct);

                    signal = apply_reverse(signal, &cfg, breadth_pct);
                    if (signal == SIG_NEUTRAL) { continue; }
                    signal = apply_regime_filter(signal, &cfg, breadth_pct);
                    if (signal == SIG_NEUTRAL) { continue; }

                    if (!conf_meets_min(confidence, cfg.entry_min_confidence)) { continue; }

                    unsigned int desired_type = (signal == SIG_BUY) ? POS_LONG : POS_SHORT;
                    if (is_pesc_blocked(&state, sym, desired_type, hybrid.t_sec, hybrid.adx, &cfg)) { continue; }

                    if (cfg.enable_ssf_filter != 0u) {
                        if (signal == SIG_BUY && hybrid.macd_hist <= 0.0f) { continue; }
                        if (signal == SIG_SELL && hybrid.macd_hist >= 0.0f) { continue; }
                    }

                    if (cfg.enable_reef_filter != 0u) {
                        if (signal == SIG_BUY) {
                            if (hybrid.adx < cfg.reef_adx_threshold) {
                                if (hybrid.rsi > cfg.reef_long_rsi_block_gt) { continue; }
                            } else {
                                if (hybrid.rsi > cfg.reef_long_rsi_extreme_gt) { continue; }
                            }
                        }
                        if (signal == SIG_SELL) {
                            if (hybrid.adx < cfg.reef_adx_threshold) {
                                if (hybrid.rsi < cfg.reef_short_rsi_block_lt) { continue; }
                            } else {
                                if (hybrid.rsi < cfg.reef_short_rsi_extreme_lt) { continue; }
                            }
                        }
                    }

                    float conf_rank = (float)(confidence);
                    float score = conf_rank * 100.0f + hybrid.adx;

                    EntryCandidate cand;
                    cand.sym_idx = sym;
                    cand.signal = signal;
                    cand.confidence = confidence;
                    cand.score = score;
                    cand.adx = hybrid.adx;
                    cand.atr = atr;
                    cand.entry_adx_threshold = entry_adx_thresh;
                    candidates[num_cands] = cand;
                    num_cands += 1u;
                }

                // Rank candidates for this sub-bar tick
                for (unsigned int i = 1u; i < num_cands; i++) {
                    EntryCandidate key = candidates[i];
                    unsigned int j = i;
                    while (j > 0u && candidates[j - 1u].score < key.score) {
                        candidates[j] = candidates[j - 1u];
                        j -= 1u;
                    }
                    candidates[j] = key;
                }

                // Compute equity = balance + unrealized PnL (mirrors CPU engine.rs:1094-1095)
                float hybrid_entry_equity = (float)state.balance;
                for (unsigned int eq_s = 0u; eq_s < ns; eq_s++) {
                    if (state.positions[eq_s].active != POS_EMPTY) {
                        const GpuSnapshot& eq_snap = snapshots[cfg.snapshot_offset + bar * ns + eq_s];
                        if (eq_snap.valid != 0u) {
                            hybrid_entry_equity += profit_usd(state.positions[eq_s], eq_snap.close);
                        }
                    }
                }
                if (hybrid_entry_equity < 0.0f) { hybrid_entry_equity = 0.0f; }

                // Execute ranked entries for this sub-bar tick
                for (unsigned int i = 0u; i < num_cands; i++) {
                    if (state.entries_this_bar >= cfg.max_entry_orders_per_loop) { break; }
                    if (state.num_open >= cfg.max_open_positions) { break; }

                    const EntryCandidate& cand = candidates[i];

                    // Re-read sub-bar candle for fill price
                    const GpuRawCandle& sc = sub_candles[(bar * max_sub + sub_i) * ns + cand.sym_idx];
                    const GpuSnapshot& ind_snap = snapshots[cfg.snapshot_offset + bar * ns + cand.sym_idx];

                    GpuSnapshot hybrid = ind_snap;
                    hybrid.close = sc.close;
                    hybrid.high = sc.high;
                    hybrid.low = sc.low;
                    hybrid.open = sc.open;
                    hybrid.t_sec = sc.t_sec;

                    // Margin cap
                    float total_margin = 0.0f;
                    for (unsigned int s = 0u; s < ns; s++) {
                        if (state.positions[s].active != POS_EMPTY) {
                            total_margin += state.positions[s].margin_used;
                        }
                    }
                    float headroom = hybrid_entry_equity * cfg.max_total_margin_pct - total_margin;
                    if (headroom <= 0.0f) { continue; }

                    SizingResult sizing = compute_entry_size(hybrid_entry_equity, hybrid.close, cand.confidence,
                                                             cand.atr, hybrid, &cfg);
                    float size = sizing.size;
                    float margin = sizing.margin;
                    float lev = sizing.leverage;

                    if (margin > headroom) {
                        float ratio = headroom / margin;
                        size *= ratio;
                        margin = headroom;
                    }

                    float notional = size * hybrid.close;
                    if (notional < cfg.min_notional_usd) {
                        if (cfg.bump_to_min_notional != 0u && hybrid.close > 0.0f) {
                            size = cfg.min_notional_usd / hybrid.close;
                            margin = size * hybrid.close / lev;
                            notional = size * hybrid.close;
                        } else {
                            continue;
                        }
                    }

                    float slip = (cand.signal == SIG_BUY) ? cfg.slippage_bps : -cfg.slippage_bps;
                    float fill_price = hybrid.close * (1.0f + slip / 10000.0f);
                    float fee = notional * fee_rate;
                    state.balance -= fee;
                    state.total_fees += fee;

                    GpuPosition new_pos;
                    new_pos.active = (cand.signal == SIG_BUY) ? POS_LONG : POS_SHORT;
                    new_pos.entry_price = fill_price;
                    new_pos.size = size;
                    new_pos.confidence = cand.confidence;
                    new_pos.entry_atr = cand.atr;
                    new_pos.entry_adx_threshold = cand.entry_adx_threshold;
                    new_pos.trailing_sl = 0.0f;
                    new_pos.leverage = lev;
                    new_pos.margin_used = margin;
                    new_pos.adds_count = 0u;
                    new_pos.tp1_taken = 0u;
                    new_pos.open_time_sec = hybrid.t_sec;
                    new_pos.last_add_time_sec = hybrid.t_sec;
                    new_pos._pad[0] = 0u;
                    new_pos._pad[1] = 0u;
                    new_pos._pad[2] = 0u;
                    state.positions[cand.sym_idx] = new_pos;

                    state.num_open += 1u;
                    state.entries_this_bar += 1u;
                }
            }

        } else {
            // ================================================================
            // MAIN-BAR PATH: original behavior (backwards compatible)
            // ================================================================

            // == Phase 1: Exits (per symbol, immediate) ======================
            for (unsigned int sym = 0u; sym < ns; sym++) {
                const GpuSnapshot& snap = snapshots[cfg.snapshot_offset + bar * ns + sym];
                if (snap.valid == 0u) { continue; }

                const GpuPosition& pos = state.positions[sym];
                if (pos.active == POS_EMPTY) { continue; }

                float p_atr = profit_atr(pos, snap.close);

                // Glitch guard (CPU semantics): block ALL exit processing including trailing update.
                bool block_exits = false;
                if (cfg.block_exits_on_extreme_dev != 0u && snap.prev_close > 0.0f) {
                    float price_change_pct = fabsf(snap.close - snap.prev_close) / snap.prev_close;
                    block_exits = (price_change_pct > cfg.glitch_price_dev_pct)
                        || (snap.atr > 0.0f
                            && fabsf(snap.close - snap.prev_close) > snap.atr * cfg.glitch_atr_mult);
                }
                if (block_exits) {
                    continue;  // CPU returns Hold — skip trailing update too
                }

                // Stop loss
                if (check_stop_loss(pos, snap, &cfg)) {
                    apply_close(&state, sym, snap, false, fee_rate, cfg.slippage_bps);
                    continue;
                }

                // Update trailing stop
                float new_tsl = compute_trailing(pos, snap, &cfg, p_atr);
                if (new_tsl > 0.0f) {
                    state.positions[sym].trailing_sl = new_tsl;
                }

                // Trailing stop exit
                if (check_trailing_exit(state.positions[sym], snap)) {
                    apply_close(&state, sym, snap, false, fee_rate, cfg.slippage_bps);
                    continue;
                }

                // Take profit
                float tp_mult = get_tp_mult(snap, &cfg);
                unsigned int tp_result = check_tp(pos, snap, &cfg, tp_mult);
                if (tp_result == 1u) {
                    apply_partial_close(&state, sym, snap, cfg.tp_partial_pct, fee_rate, cfg.slippage_bps);
                    continue;
                }
                if (tp_result == 2u) {
                    apply_close(&state, sym, snap, false, fee_rate, cfg.slippage_bps);
                    continue;
                }

                // Smart exits
                if (check_smart_exits(pos, snap, &cfg, p_atr)) {
                    apply_close(&state, sym, snap, false, fee_rate, cfg.slippage_bps);
                    continue;
                }
            }

            // == Phase 2: Entry collection ===================================
            EntryCandidate candidates[MAX_CANDIDATES];
            unsigned int num_cands = 0u;

            for (unsigned int sym = 0u; sym < ns; sym++) {
                if (num_cands >= MAX_CANDIDATES) { break; }

                const GpuSnapshot& snap = snapshots[cfg.snapshot_offset + bar * ns + sym];
                if (snap.valid == 0u) { continue; }

                const GpuPosition& pos = state.positions[sym];

                // Pyramiding (same-direction add) -- not ranked, immediate
                if (pos.active != POS_EMPTY && cfg.enable_pyramiding != 0u) {
                    if (pos.adds_count < cfg.max_adds_per_symbol) {
                        // AQC-1252: confidence gate for pyramid adds
                        if (cfg.add_min_confidence > 0u) {
                            bool is_btc_sym_pyr = (sym == params->btc_sym_idx);
                            GateResult gates_pyr = check_gates(snap, &cfg, snap.ema_slow_slope_pct,
                                                                btc_bull, is_btc_sym_pyr);
                            SignalResultLegacy sig_pyr = generate_signal(
                                snap, &cfg, gates_pyr, btc_bull, is_btc_sym_pyr,
                                snap.ema_slow_slope_pct);
                            if (sig_pyr.confidence < cfg.add_min_confidence) {
                                continue;
                            }
                        }
                        float p_atr_pyr = profit_atr(pos, snap.close);
                        if (p_atr_pyr >= cfg.add_min_profit_atr) {
                            unsigned int elapsed_sec = snap.t_sec - pos.last_add_time_sec;
                            if (elapsed_sec >= cfg.add_cooldown_minutes * 60u) {
                                float equity = (float)state.balance;
                                float base_margin = equity * cfg.allocation_pct;
                                float add_margin = base_margin * cfg.add_fraction_of_base_margin;
                                float lev = pos.leverage;
                                float add_notional = add_margin * lev;
                                float add_size = add_notional / snap.close;

                                if (add_notional < cfg.min_notional_usd) {
                                    if (cfg.bump_to_min_notional != 0u && snap.close > 0.0f) {
                                        add_notional = cfg.min_notional_usd;
                                        add_size = add_notional / snap.close;
                                    } else {
                                        continue;
                                    }
                                }

                                float fee = add_notional * fee_rate;
                                state.balance -= fee;
                                state.total_fees += fee;

                                float old_size = pos.size;
                                float new_size = old_size + add_size;
                                float new_entry = (pos.entry_price * old_size + snap.close * add_size) / new_size;
                                state.positions[sym].entry_price = new_entry;
                                state.positions[sym].size = new_size;
                                state.positions[sym].margin_used += add_margin;
                                state.positions[sym].adds_count += 1u;
                                state.positions[sym].last_add_time_sec = snap.t_sec;
                            }
                        }
                    }
                    continue;
                }

                if (pos.active != POS_EMPTY) { continue; }

                // AQC-1213: gates and signal now use codegen (double precision)
                bool is_btc_symbol = (sym == params->btc_sym_idx);
                GateResult gates = check_gates(snap, &cfg, snap.ema_slow_slope_pct,
                                                btc_bull, is_btc_symbol);
                SignalResultLegacy sig_result = generate_signal(
                    snap,
                    &cfg,
                    gates,
                    btc_bull,
                    is_btc_symbol,
                    snap.ema_slow_slope_pct
                );
                unsigned int signal = sig_result.signal;
                unsigned int confidence = sig_result.confidence;
                float entry_adx_thresh = sig_result.entry_adx_threshold;

                if (signal == SIG_NEUTRAL) { continue; }

                float atr = apply_atr_floor(snap.atr, snap.close, cfg.min_atr_pct);

                signal = apply_reverse(signal, &cfg, breadth_pct);
                if (signal == SIG_NEUTRAL) { continue; }
                signal = apply_regime_filter(signal, &cfg, breadth_pct);
                if (signal == SIG_NEUTRAL) { continue; }

                if (!conf_meets_min(confidence, cfg.entry_min_confidence)) { continue; }

                unsigned int desired_type = (signal == SIG_BUY) ? POS_LONG : POS_SHORT;
                if (is_pesc_blocked(&state, sym, desired_type, snap.t_sec, snap.adx, &cfg)) { continue; }

                if (cfg.enable_ssf_filter != 0u) {
                    if (signal == SIG_BUY && snap.macd_hist <= 0.0f) { continue; }
                    if (signal == SIG_SELL && snap.macd_hist >= 0.0f) { continue; }
                }

                if (cfg.enable_reef_filter != 0u) {
                    if (signal == SIG_BUY) {
                        if (snap.adx < cfg.reef_adx_threshold) {
                            if (snap.rsi > cfg.reef_long_rsi_block_gt) { continue; }
                        } else {
                            if (snap.rsi > cfg.reef_long_rsi_extreme_gt) { continue; }
                        }
                    }
                    if (signal == SIG_SELL) {
                        if (snap.adx < cfg.reef_adx_threshold) {
                            if (snap.rsi < cfg.reef_short_rsi_block_lt) { continue; }
                        } else {
                            if (snap.rsi < cfg.reef_short_rsi_extreme_lt) { continue; }
                        }
                    }
                }

                float conf_rank = (float)(confidence);
                float score = conf_rank * 100.0f + snap.adx;

                EntryCandidate cand;
                cand.sym_idx = sym;
                cand.signal = signal;
                cand.confidence = confidence;
                cand.score = score;
                cand.adx = snap.adx;
                cand.atr = atr;
                cand.entry_adx_threshold = entry_adx_thresh;
                candidates[num_cands] = cand;
                num_cands += 1u;
            }

            // == Phase 3: Rank candidates ====================================
            for (unsigned int i = 1u; i < num_cands; i++) {
                EntryCandidate key = candidates[i];
                unsigned int j = i;
                while (j > 0u && candidates[j - 1u].score < key.score) {
                    candidates[j] = candidates[j - 1u];
                    j -= 1u;
                }
                candidates[j] = key;
            }

            // == Phase 4: Execute ranked entries =============================
            // Compute equity = balance + unrealized PnL (mirrors CPU engine.rs:1094-1095)
            float main_entry_equity = (float)state.balance;
            for (unsigned int eq_s3 = 0u; eq_s3 < ns; eq_s3++) {
                if (state.positions[eq_s3].active != POS_EMPTY) {
                    const GpuSnapshot& eq_snap3 = snapshots[cfg.snapshot_offset + bar * ns + eq_s3];
                    if (eq_snap3.valid != 0u) {
                        main_entry_equity += profit_usd(state.positions[eq_s3], eq_snap3.close);
                    }
                }
            }
            if (main_entry_equity < 0.0f) { main_entry_equity = 0.0f; }

            for (unsigned int i = 0u; i < num_cands; i++) {
                if (state.entries_this_bar >= cfg.max_entry_orders_per_loop) { break; }
                if (state.num_open >= cfg.max_open_positions) { break; }

                const EntryCandidate& cand = candidates[i];
                const GpuSnapshot& snap = snapshots[cfg.snapshot_offset + bar * ns + cand.sym_idx];

                float total_margin = 0.0f;
                for (unsigned int s = 0u; s < ns; s++) {
                    if (state.positions[s].active != POS_EMPTY) {
                        total_margin += state.positions[s].margin_used;
                    }
                }
                float headroom = main_entry_equity * cfg.max_total_margin_pct - total_margin;
                if (headroom <= 0.0f) { continue; }

                SizingResult sizing = compute_entry_size(main_entry_equity, snap.close, cand.confidence,
                                                         cand.atr, snap, &cfg);
                float size = sizing.size;
                float margin = sizing.margin;
                float lev = sizing.leverage;

                if (margin > headroom) {
                    float ratio = headroom / margin;
                    size *= ratio;
                    margin = headroom;
                }

                float notional = size * snap.close;
                if (notional < cfg.min_notional_usd) {
                    if (cfg.bump_to_min_notional != 0u && snap.close > 0.0f) {
                        size = cfg.min_notional_usd / snap.close;
                        margin = size * snap.close / lev;
                        notional = size * snap.close;
                    } else {
                        continue;
                    }
                }

                float slip = (cand.signal == SIG_BUY) ? cfg.slippage_bps : -cfg.slippage_bps;
                float fill_price = snap.close * (1.0f + slip / 10000.0f);
                float fee = notional * fee_rate;
                state.balance -= fee;
                state.total_fees += fee;

                GpuPosition new_pos;
                new_pos.active = (cand.signal == SIG_BUY) ? POS_LONG : POS_SHORT;
                new_pos.entry_price = fill_price;
                new_pos.size = size;
                new_pos.confidence = cand.confidence;
                new_pos.entry_atr = cand.atr;
                new_pos.entry_adx_threshold = cand.entry_adx_threshold;
                new_pos.trailing_sl = 0.0f;
                new_pos.leverage = lev;
                new_pos.margin_used = margin;
                new_pos.adds_count = 0u;
                new_pos.tp1_taken = 0u;
                new_pos.open_time_sec = snap.t_sec;
                new_pos.last_add_time_sec = snap.t_sec;
                new_pos._pad[0] = 0u;
                new_pos._pad[1] = 0u;
                new_pos._pad[2] = 0u;
                state.positions[cand.sym_idx] = new_pos;

                state.num_open += 1u;
                state.entries_this_bar += 1u;
            }
        } // end if/else max_sub

        // == Equity tracking (double precision) ==================================
        double equity = state.balance;
        for (unsigned int s = 0u; s < ns; s++) {
            const GpuPosition& p = state.positions[s];
            if (p.active != POS_EMPTY) {
                const GpuSnapshot& snap = snapshots[cfg.snapshot_offset + bar * ns + s];
                if (snap.valid != 0u) {
                    equity += (double)profit_usd(p, snap.close);
                }
            }
        }
        if (equity > state.peak_equity) { state.peak_equity = equity; }
        if (state.peak_equity > 0.0) {
            double dd = (state.peak_equity - equity) / state.peak_equity;
            if (dd > state.max_drawdown) { state.max_drawdown = dd; }
        }
    }

    // Write back state
    states[combo_id] = state;

    // Pack result (on last chunk)
    if (params->chunk_end >= params->trade_end_bar) {
        unsigned int total_losses = (state.total_trades >= state.total_wins)
                                    ? (state.total_trades - state.total_wins)
                                    : 0u;
        GpuResult res;
        res.final_balance = (float)state.balance;
        res.total_pnl = (float)state.total_pnl;
        res.total_fees = (float)state.total_fees;
        res.total_trades = state.total_trades;
        res.total_wins = state.total_wins;
        res.total_losses = total_losses;
        res.gross_profit = (float)state.gross_profit;
        res.gross_loss = (float)state.gross_loss;
        res.max_drawdown_pct = (float)state.max_drawdown;
        res._pad[0] = 0u;
        res._pad[1] = 0u;
        res._pad[2] = 0u;
        results[combo_id] = res;
    }
}
