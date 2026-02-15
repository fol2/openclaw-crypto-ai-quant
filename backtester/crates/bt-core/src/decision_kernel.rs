//! Versioned decision kernel types and a deterministic transition function.
//!
//! Invariants:
//! 1) `step(state, event, params)` is pure and deterministic for identical inputs.
//! 2) Every participating payload must use `schema_version == KERNEL_SCHEMA_VERSION`.
//!    A mismatch results in an error and no state change.
//! 3) All monetary and size values are rounded to a stable 1e-12 resolution before
//!    being written into state so that repeated replays remain stable.

use crate::accounting;
use crate::indicators::IndicatorSnapshot;
use crate::kernel_exits::{ExitParams, KernelExitResult};
use crate::signals::gates::GateResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const KERNEL_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MarketSignal {
    Buy,
    Sell,
    Neutral,
    Funding,
    /// Per-bar price tick: kernel evaluates exit conditions for existing positions.
    PriceUpdate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionSide {
    Long,
    Short,
}

impl PositionSide {
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderIntentKind {
    Open,
    Add,
    Close,
    Hold,
    Reverse,
}

/// Incoming signal-like event from market data/feature extraction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarketEvent {
    pub schema_version: u32,
    pub event_id: u64,
    pub timestamp_ms: i64,
    pub symbol: String,
    pub signal: MarketSignal,
    pub price: f64,
    #[serde(default)]
    pub notional_hint_usd: Option<f64>,
    /// Fraction of position to close: `None` or `Some(1.0)` = full close,
    /// `Some(0.5)` = close 50%.  Only meaningful when the event triggers a
    /// close (opposite-side signal with an existing position).
    #[serde(default)]
    pub close_fraction: Option<f64>,
    /// Fee role override: `None` defaults to Taker.
    #[serde(default)]
    pub fee_role: Option<accounting::FeeRole>,
    /// Funding rate for settlement events.  Only meaningful when
    /// `signal == MarketSignal::Funding`.  Positive rate means longs pay shorts.
    #[serde(default)]
    pub funding_rate: Option<f64>,
    /// Indicator snapshot for exit evaluation.  Only meaningful when
    /// `signal == MarketSignal::PriceUpdate`.
    #[serde(default)]
    pub indicators: Option<IndicatorSnapshot>,
    /// Pre-computed gate evaluation result. When present on Buy/Sell events,
    /// the kernel uses this to filter entry intents (block if gates fail).
    #[serde(default)]
    pub gate_result: Option<GateResult>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub side: PositionSide,
    pub quantity: f64,
    pub avg_entry_price: f64,
    pub opened_at_ms: i64,
    pub updated_at_ms: i64,
    pub notional_usd: f64,
    /// Margin (collateral) locked for this position = notional / leverage.
    #[serde(default)]
    pub margin_usd: f64,
    /// Signal confidence at entry (0=Low, 1=Medium, 2=High).
    #[serde(default)]
    pub confidence: Option<u8>,
    /// ATR value at the time of entry, used for exit sizing.
    #[serde(default)]
    pub entry_atr: Option<f64>,
    /// ADX threshold at entry time, used for smart exit exhaustion check.
    #[serde(default)]
    pub entry_adx_threshold: Option<f64>,
    /// Number of ADD (pyramid) fills applied to this position.
    #[serde(default)]
    pub adds_count: u32,
    /// Whether partial take-profit (TP1) has been taken.
    #[serde(default)]
    pub tp1_taken: bool,
    /// Trailing stop-loss price, if active.
    #[serde(default)]
    pub trailing_sl: Option<f64>,
    /// Maximum adverse excursion in USD (worst unrealised drawdown).
    #[serde(default)]
    pub mae_usd: f64,
    /// Maximum favorable excursion in USD (best unrealised profit).
    #[serde(default)]
    pub mfe_usd: f64,
    /// Timestamp (ms) of the last funding settlement applied to this position.
    #[serde(default)]
    pub last_funding_ms: Option<i64>,
}

/// Canonical strategy state persisted between steps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrategyState {
    pub schema_version: u32,
    pub timestamp_ms: i64,
    pub step: u64,
    pub cash_usd: f64,
    pub positions: BTreeMap<String, Position>,
}

impl StrategyState {
    pub fn new(cash_usd: f64, timestamp_ms: i64) -> Self {
        Self {
            schema_version: KERNEL_SCHEMA_VERSION,
            timestamp_ms,
            step: 0,
            cash_usd,
            positions: BTreeMap::new(),
        }
    }
}

/// Canonical order request produced by the kernel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderIntent {
    pub schema_version: u32,
    pub intent_id: u64,
    pub symbol: String,
    pub kind: OrderIntentKind,
    pub side: PositionSide,
    pub quantity: f64,
    pub price: f64,
    pub notional_usd: f64,
    pub fee_rate: f64,
}

/// Canonical fill event produced from intent execution simulation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FillEvent {
    pub schema_version: u32,
    pub intent_id: u64,
    pub symbol: String,
    pub side: PositionSide,
    pub quantity: f64,
    pub price: f64,
    pub notional_usd: f64,
    pub fee_usd: f64,
    pub pnl_usd: f64,
}

/// Per-step diagnostics.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Diagnostics {
    pub schema_version: u32,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub intent_count: usize,
    pub fill_count: usize,
    pub step: u64,
    /// Gate evaluation summary for audit trail.
    #[serde(default)]
    pub gate_blocked: bool,
    /// Which gates blocked entry (if any).
    #[serde(default)]
    pub gate_block_reasons: Vec<String>,
}

impl Diagnostics {
    fn new(step: u64) -> Self {
        Self {
            schema_version: KERNEL_SCHEMA_VERSION,
            step,
            ..Default::default()
        }
    }

    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

/// Output of a deterministic kernel transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionResult {
    pub schema_version: u32,
    pub state: StrategyState,
    pub intents: Vec<OrderIntent>,
    pub fills: Vec<FillEvent>,
    pub diagnostics: Diagnostics,
}

/// Kernel params controlling risk/performance behaviour.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelParams {
    pub schema_version: u32,
    pub default_notional_usd: f64,
    pub min_notional_usd: f64,
    pub max_notional_usd: f64,
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub allow_pyramid: bool,
    pub allow_reverse: bool,
    /// Leverage used for margin calculation.  Kernel tracks margin (= notional /
    /// leverage) in its cash model rather than full notional, matching the
    /// exchange's margined-perpetual accounting.  Default 1.0 (spot-equivalent).
    pub leverage: f64,
    /// Exit evaluation parameters.  When `Some`, PriceUpdate events will evaluate
    /// exit conditions (SL, trailing, TP) for existing positions.
    #[serde(default)]
    pub exit_params: Option<ExitParams>,
}

impl Default for KernelParams {
    fn default() -> Self {
        Self {
            schema_version: KERNEL_SCHEMA_VERSION,
            default_notional_usd: 10_000.0,
            min_notional_usd: 10.0,
            max_notional_usd: 100_000.0,
            maker_fee_bps: accounting::DEFAULT_MAKER_FEE_BPS,
            taker_fee_bps: accounting::DEFAULT_TAKER_FEE_BPS,
            allow_pyramid: true,
            allow_reverse: true,
            leverage: 1.0,
            exit_params: None,
        }
    }
}

fn quantise(value: f64) -> f64 {
    accounting::quantize(value)
}

fn check_versions(
    state: &StrategyState,
    event: &MarketEvent,
    params: &KernelParams,
    diagnostics: &mut Diagnostics,
) {
    if state.schema_version != KERNEL_SCHEMA_VERSION {
        diagnostics.errors.push(format!(
            "Unsupported StrategyState schema version {}",
            state.schema_version
        ));
    }
    if event.schema_version != KERNEL_SCHEMA_VERSION {
        diagnostics.errors.push(format!(
            "Unsupported MarketEvent schema version {}",
            event.schema_version
        ));
    }
    if params.schema_version != KERNEL_SCHEMA_VERSION {
        diagnostics.errors.push(format!(
            "Unsupported KernelParams schema version {}",
            params.schema_version
        ));
    }
}

fn side_from_signal(signal: MarketSignal) -> Option<PositionSide> {
    match signal {
        MarketSignal::Buy => Some(PositionSide::Long),
        MarketSignal::Sell => Some(PositionSide::Short),
        MarketSignal::Neutral | MarketSignal::Funding | MarketSignal::PriceUpdate => None,
    }
}

fn with_intent_id(step: u64, offset: u64) -> u64 {
    step.saturating_mul(1000).saturating_add(offset)
}

fn apply_open(
    state: &mut StrategyState,
    symbol: &str,
    side: PositionSide,
    notional: f64,
    price: f64,
    fee_rate: f64,
    leverage: f64,
    timestamp_ms: i64,
    intent_id: u64,
    kind: OrderIntentKind,
    diagnostics: &mut Diagnostics,
) -> Option<(OrderIntent, FillEvent)> {
    if notional <= 0.0 {
        diagnostics
            .warnings
            .push(format!("skip open for {symbol}: non-positive notional"));
        return None;
    }
    if price <= 0.0 {
        diagnostics
            .warnings
            .push(format!("skip open for {symbol}: non-positive price"));
        return None;
    }

    let open = accounting::apply_open_fill(notional, fee_rate);
    let effective_leverage = leverage.max(1.0);
    let margin = quantise(open.notional / effective_leverage);

    // Margin-based cash check: only the collateral (margin) is locked, not full notional.
    if margin + open.fee_usd > state.cash_usd {
        diagnostics
            .warnings
            .push(format!("skip open for {symbol}: insufficient cash"));
        return None;
    }

    let quantity = quantise(open.notional / price);
    let notional = open.notional;

    if quantity <= 0.0 {
        diagnostics
            .warnings
            .push(format!("skip open for {symbol}: computed quantity is zero"));
        return None;
    }

    // Deduct margin + fee from cash (not full notional).
    state.cash_usd = quantise(state.cash_usd - margin - open.fee_usd);
    if let Some(existing) = state.positions.get_mut(symbol) {
        if existing.side == side {
            // Weighted average with additional stake.
            let existing_notional = existing.notional_usd;
            let added_notional = notional;
            let total_notional = existing_notional + added_notional;
            let existing_qty = existing.quantity;
            let total_qty = existing_qty + quantity;
            let avg_entry = if existing_notional <= 0.0 {
                price
            } else {
                (existing.avg_entry_price * existing_qty + price * quantity) / total_qty
            };
            existing.notional_usd = accounting::quantize(total_notional);
            existing.margin_usd = accounting::quantize(existing.margin_usd + margin);
            existing.quantity = accounting::quantize(total_qty);
            existing.avg_entry_price = accounting::quantize(avg_entry);
            existing.updated_at_ms = timestamp_ms;
            existing.adds_count = existing.adds_count.saturating_add(1);
        } else {
            return None;
        }
    } else {
        state.positions.insert(
            symbol.to_string(),
            Position {
                symbol: symbol.to_string(),
                side,
                quantity,
                avg_entry_price: quantise(price),
                opened_at_ms: timestamp_ms,
                updated_at_ms: timestamp_ms,
                notional_usd: notional,
                margin_usd: margin,
                confidence: None,
                entry_atr: None,
                entry_adx_threshold: None,
                adds_count: 0,
                tp1_taken: false,
                trailing_sl: None,
                mae_usd: 0.0,
                mfe_usd: 0.0,
                last_funding_ms: None,
            },
        );
    }

    let intent = OrderIntent {
        schema_version: KERNEL_SCHEMA_VERSION,
        intent_id,
        symbol: symbol.to_string(),
        kind,
        side,
        quantity,
        price: quantise(price),
        notional_usd: notional,
        fee_rate,
    };
    let fill = FillEvent {
        schema_version: KERNEL_SCHEMA_VERSION,
        intent_id,
        symbol: symbol.to_string(),
        side,
        quantity,
        price: quantise(price),
        notional_usd: notional,
        fee_usd: open.fee_usd,
        pnl_usd: 0.0,
    };

    Some((intent, fill))
}

fn apply_close(
    state: &mut StrategyState,
    symbol: &str,
    side: PositionSide,
    price: f64,
    fee_rate: f64,
    close_fraction: Option<f64>,
    intent_id: u64,
    diagnostics: &mut Diagnostics,
) -> Option<(OrderIntent, FillEvent)> {
    let position = match state.positions.get(symbol) {
        Some(pos) => pos.clone(),
        None => {
            diagnostics
                .warnings
                .push(format!("close skipped for {symbol}: no position"));
            return None;
        }
    };

    if position.quantity <= 0.0 {
        diagnostics
            .warnings
            .push(format!("close skipped for {symbol}: position quantity not positive"));
        return None;
    }
    if price <= 0.0 {
        diagnostics
            .warnings
            .push(format!("close skipped for {symbol}: non-positive price"));
        return None;
    }

    // Determine effective fraction: None or >= 1.0 means full close.
    let frac = close_fraction.unwrap_or(1.0).clamp(0.0, 1.0);
    let is_full_close = frac >= 1.0 - 1e-15;

    let plan = accounting::build_partial_close_plan(position.quantity, position.margin_usd, frac);

    if plan.closed_size <= 0.0 {
        diagnostics
            .warnings
            .push(format!("close skipped for {symbol}: closed size is zero"));
        return None;
    }

    let close_qty = plan.closed_size;
    let close = accounting::apply_close_fill(
        side == PositionSide::Long,
        position.avg_entry_price,
        price,
        close_qty,
        fee_rate,
    );

    // Return proportional margin + PnL - fee to cash.
    let returned_margin = quantise(position.margin_usd - plan.remaining_margin);
    state.cash_usd = quantise(state.cash_usd + returned_margin + close.pnl - close.fee_usd);

    if is_full_close {
        state.positions.remove(symbol);
    } else {
        // Update position in-place with reduced values.
        let pos = state.positions.get_mut(symbol).unwrap();
        pos.quantity = plan.remaining_size;
        pos.margin_usd = plan.remaining_margin;
        pos.notional_usd = quantise(pos.avg_entry_price * plan.remaining_size);
        pos.updated_at_ms = state.timestamp_ms;
    }

    let intent = OrderIntent {
        schema_version: KERNEL_SCHEMA_VERSION,
        intent_id,
        symbol: symbol.to_string(),
        kind: OrderIntentKind::Close,
        side,
        quantity: close_qty,
        price: quantise(price),
        notional_usd: close.notional,
        fee_rate,
    };
    let fill = FillEvent {
        schema_version: KERNEL_SCHEMA_VERSION,
        intent_id,
        symbol: symbol.to_string(),
        side,
        quantity: close_qty,
        price: quantise(price),
        notional_usd: close.notional,
        fee_usd: close.fee_usd,
        pnl_usd: close.pnl,
    };
    Some((intent, fill))
}

/// Execute one deterministic state transition.
///
/// This function is a pure transformation over `(state, event, params)` with no
/// side effects and deterministic outputs.
pub fn step(state: &StrategyState, event: &MarketEvent, params: &KernelParams) -> DecisionResult {
    let mut diagnostics = Diagnostics::new(state.step.saturating_add(1));
    check_versions(state, event, params, &mut diagnostics);
    if diagnostics.has_errors() {
        return DecisionResult {
            schema_version: KERNEL_SCHEMA_VERSION,
            state: state.clone(),
            intents: vec![],
            fills: vec![],
            diagnostics,
        };
    }

    // ---- Funding settlement: cash-only adjustment, no order intents. ----
    if event.signal == MarketSignal::Funding {
        let mut next_state = state.clone();
        next_state.step = next_state.step.saturating_add(1);
        next_state.timestamp_ms = event.timestamp_ms;

        if let Some(rate) = event.funding_rate {
            if rate != 0.0 {
                if let Some(pos) = next_state.positions.get_mut(&event.symbol) {
                    let is_long = pos.side == PositionSide::Long;
                    let delta =
                        accounting::funding_delta(is_long, pos.quantity, event.price, rate);
                    next_state.cash_usd = quantise(next_state.cash_usd + delta);
                    pos.last_funding_ms = Some(event.timestamp_ms);
                    pos.updated_at_ms = event.timestamp_ms;
                }
            }
        }

        diagnostics.intent_count = 0;
        diagnostics.fill_count = 0;
        return DecisionResult {
            schema_version: KERNEL_SCHEMA_VERSION,
            state: next_state,
            intents: vec![],
            fills: vec![],
            diagnostics,
        };
    }

    // ---- PriceUpdate: evaluate exit conditions for existing positions. ----
    if event.signal == MarketSignal::PriceUpdate {
        let mut next_state = state.clone();
        next_state.step = next_state.step.saturating_add(1);
        next_state.timestamp_ms = event.timestamp_ms;

        let mut intents = Vec::new();
        let mut fills = Vec::new();

        if let (Some(exit_params), Some(snap)) = (&params.exit_params, &event.indicators) {
            // Evaluate exits, then act on the result (two-phase to satisfy borrow checker).
            let exit_result = {
                if let Some(pos) = next_state.positions.get_mut(&event.symbol) {
                    Some(crate::kernel_exits::evaluate_exits(
                        pos,
                        snap,
                        exit_params,
                        event.timestamp_ms,
                    ))
                } else {
                    None
                }
            };

            if let Some(result) = exit_result {
                let fee_model = accounting::FeeModel {
                    maker_fee_bps: params.maker_fee_bps,
                    taker_fee_bps: params.taker_fee_bps,
                };
                let role = event.fee_role.unwrap_or(accounting::FeeRole::Taker);
                let fee_rate = fee_model.role_rate(role);
                let close_id = with_intent_id(next_state.step, 1);

                match result {
                    KernelExitResult::Hold => {}
                    KernelExitResult::FullClose { exit_price, .. } => {
                        if let Some(pos) = next_state.positions.get(&event.symbol) {
                            let closed_side = pos.side;
                            if let Some((intent, fill)) = apply_close(
                                &mut next_state,
                                &event.symbol,
                                closed_side,
                                exit_price,
                                fee_rate,
                                Some(1.0),
                                close_id,
                                &mut diagnostics,
                            ) {
                                intents.push(intent);
                                fills.push(fill);
                            }
                        }
                    }
                    KernelExitResult::PartialClose {
                        exit_price,
                        fraction,
                        ..
                    } => {
                        if let Some(pos) = next_state.positions.get(&event.symbol) {
                            let closed_side = pos.side;
                            if let Some((intent, fill)) = apply_close(
                                &mut next_state,
                                &event.symbol,
                                closed_side,
                                exit_price,
                                fee_rate,
                                Some(fraction),
                                close_id,
                                &mut diagnostics,
                            ) {
                                intents.push(intent);
                                fills.push(fill);
                                // Mark tp1_taken after successful partial close.
                                if let Some(pos) = next_state.positions.get_mut(&event.symbol) {
                                    pos.tp1_taken = true;
                                    // Lock trailing_sl to at least entry (breakeven).
                                    let entry = pos.avg_entry_price;
                                    match pos.side {
                                        PositionSide::Long => {
                                            pos.trailing_sl = Some(match pos.trailing_sl {
                                                Some(prev) => prev.max(entry),
                                                None => entry,
                                            });
                                        }
                                        PositionSide::Short => {
                                            pos.trailing_sl = Some(match pos.trailing_sl {
                                                Some(prev) => prev.min(entry),
                                                None => entry,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        diagnostics.intent_count = intents.len();
        diagnostics.fill_count = fills.len();
        return DecisionResult {
            schema_version: KERNEL_SCHEMA_VERSION,
            state: next_state,
            intents,
            fills,
            diagnostics,
        };
    }

    let requested_side = match side_from_signal(event.signal) {
        Some(side) => side,
        None => {
            return DecisionResult {
                schema_version: KERNEL_SCHEMA_VERSION,
                state: state.clone(),
                intents: vec![],
                fills: vec![],
                diagnostics,
            };
        }
    };

    let mut next_state = state.clone();
    next_state.step = next_state.step.saturating_add(1);
    next_state.timestamp_ms = event.timestamp_ms;

    // --- Gate pre-filter for entry decisions ---
    if let Some(ref gr) = event.gate_result {
        let mut blocked_reasons = Vec::new();

        // Core gates
        if !gr.all_gates_pass {
            if gr.is_ranging {
                blocked_reasons.push("ranging".to_string());
            }
            if gr.is_anomaly {
                blocked_reasons.push("anomaly".to_string());
            }
            if gr.is_extended {
                blocked_reasons.push("extension".to_string());
            }
            if !gr.adx_above_min {
                blocked_reasons.push("adx_low".to_string());
            }
            if !gr.is_trending_up {
                blocked_reasons.push("adx_not_rising".to_string());
            }
            if !gr.vol_confirm {
                blocked_reasons.push("volume".to_string());
            }
        }

        // Directional alignment checks
        match requested_side {
            PositionSide::Long => {
                if !gr.bullish_alignment {
                    blocked_reasons.push("bearish_alignment".to_string());
                }
                if !gr.btc_ok_long {
                    blocked_reasons.push("btc_alignment".to_string());
                }
            }
            PositionSide::Short => {
                if !gr.bearish_alignment {
                    blocked_reasons.push("bullish_alignment".to_string());
                }
                if !gr.btc_ok_short {
                    blocked_reasons.push("btc_alignment".to_string());
                }
            }
        }

        if !blocked_reasons.is_empty() {
            let has_position = next_state.positions.contains_key(&event.symbol);
            let is_same_side = next_state
                .positions
                .get(&event.symbol)
                .map_or(false, |p| p.side == requested_side);

            // Block new entries (no existing position) and same-side pyramids
            if !has_position || is_same_side {
                diagnostics.gate_blocked = true;
                diagnostics.gate_block_reasons = blocked_reasons;
                diagnostics.intent_count = 0;
                diagnostics.fill_count = 0;
                return DecisionResult {
                    schema_version: KERNEL_SCHEMA_VERSION,
                    state: next_state,
                    intents: vec![],
                    fills: vec![],
                    diagnostics,
                };
            }

            // Opposite-side: allow close/reverse but record reasons for audit
            diagnostics.gate_block_reasons = blocked_reasons;
        }
    }

    let fee_model = accounting::FeeModel {
        maker_fee_bps: params.maker_fee_bps,
        taker_fee_bps: params.taker_fee_bps,
    };
    let role = event.fee_role.unwrap_or(accounting::FeeRole::Taker);
    let fee_rate = fee_model.role_rate(role);
    let leverage = params.leverage;
    let notional = event
        .notional_hint_usd
        .unwrap_or(params.default_notional_usd)
        .clamp(params.min_notional_usd, params.max_notional_usd);
    let mut intents = Vec::with_capacity(2);
    let mut fills = Vec::with_capacity(2);
    let open_id = with_intent_id(next_state.step, 1);
    let close_id = with_intent_id(next_state.step, 2);
    let reverse_id = with_intent_id(next_state.step, 3);

    let existing = next_state.positions.get(&event.symbol).cloned();
    match existing {
        None => {
            if let Some((intent, fill)) = apply_open(
                &mut next_state,
                &event.symbol,
                requested_side,
                notional,
                event.price,
                fee_rate,
                leverage,
                event.timestamp_ms,
                open_id,
                OrderIntentKind::Open,
                &mut diagnostics,
            ) {
                intents.push(intent);
                fills.push(fill);
            }
        }
        Some(position) if position.side == requested_side => {
            if params.allow_pyramid {
                if let Some((intent, fill)) = apply_open(
                    &mut next_state,
                    &event.symbol,
                    requested_side,
                    notional,
                    event.price,
                    fee_rate,
                    leverage,
                    event.timestamp_ms,
                    open_id,
                    OrderIntentKind::Add,
                    &mut diagnostics,
                ) {
                    intents.push(intent);
                    fills.push(fill);
                }
            } else {
                diagnostics.warnings.push(format!(
                    "ignore same-side signal for {}: pyramiding disabled",
                    event.symbol
                ));
            }
        }
        Some(position) => {
            let closed_side = position.side;
            if let Some((intent, fill)) = apply_close(
                &mut next_state,
                &event.symbol,
                closed_side,
                event.price,
                fee_rate,
                event.close_fraction,
                close_id,
                &mut diagnostics,
            ) {
                intents.push(OrderIntent {
                    kind: if params.allow_reverse {
                        OrderIntentKind::Reverse
                    } else {
                        OrderIntentKind::Close
                    },
                    ..intent
                });
                fills.push(fill);
            }

            if params.allow_reverse {
                if let Some((intent, fill)) = apply_open(
                    &mut next_state,
                    &event.symbol,
                    requested_side,
                    notional,
                    event.price,
                    fee_rate,
                    leverage,
                    event.timestamp_ms,
                    reverse_id,
                    OrderIntentKind::Open,
                    &mut diagnostics,
                ) {
                    intents.push(intent);
                    fills.push(fill);
                }
            }
        }
    }

    next_state.cash_usd = quantise(next_state.cash_usd);
    next_state.timestamp_ms = event.timestamp_ms;
    diagnostics.intent_count = intents.len();
    diagnostics.fill_count = fills.len();

    DecisionResult {
        schema_version: KERNEL_SCHEMA_VERSION,
        state: next_state,
        intents,
        fills,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_state() -> StrategyState {
        StrategyState::new(100_000.0, 0)
    }

    fn event_with_signal(symbol: &str, signal: MarketSignal) -> MarketEvent {
        let notional_hint = match symbol {
            "BTC" => Some(5_000.0),
            _ => Some(10_000.0),
        };

        MarketEvent {
            schema_version: KERNEL_SCHEMA_VERSION,
            event_id: 100,
            timestamp_ms: 1_000,
            symbol: symbol.to_string(),
            signal,
            price: 10_000.0,
            notional_hint_usd: notional_hint,
            close_fraction: None,
            fee_role: None,
            funding_rate: None,
            indicators: None,
            gate_result: None,
        }
    }

    #[test]
    fn deterministic_state_transition() {
        let state = init_state();
        let params = KernelParams::default();
        let event = event_with_signal("BTC", MarketSignal::Buy);

        let first = step(&state, &event, &params);
        let second = step(&state, &event, &params);

        assert_eq!(first, second);
        assert_eq!(first.intents.len(), 1);
        assert_eq!(first.fills.len(), 1);
        assert!(!first.diagnostics.has_errors());
        assert_eq!(first.state.positions.len(), 1);
    }

    #[test]
    fn state_transition_updates_position_and_cash() {
        let state = init_state();
        let params = KernelParams {
            default_notional_usd: 10_000.0,
            min_notional_usd: 100.0,
            max_notional_usd: 10_000.0,
            taker_fee_bps: accounting::DEFAULT_TAKER_FEE_BPS,
            maker_fee_bps: accounting::DEFAULT_MAKER_FEE_BPS,
            ..KernelParams::default()
        };
        let mut event = event_with_signal("BTC", MarketSignal::Buy);
        event.notional_hint_usd = Some(10_000.0);

        let result = step(&state, &event, &params);
        let expected_cost = 10_000.0 * (1.0 + accounting::DEFAULT_TAKER_FEE_RATE);
        let expected_qty = 10_000.0 / 10_000.0;

        assert_eq!(result.state.positions.len(), 1);
        let pos = result.state.positions.get("BTC").expect("position exists");
        assert_eq!(pos.side, PositionSide::Long);
        assert!((pos.quantity - expected_qty).abs() < 1e-12);
        assert!((result.state.cash_usd - (100_000.0 - expected_cost)).abs() < 1e-12);
    }

    #[test]
    fn round_trip_close_pnl_matches_margin_accounting() {
        let initial_state = init_state();
        let params = KernelParams {
            // Disable reverse so the Sell signal only closes, not re-opens.
            allow_reverse: false,
            ..KernelParams::default() // leverage = 1.0
        };
        let mut open_event = event_with_signal("BTC", MarketSignal::Buy);
        open_event.notional_hint_usd = Some(10_000.0);

        let open_result = step(&initial_state, &open_event, &params);
        let open_fill = accounting::apply_open_fill(10_000.0, accounting::DEFAULT_TAKER_FEE_RATE);
        let margin = accounting::quantize(10_000.0 / params.leverage.max(1.0));
        // Kernel deducts margin + fee from cash.
        let expected_cash_after_open = accounting::quantize(initial_state.cash_usd - margin - open_fill.fee_usd);
        assert!(
            (open_result.state.cash_usd - expected_cash_after_open).abs() < 1e-12,
            "open cash: {} != expected {}",
            open_result.state.cash_usd,
            expected_cash_after_open,
        );

        let close_event = MarketEvent {
            schema_version: 1,
            event_id: 101,
            timestamp_ms: 2_000,
            symbol: "BTC".to_string(),
            signal: MarketSignal::Sell,
            price: 10_200.0,
            notional_hint_usd: None,
            close_fraction: None,
            fee_role: None,
            funding_rate: None,
            indicators: None,
            gate_result: None,
        };
        let close_state = open_result.state;
        let close_result = step(&close_state, &close_event, &params);

        let expected_close = accounting::apply_close_fill(true, 10_000.0, 10_200.0, 1.0, accounting::DEFAULT_TAKER_FEE_RATE);
        // Round-trip: initial - margin - open_fee + margin + pnl - close_fee = initial + pnl - total_fees
        let expected_cash = accounting::quantize(
            initial_state.cash_usd + expected_close.pnl - open_fill.fee_usd - expected_close.fee_usd,
        );
        assert!(
            (close_result.state.cash_usd - expected_cash).abs() < 1e-12,
            "close cash: {} != expected {}",
            close_result.state.cash_usd,
            expected_cash,
        );
        assert!((close_result.fills[0].pnl_usd - expected_close.pnl).abs() < 1e-12);
    }

    #[test]
    fn schema_versioning_rejects_mismatch() {
        let state = init_state();
        let params = KernelParams::default();
        let mut event = event_with_signal("BTC", MarketSignal::Buy);
        event.schema_version = 2;

        let result = step(&state, &event, &params);

        assert!(result.diagnostics.has_errors());
        assert_eq!(result.state, state);
        assert_eq!(result.intents.len(), 0);
    }

    // ---- Partial close tests ----

    /// Helper: open a long BTC position with given notional and return resulting state.
    fn open_long_btc(notional: f64, price: f64) -> StrategyState {
        let state = init_state();
        let params = KernelParams {
            allow_reverse: false,
            ..KernelParams::default()
        };
        let mut event = event_with_signal("BTC", MarketSignal::Buy);
        event.notional_hint_usd = Some(notional);
        event.price = price;
        step(&state, &event, &params).state
    }

    fn partial_close_event(fraction: f64, price: f64) -> MarketEvent {
        MarketEvent {
            schema_version: KERNEL_SCHEMA_VERSION,
            event_id: 200,
            timestamp_ms: 2_000,
            symbol: "BTC".to_string(),
            signal: MarketSignal::Sell,
            price,
            notional_hint_usd: None,
            close_fraction: Some(fraction),
            fee_role: None,
            funding_rate: None,
            indicators: None,
            gate_result: None,
        }
    }

    #[test]
    fn partial_close_50_pct_keeps_position() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = KernelParams {
            allow_reverse: false,
            ..KernelParams::default()
        };
        let pos_before = state.positions.get("BTC").unwrap().clone();
        let event = partial_close_event(0.5, 10_200.0);

        let result = step(&state, &event, &params);

        // Position should still exist with halved quantity.
        let pos = result.state.positions.get("BTC").expect("position remains");
        assert!((pos.quantity - accounting::quantize(pos_before.quantity * 0.5)).abs() < 1e-12);
        assert!((pos.margin_usd - accounting::quantize(pos_before.margin_usd * 0.5)).abs() < 1e-12);
        assert_eq!(pos.side, PositionSide::Long);

        // Fill should reflect partial quantity.
        assert_eq!(result.fills.len(), 1);
        assert!((result.fills[0].quantity - accounting::quantize(pos_before.quantity * 0.5)).abs() < 1e-12);
        assert!(result.fills[0].pnl_usd > 0.0); // price rose
    }

    #[test]
    fn partial_close_25_pct() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = KernelParams {
            allow_reverse: false,
            ..KernelParams::default()
        };
        let pos_before = state.positions.get("BTC").unwrap().clone();
        let event = partial_close_event(0.25, 10_000.0);

        let result = step(&state, &event, &params);

        let pos = result.state.positions.get("BTC").expect("position remains");
        assert!((pos.quantity - accounting::quantize(pos_before.quantity * 0.75)).abs() < 1e-12);
        assert!((pos.margin_usd - accounting::quantize(pos_before.margin_usd * 0.75)).abs() < 1e-12);
    }

    #[test]
    fn partial_close_75_pct() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = KernelParams {
            allow_reverse: false,
            ..KernelParams::default()
        };
        let pos_before = state.positions.get("BTC").unwrap().clone();
        let event = partial_close_event(0.75, 9_800.0);

        let result = step(&state, &event, &params);

        let pos = result.state.positions.get("BTC").expect("position remains");
        assert!((pos.quantity - accounting::quantize(pos_before.quantity * 0.25)).abs() < 1e-12);
        assert!((pos.margin_usd - accounting::quantize(pos_before.margin_usd * 0.25)).abs() < 1e-12);
        // Price dropped → PnL should be negative.
        assert!(result.fills[0].pnl_usd < 0.0);
    }

    #[test]
    fn partial_close_100_pct_removes_position() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = KernelParams {
            allow_reverse: false,
            ..KernelParams::default()
        };
        let event = partial_close_event(1.0, 10_100.0);

        let result = step(&state, &event, &params);

        assert!(
            result.state.positions.get("BTC").is_none(),
            "full close should remove position"
        );
        assert_eq!(result.fills.len(), 1);
    }

    #[test]
    fn partial_close_none_fraction_is_full_close() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = KernelParams {
            allow_reverse: false,
            ..KernelParams::default()
        };
        let mut event = partial_close_event(1.0, 10_100.0);
        event.close_fraction = None;

        let result = step(&state, &event, &params);

        assert!(
            result.state.positions.get("BTC").is_none(),
            "None fraction should behave as full close"
        );
    }

    #[test]
    fn partial_close_cash_accounting_round_trip() {
        let initial_cash = 100_000.0;
        let entry_price = 10_000.0;
        let exit_price = 10_400.0;
        let notional = 10_000.0;
        let frac = 0.5;
        let fee_rate = accounting::DEFAULT_TAKER_FEE_RATE;

        let state = open_long_btc(notional, entry_price);
        let cash_after_open = state.cash_usd;
        let pos = state.positions.get("BTC").unwrap();
        let full_qty = pos.quantity;
        let full_margin = pos.margin_usd;

        let params = KernelParams {
            allow_reverse: false,
            ..KernelParams::default()
        };
        let event = partial_close_event(frac, exit_price);
        let result = step(&state, &event, &params);

        // Manually compute expected cash after partial close.
        let close_qty = accounting::quantize(full_qty * frac);
        let returned_margin = accounting::quantize(full_margin * frac);
        let close_acc = accounting::apply_close_fill(true, entry_price, exit_price, close_qty, fee_rate);
        let expected_cash = accounting::quantize(cash_after_open + returned_margin + close_acc.pnl - close_acc.fee_usd);

        assert!(
            (result.state.cash_usd - expected_cash).abs() < 1e-12,
            "partial close cash: {} != expected {}",
            result.state.cash_usd,
            expected_cash,
        );

        // Close the remaining 50%.
        let event2 = MarketEvent {
            event_id: 300,
            timestamp_ms: 3_000,
            close_fraction: Some(1.0),
            price: exit_price,
            ..event
        };
        let result2 = step(&result.state, &event2, &params);
        assert!(result2.state.positions.get("BTC").is_none());

        // Total round-trip: initial_cash + total_pnl - total_fees.
        let open_fee = accounting::apply_open_fill(notional, fee_rate).fee_usd;
        let total_pnl = accounting::mark_to_market_pnl(true, entry_price, exit_price, full_qty);
        let close1_fee = close_acc.fee_usd;
        let close2_qty = accounting::quantize(full_qty - close_qty);
        let close2_fee = accounting::apply_close_fill(true, entry_price, exit_price, close2_qty, fee_rate).fee_usd;
        let expected_final = accounting::quantize(initial_cash + total_pnl - open_fee - close1_fee - close2_fee);

        assert!(
            (result2.state.cash_usd - expected_final).abs() < 1e-12,
            "round-trip cash: {} != expected {}",
            result2.state.cash_usd,
            expected_final,
        );
    }

    #[test]
    fn partial_close_deterministic() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = KernelParams {
            allow_reverse: false,
            ..KernelParams::default()
        };
        let event = partial_close_event(0.5, 10_200.0);

        let r1 = step(&state, &event, &params);
        let r2 = step(&state, &event, &params);
        assert_eq!(r1, r2, "partial close must be deterministic");
    }

    // ---- ADD weighted-average entry price tests ----

    fn pyramid_params() -> KernelParams {
        KernelParams {
            allow_pyramid: true,
            allow_reverse: false,
            ..KernelParams::default()
        }
    }

    #[test]
    fn add_recalculates_weighted_avg_entry_price() {
        // Open at 10_000, then ADD at 12_000. Expect weighted average.
        let state = init_state();
        let params = pyramid_params();

        // Step 1: open long BTC @ 10_000, notional 10_000 → qty = 1.0
        let mut open_evt = event_with_signal("BTC", MarketSignal::Buy);
        open_evt.notional_hint_usd = Some(10_000.0);
        open_evt.price = 10_000.0;
        let r1 = step(&state, &open_evt, &params);
        let pos1 = r1.state.positions.get("BTC").unwrap();
        assert!((pos1.avg_entry_price - 10_000.0).abs() < 1e-12);
        assert!((pos1.quantity - 1.0).abs() < 1e-12);

        // Step 2: ADD long BTC @ 12_000, notional 12_000 → qty = 1.0
        let mut add_evt = MarketEvent {
            event_id: 101,
            timestamp_ms: 2_000,
            price: 12_000.0,
            notional_hint_usd: Some(12_000.0),
            signal: MarketSignal::Buy,
            close_fraction: None,
            ..open_evt
        };
        add_evt.symbol = "BTC".to_string();
        let r2 = step(&r1.state, &add_evt, &params);

        let pos2 = r2.state.positions.get("BTC").unwrap();
        // Weighted avg = (10_000 * 1.0 + 12_000 * 1.0) / (1.0 + 1.0) = 11_000
        let expected_avg = accounting::quantize(
            (10_000.0 * pos1.quantity + 12_000.0 * 1.0) / (pos1.quantity + 1.0),
        );
        assert!(
            (pos2.avg_entry_price - expected_avg).abs() < 1e-12,
            "avg_entry: {} != expected {}",
            pos2.avg_entry_price,
            expected_avg,
        );
        // Total qty should be 2.0
        assert!((pos2.quantity - 2.0).abs() < 1e-12);
        assert_eq!(r2.intents[0].kind, OrderIntentKind::Add);
    }

    #[test]
    fn add_at_lower_price_brings_avg_down() {
        // Open at 10_000, ADD at 8_000 — avg should decrease.
        let state = init_state();
        let params = pyramid_params();

        let mut open_evt = event_with_signal("BTC", MarketSignal::Buy);
        open_evt.notional_hint_usd = Some(10_000.0);
        open_evt.price = 10_000.0;
        let r1 = step(&state, &open_evt, &params);
        let pos1 = r1.state.positions.get("BTC").unwrap();
        let qty1 = pos1.quantity;

        // ADD at 8_000, notional 8_000 → qty = 1.0
        let add_evt = MarketEvent {
            schema_version: KERNEL_SCHEMA_VERSION,
            event_id: 102,
            timestamp_ms: 3_000,
            symbol: "BTC".to_string(),
            signal: MarketSignal::Buy,
            price: 8_000.0,
            notional_hint_usd: Some(8_000.0),
            close_fraction: None,
            fee_role: None,
            funding_rate: None,
            indicators: None,
            gate_result: None,
        };
        let r2 = step(&r1.state, &add_evt, &params);
        let pos2 = r2.state.positions.get("BTC").unwrap();

        let add_qty = accounting::quantize(8_000.0 / 8_000.0); // 1.0
        let expected_avg = accounting::quantize(
            (10_000.0 * qty1 + 8_000.0 * add_qty) / (qty1 + add_qty),
        );
        assert!(
            (pos2.avg_entry_price - expected_avg).abs() < 1e-12,
            "avg_entry after lower add: {} != expected {}",
            pos2.avg_entry_price,
            expected_avg,
        );
        assert!(pos2.avg_entry_price < 10_000.0, "avg should decrease");
    }

    #[test]
    fn add_accumulates_notional_and_margin() {
        let state = init_state();
        let params = pyramid_params();
        let fee_rate = accounting::DEFAULT_TAKER_FEE_RATE;

        let mut open_evt = event_with_signal("BTC", MarketSignal::Buy);
        open_evt.notional_hint_usd = Some(10_000.0);
        open_evt.price = 10_000.0;
        let r1 = step(&state, &open_evt, &params);
        let pos1 = r1.state.positions.get("BTC").unwrap();
        let notional1 = pos1.notional_usd;
        let margin1 = pos1.margin_usd;

        // ADD with notional 5_000
        let add_evt = MarketEvent {
            schema_version: KERNEL_SCHEMA_VERSION,
            event_id: 103,
            timestamp_ms: 4_000,
            symbol: "BTC".to_string(),
            signal: MarketSignal::Buy,
            price: 11_000.0,
            notional_hint_usd: Some(5_000.0),
            close_fraction: None,
            fee_role: None,
            funding_rate: None,
            indicators: None,
            gate_result: None,
        };
        let r2 = step(&r1.state, &add_evt, &params);
        let pos2 = r2.state.positions.get("BTC").unwrap();

        // Notional should be cumulative.
        let add_notional = accounting::apply_open_fill(5_000.0, fee_rate).notional;
        let expected_notional = accounting::quantize(notional1 + add_notional);
        assert!(
            (pos2.notional_usd - expected_notional).abs() < 1e-12,
            "notional: {} != expected {}",
            pos2.notional_usd,
            expected_notional,
        );

        // Margin should accumulate.
        let add_margin = accounting::quantize(add_notional / params.leverage.max(1.0));
        let expected_margin = accounting::quantize(margin1 + add_margin);
        assert!(
            (pos2.margin_usd - expected_margin).abs() < 1e-12,
            "margin: {} != expected {}",
            pos2.margin_usd,
            expected_margin,
        );
    }

    #[test]
    fn add_three_levels_weighted_avg_correct() {
        // Open @ 10_000, ADD @ 11_000, ADD @ 9_000
        let state = init_state();
        let params = pyramid_params();

        let mut e1 = event_with_signal("BTC", MarketSignal::Buy);
        e1.notional_hint_usd = Some(10_000.0);
        e1.price = 10_000.0;
        let r1 = step(&state, &e1, &params);

        let e2 = MarketEvent {
            schema_version: KERNEL_SCHEMA_VERSION,
            event_id: 201,
            timestamp_ms: 2_000,
            symbol: "BTC".to_string(),
            signal: MarketSignal::Buy,
            price: 11_000.0,
            notional_hint_usd: Some(11_000.0),
            close_fraction: None,
            fee_role: None,
            funding_rate: None,
            indicators: None,
            gate_result: None,
        };
        let r2 = step(&r1.state, &e2, &params);

        let e3 = MarketEvent {
            schema_version: KERNEL_SCHEMA_VERSION,
            event_id: 202,
            timestamp_ms: 3_000,
            symbol: "BTC".to_string(),
            signal: MarketSignal::Buy,
            price: 9_000.0,
            notional_hint_usd: Some(9_000.0),
            close_fraction: None,
            fee_role: None,
            funding_rate: None,
            indicators: None,
            gate_result: None,
        };
        let r3 = step(&r2.state, &e3, &params);
        let pos3 = r3.state.positions.get("BTC").unwrap();

        // qty1=1.0 @ 10k, qty2=1.0 @ 11k, qty3=1.0 @ 9k
        // weighted avg = (10k*1 + 11k*1 + 9k*1) / 3 = 10_000
        let p1 = r1.state.positions.get("BTC").unwrap();
        let p2 = r2.state.positions.get("BTC").unwrap();
        // Verify step-by-step: after step 2, avg = (10k*1 + 11k*1)/2 = 10_500
        let expected_avg2 = accounting::quantize(
            (10_000.0 * p1.quantity + 11_000.0 * (p2.quantity - p1.quantity))
                / p2.quantity,
        );
        assert!(
            (p2.avg_entry_price - expected_avg2).abs() < 1e-12,
            "avg after 2nd add: {} != {}",
            p2.avg_entry_price,
            expected_avg2,
        );

        // After step 3: avg = (prev_avg * prev_qty + 9k * new_qty) / total_qty
        let qty3_added = accounting::quantize(9_000.0 / 9_000.0);
        let expected_avg3 = accounting::quantize(
            (p2.avg_entry_price * p2.quantity + 9_000.0 * qty3_added)
                / (p2.quantity + qty3_added),
        );
        assert!(
            (pos3.avg_entry_price - expected_avg3).abs() < 1e-12,
            "avg after 3rd add: {} != {}",
            pos3.avg_entry_price,
            expected_avg3,
        );
        assert!((pos3.quantity - 3.0).abs() < 1e-12);
    }

    // ---- Position tracking metadata tests ----

    #[test]
    fn adds_count_increments_on_each_add() {
        let state = init_state();
        let params = pyramid_params();

        let mut open_evt = event_with_signal("BTC", MarketSignal::Buy);
        open_evt.notional_hint_usd = Some(10_000.0);
        open_evt.price = 10_000.0;
        let r1 = step(&state, &open_evt, &params);
        assert_eq!(r1.state.positions.get("BTC").unwrap().adds_count, 0);

        let add1 = MarketEvent {
            schema_version: KERNEL_SCHEMA_VERSION,
            event_id: 301,
            timestamp_ms: 2_000,
            symbol: "BTC".to_string(),
            signal: MarketSignal::Buy,
            price: 10_500.0,
            notional_hint_usd: Some(5_000.0),
            close_fraction: None,
            fee_role: None,
            funding_rate: None,
            indicators: None,
            gate_result: None,
        };
        let r2 = step(&r1.state, &add1, &params);
        assert_eq!(r2.state.positions.get("BTC").unwrap().adds_count, 1);

        let add2 = MarketEvent {
            event_id: 302,
            timestamp_ms: 3_000,
            price: 11_000.0,
            ..add1.clone()
        };
        let r3 = step(&r2.state, &add2, &params);
        assert_eq!(r3.state.positions.get("BTC").unwrap().adds_count, 2);
    }

    #[test]
    fn metadata_defaults_on_new_position() {
        let state = init_state();
        let params = KernelParams::default();

        let mut open_evt = event_with_signal("BTC", MarketSignal::Buy);
        open_evt.notional_hint_usd = Some(10_000.0);
        let result = step(&state, &open_evt, &params);
        let pos = result.state.positions.get("BTC").unwrap();

        assert_eq!(pos.confidence, None);
        assert_eq!(pos.entry_atr, None);
        assert_eq!(pos.adds_count, 0);
        assert!(!pos.tp1_taken);
        assert_eq!(pos.trailing_sl, None);
        assert!((pos.mae_usd - 0.0).abs() < f64::EPSILON);
        assert!((pos.mfe_usd - 0.0).abs() < f64::EPSILON);
        assert_eq!(pos.last_funding_ms, None);
    }

    #[test]
    fn caller_set_metadata_survives_add_cycle() {
        let state = init_state();
        let params = pyramid_params();

        // Open position.
        let mut open_evt = event_with_signal("BTC", MarketSignal::Buy);
        open_evt.notional_hint_usd = Some(10_000.0);
        open_evt.price = 10_000.0;
        let mut r1 = step(&state, &open_evt, &params);

        // Simulate caller setting metadata after open.
        let pos = r1.state.positions.get_mut("BTC").unwrap();
        pos.confidence = Some(2);
        pos.entry_atr = Some(350.0);
        pos.tp1_taken = true;
        pos.trailing_sl = Some(9_500.0);
        pos.mae_usd = -150.0;
        pos.mfe_usd = 200.0;

        // ADD to position — metadata should survive.
        let add_evt = MarketEvent {
            schema_version: KERNEL_SCHEMA_VERSION,
            event_id: 401,
            timestamp_ms: 2_000,
            symbol: "BTC".to_string(),
            signal: MarketSignal::Buy,
            price: 10_500.0,
            notional_hint_usd: Some(5_000.0),
            close_fraction: None,
            fee_role: None,
            funding_rate: None,
            indicators: None,
            gate_result: None,
        };
        let r2 = step(&r1.state, &add_evt, &params);
        let pos2 = r2.state.positions.get("BTC").unwrap();

        assert_eq!(pos2.confidence, Some(2));
        assert_eq!(pos2.entry_atr, Some(350.0));
        assert!(pos2.tp1_taken);
        assert_eq!(pos2.trailing_sl, Some(9_500.0));
        assert!((pos2.mae_usd - (-150.0)).abs() < f64::EPSILON);
        assert!((pos2.mfe_usd - 200.0).abs() < f64::EPSILON);
        assert_eq!(pos2.adds_count, 1); // incremented by kernel
    }

    #[test]
    fn position_json_round_trip_with_metadata() {
        let pos = Position {
            symbol: "ETH".to_string(),
            side: PositionSide::Short,
            quantity: 5.0,
            avg_entry_price: 3_200.0,
            opened_at_ms: 1_000,
            updated_at_ms: 2_000,
            notional_usd: 16_000.0,
            margin_usd: 8_000.0,
            confidence: Some(1),
            entry_atr: Some(120.5),
            entry_adx_threshold: Some(18.0),
            adds_count: 3,
            tp1_taken: true,
            trailing_sl: Some(3_300.0),
            mae_usd: -400.0,
            mfe_usd: 600.0,
            last_funding_ms: Some(1_500),
        };

        let json = serde_json::to_string(&pos).unwrap();
        let deser: Position = serde_json::from_str(&json).unwrap();
        assert_eq!(pos, deser);
    }

    #[test]
    fn position_json_round_trip_without_optional_metadata() {
        // Deserialise JSON missing all optional/default fields.
        let json = r#"{
            "symbol": "BTC",
            "side": "long",
            "quantity": 1.0,
            "avg_entry_price": 10000.0,
            "opened_at_ms": 0,
            "updated_at_ms": 0,
            "notional_usd": 10000.0
        }"#;
        let pos: Position = serde_json::from_str(json).unwrap();
        assert_eq!(pos.margin_usd, 0.0);
        assert_eq!(pos.confidence, None);
        assert_eq!(pos.entry_atr, None);
        assert_eq!(pos.adds_count, 0);
        assert!(!pos.tp1_taken);
        assert_eq!(pos.trailing_sl, None);
        assert!((pos.mae_usd).abs() < f64::EPSILON);
        assert!((pos.mfe_usd).abs() < f64::EPSILON);
    }

    // ---- Fee role selection tests ----

    fn fee_role_params() -> KernelParams {
        KernelParams {
            maker_fee_bps: 1.0,  // 0.01%
            taker_fee_bps: 3.5,  // 0.035%
            ..KernelParams::default()
        }
    }

    #[test]
    fn open_with_default_none_fee_role_uses_taker() {
        let state = init_state();
        let params = fee_role_params();
        let mut event = event_with_signal("BTC", MarketSignal::Buy);
        event.notional_hint_usd = Some(10_000.0);
        // fee_role is None → taker

        let result = step(&state, &event, &params);
        let fill = &result.fills[0];

        let expected_fee = accounting::quantize(10_000.0 * (3.5 / 10_000.0));
        assert!(
            (fill.fee_usd - expected_fee).abs() < 1e-12,
            "default fee: {} != expected taker fee {}",
            fill.fee_usd,
            expected_fee,
        );
    }

    #[test]
    fn open_with_explicit_maker_fee_role() {
        let state = init_state();
        let params = fee_role_params();
        let mut event = event_with_signal("BTC", MarketSignal::Buy);
        event.notional_hint_usd = Some(10_000.0);
        event.fee_role = Some(accounting::FeeRole::Maker);

        let result = step(&state, &event, &params);
        let fill = &result.fills[0];

        let expected_fee = accounting::quantize(10_000.0 * (1.0 / 10_000.0));
        assert!(
            (fill.fee_usd - expected_fee).abs() < 1e-12,
            "maker fee: {} != expected {}",
            fill.fee_usd,
            expected_fee,
        );
    }

    #[test]
    fn open_with_explicit_taker_matches_default() {
        let state = init_state();
        let params = fee_role_params();

        let mut evt_default = event_with_signal("BTC", MarketSignal::Buy);
        evt_default.notional_hint_usd = Some(10_000.0);
        // fee_role = None (taker by default)

        let mut evt_explicit = evt_default.clone();
        evt_explicit.fee_role = Some(accounting::FeeRole::Taker);

        let r_default = step(&state, &evt_default, &params);
        let r_explicit = step(&state, &evt_explicit, &params);

        assert_eq!(
            r_default.fills[0].fee_usd, r_explicit.fills[0].fee_usd,
            "explicit taker should match default"
        );
        assert_eq!(r_default.state.cash_usd, r_explicit.state.cash_usd);
    }

    #[test]
    fn maker_vs_taker_cash_difference_matches_bps_delta() {
        let state = init_state();
        let params = fee_role_params();
        let notional = 10_000.0;

        let mut evt_maker = event_with_signal("BTC", MarketSignal::Buy);
        evt_maker.notional_hint_usd = Some(notional);
        evt_maker.fee_role = Some(accounting::FeeRole::Maker);

        let mut evt_taker = event_with_signal("BTC", MarketSignal::Buy);
        evt_taker.notional_hint_usd = Some(notional);
        // fee_role = None → taker

        let r_maker = step(&state, &evt_maker, &params);
        let r_taker = step(&state, &evt_taker, &params);

        // Maker saves (taker_bps - maker_bps) / 10_000 * notional in fees.
        let bps_delta = params.taker_fee_bps - params.maker_fee_bps; // 2.5
        let expected_savings = accounting::quantize(notional * bps_delta / 10_000.0);
        let actual_savings = accounting::quantize(r_maker.state.cash_usd - r_taker.state.cash_usd);
        assert!(
            (actual_savings - expected_savings).abs() < 1e-12,
            "cash savings: {} != expected {}",
            actual_savings,
            expected_savings,
        );
    }

    // ---- Funding settlement tests ----

    fn funding_event(symbol: &str, price: f64, rate: f64) -> MarketEvent {
        MarketEvent {
            schema_version: KERNEL_SCHEMA_VERSION,
            event_id: 500,
            timestamp_ms: 5_000,
            symbol: symbol.to_string(),
            signal: MarketSignal::Funding,
            price,
            notional_hint_usd: None,
            close_fraction: None,
            fee_role: None,
            funding_rate: Some(rate),
            indicators: None,
            gate_result: None,
        }
    }

    #[test]
    fn funding_long_positive_rate_decreases_cash() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = KernelParams::default();
        let cash_before = state.cash_usd;
        let pos = state.positions.get("BTC").unwrap();
        let qty = pos.quantity;

        let rate = 0.0001; // positive
        let event = funding_event("BTC", 10_000.0, rate);
        let result = step(&state, &event, &params);

        // Long pays positive funding: cash should decrease.
        let expected_delta = accounting::funding_delta(true, qty, 10_000.0, rate);
        assert!(expected_delta < 0.0, "long+positive rate delta should be negative");
        let expected_cash = accounting::quantize(cash_before + expected_delta);
        assert!(
            (result.state.cash_usd - expected_cash).abs() < 1e-12,
            "funding long cash: {} != expected {}",
            result.state.cash_usd,
            expected_cash,
        );
        // No intents or fills.
        assert!(result.intents.is_empty());
        assert!(result.fills.is_empty());
        // Position still exists, unchanged quantity.
        let pos_after = result.state.positions.get("BTC").unwrap();
        assert!((pos_after.quantity - qty).abs() < 1e-12);
        assert_eq!(pos_after.last_funding_ms, Some(5_000));
    }

    #[test]
    fn funding_short_positive_rate_increases_cash() {
        // Open a short position.
        let state = init_state();
        let params = KernelParams {
            allow_reverse: false,
            ..KernelParams::default()
        };
        let mut open_evt = event_with_signal("BTC", MarketSignal::Sell);
        open_evt.notional_hint_usd = Some(10_000.0);
        let open_result = step(&state, &open_evt, &params);
        let short_state = open_result.state;
        let cash_before = short_state.cash_usd;
        let pos = short_state.positions.get("BTC").unwrap();
        let qty = pos.quantity;
        assert_eq!(pos.side, PositionSide::Short);

        let rate = 0.0001;
        let event = funding_event("BTC", 10_000.0, rate);
        let result = step(&short_state, &event, &params);

        // Short receives positive funding: cash should increase.
        let expected_delta = accounting::funding_delta(false, qty, 10_000.0, rate);
        assert!(expected_delta > 0.0, "short+positive rate delta should be positive");
        let expected_cash = accounting::quantize(cash_before + expected_delta);
        assert!(
            (result.state.cash_usd - expected_cash).abs() < 1e-12,
            "funding short cash: {} != expected {}",
            result.state.cash_usd,
            expected_cash,
        );
        assert!(result.intents.is_empty());
        assert!(result.fills.is_empty());
    }

    #[test]
    fn funding_zero_rate_no_cash_change() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = KernelParams::default();
        let cash_before = state.cash_usd;

        let event = funding_event("BTC", 10_000.0, 0.0);
        let result = step(&state, &event, &params);

        assert!(
            (result.state.cash_usd - cash_before).abs() < 1e-12,
            "zero rate should not change cash",
        );
        // last_funding_ms should NOT be updated for zero rate.
        let pos = result.state.positions.get("BTC").unwrap();
        assert_eq!(pos.last_funding_ms, None);
    }

    #[test]
    fn funding_no_position_no_cash_change() {
        let state = init_state(); // no positions
        let params = KernelParams::default();
        let cash_before = state.cash_usd;

        let event = funding_event("BTC", 10_000.0, 0.0001);
        let result = step(&state, &event, &params);

        assert!(
            (result.state.cash_usd - cash_before).abs() < 1e-12,
            "no position → no funding effect",
        );
        assert!(result.intents.is_empty());
        assert!(result.fills.is_empty());
    }

    #[test]
    fn funding_no_rate_field_no_cash_change() {
        // Funding signal but funding_rate = None → no effect.
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = KernelParams::default();
        let cash_before = state.cash_usd;

        let mut event = funding_event("BTC", 10_000.0, 0.0);
        event.funding_rate = None;
        let result = step(&state, &event, &params);

        assert!(
            (result.state.cash_usd - cash_before).abs() < 1e-12,
            "None funding_rate should not change cash",
        );
    }

    #[test]
    fn funding_advances_step_counter() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = KernelParams::default();
        let step_before = state.step;

        let event = funding_event("BTC", 10_000.0, 0.0001);
        let result = step(&state, &event, &params);

        assert_eq!(result.state.step, step_before + 1);
        assert_eq!(result.state.timestamp_ms, 5_000);
    }

    #[test]
    fn funding_deterministic() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = KernelParams::default();

        let event = funding_event("BTC", 10_000.0, 0.0001);
        let r1 = step(&state, &event, &params);
        let r2 = step(&state, &event, &params);
        assert_eq!(r1, r2, "funding settlement must be deterministic");
    }

    #[test]
    fn funding_wrong_symbol_no_effect() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = KernelParams::default();
        let cash_before = state.cash_usd;

        // Funding for ETH, but only BTC position exists.
        let event = funding_event("ETH", 3_000.0, 0.0001);
        let result = step(&state, &event, &params);

        assert!(
            (result.state.cash_usd - cash_before).abs() < 1e-12,
            "funding for wrong symbol should not change cash",
        );
    }

    // ---- PriceUpdate / exit evaluation integration tests ----

    fn test_snap(close: f64) -> IndicatorSnapshot {
        IndicatorSnapshot {
            close,
            high: close,
            low: close,
            open: close,
            volume: 0.0,
            t: 0,
            ema_slow: close,
            ema_fast: close,
            ema_macro: close,
            adx: 30.0,
            adx_pos: 15.0,
            adx_neg: 15.0,
            adx_slope: 0.0,
            bb_upper: close * 1.02,
            bb_lower: close * 0.98,
            bb_width: 0.04,
            bb_width_avg: 0.04,
            bb_width_ratio: 1.0,
            atr: close * 0.01,
            atr_slope: 0.0,
            avg_atr: close * 0.01,
            rsi: 50.0,
            stoch_rsi_k: 50.0,
            stoch_rsi_d: 50.0,
            macd_hist: 0.0,
            prev_macd_hist: 0.0,
            prev2_macd_hist: 0.0,
            prev3_macd_hist: 0.0,
            vol_sma: 100.0,
            vol_trend: false,
            prev_close: close,
            prev_ema_fast: close,
            prev_ema_slow: close,
            bar_count: 200,
            funding_rate: 0.0,
        }
    }

    fn exit_params_for_test() -> KernelParams {
        KernelParams {
            allow_reverse: false,
            exit_params: Some(ExitParams::default()),
            ..KernelParams::default()
        }
    }

    fn price_update_event(symbol: &str, price: f64, snap: IndicatorSnapshot) -> MarketEvent {
        MarketEvent {
            schema_version: KERNEL_SCHEMA_VERSION,
            event_id: 900,
            timestamp_ms: 5_000,
            symbol: symbol.to_string(),
            signal: MarketSignal::PriceUpdate,
            price,
            notional_hint_usd: None,
            close_fraction: None,
            fee_role: None,
            funding_rate: None,
            indicators: Some(snap),
            gate_result: None,
        }
    }

    #[test]
    fn price_update_sl_triggers_close() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let mut state_with_atr = state.clone();
        let pos = state_with_atr.positions.get_mut("BTC").unwrap();
        pos.entry_atr = Some(100.0);

        let params = exit_params_for_test();
        let snap = test_snap(9_750.0);
        let event = price_update_event("BTC", 9_750.0, snap);

        let result = step(&state_with_atr, &event, &params);

        assert!(!result.intents.is_empty(), "should emit close intent");
        assert_eq!(result.intents[0].kind, OrderIntentKind::Close);
        assert!(
            result.state.positions.get("BTC").is_none(),
            "position should be closed"
        );
    }

    #[test]
    fn price_update_no_exit_on_normal_price() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let mut state_with_atr = state.clone();
        let pos = state_with_atr.positions.get_mut("BTC").unwrap();
        pos.entry_atr = Some(100.0);

        let params = exit_params_for_test();
        let snap = test_snap(10_050.0);
        let event = price_update_event("BTC", 10_050.0, snap);

        let result = step(&state_with_atr, &event, &params);

        assert!(result.intents.is_empty(), "no exit should trigger");
        assert!(
            result.state.positions.get("BTC").is_some(),
            "position should remain"
        );
    }

    #[test]
    fn price_update_partial_tp_emits_partial_close() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let mut state_with_atr = state.clone();
        let pos = state_with_atr.positions.get_mut("BTC").unwrap();
        pos.entry_atr = Some(100.0);

        let params = exit_params_for_test();
        let snap = test_snap(10_400.0);
        let event = price_update_event("BTC", 10_400.0, snap);

        let result = step(&state_with_atr, &event, &params);

        assert!(!result.intents.is_empty(), "should emit close intent");
        assert_eq!(result.intents[0].kind, OrderIntentKind::Close);
        let pos = result.state.positions.get("BTC");
        assert!(pos.is_some(), "position should remain after partial TP");
        let pos = pos.unwrap();
        assert!(pos.tp1_taken, "tp1_taken should be set after partial TP");
    }

    #[test]
    fn price_update_trailing_updates_position() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let mut state_with_atr = state.clone();
        let pos = state_with_atr.positions.get_mut("BTC").unwrap();
        pos.entry_atr = Some(100.0);

        let params = exit_params_for_test();
        let snap = test_snap(10_200.0);
        let event = price_update_event("BTC", 10_200.0, snap);

        let result = step(&state_with_atr, &event, &params);

        let pos = result.state.positions.get("BTC").unwrap();
        assert!(
            pos.trailing_sl.is_some(),
            "trailing_sl should be set after sufficient profit"
        );
    }

    #[test]
    fn price_update_without_exit_params_is_noop() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = KernelParams {
            allow_reverse: false,
            exit_params: None,
            ..KernelParams::default()
        };
        let snap = test_snap(9_500.0);
        let event = price_update_event("BTC", 9_500.0, snap);

        let result = step(&state, &event, &params);

        assert!(result.intents.is_empty(), "no intents without exit_params");
        assert!(
            result.state.positions.get("BTC").is_some(),
            "position should remain"
        );
    }

    #[test]
    fn price_update_without_indicators_is_noop() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = exit_params_for_test();
        let event = MarketEvent {
            schema_version: KERNEL_SCHEMA_VERSION,
            event_id: 901,
            timestamp_ms: 5_000,
            symbol: "BTC".to_string(),
            signal: MarketSignal::PriceUpdate,
            price: 9_500.0,
            notional_hint_usd: None,
            close_fraction: None,
            fee_role: None,
            funding_rate: None,
            indicators: None,
            gate_result: None,
        };

        let result = step(&state, &event, &params);

        assert!(result.intents.is_empty());
    }

    #[test]
    fn price_update_advances_step_counter() {
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = exit_params_for_test();
        let snap = test_snap(10_050.0);
        let event = price_update_event("BTC", 10_050.0, snap);
        let step_before = state.step;

        let result = step(&state, &event, &params);

        assert_eq!(result.state.step, step_before + 1);
    }

    // ---- Gate pre-filter tests ----

    fn failing_gate_result() -> GateResult {
        GateResult {
            is_ranging: true,
            is_anomaly: false,
            is_extended: false,
            vol_confirm: true,
            is_trending_up: true,
            adx_above_min: true,
            bullish_alignment: false,
            bearish_alignment: false,
            btc_ok_long: true,
            btc_ok_short: true,
            effective_min_adx: 22.0,
            bb_width_ratio: 1.0,
            dynamic_tp_mult: 5.0,
            rsi_long_limit: 55.0,
            rsi_short_limit: 45.0,
            stoch_k: 0.5,
            stoch_d: 0.5,
            stoch_rsi_active: false,
            all_gates_pass: false,
        }
    }

    fn passing_gate_result() -> GateResult {
        GateResult {
            is_ranging: false,
            is_anomaly: false,
            is_extended: false,
            vol_confirm: true,
            is_trending_up: true,
            adx_above_min: true,
            bullish_alignment: true,
            bearish_alignment: true,
            btc_ok_long: true,
            btc_ok_short: true,
            effective_min_adx: 22.0,
            bb_width_ratio: 1.0,
            dynamic_tp_mult: 5.0,
            rsi_long_limit: 55.0,
            rsi_short_limit: 45.0,
            stoch_k: 0.5,
            stoch_d: 0.5,
            stoch_rsi_active: false,
            all_gates_pass: true,
        }
    }

    #[test]
    fn gate_blocks_new_entry() {
        let state = init_state();
        let params = KernelParams::default();
        let mut event = event_with_signal("BTC", MarketSignal::Buy);
        event.gate_result = Some(failing_gate_result());

        let result = step(&state, &event, &params);

        assert!(
            result.state.positions.get("BTC").is_none(),
            "failing gates should block new entry"
        );
        assert!(result.intents.is_empty());
        assert!(result.fills.is_empty());
        assert!(result.diagnostics.gate_blocked);
    }

    #[test]
    fn gate_allows_entry_when_passing() {
        let state = init_state();
        let params = KernelParams::default();
        let mut event = event_with_signal("BTC", MarketSignal::Buy);
        event.gate_result = Some(passing_gate_result());

        let result = step(&state, &event, &params);

        assert!(
            result.state.positions.get("BTC").is_some(),
            "passing gates should allow entry"
        );
        assert_eq!(result.intents.len(), 1);
        assert!(!result.diagnostics.gate_blocked);
    }

    #[test]
    fn gate_allows_close_despite_failing() {
        // Open a long position, then send a Sell signal with failing gates.
        // The close should still happen (we want to be able to exit).
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = KernelParams {
            allow_reverse: false,
            ..KernelParams::default()
        };
        let event = MarketEvent {
            schema_version: KERNEL_SCHEMA_VERSION,
            event_id: 600,
            timestamp_ms: 2_000,
            symbol: "BTC".to_string(),
            signal: MarketSignal::Sell,
            price: 10_200.0,
            notional_hint_usd: None,
            close_fraction: None,
            fee_role: None,
            funding_rate: None,
            indicators: None,
            gate_result: Some(failing_gate_result()),
        };

        let result = step(&state, &event, &params);

        assert!(
            result.state.positions.get("BTC").is_none(),
            "opposite-side signal should still close despite failing gates"
        );
        assert!(!result.intents.is_empty());
        assert!(!result.diagnostics.gate_blocked);
        // But gate_block_reasons should still be populated for audit
        assert!(!result.diagnostics.gate_block_reasons.is_empty());
    }

    #[test]
    fn gate_none_allows_entry() {
        // Backwards compatibility: gate_result = None should not block anything.
        let state = init_state();
        let params = KernelParams::default();
        let mut event = event_with_signal("BTC", MarketSignal::Buy);
        event.gate_result = None;

        let result = step(&state, &event, &params);

        assert!(
            result.state.positions.get("BTC").is_some(),
            "None gate_result should allow entry (backwards compatible)"
        );
        assert!(!result.diagnostics.gate_blocked);
    }

    #[test]
    fn gate_blocks_pyramid() {
        // Open a long position, then try to pyramid with failing gates.
        let state = open_long_btc(10_000.0, 10_000.0);
        let params = pyramid_params();
        let event = MarketEvent {
            schema_version: KERNEL_SCHEMA_VERSION,
            event_id: 601,
            timestamp_ms: 2_000,
            symbol: "BTC".to_string(),
            signal: MarketSignal::Buy,
            price: 10_500.0,
            notional_hint_usd: Some(5_000.0),
            close_fraction: None,
            fee_role: None,
            funding_rate: None,
            indicators: None,
            gate_result: Some(failing_gate_result()),
        };

        let pos_before = state.positions.get("BTC").unwrap().clone();
        let result = step(&state, &event, &params);

        // Position should remain unchanged (no pyramid added).
        let pos_after = result.state.positions.get("BTC").unwrap();
        assert!(
            (pos_after.quantity - pos_before.quantity).abs() < 1e-12,
            "failing gates should block pyramid"
        );
        assert_eq!(pos_after.adds_count, 0);
        assert!(result.diagnostics.gate_blocked);
        assert!(result.intents.is_empty());
    }

    #[test]
    fn gate_blocked_diagnostics() {
        let state = init_state();
        let params = KernelParams::default();
        let mut event = event_with_signal("BTC", MarketSignal::Buy);
        // Create a gate result that fails on multiple fronts
        let mut gr = failing_gate_result();
        gr.is_anomaly = true;
        gr.btc_ok_long = false;
        event.gate_result = Some(gr);

        let result = step(&state, &event, &params);

        assert!(result.diagnostics.gate_blocked);
        assert!(result.diagnostics.gate_block_reasons.contains(&"ranging".to_string()));
        assert!(result.diagnostics.gate_block_reasons.contains(&"anomaly".to_string()));
        assert!(result.diagnostics.gate_block_reasons.contains(&"bearish_alignment".to_string()));
        assert!(result.diagnostics.gate_block_reasons.contains(&"btc_alignment".to_string()));
        assert_eq!(result.diagnostics.intent_count, 0);
        assert_eq!(result.diagnostics.fill_count, 0);
    }
}
