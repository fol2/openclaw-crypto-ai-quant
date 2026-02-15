//! Versioned decision kernel types and a deterministic transition function.
//!
//! Invariants:
//! 1) `step(state, event, params)` is pure and deterministic for identical inputs.
//! 2) Every participating payload must use `schema_version == KERNEL_SCHEMA_VERSION`.
//!    A mismatch results in an error and no state change.
//! 3) All monetary and size values are rounded to a stable 1e-12 resolution before
//!    being written into state so that repeated replays remain stable.

use crate::accounting;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const KERNEL_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MarketSignal {
    Buy,
    Sell,
    Neutral,
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
        MarketSignal::Neutral => None,
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
    intent_id: u64,
    diagnostics: &mut Diagnostics,
) -> Option<(OrderIntent, FillEvent)> {
    let position = match state.positions.remove(symbol) {
        Some(pos) => pos,
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

    let quantity = accounting::quantize(position.quantity);
    let close = accounting::apply_close_fill(
        side == PositionSide::Long,
        position.avg_entry_price,
        price,
        quantity,
        fee_rate,
    );

    // Return margin (collateral) + PnL - fee to cash.
    let margin = position.margin_usd;
    state.cash_usd = quantise(state.cash_usd + margin + close.pnl - close.fee_usd);

    let intent = OrderIntent {
        schema_version: KERNEL_SCHEMA_VERSION,
        intent_id,
        symbol: symbol.to_string(),
        kind: OrderIntentKind::Close,
        side,
        quantity,
        price: quantise(price),
        notional_usd: close.notional,
        fee_rate,
    };
    let fill = FillEvent {
        schema_version: KERNEL_SCHEMA_VERSION,
        intent_id,
        symbol: symbol.to_string(),
        side,
        quantity,
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

    let fee_model = accounting::FeeModel {
        maker_fee_bps: params.maker_fee_bps,
        taker_fee_bps: params.taker_fee_bps,
    };
    let fee_rate = fee_model.role_rate(accounting::FeeRole::Taker);
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
}
