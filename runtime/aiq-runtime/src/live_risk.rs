use chrono::{TimeZone, Utc};
use std::collections::VecDeque;
use std::env;
use std::fs;
use std::path::PathBuf;

const MIN_ORDERS_PER_MIN: f64 = 1.0;
const MAX_ORDERS_PER_MIN: f64 = 1000.0;
const MIN_SLIPPAGE_WINDOW_FILLS: usize = 1;
const MAX_SLIPPAGE_WINDOW_FILLS: usize = 1000;
const MAX_ORDER_GAP_MS: i64 = 60_000;
const MILLIS_PER_MINUTE: f64 = 60_000.0;
const BASIS_POINTS_PER_UNIT: f64 = 10_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KillMode {
    Off,
    CloseOnly,
    HaltAll,
}

impl KillMode {
    fn from_env_value(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "close_only" | "close" => Some(Self::CloseOnly),
            "halt_all" | "halt" => Some(Self::HaltAll),
            "off" => Some(Self::Off),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderAction {
    Open,
    Add,
    Close,
    Reduce,
}

impl OrderAction {
    fn is_entry(self) -> bool {
        matches!(self, Self::Open | Self::Add)
    }

    fn is_exit(self) -> bool {
        matches!(self, Self::Close | Self::Reduce)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiskDecision {
    pub allowed: bool,
    pub reason: String,
    pub kill_mode: Option<KillMode>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveRiskStatus {
    pub kill_mode: KillMode,
    pub kill_reason: Option<String>,
    pub kill_since_ms: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct LiveRiskConfig {
    pub kill_switch_file: Option<PathBuf>,
    pub kill_switch_mode: KillMode,
    pub max_entry_orders_per_min: f64,
    pub max_exit_orders_per_min: f64,
    pub max_cancels_per_min: f64,
    pub min_order_gap_ms: i64,
    pub max_drawdown_pct: f64,
    pub max_daily_loss_usd: f64,
    pub slippage_guard_enabled: bool,
    pub slippage_guard_window_fills: usize,
    pub slippage_guard_max_median_bps: f64,
}

impl Default for LiveRiskConfig {
    fn default() -> Self {
        Self {
            kill_switch_file: None,
            kill_switch_mode: KillMode::CloseOnly,
            max_entry_orders_per_min: 30.0,
            max_exit_orders_per_min: 120.0,
            max_cancels_per_min: 120.0,
            min_order_gap_ms: 0,
            max_drawdown_pct: 0.0,
            max_daily_loss_usd: 0.0,
            slippage_guard_enabled: false,
            slippage_guard_window_fills: 20,
            slippage_guard_max_median_bps: 0.0,
        }
    }
}

impl LiveRiskConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        config.apply_env_overrides();
        config
    }

    fn apply_env_overrides(&mut self) {
        if let Some(raw) = env_raw("AI_QUANT_KILL_SWITCH_FILE") {
            self.kill_switch_file = if raw.is_empty() {
                None
            } else {
                Some(PathBuf::from(raw))
            };
        }
        if let Some(raw) = env_raw("AI_QUANT_KILL_SWITCH_MODE")
            .and_then(|value| KillMode::from_env_value(&value))
            .filter(|mode| *mode != KillMode::Off)
        {
            self.kill_switch_mode = raw;
        }

        if let Some(raw) = env_raw("AI_QUANT_RISK_MAX_DRAWDOWN_PCT") {
            if let Some(value) = parse_f64(&raw) {
                self.max_drawdown_pct = value.clamp(0.0, 100.0);
            }
        }
        if let Some(raw) = env_raw("AI_QUANT_RISK_MAX_DAILY_LOSS_USD") {
            if let Some(value) = parse_f64(&raw) {
                self.max_daily_loss_usd = value.max(0.0);
            }
        }
        if let Some(raw) = env_raw("AI_QUANT_RISK_SLIPPAGE_GUARD_ENABLED") {
            self.slippage_guard_enabled = parse_bool(&raw).unwrap_or(false);
        }
        if let Some(raw) = env_raw("AI_QUANT_RISK_SLIPPAGE_GUARD_WINDOW_FILLS") {
            if let Some(value) = parse_usize(&raw) {
                self.slippage_guard_window_fills =
                    value.clamp(MIN_SLIPPAGE_WINDOW_FILLS, MAX_SLIPPAGE_WINDOW_FILLS);
            }
        }
        if let Some(raw) = env_raw("AI_QUANT_RISK_SLIPPAGE_GUARD_MAX_MEDIAN_BPS") {
            if let Some(value) = parse_f64(&raw) {
                self.slippage_guard_max_median_bps = value.max(0.0);
            }
        }
        if let Some(raw) = env_raw("AI_QUANT_RISK_MAX_ENTRY_ORDERS_PER_MIN") {
            if let Some(value) = parse_f64(&raw) {
                self.max_entry_orders_per_min = clamp_orders_per_min(value);
            }
        }
        if let Some(raw) = env_raw("AI_QUANT_RISK_MAX_EXIT_ORDERS_PER_MIN") {
            if let Some(value) = parse_f64(&raw) {
                self.max_exit_orders_per_min = clamp_orders_per_min(value);
            }
        }
        if let Some(raw) = env_raw("AI_QUANT_RISK_MAX_CANCELS_PER_MIN") {
            if let Some(value) = parse_f64(&raw) {
                self.max_cancels_per_min = clamp_orders_per_min(value);
            }
        }
        if let Some(raw) = env_raw("AI_QUANT_RISK_MIN_ORDER_GAP_MS") {
            if let Some(value) = parse_i64(&raw) {
                self.min_order_gap_ms = value.clamp(0, MAX_ORDER_GAP_MS);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OrderCheck<'a> {
    pub now_ms: i64,
    pub symbol: &'a str,
    pub action: OrderAction,
    pub reduce_risk: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct OrderRecord<'a> {
    pub now_ms: i64,
    pub symbol: &'a str,
    pub action: OrderAction,
    pub reduce_risk: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct CancelCheck<'a> {
    pub now_ms: i64,
    pub symbol: &'a str,
    pub exchange_order_id: Option<&'a str>,
}

#[derive(Debug, Clone, Copy)]
pub struct FillEvent<'a> {
    pub ts_ms: i64,
    pub symbol: &'a str,
    pub action: OrderAction,
    pub pnl_usd: f64,
    pub fee_usd: f64,
    pub fill_price: Option<f64>,
    pub side: Option<OrderSide>,
    pub ref_mid: Option<f64>,
    pub ref_bid: Option<f64>,
    pub ref_ask: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct LiveRiskManager {
    config: LiveRiskConfig,
    env_runtime_enabled: bool,
    kill_mode: KillMode,
    kill_reason: Option<String>,
    kill_since_ms: Option<i64>,
    entry_bucket: TokenBucket,
    exit_bucket: TokenBucket,
    cancel_bucket: TokenBucket,
    last_order_ts_ms: Option<i64>,
    equity_peak: Option<f64>,
    drawdown_peak_utc_day: Option<String>,
    daily_utc_day: Option<String>,
    daily_realised_pnl_usd: f64,
    daily_fees_usd: f64,
    slippage_bps: VecDeque<f64>,
    last_slippage_symbol: Option<String>,
}

impl LiveRiskManager {
    pub fn new(config: LiveRiskConfig) -> Self {
        Self::build(config, false)
    }

    pub fn from_env() -> Self {
        Self::build(LiveRiskConfig::from_env(), true)
    }

    fn build(config: LiveRiskConfig, env_runtime_enabled: bool) -> Self {
        Self {
            entry_bucket: TokenBucket::per_minute(config.max_entry_orders_per_min),
            exit_bucket: TokenBucket::per_minute(config.max_exit_orders_per_min),
            cancel_bucket: TokenBucket::per_minute(config.max_cancels_per_min),
            config,
            env_runtime_enabled,
            kill_mode: KillMode::Off,
            kill_reason: None,
            kill_since_ms: None,
            last_order_ts_ms: None,
            equity_peak: None,
            drawdown_peak_utc_day: None,
            daily_utc_day: None,
            daily_realised_pnl_usd: 0.0,
            daily_fees_usd: 0.0,
            slippage_bps: VecDeque::new(),
            last_slippage_symbol: None,
        }
    }

    pub fn status(&self) -> LiveRiskStatus {
        LiveRiskStatus {
            kill_mode: self.kill_mode,
            kill_reason: self.kill_reason.clone(),
            kill_since_ms: self.kill_since_ms,
        }
    }

    pub fn kill_mode(&self) -> KillMode {
        self.kill_mode
    }

    pub fn kill_reason(&self) -> Option<&str> {
        self.kill_reason.as_deref()
    }

    pub fn kill_since_ms(&self) -> Option<i64> {
        self.kill_since_ms
    }

    pub fn kill(&mut self, now_ms: i64, mode: KillMode, reason: impl Into<String>) {
        let mode = match mode {
            KillMode::Off => KillMode::CloseOnly,
            other => other,
        };
        let reason = reason.into();
        if self.kill_mode == mode
            && self.kill_reason.as_deref() == Some(reason.as_str())
            && self.kill_since_ms.is_some()
        {
            return;
        }
        self.kill_mode = mode;
        self.kill_reason = Some(reason);
        self.kill_since_ms = Some(now_ms);
    }

    pub fn clear_kill(&mut self, reason: impl Into<String>) {
        self.equity_peak = None;
        self.drawdown_peak_utc_day = None;
        self.daily_utc_day = None;
        self.daily_realised_pnl_usd = 0.0;
        self.daily_fees_usd = 0.0;
        self.slippage_bps.clear();
        self.last_slippage_symbol = None;
        self.kill_mode = KillMode::Off;
        self.kill_reason = Some(reason.into());
        self.kill_since_ms = None;
    }

    pub fn refresh(&mut self, now_ms: i64, equity_usd: Option<f64>) {
        if self.env_runtime_enabled {
            let prior_entry_per_min = self.config.max_entry_orders_per_min;
            let prior_exit_per_min = self.config.max_exit_orders_per_min;
            let prior_cancel_per_min = self.config.max_cancels_per_min;
            self.config.apply_env_overrides();
            if self.config.max_entry_orders_per_min != prior_entry_per_min {
                self.entry_bucket = TokenBucket::per_minute(self.config.max_entry_orders_per_min);
            }
            if self.config.max_exit_orders_per_min != prior_exit_per_min {
                self.exit_bucket = TokenBucket::per_minute(self.config.max_exit_orders_per_min);
            }
            if self.config.max_cancels_per_min != prior_cancel_per_min {
                self.cancel_bucket = TokenBucket::per_minute(self.config.max_cancels_per_min);
            }
            self.trim_slippage_window();
        }
        self.refresh_manual_kill(now_ms);
        self.refresh_drawdown(now_ms, equity_usd);
        self.refresh_daily_loss(now_ms);
        self.refresh_slippage_guard(now_ms, None);
    }

    pub fn note_fill(&mut self, fill: FillEvent<'_>) {
        if fill.action.is_exit() {
            self.record_daily_loss(fill.ts_ms, fill.pnl_usd, fill.fee_usd);
            self.refresh_daily_loss(fill.ts_ms);
        }

        if fill.action.is_entry() {
            if let Some(slippage_bps) = self.compute_slippage_bps(&fill) {
                self.slippage_bps.push_back(slippage_bps.max(0.0));
                self.last_slippage_symbol = Some(normalise_symbol(fill.symbol));
                self.trim_slippage_window();
                self.refresh_slippage_guard(fill.ts_ms, Some(fill.symbol));
            }
        }
    }

    pub fn allow_order(&mut self, order: OrderCheck<'_>) -> RiskDecision {
        if self.kill_mode == KillMode::HaltAll {
            return self.block("halt_all");
        }
        if self.kill_mode == KillMode::CloseOnly && !order.reduce_risk && order.action.is_entry() {
            return self.block("close_only");
        }

        if self.config.min_order_gap_ms > 0 {
            if let Some(last_order_ts_ms) = self.last_order_ts_ms {
                if order.now_ms.saturating_sub(last_order_ts_ms) < self.config.min_order_gap_ms {
                    return self.block("min_order_gap");
                }
            }
        }

        if !order.reduce_risk
            && order.action.is_entry()
            && !self.entry_bucket.allow(order.now_ms, 1.0)
        {
            return self.block("entry_rate");
        }

        if (order.reduce_risk || order.action.is_exit())
            && !self.exit_bucket.allow(order.now_ms, 1.0)
        {
            return self.block("exit_rate");
        }

        let _ = order.symbol;
        self.ok()
    }

    pub fn note_order_sent(&mut self, order: OrderRecord<'_>) {
        if (!order.reduce_risk && order.action.is_entry()) || order.action.is_exit() {
            self.last_order_ts_ms = Some(order.now_ms);
        }
        let _ = order.symbol;
    }

    pub fn allow_cancel(&mut self, cancel: CancelCheck<'_>) -> RiskDecision {
        let _ = cancel.symbol;
        let _ = cancel.exchange_order_id;
        if !self.cancel_bucket.allow(cancel.now_ms, 1.0) {
            return self.block("cancel_rate");
        }
        self.ok()
    }

    fn ok(&self) -> RiskDecision {
        RiskDecision {
            allowed: true,
            reason: "ok".to_string(),
            kill_mode: match self.kill_mode {
                KillMode::Off => None,
                mode => Some(mode),
            },
        }
    }

    fn block(&self, reason: &str) -> RiskDecision {
        RiskDecision {
            allowed: false,
            reason: reason.to_string(),
            kill_mode: match self.kill_mode {
                KillMode::Off => None,
                mode => Some(mode),
            },
        }
    }

    fn refresh_manual_kill(&mut self, now_ms: i64) {
        if self.env_runtime_enabled {
            if let Some(raw) = env_raw("AI_QUANT_KILL_SWITCH") {
                let value = raw.trim().to_ascii_lowercase();
                if matches!(value.as_str(), "clear" | "resume" | "off" | "unpause") {
                    self.clear_kill("env_clear");
                    return;
                }
                if matches!(
                    value.as_str(),
                    "1" | "true" | "yes" | "y" | "on" | "close_only" | "close"
                ) {
                    self.kill(now_ms, self.config.kill_switch_mode, "env");
                } else if matches!(value.as_str(), "halt" | "halt_all" | "2" | "stop" | "full") {
                    self.kill(now_ms, KillMode::HaltAll, "env");
                }
            }
        }

        let Some(path) = self.config.kill_switch_file.as_ref() else {
            return;
        };
        if !path.exists() {
            return;
        }

        let mode = match fs::read_to_string(path) {
            Ok(contents) => {
                let value = contents.trim().to_ascii_lowercase();
                if contains_clear_token(&value) {
                    self.clear_kill(format!("file_clear:{}", path.display()));
                    return;
                }
                if value.contains("halt") || value.contains("full") {
                    KillMode::HaltAll
                } else {
                    KillMode::CloseOnly
                }
            }
            Err(_) => KillMode::CloseOnly,
        };
        self.kill(now_ms, mode, format!("file:{}", path.display()));
    }

    fn refresh_drawdown(&mut self, now_ms: i64, equity_usd: Option<f64>) {
        let Some(day) = utc_day(now_ms) else {
            return;
        };
        if self.drawdown_peak_utc_day.as_deref() != Some(day.as_str()) {
            self.drawdown_peak_utc_day = Some(day);
            self.equity_peak = None;
        }

        if self.config.max_drawdown_pct <= 0.0 {
            return;
        }

        let Some(equity) = equity_usd.filter(|value| *value > 0.0) else {
            return;
        };

        match self.equity_peak {
            None => {
                self.equity_peak = Some(equity);
                return;
            }
            Some(peak) if equity > peak => {
                self.equity_peak = Some(equity);
                return;
            }
            Some(peak) if peak > 0.0 => {
                let drawdown_pct = ((peak - equity).max(0.0) / peak) * 100.0;
                if drawdown_pct >= self.config.max_drawdown_pct {
                    self.kill(now_ms, KillMode::CloseOnly, "drawdown");
                }
            }
            _ => {}
        }
    }

    fn record_daily_loss(&mut self, ts_ms: i64, pnl_usd: f64, fee_usd: f64) {
        self.roll_daily_state(ts_ms);
        self.daily_realised_pnl_usd += pnl_usd;
        self.daily_fees_usd += fee_usd;
    }

    fn refresh_daily_loss(&mut self, now_ms: i64) {
        self.roll_daily_state(now_ms);
        if self.config.max_daily_loss_usd <= 0.0 {
            return;
        }
        let Some(day) = self.daily_utc_day.clone() else {
            return;
        };
        let net_pnl = self.daily_realised_pnl_usd - self.daily_fees_usd;
        if net_pnl <= -self.config.max_daily_loss_usd {
            self.kill(now_ms, KillMode::CloseOnly, format!("daily_loss:{day}"));
        }
    }

    fn refresh_slippage_guard(&mut self, now_ms: i64, symbol: Option<&str>) {
        self.trim_slippage_window();
        if !self.config.slippage_guard_enabled || self.config.slippage_guard_max_median_bps <= 0.0 {
            return;
        }
        if self.slippage_bps.len() < self.config.slippage_guard_window_fills {
            return;
        }

        let median_bps = median(self.slippage_bps.iter().copied());
        if median_bps <= self.config.slippage_guard_max_median_bps {
            return;
        }

        let source_symbol = symbol
            .map(normalise_symbol)
            .or_else(|| self.last_slippage_symbol.clone())
            .unwrap_or_else(|| "SYSTEM".to_string());
        let _ = source_symbol;
        self.kill(now_ms, KillMode::CloseOnly, "slippage_guard");
    }

    fn roll_daily_state(&mut self, ts_ms: i64) {
        let Some(day) = utc_day(ts_ms) else {
            return;
        };
        if self.daily_utc_day.as_deref() != Some(day.as_str()) {
            self.daily_utc_day = Some(day);
            self.daily_realised_pnl_usd = 0.0;
            self.daily_fees_usd = 0.0;
        }
    }

    fn trim_slippage_window(&mut self) {
        while self.slippage_bps.len() > self.config.slippage_guard_window_fills {
            let _ = self.slippage_bps.pop_front();
        }
    }

    fn compute_slippage_bps(&self, fill: &FillEvent<'_>) -> Option<f64> {
        if !self.config.slippage_guard_enabled || self.config.slippage_guard_max_median_bps <= 0.0 {
            return None;
        }

        let fill_price = fill.fill_price.filter(|price| *price > 0.0)?;
        let side = fill.side?;
        let reference_price = match side {
            OrderSide::Buy => fill.ref_ask.filter(|price| *price > 0.0).or(fill.ref_mid),
            OrderSide::Sell => fill.ref_bid.filter(|price| *price > 0.0).or(fill.ref_mid),
        }
        .filter(|price| *price > 0.0)?;

        let slippage_bps = match side {
            OrderSide::Buy => {
                ((fill_price - reference_price) / reference_price) * BASIS_POINTS_PER_UNIT
            }
            OrderSide::Sell => {
                ((reference_price - fill_price) / reference_price) * BASIS_POINTS_PER_UNIT
            }
        };
        Some(slippage_bps.max(0.0))
    }
}

#[derive(Debug, Clone)]
struct TokenBucket {
    capacity: f64,
    refill_per_ms: f64,
    tokens: f64,
    last_refill_ms: Option<i64>,
}

impl TokenBucket {
    fn per_minute(rate_per_min: f64) -> Self {
        let capacity = clamp_orders_per_min(rate_per_min);
        Self {
            capacity,
            refill_per_ms: capacity / MILLIS_PER_MINUTE,
            tokens: capacity,
            last_refill_ms: None,
        }
    }

    fn allow(&mut self, now_ms: i64, cost: f64) -> bool {
        let cost = cost.max(0.0);
        self.refill(now_ms);
        if self.tokens + f64::EPSILON >= cost {
            self.tokens = (self.tokens - cost).max(0.0);
            true
        } else {
            false
        }
    }

    fn refill(&mut self, now_ms: i64) {
        let now_ms = match self.last_refill_ms {
            Some(last_refill_ms) => now_ms.max(last_refill_ms),
            None => {
                self.last_refill_ms = Some(now_ms);
                return;
            }
        };
        let elapsed_ms = now_ms.saturating_sub(self.last_refill_ms.unwrap_or(now_ms));
        self.last_refill_ms = Some(now_ms);
        if elapsed_ms <= 0 || self.refill_per_ms <= 0.0 {
            return;
        }
        self.tokens = (self.tokens + (elapsed_ms as f64 * self.refill_per_ms)).min(self.capacity);
    }
}

fn clamp_orders_per_min(value: f64) -> f64 {
    value.clamp(MIN_ORDERS_PER_MIN, MAX_ORDERS_PER_MIN)
}

fn parse_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "on" => Some(true),
        "0" | "false" | "no" | "n" | "off" => Some(false),
        _ => None,
    }
}

fn parse_f64(raw: &str) -> Option<f64> {
    raw.trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

fn parse_i64(raw: &str) -> Option<i64> {
    parse_f64(raw).map(|value| value as i64)
}

fn parse_usize(raw: &str) -> Option<usize> {
    parse_f64(raw)
        .filter(|value| *value >= 0.0)
        .map(|value| value as usize)
}

fn env_raw(name: &str) -> Option<String> {
    env::var(name).ok().map(|value| value.trim().to_string())
}

fn normalise_symbol(symbol: &str) -> String {
    symbol.trim().to_ascii_uppercase()
}

fn contains_clear_token(raw: &str) -> bool {
    ["clear", "resume", "off", "unpause"]
        .iter()
        .any(|token| raw.contains(token))
}

fn utc_day(ts_ms: i64) -> Option<String> {
    Utc.timestamp_millis_opt(ts_ms)
        .single()
        .map(|dt| dt.date_naive().to_string())
}

fn median<I>(values: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut values: Vec<f64> = values.into_iter().collect();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{env_lock, EnvGuard};
    use tempfile::tempdir;

    fn utc_ms(year: i32, month: u32, day: u32) -> i64 {
        Utc.with_ymd_and_hms(year, month, day, 0, 0, 0)
            .single()
            .unwrap()
            .timestamp_millis()
    }

    #[test]
    fn manual_kill_from_env_blocks_entries_but_not_reducing_orders() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let _env = EnvGuard::set(&[
            ("AI_QUANT_KILL_SWITCH", Some("1")),
            ("AI_QUANT_KILL_SWITCH_MODE", Some("close_only")),
            ("AI_QUANT_KILL_SWITCH_FILE", None),
        ]);
        let day_one_ms = utc_ms(2025, 3, 8);

        let mut risk = LiveRiskManager::from_env();
        risk.refresh(day_one_ms, Some(1_000.0));

        let blocked = risk.allow_order(OrderCheck {
            now_ms: day_one_ms,
            symbol: "ETH",
            action: OrderAction::Open,
            reduce_risk: false,
        });
        assert!(!blocked.allowed);
        assert_eq!(blocked.reason, "close_only");
        assert_eq!(blocked.kill_mode, Some(KillMode::CloseOnly));

        let allowed = risk.allow_order(OrderCheck {
            now_ms: day_one_ms + 1,
            symbol: "ETH",
            action: OrderAction::Close,
            reduce_risk: true,
        });
        assert!(allowed.allowed);
        assert_eq!(risk.kill_reason(), Some("env"));
    }

    #[test]
    fn kill_file_can_halt_and_clear_runtime() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempdir().unwrap();
        let kill_file = temp_dir.path().join("kill-switch.txt");
        fs::write(&kill_file, "halt_all").unwrap();
        let kill_path = kill_file.to_string_lossy().into_owned();
        let _env = EnvGuard::set(&[
            ("AI_QUANT_KILL_SWITCH", None),
            ("AI_QUANT_KILL_SWITCH_FILE", Some(kill_path.as_str())),
        ]);
        let day_one_ms = utc_ms(2025, 3, 8);

        let mut risk = LiveRiskManager::from_env();
        risk.refresh(day_one_ms, Some(1_000.0));
        assert_eq!(risk.kill_mode(), KillMode::HaltAll);
        assert_eq!(
            risk.kill_reason(),
            Some(format!("file:{}", kill_file.display()).as_str())
        );

        fs::write(&kill_file, "clear").unwrap();
        risk.refresh(day_one_ms + 1_000, Some(1_000.0));
        assert_eq!(risk.kill_mode(), KillMode::Off);
        assert_eq!(
            risk.kill_reason(),
            Some(format!("file_clear:{}", kill_file.display()).as_str())
        );
    }

    #[test]
    fn entry_and_cancel_token_buckets_enforce_rate_limits() {
        let day_one_ms = utc_ms(2025, 3, 8);
        let mut risk = LiveRiskManager::new(LiveRiskConfig {
            max_entry_orders_per_min: 1.0,
            max_cancels_per_min: 1.0,
            ..LiveRiskConfig::default()
        });

        let first_entry = risk.allow_order(OrderCheck {
            now_ms: day_one_ms,
            symbol: "BTC",
            action: OrderAction::Open,
            reduce_risk: false,
        });
        assert!(first_entry.allowed);

        let second_entry = risk.allow_order(OrderCheck {
            now_ms: day_one_ms + 1,
            symbol: "BTC",
            action: OrderAction::Add,
            reduce_risk: false,
        });
        assert!(!second_entry.allowed);
        assert_eq!(second_entry.reason, "entry_rate");

        let first_cancel = risk.allow_cancel(CancelCheck {
            now_ms: day_one_ms,
            symbol: "BTC",
            exchange_order_id: Some("123"),
        });
        assert!(first_cancel.allowed);

        let second_cancel = risk.allow_cancel(CancelCheck {
            now_ms: day_one_ms + 1,
            symbol: "BTC",
            exchange_order_id: Some("456"),
        });
        assert!(!second_cancel.allowed);
        assert_eq!(second_cancel.reason, "cancel_rate");
    }

    #[test]
    fn env_refresh_rebuilds_rate_limit_buckets_when_limits_change() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let day_one_ms = utc_ms(2025, 3, 8);
        let initial_env = EnvGuard::set(&[
            ("AI_QUANT_RISK_MAX_ENTRY_ORDERS_PER_MIN", Some("1")),
            ("AI_QUANT_RISK_MAX_EXIT_ORDERS_PER_MIN", Some("1")),
            ("AI_QUANT_RISK_MAX_CANCELS_PER_MIN", Some("1")),
        ]);

        let mut risk = LiveRiskManager::from_env();
        assert!(
            risk.allow_order(OrderCheck {
                now_ms: day_one_ms,
                symbol: "BTC",
                action: OrderAction::Open,
                reduce_risk: false,
            })
            .allowed
        );
        assert!(
            risk.allow_order(OrderCheck {
                now_ms: day_one_ms,
                symbol: "BTC",
                action: OrderAction::Close,
                reduce_risk: true,
            })
            .allowed
        );
        assert!(
            risk.allow_cancel(CancelCheck {
                now_ms: day_one_ms,
                symbol: "BTC",
                exchange_order_id: Some("1"),
            })
            .allowed
        );

        assert_eq!(
            risk.allow_order(OrderCheck {
                now_ms: day_one_ms + 1,
                symbol: "BTC",
                action: OrderAction::Add,
                reduce_risk: false,
            })
            .reason,
            "entry_rate"
        );
        assert_eq!(
            risk.allow_order(OrderCheck {
                now_ms: day_one_ms + 1,
                symbol: "BTC",
                action: OrderAction::Reduce,
                reduce_risk: true,
            })
            .reason,
            "exit_rate"
        );
        assert_eq!(
            risk.allow_cancel(CancelCheck {
                now_ms: day_one_ms + 1,
                symbol: "BTC",
                exchange_order_id: Some("2"),
            })
            .reason,
            "cancel_rate"
        );

        drop(initial_env);
        let _updated_env = EnvGuard::set(&[
            ("AI_QUANT_RISK_MAX_ENTRY_ORDERS_PER_MIN", Some("2")),
            ("AI_QUANT_RISK_MAX_EXIT_ORDERS_PER_MIN", Some("2")),
            ("AI_QUANT_RISK_MAX_CANCELS_PER_MIN", Some("2")),
        ]);
        risk.refresh(day_one_ms + 5_000, Some(1_000.0));

        assert!(
            risk.allow_order(OrderCheck {
                now_ms: day_one_ms + 5_001,
                symbol: "BTC",
                action: OrderAction::Add,
                reduce_risk: false,
            })
            .allowed
        );
        assert!(
            risk.allow_order(OrderCheck {
                now_ms: day_one_ms + 5_001,
                symbol: "BTC",
                action: OrderAction::Reduce,
                reduce_risk: true,
            })
            .allowed
        );
        assert!(
            risk.allow_cancel(CancelCheck {
                now_ms: day_one_ms + 5_001,
                symbol: "BTC",
                exchange_order_id: Some("3"),
            })
            .allowed
        );
    }

    #[test]
    fn note_order_sent_enforces_minimum_order_gap() {
        let day_one_ms = utc_ms(2025, 3, 8);
        let mut risk = LiveRiskManager::new(LiveRiskConfig {
            min_order_gap_ms: 1_000,
            max_entry_orders_per_min: 60.0,
            ..LiveRiskConfig::default()
        });

        risk.note_order_sent(OrderRecord {
            now_ms: day_one_ms,
            symbol: "SOL",
            action: OrderAction::Open,
            reduce_risk: false,
        });

        let blocked = risk.allow_order(OrderCheck {
            now_ms: day_one_ms + 500,
            symbol: "SOL",
            action: OrderAction::Add,
            reduce_risk: false,
        });
        assert!(!blocked.allowed);
        assert_eq!(blocked.reason, "min_order_gap");

        let allowed = risk.allow_order(OrderCheck {
            now_ms: day_one_ms + 1_001,
            symbol: "SOL",
            action: OrderAction::Add,
            reduce_risk: false,
        });
        assert!(allowed.allowed);
    }

    #[test]
    fn refresh_kills_on_drawdown_and_resets_peak_after_clear() {
        let day_one_ms = utc_ms(2025, 3, 8);
        let day_two_ms = utc_ms(2025, 3, 9);
        let mut risk = LiveRiskManager::new(LiveRiskConfig {
            max_drawdown_pct: 10.0,
            ..LiveRiskConfig::default()
        });

        risk.refresh(day_one_ms, Some(1_000.0));
        risk.refresh(day_one_ms + 1_000, Some(890.0));
        assert_eq!(risk.kill_mode(), KillMode::CloseOnly);
        assert_eq!(risk.kill_reason(), Some("drawdown"));

        risk.clear_kill("manual_clear");
        risk.refresh(day_two_ms, Some(900.0));
        assert_eq!(risk.kill_mode(), KillMode::Off);
        assert_eq!(risk.kill_since_ms(), None);
    }

    #[test]
    fn close_fills_trip_daily_loss_and_reset_on_new_utc_day() {
        let day_one_ms = utc_ms(2025, 3, 8);
        let day_two_ms = utc_ms(2025, 3, 9);
        let mut risk = LiveRiskManager::new(LiveRiskConfig {
            max_daily_loss_usd: 100.0,
            ..LiveRiskConfig::default()
        });

        risk.note_fill(FillEvent {
            ts_ms: day_one_ms,
            symbol: "BTC",
            action: OrderAction::Close,
            pnl_usd: -90.0,
            fee_usd: 15.0,
            fill_price: None,
            side: None,
            ref_mid: None,
            ref_bid: None,
            ref_ask: None,
        });
        assert_eq!(risk.kill_reason(), Some("daily_loss:2025-03-08"));

        risk.clear_kill("manual_clear");
        risk.refresh(day_two_ms, None);
        assert_eq!(risk.kill_mode(), KillMode::Off);

        risk.note_fill(FillEvent {
            ts_ms: day_two_ms,
            symbol: "BTC",
            action: OrderAction::Reduce,
            pnl_usd: -50.0,
            fee_usd: 5.0,
            fill_price: None,
            side: None,
            ref_mid: None,
            ref_bid: None,
            ref_ask: None,
        });
        assert_eq!(risk.kill_mode(), KillMode::Off);
    }

    #[test]
    fn entry_fill_slippage_guard_uses_median_bps() {
        let day_one_ms = utc_ms(2025, 3, 8);
        let mut risk = LiveRiskManager::new(LiveRiskConfig {
            slippage_guard_enabled: true,
            slippage_guard_window_fills: 3,
            slippage_guard_max_median_bps: 10.0,
            ..LiveRiskConfig::default()
        });

        for fill_price in [100.05, 100.15, 100.20] {
            risk.note_fill(FillEvent {
                ts_ms: day_one_ms,
                symbol: "ETH",
                action: OrderAction::Open,
                pnl_usd: 0.0,
                fee_usd: 0.0,
                fill_price: Some(fill_price),
                side: Some(OrderSide::Buy),
                ref_mid: None,
                ref_bid: None,
                ref_ask: Some(100.0),
            });
        }

        assert_eq!(risk.kill_mode(), KillMode::CloseOnly);
        assert_eq!(risk.kill_reason(), Some("slippage_guard"));
    }
}
