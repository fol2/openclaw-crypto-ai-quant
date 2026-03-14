use aiq_runtime_core::paper::{restore_paper_state, PaperBootstrapReport, PaperPositionState};
use aiq_runtime_core::runtime::RuntimeBootstrap;
use anyhow::{anyhow, Result};
use bt_core::behaviour::ResolvedBehaviourPlan;
use bt_core::candle::OhlcvBar;
use bt_core::config::{Confidence, MacdMode, Signal, StrategyConfig};
use bt_core::decision_kernel::{
    self, CooldownParams, FillEvent, KernelParams, MarketEvent, MarketSignal, OrderIntent,
    OrderIntentKind, PositionSide, StrategyState,
};
use bt_core::indicators::{IndicatorBank, IndicatorSnapshot};
use bt_core::kernel_entries::{evaluate_entry_with_behaviour_plan, EntryParams};
use bt_core::kernel_exits::ExitParams;
use bt_core::signals::behaviour::BehaviourTrace;
use bt_core::signals::gates::{self, GateResult};
use chrono::{DateTime, TimeZone, Utc};
use risk_core::{
    compute_entry_sizing, compute_pyramid_sizing, evaluate_exposure_guard, ConfidenceTier,
    EntrySizingInput, ExposureGuardInput, PyramidSizingInput,
};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::Serialize;
use std::path::Path;

use crate::decision_events;
use crate::paper_export;

pub struct PaperRunOnceInput<'a> {
    pub config: &'a StrategyConfig,
    pub runtime_bootstrap: RuntimeBootstrap,
    pub paper_db: &'a Path,
    pub candles_db: &'a Path,
    pub symbol: &'a str,
    pub btc_symbol: &'a str,
    pub lookback_bars: usize,
    pub exported_at_ms: Option<i64>,
    pub dry_run: bool,
}

pub(crate) struct PrepareSymbolStepInput<'a> {
    pub(crate) config: &'a StrategyConfig,
    pub(crate) profile_override: Option<&'a str>,
    pub(crate) prior_position: Option<&'a PaperPositionState>,
    pub(crate) candles_db: &'a Path,
    pub(crate) symbol: &'a str,
    pub(crate) btc_symbol: &'a str,
    pub(crate) lookback_bars: usize,
    pub(crate) step_close_ts_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExecutionPlan {
    pub(crate) requested_notional_usd: Option<f64>,
    pub(crate) leverage: f64,
    pub(crate) allow_pyramid: bool,
    pub(crate) warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ExecutionMetadata {
    pub(crate) signal: Signal,
    pub(crate) confidence: Confidence,
    pub(crate) entry_adx_threshold: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ProjectionInput<'a> {
    pub(crate) config_fingerprint: &'a str,
    pub(crate) run_fingerprint: &'a str,
    pub(crate) symbol: &'a str,
    pub(crate) pre_state: &'a StrategyState,
    pub(crate) prior_position: Option<&'a PaperPositionState>,
    pub(crate) post_state: &'a StrategyState,
    pub(crate) intents: &'a [OrderIntent],
    pub(crate) fills: &'a [FillEvent],
    pub(crate) snap: &'a IndicatorSnapshot,
    pub(crate) execution_metadata: ExecutionMetadata,
    pub(crate) ts_ms: i64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BuildExecutionPlanInput<'a> {
    pub(crate) cfg: &'a StrategyConfig,
    pub(crate) behaviour_plan: &'a ResolvedBehaviourPlan,
    pub(crate) pre_state: &'a StrategyState,
    pub(crate) symbol: &'a str,
    pub(crate) execution_ts_ms: i64,
    pub(crate) prior_position: Option<&'a PaperPositionState>,
    pub(crate) snap: &'a IndicatorSnapshot,
    pub(crate) execution_metadata: ExecutionMetadata,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedSymbolStep {
    pub symbol: String,
    pub snap: IndicatorSnapshot,
    pub gate_result: GateResult,
    pub gate_behaviour_trace: Vec<BehaviourTrace>,
    pub signal_behaviour_trace: Vec<BehaviourTrace>,
    pub ema_slow_slope_pct: f64,
    pub execution_metadata: ExecutionMetadata,
    pub behaviour_plan: ResolvedBehaviourPlan,
    pub config: StrategyConfig,
    pub prior_position: Option<PaperPositionState>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperRunOnceReport {
    pub ok: bool,
    pub dry_run: bool,
    pub symbol: String,
    pub interval: String,
    pub runtime_bootstrap: RuntimeBootstrap,
    pub paper_bootstrap: PaperBootstrapReport,
    pub snapshot_source: String,
    pub snapshot_exported_at_ms: i64,
    pub price: f64,
    pub btc_bullish: Option<bool>,
    pub ema_slow_slope_pct: f64,
    pub intent_count: usize,
    pub fill_count: usize,
    pub action_codes: Vec<String>,
    pub trades_written: usize,
    pub position_state_written: bool,
    pub runtime_cooldowns_written: bool,
    pub behaviour_trace: Vec<BehaviourTrace>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

pub(crate) fn action_codes_for_symbol(
    symbol: &str,
    pre_state: &StrategyState,
    intents: &[OrderIntent],
    fills: &[FillEvent],
) -> Vec<String> {
    let mut codes = Vec::new();
    let mut current_qty = pre_state
        .positions
        .get(symbol)
        .map(|position| position.quantity)
        .unwrap_or(0.0);

    for (intent, fill) in intents.iter().zip(fills.iter()) {
        if intent.symbol != symbol {
            continue;
        }
        let code = match intent.kind {
            OrderIntentKind::Open => {
                current_qty = fill.quantity;
                "OPEN"
            }
            OrderIntentKind::Add => {
                current_qty += fill.quantity;
                "ADD"
            }
            OrderIntentKind::Close => {
                if fill.quantity < current_qty {
                    current_qty = (current_qty - fill.quantity).max(0.0);
                    "REDUCE"
                } else {
                    current_qty = 0.0;
                    "CLOSE"
                }
            }
            OrderIntentKind::Hold => continue,
            OrderIntentKind::Reverse => "REVERSE",
        };
        codes.push(code.to_string());
    }

    codes
}

pub fn run_once(input: PaperRunOnceInput<'_>) -> Result<PaperRunOnceReport> {
    if !input.paper_db.exists() {
        anyhow::bail!("paper db not found: {}", input.paper_db.display());
    }
    if !input.candles_db.exists() {
        anyhow::bail!("candles db not found: {}", input.candles_db.display());
    }

    let symbol = input.symbol.trim().to_ascii_uppercase();
    if symbol.is_empty() {
        anyhow::bail!("symbol must be non-empty");
    }
    let btc_symbol = input.btc_symbol.trim().to_ascii_uppercase();
    let interval = input.config.engine.interval.trim().to_string();

    let symbol_bars = load_recent_bars(
        input.candles_db,
        &symbol,
        &interval,
        input.lookback_bars.max(64),
    )?;
    let (snap, ema_history) = build_latest_snapshot(input.config, &symbol_bars)?;
    let exported_at_ms = input
        .exported_at_ms
        .unwrap_or_else(|| Utc::now().timestamp_millis());

    let snapshot = paper_export::export_paper_snapshot(input.paper_db, exported_at_ms)?;
    let (paper_state, paper_bootstrap) =
        restore_paper_state(&snapshot).map_err(anyhow::Error::msg)?;
    let prior_position = paper_state.positions.get(&symbol).cloned();
    let pre_state = paper_state.into_strategy_state();

    let btc_bullish = if btc_symbol != symbol {
        let btc_bars = load_recent_bars(
            input.candles_db,
            &btc_symbol,
            &interval,
            input.lookback_bars.max(64),
        )?;
        let (btc_snap, _) = build_latest_snapshot(input.config, &btc_bars)?;
        Some(btc_snap.close > btc_snap.ema_slow)
    } else {
        Some(snap.close > snap.ema_slow)
    };

    let slope_window = input.config.thresholds.entry.slow_drift_slope_window.max(1);
    let ema_slow_slope_pct = compute_ema_slow_slope(&ema_history, slope_window, snap.close);
    let behaviour_plan = input
        .config
        .resolve_behaviour_plan(None)
        .map_err(anyhow::Error::msg)?;
    let gate_report = gates::check_gates_with_behaviour_plan(
        &snap,
        input.config,
        &symbol,
        btc_bullish,
        ema_slow_slope_pct,
        &behaviour_plan.gates,
    );

    let entry_params = build_entry_params(input.config);
    let entry_result = evaluate_entry_with_behaviour_plan(
        &snap,
        &gate_report.result,
        &entry_params,
        ema_slow_slope_pct,
        &behaviour_plan.signal_modes,
        &behaviour_plan.signal_confidence,
    );
    let execution_metadata = ExecutionMetadata {
        signal: entry_result.signal,
        confidence: to_bt_confidence(entry_result.confidence),
        entry_adx_threshold: entry_result.entry_adx_threshold,
    };
    let execution_plan = build_execution_plan(BuildExecutionPlanInput {
        cfg: input.config,
        behaviour_plan: &behaviour_plan,
        pre_state: &pre_state,
        symbol: &symbol,
        execution_ts_ms: exported_at_ms,
        prior_position: prior_position.as_ref(),
        snap: &snap,
        execution_metadata,
    });
    let params = build_kernel_params(
        input.config,
        &behaviour_plan,
        execution_plan.leverage,
        execution_plan.allow_pyramid,
    );
    let event = MarketEvent {
        schema_version: 1,
        event_id: snap.t.max(0) as u64,
        timestamp_ms: snap.t,
        symbol: symbol.clone(),
        signal: MarketSignal::Evaluate,
        price: snap.close,
        notional_hint_usd: execution_plan.requested_notional_usd,
        close_fraction: None,
        fee_role: None,
        funding_rate: None,
        indicators: Some(snap.clone()),
        gate_result: Some(gate_report.result.clone()),
        ema_slow_slope_pct: Some(ema_slow_slope_pct),
    };
    let decision = decision_kernel::step(&pre_state, &event, &params);

    let (trades_written, position_state_written, runtime_cooldowns_written) = if input.dry_run {
        (0usize, false, false)
    } else {
        {
            let projection = ProjectionInput {
                config_fingerprint: input.runtime_bootstrap.config_fingerprint.as_str(),
                run_fingerprint: "paper_run_once",
                symbol: &symbol,
                pre_state: &pre_state,
                prior_position: prior_position.as_ref(),
                post_state: &decision.state,
                intents: &decision.intents,
                fills: &decision.fills,
                snap: &snap,
                execution_metadata,
                ts_ms: exported_at_ms,
            };
            apply_decision_projection(input.paper_db, &projection)?
        }
    };

    Ok(PaperRunOnceReport {
        ok: decision.diagnostics.errors.is_empty(),
        dry_run: input.dry_run,
        symbol: symbol.clone(),
        interval,
        runtime_bootstrap: input.runtime_bootstrap,
        paper_bootstrap,
        snapshot_source: snapshot.source,
        snapshot_exported_at_ms: snapshot.exported_at_ms,
        price: snap.close,
        btc_bullish,
        ema_slow_slope_pct,
        intent_count: decision.intents.len(),
        fill_count: decision.fills.len(),
        action_codes: action_codes_for_symbol(
            &symbol,
            &pre_state,
            &decision.intents,
            &decision.fills,
        ),
        trades_written,
        position_state_written,
        runtime_cooldowns_written,
        behaviour_trace: gate_report
            .behaviour_trace
            .iter()
            .chain(entry_result.behaviour_trace.iter())
            .cloned()
            .chain(decision.diagnostics.behaviour_trace.clone())
            .collect(),
        warnings: execution_plan
            .warnings
            .into_iter()
            .chain(decision.diagnostics.warnings)
            .collect(),
        errors: decision.diagnostics.errors,
    })
}

pub(crate) fn prepare_symbol_step(input: PrepareSymbolStepInput<'_>) -> Result<PreparedSymbolStep> {
    let PrepareSymbolStepInput {
        config,
        profile_override,
        prior_position,
        candles_db,
        symbol,
        btc_symbol,
        lookback_bars,
        step_close_ts_ms,
    } = input;
    let interval = config.engine.interval.trim().to_string();
    let symbol_bars = load_recent_bars_as_of(
        candles_db,
        symbol,
        &interval,
        lookback_bars.max(64),
        step_close_ts_ms,
    )?;
    let (snap, ema_history) = build_latest_snapshot(config, &symbol_bars)?;

    let btc_bullish = if btc_symbol != symbol {
        let btc_bars = load_recent_bars_as_of(
            candles_db,
            btc_symbol,
            &interval,
            lookback_bars.max(64),
            step_close_ts_ms,
        )?;
        let (btc_snap, _) = build_latest_snapshot(config, &btc_bars)?;
        Some(btc_snap.close > btc_snap.ema_slow)
    } else {
        Some(snap.close > snap.ema_slow)
    };

    let slope_window = config.thresholds.entry.slow_drift_slope_window.max(1);
    let ema_slow_slope_pct = compute_ema_slow_slope(&ema_history, slope_window, snap.close);
    let behaviour_plan = config
        .resolve_behaviour_plan(profile_override)
        .map_err(anyhow::Error::msg)?;
    let gate_report = gates::check_gates_with_behaviour_plan(
        &snap,
        config,
        symbol,
        btc_bullish,
        ema_slow_slope_pct,
        &behaviour_plan.gates,
    );

    let entry_report = bt_core::signals::entry::generate_signal_with_behaviour_plan(
        &snap,
        &gate_report.result,
        config,
        ema_slow_slope_pct,
        &behaviour_plan.signal_modes,
        &behaviour_plan.signal_confidence,
    );
    let execution_metadata = ExecutionMetadata {
        signal: entry_report.signal,
        confidence: entry_report.confidence,
        entry_adx_threshold: entry_report.entry_adx_threshold,
    };

    Ok(PreparedSymbolStep {
        symbol: symbol.to_string(),
        snap,
        gate_result: gate_report.result,
        gate_behaviour_trace: gate_report.behaviour_trace,
        signal_behaviour_trace: entry_report.behaviour_trace,
        ema_slow_slope_pct,
        execution_metadata,
        behaviour_plan,
        config: config.clone(),
        prior_position: prior_position.cloned(),
    })
}

pub(crate) fn execute_prepared_symbol_step(
    pre_state: &StrategyState,
    prepared: &PreparedSymbolStep,
    decision_ts_ms: i64,
) -> (decision_kernel::DecisionResult, ExecutionPlan) {
    execute_prepared_symbol_step_with_allow_pyramid_override(
        pre_state,
        prepared,
        decision_ts_ms,
        None,
    )
}

pub(crate) fn execute_prepared_symbol_step_with_allow_pyramid_override(
    pre_state: &StrategyState,
    prepared: &PreparedSymbolStep,
    decision_ts_ms: i64,
    allow_pyramid_override: Option<bool>,
) -> (decision_kernel::DecisionResult, ExecutionPlan) {
    let execution_plan = build_execution_plan(BuildExecutionPlanInput {
        cfg: &prepared.config,
        behaviour_plan: &prepared.behaviour_plan,
        pre_state,
        symbol: &prepared.symbol,
        execution_ts_ms: decision_ts_ms,
        prior_position: prepared.prior_position.as_ref(),
        snap: &prepared.snap,
        execution_metadata: prepared.execution_metadata,
    });
    let params = build_kernel_params(
        &prepared.config,
        &prepared.behaviour_plan,
        execution_plan.leverage,
        allow_pyramid_override.unwrap_or(execution_plan.allow_pyramid),
    );
    let event = MarketEvent {
        schema_version: 1,
        event_id: decision_ts_ms.max(0) as u64,
        timestamp_ms: decision_ts_ms,
        symbol: prepared.symbol.clone(),
        signal: MarketSignal::Evaluate,
        price: prepared.snap.close,
        notional_hint_usd: execution_plan.requested_notional_usd,
        close_fraction: None,
        fee_role: None,
        funding_rate: None,
        indicators: Some(prepared.snap.clone()),
        gate_result: Some(prepared.gate_result.clone()),
        ema_slow_slope_pct: Some(prepared.ema_slow_slope_pct),
    };
    let decision = decision_kernel::step(pre_state, &event, &params);
    (decision, execution_plan)
}

pub(crate) fn load_recent_bars(
    db_path: &Path,
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<OhlcvBar>> {
    load_recent_bars_as_of(db_path, symbol, interval, limit, None)
}

pub(crate) fn load_recent_bars_as_of(
    db_path: &Path,
    symbol: &str,
    interval: &str,
    limit: usize,
    max_close_ts_ms: Option<i64>,
) -> Result<Vec<OhlcvBar>> {
    let conn = Connection::open(db_path)?;
    let sql = if max_close_ts_ms.is_some() {
        "SELECT t, COALESCE(t_close, t), o, h, l, c, v, COALESCE(n, 0)
         FROM candles
         WHERE symbol = ?1 AND interval = ?2 AND COALESCE(t_close, t) <= ?3
         ORDER BY COALESCE(t_close, t) DESC, t DESC
         LIMIT ?4"
    } else {
        "SELECT t, COALESCE(t_close, t), o, h, l, c, v, COALESCE(n, 0)
         FROM candles
         WHERE symbol = ?1 AND interval = ?2
         ORDER BY COALESCE(t_close, t) DESC, t DESC
         LIMIT ?3"
    };
    let mut stmt = conn.prepare(sql)?;
    let mut bars = if let Some(max_close_ts_ms) = max_close_ts_ms {
        stmt.query_map((symbol, interval, max_close_ts_ms, limit as i64), |row| {
            Ok(OhlcvBar {
                t: row.get(0)?,
                t_close: row.get(1)?,
                o: row.get(2)?,
                h: row.get(3)?,
                l: row.get(4)?,
                c: row.get(5)?,
                v: row.get(6)?,
                n: row.get(7)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?
    } else {
        stmt.query_map((symbol, interval, limit as i64), |row| {
            Ok(OhlcvBar {
                t: row.get(0)?,
                t_close: row.get(1)?,
                o: row.get(2)?,
                h: row.get(3)?,
                l: row.get(4)?,
                c: row.get(5)?,
                v: row.get(6)?,
                n: row.get(7)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?
    };
    bars.reverse();
    if bars.len() < 2 {
        anyhow::bail!("not enough bars for {} {}", symbol, interval);
    }
    Ok(bars)
}

pub(crate) fn build_latest_snapshot(
    cfg: &StrategyConfig,
    bars: &[OhlcvBar],
) -> Result<(IndicatorSnapshot, Vec<f64>)> {
    let mut bank = IndicatorBank::new_with_ave_window(
        &cfg.indicators,
        cfg.filters.use_stoch_rsi_filter,
        cfg.effective_ave_avg_atr_window(),
    );
    let mut ema_history = Vec::with_capacity(bars.len());
    let mut latest: Option<IndicatorSnapshot> = None;
    for bar in bars {
        let snap = bank.update(bar);
        ema_history.push(snap.ema_slow);
        latest = Some(snap);
    }
    let latest = latest.ok_or_else(|| anyhow!("indicator snapshot unavailable"))?;
    Ok((latest, ema_history))
}

pub(crate) fn compute_ema_slow_slope(history: &[f64], window: usize, current_close: f64) -> f64 {
    if history.len() < window || current_close <= 0.0 {
        return 0.0;
    }
    let current = history[history.len() - 1];
    let past = history[history.len() - window];
    (current - past) / current_close
}

pub(crate) fn build_kernel_params(
    cfg: &StrategyConfig,
    behaviour_plan: &ResolvedBehaviourPlan,
    leverage: f64,
    allow_pyramid: bool,
) -> KernelParams {
    KernelParams {
        default_notional_usd: 0.0,
        min_notional_usd: 0.0,
        max_notional_usd: f64::MAX,
        allow_pyramid,
        allow_reverse: false,
        leverage: leverage.max(1.0),
        exit_params: Some(build_exit_params(cfg)),
        entry_params: Some(build_entry_params(cfg)),
        cooldown_params: Some(build_cooldown_params(cfg)),
        behaviour_plan: behaviour_plan.clone(),
        ..KernelParams::default()
    }
}

pub(crate) fn build_exit_params(cfg: &StrategyConfig) -> ExitParams {
    let t = &cfg.trade;
    ExitParams {
        sl_atr_mult: t.sl_atr_mult,
        tp_atr_mult: t.tp_atr_mult,
        trailing_start_atr: t.trailing_start_atr,
        trailing_distance_atr: t.trailing_distance_atr,
        enable_partial_tp: t.enable_partial_tp,
        tp_partial_pct: t.tp_partial_pct,
        tp_partial_atr_mult: t.tp_partial_atr_mult,
        tp_partial_min_notional_usd: t.tp_partial_min_notional_usd,
        enable_breakeven_stop: t.enable_breakeven_stop,
        breakeven_start_atr: t.breakeven_start_atr,
        breakeven_buffer_atr: t.breakeven_buffer_atr,
        enable_vol_buffered_trailing: t.enable_vol_buffered_trailing,
        block_exits_on_extreme_dev: t.block_exits_on_extreme_dev,
        glitch_price_dev_pct: t.glitch_price_dev_pct,
        glitch_atr_mult: t.glitch_atr_mult,
        smart_exit_adx_exhaustion_lt: t.smart_exit_adx_exhaustion_lt,
        tsme_min_profit_atr: t.tsme_min_profit_atr,
        tsme_require_adx_slope_negative: t.tsme_require_adx_slope_negative,
        enable_rsi_overextension_exit: t.enable_rsi_overextension_exit,
        rsi_exit_profit_atr_switch: t.rsi_exit_profit_atr_switch,
        rsi_exit_ub_lo_profit: t.rsi_exit_ub_lo_profit,
        rsi_exit_ub_hi_profit: t.rsi_exit_ub_hi_profit,
        rsi_exit_lb_lo_profit: t.rsi_exit_lb_lo_profit,
        rsi_exit_lb_hi_profit: t.rsi_exit_lb_hi_profit,
        rsi_exit_ub_lo_profit_low_conf: t.rsi_exit_ub_lo_profit_low_conf,
        rsi_exit_lb_lo_profit_low_conf: t.rsi_exit_lb_lo_profit_low_conf,
        rsi_exit_ub_hi_profit_low_conf: t.rsi_exit_ub_hi_profit_low_conf,
        rsi_exit_lb_hi_profit_low_conf: t.rsi_exit_lb_hi_profit_low_conf,
        smart_exit_adx_exhaustion_lt_low_conf: t.smart_exit_adx_exhaustion_lt_low_conf,
        require_macro_alignment: cfg.filters.require_macro_alignment,
        trailing_start_atr_low_conf: t.trailing_start_atr_low_conf,
        trailing_distance_atr_low_conf: t.trailing_distance_atr_low_conf,
    }
}

pub(crate) fn build_entry_params(cfg: &StrategyConfig) -> EntryParams {
    let entry = &cfg.thresholds.entry;
    let stoch = &cfg.thresholds.stoch_rsi;
    EntryParams {
        macd_mode: match entry.macd_hist_entry_mode {
            MacdMode::Accel => 0,
            MacdMode::Sign => 1,
            MacdMode::None => 2,
        },
        stoch_block_long_gt: stoch.block_long_if_k_gt,
        stoch_block_short_lt: stoch.block_short_if_k_lt,
        high_conf_volume_mult: entry.high_conf_volume_mult,
        enable_pullback: entry.enable_pullback_entries,
        pullback_confidence: match entry.pullback_confidence {
            Confidence::Low => 0,
            Confidence::Medium => 1,
            Confidence::High => 2,
        },
        pullback_min_adx: entry.pullback_min_adx,
        pullback_rsi_long_min: entry.pullback_rsi_long_min,
        pullback_rsi_short_max: entry.pullback_rsi_short_max,
        pullback_require_macd_sign: entry.pullback_require_macd_sign,
        enable_slow_drift: entry.enable_slow_drift_entries,
        slow_drift_min_slope_pct: entry.slow_drift_min_slope_pct,
        slow_drift_min_adx: entry.slow_drift_min_adx,
        slow_drift_rsi_long_min: entry.slow_drift_rsi_long_min,
        slow_drift_rsi_short_max: entry.slow_drift_rsi_short_max,
        slow_drift_require_macd_sign: entry.slow_drift_require_macd_sign,
    }
}

pub(crate) fn build_cooldown_params(cfg: &StrategyConfig) -> CooldownParams {
    CooldownParams {
        entry_cooldown_s: cfg.trade.entry_cooldown_s as u32,
        exit_cooldown_s: cfg.trade.exit_cooldown_s as u32,
        reentry_cooldown_minutes: cfg.trade.reentry_cooldown_minutes as u32,
        reentry_cooldown_min_mins: cfg.trade.reentry_cooldown_min_mins as u32,
        reentry_cooldown_max_mins: cfg.trade.reentry_cooldown_max_mins as u32,
    }
}

pub(crate) fn build_execution_plan(input: BuildExecutionPlanInput<'_>) -> ExecutionPlan {
    let BuildExecutionPlanInput {
        cfg,
        behaviour_plan,
        pre_state,
        symbol,
        execution_ts_ms,
        prior_position,
        snap,
        execution_metadata,
    } = input;
    let entry_side = signal_to_side(execution_metadata.signal);
    let base_leverage = prior_position
        .map(|position| position.leverage.max(1.0))
        .unwrap_or_else(|| cfg.trade.leverage.max(1.0));
    let mut plan = ExecutionPlan {
        requested_notional_usd: None,
        leverage: base_leverage,
        allow_pyramid: cfg.trade.enable_pyramiding
            && behaviour_plan
                .entry_progression
                .is_enabled("entry.progression.pyramiding"),
        warnings: Vec::new(),
    };

    let Some(entry_side) = entry_side else {
        plan.allow_pyramid = false;
        return plan;
    };

    let equity = pre_state.cash_usd.max(0.0);
    let total_margin_used = total_margin_used(pre_state);
    let confidence = to_confidence_tier(execution_metadata.confidence);

    match pre_state.positions.get_key_value(symbol) {
        Some((_symbol, position)) if position.side == entry_side => {
            if !behaviour_plan
                .entry_progression
                .is_enabled("entry.progression.pyramiding")
                || !cfg.trade.enable_pyramiding
            {
                plan.allow_pyramid = false;
                return plan;
            }
            if position.adds_count >= cfg.trade.max_adds_per_symbol as u32 {
                plan.allow_pyramid = false;
                plan.warnings.push(format!(
                    "skip add for {}: adds_count {} reached max_adds_per_symbol {}",
                    position.symbol, position.adds_count, cfg.trade.max_adds_per_symbol
                ));
                return plan;
            }
            if !execution_metadata
                .confidence
                .meets_min(cfg.trade.add_min_confidence)
            {
                plan.allow_pyramid = false;
                plan.warnings.push(format!(
                    "skip add for {}: confidence {:?} below add_min_confidence {:?}",
                    position.symbol, execution_metadata.confidence, cfg.trade.add_min_confidence
                ));
                return plan;
            }
            if behaviour_plan
                .entry_progression
                .is_enabled("entry.progression.tp1_block_add")
                && prior_position
                    .map(|position| position.tp1_taken)
                    .unwrap_or(position.tp1_taken)
            {
                plan.allow_pyramid = false;
                plan.warnings.push(format!(
                    "skip add for {}: tp1_taken blocks same-direction pyramiding",
                    position.symbol
                ));
                return plan;
            }
            let last_add_time_ms = prior_position
                .map(|position| position.last_add_time_ms)
                .unwrap_or(0);
            if behaviour_plan
                .entry_progression
                .is_enabled("entry.progression.add_cooldown")
                && last_add_time_ms > 0
                && execution_ts_ms.saturating_sub(last_add_time_ms)
                    < (cfg.trade.add_cooldown_minutes as i64 * 60_000)
            {
                plan.allow_pyramid = false;
                plan.warnings.push(format!(
                    "skip add for {}: add cooldown is still active",
                    position.symbol
                ));
                return plan;
            }
            match compute_pyramid_sizing(PyramidSizingInput {
                equity,
                price: snap.close,
                leverage: prior_position
                    .map(|position| position.leverage.max(1.0))
                    .unwrap_or_else(|| position.notional_usd / position.margin_usd.max(1e-12)),
                allocation_pct: cfg.trade.allocation_pct,
                add_fraction_of_base_margin: cfg.trade.add_fraction_of_base_margin,
                min_notional_usd: cfg.trade.min_notional_usd,
                bump_to_min_notional: cfg.trade.bump_to_min_notional
                    && behaviour_plan
                        .entry_sizing
                        .is_enabled("entry.sizing.min_notional_bump"),
            }) {
                Some(add_sizing) => {
                    let exposure = if behaviour_plan.risk.is_enabled("risk.exposure_guard") {
                        evaluate_exposure_guard(ExposureGuardInput {
                            open_positions: pre_state.positions.len(),
                            max_open_positions: None,
                            total_margin_used: total_margin_used + add_sizing.add_margin,
                            equity,
                            max_total_margin_pct: cfg.trade.max_total_margin_pct,
                            allow_zero_margin_headroom: true,
                        })
                    } else {
                        evaluate_exposure_guard(ExposureGuardInput {
                            open_positions: pre_state.positions.len(),
                            max_open_positions: None,
                            total_margin_used,
                            equity,
                            max_total_margin_pct: 0.0,
                            allow_zero_margin_headroom: true,
                        })
                    };
                    if exposure.allowed {
                        plan.requested_notional_usd = Some(add_sizing.add_notional);
                        plan.leverage = add_sizing.add_notional / add_sizing.add_margin.max(1e-12);
                    } else {
                        plan.allow_pyramid = false;
                        plan.warnings.push(format!(
                            "skip add for {}: exposure guard blocked margin headroom {:.4}",
                            position.symbol, exposure.margin_headroom
                        ));
                    }
                }
                None => {
                    plan.allow_pyramid = false;
                    plan.warnings.push(format!(
                        "skip add for {}: add sizing returned no executable notional",
                        position.symbol
                    ));
                }
            }
        }
        Some((symbol, _position)) => {
            plan.allow_pyramid = false;
            plan.warnings.push(format!(
                "same-symbol position on {} is opposite-side; entry notional deferred to close-only path",
                symbol
            ));
        }
        None => {
            if !execution_metadata
                .confidence
                .meets_min(cfg.trade.entry_min_confidence)
            {
                plan.allow_pyramid = false;
                plan.warnings.push(format!(
                    "skip entry for {}: confidence {:?} below entry_min_confidence {:?}",
                    symbol, execution_metadata.confidence, cfg.trade.entry_min_confidence
                ));
                return plan;
            }
            let exposure = if behaviour_plan.risk.is_enabled("risk.exposure_guard") {
                evaluate_exposure_guard(ExposureGuardInput {
                    open_positions: pre_state.positions.len(),
                    max_open_positions: Some(cfg.trade.max_open_positions),
                    total_margin_used,
                    equity,
                    max_total_margin_pct: cfg.trade.max_total_margin_pct,
                    allow_zero_margin_headroom: false,
                })
            } else {
                evaluate_exposure_guard(ExposureGuardInput {
                    open_positions: pre_state.positions.len(),
                    max_open_positions: None,
                    total_margin_used,
                    equity,
                    max_total_margin_pct: 0.0,
                    allow_zero_margin_headroom: true,
                })
            };
            if !exposure.allowed {
                plan.allow_pyramid = false;
                plan.warnings.push(format!(
                    "skip entry for {}: exposure guard blocked margin headroom {:.4}",
                    symbol, exposure.margin_headroom
                ));
                return plan;
            }

            let sizing = compute_entry_sizing(EntrySizingInput {
                equity,
                price: snap.close,
                atr: snap.atr,
                adx: snap.adx,
                confidence,
                allocation_pct: cfg.trade.allocation_pct,
                enable_dynamic_sizing: cfg.trade.enable_dynamic_sizing
                    && behaviour_plan
                        .entry_sizing
                        .is_enabled("entry.sizing.dynamic"),
                confidence_mult_high: if behaviour_plan
                    .entry_sizing
                    .is_enabled("entry.sizing.confidence_multiplier")
                {
                    cfg.trade.confidence_mult_high
                } else {
                    1.0
                },
                confidence_mult_medium: if behaviour_plan
                    .entry_sizing
                    .is_enabled("entry.sizing.confidence_multiplier")
                {
                    cfg.trade.confidence_mult_medium
                } else {
                    1.0
                },
                confidence_mult_low: if behaviour_plan
                    .entry_sizing
                    .is_enabled("entry.sizing.confidence_multiplier")
                {
                    cfg.trade.confidence_mult_low
                } else {
                    1.0
                },
                adx_sizing_min_mult: if behaviour_plan
                    .entry_sizing
                    .is_enabled("entry.sizing.adx_multiplier")
                {
                    cfg.trade.adx_sizing_min_mult
                } else {
                    1.0
                },
                adx_sizing_full_adx: if behaviour_plan
                    .entry_sizing
                    .is_enabled("entry.sizing.adx_multiplier")
                {
                    cfg.trade.adx_sizing_full_adx
                } else {
                    0.0
                },
                vol_baseline_pct: if behaviour_plan
                    .entry_sizing
                    .is_enabled("entry.sizing.volatility_scalar")
                {
                    cfg.trade.vol_baseline_pct
                } else {
                    0.0
                },
                vol_scalar_min: if behaviour_plan
                    .entry_sizing
                    .is_enabled("entry.sizing.volatility_scalar")
                {
                    cfg.trade.vol_scalar_min
                } else {
                    1.0
                },
                vol_scalar_max: if behaviour_plan
                    .entry_sizing
                    .is_enabled("entry.sizing.volatility_scalar")
                {
                    cfg.trade.vol_scalar_max
                } else {
                    1.0
                },
                leverage_low: cfg.trade.leverage_low,
                leverage_medium: cfg.trade.leverage_medium,
                leverage_high: cfg.trade.leverage_high,
            });
            let mut margin_used = sizing.margin_used;
            let mut notional = sizing.notional;
            plan.leverage = sizing.leverage.max(1.0);

            if margin_used > exposure.margin_headroom {
                let ratio = if margin_used > 1e-12 {
                    exposure.margin_headroom / margin_used
                } else {
                    0.0
                };
                margin_used = exposure.margin_headroom.max(0.0);
                notional *= ratio.max(0.0);
            }

            if notional < cfg.trade.min_notional_usd {
                if cfg.trade.bump_to_min_notional
                    && behaviour_plan
                        .entry_sizing
                        .is_enabled("entry.sizing.min_notional_bump")
                {
                    let bumped_margin = cfg.trade.min_notional_usd / plan.leverage.max(1.0);
                    if bumped_margin > exposure.margin_headroom {
                        plan.warnings.push(format!(
                            "skip entry: bumped min-notional margin {:.4} exceeds headroom {:.4}",
                            bumped_margin, exposure.margin_headroom
                        ));
                        return plan;
                    }
                    notional = cfg.trade.min_notional_usd;
                    margin_used = bumped_margin;
                } else {
                    let _ = margin_used;
                    plan.warnings.push(format!(
                        "skip entry: requested notional {:.4} is below min_notional_usd {:.4}",
                        notional, cfg.trade.min_notional_usd
                    ));
                    return plan;
                }
            }

            if notional > 0.0 && margin_used >= 0.0 {
                plan.requested_notional_usd = Some(notional);
            }
            plan.allow_pyramid = false;
        }
    }

    plan
}

pub(crate) fn signal_to_side(signal: Signal) -> Option<PositionSide> {
    match signal {
        Signal::Buy => Some(PositionSide::Long),
        Signal::Sell => Some(PositionSide::Short),
        Signal::Neutral => None,
    }
}

pub(crate) fn to_bt_confidence(raw: u8) -> Confidence {
    match raw {
        2 => Confidence::High,
        1 => Confidence::Medium,
        _ => Confidence::Low,
    }
}

pub(crate) fn to_confidence_tier(confidence: Confidence) -> ConfidenceTier {
    match confidence {
        Confidence::High => ConfidenceTier::High,
        Confidence::Medium => ConfidenceTier::Medium,
        Confidence::Low => ConfidenceTier::Low,
    }
}

pub(crate) fn total_margin_used(state: &StrategyState) -> f64 {
    state
        .positions
        .values()
        .map(|position| position.margin_usd.max(0.0))
        .sum()
}

pub(crate) fn apply_decision_projection(
    db_path: &Path,
    input: &ProjectionInput<'_>,
) -> Result<(usize, bool, bool)> {
    let mut conn = Connection::open(db_path)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let result = apply_decision_projection_with_tx(&tx, input)?;
    tx.commit()?;
    Ok(result)
}

pub(crate) fn apply_decision_projection_with_tx(
    tx: &rusqlite::Transaction<'_>,
    input: &ProjectionInput<'_>,
) -> Result<(usize, bool, bool)> {
    let symbol = input.symbol;
    let pre_state = input.pre_state;
    let prior_position = input.prior_position;
    let post_state = input.post_state;
    let intents = input.intents;
    let fills = input.fills;
    let snap = input.snap;
    let execution_metadata = input.execution_metadata;
    let ts_ms = input.ts_ms;
    let ts_iso = iso_from_ms(ts_ms);
    let existing_open_trade_id = tx
        .query_row(
            "SELECT open_trade_id FROM position_state WHERE symbol = ?1 LIMIT 1",
            [symbol],
            |row| row.get::<_, Option<i64>>(0),
        )
        .optional()?
        .flatten();
    let mut active_open_trade_id = existing_open_trade_id;
    let mut current_qty = pre_state
        .positions
        .get(symbol)
        .map(|position| position.quantity)
        .unwrap_or(0.0);
    let mut trades_written = 0usize;
    let mut saw_open = false;
    let mut saw_add = false;
    let mut latest_full_close: Option<(String, String)> = None;

    for (idx, (intent, fill)) in intents.iter().zip(fills.iter()).enumerate() {
        let action = match intent.kind {
            OrderIntentKind::Open => {
                current_qty = fill.quantity;
                saw_open = true;
                "OPEN"
            }
            OrderIntentKind::Add => {
                current_qty += fill.quantity;
                saw_add = true;
                "ADD"
            }
            OrderIntentKind::Close => {
                if fill.quantity < current_qty {
                    current_qty = (current_qty - fill.quantity).max(0.0);
                    "REDUCE"
                } else {
                    current_qty = 0.0;
                    "CLOSE"
                }
            }
            OrderIntentKind::Hold => continue,
            OrderIntentKind::Reverse => continue,
        };

        tx.execute(
            "INSERT INTO trades (timestamp, symbol, action, type, price, size, notional, reason, reason_code, confidence, pnl, fee_usd, fee_rate, balance, entry_atr, leverage, margin_used, meta_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
            params![
                &ts_iso,
                symbol,
                action,
                if intent.side == PositionSide::Long { "LONG" } else { "SHORT" },
                fill.price,
                fill.quantity,
                fill.notional_usd,
                if intent.reason.is_empty() { "rust_paper_run_once" } else { intent.reason.as_str() },
                if intent.reason_code.is_empty() { "entry_signal" } else { intent.reason_code.as_str() },
                projected_confidence_label(intent, pre_state, execution_metadata),
                fill.pnl_usd,
                fill.fee_usd,
                fill.fee_usd / fill.notional_usd.max(1e-12),
                if idx + 1 == fills.len() { Some(post_state.cash_usd) } else { None },
                snap.atr,
                infer_leverage(post_state, intent, fill),
                infer_margin_used(post_state, intent, fill),
                format!("{{\"source\":\"paper_run_once\",\"intent_kind\":\"{:?}\"}}", intent.kind),
            ],
        )?;
        trades_written += 1;
        let trade_id = tx.last_insert_rowid();
        let decision_event_id = format!("{}:{}:{trade_id}", input.run_fingerprint, idx);
        decision_events::insert_fill_tx(
            tx,
            &decision_events::FillDecisionEvent {
                id: &decision_event_id,
                timestamp_ms: ts_ms,
                symbol,
                trade_id,
                config_fingerprint: input.config_fingerprint,
                run_fingerprint: input.run_fingerprint,
                reason_code: (!intent.reason_code.is_empty())
                    .then_some(intent.reason_code.as_str()),
            },
        )?;

        if action == "OPEN" {
            active_open_trade_id = Some(trade_id);
        }
        if action == "CLOSE" {
            active_open_trade_id = None;
            latest_full_close = Some((
                if intent.side == PositionSide::Long {
                    "long".to_string()
                } else {
                    "short".to_string()
                },
                if intent.reason.is_empty() {
                    "rust_paper_run_once".to_string()
                } else {
                    intent.reason.clone()
                },
            ));
        }
    }

    let position_state_written = if let Some(position) = post_state.positions.get(symbol) {
        let last_funding_time_ms = position
            .last_funding_ms
            .filter(|value| *value > 0)
            .or_else(|| {
                if saw_open {
                    Some(ts_ms)
                } else {
                    prior_position
                        .map(|position| position.last_funding_time_ms)
                        .filter(|value| *value > 0)
                }
            })
            .unwrap_or(ts_ms.max(0));
        let last_add_time_ms = if saw_add {
            ts_ms
        } else if saw_open {
            0
        } else {
            prior_position
                .map(|position| position.last_add_time_ms)
                .unwrap_or(0)
        };
        tx.execute(
            "INSERT OR REPLACE INTO position_state (symbol, open_trade_id, trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            (
                symbol,
                active_open_trade_id,
                position.trailing_sl,
                last_funding_time_ms,
                i64::from(position.adds_count),
                if position.tp1_taken { 1_i64 } else { 0_i64 },
                last_add_time_ms,
                if saw_open {
                    execution_metadata.entry_adx_threshold
                } else {
                    position
                        .entry_adx_threshold
                        .or_else(|| prior_position.map(|position| position.entry_adx_threshold))
                        .unwrap_or(0.0)
                },
                &ts_iso,
            ),
        )?;
        true
    } else {
        tx.execute("DELETE FROM position_state WHERE symbol = ?1", [symbol])?;
        true
    };

    tx.execute(
        "INSERT OR REPLACE INTO runtime_cooldowns (symbol, last_entry_attempt_s, last_exit_attempt_s, updated_at)
         VALUES (?1, ?2, ?3, ?4)",
        (
            symbol,
            if saw_open || saw_add {
                Some(ms_to_secs(ts_ms))
            } else {
                post_state.last_entry_ms.get(symbol).map(|value| ms_to_secs(*value))
            },
            if intents.iter().any(|intent| intent.kind == OrderIntentKind::Close) {
                Some(ms_to_secs(ts_ms))
            } else {
                post_state.last_exit_ms.get(symbol).map(|value| ms_to_secs(*value))
            },
            &ts_iso,
        ),
    )?;

    if let Some((close_side, close_reason)) = latest_full_close {
        if table_exists_in_tx(tx, "runtime_last_closes")? {
            tx.execute(
                "INSERT OR REPLACE INTO runtime_last_closes (symbol, close_ts_ms, side, reason, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                (symbol, ts_ms, close_side, close_reason, &ts_iso),
            )?;
        }
    }

    Ok((trades_written, position_state_written, true))
}

pub(crate) fn projected_confidence_label(
    intent: &OrderIntent,
    pre_state: &StrategyState,
    execution_metadata: ExecutionMetadata,
) -> &'static str {
    if matches!(intent.kind, OrderIntentKind::Open | OrderIntentKind::Add) {
        return match execution_metadata.confidence {
            Confidence::High => "high",
            Confidence::Medium => "medium",
            Confidence::Low => "low",
        };
    }
    if let Some(position) = pre_state.positions.get(&intent.symbol) {
        return match position.confidence {
            Some(2) => "high",
            Some(1) => "medium",
            _ => "low",
        };
    }
    "medium"
}

pub(crate) fn infer_leverage(
    post_state: &StrategyState,
    intent: &OrderIntent,
    fill: &FillEvent,
) -> f64 {
    if let Some(position) = post_state.positions.get(&intent.symbol) {
        if position.margin_usd > 0.0 {
            return position.notional_usd / position.margin_usd;
        }
    }
    (fill.notional_usd / fill.quantity.max(1e-12)) / fill.price.max(1e-12)
}

pub(crate) fn infer_margin_used(
    post_state: &StrategyState,
    intent: &OrderIntent,
    fill: &FillEvent,
) -> f64 {
    if let Some(position) = post_state.positions.get(&intent.symbol) {
        return position.margin_usd;
    }
    fill.notional_usd
}

pub(crate) fn iso_from_ms(ms: i64) -> String {
    DateTime::from_timestamp_millis(ms)
        .map(|dt| dt.with_timezone(&Utc).to_rfc3339())
        .unwrap_or_else(|| Utc.timestamp_millis_opt(0).single().unwrap().to_rfc3339())
}

pub(crate) fn ms_to_secs(value: i64) -> f64 {
    (value as f64) / 1000.0
}

pub(crate) fn table_exists_in_tx(tx: &rusqlite::Transaction<'_>, table_name: &str) -> Result<bool> {
    let row = tx
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            [table_name],
            |_| Ok(()),
        )
        .optional()?;
    Ok(row.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiq_runtime_core::runtime::{build_bootstrap, RuntimeMode};
    use rusqlite::OptionalExtension;
    use tempfile::tempdir;

    const FIXED_TS_MS: i64 = 1_772_676_900_000;

    fn create_paper_schema(path: &Path, include_runtime_cooldowns: bool) {
        let conn = Connection::open(path).unwrap();
        let runtime_cooldowns_sql = if include_runtime_cooldowns {
            r#"
            CREATE TABLE runtime_cooldowns (
                symbol TEXT PRIMARY KEY,
                last_entry_attempt_s REAL,
                last_exit_attempt_s REAL,
                updated_at TEXT
            );
            "#
        } else {
            ""
        };
        conn.execute_batch(&format!(
            r#"
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                type TEXT,
                action TEXT,
                price REAL,
                size REAL,
                notional REAL,
                reason TEXT,
                reason_code TEXT,
                confidence TEXT,
                pnl REAL,
                fee_usd REAL,
                fee_token TEXT,
                fee_rate REAL,
                balance REAL,
                entry_atr REAL,
                leverage REAL,
                margin_used REAL,
                meta_json TEXT,
                run_fingerprint TEXT,
                fill_hash TEXT,
                fill_tid INTEGER
            );
            CREATE TABLE position_state (
                symbol TEXT PRIMARY KEY,
                open_trade_id INTEGER,
                trailing_sl REAL,
                last_funding_time INTEGER,
                adds_count INTEGER,
                tp1_taken INTEGER,
                last_add_time INTEGER,
                entry_adx_threshold REAL,
                updated_at TEXT
            );
            CREATE TABLE runtime_last_closes (
                symbol TEXT PRIMARY KEY,
                close_ts_ms INTEGER NOT NULL,
                side TEXT NOT NULL,
                reason TEXT,
                updated_at TEXT NOT NULL
            );
            {runtime_cooldowns_sql}
            "#,
        ))
        .unwrap();
    }

    fn create_legacy_decision_events_schema(path: &Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE decision_events (
                id TEXT PRIMARY KEY,
                timestamp_ms INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                decision_phase TEXT NOT NULL,
                parent_decision_id TEXT,
                trade_id INTEGER,
                triggered_by TEXT,
                action_taken TEXT,
                rejection_reason TEXT,
                context_json TEXT,
                config_fingerprint TEXT,
                run_fingerprint TEXT,
                reason_code TEXT
            );
            "#,
        )
        .unwrap();
    }

    fn seed_paper_db(path: &Path) {
        create_paper_schema(path, true);
        let conn = Connection::open(path).unwrap();
        conn.execute(
            "INSERT INTO trades (timestamp,symbol,action,type,price,size,notional,reason,confidence,balance,pnl,fee_usd,fee_rate,entry_atr,leverage,margin_used,meta_json)
             VALUES ('2026-03-05T10:00:00+00:00','ETH','OPEN','LONG',100.0,1.0,100.0,'seed','medium',1000.0,0.0,0.0,0.0,5.0,3.0,33.3,'{}')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO position_state VALUES ('ETH',1,95.0,1772676500000,0,0,1772676600000,22.0,'2026-03-05T10:08:20+00:00')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO runtime_cooldowns VALUES ('ETH',1772676500.0,1772676550.0,'2026-03-05T10:15:00+00:00')",
            [],
        )
        .unwrap();
        conn.close().unwrap();
    }

    fn seed_flat_paper_db(path: &Path, include_runtime_cooldowns: bool) {
        create_paper_schema(path, include_runtime_cooldowns);
        let conn = Connection::open(path).unwrap();
        conn.execute(
            "INSERT INTO trades (timestamp,symbol,action,type,price,size,notional,reason,reason_code,confidence,balance,pnl,fee_usd,fee_rate,entry_atr,leverage,margin_used,meta_json)
             VALUES ('2026-03-05T09:30:00+00:00','ETH','CLOSE','SHORT',100.0,1.0,100.0,'seed_close','signal_flip','medium',1000.0,0.0,0.0,0.0,5.0,3.0,0.0,'{}')",
            [],
        )
        .unwrap();
        if include_runtime_cooldowns {
            conn.execute(
                "INSERT INTO runtime_cooldowns VALUES ('ETH',1772676500.0,1772676550.0,'2026-03-05T09:30:00+00:00')",
                [],
            )
            .unwrap();
        }
        conn.close().unwrap();
    }

    fn load_cfg(symbol: &str) -> StrategyConfig {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        bt_core::config::load_config_checked(
            repo_root
                .join("config/strategy_overrides.yaml.example")
                .to_str()
                .unwrap(),
            Some(symbol),
            false,
        )
        .unwrap()
    }

    fn seed_candles_db(path: &Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE candles (
                symbol TEXT,
                interval TEXT,
                t INTEGER,
                t_close INTEGER,
                o REAL,
                h REAL,
                l REAL,
                c REAL,
                v REAL,
                n INTEGER
            );
            "#,
        )
        .unwrap();

        let base = 1_772_670_000_000_i64;
        for (symbol, start, drift) in [("ETH", 100.0, 0.25), ("BTC", 50_000.0, 20.0)] {
            let mut price: f64 = start;
            for idx in 0..420_i64 {
                let t = base + (idx * 1_800_000);
                let open: f64 = price;
                let close: f64 = price + drift;
                let high = open.max(close) + 0.5;
                let low = open.min(close) - 0.5;
                let volume = 1000.0 + idx as f64;
                conn.execute(
                    "INSERT INTO candles VALUES (?1, '30m', ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1)",
                    (symbol, t, t + 1_800_000, open, high, low, close, volume),
                )
                .unwrap();
                price = close;
            }
        }
    }

    fn sample_snap() -> IndicatorSnapshot {
        IndicatorSnapshot {
            close: 100.0,
            high: 101.0,
            low: 99.0,
            open: 99.5,
            volume: 1_500.0,
            t: FIXED_TS_MS,
            ema_slow: 95.0,
            ema_fast: 99.0,
            ema_macro: 90.0,
            adx: 32.0,
            adx_pos: 24.0,
            adx_neg: 10.0,
            adx_slope: 0.8,
            bb_upper: 103.0,
            bb_lower: 97.0,
            bb_width: 0.06,
            bb_width_avg: 0.05,
            bb_width_ratio: 1.2,
            atr: 5.0,
            atr_slope: 0.2,
            avg_atr: 4.5,
            rsi: 58.0,
            stoch_rsi_k: 40.0,
            stoch_rsi_d: 35.0,
            macd_hist: 1.0,
            prev_macd_hist: 0.6,
            prev2_macd_hist: 0.3,
            prev3_macd_hist: 0.1,
            vol_sma: 1_000.0,
            vol_trend: true,
            prev_close: 99.0,
            prev_ema_fast: 98.0,
            prev_ema_slow: 94.5,
            bar_count: 200,
            funding_rate: 0.0,
        }
    }

    #[test]
    fn paper_run_once_restores_executes_and_projects() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_paper_db(&paper_db);
        seed_candles_db(&candles_db);

        let cfg = load_cfg("ETH");
        let runtime_bootstrap = build_bootstrap(&cfg, RuntimeMode::Paper, None).unwrap();
        let report = run_once(PaperRunOnceInput {
            config: &cfg,
            runtime_bootstrap,
            paper_db: &paper_db,
            candles_db: &candles_db,
            symbol: "ETH",
            btc_symbol: "BTC",
            lookback_bars: 400,
            exported_at_ms: Some(FIXED_TS_MS),
            dry_run: false,
        })
        .unwrap();

        assert_eq!(report.symbol, "ETH");
        assert!(report.intent_count >= report.fill_count);
        assert!(report.trades_written > 0);
        assert!(report.position_state_written);
        assert!(report.runtime_cooldowns_written);
        assert_eq!(report.snapshot_exported_at_ms, FIXED_TS_MS);

        let conn = Connection::open(&paper_db).unwrap();
        let position_state = conn
            .query_row(
                "SELECT last_funding_time, last_add_time, updated_at FROM position_state WHERE symbol = 'ETH'",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(position_state.0, 1_772_676_500_000);
        assert_eq!(position_state.1, 1_772_676_600_000);
        assert_eq!(position_state.2, iso_from_ms(FIXED_TS_MS));
    }

    #[test]
    fn paper_run_once_dry_run_skips_writes() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_paper_db(&paper_db);
        seed_candles_db(&candles_db);

        let cfg = load_cfg("ETH");
        let runtime_bootstrap = build_bootstrap(&cfg, RuntimeMode::Paper, None).unwrap();
        let report = run_once(PaperRunOnceInput {
            config: &cfg,
            runtime_bootstrap,
            paper_db: &paper_db,
            candles_db: &candles_db,
            symbol: "ETH",
            btc_symbol: "BTC",
            lookback_bars: 400,
            exported_at_ms: Some(FIXED_TS_MS),
            dry_run: true,
        })
        .unwrap();

        assert_eq!(report.trades_written, 0);
        assert!(!report.position_state_written);
        assert!(!report.runtime_cooldowns_written);
    }

    #[test]
    fn build_execution_plan_uses_runtime_sizing_instead_of_kernel_default() {
        let cfg = load_cfg("ETH");
        let pre_state = StrategyState::new(1_000.0, FIXED_TS_MS);
        let plan = build_execution_plan(BuildExecutionPlanInput {
            cfg: &cfg,
            behaviour_plan: &ResolvedBehaviourPlan::production(),
            pre_state: &pre_state,
            symbol: "ETH",
            execution_ts_ms: FIXED_TS_MS,
            prior_position: None,
            snap: &sample_snap(),
            execution_metadata: ExecutionMetadata {
                signal: Signal::Buy,
                confidence: Confidence::High,
                entry_adx_threshold: 22.0,
            },
        });

        assert!(plan.requested_notional_usd.is_some());
        let requested = plan.requested_notional_usd.unwrap();
        assert!(requested > 0.0);
        assert!(requested < 1_000.0);
        assert_eq!(plan.leverage, cfg.trade.leverage_high);
    }

    #[test]
    fn apply_decision_projection_open_sets_funding_and_resets_add_marker() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        seed_flat_paper_db(&paper_db, true);

        let pre_state = StrategyState::new(1_000.0, FIXED_TS_MS);
        let mut post_state = StrategyState::new(970.0, FIXED_TS_MS);
        post_state.positions.insert(
            "ETH".to_string(),
            bt_core::decision_kernel::Position {
                symbol: "ETH".to_string(),
                side: PositionSide::Long,
                quantity: 1.5,
                avg_entry_price: 100.0,
                opened_at_ms: FIXED_TS_MS,
                updated_at_ms: FIXED_TS_MS,
                notional_usd: 150.0,
                margin_usd: 30.0,
                confidence: Some(2),
                entry_atr: Some(5.0),
                entry_adx_threshold: Some(22.0),
                adds_count: 0,
                tp1_taken: false,
                trailing_sl: None,
                mae_usd: 0.0,
                mfe_usd: 0.0,
                last_funding_ms: None,
            },
        );
        let intent = OrderIntent {
            schema_version: 1,
            intent_id: 1,
            symbol: "ETH".to_string(),
            kind: OrderIntentKind::Open,
            side: PositionSide::Long,
            quantity: 1.5,
            price: 100.0,
            notional_usd: 150.0,
            fee_rate: 0.0,
            reason: "Signal Trigger".to_string(),
            reason_code: "entry_signal".to_string(),
        };
        let fill = FillEvent {
            schema_version: 1,
            intent_id: 1,
            symbol: "ETH".to_string(),
            side: PositionSide::Long,
            price: 100.0,
            quantity: 1.5,
            notional_usd: 150.0,
            fee_usd: 0.0,
            pnl_usd: 0.0,
        };
        let metadata = ExecutionMetadata {
            signal: Signal::Buy,
            confidence: Confidence::High,
            entry_adx_threshold: 22.0,
        };
        let projection = ProjectionInput {
            config_fingerprint: "cfg",
            run_fingerprint: "run",
            symbol: "ETH",
            pre_state: &pre_state,
            prior_position: None,
            post_state: &post_state,
            intents: &[intent],
            fills: &[fill],
            snap: &sample_snap(),
            execution_metadata: metadata,
            ts_ms: FIXED_TS_MS,
        };
        let (trades_written, position_state_written, runtime_cooldowns_written) =
            apply_decision_projection(&paper_db, &projection).unwrap();

        assert_eq!(trades_written, 1);
        assert!(position_state_written);
        assert!(runtime_cooldowns_written);
        let conn = Connection::open(&paper_db).unwrap();
        let last_trade = conn
            .query_row(
                "SELECT action, notional, timestamp, confidence FROM trades WHERE symbol = 'ETH' ORDER BY id DESC LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(last_trade.0, "OPEN");
        assert!(last_trade.1 > 0.0);
        assert!(last_trade.1 < 1_000.0);
        assert_eq!(last_trade.2, iso_from_ms(FIXED_TS_MS));
        assert_eq!(last_trade.3, "high");

        let markers = conn
            .query_row(
                "SELECT last_funding_time, last_add_time, entry_adx_threshold FROM position_state WHERE symbol = 'ETH'",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, f64>(2)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(markers.0, FIXED_TS_MS);
        assert_eq!(markers.1, 0);
        assert_eq!(markers.2, 22.0);
        let cooldown = conn
            .query_row(
                "SELECT last_entry_attempt_s FROM runtime_cooldowns WHERE symbol = 'ETH'",
                [],
                |row| row.get::<_, Option<f64>>(0),
            )
            .unwrap();
        assert_eq!(cooldown, Some(ms_to_secs(FIXED_TS_MS)));
    }

    #[test]
    fn apply_decision_projection_writes_fill_events_for_legacy_decision_schema() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        seed_flat_paper_db(&paper_db, true);
        create_legacy_decision_events_schema(&paper_db);

        let pre_state = StrategyState::new(1_000.0, FIXED_TS_MS);
        let mut post_state = StrategyState::new(970.0, FIXED_TS_MS);
        post_state.positions.insert(
            "ETH".to_string(),
            bt_core::decision_kernel::Position {
                symbol: "ETH".to_string(),
                side: PositionSide::Long,
                quantity: 1.5,
                avg_entry_price: 100.0,
                opened_at_ms: FIXED_TS_MS,
                updated_at_ms: FIXED_TS_MS,
                notional_usd: 150.0,
                margin_usd: 30.0,
                confidence: Some(2),
                entry_atr: Some(5.0),
                entry_adx_threshold: Some(22.0),
                adds_count: 0,
                tp1_taken: false,
                trailing_sl: None,
                mae_usd: 0.0,
                mfe_usd: 0.0,
                last_funding_ms: None,
            },
        );
        let intent = OrderIntent {
            schema_version: 1,
            intent_id: 1,
            symbol: "ETH".to_string(),
            kind: OrderIntentKind::Open,
            side: PositionSide::Long,
            quantity: 1.5,
            price: 100.0,
            notional_usd: 150.0,
            fee_rate: 0.0,
            reason: "Signal Trigger".to_string(),
            reason_code: "entry_signal".to_string(),
        };
        let fill = FillEvent {
            schema_version: 1,
            intent_id: 1,
            symbol: "ETH".to_string(),
            side: PositionSide::Long,
            price: 100.0,
            quantity: 1.5,
            notional_usd: 150.0,
            fee_usd: 0.0,
            pnl_usd: 0.0,
        };
        let projection = ProjectionInput {
            config_fingerprint: "cfg",
            run_fingerprint: "run",
            symbol: "ETH",
            pre_state: &pre_state,
            prior_position: None,
            post_state: &post_state,
            intents: &[intent],
            fills: &[fill],
            snap: &sample_snap(),
            execution_metadata: ExecutionMetadata {
                signal: Signal::Buy,
                confidence: Confidence::High,
                entry_adx_threshold: 22.0,
            },
            ts_ms: FIXED_TS_MS,
        };

        apply_decision_projection(&paper_db, &projection).unwrap();

        let conn = Connection::open(&paper_db).unwrap();
        let event = conn
            .query_row(
                "SELECT timestamp_ms, symbol, decision_phase, trade_id, config_fingerprint, run_fingerprint, reason_code
                 FROM decision_events ORDER BY rowid DESC LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, Option<String>>(5)?,
                        row.get::<_, Option<String>>(6)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(event.0, FIXED_TS_MS);
        assert_eq!(event.1, "ETH");
        assert_eq!(event.2, "execution");
        assert!(event.3 > 0);
        assert_eq!(event.4.as_deref(), Some("cfg"));
        assert_eq!(event.5.as_deref(), Some("run"));
        assert_eq!(event.6.as_deref(), Some("entry_signal"));
    }

    #[test]
    fn build_execution_plan_blocks_adds_that_violate_pyramid_gates() {
        let cfg = load_cfg("ETH");
        let mut pre_state = StrategyState::new(1_000.0, FIXED_TS_MS);
        pre_state.positions.insert(
            "ETH".to_string(),
            bt_core::decision_kernel::Position {
                symbol: "ETH".to_string(),
                side: PositionSide::Long,
                quantity: 1.0,
                avg_entry_price: 100.0,
                opened_at_ms: FIXED_TS_MS - 3_600_000,
                updated_at_ms: FIXED_TS_MS - 1_800_000,
                notional_usd: 100.0,
                margin_usd: 33.3333333333,
                confidence: Some(2),
                entry_atr: Some(5.0),
                entry_adx_threshold: Some(22.0),
                adds_count: 1,
                tp1_taken: true,
                trailing_sl: None,
                mae_usd: 0.0,
                mfe_usd: 0.0,
                last_funding_ms: Some(FIXED_TS_MS - 1_800_000),
            },
        );
        let plan = build_execution_plan(BuildExecutionPlanInput {
            cfg: &cfg,
            behaviour_plan: &ResolvedBehaviourPlan::production(),
            pre_state: &pre_state,
            symbol: "ETH",
            execution_ts_ms: FIXED_TS_MS,
            prior_position: Some(&PaperPositionState {
                symbol: "ETH".to_string(),
                side: "long".to_string(),
                size: 1.0,
                entry_price: 100.0,
                entry_atr: 5.0,
                trailing_sl: None,
                confidence: "high".to_string(),
                leverage: 3.0,
                margin_used: 33.3333333333,
                adds_count: 1,
                tp1_taken: true,
                open_time_ms: FIXED_TS_MS - 3_600_000,
                last_funding_time_ms: FIXED_TS_MS - 1_800_000,
                last_add_time_ms: FIXED_TS_MS - 1_000,
                entry_adx_threshold: 22.0,
            }),
            snap: &sample_snap(),
            execution_metadata: ExecutionMetadata {
                signal: Signal::Buy,
                confidence: Confidence::High,
                entry_adx_threshold: 22.0,
            },
        });

        assert!(plan.requested_notional_usd.is_none());
        assert!(!plan.allow_pyramid);
        assert!(!plan.warnings.is_empty());
    }

    #[test]
    fn apply_decision_projection_add_updates_last_add_time() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        seed_paper_db(&paper_db);

        let mut pre_state = StrategyState::new(1_000.0, FIXED_TS_MS - 60_000);
        pre_state.positions.insert(
            "ETH".to_string(),
            bt_core::decision_kernel::Position {
                symbol: "ETH".to_string(),
                side: PositionSide::Long,
                quantity: 1.0,
                avg_entry_price: 100.0,
                opened_at_ms: FIXED_TS_MS - 3_600_000,
                updated_at_ms: FIXED_TS_MS - 3_600_000,
                notional_usd: 100.0,
                margin_usd: 33.3,
                confidence: Some(2),
                entry_atr: Some(5.0),
                entry_adx_threshold: Some(22.0),
                adds_count: 0,
                tp1_taken: false,
                trailing_sl: Some(95.0),
                mae_usd: 0.0,
                mfe_usd: 0.0,
                last_funding_ms: Some(FIXED_TS_MS - 3_600_000),
            },
        );
        let mut post_state = pre_state.clone();
        if let Some(position) = post_state.positions.get_mut("ETH") {
            position.quantity = 1.5;
            position.notional_usd = 150.0;
            position.margin_usd = 50.0;
            position.adds_count = 1;
            position.updated_at_ms = FIXED_TS_MS;
        }
        let intent = OrderIntent {
            schema_version: 1,
            intent_id: 2,
            symbol: "ETH".to_string(),
            kind: OrderIntentKind::Add,
            side: PositionSide::Long,
            quantity: 0.5,
            price: 100.0,
            notional_usd: 50.0,
            fee_rate: 0.0,
            reason: "Pyramid Add".to_string(),
            reason_code: "entry_signal".to_string(),
        };
        let fill = FillEvent {
            schema_version: 1,
            intent_id: 2,
            symbol: "ETH".to_string(),
            side: PositionSide::Long,
            price: 100.0,
            quantity: 0.5,
            notional_usd: 50.0,
            fee_usd: 0.0,
            pnl_usd: 0.0,
        };
        let metadata = ExecutionMetadata {
            signal: Signal::Buy,
            confidence: Confidence::High,
            entry_adx_threshold: 22.0,
        };
        let prior_position = PaperPositionState {
            symbol: "ETH".to_string(),
            side: "long".to_string(),
            size: 1.0,
            entry_price: 100.0,
            entry_atr: 5.0,
            trailing_sl: Some(95.0),
            confidence: "high".to_string(),
            leverage: 3.0,
            margin_used: 33.3,
            adds_count: 0,
            tp1_taken: false,
            open_time_ms: FIXED_TS_MS - 3_600_000,
            last_funding_time_ms: FIXED_TS_MS - 3_600_000,
            last_add_time_ms: FIXED_TS_MS - 7_200_000,
            entry_adx_threshold: 22.0,
        };
        let projection = ProjectionInput {
            config_fingerprint: "cfg",
            run_fingerprint: "run",
            symbol: "ETH",
            pre_state: &pre_state,
            prior_position: Some(&prior_position),
            post_state: &post_state,
            intents: &[intent],
            fills: &[fill],
            snap: &sample_snap(),
            execution_metadata: metadata,
            ts_ms: FIXED_TS_MS,
        };

        apply_decision_projection(&paper_db, &projection).unwrap();

        let conn = Connection::open(&paper_db).unwrap();
        let state_row = conn
            .query_row(
                "SELECT adds_count, last_add_time, last_entry_attempt_s FROM position_state JOIN runtime_cooldowns USING(symbol) WHERE symbol = 'ETH'",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, Option<f64>>(2)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(state_row.0, 1);
        assert_eq!(state_row.1, FIXED_TS_MS);
        assert_eq!(state_row.2, Some(ms_to_secs(FIXED_TS_MS)));
    }

    #[test]
    fn full_close_projection_refreshes_runtime_last_closes() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        seed_paper_db(&paper_db);

        let conn = Connection::open(&paper_db).unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO runtime_last_closes (symbol, close_ts_ms, side, reason, updated_at)
             VALUES ('ETH', ?1, 'long', 'stale-close', '2026-03-05T10:00:00+00:00')",
            [FIXED_TS_MS - 60_000],
        )
        .unwrap();
        conn.close().unwrap();

        let mut pre_state = StrategyState::new(1_000.0, FIXED_TS_MS - 60_000);
        pre_state.positions.insert(
            "ETH".to_string(),
            bt_core::decision_kernel::Position {
                symbol: "ETH".to_string(),
                side: PositionSide::Long,
                quantity: 1.0,
                avg_entry_price: 100.0,
                opened_at_ms: FIXED_TS_MS - 3_600_000,
                updated_at_ms: FIXED_TS_MS - 60_000,
                notional_usd: 100.0,
                margin_usd: 33.3,
                confidence: Some(2),
                entry_atr: Some(5.0),
                entry_adx_threshold: Some(22.0),
                adds_count: 0,
                tp1_taken: false,
                trailing_sl: Some(95.0),
                mae_usd: 0.0,
                mfe_usd: 0.0,
                last_funding_ms: Some(FIXED_TS_MS - 3_600_000),
            },
        );
        let post_state = StrategyState::new(1_050.0, FIXED_TS_MS);
        let intent = OrderIntent {
            schema_version: 1,
            intent_id: 3,
            symbol: "ETH".to_string(),
            kind: OrderIntentKind::Close,
            side: PositionSide::Long,
            quantity: 1.0,
            price: 105.0,
            notional_usd: 105.0,
            fee_rate: 0.0,
            reason: "Signal Trigger".to_string(),
            reason_code: "exit_signal".to_string(),
        };
        let fill = FillEvent {
            schema_version: 1,
            intent_id: 3,
            symbol: "ETH".to_string(),
            side: PositionSide::Long,
            price: 105.0,
            quantity: 1.0,
            notional_usd: 105.0,
            fee_usd: 0.0,
            pnl_usd: 5.0,
        };

        let prior_position = PaperPositionState {
            symbol: "ETH".to_string(),
            side: "long".to_string(),
            size: 1.0,
            entry_price: 100.0,
            entry_atr: 5.0,
            trailing_sl: Some(95.0),
            confidence: "high".to_string(),
            leverage: 3.0,
            margin_used: 33.3,
            adds_count: 0,
            tp1_taken: false,
            open_time_ms: FIXED_TS_MS - 3_600_000,
            last_funding_time_ms: FIXED_TS_MS - 3_600_000,
            last_add_time_ms: 0,
            entry_adx_threshold: 22.0,
        };
        let projection = ProjectionInput {
            config_fingerprint: "cfg",
            run_fingerprint: "run",
            symbol: "ETH",
            pre_state: &pre_state,
            prior_position: Some(&prior_position),
            post_state: &post_state,
            intents: &[intent],
            fills: &[fill],
            snap: &sample_snap(),
            execution_metadata: ExecutionMetadata {
                signal: Signal::Sell,
                confidence: Confidence::High,
                entry_adx_threshold: 22.0,
            },
            ts_ms: FIXED_TS_MS,
        };

        apply_decision_projection(&paper_db, &projection).unwrap();

        let snapshot = crate::paper_export::export_paper_snapshot(&paper_db, FIXED_TS_MS).unwrap();
        let close_info = snapshot
            .runtime
            .as_ref()
            .unwrap()
            .last_close_info_by_symbol
            .get("ETH")
            .unwrap();
        assert_eq!(close_info.timestamp_ms, FIXED_TS_MS);
        assert_eq!(close_info.reason, "Signal Trigger");
    }

    #[test]
    fn paper_run_once_projection_rolls_back_when_runtime_cooldowns_missing() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_flat_paper_db(&paper_db, false);
        seed_candles_db(&candles_db);

        let cfg = load_cfg("ETH");
        let runtime_bootstrap = build_bootstrap(&cfg, RuntimeMode::Paper, None).unwrap();
        let err = run_once(PaperRunOnceInput {
            config: &cfg,
            runtime_bootstrap,
            paper_db: &paper_db,
            candles_db: &candles_db,
            symbol: "ETH",
            btc_symbol: "BTC",
            lookback_bars: 400,
            exported_at_ms: Some(FIXED_TS_MS),
            dry_run: false,
        })
        .unwrap_err();

        assert!(err.to_string().contains("runtime_cooldowns"));

        let conn = Connection::open(&paper_db).unwrap();
        let trade_count = conn
            .query_row("SELECT COUNT(*) FROM trades", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap();
        assert_eq!(trade_count, 1);
        let maybe_position_state = conn
            .query_row(
                "SELECT symbol FROM position_state WHERE symbol = 'ETH'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .unwrap();
        assert!(maybe_position_state.is_none());
    }
}
