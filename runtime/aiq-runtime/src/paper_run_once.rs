use aiq_runtime_core::paper::{restore_paper_state, PaperBootstrapReport};
use aiq_runtime_core::runtime::RuntimeBootstrap;
use anyhow::{anyhow, Result};
use bt_core::candle::OhlcvBar;
use bt_core::config::{Confidence, MacdMode, StrategyConfig};
use bt_core::decision_kernel::{
    self, CooldownParams, FillEvent, KernelParams, MarketEvent, MarketSignal, OrderIntent,
    OrderIntentKind, PositionSide, StrategyState,
};
use bt_core::indicators::{IndicatorBank, IndicatorSnapshot};
use bt_core::kernel_entries::EntryParams;
use bt_core::kernel_exits::ExitParams;
use bt_core::signals::gates;
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;
use std::path::Path;

use crate::paper_export;

pub struct PaperRunOnceInput<'a> {
    pub config: &'a StrategyConfig,
    pub runtime_bootstrap: RuntimeBootstrap,
    pub paper_db: &'a Path,
    pub candles_db: &'a Path,
    pub symbol: &'a str,
    pub btc_symbol: &'a str,
    pub lookback_bars: usize,
    pub dry_run: bool,
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
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
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
    let exported_at_ms = Utc::now().timestamp_millis();

    let snapshot = paper_export::export_paper_snapshot(input.paper_db, exported_at_ms)?;
    let (paper_state, paper_bootstrap) =
        restore_paper_state(&snapshot).map_err(anyhow::Error::msg)?;

    let symbol_bars = load_recent_bars(
        input.candles_db,
        &symbol,
        &interval,
        input.lookback_bars.max(64),
    )?;
    let (snap, ema_history) = build_latest_snapshot(input.config, &symbol_bars)?;

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
    let gate_result = gates::check_gates(
        &snap,
        input.config,
        &symbol,
        btc_bullish,
        ema_slow_slope_pct,
    );

    let pre_state = paper_state.into_strategy_state();
    let params = build_kernel_params(input.config);
    let event = MarketEvent {
        schema_version: 1,
        event_id: snap.t.max(0) as u64,
        timestamp_ms: snap.t,
        symbol: symbol.clone(),
        signal: MarketSignal::Evaluate,
        price: snap.close,
        notional_hint_usd: None,
        close_fraction: None,
        fee_role: None,
        funding_rate: None,
        indicators: Some(snap.clone()),
        gate_result: Some(gate_result),
        ema_slow_slope_pct: Some(ema_slow_slope_pct),
    };
    let decision = decision_kernel::step(&pre_state, &event, &params);

    let (trades_written, position_state_written, runtime_cooldowns_written) = if input.dry_run {
        (0usize, false, false)
    } else {
        apply_decision_projection(
            input.paper_db,
            &symbol,
            &pre_state,
            &decision.state,
            &decision.intents,
            &decision.fills,
            &snap,
            exported_at_ms,
        )?
    };

    Ok(PaperRunOnceReport {
        ok: decision.diagnostics.errors.is_empty(),
        dry_run: input.dry_run,
        symbol,
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
        action_codes: project_action_codes(&pre_state, &decision.intents, &decision.fills),
        trades_written,
        position_state_written,
        runtime_cooldowns_written,
        warnings: decision.diagnostics.warnings,
        errors: decision.diagnostics.errors,
    })
}

fn load_recent_bars(
    db_path: &Path,
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<OhlcvBar>> {
    let conn = Connection::open(db_path)?;
    let mut stmt = conn.prepare(
        "SELECT t, COALESCE(t_close, t), o, h, l, c, v, COALESCE(n, 0)
         FROM candles
         WHERE symbol = ?1 AND interval = ?2
         ORDER BY t DESC
         LIMIT ?3",
    )?;
    let mut bars = stmt
        .query_map((symbol, interval, limit as i64), |row| {
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
        .collect::<rusqlite::Result<Vec<_>>>()?;
    bars.reverse();
    if bars.len() < 2 {
        anyhow::bail!("not enough bars for {} {}", symbol, interval);
    }
    Ok(bars)
}

fn build_latest_snapshot(
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

fn compute_ema_slow_slope(history: &[f64], window: usize, current_close: f64) -> f64 {
    if history.len() < window || current_close <= 0.0 {
        return 0.0;
    }
    let current = history[history.len() - 1];
    let past = history[history.len() - window];
    (current - past) / current_close
}

fn build_kernel_params(cfg: &StrategyConfig) -> KernelParams {
    let mut params = KernelParams::default();
    params.allow_pyramid = cfg.trade.enable_pyramiding;
    params.allow_reverse = false;
    params.leverage = cfg.trade.leverage.max(1.0);
    params.exit_params = Some(build_exit_params(cfg));
    params.entry_params = Some(build_entry_params(cfg));
    params.cooldown_params = Some(build_cooldown_params(cfg));
    params
}

fn build_exit_params(cfg: &StrategyConfig) -> ExitParams {
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

fn build_entry_params(cfg: &StrategyConfig) -> EntryParams {
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

fn build_cooldown_params(cfg: &StrategyConfig) -> CooldownParams {
    CooldownParams {
        entry_cooldown_s: cfg.trade.entry_cooldown_s as u32,
        exit_cooldown_s: cfg.trade.exit_cooldown_s as u32,
        reentry_cooldown_minutes: cfg.trade.reentry_cooldown_minutes as u32,
        reentry_cooldown_min_mins: cfg.trade.reentry_cooldown_min_mins as u32,
        reentry_cooldown_max_mins: cfg.trade.reentry_cooldown_max_mins as u32,
    }
}

fn project_action_codes(
    pre_state: &StrategyState,
    intents: &[OrderIntent],
    fills: &[FillEvent],
) -> Vec<String> {
    let mut codes = Vec::new();
    let mut current_qty = pre_state
        .positions
        .values()
        .next()
        .map(|position| position.quantity)
        .unwrap_or(0.0);

    for (intent, fill) in intents.iter().zip(fills.iter()) {
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

fn apply_decision_projection(
    db_path: &Path,
    symbol: &str,
    pre_state: &StrategyState,
    post_state: &StrategyState,
    intents: &[OrderIntent],
    fills: &[FillEvent],
    snap: &IndicatorSnapshot,
    ts_ms: i64,
) -> Result<(usize, bool, bool)> {
    let conn = Connection::open(db_path)?;
    let ts_iso = iso_from_ms(ts_ms);
    let existing_open_trade_id = conn
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

    for (idx, (intent, fill)) in intents.iter().zip(fills.iter()).enumerate() {
        let action = match intent.kind {
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
            OrderIntentKind::Reverse => continue,
        };

        conn.execute(
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
                inferred_confidence(intent, pre_state),
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

        if action == "OPEN" {
            active_open_trade_id = Some(conn.last_insert_rowid());
        }
        if action == "CLOSE" {
            active_open_trade_id = None;
        }
    }

    let position_state_written = if let Some(position) = post_state.positions.get(symbol) {
        conn.execute(
            "INSERT OR REPLACE INTO position_state (symbol, open_trade_id, trailing_sl, last_funding_time, adds_count, tp1_taken, last_add_time, entry_adx_threshold, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            (
                symbol,
                active_open_trade_id,
                position.trailing_sl,
                position.last_funding_ms.unwrap_or(position.opened_at_ms),
                i64::from(position.adds_count),
                if position.tp1_taken { 1_i64 } else { 0_i64 },
                position.updated_at_ms,
                position.entry_adx_threshold.unwrap_or(0.0),
                &ts_iso,
            ),
        )?;
        true
    } else {
        conn.execute("DELETE FROM position_state WHERE symbol = ?1", [symbol])?;
        true
    };

    conn.execute(
        "INSERT OR REPLACE INTO runtime_cooldowns (symbol, last_entry_attempt_s, last_exit_attempt_s, updated_at)
         VALUES (?1, ?2, ?3, ?4)",
        (
            symbol,
            post_state.last_entry_ms.get(symbol).map(|value| (*value as f64) / 1000.0),
            post_state.last_exit_ms.get(symbol).map(|value| (*value as f64) / 1000.0),
            &ts_iso,
        ),
    )?;

    Ok((trades_written, position_state_written, true))
}

fn inferred_confidence(intent: &OrderIntent, pre_state: &StrategyState) -> &'static str {
    if let Some(position) = pre_state.positions.get(&intent.symbol) {
        return match position.confidence {
            Some(2) => "high",
            Some(1) => "medium",
            _ => "medium",
        };
    }
    "medium"
}

fn infer_leverage(post_state: &StrategyState, intent: &OrderIntent, fill: &FillEvent) -> f64 {
    if let Some(position) = post_state.positions.get(&intent.symbol) {
        if position.margin_usd > 0.0 {
            return position.notional_usd / position.margin_usd;
        }
    }
    (fill.notional_usd / fill.quantity.max(1e-12)) / fill.price.max(1e-12)
}

fn infer_margin_used(post_state: &StrategyState, intent: &OrderIntent, fill: &FillEvent) -> f64 {
    if let Some(position) = post_state.positions.get(&intent.symbol) {
        return position.margin_usd;
    }
    fill.notional_usd
}

fn iso_from_ms(ms: i64) -> String {
    DateTime::from_timestamp_millis(ms)
        .map(|dt| dt.with_timezone(&Utc).to_rfc3339())
        .unwrap_or_else(|| Utc::now().to_rfc3339())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiq_runtime_core::runtime::{build_bootstrap, RuntimeMode};
    use tempfile::tempdir;

    fn seed_paper_db(path: &Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
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
            CREATE TABLE runtime_cooldowns (
                symbol TEXT PRIMARY KEY,
                last_entry_attempt_s REAL,
                last_exit_attempt_s REAL,
                updated_at TEXT
            );
            "#,
        )
        .unwrap();
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

    #[test]
    fn paper_run_once_restores_executes_and_projects() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        seed_paper_db(&paper_db);
        seed_candles_db(&candles_db);

        let cfg = bt_core::config::load_config_checked(
            repo_root
                .join("config/strategy_overrides.yaml.example")
                .to_str()
                .unwrap(),
            Some("ETH"),
            false,
        )
        .unwrap();
        let runtime_bootstrap = build_bootstrap(&cfg, RuntimeMode::Paper, None).unwrap();
        let report = run_once(PaperRunOnceInput {
            config: &cfg,
            runtime_bootstrap,
            paper_db: &paper_db,
            candles_db: &candles_db,
            symbol: "ETH",
            btc_symbol: "BTC",
            lookback_bars: 400,
            dry_run: false,
        })
        .unwrap();

        assert_eq!(report.symbol, "ETH");
        assert!(report.intent_count >= report.fill_count);
        assert!(report.position_state_written);
        assert!(report.runtime_cooldowns_written);
    }

    #[test]
    fn paper_run_once_dry_run_skips_writes() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        seed_paper_db(&paper_db);
        seed_candles_db(&candles_db);

        let cfg = bt_core::config::load_config_checked(
            repo_root
                .join("config/strategy_overrides.yaml.example")
                .to_str()
                .unwrap(),
            Some("ETH"),
            false,
        )
        .unwrap();
        let runtime_bootstrap = build_bootstrap(&cfg, RuntimeMode::Paper, None).unwrap();
        let report = run_once(PaperRunOnceInput {
            config: &cfg,
            runtime_bootstrap,
            paper_db: &paper_db,
            candles_db: &candles_db,
            symbol: "ETH",
            btc_symbol: "BTC",
            lookback_bars: 400,
            dry_run: true,
        })
        .unwrap();

        assert_eq!(report.trades_written, 0);
        assert!(!report.position_state_written);
        assert!(!report.runtime_cooldowns_written);
    }
}
