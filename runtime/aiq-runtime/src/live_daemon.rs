use anyhow::{Context, Result};
use chrono::Utc;
use fs2::FileExt;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use serde_json::json;
use signal_hook::{consts::signal::SIGINT, consts::signal::SIGTERM, flag, SigId};
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::live_cycle::{self, LiveActionPlan, LiveCycleInput, LiveCycleReport};
use crate::live_hyperliquid::{
    extract_exchange_order_id, response_has_embedded_error, HyperliquidClient,
};
use crate::live_oms::{
    CreateIntentRequest, InsertFillRequest, LiveOms, SentOrderRequest, SubmitUnknownRequest,
};
use crate::live_risk::{
    FillEvent, LiveRiskManager, OrderAction, OrderCheck, OrderRecord, OrderSide,
};
use crate::live_safety;
use crate::live_secrets::load_live_secrets;
use crate::live_state::{
    build_strategy_state, ensure_live_runtime_tables, note_submitted_order, record_full_close,
    sync_exchange_positions, SubmittedOrderAction, SubmittedOrderProjection,
};
use crate::paper_config::PaperEffectiveConfig;

pub struct LiveDaemonInput<'a> {
    pub effective_config: PaperEffectiveConfig,
    pub runtime_bootstrap: aiq_runtime_core::runtime::RuntimeBootstrap,
    pub live_db: &'a Path,
    pub candles_db: &'a Path,
    pub explicit_symbols: &'a [String],
    pub symbols_file: Option<&'a Path>,
    pub btc_symbol: &'a str,
    pub lookback_bars: usize,
    pub secrets_path: &'a Path,
    pub lock_path: Option<&'a Path>,
    pub status_path: Option<&'a Path>,
    pub idle_sleep_ms: u64,
    pub max_idle_polls: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LiveDaemonReport {
    pub ok: bool,
    pub pid: u32,
    pub lock_path: String,
    pub status_path: String,
    pub started_at_ms: i64,
    pub stopped_at_ms: i64,
    pub stop_requested: bool,
    pub last_fill_cursor_ms: i64,
    pub runtime_bootstrap: aiq_runtime_core::runtime::RuntimeBootstrap,
    pub last_cycle: Option<LiveCycleReport>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LiveDaemonStatus {
    pub ok: bool,
    pub running: bool,
    pub pid: u32,
    pub config_path: String,
    pub live_db: String,
    pub candles_db: String,
    pub lock_path: String,
    pub status_path: String,
    pub started_at_ms: i64,
    pub updated_at_ms: i64,
    pub stopped_at_ms: Option<i64>,
    pub stop_requested: bool,
    pub runtime_bootstrap: aiq_runtime_core::runtime::RuntimeBootstrap,
    pub btc_symbol: String,
    pub lookback_bars: usize,
    pub explicit_symbols: Vec<String>,
    pub symbols_file: Option<String>,
    pub latest_common_close_ts_ms: Option<i64>,
    pub last_step_close_ts_ms: Option<i64>,
    pub last_fill_cursor_ms: i64,
    pub last_plans_count: usize,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub(crate) struct LiveDaemonStatusSnapshot {
    pub ok: bool,
    pub running: bool,
    pub pid: u32,
    #[serde(default)]
    pub config_path: String,
    #[serde(default)]
    pub live_db: String,
    #[serde(default)]
    pub candles_db: String,
    pub lock_path: String,
    pub status_path: String,
    pub started_at_ms: i64,
    pub updated_at_ms: i64,
    pub stopped_at_ms: Option<i64>,
    pub stop_requested: bool,
    pub runtime_bootstrap: LiveDaemonStatusRuntimeBootstrap,
    #[serde(default)]
    pub btc_symbol: String,
    #[serde(default)]
    pub lookback_bars: usize,
    #[serde(default)]
    pub explicit_symbols: Vec<String>,
    pub symbols_file: Option<String>,
    pub latest_common_close_ts_ms: Option<i64>,
    pub last_step_close_ts_ms: Option<i64>,
    pub last_fill_cursor_ms: i64,
    pub last_plans_count: usize,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub(crate) struct LiveDaemonStatusRuntimeBootstrap {
    pub config_fingerprint: String,
    pub pipeline: LiveDaemonStatusRuntimePipeline,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub(crate) struct LiveDaemonStatusRuntimePipeline {
    pub profile: String,
}

pub(crate) fn load_status_file(status_path: &Path) -> Result<Option<LiveDaemonStatusSnapshot>> {
    if !status_path.exists() {
        return Ok(None);
    }

    let payload = std::fs::read(status_path).with_context(|| {
        format!(
            "failed to read live daemon status file: {}",
            status_path.display()
        )
    })?;
    let status =
        serde_json::from_slice::<LiveDaemonStatusSnapshot>(&payload).with_context(|| {
            format!(
                "failed to parse live daemon status file: {}",
                status_path.display()
            )
        })?;
    Ok(Some(status))
}

pub(crate) fn probe_lock_owner(lock_path: &Path) -> Result<Option<u32>> {
    if !lock_path.exists() {
        return Ok(None);
    }

    let mut lock_file = OpenOptions::new()
        .create(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .with_context(|| {
            format!(
                "failed to open live daemon lock file: {}",
                lock_path.display()
            )
        })?;
    match lock_file.try_lock_exclusive() {
        Ok(()) => {
            lock_file.unlock().with_context(|| {
                format!(
                    "failed to unlock live daemon probe file: {}",
                    lock_path.display()
                )
            })?;
            Ok(None)
        }
        Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
            let pid = read_pid(&mut lock_file)?.context(format!(
                "live daemon lock is held but its pid file is empty or invalid: {}",
                lock_path.display()
            ))?;
            Ok(Some(pid))
        }
        Err(err) => Err(err).with_context(|| {
            format!(
                "failed to probe the live daemon lock owner: {}",
                lock_path.display()
            )
        }),
    }
}

pub fn run_daemon(input: LiveDaemonInput<'_>) -> Result<LiveDaemonReport> {
    if !input.live_db.exists() {
        anyhow::bail!("live db not found: {}", input.live_db.display());
    }
    if !input.candles_db.exists() {
        anyhow::bail!("candles db not found: {}", input.candles_db.display());
    }

    let secrets = load_live_secrets(input.secrets_path)?;
    let client = HyperliquidClient::new(&secrets, Some(4.0))?;
    let oms = LiveOms::new(input.live_db)?;
    let mut risk = LiveRiskManager::from_env();
    let lock_path = resolve_lock_path(input.lock_path);
    let status_path = resolve_status_path(input.status_path, &lock_path);
    let _lock_file = acquire_lock(&lock_path)?;
    let started_at_ms = Utc::now().timestamp_millis();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let _signal_guard = install_signal_handlers(&stop_flag)?;

    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    let mut stop_requested = false;
    let mut idle_polls = 0usize;
    let mut last_step_close_ts_ms = None;
    let mut last_fill_cursor_ms = Utc::now().timestamp_millis() - 2 * 60 * 60 * 1000;
    let mut last_cycle = None;

    ensure_live_runtime_tables(&rusqlite::Connection::open(input.live_db)?)?;
    write_status(
        &status_path,
        &build_status(
            &input,
            &lock_path,
            started_at_ms,
            None,
            last_step_close_ts_ms,
            last_fill_cursor_ms,
            &[],
            &[],
            None,
            false,
        ),
    )?;

    loop {
        if stop_flag.load(Ordering::Relaxed) {
            stop_requested = true;
            warnings.push("live daemon stop requested".to_string());
            break;
        }

        let symbols = load_symbols(input.explicit_symbols, input.symbols_file)?;
        let account_snapshot = client.account_snapshot()?;
        let exchange_positions = client.positions()?;
        sync_exchange_positions(
            input.live_db,
            &exchange_positions,
            Utc::now().timestamp_millis(),
        )?;
        risk.refresh(
            Utc::now().timestamp_millis(),
            Some(account_snapshot.account_value_usd),
        );
        ingest_recent_fills(
            &client,
            &oms,
            input.live_db,
            &mut risk,
            &mut last_fill_cursor_ms,
            &mut warnings,
        )?;

        let state = build_strategy_state(
            input.live_db,
            &account_snapshot,
            &exchange_positions,
            Utc::now().timestamp_millis(),
        )?;
        let active_symbols = active_symbols(&symbols, &state);
        if active_symbols.is_empty() {
            warnings.push("live daemon idle: no configured symbols or open positions".to_string());
            idle_polls = idle_polls.saturating_add(1);
            if input.max_idle_polls > 0 && idle_polls >= input.max_idle_polls {
                break;
            }
            sleep_with_stop_flag(input.idle_sleep_ms, &stop_flag);
            continue;
        }

        let interval = input
            .effective_config
            .resolve_shared_interval(&active_symbols, true)?;
        if last_step_close_ts_ms.is_none() {
            last_step_close_ts_ms = load_last_applied_step_close_ts_ms(
                input.live_db,
                &input.runtime_bootstrap.config_fingerprint,
                &interval,
                true,
            )?;
        }
        let latest_common_close_ts_ms = latest_common_close_ts_ms(
            input.candles_db,
            &interval,
            &active_symbols,
            input.btc_symbol,
        )?;
        let interval_ms = interval_to_ms(&interval)?;
        let next_due_step_close_ts_ms = last_step_close_ts_ms
            .and_then(|value| value.checked_add(interval_ms))
            .unwrap_or(latest_common_close_ts_ms);

        if next_due_step_close_ts_ms > latest_common_close_ts_ms {
            idle_polls = idle_polls.saturating_add(1);
            write_status(
                &status_path,
                &build_status(
                    &input,
                    &lock_path,
                    started_at_ms,
                    Some(latest_common_close_ts_ms),
                    last_step_close_ts_ms,
                    last_fill_cursor_ms,
                    &warnings,
                    &errors,
                    last_cycle.as_ref(),
                    false,
                ),
            )?;
            if input.max_idle_polls > 0 && idle_polls >= input.max_idle_polls {
                warnings.push(format!(
                    "live daemon follow exhausted after {} idle poll(s)",
                    input.max_idle_polls
                ));
                break;
            }
            sleep_with_stop_flag(input.idle_sleep_ms, &stop_flag);
            continue;
        }

        let cycle_report = live_cycle::run_cycle(LiveCycleInput {
            effective_config: input.effective_config.clone(),
            runtime_bootstrap: input.runtime_bootstrap.clone(),
            state,
            explicit_symbols: &symbols,
            candles_db: input.candles_db,
            btc_symbol: input.btc_symbol,
            lookback_bars: input.lookback_bars,
            step_close_ts_ms: next_due_step_close_ts_ms,
        })?;
        warnings.extend(cycle_report.warnings.iter().cloned());
        errors.extend(cycle_report.errors.iter().cloned());
        submit_cycle_plans(
            &client,
            &oms,
            &mut risk,
            input.live_db,
            &cycle_report.plans,
            next_due_step_close_ts_ms,
        )?;
        record_runtime_cycle_step(
            input.live_db,
            &input.runtime_bootstrap.config_fingerprint,
            &interval,
            next_due_step_close_ts_ms,
            &cycle_report.active_symbols,
            cycle_report.plans.len(),
        )?;
        last_step_close_ts_ms = Some(next_due_step_close_ts_ms);
        last_cycle = Some(cycle_report);
        idle_polls = 0;

        write_status(
            &status_path,
            &build_status(
                &input,
                &lock_path,
                started_at_ms,
                Some(latest_common_close_ts_ms),
                last_step_close_ts_ms,
                last_fill_cursor_ms,
                &warnings,
                &errors,
                last_cycle.as_ref(),
                false,
            ),
        )?;
        sleep_with_stop_flag(input.idle_sleep_ms, &stop_flag);
    }

    let stopped_at_ms = Utc::now().timestamp_millis();
    write_status(
        &status_path,
        &build_status(
            &input,
            &lock_path,
            started_at_ms,
            None,
            last_step_close_ts_ms,
            last_fill_cursor_ms,
            &warnings,
            &errors,
            last_cycle.as_ref(),
            true,
        )
        .with_stopped_at(stopped_at_ms, stop_requested),
    )?;

    Ok(LiveDaemonReport {
        ok: errors.is_empty(),
        pid: std::process::id(),
        lock_path: lock_path.display().to_string(),
        status_path: status_path.display().to_string(),
        started_at_ms,
        stopped_at_ms,
        stop_requested,
        last_fill_cursor_ms,
        runtime_bootstrap: input.runtime_bootstrap,
        last_cycle,
        warnings,
        errors,
    })
}

fn submit_cycle_plans(
    client: &HyperliquidClient,
    oms: &LiveOms,
    risk: &mut LiveRiskManager,
    live_db: &Path,
    plans: &[LiveActionPlan],
    step_close_ts_ms: i64,
) -> Result<()> {
    for plan in plans {
        submit_plan(client, oms, risk, live_db, plan, step_close_ts_ms)?;
    }
    Ok(())
}

fn submit_plan(
    client: &HyperliquidClient,
    oms: &LiveOms,
    risk: &mut LiveRiskManager,
    live_db: &Path,
    plan: &LiveActionPlan,
    step_close_ts_ms: i64,
) -> Result<()> {
    let intent_meta = json!({
        "phase": plan.phase,
        "reason": plan.reason,
        "reason_code": plan.reason_code,
        "reference_price": plan.reference_price,
    });
    let intent = oms.create_intent(CreateIntentRequest {
        symbol: &plan.symbol,
        action: &plan.action,
        side: &plan.side,
        requested_size: Some(plan.quantity),
        requested_notional: Some(plan.notional_usd),
        leverage: Some(plan.leverage),
        decision_ts_ms: Some(step_close_ts_ms),
        reason: Some(&plan.reason),
        confidence: Some(&plan.confidence),
        entry_atr: None,
        meta: Some(&intent_meta),
        dedupe_open: plan.action == "OPEN",
        strategy_version: None,
        strategy_sha1: None,
    })?;
    if intent.duplicate && plan.action == "OPEN" {
        return Ok(());
    }

    let order_action = order_action_for_plan(plan)?;
    let reducing = matches!(order_action, OrderAction::Close | OrderAction::Reduce);
    let risk_decision = risk.allow_order(OrderCheck {
        now_ms: step_close_ts_ms,
        symbol: &plan.symbol,
        action: order_action,
        reduce_risk: reducing,
    });
    if !risk_decision.allowed {
        oms.mark_failed(&intent, &risk_decision.reason)?;
        return Ok(());
    }

    let send_allowed = if reducing {
        live_safety::live_orders_enabled()
    } else {
        live_safety::live_entries_enabled()
    };
    if !send_allowed {
        oms.mark_would(&intent, Some("disabled"))?;
        return Ok(());
    }

    if matches!(order_action, OrderAction::Open | OrderAction::Add) {
        let leverage = plan.leverage.round().clamp(1.0, 50.0) as u32;
        let leverage_response = client.update_leverage(&plan.symbol, leverage, true)?;
        if response_has_embedded_error(&leverage_response) {
            oms.mark_failed(&intent, "update_leverage rejected")?;
            return Ok(());
        }
    }

    let response = match order_action {
        OrderAction::Open | OrderAction::Add => client.market_open(
            &plan.symbol,
            plan.side == "BUY",
            plan.quantity,
            Some(plan.reference_price),
            0.01,
            intent.client_order_id.as_deref(),
        ),
        OrderAction::Close | OrderAction::Reduce => client.market_close(
            &plan.symbol,
            plan.side == "BUY",
            plan.quantity,
            Some(plan.reference_price),
            0.01,
            intent.client_order_id.as_deref(),
        ),
    };

    match response {
        Ok(response) if !response_has_embedded_error(&response) => {
            let exchange_order_id = extract_exchange_order_id(&response);
            oms.mark_sent(
                &intent,
                SentOrderRequest {
                    symbol: &plan.symbol,
                    side: &plan.side,
                    order_type: if reducing {
                        "market_close"
                    } else {
                        "market_open"
                    },
                    reduce_only: reducing,
                    requested_size: Some(plan.quantity),
                    result: Some(&response),
                    exchange_order_id: exchange_order_id.as_deref(),
                },
            )?;
            risk.note_order_sent(OrderRecord {
                now_ms: step_close_ts_ms,
                symbol: &plan.symbol,
                action: order_action,
                reduce_risk: reducing,
            });
            note_submitted_order(
                live_db,
                SubmittedOrderProjection {
                    symbol: &plan.symbol,
                    action: submitted_order_action(order_action),
                    ts_ms: step_close_ts_ms,
                    entry_adx_threshold: plan.entry_adx_threshold,
                },
            )?;
        }
        Ok(response) => {
            oms.mark_failed(&intent, &response.to_string())?;
        }
        Err(error) => {
            oms.mark_submit_unknown(
                &intent,
                SubmitUnknownRequest {
                    symbol: &plan.symbol,
                    side: &plan.side,
                    order_type: if reducing {
                        "market_close"
                    } else {
                        "market_open"
                    },
                    reduce_only: reducing,
                    requested_size: Some(plan.quantity),
                    error: Some(&error.to_string()),
                },
            )?;
        }
    }
    Ok(())
}

fn ingest_recent_fills(
    client: &HyperliquidClient,
    oms: &LiveOms,
    live_db: &Path,
    risk: &mut LiveRiskManager,
    last_fill_cursor_ms: &mut i64,
    warnings: &mut Vec<String>,
) -> Result<()> {
    let now_ms = Utc::now().timestamp_millis();
    let start_ms = (*last_fill_cursor_ms - 5 * 60 * 1000).max(0);
    let fills = client.user_fills_by_time(start_ms, now_ms)?;
    for fill in fills {
        if let Some(input) = parse_fill_match_input(&fill.raw)? {
            let matched = oms.match_intent_for_fill(
                &fill.raw,
                &input.symbol,
                &input.action,
                &input.side,
                input.ts_ms,
            )?;
            let inserted = oms.insert_fill(InsertFillRequest {
                ts_ms: input.ts_ms,
                symbol: &input.symbol,
                intent_id: matched.as_ref().map(|matched| matched.intent_id.as_str()),
                order_id: input
                    .exchange_order_id
                    .as_deref()
                    .and_then(|value| value.parse::<i64>().ok()),
                action: Some(&input.action),
                side: Some(&input.side),
                pos_type: Some(&input.pos_type),
                price: input.price,
                size: input.size,
                notional: input.notional_usd,
                fee_usd: Some(input.fee_usd),
                fee_token: None,
                fee_rate: None,
                pnl_usd: Some(input.pnl_usd),
                fill_hash: input.fill_hash.as_deref(),
                fill_tid: input.fill_tid,
                matched_via: matched.as_ref().map(|matched| matched.matched_via),
                raw: Some(&fill.raw),
            })?;
            if inserted {
                if input.exchange_order_id.is_none() {
                    warnings.push(format!(
                        "live fill ingested without exchange_order_id for {} {}",
                        input.symbol, input.action
                    ));
                }
                risk.note_fill(FillEvent {
                    ts_ms: input.ts_ms,
                    symbol: &input.symbol,
                    action: order_action_from_fill(&input.action)?,
                    pnl_usd: input.pnl_usd,
                    fee_usd: input.fee_usd,
                    fill_price: Some(input.price),
                    side: order_side_from_fill(&input.side),
                    ref_mid: None,
                    ref_bid: None,
                    ref_ask: None,
                });
                if input.action == "CLOSE" {
                    let close_reason = matched
                        .as_ref()
                        .and_then(|matched| {
                            oms.load_intent_reason(&matched.intent_id).ok().flatten()
                        })
                        .and_then(|reason| reason.reason_code.or(reason.reason))
                        .unwrap_or_else(|| "exchange_fill".to_string());
                    record_full_close(
                        live_db,
                        &input.symbol,
                        input.ts_ms,
                        &input.side,
                        &close_reason,
                    )?;
                }
            }
        }
    }
    *last_fill_cursor_ms = now_ms;
    ensure_live_runtime_tables(&rusqlite::Connection::open(live_db)?)?;
    Ok(())
}

struct ParsedFillInput {
    ts_ms: i64,
    symbol: String,
    action: String,
    side: String,
    pos_type: String,
    price: f64,
    size: f64,
    notional_usd: f64,
    fee_usd: f64,
    pnl_usd: f64,
    fill_hash: Option<String>,
    fill_tid: Option<i64>,
    exchange_order_id: Option<String>,
}

fn parse_fill_match_input(raw: &serde_json::Value) -> Result<Option<ParsedFillInput>> {
    let symbol = raw
        .get("coin")
        .or_else(|| raw.get("symbol"))
        .and_then(|value| value.as_str())
        .map(|value| value.trim().to_ascii_uppercase())
        .filter(|value| !value.is_empty());
    let Some(symbol) = symbol else {
        return Ok(None);
    };
    let price = raw
        .get("px")
        .and_then(|value| value.as_str())
        .and_then(|value| value.parse::<f64>().ok());
    let size = raw
        .get("sz")
        .and_then(|value| value.as_str())
        .and_then(|value| value.parse::<f64>().ok());
    let Some(price) = price.filter(|value| *value > 0.0) else {
        return Ok(None);
    };
    let Some(size) = size.filter(|value| *value > 0.0) else {
        return Ok(None);
    };
    let ts_ms = raw
        .get("time")
        .or_else(|| raw.get("timestamp"))
        .and_then(|value| value.as_i64())
        .unwrap_or_else(|| Utc::now().timestamp_millis());
    let dir = raw
        .get("dir")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let start_position = raw
        .get("startPosition")
        .and_then(|value| value.as_str())
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);
    let Some((action, pos_type, side)) = classify_fill_direction(&dir, start_position, size) else {
        return Ok(None);
    };
    let fee_usd = raw
        .get("fee")
        .and_then(|value| value.as_str())
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);
    let pnl_usd = raw
        .get("closedPnl")
        .and_then(|value| value.as_str())
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);

    Ok(Some(ParsedFillInput {
        ts_ms,
        symbol,
        action: action.to_string(),
        side: side.to_string(),
        pos_type: pos_type.to_string(),
        price,
        size,
        notional_usd: price * size,
        fee_usd,
        pnl_usd,
        fill_hash: raw
            .get("hash")
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned),
        fill_tid: raw.get("tid").and_then(|value| value.as_i64()),
        exchange_order_id: raw
            .get("oid")
            .map(|value| value.to_string().trim_matches('"').to_string())
            .filter(|value| !value.is_empty()),
    }))
}

fn classify_fill_direction(
    dir: &str,
    start_position: f64,
    fill_size: f64,
) -> Option<(&'static str, &'static str, &'static str)> {
    let pos_type = if dir.contains("long") {
        "LONG"
    } else if dir.contains("short") {
        "SHORT"
    } else {
        return None;
    };
    if dir.starts_with("open") {
        let action = if start_position.abs() < 1e-9 {
            "OPEN"
        } else {
            "ADD"
        };
        let side = if pos_type == "LONG" { "BUY" } else { "SELL" };
        return Some((action, pos_type, side));
    }
    if dir.starts_with("close") {
        let ending_position = if pos_type == "LONG" {
            start_position - fill_size
        } else {
            start_position + fill_size
        };
        let action = if ending_position.abs() < 1e-9 {
            "CLOSE"
        } else {
            "REDUCE"
        };
        let side = if pos_type == "LONG" { "SELL" } else { "BUY" };
        return Some((action, pos_type, side));
    }
    None
}

fn active_symbols(
    symbols: &[String],
    state: &bt_core::decision_kernel::StrategyState,
) -> Vec<String> {
    let mut merged = BTreeSet::new();
    merged.extend(
        symbols
            .iter()
            .map(|symbol| symbol.trim().to_ascii_uppercase())
            .filter(|symbol| !symbol.is_empty()),
    );
    merged.extend(state.positions.keys().cloned());
    merged.into_iter().collect()
}

fn load_symbols(explicit_symbols: &[String], symbols_file: Option<&Path>) -> Result<Vec<String>> {
    let mut merged = explicit_symbols.to_vec();
    if merged.is_empty() {
        if let Ok(raw) = std::env::var("AI_QUANT_SYMBOLS") {
            merged.extend(
                raw.split(',')
                    .map(str::trim)
                    .filter(|symbol| !symbol.is_empty())
                    .map(ToOwned::to_owned),
            );
        }
    }
    if let Some(symbols_file) = symbols_file {
        let file_symbols = std::fs::read_to_string(symbols_file)
            .with_context(|| format!("failed to read symbols file: {}", symbols_file.display()))?
            .lines()
            .map(str::trim)
            .filter(|symbol| !symbol.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        merged.extend(file_symbols);
    }
    let mut symbols = merged
        .into_iter()
        .map(|symbol| symbol.trim().to_ascii_uppercase())
        .filter(|symbol| !symbol.is_empty())
        .collect::<Vec<_>>();
    symbols.sort();
    symbols.dedup();
    Ok(symbols)
}

fn order_action_for_plan(plan: &LiveActionPlan) -> Result<OrderAction> {
    match plan.action.as_str() {
        "OPEN" => Ok(OrderAction::Open),
        "ADD" => Ok(OrderAction::Add),
        "CLOSE" => Ok(OrderAction::Close),
        "REDUCE" => Ok(OrderAction::Reduce),
        other => anyhow::bail!("unsupported live action plan: {other}"),
    }
}

fn order_action_from_fill(action: &str) -> Result<OrderAction> {
    match action {
        "OPEN" => Ok(OrderAction::Open),
        "ADD" => Ok(OrderAction::Add),
        "CLOSE" => Ok(OrderAction::Close),
        "REDUCE" => Ok(OrderAction::Reduce),
        other => anyhow::bail!("unsupported ingested fill action: {other}"),
    }
}

fn submitted_order_action(action: OrderAction) -> SubmittedOrderAction {
    match action {
        OrderAction::Open => SubmittedOrderAction::Open,
        OrderAction::Add => SubmittedOrderAction::Add,
        OrderAction::Close => SubmittedOrderAction::Close,
        OrderAction::Reduce => SubmittedOrderAction::Reduce,
    }
}

fn order_side_from_fill(side: &str) -> Option<OrderSide> {
    match side {
        "BUY" => Some(OrderSide::Buy),
        "SELL" => Some(OrderSide::Sell),
        _ => None,
    }
}

fn latest_common_close_ts_ms(
    candles_db: &Path,
    interval: &str,
    active_symbols: &[String],
    btc_symbol: &str,
) -> Result<i64> {
    let conn = rusqlite::Connection::open(candles_db)?;
    let mut symbols = BTreeSet::new();
    symbols.extend(active_symbols.iter().cloned());
    let btc_symbol = btc_symbol.trim().to_ascii_uppercase();
    if !btc_symbol.is_empty() {
        symbols.insert(btc_symbol);
    }
    let mut latest_common: Option<i64> = None;
    for symbol in symbols {
        let latest_symbol_close = conn
            .query_row(
                "SELECT MAX(COALESCE(t_close, t)) FROM candles WHERE symbol = ?1 AND interval = ?2",
                params![symbol, interval],
                |row| row.get::<_, Option<i64>>(0),
            )?
            .with_context(|| {
                format!(
                    "live daemon requires candle coverage for {} at {}",
                    symbol, interval
                )
            })?;
        latest_common = Some(match latest_common {
            Some(current) => current.min(latest_symbol_close),
            None => latest_symbol_close,
        });
    }
    latest_common.context("live daemon requires at least one available candle close")
}

fn load_last_applied_step_close_ts_ms(
    live_db: &Path,
    config_fingerprint: &str,
    interval: &str,
    live: bool,
) -> Result<Option<i64>> {
    let conn = rusqlite::Connection::open(live_db)?;
    let exists: i64 = conn.query_row(
        "SELECT COUNT(1) FROM sqlite_master WHERE type = 'table' AND name = 'runtime_cycle_steps'",
        [],
        |row| row.get(0),
    )?;
    if exists == 0 {
        return Ok(None);
    }
    let mut stmt = conn.prepare(
        "SELECT step_id, step_close_ts_ms
         FROM runtime_cycle_steps
         WHERE interval = ?1
         ORDER BY step_close_ts_ms DESC",
    )?;
    let rows = stmt.query_map([interval], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })?;
    for row in rows {
        let (step_id, step_close_ts_ms) = row?;
        let expected = crate::paper_cycle::derive_step_id(
            config_fingerprint,
            interval,
            step_close_ts_ms,
            live,
        );
        if step_id == expected {
            return Ok(Some(step_close_ts_ms));
        }
    }
    Ok(None)
}

fn record_runtime_cycle_step(
    live_db: &Path,
    config_fingerprint: &str,
    interval: &str,
    step_close_ts_ms: i64,
    active_symbols: &[String],
    execution_count: usize,
) -> Result<()> {
    let conn = rusqlite::Connection::open(live_db)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS runtime_cycle_steps (
            step_id TEXT PRIMARY KEY,
            step_close_ts_ms INTEGER NOT NULL,
            interval TEXT NOT NULL,
            symbols_json TEXT NOT NULL,
            snapshot_exported_at_ms INTEGER NOT NULL,
            execution_count INTEGER NOT NULL,
            trades_written INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )",
    )?;
    let step_id =
        crate::paper_cycle::derive_step_id(config_fingerprint, interval, step_close_ts_ms, true);
    conn.execute(
        "INSERT OR REPLACE INTO runtime_cycle_steps (
            step_id, step_close_ts_ms, interval, symbols_json, snapshot_exported_at_ms,
            execution_count, trades_written, created_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            step_id,
            step_close_ts_ms,
            interval,
            serde_json::to_string(active_symbols)?,
            step_close_ts_ms,
            execution_count as i64,
            0_i64,
            Utc::now().to_rfc3339()
        ],
    )?;
    Ok(())
}

fn interval_to_ms(interval: &str) -> Result<i64> {
    let interval = interval.trim();
    if let Some(minutes) = interval.strip_suffix('m') {
        let minutes = minutes.parse::<i64>().context("invalid minute interval")?;
        return Ok(minutes * 60_000);
    }
    if let Some(hours) = interval.strip_suffix('h') {
        let hours = hours.parse::<i64>().context("invalid hour interval")?;
        return Ok(hours * 60 * 60_000);
    }
    anyhow::bail!("unsupported interval: {interval}")
}

fn resolve_lock_path(lock_path: Option<&Path>) -> PathBuf {
    lock_path.map(Path::to_path_buf).unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("ai_quant_v8_live.lock")
    })
}

fn resolve_status_path(status_path: Option<&Path>, lock_path: &Path) -> PathBuf {
    status_path.map(Path::to_path_buf).unwrap_or_else(|| {
        let name = lock_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("ai_quant_v8_live.lock");
        let status_name = name
            .strip_suffix(".lock")
            .map(|name| format!("{name}.status.json"))
            .unwrap_or_else(|| format!("{name}.status.json"));
        lock_path
            .parent()
            .map(|parent| parent.join(&status_name))
            .unwrap_or_else(|| PathBuf::from(status_name))
    })
}

fn acquire_lock(lock_path: &Path) -> Result<File> {
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut lock_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(lock_path)
        .with_context(|| format!("failed to open live lock file: {}", lock_path.display()))?;
    lock_file
        .try_lock_exclusive()
        .with_context(|| format!("failed to acquire live lock {}", lock_path.display()))?;
    lock_file.set_len(0)?;
    lock_file.seek(SeekFrom::Start(0))?;
    write!(lock_file, "{}", std::process::id())?;
    lock_file.flush()?;
    Ok(lock_file)
}

fn read_pid(lock_file: &mut File) -> Result<Option<u32>> {
    let mut payload = String::new();
    lock_file
        .seek(SeekFrom::Start(0))
        .context("failed to seek live daemon lock file while reading its pid")?;
    std::io::Read::read_to_string(lock_file, &mut payload)
        .context("failed to read live daemon lock pid")?;
    let payload = payload.trim();
    if payload.is_empty() {
        return Ok(None);
    }
    let pid = payload
        .parse::<u32>()
        .with_context(|| format!("failed to parse live daemon lock pid: {payload}"))?;
    Ok(Some(pid))
}

fn install_signal_handlers(stop_flag: &Arc<AtomicBool>) -> Result<Vec<SigId>> {
    Ok(vec![
        flag::register(SIGINT, Arc::clone(stop_flag))?,
        flag::register(SIGTERM, Arc::clone(stop_flag))?,
    ])
}

fn sleep_with_stop_flag(idle_sleep_ms: u64, stop_flag: &AtomicBool) {
    let sleep = Duration::from_millis(idle_sleep_ms.max(100));
    let step = Duration::from_millis(100);
    let mut elapsed = Duration::ZERO;
    while elapsed < sleep {
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }
        let current = std::cmp::min(step, sleep.saturating_sub(elapsed));
        std::thread::sleep(current);
        elapsed += current;
    }
}

fn build_status(
    input: &LiveDaemonInput<'_>,
    lock_path: &Path,
    started_at_ms: i64,
    latest_common_close_ts_ms: Option<i64>,
    last_step_close_ts_ms: Option<i64>,
    last_fill_cursor_ms: i64,
    warnings: &[String],
    errors: &[String],
    last_cycle: Option<&LiveCycleReport>,
    stopped: bool,
) -> LiveDaemonStatus {
    LiveDaemonStatus {
        ok: errors.is_empty(),
        running: !stopped,
        pid: std::process::id(),
        config_path: input.effective_config.config_path().display().to_string(),
        live_db: input.live_db.display().to_string(),
        candles_db: input.candles_db.display().to_string(),
        lock_path: lock_path.display().to_string(),
        status_path: resolve_status_path(input.status_path, lock_path)
            .display()
            .to_string(),
        started_at_ms,
        updated_at_ms: Utc::now().timestamp_millis(),
        stopped_at_ms: None,
        stop_requested: false,
        runtime_bootstrap: input.runtime_bootstrap.clone(),
        btc_symbol: input.btc_symbol.trim().to_ascii_uppercase(),
        lookback_bars: input.lookback_bars,
        explicit_symbols: input.explicit_symbols.to_vec(),
        symbols_file: input.symbols_file.map(|path| path.display().to_string()),
        latest_common_close_ts_ms,
        last_step_close_ts_ms,
        last_fill_cursor_ms,
        last_plans_count: last_cycle.map(|cycle| cycle.plans.len()).unwrap_or(0),
        warnings: warnings.to_vec(),
        errors: errors.to_vec(),
    }
}

impl LiveDaemonStatus {
    fn with_stopped_at(mut self, stopped_at_ms: i64, stop_requested: bool) -> Self {
        self.running = false;
        self.stopped_at_ms = Some(stopped_at_ms);
        self.stop_requested = stop_requested;
        self.updated_at_ms = stopped_at_ms;
        self
    }
}

fn write_status(status_path: &Path, status: &LiveDaemonStatus) -> Result<()> {
    if let Some(parent) = status_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp_path = status_path.with_extension("status.json.tmp");
    let payload = serde_json::to_vec_pretty(status)?;
    std::fs::write(&tmp_path, payload)?;
    std::fs::rename(&tmp_path, status_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_fill_direction_maps_open_and_close_actions() {
        assert_eq!(
            classify_fill_direction("open long", 0.0, 0.25),
            Some(("OPEN", "LONG", "BUY"))
        );
        assert_eq!(
            classify_fill_direction("open short", -0.5, 0.1),
            Some(("ADD", "SHORT", "SELL"))
        );
        assert_eq!(
            classify_fill_direction("close long", 0.25, 0.25),
            Some(("CLOSE", "LONG", "SELL"))
        );
        assert_eq!(
            classify_fill_direction("close short", -0.5, 0.2),
            Some(("REDUCE", "SHORT", "BUY"))
        );
    }

    #[test]
    fn parse_fill_match_input_extracts_hyperliquid_fill_shape() {
        let fill = serde_json::json!({
            "coin": "ETH",
            "px": "2500.5",
            "sz": "0.25",
            "time": 1_700_000_000_000_i64,
            "dir": "Close Long",
            "startPosition": "0.25",
            "fee": "0.12",
            "closedPnl": "5.4",
            "hash": "0xabc",
            "tid": 42,
            "oid": 987654321_i64
        });

        let parsed = parse_fill_match_input(&fill)
            .expect("fill parsing should succeed")
            .expect("fill should be recognised");

        assert_eq!(parsed.symbol, "ETH");
        assert_eq!(parsed.action, "CLOSE");
        assert_eq!(parsed.side, "SELL");
        assert_eq!(parsed.pos_type, "LONG");
        assert_eq!(parsed.ts_ms, 1_700_000_000_000_i64);
        assert_eq!(parsed.exchange_order_id.as_deref(), Some("987654321"));
        assert_eq!(parsed.fill_hash.as_deref(), Some("0xabc"));
        assert_eq!(parsed.fill_tid, Some(42));
        assert!((parsed.notional_usd - 625.125).abs() < 1e-9);
    }

    #[test]
    fn resolve_status_path_derives_default_from_lock_path() {
        let lock_path = PathBuf::from("/tmp/ai_quant_v8_live.lock");
        let status_path = resolve_status_path(None, &lock_path);
        assert_eq!(
            status_path,
            PathBuf::from("/tmp/ai_quant_v8_live.status.json")
        );

        let explicit = PathBuf::from("/tmp/custom-live-status.json");
        assert_eq!(
            resolve_status_path(Some(explicit.as_path()), &lock_path),
            explicit
        );
    }
}
