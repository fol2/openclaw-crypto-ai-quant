use aiq_runtime_core::paper::PaperPositionState;
use aiq_runtime_core::{PipelinePlan, StageId};
use anyhow::Result;
use bt_core::config::Confidence;
use bt_core::decision_kernel::{OrderIntentKind, PositionSide, StrategyState};
use serde::Serialize;

use crate::paper_config::PaperEffectiveConfig;
use crate::paper_run_once::{
    action_codes_for_symbol, execute_prepared_symbol_step,
    execute_prepared_symbol_step_with_allow_pyramid_override, prepare_symbol_step,
    PreparedSymbolStep,
};

pub struct LiveCycleInput<'a> {
    pub effective_config: PaperEffectiveConfig,
    pub runtime_bootstrap: aiq_runtime_core::runtime::RuntimeBootstrap,
    pub state: StrategyState,
    pub explicit_symbols: &'a [String],
    pub candles_db: &'a std::path::Path,
    pub btc_symbol: &'a str,
    pub lookback_bars: usize,
    pub step_close_ts_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LiveCycleStageTrace {
    pub stage: StageId,
    pub enabled: bool,
    pub status: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LiveActionPlan {
    pub symbol: String,
    pub phase: String,
    pub action: String,
    pub side: String,
    pub quantity: f64,
    pub reference_price: f64,
    pub notional_usd: f64,
    pub leverage: f64,
    pub confidence: String,
    pub reason: String,
    pub reason_code: String,
    pub entry_adx_threshold: Option<f64>,
    pub score: Option<f64>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LiveCycleReport {
    pub ok: bool,
    pub interval: String,
    pub active_symbols: Vec<String>,
    pub candidate_count: usize,
    pub executed_entry_count: usize,
    pub plans: Vec<LiveActionPlan>,
    pub pipeline_trace: Vec<LiveCycleStageTrace>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
struct EntryCandidate {
    prepared: PreparedSymbolStep,
    score: f64,
}

#[derive(Debug, Clone)]
struct PlannedExecution {
    symbol: String,
    phase: &'static str,
    score: Option<f64>,
    prepared: PreparedSymbolStep,
    pre_state: StrategyState,
    decision: bt_core::decision_kernel::DecisionResult,
    leverage: f64,
    warnings: Vec<String>,
}

pub fn run_cycle(input: LiveCycleInput<'_>) -> Result<LiveCycleReport> {
    if !input.candles_db.exists() {
        anyhow::bail!("candles db not found: {}", input.candles_db.display());
    }
    if input.step_close_ts_ms <= 0 {
        anyhow::bail!("step_close_ts_ms must be positive");
    }

    let mut explicit_symbols = normalise_symbols(input.explicit_symbols);
    let open_symbols = input.state.positions.keys().cloned().collect::<Vec<_>>();
    let mut active_symbols = std::collections::BTreeSet::new();
    active_symbols.extend(explicit_symbols.iter().cloned());
    active_symbols.extend(open_symbols.iter().cloned());
    if active_symbols.is_empty() {
        anyhow::bail!("live cycle requires explicit symbols or open exchange positions");
    }
    explicit_symbols.sort();
    let active_symbols = active_symbols.into_iter().collect::<Vec<_>>();

    let base_cfg = input.effective_config.load_config(None, true)?;
    let max_entries = base_cfg.trade.max_entry_orders_per_loop;
    let pipeline = &input.runtime_bootstrap.pipeline;
    let ordered_stage_ids = pipeline.ordered_stage_ids();
    let ranking_order_supported = !pipeline.is_enabled(StageId::Ranking)
        || !pipeline.is_enabled(StageId::IntentGeneration)
        || stage_precedes(
            &ordered_stage_ids,
            StageId::Ranking,
            StageId::IntentGeneration,
        );
    let state_order_supported = !pipeline.is_enabled(StageId::IntentGeneration)
        || !pipeline.is_enabled(StageId::OmsTransition)
        || stage_precedes(
            &ordered_stage_ids,
            StageId::IntentGeneration,
            StageId::OmsTransition,
        );
    let broker_order_supported = !pipeline.is_enabled(StageId::BrokerExecution)
        || !pipeline.is_enabled(StageId::FillReconciliation)
        || stage_precedes(
            &ordered_stage_ids,
            StageId::BrokerExecution,
            StageId::FillReconciliation,
        );
    let ranking_applied = pipeline.is_enabled(StageId::Ranking) && ranking_order_supported;
    let intent_enabled = pipeline.is_enabled(StageId::IntentGeneration);
    let state_progression_enabled =
        intent_enabled && pipeline.is_enabled(StageId::OmsTransition) && state_order_supported;
    let execution_enabled = state_progression_enabled
        && pipeline.is_enabled(StageId::BrokerExecution)
        && broker_order_supported;

    let mut interval: Option<String> = None;
    let mut candidate_entries = Vec::new();
    let mut planned = Vec::new();
    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    let mut current_state = input.state.clone();
    let mut used_entry_budget = 0usize;

    if pipeline.is_enabled(StageId::Ranking) && !ranking_order_supported {
        warnings.push(format!(
            "pipeline profile {} orders ranking after intent_generation; live cycle fell back to collection order",
            pipeline.profile
        ));
    }
    if intent_enabled && pipeline.is_enabled(StageId::OmsTransition) && !state_order_supported {
        warnings.push(format!(
            "pipeline profile {} orders oms_transition before intent_generation; live cycle stayed preview-only for state progression",
            pipeline.profile
        ));
    }
    if pipeline.is_enabled(StageId::BrokerExecution)
        && pipeline.is_enabled(StageId::FillReconciliation)
        && !broker_order_supported
    {
        warnings.push(format!(
            "pipeline profile {} orders fill_reconciliation before broker_execution; live cycle disabled broker submission planning",
            pipeline.profile
        ));
    }

    for symbol in &active_symbols {
        let config = input.effective_config.load_config(Some(symbol), true)?;
        match &interval {
            Some(current_interval) if current_interval != &config.engine.interval => {
                anyhow::bail!(
                    "live cycle requires a shared interval; {} resolved to {} but prior symbols use {}",
                    symbol,
                    config.engine.interval,
                    current_interval
                );
            }
            None => interval = Some(config.engine.interval.clone()),
            _ => {}
        }
        let prior_position = input
            .state
            .positions
            .get(symbol)
            .map(|position| PaperPositionState {
                symbol: position.symbol.clone(),
                side: match position.side {
                    PositionSide::Long => "long".to_string(),
                    PositionSide::Short => "short".to_string(),
                },
                size: position.quantity,
                entry_price: position.avg_entry_price,
                entry_atr: position.entry_atr.unwrap_or(0.0),
                trailing_sl: position.trailing_sl,
                confidence: match position.confidence.unwrap_or(1) {
                    2 => "high".to_string(),
                    1 => "medium".to_string(),
                    _ => "low".to_string(),
                },
                leverage: if position.margin_usd > 0.0 {
                    position.notional_usd / position.margin_usd
                } else {
                    1.0
                },
                margin_used: position.margin_usd,
                adds_count: position.adds_count,
                tp1_taken: position.tp1_taken,
                open_time_ms: position.opened_at_ms,
                last_funding_time_ms: position.last_funding_ms.unwrap_or(position.opened_at_ms),
                last_add_time_ms: 0,
                entry_adx_threshold: position.entry_adx_threshold.unwrap_or(0.0),
            });
        let prepared = prepare_symbol_step(
            &config,
            prior_position.as_ref(),
            input.candles_db,
            symbol,
            &input.btc_symbol.trim().to_ascii_uppercase(),
            input.lookback_bars,
            Some(input.step_close_ts_ms),
        )?;
        let pre_state = current_state.clone();
        let (decision, execution_plan) =
            execute_prepared_symbol_step(&pre_state, &prepared, input.step_close_ts_ms);

        if pre_state.positions.contains_key(symbol) {
            if state_progression_enabled
                && decision_consumes_entry_budget(&decision)
                && used_entry_budget >= max_entries
            {
                let (budgeted_decision, mut budgeted_plan) =
                    execute_prepared_symbol_step_with_allow_pyramid_override(
                        &pre_state,
                        &prepared,
                        input.step_close_ts_ms,
                        Some(false),
                    );
                let budget_warning = format!(
                    "skip open-position entry for {}: max_entry_orders_per_loop {} exhausted",
                    symbol, max_entries
                );
                warnings.push(budget_warning.clone());
                budgeted_plan.warnings.push(budget_warning);
                errors.extend(budgeted_decision.diagnostics.errors.clone());
                warnings.extend(budgeted_plan.warnings.clone());
                if !budgeted_decision.intents.is_empty() || !budgeted_decision.fills.is_empty() {
                    if state_progression_enabled {
                        current_state = budgeted_decision.state.clone();
                    }
                    planned.push(PlannedExecution {
                        symbol: symbol.clone(),
                        phase: if execution_enabled {
                            "open_position"
                        } else if state_progression_enabled {
                            "open_position_staged"
                        } else {
                            "open_position_preview"
                        },
                        score: None,
                        prepared,
                        pre_state,
                        leverage: budgeted_plan.leverage,
                        decision: budgeted_decision,
                        warnings: budgeted_plan.warnings,
                    });
                }
                continue;
            }

            warnings.extend(execution_plan.warnings.clone());
            errors.extend(decision.diagnostics.errors.clone());
            if state_progression_enabled && decision_consumes_entry_budget(&decision) {
                used_entry_budget += 1;
            }
            if !decision.intents.is_empty() || !decision.fills.is_empty() {
                if state_progression_enabled {
                    current_state = decision.state.clone();
                }
                planned.push(PlannedExecution {
                    symbol: symbol.clone(),
                    phase: if execution_enabled {
                        "open_position"
                    } else if state_progression_enabled {
                        "open_position_staged"
                    } else {
                        "open_position_preview"
                    },
                    score: None,
                    prepared,
                    pre_state,
                    leverage: execution_plan.leverage,
                    decision,
                    warnings: execution_plan.warnings,
                });
            }
            continue;
        }

        if decision
            .intents
            .iter()
            .any(|intent| matches!(intent.kind, OrderIntentKind::Open | OrderIntentKind::Add))
        {
            candidate_entries.push(EntryCandidate {
                score: entry_score(prepared.execution_metadata.confidence, prepared.snap.adx),
                prepared,
            });
        } else {
            warnings.extend(execution_plan.warnings);
            errors.extend(decision.diagnostics.errors);
        }
    }

    if ranking_applied {
        sort_entry_candidates(pipeline.ranker.as_str(), &mut candidate_entries);
    }
    let candidate_count = candidate_entries.len();
    let mut executed_entry_count = 0usize;

    for candidate in candidate_entries {
        if state_progression_enabled && used_entry_budget >= max_entries {
            warnings.push(format!(
                "live cycle entry limit reached at {} candidate(s)",
                max_entries
            ));
            break;
        }
        let pre_state = current_state.clone();
        let (decision, execution_plan) =
            execute_prepared_symbol_step(&pre_state, &candidate.prepared, input.step_close_ts_ms);
        warnings.extend(execution_plan.warnings.clone());
        errors.extend(decision.diagnostics.errors.clone());
        if decision
            .intents
            .iter()
            .any(|intent| matches!(intent.kind, OrderIntentKind::Open | OrderIntentKind::Add))
        {
            if state_progression_enabled {
                current_state = decision.state.clone();
                used_entry_budget += 1;
                executed_entry_count += 1;
            }
            planned.push(PlannedExecution {
                symbol: candidate.prepared.symbol.clone(),
                phase: if execution_enabled {
                    if ranking_applied {
                        "ranked_entry"
                    } else {
                        "entry"
                    }
                } else if state_progression_enabled {
                    if ranking_applied {
                        "ranked_entry_staged"
                    } else {
                        "entry_staged"
                    }
                } else if ranking_applied {
                    "ranked_entry_preview"
                } else {
                    "entry_preview"
                },
                score: if ranking_applied {
                    Some(candidate.score)
                } else {
                    None
                },
                prepared: candidate.prepared,
                pre_state,
                leverage: execution_plan.leverage,
                decision,
                warnings: execution_plan.warnings,
            });
        }
    }

    let plans = planned
        .into_iter()
        .filter_map(build_live_action_plan)
        .collect::<Vec<_>>();
    let pipeline_trace = build_pipeline_trace(
        pipeline,
        candidate_count,
        executed_entry_count,
        ranking_applied,
        state_progression_enabled,
        execution_enabled,
    );

    Ok(LiveCycleReport {
        ok: errors.is_empty(),
        interval: interval.unwrap_or_else(|| "unknown".to_string()),
        active_symbols,
        candidate_count,
        executed_entry_count,
        plans,
        pipeline_trace,
        warnings,
        errors,
    })
}

fn build_live_action_plan(execution: PlannedExecution) -> Option<LiveActionPlan> {
    let intent = execution.decision.intents.first()?;
    let fill = execution.decision.fills.first()?;
    let action_codes = action_codes_for_symbol(
        &execution.symbol,
        &execution.pre_state,
        &execution.decision.intents,
        &execution.decision.fills,
    );
    let action = action_codes.first()?.to_string();
    let entry_adx_threshold = matches!(action.as_str(), "OPEN" | "ADD")
        .then_some(execution.prepared.execution_metadata.entry_adx_threshold)
        .filter(|value| *value > 0.0);
    let side = order_side_for_action(&action, fill.side)?;
    Some(LiveActionPlan {
        symbol: execution.symbol,
        phase: execution.phase.to_string(),
        action,
        side,
        quantity: fill.quantity,
        reference_price: fill.price,
        notional_usd: fill.notional_usd,
        leverage: execution.leverage,
        confidence: confidence_label(execution.prepared.execution_metadata.confidence).to_string(),
        reason: intent.reason.clone(),
        reason_code: intent.reason_code.clone(),
        entry_adx_threshold,
        score: execution.score,
        warnings: execution
            .warnings
            .into_iter()
            .chain(execution.decision.diagnostics.warnings)
            .collect(),
        errors: execution.decision.diagnostics.errors,
    })
}

fn confidence_label(confidence: Confidence) -> &'static str {
    match confidence {
        Confidence::High => "high",
        Confidence::Medium => "medium",
        Confidence::Low => "low",
    }
}

fn order_side_for_action(action: &str, side: PositionSide) -> Option<String> {
    let action = action.trim().to_ascii_uppercase();
    let value = match (action.as_str(), side) {
        ("OPEN", PositionSide::Long) | ("ADD", PositionSide::Long) => "BUY",
        ("OPEN", PositionSide::Short) | ("ADD", PositionSide::Short) => "SELL",
        ("CLOSE", PositionSide::Long) | ("REDUCE", PositionSide::Long) => "SELL",
        ("CLOSE", PositionSide::Short) | ("REDUCE", PositionSide::Short) => "BUY",
        _ => return None,
    };
    Some(value.to_string())
}

fn normalise_symbols(raw: &[String]) -> Vec<String> {
    let mut symbols = raw
        .iter()
        .map(|symbol| symbol.trim().to_ascii_uppercase())
        .filter(|symbol| !symbol.is_empty())
        .collect::<Vec<_>>();
    symbols.sort();
    symbols.dedup();
    symbols
}

fn entry_score(confidence: Confidence, adx: f64) -> f64 {
    let conf_rank = match confidence {
        Confidence::Low => 0,
        Confidence::Medium => 1,
        Confidence::High => 2,
    };
    f64::from(conf_rank * 100) + adx
}

fn sort_entry_candidates(ranker: &str, candidates: &mut [EntryCandidate]) {
    let _ = ranker.trim();
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.prepared.symbol.cmp(&right.prepared.symbol))
    });
}

fn stage_precedes(ordered_stage_ids: &[StageId], left: StageId, right: StageId) -> bool {
    let left_index = ordered_stage_ids.iter().position(|stage| *stage == left);
    let right_index = ordered_stage_ids.iter().position(|stage| *stage == right);
    match (left_index, right_index) {
        (Some(left_index), Some(right_index)) => left_index < right_index,
        _ => false,
    }
}

fn decision_consumes_entry_budget(decision: &bt_core::decision_kernel::DecisionResult) -> bool {
    decision
        .intents
        .iter()
        .any(|intent| matches!(intent.kind, OrderIntentKind::Open | OrderIntentKind::Add))
}

fn build_pipeline_trace(
    pipeline: &PipelinePlan,
    candidate_count: usize,
    executed_entry_count: usize,
    ranking_applied: bool,
    state_progression_enabled: bool,
    execution_enabled: bool,
) -> Vec<LiveCycleStageTrace> {
    pipeline
        .stages
        .iter()
        .map(|stage| {
            let (status, detail) = match stage.id {
                StageId::MarketDataNormalisation => (
                    "executed",
                    "loaded the live symbol universe and aligned exchange-backed state".to_string(),
                ),
                StageId::IndicatorBuild => (
                    "executed",
                    "built indicator snapshots during symbol preparation".to_string(),
                ),
                StageId::GateEvaluation => (
                    "executed",
                    "evaluated pre-signal gates during symbol preparation".to_string(),
                ),
                StageId::SignalGeneration => (
                    "executed",
                    "generated kernel intents and fills for each active symbol".to_string(),
                ),
                StageId::RiskChecks => (
                    if stage.enabled { "executed" } else { "skipped" },
                    if stage.enabled {
                        "live cycle kept kernel risk semantics active for sizing and cooldowns"
                            .to_string()
                    } else {
                        "pipeline disabled risk_checks; only runtime broker/risk guards will remain downstream"
                            .to_string()
                    },
                ),
                StageId::Ranking => (
                    if !stage.enabled {
                        "skipped"
                    } else if ranking_applied {
                        "executed"
                    } else if candidate_count > 0 {
                        "deferred"
                    } else {
                        "executed"
                    },
                    if candidate_count == 0 {
                        "no ranked entry candidates were available in this cycle".to_string()
                    } else if ranking_applied {
                        format!("ranked {candidate_count} live entry candidate(s)")
                    } else {
                        "pipeline ordering prevented ranking from applying before execution".to_string()
                    },
                ),
                StageId::IntentGeneration => (
                    if stage.enabled { "executed" } else { "skipped" },
                    "prepared runtime action plans from the kernel decision outputs".to_string(),
                ),
                StageId::OmsTransition => (
                    if !stage.enabled {
                        "skipped"
                    } else if state_progression_enabled {
                        "executed"
                    } else {
                        "deferred"
                    },
                    if state_progression_enabled {
                        "live cycle advanced staged action plans with OMS-ready intent semantics"
                            .to_string()
                    } else {
                        "pipeline ordering prevented OMS transition planning from becoming authoritative"
                            .to_string()
                    },
                ),
                StageId::BrokerExecution => (
                    if !stage.enabled {
                        "skipped"
                    } else if execution_enabled {
                        "executed"
                    } else {
                        "deferred"
                    },
                    if execution_enabled {
                        format!(
                            "prepared {} broker submission plan(s) for downstream execution",
                            executed_entry_count
                        )
                    } else {
                        "broker execution remains disabled until the live adapter is attached"
                            .to_string()
                    },
                ),
                StageId::FillReconciliation => (
                    if stage.enabled { "pending" } else { "skipped" },
                    "fill reconciliation remains a downstream live daemon responsibility".to_string(),
                ),
                StageId::PersistenceAudit => (
                    if stage.enabled { "pending" } else { "skipped" },
                    "persistence audit is deferred to the live OMS/fill sink".to_string(),
                ),
            };
            LiveCycleStageTrace {
                stage: stage.id,
                enabled: stage.enabled,
                status: status.to_string(),
                detail,
            }
        })
        .collect()
}
