use aiq_runtime_core::paper::{restore_paper_state, PaperBootstrapReport};
use aiq_runtime_core::runtime::RuntimeBootstrap;
use aiq_runtime_core::StageId;
use anyhow::Result;
use bt_core::config::Confidence;
use bt_core::decision_kernel::{self, OrderIntentKind, StrategyState};
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::Serialize;
use std::collections::BTreeSet;
use std::path::Path;

use crate::paper_config::PaperEffectiveConfig;
use crate::paper_export;
use crate::paper_run_once::{
    action_codes_for_symbol, apply_decision_projection_with_tx, iso_from_ms, prepare_symbol_step,
    PreparedSymbolStep, ProjectionInput,
};

pub struct PaperCycleInput<'a> {
    pub effective_config: PaperEffectiveConfig,
    pub runtime_bootstrap: RuntimeBootstrap,
    pub live: bool,
    pub paper_db: &'a Path,
    pub candles_db: &'a Path,
    pub explicit_symbols: &'a [String],
    pub btc_symbol: &'a str,
    pub lookback_bars: usize,
    pub step_close_ts_ms: i64,
    pub exported_at_ms: Option<i64>,
    pub dry_run: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperCycleSymbolReport {
    pub symbol: String,
    pub phase: String,
    pub score: Option<f64>,
    pub intent_count: usize,
    pub fill_count: usize,
    pub action_codes: Vec<String>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperCycleStageTrace {
    pub stage: StageId,
    pub enabled: bool,
    pub status: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperCycleReport {
    pub ok: bool,
    pub dry_run: bool,
    pub step_id: String,
    pub step_close_ts_ms: i64,
    pub snapshot_exported_at_ms: i64,
    pub interval: String,
    pub runtime_bootstrap: RuntimeBootstrap,
    pub paper_bootstrap: PaperBootstrapReport,
    pub explicit_symbols: Vec<String>,
    pub active_symbols: Vec<String>,
    pub candidate_count: usize,
    pub executed_entry_count: usize,
    pub trades_written: usize,
    pub runtime_step_recorded: bool,
    pub executions: Vec<PaperCycleSymbolReport>,
    pub pipeline_trace: Vec<PaperCycleStageTrace>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
struct EntryCandidate {
    prepared: PreparedSymbolStep,
    score: f64,
    requested_notional_usd: f64,
}

#[derive(Debug, Clone)]
struct ExecutedStep {
    symbol: String,
    phase: &'static str,
    score: Option<f64>,
    prepared: PreparedSymbolStep,
    pre_state: StrategyState,
    decision: decision_kernel::DecisionResult,
    warnings: Vec<String>,
}

pub fn run_cycle(input: PaperCycleInput<'_>) -> Result<PaperCycleReport> {
    if !input.paper_db.exists() {
        anyhow::bail!("paper db not found: {}", input.paper_db.display());
    }
    if !input.candles_db.exists() {
        anyhow::bail!("candles db not found: {}", input.candles_db.display());
    }
    if input.step_close_ts_ms <= 0 {
        anyhow::bail!("step_close_ts_ms must be positive");
    }

    let exported_at_ms = input
        .exported_at_ms
        .unwrap_or_else(|| Utc::now().timestamp_millis());
    let snapshot = paper_export::export_paper_snapshot(input.paper_db, exported_at_ms)?;
    let (paper_state, paper_bootstrap) =
        restore_paper_state(&snapshot).map_err(anyhow::Error::msg)?;
    let mut current_state = paper_state.clone().into_strategy_state();
    let initial_positions = paper_state.positions;

    let mut explicit_symbols = normalise_symbols(input.explicit_symbols);
    let open_symbols = initial_positions.keys().cloned().collect::<Vec<_>>();
    let mut active_symbols = BTreeSet::new();
    active_symbols.extend(explicit_symbols.iter().cloned());
    active_symbols.extend(open_symbols.iter().cloned());
    if active_symbols.is_empty() {
        anyhow::bail!("paper cycle requires explicit symbols or open paper positions");
    }
    explicit_symbols.sort();
    let active_symbols = active_symbols.into_iter().collect::<Vec<_>>();
    let base_cfg = input.effective_config.load_config(None, input.live)?;
    let max_entries = base_cfg.trade.max_entry_orders_per_loop;
    let mut used_entry_budget = 0usize;
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
    let persistence_order_supported = !pipeline.is_enabled(StageId::PersistenceAudit)
        || ((!pipeline.is_enabled(StageId::OmsTransition)
            || stage_precedes(
                &ordered_stage_ids,
                StageId::OmsTransition,
                StageId::PersistenceAudit,
            ))
            && (!pipeline.is_enabled(StageId::FillReconciliation)
                || stage_precedes(
                    &ordered_stage_ids,
                    StageId::FillReconciliation,
                    StageId::PersistenceAudit,
                )));
    let ranking_applied = pipeline.is_enabled(StageId::Ranking) && ranking_order_supported;
    let intent_enabled = pipeline.is_enabled(StageId::IntentGeneration);
    let state_progression_enabled =
        intent_enabled && pipeline.is_enabled(StageId::OmsTransition) && state_order_supported;
    let projection_enabled = state_progression_enabled
        && pipeline.is_enabled(StageId::BrokerExecution)
        && pipeline.is_enabled(StageId::FillReconciliation)
        && broker_order_supported;
    let persistence_enabled = pipeline.is_enabled(StageId::PersistenceAudit)
        && !input.dry_run
        && persistence_order_supported;

    let mut interval: Option<String> = None;
    let mut candidate_entries = Vec::new();
    let mut executed_steps = Vec::new();
    let mut warnings = Vec::new();
    let mut errors = Vec::new();

    if pipeline.is_enabled(StageId::Ranking) && !ranking_order_supported {
        warnings.push(format!(
            "pipeline profile {} orders ranking after intent_generation; paper cycle fell back to collection order",
            pipeline.profile
        ));
    }
    if intent_enabled && pipeline.is_enabled(StageId::OmsTransition) && !state_order_supported {
        warnings.push(format!(
            "pipeline profile {} orders oms_transition before intent_generation; paper cycle stayed preview-only for state progression",
            pipeline.profile
        ));
    }
    if pipeline.is_enabled(StageId::BrokerExecution)
        && pipeline.is_enabled(StageId::FillReconciliation)
        && !broker_order_supported
    {
        warnings.push(format!(
            "pipeline profile {} orders fill_reconciliation before broker_execution; paper cycle disabled trade projection writes",
            pipeline.profile
        ));
    }
    if pipeline.is_enabled(StageId::PersistenceAudit) && !persistence_order_supported {
        warnings.push(format!(
            "pipeline profile {} orders persistence_audit before upstream execution stages; paper cycle deferred persistence",
            pipeline.profile
        ));
    }
    if !pipeline.is_enabled(StageId::RiskChecks) && state_progression_enabled {
        warnings.push(
            "pipeline disables risk_checks, but paper cycle still uses the kernel risk path while execution remains enabled".to_string(),
        );
    }

    for symbol in &active_symbols {
        let config = input
            .effective_config
            .load_config(Some(symbol), input.live)?;
        match &interval {
            Some(current_interval) if current_interval != &config.engine.interval => {
                anyhow::bail!(
                    "paper cycle requires a shared interval; {} resolved to {} but prior symbols use {}",
                    symbol,
                    config.engine.interval,
                    current_interval
                );
            }
            None => interval = Some(config.engine.interval.clone()),
            _ => {}
        }
        let prepared = prepare_symbol_step(
            &config,
            initial_positions.get(symbol),
            input.candles_db,
            symbol,
            &input.btc_symbol.trim().to_ascii_uppercase(),
            input.lookback_bars,
            Some(input.step_close_ts_ms),
        )?;
        let pre_state = current_state.clone();
        let (decision, execution_plan) = crate::paper_run_once::execute_prepared_symbol_step(
            &pre_state,
            &prepared,
            input.step_close_ts_ms,
        );

        if pre_state.positions.contains_key(symbol) {
            if state_progression_enabled
                && decision_consumes_entry_budget(&decision)
                && used_entry_budget >= max_entries
            {
                let (budgeted_decision, mut budgeted_plan) =
                    crate::paper_run_once::execute_prepared_symbol_step_with_allow_pyramid_override(
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
                    if !state_progression_enabled {
                        budgeted_plan.warnings.push(execution_disabled_warning(
                            pipeline.profile.as_str(),
                            projection_enabled,
                            state_progression_enabled,
                        ));
                    }
                    executed_steps.push(ExecutedStep {
                        symbol: symbol.clone(),
                        phase: if projection_enabled {
                            "open_position"
                        } else if state_progression_enabled {
                            "open_position_staged"
                        } else {
                            "open_position_preview"
                        },
                        score: None,
                        prepared,
                        pre_state,
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
                let mut execution_warnings = execution_plan.warnings;
                if state_progression_enabled {
                    current_state = decision.state.clone();
                } else {
                    execution_warnings.push(execution_disabled_warning(
                        pipeline.profile.as_str(),
                        projection_enabled,
                        state_progression_enabled,
                    ));
                }
                executed_steps.push(ExecutedStep {
                    symbol: symbol.clone(),
                    phase: if projection_enabled {
                        "open_position"
                    } else if state_progression_enabled {
                        "open_position_staged"
                    } else {
                        "open_position_preview"
                    },
                    score: None,
                    prepared,
                    pre_state,
                    decision,
                    warnings: execution_warnings,
                });
            }
            continue;
        }

        if decision
            .intents
            .iter()
            .any(|intent| matches!(intent.kind, OrderIntentKind::Open | OrderIntentKind::Add))
        {
            let score = entry_score(prepared.execution_metadata.confidence, prepared.snap.adx);
            candidate_entries.push(EntryCandidate {
                prepared,
                score,
                requested_notional_usd: execution_plan.requested_notional_usd.unwrap_or(0.0),
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
                "paper cycle entry limit reached at {} candidate(s)",
                max_entries
            ));
            break;
        }
        let pre_state = current_state.clone();
        let (decision, execution_plan) = crate::paper_run_once::execute_prepared_symbol_step(
            &pre_state,
            &candidate.prepared,
            input.step_close_ts_ms,
        );
        warnings.extend(execution_plan.warnings.clone());
        errors.extend(decision.diagnostics.errors.clone());
        if decision
            .intents
            .iter()
            .any(|intent| matches!(intent.kind, OrderIntentKind::Open | OrderIntentKind::Add))
        {
            let mut execution_warnings = execution_plan.warnings;
            if state_progression_enabled {
                current_state = decision.state.clone();
                used_entry_budget += 1;
                executed_entry_count += 1;
            } else {
                execution_warnings.push(execution_disabled_warning(
                    pipeline.profile.as_str(),
                    projection_enabled,
                    state_progression_enabled,
                ));
            }
            executed_steps.push(ExecutedStep {
                symbol: candidate.prepared.symbol.clone(),
                phase: if projection_enabled {
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
                decision,
                warnings: execution_warnings,
            });
        }
    }

    let step_id = derive_step_id(
        &input.runtime_bootstrap.config_fingerprint,
        interval.as_deref().unwrap_or("unknown"),
        input.step_close_ts_ms,
        input.live,
    );

    let mut trades_written = 0usize;
    let mut runtime_step_recorded = false;
    if persistence_enabled {
        let mut conn = Connection::open(input.paper_db)?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        ensure_cycle_tables(&tx)?;
        if cycle_step_exists(&tx, &step_id)? {
            anyhow::bail!("paper cycle step already applied: {}", step_id);
        }

        if projection_enabled {
            for execution in &executed_steps {
                let projection = ProjectionInput {
                    symbol: &execution.symbol,
                    pre_state: &execution.pre_state,
                    prior_position: execution.prepared.prior_position.as_ref(),
                    post_state: &execution.decision.state,
                    intents: &execution.decision.intents,
                    fills: &execution.decision.fills,
                    snap: &execution.prepared.snap,
                    execution_metadata: execution.prepared.execution_metadata,
                    ts_ms: input.step_close_ts_ms,
                };
                let (written, _, _) = apply_decision_projection_with_tx(&tx, &projection)?;
                trades_written += written;
            }
        }

        let cycle_step = CycleStepRecord {
            step_id: &step_id,
            step_close_ts_ms: input.step_close_ts_ms,
            interval: interval.as_deref().unwrap_or("unknown"),
            active_symbols: &active_symbols,
            snapshot_exported_at_ms: snapshot.exported_at_ms,
            execution_count: executed_steps.len(),
            trades_written,
        };
        record_cycle_step(&tx, &cycle_step)?;
        tx.commit()?;
        runtime_step_recorded = true;
    } else if input.dry_run {
        warnings.push(
            "persistence_audit disabled by dry_run; no runtime cycle step was recorded".to_string(),
        );
    } else if !pipeline.is_enabled(StageId::PersistenceAudit) {
        warnings.push(format!(
            "pipeline profile {} disabled persistence_audit; paper cycle stayed non-persistent",
            pipeline.profile
        ));
    }

    let executions = executed_steps
        .into_iter()
        .map(|execution| PaperCycleSymbolReport {
            symbol: execution.symbol.clone(),
            phase: execution.phase.to_string(),
            score: execution.score,
            intent_count: execution.decision.intents.len(),
            fill_count: execution.decision.fills.len(),
            action_codes: action_codes_for_symbol(
                &execution.symbol,
                &execution.pre_state,
                &execution.decision.intents,
                &execution.decision.fills,
            ),
            warnings: execution
                .warnings
                .into_iter()
                .chain(execution.decision.diagnostics.warnings)
                .collect(),
            errors: execution.decision.diagnostics.errors,
        })
        .collect::<Vec<_>>();
    let pipeline_trace = build_pipeline_trace(
        pipeline,
        candidate_count,
        executed_entry_count,
        ranking_applied,
        state_progression_enabled,
        projection_enabled,
        persistence_enabled,
    );

    Ok(PaperCycleReport {
        ok: errors.is_empty(),
        dry_run: input.dry_run,
        step_id,
        step_close_ts_ms: input.step_close_ts_ms,
        snapshot_exported_at_ms: snapshot.exported_at_ms,
        interval: interval.unwrap_or_else(|| "unknown".to_string()),
        runtime_bootstrap: input.runtime_bootstrap,
        paper_bootstrap,
        explicit_symbols,
        active_symbols,
        candidate_count,
        executed_entry_count,
        trades_written,
        runtime_step_recorded,
        executions,
        pipeline_trace,
        warnings,
        errors,
    })
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
    match ranker.trim() {
        "raw_notional_desc" => candidates.sort_by(|left, right| {
            right
                .requested_notional_usd
                .partial_cmp(&left.requested_notional_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.prepared.symbol.cmp(&right.prepared.symbol))
        }),
        _ => candidates.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.prepared.symbol.cmp(&right.prepared.symbol))
        }),
    }
}

fn execution_disabled_warning(
    profile: &str,
    projection_enabled: bool,
    state_progression_enabled: bool,
) -> String {
    if state_progression_enabled && !projection_enabled {
        format!(
            "pipeline profile {} disabled broker/fill projection stages; this cycle advanced intents/OMS in memory without trade writes",
            profile
        )
    } else {
        format!(
            "pipeline profile {} disabled one or more execution stages; this cycle ran in preview-only mode",
            profile
        )
    }
}

fn stage_precedes(ordered_stage_ids: &[StageId], left: StageId, right: StageId) -> bool {
    let left_index = ordered_stage_ids.iter().position(|stage| *stage == left);
    let right_index = ordered_stage_ids.iter().position(|stage| *stage == right);
    match (left_index, right_index) {
        (Some(left_index), Some(right_index)) => left_index < right_index,
        _ => false,
    }
}

fn build_pipeline_trace(
    pipeline: &aiq_runtime_core::PipelinePlan,
    candidate_count: usize,
    executed_entry_count: usize,
    ranking_applied: bool,
    state_progression_enabled: bool,
    projection_enabled: bool,
    persistence_enabled: bool,
) -> Vec<PaperCycleStageTrace> {
    pipeline
        .stages
        .iter()
        .map(|stage| {
            let (status, detail) = match stage.id {
                StageId::MarketDataNormalisation => (
                    "executed",
                    "resolved paper snapshot and symbol universe for this cycle".to_string(),
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
                    format!("prepared {} candidate symbol(s) for this cycle", candidate_count),
                ),
                StageId::Ranking => {
                    if stage.enabled && ranking_applied {
                        (
                            "executed",
                            format!("ranker={} candidates={candidate_count}", pipeline.ranker),
                        )
                    } else if stage.enabled {
                        (
                            "deferred",
                            "ranking is enabled in the plan, but the configured stage order keeps it out of the active execution path".to_string(),
                        )
                    } else {
                        (
                            "skipped",
                            "ranking disabled; candidate order remained collection order".to_string(),
                        )
                    }
                }
                StageId::RiskChecks => {
                    if stage.enabled {
                        (
                            "executed",
                            "paper cycle kept kernel risk checks enabled".to_string(),
                        )
                    } else {
                        (
                            "deferred",
                            "paper cycle still keeps kernel risk semantics active unless execution is fully disabled".to_string(),
                        )
                    }
                }
                StageId::IntentGeneration | StageId::OmsTransition | StageId::BrokerExecution | StageId::FillReconciliation => {
                    match stage.id {
                        StageId::IntentGeneration | StageId::OmsTransition => {
                            if stage.enabled && state_progression_enabled {
                                (
                                    "executed",
                                    format!("executed_entries={executed_entry_count}"),
                                )
                            } else {
                                (
                                    "skipped",
                                    "intent/OMS progression disabled; cycle remained preview-only".to_string(),
                                )
                            }
                        }
                        StageId::BrokerExecution | StageId::FillReconciliation => {
                            if stage.enabled && projection_enabled {
                                (
                                    "executed",
                                    format!("executed_entries={executed_entry_count}"),
                                )
                            } else {
                                (
                                    "skipped",
                                    "trade projection disabled; cycle avoided broker/fill writes".to_string(),
                                )
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                StageId::PersistenceAudit => {
                    if persistence_enabled {
                        (
                            "executed",
                            "persisted paper projections and runtime_cycle_steps".to_string(),
                        )
                    } else {
                        (
                            "skipped",
                            "persistence disabled by profile or dry-run".to_string(),
                        )
                    }
                }
            };

            PaperCycleStageTrace {
                stage: stage.id,
                enabled: stage.enabled,
                status: status.to_string(),
                detail,
            }
        })
        .collect()
}

pub(crate) fn derive_step_id(
    config_fingerprint: &str,
    interval: &str,
    step_close_ts_ms: i64,
    live: bool,
) -> String {
    format!(
        "paper_cycle:{}:{}:{}:{}",
        interval,
        step_close_ts_ms,
        if live { "live" } else { "paper" },
        config_fingerprint,
    )
}

fn decision_consumes_entry_budget(decision: &decision_kernel::DecisionResult) -> bool {
    decision
        .intents
        .iter()
        .any(|intent| matches!(intent.kind, OrderIntentKind::Open | OrderIntentKind::Add))
}

fn ensure_cycle_tables(tx: &rusqlite::Transaction<'_>) -> Result<()> {
    tx.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS runtime_cycle_steps (
            step_id TEXT PRIMARY KEY,
            step_close_ts_ms INTEGER NOT NULL,
            interval TEXT NOT NULL,
            symbols_json TEXT NOT NULL,
            snapshot_exported_at_ms INTEGER NOT NULL,
            execution_count INTEGER NOT NULL,
            trades_written INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        "#,
    )?;
    Ok(())
}

fn cycle_step_exists(tx: &rusqlite::Transaction<'_>, step_id: &str) -> Result<bool> {
    let row = tx
        .query_row(
            "SELECT 1 FROM runtime_cycle_steps WHERE step_id = ?1 LIMIT 1",
            [step_id],
            |_| Ok(()),
        )
        .optional()?;
    Ok(row.is_some())
}

struct CycleStepRecord<'a> {
    step_id: &'a str,
    step_close_ts_ms: i64,
    interval: &'a str,
    active_symbols: &'a [String],
    snapshot_exported_at_ms: i64,
    execution_count: usize,
    trades_written: usize,
}

fn record_cycle_step(tx: &rusqlite::Transaction<'_>, record: &CycleStepRecord<'_>) -> Result<()> {
    tx.execute(
        "INSERT INTO runtime_cycle_steps (step_id, step_close_ts_ms, interval, symbols_json, snapshot_exported_at_ms, execution_count, trades_written, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            record.step_id,
            record.step_close_ts_ms,
            record.interval,
            serde_json::to_string(record.active_symbols)?,
            record.snapshot_exported_at_ms,
            record.execution_count as i64,
            record.trades_written as i64,
            iso_from_ms(record.snapshot_exported_at_ms),
        ],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiq_runtime_core::runtime::{build_bootstrap, RuntimeMode};
    use bt_core::config::StrategyConfig;
    use rusqlite::Connection;
    use tempfile::tempdir;

    const FIXED_STEP_CLOSE_TS_MS: i64 = 1_773_424_200_000;
    const FIXED_EXPORTED_AT_MS: i64 = 1_772_676_900_000;

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
            CREATE TABLE runtime_last_closes (
                symbol TEXT PRIMARY KEY,
                close_ts_ms INTEGER NOT NULL,
                side TEXT NOT NULL,
                reason TEXT,
                updated_at TEXT NOT NULL
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

    fn seed_btc_paper_db(path: &Path) {
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
            CREATE TABLE runtime_last_closes (
                symbol TEXT PRIMARY KEY,
                close_ts_ms INTEGER NOT NULL,
                side TEXT NOT NULL,
                reason TEXT,
                updated_at TEXT NOT NULL
            );
            "#,
        )
        .unwrap();
        conn.execute(
            "INSERT INTO trades (timestamp,symbol,action,type,price,size,notional,reason,confidence,balance,pnl,fee_usd,fee_rate,entry_atr,leverage,margin_used,meta_json)
             VALUES ('2026-03-05T10:00:00+00:00','BTC','OPEN','LONG',50000.0,0.02,1000.0,'seed','medium',1000.0,0.0,0.0,0.0,1000.0,3.0,333.3,'{}')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO position_state VALUES ('BTC',1,49000.0,1772676500000,0,0,0,22.0,'2026-03-05T10:08:20+00:00')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO runtime_cooldowns VALUES ('BTC',1772676500.0,1772676550.0,'2026-03-05T10:15:00+00:00')",
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

    fn load_cfg(symbol: Option<&str>) -> StrategyConfig {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        bt_core::config::load_config_checked(
            repo_root
                .join("config/strategy_overrides.yaml.example")
                .to_str()
                .unwrap(),
            symbol,
            false,
        )
        .unwrap()
    }

    fn config_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("config/strategy_overrides.yaml.example")
    }

    fn effective_config(path: &Path) -> PaperEffectiveConfig {
        PaperEffectiveConfig::resolve(Some(path), None, None).unwrap()
    }

    #[test]
    fn cycle_rejects_duplicate_step_replay() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_paper_db(&paper_db);
        seed_candles_db(&candles_db);

        let base_cfg = load_cfg(None);
        let runtime_bootstrap = build_bootstrap(&base_cfg, RuntimeMode::Paper, None).unwrap();
        let cfg_path = config_path();
        let input = PaperCycleInput {
            effective_config: effective_config(&cfg_path),
            runtime_bootstrap,
            live: false,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["ETH".to_string()],
            btc_symbol: "BTC",
            lookback_bars: 400,
            step_close_ts_ms: FIXED_STEP_CLOSE_TS_MS,
            exported_at_ms: Some(FIXED_EXPORTED_AT_MS),
            dry_run: false,
        };

        let first = run_cycle(input).unwrap();
        assert!(first.runtime_step_recorded);
        assert!(first.trades_written > 0);
        let conn = Connection::open(&paper_db).unwrap();
        let before_counts = (
            conn.query_row("SELECT COUNT(*) FROM trades", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
            conn.query_row("SELECT COUNT(*) FROM runtime_cycle_steps", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        );
        conn.close().unwrap();

        let runtime_bootstrap = build_bootstrap(&base_cfg, RuntimeMode::Paper, None).unwrap();
        let cfg_path = config_path();
        let err = run_cycle(PaperCycleInput {
            effective_config: effective_config(&cfg_path),
            runtime_bootstrap,
            live: false,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["ETH".to_string(), "BTC".to_string()],
            btc_symbol: "BTC",
            lookback_bars: 400,
            step_close_ts_ms: FIXED_STEP_CLOSE_TS_MS,
            exported_at_ms: Some(FIXED_EXPORTED_AT_MS),
            dry_run: false,
        })
        .unwrap_err();
        assert!(err.to_string().contains("already applied"));
        let conn = Connection::open(&paper_db).unwrap();
        let after_counts = (
            conn.query_row("SELECT COUNT(*) FROM trades", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
            conn.query_row("SELECT COUNT(*) FROM runtime_cycle_steps", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        );
        assert_eq!(before_counts, after_counts);
    }

    #[test]
    fn cycle_persists_step_time_not_exported_time() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_paper_db(&paper_db);
        seed_candles_db(&candles_db);

        let base_cfg = load_cfg(None);
        let runtime_bootstrap = build_bootstrap(&base_cfg, RuntimeMode::Paper, None).unwrap();
        let cfg_path = config_path();
        let report = run_cycle(PaperCycleInput {
            effective_config: effective_config(&cfg_path),
            runtime_bootstrap,
            live: false,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["ETH".to_string()],
            btc_symbol: "BTC",
            lookback_bars: 400,
            step_close_ts_ms: FIXED_STEP_CLOSE_TS_MS,
            exported_at_ms: Some(FIXED_EXPORTED_AT_MS),
            dry_run: false,
        })
        .unwrap();

        assert_eq!(report.step_close_ts_ms, FIXED_STEP_CLOSE_TS_MS);
        let conn = Connection::open(&paper_db).unwrap();
        let latest_trade_ts: String = conn
            .query_row(
                "SELECT timestamp FROM trades WHERE symbol = 'ETH' ORDER BY id DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(latest_trade_ts, iso_from_ms(FIXED_STEP_CLOSE_TS_MS));
        let last_exit_attempt_s: Option<f64> = conn
            .query_row(
                "SELECT last_exit_attempt_s FROM runtime_cooldowns WHERE symbol = 'ETH'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            last_exit_attempt_s,
            Some((FIXED_STEP_CLOSE_TS_MS as f64) / 1000.0)
        );
    }

    #[test]
    fn cycle_dry_run_keeps_db_unchanged() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_paper_db(&paper_db);
        seed_candles_db(&candles_db);

        let conn = Connection::open(&paper_db).unwrap();
        let before_trades: i64 = conn
            .query_row("SELECT COUNT(*) FROM trades", [], |row| row.get(0))
            .unwrap();
        conn.close().unwrap();

        let base_cfg = load_cfg(None);
        let runtime_bootstrap = build_bootstrap(&base_cfg, RuntimeMode::Paper, None).unwrap();
        let cfg_path = config_path();
        let report = run_cycle(PaperCycleInput {
            effective_config: effective_config(&cfg_path),
            runtime_bootstrap,
            live: false,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["ETH".to_string()],
            btc_symbol: "BTC",
            lookback_bars: 400,
            step_close_ts_ms: FIXED_STEP_CLOSE_TS_MS,
            exported_at_ms: Some(FIXED_EXPORTED_AT_MS),
            dry_run: true,
        })
        .unwrap();

        assert!(report.dry_run);
        assert_eq!(report.trades_written, 0);
        let conn = Connection::open(&paper_db).unwrap();
        let after_trades: i64 = conn
            .query_row("SELECT COUNT(*) FROM trades", [], |row| row.get(0))
            .unwrap();
        let step_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'runtime_cycle_steps'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(before_trades, after_trades);
        assert_eq!(step_count, 0);
    }

    #[test]
    fn cycle_includes_open_btc_positions_even_when_btc_is_anchor() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_btc_paper_db(&paper_db);
        seed_candles_db(&candles_db);

        let base_cfg = load_cfg(None);
        let runtime_bootstrap = build_bootstrap(&base_cfg, RuntimeMode::Paper, None).unwrap();
        let cfg_path = config_path();
        let report = run_cycle(PaperCycleInput {
            effective_config: effective_config(&cfg_path),
            runtime_bootstrap,
            live: false,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &[],
            btc_symbol: "BTC",
            lookback_bars: 400,
            step_close_ts_ms: FIXED_STEP_CLOSE_TS_MS,
            exported_at_ms: Some(FIXED_EXPORTED_AT_MS),
            dry_run: true,
        })
        .unwrap();

        assert!(report.active_symbols.iter().any(|symbol| symbol == "BTC"));
    }

    #[test]
    fn stage_debug_profile_runs_preview_only_and_skips_persistence() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_paper_db(&paper_db);
        seed_candles_db(&candles_db);

        let base_cfg = load_cfg(None);
        let runtime_bootstrap =
            build_bootstrap(&base_cfg, RuntimeMode::Paper, Some("stage_debug")).unwrap();
        let cfg_path = config_path();
        let report = run_cycle(PaperCycleInput {
            effective_config: effective_config(&cfg_path),
            runtime_bootstrap,
            live: false,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["ETH".to_string()],
            btc_symbol: "BTC",
            lookback_bars: 400,
            step_close_ts_ms: FIXED_STEP_CLOSE_TS_MS,
            exported_at_ms: Some(FIXED_EXPORTED_AT_MS),
            dry_run: false,
        })
        .unwrap();

        assert_eq!(report.runtime_bootstrap.pipeline.profile, "stage_debug");
        assert_eq!(report.trades_written, 0);
        assert!(!report.runtime_step_recorded);
        assert!(report
            .executions
            .iter()
            .all(|execution| execution.phase.ends_with("_preview")));
        assert!(report
            .pipeline_trace
            .iter()
            .any(|trace| trace.stage == StageId::Ranking && trace.status == "skipped"));
        assert!(report.pipeline_trace.iter().any(|trace| {
            trace.stage == StageId::PersistenceAudit && trace.status == "skipped"
        }));
    }

    #[test]
    fn sort_entry_candidates_honours_configured_ranker() {
        let dir = tempdir().unwrap();
        let candles_db = dir.path().join("candles.db");
        seed_candles_db(&candles_db);

        let btc = EntryCandidate {
            prepared: prepare_symbol_step(
                &load_cfg(Some("BTC")),
                None,
                &candles_db,
                "BTC",
                "BTC",
                400,
                Some(FIXED_STEP_CLOSE_TS_MS),
            )
            .unwrap(),
            score: 10.0,
            requested_notional_usd: 50.0,
        };
        let eth = EntryCandidate {
            prepared: prepare_symbol_step(
                &load_cfg(Some("ETH")),
                None,
                &candles_db,
                "ETH",
                "BTC",
                400,
                Some(FIXED_STEP_CLOSE_TS_MS),
            )
            .unwrap(),
            score: 20.0,
            requested_notional_usd: 25.0,
        };

        let mut by_score = vec![btc.clone(), eth.clone()];
        sort_entry_candidates("confidence_adx", &mut by_score);
        assert_eq!(by_score[0].prepared.symbol, "ETH");

        let mut by_notional = vec![btc, eth];
        sort_entry_candidates("raw_notional_desc", &mut by_notional);
        assert_eq!(by_notional[0].prepared.symbol, "BTC");
    }

    #[test]
    fn parity_baseline_keeps_intent_and_oms_stages_active() {
        let dir = tempdir().unwrap();
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_paper_db(&paper_db);
        seed_candles_db(&candles_db);

        let base_cfg = load_cfg(None);
        let runtime_bootstrap =
            build_bootstrap(&base_cfg, RuntimeMode::Paper, Some("parity_baseline")).unwrap();
        let cfg_path = config_path();
        let report = run_cycle(PaperCycleInput {
            effective_config: effective_config(&cfg_path),
            runtime_bootstrap,
            live: false,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["ETH".to_string()],
            btc_symbol: "BTC",
            lookback_bars: 400,
            step_close_ts_ms: FIXED_STEP_CLOSE_TS_MS,
            exported_at_ms: Some(FIXED_EXPORTED_AT_MS),
            dry_run: false,
        })
        .unwrap();

        assert_eq!(report.runtime_bootstrap.pipeline.profile, "parity_baseline");
        assert!(report.runtime_step_recorded);
        assert_eq!(report.trades_written, 0);
        assert!(report.pipeline_trace.iter().any(|trace| {
            trace.stage == StageId::IntentGeneration && trace.status == "executed"
        }));
        assert!(report
            .pipeline_trace
            .iter()
            .any(|trace| { trace.stage == StageId::OmsTransition && trace.status == "executed" }));
        assert!(report
            .pipeline_trace
            .iter()
            .any(|trace| { trace.stage == StageId::BrokerExecution && trace.status == "skipped" }));
    }

    #[test]
    fn custom_stage_order_can_defer_ranking_in_execution() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let paper_db = dir.path().join("paper.db");
        let candles_db = dir.path().join("candles.db");
        seed_paper_db(&paper_db);
        seed_candles_db(&candles_db);
        std::fs::write(
            &config_path,
            r#"
global:
  engine:
    interval: 30m
  runtime:
    profile: late_ranker
  pipeline:
    profiles:
      late_ranker:
        stage_order:
          - market_data_normalisation
          - indicator_build
          - gate_evaluation
          - signal_generation
          - intent_generation
          - ranking
          - risk_checks
          - oms_transition
          - broker_execution
          - fill_reconciliation
          - persistence_audit
"#,
        )
        .unwrap();

        let effective = PaperEffectiveConfig::resolve(Some(&config_path), None, None).unwrap();
        let base_cfg = effective.load_config(None, false).unwrap();
        let runtime_bootstrap =
            build_bootstrap(&base_cfg, RuntimeMode::Paper, Some("late_ranker")).unwrap();
        let report = run_cycle(PaperCycleInput {
            effective_config: effective,
            runtime_bootstrap,
            live: false,
            paper_db: &paper_db,
            candles_db: &candles_db,
            explicit_symbols: &["BTC".to_string()],
            btc_symbol: "BTC",
            lookback_bars: 400,
            step_close_ts_ms: FIXED_STEP_CLOSE_TS_MS,
            exported_at_ms: Some(FIXED_EXPORTED_AT_MS),
            dry_run: true,
        })
        .unwrap();

        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("orders ranking after intent_generation")));
        assert!(report
            .pipeline_trace
            .iter()
            .any(|trace| { trace.stage == StageId::Ranking && trace.status == "deferred" }));
    }
}
