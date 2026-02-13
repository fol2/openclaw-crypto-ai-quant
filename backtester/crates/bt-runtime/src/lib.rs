//! Python binding entrypoint for the Rust decision kernel.
//!
//! The module exposes a small JSON envelope API so Python callers can execute the
//! canonical `decision_kernel::step` transition without mirroring Rust structs.
//! The response is always JSON, which keeps the interface language-agnostic and
//! easy to persist in test artefacts.

use pyo3::prelude::*;
use serde::Serialize;
use serde_json::{self, Value, json};

use bt_core::decision_kernel;

const ERROR_OK_PREFIX: &str = "bt-runtime:";

#[derive(Serialize)]
struct ErrorObject {
    code: String,
    message: String,
    details: Vec<String>,
}

#[derive(Serialize)]
struct ErrorEnvelope {
    ok: bool,
    error: ErrorObject,
}

#[derive(Serialize)]
struct StepEnvelope {
    ok: bool,
    decision: decision_kernel::DecisionResult,
}

fn validation_error(code: &str, message: &str, details: Vec<String>) -> String {
    serde_json::to_string(&ErrorEnvelope {
        ok: false,
        error: ErrorObject {
            code: code.to_string(),
            message: message.to_string(),
            details,
        },
    })
    .unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": {
                "code": code,
                "message": message,
                "details": details,
            }
        })
        .to_string()
    })
}

fn as_u32(payload: &str, label: &str) -> PyResult<u32> {
    let payload: Value = serde_json::from_str(payload).map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{ERROR_OK_PREFIX} cannot parse {label} JSON payload: {err}"
        ))
    })?;
    payload
        .as_object()
        .and_then(|obj| obj.get("schema_version"))
        .and_then(|value| value.as_u64())
        .map(|v| v as u32)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "{ERROR_OK_PREFIX} missing schema_version in {label}"
            ))
        })
}

fn ensure_matching_schema_version(
    state_version: u32,
    event_version: u32,
    params_version: u32,
    expected_version: u32,
) -> Option<Vec<String>> {
    if state_version == expected_version
        && event_version == expected_version
        && params_version == expected_version
    {
        return None;
    }

    let mut errors = Vec::with_capacity(3);
    if state_version != expected_version {
        errors.push(format!("state schema_version={state_version}, expected={expected_version}"));
    }
    if event_version != expected_version {
        errors.push(format!("event schema_version={event_version}, expected={expected_version}"));
    }
    if params_version != expected_version {
        errors.push(format!(
            "params schema_version={params_version}, expected={expected_version}"
        ));
    }
    Some(errors)
}

fn default_schema_version() -> u32 {
    decision_kernel::StrategyState::new(0.0, 0).schema_version
}

fn step_envelope(decision: decision_kernel::DecisionResult) -> String {
    serde_json::to_string(&StepEnvelope { ok: true, decision }).unwrap_or_else(|err| {
        validation_error(
            "SERIALIZATION_FAILED",
            "Failed to serialise kernel decision envelope",
            vec![err.to_string()],
        )
    })
}

#[pyfunction]
fn step_decision(state_json: &str, event_json: &str, params_json: &str) -> PyResult<String> {
    let state_version = match as_u32(state_json, "state") {
        Ok(v) => v,
        Err(err) => {
            return Ok(validation_error(
                "INVALID_JSON",
                "Failed to parse state payload",
                vec![err.to_string()],
            ));
        }
    };

    let event_version = match as_u32(event_json, "event") {
        Ok(v) => v,
        Err(err) => {
            return Ok(validation_error(
                "INVALID_JSON",
                "Failed to parse event payload",
                vec![err.to_string()],
            ));
        }
    };

    let params_version = match as_u32(params_json, "params") {
        Ok(v) => v,
        Err(err) => {
            return Ok(validation_error(
                "INVALID_JSON",
                "Failed to parse params payload",
                vec![err.to_string()],
            ));
        }
    };

    let expected = default_schema_version();
    if let Some(details) = ensure_matching_schema_version(state_version, event_version, params_version, expected)
    {
        return Ok(validation_error(
            "SCHEMA_VERSION_MISMATCH",
            "Schema version mismatch",
            details,
        ));
    }

    let state_result = serde_json::from_str::<decision_kernel::StrategyState>(state_json);
    let state = match state_result {
        Ok(value) => value,
        Err(err) => {
            return Ok(validation_error(
                "INVALID_JSON",
                "Failed to parse StrategyState",
                vec![err.to_string()],
            ));
        }
    };

    let event_result = serde_json::from_str::<decision_kernel::MarketEvent>(event_json);
    let event = match event_result {
        Ok(value) => value,
        Err(err) => {
            return Ok(validation_error(
                "INVALID_JSON",
                "Failed to parse MarketEvent",
                vec![err.to_string()],
            ));
        }
    };

    let params_result = serde_json::from_str::<decision_kernel::KernelParams>(params_json);
    let params = match params_result {
        Ok(value) => value,
        Err(err) => {
            return Ok(validation_error(
                "INVALID_JSON",
                "Failed to parse KernelParams",
                vec![err.to_string()],
            ));
        }
    };

    let decision = decision_kernel::step(&state, &event, &params);
    if !decision.diagnostics.errors.is_empty() {
        return Ok(validation_error(
            "KERNEL_DECISION_REJECTED",
            "Kernel diagnostics reported errors",
            decision.diagnostics.errors.clone(),
        ));
    }

    Ok(step_envelope(decision))
}

#[pyfunction]
fn default_kernel_state_json(initial_cash_usd: f64, timestamp_ms: i64) -> PyResult<String> {
    let state = decision_kernel::StrategyState::new(initial_cash_usd, timestamp_ms);
    serde_json::to_string(&state).map_err(|err| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to serialise default state: {err}"
        ))
    })
}

#[pyfunction]
fn default_kernel_params_json() -> PyResult<String> {
    let params = decision_kernel::KernelParams::default();
    serde_json::to_string(&params).map_err(|err| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to serialise default params: {err}"
        ))
    })
}

#[pyfunction]
fn schema_version(payload_json: &str) -> PyResult<u32> {
    as_u32(payload_json, "payload")
}

#[pymodule]
fn bt_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(step_decision, m)?)?;
    m.add_function(wrap_pyfunction!(default_kernel_state_json, m)?)?;
    m.add_function(wrap_pyfunction!(default_kernel_params_json, m)?)?;
    m.add_function(wrap_pyfunction!(schema_version, m)?)?;
    Ok(())
}
