use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::json;

use crate::config::HubConfig;

pub const FACTORY_ENABLE_ENV: &str = "AI_QUANT_FACTORY_ENABLE";
pub const FACTORY_SETTINGS_PATH: &str = "config/factory_defaults.yaml";
pub const FACTORY_SERVICE_UNITS: [&str; 2] = [
    "openclaw-ai-quant-factory-v8",
    "openclaw-ai-quant-factory-v8-deep",
];
const FACTORY_EXECUTION_WIRED: bool = false;

#[derive(Debug, Clone, Serialize)]
pub struct FactoryCapability {
    pub compiled: bool,
    pub policy_enabled: bool,
    pub executor_wired: bool,
    pub execution_enabled: bool,
    pub mode: &'static str,
    pub reason: String,
    pub enable_env: &'static str,
    pub settings_path: &'static str,
    pub service_units: [&'static str; 2],
}

impl FactoryCapability {
    pub fn current(config: &HubConfig) -> Self {
        let compiled = cfg!(feature = "factory");
        let policy_enabled = config.factory_enabled;
        let executor_wired = FACTORY_EXECUTION_WIRED;
        let execution_enabled = compiled && policy_enabled && executor_wired;
        let reason = match (compiled, policy_enabled, executor_wired) {
            (true, true, true) => {
                "Factory execution is explicitly enabled for this Hub instance.".to_string()
            }
            (true, true, false) => format!(
                "Factory execution was requested, but this Hub build still exposes only the dormant contract. A future build must wire the executor before {}=1 can activate it.",
                FACTORY_ENABLE_ENV
            ),
            (true, false, _) => format!(
                "Factory execution is compiled in, but policy keeps it dormant until {}=1.",
                FACTORY_ENABLE_ENV
            ),
            (false, true, _) => {
                "Factory policy is enabled, but this Hub build was compiled without the `factory` feature."
                    .to_string()
            }
            (false, false, _) => format!(
                "Factory execution is dormant by default. Re-enable it only with a build that includes the `factory` feature and {}=1.",
                FACTORY_ENABLE_ENV
            ),
        };

        Self {
            compiled,
            policy_enabled,
            executor_wired,
            execution_enabled,
            mode: if execution_enabled {
                "enabled"
            } else {
                "dormant"
            },
            reason,
            enable_env: FACTORY_ENABLE_ENV,
            settings_path: FACTORY_SETTINGS_PATH,
            service_units: FACTORY_SERVICE_UNITS,
        }
    }
}

pub fn disabled_response(capability: &FactoryCapability, action: &str) -> Response {
    (
        StatusCode::FORBIDDEN,
        axum::Json(json!({
            "error": "factory_disabled",
            "action": action,
            "capability": capability,
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_factory_enabled(enabled: bool) -> HubConfig {
        let mut cfg = HubConfig::from_env();
        cfg.factory_enabled = enabled;
        cfg
    }

    #[test]
    fn dormant_when_policy_disabled() {
        let cap = FactoryCapability::current(&config_with_factory_enabled(false));
        assert!(!cap.execution_enabled);
        assert_eq!(cap.mode, "dormant");
        assert!(cap.reason.contains(FACTORY_ENABLE_ENV));
    }

    #[test]
    fn reports_missing_feature_when_policy_on() {
        let cap = FactoryCapability::current(&config_with_factory_enabled(true));
        if !cfg!(feature = "factory") {
            assert!(!cap.execution_enabled);
            assert!(cap
                .reason
                .contains("compiled without the `factory` feature"));
        }
    }

    #[test]
    fn remains_dormant_when_executor_is_not_wired() {
        let cap = FactoryCapability::current(&config_with_factory_enabled(true));
        if cfg!(feature = "factory") {
            assert!(!cap.execution_enabled);
            assert!(!cap.executor_wired);
            assert!(cap.reason.contains("dormant contract"));
        }
    }
}
