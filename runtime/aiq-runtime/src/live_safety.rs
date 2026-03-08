use std::env;

pub const LIVE_CONFIRM_VALUE: &str = "I_UNDERSTAND_THIS_CAN_LOSE_MONEY";

fn env_bool(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "y" | "on"
            )
        })
        .unwrap_or(false)
}

fn env_string(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub fn live_mode() -> String {
    env_string("AI_QUANT_MODE")
        .unwrap_or_else(|| "paper".to_string())
        .to_ascii_lowercase()
}

pub fn live_orders_enabled() -> bool {
    if env_bool("AI_QUANT_HARD_KILL_SWITCH") {
        return false;
    }
    env_bool("AI_QUANT_LIVE_ENABLE")
        && env_string("AI_QUANT_LIVE_CONFIRM")
            .map(|value| value == LIVE_CONFIRM_VALUE)
            .unwrap_or(false)
}

pub fn live_entries_enabled() -> bool {
    live_orders_enabled() && !env_bool("AI_QUANT_KILL_SWITCH")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{env_lock, EnvGuard};

    #[test]
    fn live_orders_require_enable_and_confirm() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let _guards = EnvGuard::set(&[
            ("AI_QUANT_LIVE_ENABLE", None),
            ("AI_QUANT_LIVE_CONFIRM", None),
            ("AI_QUANT_HARD_KILL_SWITCH", None),
        ]);
        assert!(!live_orders_enabled());

        let _enable = EnvGuard::set(&[("AI_QUANT_LIVE_ENABLE", Some("1"))]);
        assert!(!live_orders_enabled());

        let _confirm = EnvGuard::set(&[("AI_QUANT_LIVE_CONFIRM", Some(LIVE_CONFIRM_VALUE))]);
        assert!(live_orders_enabled());

        let _hard_kill = EnvGuard::set(&[("AI_QUANT_HARD_KILL_SWITCH", Some("1"))]);
        assert!(!live_orders_enabled());
    }

    #[test]
    fn live_entries_honour_soft_kill_switch() {
        let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
        let _guards = EnvGuard::set(&[
            ("AI_QUANT_LIVE_ENABLE", Some("1")),
            ("AI_QUANT_LIVE_CONFIRM", Some(LIVE_CONFIRM_VALUE)),
            ("AI_QUANT_KILL_SWITCH", None),
            ("AI_QUANT_HARD_KILL_SWITCH", None),
        ]);
        assert!(live_entries_enabled());

        let _soft_kill = EnvGuard::set(&[("AI_QUANT_KILL_SWITCH", Some("1"))]);
        assert!(!live_entries_enabled());
        assert!(live_orders_enabled());
    }
}
