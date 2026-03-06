use aiq_runtime_core::runtime::{build_bootstrap, RuntimeBootstrap, RuntimeMode};
use anyhow::{Context, Result};
use bt_core::config::{self, StrategyConfig};
use std::env;
use std::path::{Path, PathBuf};

const DEFAULT_CONFIG_PATH: &str = "config/strategy_overrides.yaml";
const VALID_PROMOTED_ROLES: &[&str] = &["primary", "fallback", "conservative"];
const DEFAULT_PROMOTED_SCAN_DATE_DIRS: usize = 90;
const DEFAULT_PROMOTED_SCAN_RUN_DIRS_PER_DATE: usize = 200;
const MIN_PROMOTED_SCAN_LIMIT: usize = 1;
const MAX_PROMOTED_SCAN_LIMIT: usize = 10_000;

#[derive(Debug, Clone, PartialEq)]
pub struct PaperEffectiveConfig {
    config_path: PathBuf,
    document: serde_yaml::Value,
    promoted_role: Option<String>,
    promoted_config_path: Option<PathBuf>,
    strategy_mode: Option<String>,
    strategy_mode_source: Option<String>,
    warnings: Vec<String>,
}

impl PaperEffectiveConfig {
    pub fn resolve(config: Option<&Path>) -> Result<Self> {
        let config_path = resolve_config_path(config);
        let mut warnings = Vec::new();
        let mut document = config::load_yaml_document_checked(
            config_path
                .to_str()
                .context("config path must be valid UTF-8")?,
        )
        .map_err(anyhow::Error::msg)?;

        let promoted_role =
            env_string("AI_QUANT_PROMOTED_ROLE").map(|role| role.to_ascii_lowercase());
        let mut promoted_config_path = None;
        if let Some(role) = promoted_role.as_deref() {
            if !VALID_PROMOTED_ROLES.contains(&role) {
                warnings.push(format!(
                    "unsupported AI_QUANT_PROMOTED_ROLE={}, ignoring promoted config overlay",
                    role
                ));
            } else {
                let artifacts_dir = env_path("AI_QUANT_ARTIFACTS_DIR")
                    .unwrap_or_else(|| project_root().join("artifacts"));
                match find_latest_promoted_config(&artifacts_dir, role) {
                    Some(promoted_path) => {
                        let promoted_document = config::load_yaml_document_checked(
                            promoted_path
                                .to_str()
                                .context("promoted config path must be valid UTF-8")?,
                        )
                        .map_err(anyhow::Error::msg)?;
                        config::deep_merge_yaml_value(&mut document, &promoted_document);
                        promoted_config_path = Some(promoted_path);
                    }
                    None => warnings.push(format!(
                        "no promoted config found for role={} under {}",
                        role,
                        artifacts_dir.display()
                    )),
                }
            }
        }

        let (strategy_mode, strategy_mode_source, strategy_mode_warning) = resolve_strategy_mode();
        if let Some(strategy_mode_warning) = strategy_mode_warning {
            warnings.push(strategy_mode_warning);
        }

        Ok(Self {
            config_path,
            document,
            promoted_role,
            promoted_config_path,
            strategy_mode,
            strategy_mode_source,
            warnings,
        })
    }

    pub fn config_path(&self) -> &Path {
        &self.config_path
    }

    pub fn promoted_role(&self) -> Option<&str> {
        self.promoted_role.as_deref()
    }

    pub fn promoted_config_path(&self) -> Option<&Path> {
        self.promoted_config_path.as_deref()
    }

    pub fn strategy_mode(&self) -> Option<&str> {
        self.strategy_mode.as_deref()
    }

    pub fn strategy_mode_source(&self) -> Option<&str> {
        self.strategy_mode_source.as_deref()
    }

    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    pub fn load_config(&self, symbol: Option<&str>, live: bool) -> Result<StrategyConfig> {
        config::load_config_document_checked(
            &self.document,
            symbol,
            live,
            self.strategy_mode.as_deref(),
        )
        .map_err(anyhow::Error::msg)
    }

    pub fn build_runtime_bootstrap(
        &self,
        live: bool,
        profile_override: Option<&str>,
    ) -> Result<RuntimeBootstrap> {
        let config = self.load_config(None, live)?;
        build_bootstrap(&config, RuntimeMode::Paper, profile_override).map_err(anyhow::Error::msg)
    }

    pub fn resolve_shared_interval(&self, active_symbols: &[String], live: bool) -> Result<String> {
        let mut interval = None;
        for symbol in active_symbols {
            let config = self.load_config(Some(symbol), live)?;
            match interval.as_deref() {
                Some(current_interval) if current_interval != config.engine.interval => {
                    anyhow::bail!(
                        "paper loop requires a shared interval; {} resolved to {} but prior symbols use {}",
                        symbol,
                        config.engine.interval,
                        current_interval
                    );
                }
                None => interval = Some(config.engine.interval.clone()),
                _ => {}
            }
        }
        interval.context("paper loop requires at least one active symbol interval")
    }
}

pub(crate) fn resolve_config_path(config: Option<&Path>) -> PathBuf {
    let configured = config
        .map(Path::to_path_buf)
        .or_else(|| env_path("AI_QUANT_STRATEGY_YAML"))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG_PATH));

    if configured.exists() {
        return configured;
    }

    let fallback = PathBuf::from(format!("{}.example", configured.display()));
    if fallback.exists() {
        return fallback;
    }

    configured
}

fn project_root() -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    path.canonicalize().unwrap_or(path)
}

fn resolve_strategy_mode() -> (Option<String>, Option<String>, Option<String>) {
    if let Some(strategy_mode) = env_string("AI_QUANT_STRATEGY_MODE") {
        return (
            config::normalise_strategy_mode_key(&strategy_mode),
            Some("env".to_string()),
            None,
        );
    }

    let mode_file = env_path("AI_QUANT_STRATEGY_MODE_FILE")
        .unwrap_or_else(|| project_root().join("artifacts/state/strategy_mode.txt"));
    if !mode_file.exists() {
        return (None, None, None);
    }

    match std::fs::read_to_string(&mode_file) {
        Ok(contents) => (
            config::normalise_strategy_mode_key(contents.trim()),
            Some("file".to_string()),
            None,
        ),
        Err(err) => (
            None,
            None,
            Some(format!(
                "failed to read strategy mode file {}: {}",
                mode_file.display(),
                err
            )),
        ),
    }
}

fn env_string(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_path(name: &str) -> Option<PathBuf> {
    env_string(name).map(PathBuf::from)
}

fn promoted_scan_limit(env_name: &str, default: usize) -> usize {
    let Some(raw) = env_string(env_name) else {
        return default;
    };
    let Ok(value) = raw.parse::<usize>() else {
        return default;
    };
    value.clamp(MIN_PROMOTED_SCAN_LIMIT, MAX_PROMOTED_SCAN_LIMIT)
}

fn find_latest_promoted_config(artifacts_dir: &Path, role: &str) -> Option<PathBuf> {
    if !artifacts_dir.is_dir() {
        return None;
    }

    let filename = format!("{role}.yaml");
    let max_dates = promoted_scan_limit(
        "AI_QUANT_PROMOTED_SCAN_DATE_DIRS",
        DEFAULT_PROMOTED_SCAN_DATE_DIRS,
    );
    let max_runs = promoted_scan_limit(
        "AI_QUANT_PROMOTED_SCAN_RUN_DIRS_PER_DATE",
        DEFAULT_PROMOTED_SCAN_RUN_DIRS_PER_DATE,
    );

    let mut date_dirs = std::fs::read_dir(artifacts_dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(is_date_dir_name)
        })
        .collect::<Vec<_>>();
    date_dirs.sort();
    date_dirs.reverse();

    scan_promoted_dirs(&date_dirs, &filename, max_dates, max_runs)
        .or_else(|| scan_promoted_dirs(&date_dirs, &filename, usize::MAX, usize::MAX))
}

fn scan_promoted_dirs(
    date_dirs: &[PathBuf],
    filename: &str,
    max_dates: usize,
    max_runs: usize,
) -> Option<PathBuf> {
    for date_dir in date_dirs.iter().take(max_dates) {
        let mut run_dirs = std::fs::read_dir(date_dir)
            .ok()?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("run_"))
            })
            .collect::<Vec<_>>();
        run_dirs.sort();
        run_dirs.reverse();

        for run_dir in run_dirs.iter().take(max_runs) {
            let candidate = run_dir.join("promoted_configs").join(filename);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn is_date_dir_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(idx, byte)| matches!(idx, 4 | 7) || byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    use crate::test_support::env_lock;

    struct EnvGuard {
        saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
    }

    impl EnvGuard {
        fn set(vars: &[(&'static str, Option<&str>)]) -> Self {
            let mut saved = Vec::new();
            for (name, value) in vars {
                saved.push((*name, env::var_os(name)));
                match value {
                    Some(value) => env::set_var(name, value),
                    None => env::remove_var(name),
                }
            }
            Self { saved }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in self.saved.drain(..) {
                match value {
                    Some(value) => env::set_var(name, value),
                    None => env::remove_var(name),
                }
            }
        }
    }
    #[test]
    fn effective_config_applies_promoted_overlay_and_mode_file() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let mode_file = dir.path().join("strategy_mode.txt");
        let artifacts_dir = dir.path().join("artifacts");
        let promoted_dir = artifacts_dir.join("2026-03-06/run_nightly/promoted_configs");
        fs::create_dir_all(&promoted_dir).unwrap();
        fs::write(
            &config_path,
            r#"
global:
  engine:
    interval: 30m
  trade:
    leverage: 2.0
modes:
  fallback:
    global:
      engine:
        interval: 1h
"#,
        )
        .unwrap();
        fs::write(
            promoted_dir.join("primary.yaml"),
            "global:\n  trade:\n    leverage: 7.5\n",
        )
        .unwrap();
        fs::write(&mode_file, "fallback\n").unwrap();

        let _env = EnvGuard::set(&[
            ("AI_QUANT_PROMOTED_ROLE", Some("primary")),
            (
                "AI_QUANT_ARTIFACTS_DIR",
                Some(artifacts_dir.to_str().unwrap()),
            ),
            ("AI_QUANT_STRATEGY_MODE", None),
            (
                "AI_QUANT_STRATEGY_MODE_FILE",
                Some(mode_file.to_str().unwrap()),
            ),
        ]);

        let effective = PaperEffectiveConfig::resolve(Some(&config_path)).unwrap();
        let cfg = effective.load_config(None, false).unwrap();

        assert_eq!(effective.promoted_role(), Some("primary"));
        assert_eq!(effective.strategy_mode(), Some("fallback"));
        assert_eq!(effective.strategy_mode_source(), Some("file"));
        assert_eq!(cfg.engine.interval, "1h");
        assert!((cfg.trade.leverage - 7.5).abs() < f64::EPSILON);
        assert!(effective.promoted_config_path().is_some());
    }

    #[test]
    fn effective_config_prefers_strategy_mode_env_over_file() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let mode_file = dir.path().join("strategy_mode.txt");
        fs::write(
            &config_path,
            "global:\n  engine:\n    interval: 30m\nmodes:\n  fallback:\n    global:\n      engine:\n        interval: 1h\n",
        )
        .unwrap();
        fs::write(&mode_file, "fallback\n").unwrap();

        let _env = EnvGuard::set(&[
            ("AI_QUANT_PROMOTED_ROLE", None),
            ("AI_QUANT_STRATEGY_MODE", Some("primary")),
            (
                "AI_QUANT_STRATEGY_MODE_FILE",
                Some(mode_file.to_str().unwrap()),
            ),
        ]);

        let effective = PaperEffectiveConfig::resolve(Some(&config_path)).unwrap();
        assert_eq!(effective.strategy_mode(), Some("primary"));
        assert_eq!(effective.strategy_mode_source(), Some("env"));
    }
}
