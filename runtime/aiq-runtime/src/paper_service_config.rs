use anyhow::{Context, Result};
use bt_core::config::StrategyConfig;
use std::env;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

const DEFAULT_CONFIG_PATH: &str = "config/strategy_overrides.yaml";
const DEFAULT_PROMOTED_SCAN_DATE_DIRS: usize = 90;
const DEFAULT_PROMOTED_SCAN_RUN_DIRS_PER_DATE: usize = 200;
const MIN_PROMOTED_SCAN_LIMIT: usize = 1;
const MAX_PROMOTED_SCAN_LIMIT: usize = 10_000;

pub struct ResolvedPaperServiceConfig {
    base_config_path: PathBuf,
    effective_config_path: PathBuf,
    promoted_config_path: Option<PathBuf>,
    promoted_role: Option<String>,
    strategy_mode: Option<String>,
    warnings: Vec<String>,
    _merged_config_file: Option<NamedTempFile>,
}

impl ResolvedPaperServiceConfig {
    pub fn base_config_path(&self) -> &Path {
        &self.base_config_path
    }

    pub fn effective_config_path(&self) -> &Path {
        &self.effective_config_path
    }

    pub fn promoted_config_path(&self) -> Option<&Path> {
        self.promoted_config_path.as_deref()
    }

    pub fn promoted_role(&self) -> Option<&str> {
        self.promoted_role.as_deref()
    }

    pub fn strategy_mode(&self) -> Option<&str> {
        self.strategy_mode.as_deref()
    }

    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    pub fn load_config(&self, symbol: Option<&str>, live: bool) -> Result<StrategyConfig> {
        bt_core::config::load_config_checked_with_mode(
            self.effective_config_path
                .to_str()
                .context("effective config path must be valid UTF-8")?,
            symbol,
            live,
            self.strategy_mode.as_deref(),
        )
        .map_err(anyhow::Error::msg)
    }
}

pub fn resolve_paper_service_config(config: Option<&Path>) -> Result<ResolvedPaperServiceConfig> {
    let base_config_path = resolve_config_path(config);
    let requested_promoted_role = env_string("AI_QUANT_PROMOTED_ROLE");
    let strategy_mode = resolve_strategy_mode();
    let mut warnings = Vec::new();

    let promoted_role = requested_promoted_role
        .as_deref()
        .map(str::trim)
        .filter(|role| !role.is_empty())
        .map(|role| role.to_ascii_lowercase());

    let promoted_config_path = if let Some(role) = promoted_role.as_deref() {
        if !matches!(role, "primary" | "fallback" | "conservative") {
            warnings.push(format!(
                "unsupported AI_QUANT_PROMOTED_ROLE={role}; using the base config"
            ));
            None
        } else {
            let artifacts_dir = env_path("AI_QUANT_ARTIFACTS_DIR")
                .unwrap_or_else(default_artifacts_dir)
                .canonicalize()
                .unwrap_or_else(|_| {
                    env_path("AI_QUANT_ARTIFACTS_DIR").unwrap_or_else(default_artifacts_dir)
                });
            match find_latest_promoted_config(&artifacts_dir, role)? {
                Some(path) => Some(path),
                None => {
                    warnings.push(format!(
                        "no promoted config found for role={role}; using the base config"
                    ));
                    None
                }
            }
        }
    } else {
        None
    };

    let (effective_config_path, merged_file) =
        materialise_effective_config(&base_config_path, promoted_config_path.as_deref())?;

    Ok(ResolvedPaperServiceConfig {
        base_config_path,
        effective_config_path,
        promoted_config_path,
        promoted_role,
        strategy_mode,
        warnings,
        _merged_config_file: merged_file,
    })
}

pub fn resolve_config_path(config: Option<&Path>) -> PathBuf {
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

fn resolve_strategy_mode() -> Option<String> {
    env_string("AI_QUANT_STRATEGY_MODE").or_else(|| {
        let path = env_path("AI_QUANT_STRATEGY_MODE_FILE")
            .unwrap_or_else(|| project_root().join("artifacts/state/strategy_mode.txt"));
        let raw = std::fs::read_to_string(path).ok()?;
        let value = raw.trim();
        (!value.is_empty()).then(|| value.to_string())
    })
}

fn default_artifacts_dir() -> PathBuf {
    project_root().join("artifacts")
}

fn materialise_effective_config(
    base_config_path: &Path,
    promoted_config_path: Option<&Path>,
) -> Result<(PathBuf, Option<NamedTempFile>)> {
    let Some(promoted_config_path) = promoted_config_path else {
        return Ok((base_config_path.to_path_buf(), None));
    };

    let base_root = read_yaml_value(base_config_path)?;
    let promoted_root = read_yaml_value(promoted_config_path)?;
    let mut merged_root = base_root;
    deep_merge(&mut merged_root, &promoted_root);

    let mut merged_file = NamedTempFile::new().context("failed to create merged config file")?;
    serde_yaml::to_writer(merged_file.as_file_mut(), &merged_root)
        .context("failed to write merged promoted config")?;
    Ok((merged_file.path().to_path_buf(), Some(merged_file)))
}

fn read_yaml_value(path: &Path) -> Result<serde_yaml::Value> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read YAML: {}", path.display()))?;
    serde_yaml::from_str(&raw).with_context(|| format!("failed to parse YAML: {}", path.display()))
}

fn deep_merge(base: &mut serde_yaml::Value, overlay: &serde_yaml::Value) {
    match (base, overlay) {
        (serde_yaml::Value::Mapping(base_map), serde_yaml::Value::Mapping(overlay_map)) => {
            for (key, overlay_val) in overlay_map {
                if let Some(base_val) = base_map.get_mut(key) {
                    deep_merge(base_val, overlay_val);
                } else {
                    base_map.insert(key.clone(), overlay_val.clone());
                }
            }
        }
        (base, overlay) => {
            if !overlay.is_null() {
                *base = overlay.clone();
            }
        }
    }
}

fn find_latest_promoted_config(artifacts_dir: &Path, role: &str) -> Result<Option<PathBuf>> {
    if !artifacts_dir.is_dir() {
        return Ok(None);
    }

    let max_dates = promoted_scan_limit(
        "AI_QUANT_PROMOTED_SCAN_DATE_DIRS",
        DEFAULT_PROMOTED_SCAN_DATE_DIRS,
    );
    let max_runs = promoted_scan_limit(
        "AI_QUANT_PROMOTED_SCAN_RUN_DIRS_PER_DATE",
        DEFAULT_PROMOTED_SCAN_RUN_DIRS_PER_DATE,
    );
    let filename = format!("{role}.yaml");

    let mut date_dirs = std::fs::read_dir(artifacts_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(is_date_dir)
        })
        .collect::<Vec<_>>();
    date_dirs.sort_by(|left, right| right.cmp(left));

    if let Some(found) = scan_promoted_dirs(
        &date_dirs[..date_dirs.len().min(max_dates)],
        &filename,
        max_runs,
    )? {
        return Ok(Some(found));
    }

    scan_promoted_dirs(&date_dirs, &filename, usize::MAX)
}

fn scan_promoted_dirs(
    date_dirs: &[PathBuf],
    filename: &str,
    max_runs: usize,
) -> Result<Option<PathBuf>> {
    for date_dir in date_dirs {
        let mut run_dirs = std::fs::read_dir(date_dir)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_dir()
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("run_"))
            })
            .collect::<Vec<_>>();
        run_dirs.sort_by(|left, right| right.cmp(left));

        for run_dir in run_dirs.into_iter().take(max_runs) {
            let candidate = run_dir.join("promoted_configs").join(filename);
            if candidate.is_file() {
                return Ok(Some(candidate));
            }
        }
    }

    Ok(None)
}

fn is_date_dir(name: &str) -> bool {
    let bytes = name.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(idx, byte)| matches!(idx, 4 | 7) || byte.is_ascii_digit())
}

fn promoted_scan_limit(env_name: &str, default: usize) -> usize {
    let Some(value) = env_string(env_name) else {
        return default;
    };
    let parsed = value.parse::<usize>().unwrap_or(default);
    parsed.clamp(MIN_PROMOTED_SCAN_LIMIT, MAX_PROMOTED_SCAN_LIMIT)
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

fn project_root() -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    path.canonicalize().unwrap_or(path)
}

#[cfg(test)]
pub(crate) fn test_env_lock() -> &'static std::sync::Mutex<()> {
    use std::sync::{Mutex, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

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
    fn resolves_strategy_mode_overlay_from_env() {
        let _guard = test_env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        fs::write(
            &config_path,
            r#"
global:
  trade:
    leverage: 2.0
modes:
  primary:
    global:
      trade:
        leverage: 7.5
"#,
        )
        .unwrap();

        let _env = EnvGuard::set(&[
            (
                "AI_QUANT_STRATEGY_YAML",
                Some(config_path.to_str().unwrap()),
            ),
            ("AI_QUANT_STRATEGY_MODE", Some("PRIMARY")),
            ("AI_QUANT_STRATEGY_MODE_FILE", None),
            ("AI_QUANT_PROMOTED_ROLE", None),
            ("AI_QUANT_ARTIFACTS_DIR", None),
        ]);

        let resolved = resolve_paper_service_config(None).unwrap();
        let config = resolved.load_config(None, false).unwrap();

        assert!((config.trade.leverage - 7.5).abs() < f64::EPSILON);
        assert_eq!(resolved.strategy_mode(), Some("PRIMARY"));
        assert_eq!(resolved.effective_config_path(), config_path.as_path());
    }

    #[test]
    fn resolves_strategy_mode_from_file_when_env_is_unset() {
        let _guard = test_env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let mode_path = dir.path().join("strategy_mode.txt");
        fs::write(
            &config_path,
            r#"
global:
  trade:
    leverage: 2.0
modes:
  fallback:
    global:
      trade:
        leverage: 6.0
"#,
        )
        .unwrap();
        fs::write(&mode_path, "fallback\n").unwrap();

        let _env = EnvGuard::set(&[
            (
                "AI_QUANT_STRATEGY_YAML",
                Some(config_path.to_str().unwrap()),
            ),
            ("AI_QUANT_STRATEGY_MODE", None),
            (
                "AI_QUANT_STRATEGY_MODE_FILE",
                Some(mode_path.to_str().unwrap()),
            ),
            ("AI_QUANT_PROMOTED_ROLE", None),
            ("AI_QUANT_ARTIFACTS_DIR", None),
        ]);

        let resolved = resolve_paper_service_config(None).unwrap();
        let config = resolved.load_config(None, false).unwrap();

        assert!((config.trade.leverage - 6.0).abs() < f64::EPSILON);
        assert_eq!(resolved.strategy_mode(), Some("fallback"));
    }

    #[test]
    fn merges_promoted_yaml_before_mode_overlay() {
        let _guard = test_env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let artifacts_dir = dir.path().join("artifacts");
        let promoted_path = artifacts_dir
            .join("2026-03-06")
            .join("run_nightly_20260306T010000Z")
            .join("promoted_configs");
        fs::create_dir_all(&promoted_path).unwrap();
        fs::write(
            &config_path,
            r#"
global:
  trade:
    leverage: 2.0
    allocation_pct: 0.03
modes:
  primary:
    global:
      trade:
        leverage: 9.0
"#,
        )
        .unwrap();
        fs::write(
            promoted_path.join("primary.yaml"),
            r#"
global:
  trade:
    allocation_pct: 0.05
"#,
        )
        .unwrap();

        let _env = EnvGuard::set(&[
            (
                "AI_QUANT_STRATEGY_YAML",
                Some(config_path.to_str().unwrap()),
            ),
            (
                "AI_QUANT_ARTIFACTS_DIR",
                Some(artifacts_dir.to_str().unwrap()),
            ),
            ("AI_QUANT_PROMOTED_ROLE", Some("primary")),
            ("AI_QUANT_STRATEGY_MODE", Some("primary")),
            ("AI_QUANT_STRATEGY_MODE_FILE", None),
        ]);

        let resolved = resolve_paper_service_config(None).unwrap();
        let config = resolved.load_config(None, false).unwrap();

        assert!((config.trade.leverage - 9.0).abs() < f64::EPSILON);
        assert!((config.trade.allocation_pct - 0.05).abs() < f64::EPSILON);
        assert_ne!(resolved.effective_config_path(), config_path.as_path());
        assert_eq!(
            resolved.promoted_config_path(),
            Some(promoted_path.join("primary.yaml").as_path())
        );
    }
}
