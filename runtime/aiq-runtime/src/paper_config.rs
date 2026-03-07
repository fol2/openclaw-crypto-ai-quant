use aiq_runtime_core::runtime::{build_bootstrap, RuntimeBootstrap, RuntimeMode};
use anyhow::{Context, Result};
use bt_core::config::{self, StrategyConfig};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::paper_lane::PaperLane;

const DEFAULT_CONFIG_PATH: &str = "config/strategy_overrides.yaml";
const VALID_PROMOTED_ROLES: &[&str] = &["primary", "fallback", "conservative"];
const DEFAULT_PROMOTED_SCAN_DATE_DIRS: usize = 90;
const DEFAULT_PROMOTED_SCAN_RUN_DIRS_PER_DATE: usize = 200;
const MIN_PROMOTED_SCAN_LIMIT: usize = 1;
const MAX_PROMOTED_SCAN_LIMIT: usize = 10_000;

#[derive(Debug, Clone, PartialEq)]
pub struct PaperEffectiveConfig {
    base_config_path: PathBuf,
    active_yaml_path: PathBuf,
    effective_yaml_path: PathBuf,
    runtime_yaml_path: PathBuf,
    active_document: serde_yaml::Value,
    effective_document: serde_yaml::Value,
    runtime_document: serde_yaml::Value,
    promoted_role: Option<String>,
    promoted_config_path: Option<PathBuf>,
    strategy_mode: Option<String>,
    strategy_mode_source: Option<String>,
    strategy_overrides_sha1: String,
    config_id: String,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PaperEffectiveConfigReport {
    pub base_config_path: String,
    pub config_path: String,
    pub active_yaml_path: String,
    pub effective_yaml_path: String,
    pub interval: String,
    pub promoted_role: Option<String>,
    pub promoted_config_path: Option<String>,
    pub strategy_mode: Option<String>,
    pub strategy_mode_source: Option<String>,
    pub strategy_overrides_sha1: String,
    pub config_id: String,
    pub warnings: Vec<String>,
}

impl PaperEffectiveConfig {
    pub fn resolve(
        config: Option<&Path>,
        lane: Option<PaperLane>,
        project_dir: Option<&Path>,
    ) -> Result<Self> {
        let lane_defaults = match lane {
            Some(lane) => Some(lane.defaults(project_dir)?),
            None => None,
        };
        let base_config_path = resolve_config_path(config, lane_defaults.as_ref());
        let resolved_project_root = lane_defaults
            .as_ref()
            .map(|defaults| defaults.project_dir.clone())
            .unwrap_or_else(project_root);
        let materialised_output_root =
            materialised_output_root(resolved_project_root.as_path(), &base_config_path);
        let mut warnings = Vec::new();
        let base_document = config::load_yaml_document_checked(
            base_config_path
                .to_str()
                .context("base config path must be valid UTF-8")?,
        )
        .map_err(anyhow::Error::msg)?;

        let mut active_document = base_document.clone();
        let requested_promoted_role = lane_defaults
            .as_ref()
            .and_then(|defaults| defaults.promoted_role.clone())
            .or_else(|| env_string("AI_QUANT_PROMOTED_ROLE").map(|role| role.to_ascii_lowercase()));
        let promoted_role = requested_promoted_role.clone();
        let mut promoted_config_path = None;

        if let Some(role) = requested_promoted_role.as_deref() {
            if !VALID_PROMOTED_ROLES.contains(&role) {
                warnings.push(format!(
                    "unsupported AI_QUANT_PROMOTED_ROLE={role}; using the base config"
                ));
            } else {
                let artifacts_dir = env_path("AI_QUANT_ARTIFACTS_DIR")
                    .unwrap_or_else(|| resolved_project_root.join("artifacts"));
                if let Some(promoted_path) = find_latest_promoted_config(&artifacts_dir, role) {
                    match config::load_yaml_document_checked(
                        promoted_path
                            .to_str()
                            .context("promoted config path must be valid UTF-8")?,
                    ) {
                        Ok(promoted_document) => {
                            config::deep_merge_yaml_value_python_compat(
                                &mut active_document,
                                &promoted_document,
                            );
                            promoted_config_path = Some(promoted_path);
                        }
                        Err(err) => warnings.push(format!(
                            "failed to load promoted config for role={role}: {err}"
                        )),
                    }
                } else {
                    warnings.push(format!(
                        "no promoted config found for role={role}; using the base config"
                    ));
                }
            }
        }

        let active_yaml_path = if promoted_config_path.is_some() {
            let path =
                promoted_output_path(materialised_output_root.as_path(), promoted_role.as_deref());
            write_yaml_document(
                &path,
                active_document_header(
                    promoted_role.as_deref().unwrap_or_default(),
                    promoted_config_path.as_deref(),
                ),
                &active_document,
            )?;
            path
        } else {
            base_config_path.clone()
        };

        let (strategy_mode, strategy_mode_source, strategy_mode_warning) =
            resolve_strategy_mode(lane_defaults.as_ref(), resolved_project_root.as_path());
        if let Some(strategy_mode_warning) = strategy_mode_warning {
            warnings.push(strategy_mode_warning);
        }

        let (effective_document, mode_applied) =
            apply_strategy_mode_overlay(&active_document, strategy_mode.as_deref(), &mut warnings)?;
        let effective_yaml_path = if mode_applied {
            let path = effective_output_path(
                materialised_output_root.as_path(),
                &base_config_path,
                promoted_role.as_deref(),
                strategy_mode.as_deref(),
            );
            write_yaml_document(
                &path,
                effective_document_header(
                    &base_config_path,
                    promoted_role.as_deref(),
                    promoted_config_path.as_deref(),
                    strategy_mode.as_deref(),
                ),
                &effective_document,
            )?;
            path
        } else {
            active_yaml_path.clone()
        };

        let runtime_document =
            config::materialise_runtime_document(&effective_document, false, None)
                .map_err(anyhow::Error::msg)?;
        let runtime_yaml_path = runtime_output_path(
            materialised_output_root.as_path(),
            &base_config_path,
            promoted_role.as_deref(),
            strategy_mode.as_deref(),
        );
        write_yaml_document(
            &runtime_yaml_path,
            runtime_document_header(
                &base_config_path,
                promoted_role.as_deref(),
                promoted_config_path.as_deref(),
                strategy_mode.as_deref(),
            ),
            &runtime_document,
        )?;
        let config_id = yaml_document_sha256(&runtime_document);

        Ok(Self {
            base_config_path,
            active_yaml_path,
            effective_yaml_path,
            runtime_yaml_path,
            active_document,
            effective_document,
            runtime_document,
            promoted_role,
            promoted_config_path,
            strategy_mode,
            strategy_mode_source,
            strategy_overrides_sha1: config_id.clone(),
            config_id,
            warnings,
        })
    }

    pub fn base_config_path(&self) -> &Path {
        &self.base_config_path
    }

    pub fn active_yaml_path(&self) -> &Path {
        &self.active_yaml_path
    }

    pub fn effective_yaml_path(&self) -> &Path {
        &self.effective_yaml_path
    }

    pub fn config_path(&self) -> &Path {
        &self.runtime_yaml_path
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

    pub fn strategy_overrides_sha1(&self) -> &str {
        &self.strategy_overrides_sha1
    }

    pub fn config_id(&self) -> &str {
        &self.config_id
    }

    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    pub fn build_report(
        &self,
        symbol: Option<&str>,
        live: bool,
    ) -> Result<PaperEffectiveConfigReport> {
        let config = self.load_config(symbol, live)?;
        Ok(PaperEffectiveConfigReport {
            base_config_path: self.base_config_path().display().to_string(),
            config_path: self.config_path().display().to_string(),
            active_yaml_path: self.active_yaml_path().display().to_string(),
            effective_yaml_path: self.effective_yaml_path().display().to_string(),
            interval: config.engine.interval,
            promoted_role: self.promoted_role().map(ToOwned::to_owned),
            promoted_config_path: self
                .promoted_config_path()
                .map(|path| path.display().to_string()),
            strategy_mode: self.strategy_mode().map(ToOwned::to_owned),
            strategy_mode_source: self.strategy_mode_source().map(ToOwned::to_owned),
            strategy_overrides_sha1: self.strategy_overrides_sha1().to_string(),
            config_id: self.config_id().to_string(),
            warnings: self.warnings().to_vec(),
        })
    }

    pub fn load_config(&self, symbol: Option<&str>, live: bool) -> Result<StrategyConfig> {
        config::load_config_document_checked(&self.runtime_document, symbol, live, None)
            .map_err(anyhow::Error::msg)
    }

    pub fn build_runtime_bootstrap(
        &self,
        symbol: Option<&str>,
        live: bool,
        profile_override: Option<&str>,
    ) -> Result<RuntimeBootstrap> {
        let config = self.load_config(symbol, live)?;
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

pub(crate) fn resolve_config_path(
    config: Option<&Path>,
    lane_defaults: Option<&crate::paper_lane::PaperLaneDefaults>,
) -> PathBuf {
    let configured = config
        .map(Path::to_path_buf)
        .or_else(|| lane_defaults.map(|defaults| defaults.config_path.clone()))
        .or_else(|| env_path("AI_QUANT_STRATEGY_YAML"))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG_PATH));

    if configured.exists() {
        return configured;
    }

    let mut fallbacks = vec![PathBuf::from(format!("{}.example", configured.display()))];
    if let Some(file_name) = configured.file_name().and_then(|name| name.to_str()) {
        if let Some((stem, extension)) = file_name.rsplit_once('.') {
            fallbacks.push(configured.with_file_name(format!("{stem}.example.{extension}")));
        }
    }
    for fallback in fallbacks {
        if fallback.exists() {
            return fallback;
        }
    }

    configured
}

fn project_root() -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    path.canonicalize().unwrap_or(path)
}

fn resolve_strategy_mode(
    lane_defaults: Option<&crate::paper_lane::PaperLaneDefaults>,
    project_root: &Path,
) -> (Option<String>, Option<String>, Option<String>) {
    if let Some(strategy_mode) = env_string("AI_QUANT_STRATEGY_MODE") {
        return (
            config::normalise_strategy_mode_key(&strategy_mode),
            Some("env".to_string()),
            None,
        );
    }

    let mode_file = env_path("AI_QUANT_STRATEGY_MODE_FILE")
        .or_else(|| lane_defaults.map(|defaults| defaults.strategy_mode_file.clone()))
        .unwrap_or_else(|| project_root.join("artifacts/state/strategy_mode.txt"));
    if !mode_file.exists() {
        if let Some(strategy_mode) = lane_defaults
            .and_then(|defaults| defaults.default_strategy_mode.as_deref())
            .and_then(config::normalise_strategy_mode_key)
        {
            return (Some(strategy_mode), Some("lane_default".to_string()), None);
        }
        return (None, None, None);
    }

    match fs::read_to_string(&mode_file) {
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

fn apply_strategy_mode_overlay(
    document: &serde_yaml::Value,
    strategy_mode: Option<&str>,
    warnings: &mut Vec<String>,
) -> Result<(serde_yaml::Value, bool)> {
    let Some(mode_key) = strategy_mode.and_then(config::normalise_strategy_mode_key) else {
        return Ok((document.clone(), false));
    };

    let serde_yaml::Value::Mapping(root_map) = document else {
        anyhow::bail!("effective config document must have a mapping root");
    };
    let modes_key = serde_yaml::Value::String("modes".to_string());
    let Some(modes_value) = root_map.get(&modes_key) else {
        warnings.push(format!(
            "strategy mode {} requested but no modes section exists in the active YAML",
            mode_key
        ));
        return Ok((document.clone(), false));
    };
    let serde_yaml::Value::Mapping(modes_map) = modes_value else {
        warnings.push("modes section is not a mapping; ignoring strategy mode".to_string());
        return Ok((document.clone(), false));
    };

    let Some(mode_overlay) = lookup_mapping_value(modes_map, &mode_key) else {
        warnings.push(format!(
            "strategy mode {} was requested but no matching modes entry was found",
            mode_key
        ));
        return Ok((document.clone(), false));
    };

    let mut output = document.clone();
    let output_map = output
        .as_mapping_mut()
        .context("effective config document must have a mapping root")?;
    let global_key = serde_yaml::Value::String("global".to_string());
    let symbols_key = serde_yaml::Value::String("symbols".to_string());
    output_map
        .entry(global_key.clone())
        .or_insert_with(|| serde_yaml::Value::Mapping(Default::default()));
    output_map
        .entry(symbols_key.clone())
        .or_insert_with(|| serde_yaml::Value::Mapping(Default::default()));

    if let serde_yaml::Value::Mapping(mode_map) = mode_overlay {
        let has_structured_overlay =
            mode_map.contains_key(&global_key) || mode_map.contains_key(&symbols_key);
        if has_structured_overlay {
            if let Some(global_overlay) = mode_map.get(&global_key) {
                let global = output_map
                    .get_mut(&global_key)
                    .context("global section must exist")?;
                config::deep_merge_yaml_value_python_compat(global, global_overlay);
            }
            if let Some(symbols_overlay) = mode_map.get(&symbols_key) {
                let symbols = output_map
                    .get_mut(&symbols_key)
                    .context("symbols section must exist")?;
                config::deep_merge_yaml_value_python_compat(symbols, symbols_overlay);
            }
            return Ok((output, true));
        }
    }

    let global = output_map
        .get_mut(&global_key)
        .context("global section must exist")?;
    config::deep_merge_yaml_value_python_compat(global, mode_overlay);
    Ok((output, true))
}

fn lookup_mapping_value<'a>(
    map: &'a serde_yaml::Mapping,
    key: &str,
) -> Option<&'a serde_yaml::Value> {
    let exact = serde_yaml::Value::String(key.to_string());
    let upper = serde_yaml::Value::String(key.to_ascii_uppercase());
    let lower = serde_yaml::Value::String(key.to_ascii_lowercase());
    map.get(&exact)
        .or_else(|| map.get(&upper))
        .or_else(|| map.get(&lower))
}

fn materialised_output_root(project_root: &Path, base_config_path: &Path) -> PathBuf {
    env_path("AI_QUANT_EFFECTIVE_CONFIG_OUTPUT_ROOT").unwrap_or_else(|| {
        if base_config_path.is_absolute() && !base_config_path.starts_with(project_root) {
            base_config_path
                .parent()
                .unwrap_or(project_root)
                .to_path_buf()
        } else {
            project_root.to_path_buf()
        }
    })
}

fn promoted_output_path(output_root: &Path, role: Option<&str>) -> PathBuf {
    let role = role.unwrap_or("unknown");
    output_root
        .join("config")
        .join(format!("strategy_overrides._promoted_{role}.yaml"))
}

fn effective_output_path(
    output_root: &Path,
    base_config_path: &Path,
    promoted_role: Option<&str>,
    strategy_mode: Option<&str>,
) -> PathBuf {
    let stem = base_config_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("strategy_overrides");
    let role_part = promoted_role.unwrap_or("base");
    let mode_part = strategy_mode.unwrap_or("none");
    output_root
        .join("artifacts")
        .join("_effective_configs")
        .join(format!("{stem}.{role_part}.{mode_part}.yaml"))
}

fn runtime_output_path(
    output_root: &Path,
    base_config_path: &Path,
    promoted_role: Option<&str>,
    strategy_mode: Option<&str>,
) -> PathBuf {
    let stem = base_config_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("strategy_overrides");
    let role_part = promoted_role.unwrap_or("base");
    let mode_part = strategy_mode.unwrap_or("none");
    output_root
        .join("artifacts")
        .join("_runtime_configs")
        .join(format!("{stem}.{role_part}.{mode_part}.runtime.yaml"))
}

fn active_document_header(role: &str, promoted_config_path: Option<&Path>) -> String {
    let source = promoted_config_path
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "n/a".to_string());
    format!(
        "# AUTO-GENERATED by Rust paper effective-config resolver\n# Role: {role}\n# Source: {source}\n# Do not edit — this file is overwritten by the resolver.\n"
    )
}

fn effective_document_header(
    base_config_path: &Path,
    promoted_role: Option<&str>,
    promoted_config_path: Option<&Path>,
    strategy_mode: Option<&str>,
) -> String {
    format!(
        "# AUTO-GENERATED by Rust paper effective-config resolver\n# Base: {}\n# Role: {}\n# Source: {}\n# Strategy mode: {}\n# Do not edit — this file is overwritten by the resolver.\n",
        base_config_path.display(),
        promoted_role.unwrap_or("none"),
        promoted_config_path
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "n/a".to_string()),
        strategy_mode.unwrap_or("none"),
    )
}

fn runtime_document_header(
    base_config_path: &Path,
    promoted_role: Option<&str>,
    promoted_config_path: Option<&Path>,
    strategy_mode: Option<&str>,
) -> String {
    format!(
        "# AUTO-GENERATED by Rust paper effective-config resolver\n# Contract: runtime-facing materialised config\n# Base: {}\n# Role: {}\n# Source: {}\n# Strategy mode: {}\n# Do not edit — this file is overwritten by the resolver.\n",
        base_config_path.display(),
        promoted_role.unwrap_or("none"),
        promoted_config_path
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "n/a".to_string()),
        strategy_mode.unwrap_or("none"),
    )
}

fn write_yaml_document(path: &Path, header: String, document: &serde_yaml::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory: {}", parent.display()))?;
    }
    let payload =
        header + &serde_yaml::to_string(document).context("failed to serialise YAML document")?;
    if let Ok(existing) = fs::read_to_string(path) {
        if existing == payload {
            return Ok(());
        }
    }
    let unique_suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let tmp_path =
        path.with_extension(format!("yaml.tmp.{}.{}", std::process::id(), unique_suffix));
    fs::write(&tmp_path, payload)
        .with_context(|| format!("failed to write YAML: {}", tmp_path.display()))?;
    fs::rename(&tmp_path, path)
        .with_context(|| format!("failed to replace YAML: {}", path.display()))?;
    Ok(())
}

fn yaml_document_sha256(document: &serde_yaml::Value) -> String {
    let payload = serde_json::to_vec(&yaml_value_to_json(document)).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(payload);
    format!("{:x}", hasher.finalize())
}

fn yaml_value_to_json(value: &serde_yaml::Value) -> serde_json::Value {
    match value {
        serde_yaml::Value::Null => serde_json::Value::Null,
        serde_yaml::Value::Bool(value) => serde_json::Value::Bool(*value),
        serde_yaml::Value::Number(value) => {
            if let Some(integer) = value.as_i64() {
                serde_json::Value::Number(serde_json::Number::from(integer))
            } else if let Some(integer) = value.as_u64() {
                serde_json::Value::Number(serde_json::Number::from(integer))
            } else if let Some(float) = value.as_f64() {
                serde_json::json!(float)
            } else {
                serde_json::Value::Null
            }
        }
        serde_yaml::Value::String(value) => serde_json::Value::String(value.clone()),
        serde_yaml::Value::Sequence(values) => {
            serde_json::Value::Array(values.iter().map(yaml_value_to_json).collect())
        }
        serde_yaml::Value::Mapping(map) => {
            let mut ordered = BTreeMap::new();
            for (key, value) in map {
                ordered.insert(yaml_key_to_string(key), yaml_value_to_json(value));
            }
            serde_json::to_value(ordered).unwrap_or(serde_json::Value::Null)
        }
        serde_yaml::Value::Tagged(tagged) => yaml_value_to_json(&tagged.value),
    }
}

fn yaml_key_to_string(value: &serde_yaml::Value) -> String {
    match value {
        serde_yaml::Value::String(value) => value.clone(),
        other => serde_yaml::to_string(other)
            .unwrap_or_else(|_| format!("{other:?}"))
            .trim()
            .to_string(),
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

    let mut date_dirs = fs::read_dir(artifacts_dir)
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
        let mut run_dirs = fs::read_dir(date_dir)
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
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

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

    fn env_lock() -> &'static Mutex<()> {
        ENV_LOCK.get_or_init(|| Mutex::new(()))
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

        let effective = PaperEffectiveConfig::resolve(Some(&config_path), None, None).unwrap();
        let cfg = effective.load_config(None, false).unwrap();

        assert_eq!(effective.promoted_role(), Some("primary"));
        assert_eq!(effective.strategy_mode(), Some("fallback"));
        assert_eq!(effective.strategy_mode_source(), Some("file"));
        assert_eq!(cfg.engine.interval, "1h");
        assert!((cfg.trade.leverage - 7.5).abs() < f64::EPSILON);
        assert!(effective.promoted_config_path().is_some());
        assert_ne!(
            effective.active_yaml_path(),
            effective.effective_yaml_path()
        );
        assert_eq!(effective.strategy_overrides_sha1().len(), 64);
        assert_eq!(effective.config_id().len(), 64);
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

        let effective = PaperEffectiveConfig::resolve(Some(&config_path), None, None).unwrap();
        assert_eq!(effective.strategy_mode(), Some("primary"));
        assert_eq!(effective.strategy_mode_source(), Some("env"));
    }

    #[test]
    fn effective_config_respects_output_root_override() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let output_root = dir.path().join("runtime-output");
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

        let _env = EnvGuard::set(&[
            ("AI_QUANT_PROMOTED_ROLE", Some("primary")),
            (
                "AI_QUANT_ARTIFACTS_DIR",
                Some(artifacts_dir.to_str().unwrap()),
            ),
            ("AI_QUANT_STRATEGY_MODE", Some("fallback")),
            ("AI_QUANT_STRATEGY_MODE_FILE", None),
            (
                "AI_QUANT_EFFECTIVE_CONFIG_OUTPUT_ROOT",
                Some(output_root.to_str().unwrap()),
            ),
        ]);

        let effective = PaperEffectiveConfig::resolve(Some(&config_path), None, None).unwrap();

        assert!(effective.active_yaml_path().starts_with(&output_root));
        assert!(effective.effective_yaml_path().starts_with(&output_root));
        assert!(effective.config_path().starts_with(&output_root));
        assert_eq!(
            effective.active_yaml_path(),
            output_root.join("config/strategy_overrides._promoted_primary.yaml")
        );
        assert_eq!(
            effective.effective_yaml_path(),
            output_root.join("artifacts/_effective_configs/strategy.primary.fallback.yaml")
        );
        assert_eq!(
            effective.config_path(),
            output_root.join("artifacts/_runtime_configs/strategy.primary.fallback.runtime.yaml")
        );
    }

    #[test]
    fn effective_config_uses_lane_defaults_without_env() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_dir = dir.path().join("config");
        let config_path = config_dir.join("strategy_overrides.paper2.yaml");
        let output_root = dir.path().join("runtime-output");
        fs::create_dir_all(&config_dir).unwrap();
        fs::write(
            &config_path,
            "global:\n  engine:\n    interval: 30m\nmodes:\n  fallback:\n    global:\n      engine:\n        interval: 1h\n",
        )
        .unwrap();
        let _env = EnvGuard::set(&[
            ("AI_QUANT_PROMOTED_ROLE", None),
            ("AI_QUANT_STRATEGY_MODE", None),
            ("AI_QUANT_STRATEGY_MODE_FILE", None),
            (
                "AI_QUANT_EFFECTIVE_CONFIG_OUTPUT_ROOT",
                Some(output_root.to_str().unwrap()),
            ),
        ]);

        let effective =
            PaperEffectiveConfig::resolve(None, Some(PaperLane::Paper2), Some(dir.path())).unwrap();
        let report = effective.build_report(None, false).unwrap();

        assert_eq!(effective.base_config_path(), config_path.as_path());
        assert_eq!(effective.promoted_role(), Some("fallback"));
        assert_eq!(effective.strategy_mode(), Some("fallback"));
        assert_eq!(effective.strategy_mode_source(), Some("lane_default"));
        assert_eq!(report.promoted_role.as_deref(), Some("fallback"));
        assert_eq!(report.strategy_mode.as_deref(), Some("fallback"));
        assert_eq!(report.strategy_mode_source.as_deref(), Some("lane_default"));
    }

    #[test]
    fn effective_config_does_not_rewrite_runtime_yaml_when_payload_is_unchanged() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        let output_root = dir.path().join("runtime-output");
        fs::write(&config_path, "global:\n  engine:\n    interval: 30m\n").unwrap();

        let _env = EnvGuard::set(&[
            ("AI_QUANT_PROMOTED_ROLE", None),
            ("AI_QUANT_STRATEGY_MODE", None),
            ("AI_QUANT_STRATEGY_MODE_FILE", None),
            (
                "AI_QUANT_EFFECTIVE_CONFIG_OUTPUT_ROOT",
                Some(output_root.to_str().unwrap()),
            ),
        ]);

        let first = PaperEffectiveConfig::resolve(Some(&config_path), None, None).unwrap();
        let first_modified = fs::metadata(first.config_path())
            .and_then(|meta| meta.modified())
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let second = PaperEffectiveConfig::resolve(Some(&config_path), None, None).unwrap();
        let second_modified = fs::metadata(second.config_path())
            .and_then(|meta| meta.modified())
            .unwrap();

        assert_eq!(first.config_path(), second.config_path());
        assert_eq!(first_modified, second_modified);
    }

    #[test]
    fn resolve_config_path_prefers_lane_example_fallbacks() {
        let dir = tempdir().unwrap();
        let config_dir = dir.path().join("config");
        fs::create_dir_all(&config_dir).unwrap();
        let example_path = config_dir.join("strategy_overrides.paper2.example.yaml");
        fs::write(&example_path, "global:\n  engine:\n    interval: 30m\n").unwrap();

        let lane_defaults = PaperLane::Paper2.defaults(Some(dir.path())).unwrap();
        let resolved = resolve_config_path(None, Some(&lane_defaults));
        assert_eq!(resolved, example_path);
    }

    #[test]
    fn active_document_hash_ignores_yaml_order_and_comments() {
        let left = serde_yaml::from_str::<serde_yaml::Value>(
            "# comment\nsymbols:\n  ETH:\n    trade:\n      leverage: 4.0\nglobal:\n  trade:\n    allocation_pct: 0.04\n",
        )
        .unwrap();
        let right = serde_yaml::from_str::<serde_yaml::Value>(
            "global:\n  trade:\n    allocation_pct: 0.04\nsymbols:\n  ETH:\n    trade:\n      leverage: 4.0\n",
        )
        .unwrap();
        let changed = serde_yaml::from_str::<serde_yaml::Value>(
            "global:\n  trade:\n    allocation_pct: 0.05\nsymbols:\n  ETH:\n    trade:\n      leverage: 4.0\n",
        )
        .unwrap();

        assert_eq!(yaml_document_sha256(&left), yaml_document_sha256(&right));
        assert_ne!(yaml_document_sha256(&left), yaml_document_sha256(&changed));
    }

    #[test]
    fn config_id_tracks_materialised_strategy_mode_overlay() {
        let _guard = env_lock().lock().unwrap();
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("strategy.yaml");
        fs::write(
            &config_path,
            r#"
global:
  engine:
    interval: 30m
modes:
  primary:
    global:
      engine:
        interval: 5m
  fallback:
    global:
      engine:
        interval: 1h
"#,
        )
        .unwrap();

        let _base_env = EnvGuard::set(&[
            ("AI_QUANT_PROMOTED_ROLE", None),
            ("AI_QUANT_STRATEGY_MODE_FILE", None),
        ]);

        env::remove_var("AI_QUANT_STRATEGY_MODE");
        let base = PaperEffectiveConfig::resolve(Some(&config_path), None, None).unwrap();
        env::set_var("AI_QUANT_STRATEGY_MODE", "primary");
        let primary = PaperEffectiveConfig::resolve(Some(&config_path), None, None).unwrap();
        env::set_var("AI_QUANT_STRATEGY_MODE", "fallback");
        let fallback = PaperEffectiveConfig::resolve(Some(&config_path), None, None).unwrap();
        env::remove_var("AI_QUANT_STRATEGY_MODE");

        assert_ne!(base.config_id(), primary.config_id());
        assert_ne!(primary.config_id(), fallback.config_id());
        assert_eq!(base.build_report(None, false).unwrap().interval, "30m");
        assert_eq!(primary.build_report(None, false).unwrap().interval, "5m");
        assert_eq!(fallback.build_report(None, false).unwrap().interval, "1h");
    }
}
