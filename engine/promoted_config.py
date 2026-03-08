"""Compatibility helpers around Rust-owned effective-config selection.

Rust now owns the authoritative effective-config contract for paper and
live-facing control-plane consumers. This module keeps the older Python
promoted-config helpers for backward-compatible tests, while also exposing thin
wrappers that shell out to `aiq-runtime paper effective-config` /
`aiq-runtime live effective-config` for the active runtime start-up and factory
materialisation paths.

Mapping (conventional):
    paper1  → AI_QUANT_PROMOTED_ROLE=primary
    paper2  → AI_QUANT_PROMOTED_ROLE=fallback
    paper3  → AI_QUANT_PROMOTED_ROLE=conservative
    livepaper → (unchanged, uses its own config)

Environment variables:
    AI_QUANT_PROMOTED_ROLE      primary | fallback | conservative
    AI_QUANT_ARTIFACTS_DIR      Root of the artifacts tree (default: <project>/artifacts)
    AI_QUANT_STRATEGY_YAML      Base config path input for the Rust resolver
    AI_QUANT_RUNTIME_BIN        Optional absolute path to `aiq-runtime`
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .utils import deep_merge


VALID_ROLES = frozenset({"primary", "fallback", "conservative"})
DEFAULT_PROMOTED_SCAN_DATE_DIRS = 90
DEFAULT_PROMOTED_SCAN_RUN_DIRS_PER_DATE = 200
MIN_PROMOTED_SCAN_LIMIT = 1
MAX_PROMOTED_SCAN_LIMIT = 10000

# Run directory naming convention produced by factory_run.py:
#   run_<run_id>  e.g.  run_nightly_20260214T014625Z
_RUN_DIR_RE = re.compile(r"^run_")

# Legacy helper cluster below this point is frozen compatibility surface only.
# New paper/factory control-plane ownership must go through the Rust resolver
# wrappers further down in this module.


def _default_artifacts_dir() -> Path:
    """Return the default artifacts root: <project_root>/artifacts."""
    here = Path(__file__).resolve().parent  # engine/
    return (here.parent / "artifacts").resolve()


def _promoted_scan_limit(env_name: str, default: int) -> int:
    raw = os.getenv(env_name)
    if raw is None:
        return int(default)
    try:
        value = int(float(str(raw).strip()))
    except Exception:
        return int(default)
    return int(max(MIN_PROMOTED_SCAN_LIMIT, min(MAX_PROMOTED_SCAN_LIMIT, value)))


def _find_latest_promoted_config(artifacts_dir: str | Path, role: str) -> Path | None:
    """Find the most recent promoted config YAML for *role*.

    Scans ``artifacts_dir/<date_dirs>/run_*/promoted_configs/{role}.yaml``
    in reverse-chronological order (by directory name) and returns the first
    match, or ``None`` if no promoted config exists.

    Parameters
    ----------
    artifacts_dir:
        Root of the artifacts tree (e.g. ``<project>/artifacts``).
    role:
        One of ``primary``, ``fallback``, ``conservative``.

    Returns
    -------
    Path | None
        Absolute path to the promoted YAML, or None.
    """
    role = str(role).strip().lower()
    if role not in VALID_ROLES:
        return None

    root = Path(artifacts_dir).resolve()
    if not root.is_dir():
        return None

    filename = f"{role}.yaml"
    max_dates = _promoted_scan_limit("AI_QUANT_PROMOTED_SCAN_DATE_DIRS", DEFAULT_PROMOTED_SCAN_DATE_DIRS)
    max_runs = _promoted_scan_limit(
        "AI_QUANT_PROMOTED_SCAN_RUN_DIRS_PER_DATE",
        DEFAULT_PROMOTED_SCAN_RUN_DIRS_PER_DATE,
    )

    # Date directories are YYYY-MM-DD — sort descending to find newest first.
    date_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", d.name)],
        key=lambda d: d.name,
        reverse=True,
    )

    def _scan(*, date_limit: int | None, run_limit: int | None) -> Path | None:
        scan_dates = date_dirs if date_limit is None else date_dirs[:date_limit]
        for date_dir in scan_dates:
            run_dirs = sorted(
                [d for d in date_dir.iterdir() if d.is_dir() and _RUN_DIR_RE.match(d.name)],
                key=lambda d: d.name,
                reverse=True,
            )
            if run_limit is not None:
                run_dirs = run_dirs[:run_limit]
            for run_dir in run_dirs:
                candidate = run_dir / "promoted_configs" / filename
                if candidate.is_file():
                    return candidate.resolve()
        return None

    # Fast path: bounded scan for very large artifacts trees.
    found = _scan(date_limit=max_dates, run_limit=max_runs)
    if found is not None:
        return found

    # Backwards-compatible fallback: if bounded scan misses, do a full scan.
    return _scan(date_limit=None, run_limit=None)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"⚠️ promoted_config: failed to load YAML {path}: {exc}")
        return {}


def _write_text_atomic(path: Path, content: str) -> None:
    """Atomically write text content to ``path`` using ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def load_promoted_config(
    role: str,
    *,
    artifacts_dir: str | Path | None = None,
    base_yaml_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load a promoted config merged on top of the base strategy overrides.

    Parameters
    ----------
    role:
        One of ``primary``, ``fallback``, ``conservative``.
    artifacts_dir:
        Override for the artifacts root.  Falls back to ``AI_QUANT_ARTIFACTS_DIR``
        env var, then ``<project>/artifacts``.
    base_yaml_path:
        Override for the base config.  Falls back to ``AI_QUANT_STRATEGY_YAML``
        env var, then ``<project>/config/strategy_overrides.yaml``.

    Returns
    -------
    (merged_config, promoted_path)
        ``merged_config`` is the deep-merged dict (base ← promoted), or ``None``
        if no promoted config was found (caller should fall back to base).
        ``promoted_path`` is the absolute path string of the promoted YAML used,
        or ``None``.
    """
    role = str(role).strip().lower()
    if role not in VALID_ROLES:
        print(f"⚠️ promoted_config: invalid role '{role}', ignoring")
        return None, None

    # Resolve artifacts dir.
    if artifacts_dir is None:
        artifacts_dir = os.getenv("AI_QUANT_ARTIFACTS_DIR", "")
    if not artifacts_dir:
        artifacts_dir = _default_artifacts_dir()
    artifacts_dir = Path(artifacts_dir).resolve()

    # Find promoted config.
    promoted_path = _find_latest_promoted_config(artifacts_dir, role)
    if promoted_path is None:
        print(
            f"⚠️ promoted_config: no promoted config found for role='{role}' "
            f"under {artifacts_dir}; falling back to base config"
        )
        return None, None

    # Resolve base config path.
    if base_yaml_path is None:
        here = Path(__file__).resolve().parent
        base_yaml_path = os.getenv(
            "AI_QUANT_STRATEGY_YAML",
            str(here.parent / "config" / "strategy_overrides.yaml"),
        )
    base_yaml_path = Path(base_yaml_path).resolve()

    # Load both configs.
    base = _load_yaml(base_yaml_path)
    promoted = _load_yaml(promoted_path)

    if not promoted:
        print(f"⚠️ promoted_config: promoted file is empty or invalid: {promoted_path}; falling back to base config")
        return None, None

    # Deep-merge: promoted takes precedence over base.
    merged = deep_merge(base, promoted)

    print(f"📋 promoted_config: loaded role='{role}' from {promoted_path}")
    return merged, str(promoted_path)


def maybe_apply_promoted_config() -> str | None:
    """Legacy compatibility helper for the original Python-only merge path.

    Active paper start-up now calls the Rust resolver via
    :func:`apply_paper_effective_config`. This helper is retained for older tests
    and compatibility-only workflows that still expect the historical Python
    merge behaviour.

    Returns
    -------
    str | None
        The promoted role that was applied, or None if no promotion was active.
    """
    role = str(os.getenv("AI_QUANT_PROMOTED_ROLE", "") or "").strip().lower()
    if not role:
        return None

    merged, promoted_path = load_promoted_config(role)
    if merged is None:
        return None

    # Write merged config to a well-known location.
    here = Path(__file__).resolve().parent
    active_path = here.parent / "config" / f"strategy_overrides._promoted_{role}.yaml"
    try:
        # Add a comment header for traceability.
        header = (
            f"# AUTO-GENERATED by promoted_config loader\n"
            f"# Role: {role}\n"
            f"# Source: {promoted_path}\n"
            f"# Do not edit — this file is overwritten on daemon startup.\n"
        )
        content = header + yaml.dump(merged, default_flow_style=False, sort_keys=False, allow_unicode=True)
        _write_text_atomic(active_path, content)
    except Exception as exc:
        print(f"⚠️ promoted_config: failed to write merged config: {exc}")
        return None

    # Point StrategyManager at the merged file.
    os.environ["AI_QUANT_STRATEGY_YAML"] = str(active_path)
    print(f"📋 promoted_config: AI_QUANT_STRATEGY_YAML → {active_path}")
    return role


@dataclass(frozen=True)
class ResolvedEffectiveConfig:
    """Resolved Rust-owned effective-config contract for paper consumers."""

    base_config_path: str
    config_path: str
    active_yaml_path: str
    effective_yaml_path: str
    interval: str
    promoted_role: str | None
    promoted_config_path: str | None
    strategy_mode: str | None
    strategy_mode_source: str | None
    strategy_overrides_sha1: str
    config_id: str
    warnings: tuple[str, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _runtime_command() -> list[str]:
    explicit = str(os.getenv("AI_QUANT_RUNTIME_BIN", "") or "").strip()
    if explicit:
        return [str(Path(explicit).expanduser().resolve())]

    root = _repo_root()
    candidates = (
        root / "target" / "release" / "aiq-runtime",
        root / "target" / "debug" / "aiq-runtime",
    )
    for candidate in candidates:
        if candidate.is_file():
            return [str(candidate)]

    which = shutil.which("aiq-runtime")
    if which:
        return [which]

    cargo = shutil.which("cargo")
    if cargo:
        return [cargo, "run", "-q", "-p", "aiq-runtime", "--"]

    raise RuntimeError("could not locate aiq-runtime; set AI_QUANT_RUNTIME_BIN or install cargo")


def _effective_config_command(
    *,
    config_path: str | Path | None,
    live: bool,
    symbol: str | None,
    env: Mapping[str, str],
) -> list[str]:
    cmd = list(_runtime_command())
    cmd += ["live" if live else "paper", "effective-config", "--json"]
    selected_config = config_path
    if selected_config is None:
        selected_config = (
            str(env.get("AI_QUANT_BASE_STRATEGY_YAML", "") or "").strip()
            or str(env.get("AI_QUANT_STRATEGY_YAML", "") or "").strip()
            or None
        )
    if selected_config is not None:
        cmd += ["--config", str(Path(selected_config).expanduser().resolve())]
    if symbol:
        cmd += ["--symbol", str(symbol).strip().upper()]
    return cmd


def _resolved_effective_config_from_json(payload: Mapping[str, Any]) -> ResolvedEffectiveConfig:
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    return ResolvedEffectiveConfig(
        base_config_path=str(payload.get("base_config_path", "") or ""),
        config_path=str(payload.get("config_path", "") or ""),
        active_yaml_path=str(payload.get("active_yaml_path", "") or ""),
        effective_yaml_path=str(payload.get("effective_yaml_path", "") or ""),
        interval=str(payload.get("interval", "") or ""),
        promoted_role=None if payload.get("promoted_role") in {None, ""} else str(payload.get("promoted_role")),
        promoted_config_path=None
        if payload.get("promoted_config_path") in {None, ""}
        else str(payload.get("promoted_config_path")),
        strategy_mode=None if payload.get("strategy_mode") in {None, ""} else str(payload.get("strategy_mode")),
        strategy_mode_source=None
        if payload.get("strategy_mode_source") in {None, ""}
        else str(payload.get("strategy_mode_source")),
        strategy_overrides_sha1=str(payload.get("strategy_overrides_sha1", "") or ""),
        config_id=str(payload.get("config_id", "") or ""),
        warnings=tuple(str(item) for item in warnings if str(item).strip()),
    )


def resolve_effective_config(
    *,
    config_path: str | Path | None = None,
    live: bool = False,
    symbol: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ResolvedEffectiveConfig:
    """Resolve the active Rust-owned effective config for runtime consumers."""

    runtime_env = dict(os.environ if env is None else env)
    command = _effective_config_command(
        config_path=config_path,
        live=live,
        symbol=symbol,
        env=runtime_env,
    )
    proc = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(_repo_root()),
        env=runtime_env,
    )
    if int(proc.returncode) != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or f"exit code {proc.returncode}"
        raise RuntimeError(f"Rust effective-config resolver failed: {detail}")
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Rust effective-config resolver returned invalid JSON: {exc}") from exc
    return _resolved_effective_config_from_json(payload)


def apply_paper_effective_config(
    *,
    config_path: str | Path | None = None,
    symbol: str | None = None,
) -> ResolvedEffectiveConfig:
    """Resolve the paper effective config via Rust and apply it to the process env."""

    resolved = resolve_effective_config(config_path=config_path, live=False, symbol=symbol)
    _apply_effective_config_env(resolved)
    return resolved


def apply_live_effective_config(
    *,
    config_path: str | Path | None = None,
    symbol: str | None = None,
) -> ResolvedEffectiveConfig:
    """Resolve the live/dry-live effective config via Rust and apply it to the process env."""

    resolved = resolve_effective_config(config_path=config_path, live=True, symbol=symbol)
    _apply_effective_config_env(resolved)
    return resolved


def _apply_effective_config_env(resolved: ResolvedEffectiveConfig) -> None:
    """Export the Rust-owned effective-config contract into the current process env."""

    os.environ["AI_QUANT_STRATEGY_YAML"] = str(resolved.config_path)
    os.environ["AI_QUANT_ACTIVE_STRATEGY_YAML"] = str(resolved.active_yaml_path)
    os.environ["AI_QUANT_EFFECTIVE_STRATEGY_YAML"] = str(resolved.effective_yaml_path)
    os.environ["AI_QUANT_BASE_STRATEGY_YAML"] = str(resolved.base_config_path)
    os.environ["AI_QUANT_EFFECTIVE_CONFIG_ID"] = str(resolved.config_id)
    os.environ["AI_QUANT_EFFECTIVE_CONFIG_OWNER"] = "rust"
    os.environ["AI_QUANT_EFFECTIVE_CONFIG_MATERIALISED"] = (
        "1" if resolved.effective_yaml_path != resolved.active_yaml_path else "0"
    )
    if resolved.strategy_overrides_sha1:
        os.environ["AI_QUANT_EFFECTIVE_STRATEGY_SHA"] = str(resolved.strategy_overrides_sha1)
    if resolved.interval:
        os.environ["AI_QUANT_INTERVAL"] = str(resolved.interval)
    if resolved.promoted_role:
        os.environ["AI_QUANT_PROMOTED_ROLE"] = str(resolved.promoted_role)
    else:
        os.environ.pop("AI_QUANT_PROMOTED_ROLE", None)
    if resolved.promoted_config_path:
        os.environ["AI_QUANT_PROMOTED_CONFIG_PATH"] = str(resolved.promoted_config_path)
    else:
        os.environ.pop("AI_QUANT_PROMOTED_CONFIG_PATH", None)
    if resolved.strategy_mode:
        os.environ["AI_QUANT_STRATEGY_MODE"] = str(resolved.strategy_mode)
    else:
        os.environ.pop("AI_QUANT_STRATEGY_MODE", None)
    if resolved.strategy_mode_source:
        os.environ["AI_QUANT_STRATEGY_MODE_SOURCE"] = str(resolved.strategy_mode_source)
    else:
        os.environ.pop("AI_QUANT_STRATEGY_MODE_SOURCE", None)
    for warning in resolved.warnings:
        print(f"⚠️ promoted_config: {warning}")
    if resolved.promoted_role and resolved.promoted_config_path:
        print(f"📋 promoted_config: role='{resolved.promoted_role}' source={resolved.promoted_config_path}")
    if resolved.strategy_mode:
        mode_source = resolved.strategy_mode_source or "env"
        print(f"📋 promoted_config: strategy_mode='{resolved.strategy_mode}' source={mode_source}")
    print(f"📋 promoted_config: AI_QUANT_STRATEGY_YAML → {resolved.config_path}")
