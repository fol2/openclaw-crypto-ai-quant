"""Promoted config loader for paper trading daemons.

Factory runs produce `promoted_configs/{primary,fallback,conservative}.yaml` inside
their run directory.  This module locates the most recent promoted config for a given
role and merges it on top of the base `strategy_overrides.yaml` so that paper daemons
can automatically pick up factory-optimised parameters.

Mapping (conventional):
    paper1  → AI_QUANT_PROMOTED_ROLE=primary
    paper2  → AI_QUANT_PROMOTED_ROLE=fallback
    paper3  → AI_QUANT_PROMOTED_ROLE=conservative
    livepaper → (unchanged, uses its own config)

Environment variables:
    AI_QUANT_PROMOTED_ROLE      primary | fallback | conservative
    AI_QUANT_ARTIFACTS_DIR      Root of the artifacts tree (default: <project>/artifacts)
    AI_QUANT_STRATEGY_YAML      Base config path (set by StrategyManager if unset)
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .utils import deep_merge


VALID_ROLES = frozenset({"primary", "fallback", "conservative"})

# Run directory naming convention produced by factory_run.py:
#   run_<run_id>  e.g.  run_nightly_20260214T014625Z
_RUN_DIR_RE = re.compile(r"^run_")


def _default_artifacts_dir() -> Path:
    """Return the default artifacts root: <project_root>/artifacts."""
    here = Path(__file__).resolve().parent  # engine/
    return (here.parent / "artifacts").resolve()


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

    # Date directories are YYYY-MM-DD — sort descending to find newest first.
    date_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", d.name)],
        key=lambda d: d.name,
        reverse=True,
    )

    for date_dir in date_dirs:
        # Run directories inside each date dir — sort descending by name.
        run_dirs = sorted(
            [d for d in date_dir.iterdir() if d.is_dir() and _RUN_DIR_RE.match(d.name)],
            key=lambda d: d.name,
            reverse=True,
        )
        for run_dir in run_dirs:
            candidate = run_dir / "promoted_configs" / filename
            if candidate.is_file():
                return candidate.resolve()

    return None


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
        print(f"⚠️ promoted_config: no promoted config found for role='{role}' "
              f"under {artifacts_dir}; falling back to base config")
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
        print(f"⚠️ promoted_config: promoted file is empty or invalid: {promoted_path}; "
              f"falling back to base config")
        return None, None

    # Deep-merge: promoted takes precedence over base.
    merged = deep_merge(base, promoted)

    print(f"📋 promoted_config: loaded role='{role}' from {promoted_path}")
    return merged, str(promoted_path)


def maybe_apply_promoted_config() -> str | None:
    """Check AI_QUANT_PROMOTED_ROLE and, if set, write a merged config for StrategyManager.

    This should be called early in daemon startup, **before** ``StrategyManager.get()``.
    If a promoted config is found, this writes the merged YAML to a deterministic path
    and sets ``AI_QUANT_STRATEGY_YAML`` so that StrategyManager picks it up transparently.

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
