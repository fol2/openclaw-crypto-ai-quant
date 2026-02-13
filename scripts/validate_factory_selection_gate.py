"""Validate factory selection and stage-gate evidence artefacts.

The validator is intentionally strict: every deployment stage expected by
`run_factory_stage_gate.sh` must expose canonical proof metadata before the
pipeline is considered safe to continue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_SELECTION_KEYS = ("selection_stage", "deploy_stage", "promotion_stage")

REQUIRED_BUNDLE_PATHS = (
    "run_dir",
    "run_metadata_json",
    "selection_json",
    "report_json",
    "report_md",
    "selection_md",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
    return default


def _collect_candidate_by_config_id(payload: Any, config_id: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    for row in payload.get("candidate_configs", []) if isinstance(payload.get("candidate_configs"), list) else []:
        if not isinstance(row, dict):
            continue
        if str(row.get("config_id", "")).strip() == str(config_id).strip():
            return row
    return None


def _error(message: str, errors: list[str]) -> None:
    errors.append(message)


def _expect_file_path(payload: dict[str, Any], key: str, *, errors: list[str]) -> Path | None:
    raw = payload.get(key)
    if not isinstance(raw, str):
        _error(f"evidence_bundle_paths.{key} is missing or not a string", errors)
        return None
    p = Path(raw)
    if not p.exists():
        _error(f"evidence_bundle_paths.{key} does not exist: {raw}", errors)
        return None
    return p


def validate_selection_path(path: Path, *, stage: str, allow_legacy: bool = False) -> list[str]:
    """Return validation errors for one `selection.json` payload."""

    errors: list[str] = []
    try:
        selection = _load_json(path)
    except Exception as exc:
        return [f"failed to load selection.json: {type(exc).__name__}: {exc}"]

    if not isinstance(selection, dict):
        return ["selection.json payload is not a mapping"]

    for key in REQUIRED_SELECTION_KEYS:
        if key not in selection:
            _error(f"selection.json missing required key: {key}", errors)

    if str(selection.get("selection_stage", "")).strip() != "selected":
        _error("selection_stage is not \"selected\"", errors)

    stage_lower = str(stage).strip().lower()
    if stage_lower in {"dry", "smoke"} and not allow_legacy:
        if str(selection.get("deploy_stage", "")).strip() not in {"no_deploy", "skipped"}:
            _error(
                f"dry/smoke stage must be no_deploy or skipped (got {selection.get('deploy_stage', 'missing')!r})",
                errors,
            )
        if str(selection.get("promotion_stage", "")).strip() not in {"skipped", ""}:
            _error(
                f"dry/smoke stage must be skipped by default (got {selection.get('promotion_stage', 'missing')!r})",
                errors,
            )
    elif stage_lower == "real" and not allow_legacy:
        if str(selection.get("deploy_stage", "")).strip() == "pending":
            _error("real stage deployment remains pending", errors)
        if str(selection.get("promotion_stage", "")).strip() == "pending":
            _error("real stage promotion remains pending", errors)

    evidence = selection.get("evidence_bundle_paths")
    if not isinstance(evidence, dict):
        _error("selection.json missing evidence_bundle_paths map", errors)
        return errors

    bundle_paths: dict[str, Path] = {}
    for key in REQUIRED_BUNDLE_PATHS:
        p = _expect_file_path(evidence, key=key, errors=errors)
        if p is not None:
            bundle_paths[key] = p

    selected = selection.get("selected")
    if not isinstance(selected, dict):
        _error("selection.json missing selected object", errors)
        return errors

    if not allow_legacy:
        canonical_cpu_verified = _as_bool(selected.get("canonical_cpu_verified"), default=False)
        if not canonical_cpu_verified:
            _error("selected candidate is not canonical_cpu_verified", errors)

        required_candidate_keys = ("pipeline_stage", "sweep_stage", "replay_stage", "validation_gate", "canonical_cpu_verified")
        for key in required_candidate_keys:
            if key not in selected:
                _error(f"selected candidate missing required key: {key}", errors)
        if str(selected.get("replay_stage", "")).strip() == "":
            _error("selected candidate has empty replay_stage", errors)

        # Proof artefact pointers must be present and exist.
        for key in ("replay_report_path", "replay_equivalence_report_path"):
            raw = selected.get(key)
            if not isinstance(raw, str) or not raw.strip():
                _error(f"selected candidate missing required key: {key}", errors)
                continue
            rep = Path(raw)
            if not rep.exists():
                _error(f"selected candidate {key} path does not exist: {raw}", errors)

        replay_status = str(selected.get("replay_equivalence_status", "")).strip().lower()
        if replay_status != "pass":
            _error(f"selected candidate replay_equivalence_status is not pass: {replay_status!r}", errors)

        status = str(selection.get("deployment_gate_status", "")).strip()
        if status and status not in {"ok", "passed"}:
            _error(f"selection deployment_gate_status indicates failure: {status!r}", errors)

        run_metadata_path = bundle_paths.get("run_metadata_json")
        if run_metadata_path is not None:
            try:
                run_meta = _load_json(run_metadata_path)
            except Exception as exc:
                _error(f"failed to load run_metadata_json: {type(exc).__name__}: {exc}", errors)
                run_meta = None
            if isinstance(run_meta, dict):
                cid = str(selected.get("config_id", "")).strip()
                if cid:
                    row = _collect_candidate_by_config_id(run_meta, cid)
                    if row is None:
                        _error(f"selected.config_id {cid!r} not found in run_metadata candidate_configs", errors)
                    elif not _as_bool(row.get("canonical_cpu_verified"), default=True):
                        _error(f"selected.candidate_metadata canonical_cpu_verified is false for config {cid!r}", errors)

    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate factory stage-gate evidence bundle.")
    parser.add_argument("--selection-json", required=True, help="Path to reports/selection.json")
    parser.add_argument("--stage", required=False, default="smoke", help="dry/smoke/real")
    parser.add_argument("--allow-legacy", action="store_true", help="Allow legacy payloads without stage metadata")
    parser.add_argument(
        "--status-message",
        default="",
        help="Optional plain text prefix to include in output",
    )
    return parser


def _run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    path = Path(str(args.selection_json))
    errors = validate_selection_path(path, stage=str(args.stage), allow_legacy=bool(args.allow_legacy))
    if errors:
        if args.status_message:
            print(args.status_message)
        for e in errors:
            print(f"[selection-gate] ERROR: {e}")
        return 1
    if args.status_message:
        print(args.status_message)
    print(f"[selection-gate] OK for {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
