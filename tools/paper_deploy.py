#!/usr/bin/env python3
"""Deploy a selected config_id to the paper trader safely (AQC-701).

This command:
1) Looks up the fully materialised YAML for a config_id from the local registry SQLite DB.
2) Writes it to the configured strategy overrides YAML path atomically (no partial writes).
3) Emits a `deploy_event.json` artefact with who/what/when/why and basic deployment metadata.
4) Optionally restarts the paper trading service when required (e.g. main interval change).

The default integration branch is `major-v8`; do not deploy to production from this repository
without explicit operator confirmation.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tools.config_id import config_id_from_yaml_text
except ImportError:  # pragma: no cover
    from config_id import config_id_from_yaml_text  # type: ignore[no-redef]

try:
    from tools.registry_index import default_registry_db_path
except ImportError:  # pragma: no cover
    from registry_index import default_registry_db_path  # type: ignore[no-redef]


AIQ_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RestartResult:
    attempted: bool
    exit_code: int
    stdout: str
    stderr: str


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _utc_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def _load_yaml_engine_interval(yaml_text: str) -> str:
    # Best-effort parse without importing project code; YAML is expected to be small.
    try:
        import yaml  # local import to keep script dependency surface minimal

        obj = yaml.safe_load(yaml_text) or {}
        if not isinstance(obj, dict):
            return ""
        glob = obj.get("global", {}) if isinstance(obj.get("global", {}), dict) else {}
        eng = glob.get("engine", {}) if isinstance(glob.get("engine", {}), dict) else {}
        iv = eng.get("interval", "")
        return str(iv or "").strip()
    except Exception:
        return ""


def _lookup_config_yaml_text(*, registry_db: Path, config_id: str) -> str:
    con = sqlite3.connect(str(registry_db), timeout=2.0)
    try:
        row = con.execute("SELECT yaml_text FROM configs WHERE config_id = ? LIMIT 1", (str(config_id),)).fetchone()
        if not row:
            raise KeyError(f"config_id not found in registry: {config_id}")
        yaml_text = str(row[0] or "")
        if not yaml_text.strip():
            raise ValueError(f"Empty yaml_text for config_id: {config_id}")
        return yaml_text
    finally:
        con.close()


def _restart_systemd_user_service(service: str) -> RestartResult:
    service = str(service).strip()
    if not service:
        return RestartResult(attempted=False, exit_code=0, stdout="", stderr="service name is empty")

    proc = subprocess.run(
        ["systemctl", "--user", "restart", service],
        capture_output=True,
        text=True,
        check=False,
    )
    return RestartResult(
        attempted=True,
        exit_code=int(proc.returncode),
        stdout=str(proc.stdout or ""),
        stderr=str(proc.stderr or ""),
    )


def deploy_paper_config(
    *,
    config_id: str,
    artifacts_dir: Path,
    yaml_path: Path,
    out_dir: Path | None,
    reason: str,
    restart: str,
    service: str,
    dry_run: bool,
) -> Path:
    """Deploy config_id to yaml_path and return the deploy directory path."""
    config_id = str(config_id).strip()
    if not config_id:
        raise ValueError("config_id cannot be empty")

    artifacts_dir = Path(artifacts_dir).expanduser().resolve()
    yaml_path = Path(yaml_path).expanduser().resolve()
    registry_db = default_registry_db_path(artifacts_root=artifacts_dir)

    yaml_text = _lookup_config_yaml_text(registry_db=registry_db, config_id=config_id)
    computed_id = config_id_from_yaml_text(yaml_text)
    if computed_id != config_id:
        raise ValueError(f"registry yaml_text hash mismatch: expected {config_id}, got {computed_id}")

    prev_text = _read_text(yaml_path)
    prev_interval = _load_yaml_engine_interval(prev_text)
    next_interval = _load_yaml_engine_interval(yaml_text)
    restart_required = bool(prev_interval and next_interval and prev_interval != next_interval)

    ts = _utc_compact()
    short = config_id[:12]
    deploy_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else (artifacts_dir / "deployments" / "paper" / f"{ts}_{short}").resolve()
    )
    deploy_dir.mkdir(parents=True, exist_ok=True)

    event: dict[str, Any] = {
        "version": "deploy_event_v1",
        "ts_utc": _utc_now_iso(),
        "ts_compact_utc": ts,
        "who": {
            "user": getpass.getuser(),
            "hostname": socket.gethostname(),
        },
        "what": {
            "mode": "paper",
            "config_id": config_id,
            "registry_db": str(registry_db),
            "yaml_path": str(yaml_path),
            "prev_yaml_sha256": "",
            "next_yaml_sha256": "",
            "prev_engine_interval": prev_interval,
            "next_engine_interval": next_interval,
            "restart_required": bool(restart_required),
        },
        "why": {
            "reason": str(reason or "").strip(),
        },
        "dry_run": bool(dry_run),
        "restart": {
            "mode": str(restart),
            "service": str(service),
            "result": None,
        },
    }

    import hashlib

    event["what"]["prev_yaml_sha256"] = hashlib.sha256(prev_text.encode("utf-8")).hexdigest() if prev_text else ""
    event["what"]["next_yaml_sha256"] = hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()

    # Always write the artefact, even in dry-run mode.
    (deploy_dir / "deployed_config.yaml").write_text(yaml_text, encoding="utf-8")

    if not dry_run:
        _atomic_write_text(yaml_path, yaml_text)

    restart_mode = str(restart or "auto").strip().lower()
    do_restart = (restart_mode == "always") or (restart_mode == "auto" and restart_required)
    if do_restart and not dry_run:
        rr = _restart_systemd_user_service(str(service))
        event["restart"]["result"] = rr.__dict__
        if rr.attempted and rr.exit_code != 0:
            # Record the event and fail fast to avoid silent deploys without a functioning engine.
            (deploy_dir / "deploy_event.json").write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise RuntimeError(f"service restart failed: {service} (exit_code={rr.exit_code})")

    (deploy_dir / "deploy_event.json").write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return deploy_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deploy a config_id to the paper trader safely.")
    ap.add_argument("--config-id", required=True, help="config_id from the factory registry.")
    ap.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root (default: artifacts).")
    ap.add_argument(
        "--yaml-path",
        default=str(AIQ_ROOT / "config" / "strategy_overrides.yaml"),
        help="Target strategy overrides YAML path (default: config/strategy_overrides.yaml).",
    )
    ap.add_argument("--out-dir", default="", help="Deploy artefact output directory (default: artifacts/deployments/paper/...).")
    ap.add_argument("--reason", default="", help="Human reason for this deployment (recorded).")
    ap.add_argument(
        "--restart",
        default="auto",
        choices=["auto", "always", "never"],
        help="Restart policy (default: auto).",
    )
    ap.add_argument(
        "--service",
        default="openclaw-ai-quant-trader",
        help="systemd user service name for paper trader (default: openclaw-ai-quant-trader).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Write artefacts but do not modify the YAML or restart.")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else None
    try:
        deploy_paper_config(
            config_id=str(args.config_id),
            artifacts_dir=Path(args.artifacts_dir),
            yaml_path=Path(args.yaml_path),
            out_dir=out_dir,
            reason=str(args.reason),
            restart=str(args.restart),
            service=str(args.service),
            dry_run=bool(args.dry_run),
        )
        return 0
    except KeyError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    raise SystemExit(main())

