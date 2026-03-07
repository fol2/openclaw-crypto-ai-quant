from __future__ import annotations

import copy
import functools
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from engine.promoted_config import resolve_effective_config
from engine.strategy_manager import StrategyManager
from tools.config_id import config_id_from_yaml_file


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(payload, default_flow_style=False, sort_keys=False), encoding="utf-8")


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _apply_mode_overlay(base: dict, mode_key: str | None) -> dict:
    output = copy.deepcopy(base)
    if not mode_key:
        return output

    modes = output.get("modes") if isinstance(output.get("modes"), dict) else {}
    if not isinstance(modes, dict):
        return output

    overlay = modes.get(mode_key) or modes.get(str(mode_key).upper()) or modes.get(str(mode_key).lower())
    if not isinstance(overlay, dict):
        raise KeyError(f"strategy mode not found in fixture: {mode_key}")

    output.setdefault("global", {})
    output.setdefault("symbols", {})

    if "global" in overlay or "symbols" in overlay:
        if isinstance(overlay.get("global"), dict):
            _deep_merge(output["global"], overlay["global"])
        if isinstance(overlay.get("symbols"), dict):
            for symbol, symbol_overlay in overlay["symbols"].items():
                if not isinstance(symbol, str) or not isinstance(symbol_overlay, dict):
                    continue
                symbol_key = symbol.strip().upper()
                output["symbols"].setdefault(symbol_key, {})
                _deep_merge(output["symbols"][symbol_key], symbol_overlay)
        return output

    _deep_merge(output["global"], overlay)
    return output


@functools.lru_cache(maxsize=1)
def _runtime_binary() -> str:
    binary = REPO_ROOT / "target" / "debug" / "aiq-runtime"
    if not binary.is_file():
        subprocess.run(["cargo", "build", "-q", "-p", "aiq-runtime"], cwd=REPO_ROOT, check=True)
    return str(binary)


def _base_fixture() -> dict:
    return {
        "global": {
            "trade": {
                "allocation_pct": 0.03,
                "leverage": 2.0,
            },
            "engine": {
                "interval": "30m",
            },
        },
        "symbols": {
            "BTC": {
                "trade": {
                    "leverage": 4.0,
                }
            }
        },
        "modes": {
            "primary": {
                "global": {
                    "engine": {
                        "interval": "5m",
                    }
                },
                "symbols": {
                    "BTC": {
                        "trade": {
                            "leverage": 6.0,
                        }
                    }
                },
            },
            "fallback": {
                "global": {
                    "engine": {
                        "interval": "1h",
                    }
                },
                "symbols": {
                    "BTC": {
                        "trade": {
                            "leverage": 5.0,
                        }
                    }
                },
            },
        },
    }


def _promoted_fixture() -> dict:
    return {
        "global": {
            "trade": {
                "allocation_pct": 0.05,
            }
        },
        "symbols": {
            "BTC": {
                "trade": {
                    "leverage": 8.0,
                }
            }
        },
    }


@pytest.mark.parametrize(
    ("case_name", "promoted_role", "strategy_mode", "mode_source", "expected_interval", "expected_leverage"),
    [
        ("base_only", None, None, None, "30m", 4.0),
        ("promoted_only", "primary", None, None, "30m", 8.0),
        ("mode_env", None, "fallback", "env", "1h", 5.0),
        ("mode_file", None, "primary", "file", "5m", 6.0),
        ("promoted_and_mode", "primary", "fallback", "env", "1h", 5.0),
    ],
)
def test_rust_effective_config_parity_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    promoted_role: str | None,
    strategy_mode: str | None,
    mode_source: str | None,
    expected_interval: str,
    expected_leverage: float,
) -> None:
    base_config = _base_fixture()
    promoted_config = _promoted_fixture()

    base_config_path = tmp_path / "config" / "strategy.yaml"
    _write_yaml(base_config_path, base_config)

    artifacts_dir = tmp_path / "artifacts"
    promoted_dir = artifacts_dir / "2026-03-07" / "run_nightly" / "promoted_configs"
    output_root = tmp_path / "runtime-output"
    if promoted_role:
        _write_yaml(promoted_dir / f"{promoted_role}.yaml", promoted_config)

    env = {
        "AI_QUANT_RUNTIME_BIN": _runtime_binary(),
        "AI_QUANT_ARTIFACTS_DIR": str(artifacts_dir),
        "AI_QUANT_EFFECTIVE_CONFIG_OUTPUT_ROOT": str(output_root),
    }
    if promoted_role:
        env["AI_QUANT_PROMOTED_ROLE"] = promoted_role
    if strategy_mode and mode_source == "env":
        env["AI_QUANT_STRATEGY_MODE"] = strategy_mode
    if strategy_mode and mode_source == "file":
        mode_file = tmp_path / "strategy_mode.txt"
        mode_file.write_text(f"{strategy_mode}\n", encoding="utf-8")
        env["AI_QUANT_STRATEGY_MODE_FILE"] = str(mode_file)

    monkeypatch.setenv("AI_QUANT_RUNTIME_BIN", env["AI_QUANT_RUNTIME_BIN"])

    resolved = resolve_effective_config(config_path=base_config_path, env=env)

    expected_document = copy.deepcopy(base_config)
    if promoted_role:
        _deep_merge(expected_document, promoted_config)
    if strategy_mode:
        expected_document = _apply_mode_overlay(expected_document, strategy_mode)

    resolved_path = Path(resolved.config_path)
    resolved_document = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}

    assert resolved.base_config_path == str(base_config_path.resolve()), case_name
    assert resolved_document == expected_document, case_name
    assert config_id_from_yaml_file(resolved_path) == resolved.config_id, case_name

    monkeypatch.setenv("AI_QUANT_EFFECTIVE_CONFIG_OWNER", "rust")
    monkeypatch.setenv(
        "AI_QUANT_EFFECTIVE_CONFIG_MATERIALISED",
        "1" if resolved.effective_yaml_path != resolved.active_yaml_path else "0",
    )
    if resolved.strategy_mode:
        monkeypatch.setenv("AI_QUANT_STRATEGY_MODE", resolved.strategy_mode)
    else:
        monkeypatch.delenv("AI_QUANT_STRATEGY_MODE", raising=False)

    manager = StrategyManager.bootstrap(
        defaults={"trade": {}, "indicators": {}, "filters": {}, "thresholds": {}, "engine": {}},
        yaml_path=str(resolved_path),
        changelog_path=None,
    )
    cfg = manager.get_config("BTC")

    assert manager.snapshot.config_id == resolved.config_id, case_name
    assert (cfg.get("engine") or {}).get("interval") == expected_interval, case_name
    assert float((cfg.get("trade") or {}).get("leverage", 0.0)) == expected_leverage, case_name
    assert resolved.promoted_role == promoted_role, case_name
    assert resolved.strategy_mode == strategy_mode, case_name
    assert resolved.strategy_mode_source == mode_source, case_name

    if promoted_role:
        assert Path(resolved.active_yaml_path).parent == output_root / "config", case_name
        assert resolved.promoted_config_path is not None, case_name
    else:
        assert Path(resolved.active_yaml_path) == base_config_path.resolve(), case_name
        assert resolved.promoted_config_path is None, case_name

    if strategy_mode:
        assert Path(resolved.effective_yaml_path).parent == output_root / "artifacts" / "_effective_configs", case_name
        assert Path(resolved.config_path) == Path(resolved.effective_yaml_path), case_name
    else:
        assert Path(resolved.config_path) == Path(resolved.active_yaml_path), case_name
