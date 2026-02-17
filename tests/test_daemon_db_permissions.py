from __future__ import annotations

import os
import stat
from pathlib import Path

import engine.daemon as daemon


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_harden_db_permissions_sets_mode_600(tmp_path: Path) -> None:
    root_db = tmp_path / "root.db"
    nested_db = tmp_path / "nested" / "extra.db"
    nested_db.parent.mkdir(parents=True, exist_ok=True)
    root_db.write_text("", encoding="utf-8")
    nested_db.write_text("", encoding="utf-8")

    os.chmod(root_db, 0o644)
    os.chmod(nested_db, 0o644)

    daemon._harden_db_permissions(str(nested_db), project_root=tmp_path)

    assert _mode(root_db) == 0o600
    assert _mode(nested_db) == 0o600


def test_harden_db_permissions_ignores_missing_paths(tmp_path: Path) -> None:
    daemon._harden_db_permissions(str(tmp_path / "missing.db"), project_root=tmp_path)
