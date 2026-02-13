from __future__ import annotations

import argparse

import factory_run


def test_sweep_output_mode_uses_candidate_for_gpu_and_tpe() -> None:
    args = argparse.Namespace(gpu=True, tpe=False)
    assert factory_run._sweep_output_mode_from_args(args) == "candidate"

    args_tpe = argparse.Namespace(gpu=False, tpe=True)
    assert factory_run._sweep_output_mode_from_args(args_tpe) == "candidate"


def test_sweep_output_mode_defaults_to_full_for_cpu() -> None:
    args = argparse.Namespace(gpu=False, tpe=False)
    assert factory_run._sweep_output_mode_from_args(args) == "full"


def test_replay_metadata_attachment_records_validation_stage() -> None:
    args = argparse.Namespace(gpu=True, tpe=False, walk_forward=True, slippage_stress=False)
    summary: dict[str, object] = {}
    entry: dict[str, object] = {}
    factory_run._attach_replay_metadata(summary=summary, entry=entry, args=args, replay_stage="cpu_replay")

    assert summary["pipeline_stage"] == "candidate_validation"
    assert summary["sweep_stage"] == "gpu"
    assert summary["replay_stage"] == "cpu_replay"
    assert summary["canonical_cpu_verified"] is True
    assert entry["pipeline_stage"] == "candidate_validation"
    assert entry["sweep_stage"] == "gpu"
    assert entry["replay_stage"] == "cpu_replay"
    assert entry["canonical_cpu_verified"] is True
