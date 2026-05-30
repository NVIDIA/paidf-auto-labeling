# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sr_runner/seedvr2.py::SeedVR2Resolver.

Covers: GPU list parsing, sp_size derivation, torchrun command construction,
CUDA_VISIBLE_DEVICES routing, and empty_output_policy on missing output.

These tests are CI-safe: they mock the SR subprocess and stub checkpoint bootstrap,
so they do not require a GPU, HuggingFace credentials, or pre-downloaded SeedVR2 weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from al_utils.schema.config import PipelineConfig
from al_utils.schema.data import SampleConfig, SampleInputsConfig, SampleOutputConfig
from al_utils.schema.endpoints import EndpointsConfig, LlmEndpointConfig, VlmEndpointConfig
from al_utils.schema.pipeline_settings import PipelineSettings
from al_utils.schema.sr import SuperResolutionConfig
from sr_runner.seedvr2 import SeedVR2Resolver, _prepare_seedvr_runtime_root


@pytest.fixture(autouse=True)
def _skip_seedvr_hf_ckpt_download() -> None:
    """Skip checkpoint bootstrap; these unit tests only verify resolver behavior."""
    with patch("sr_runner.seedvr2.ensure_seedvr2_ckpts"):
        yield


def _config(
    *,
    use_multi_gpu: bool = False,
    empty_output_policy: str = "warn",
    window_timeout: int = 3600,
) -> PipelineConfig:
    return PipelineConfig(
        pipeline=PipelineSettings(
            use_multi_gpu=use_multi_gpu,
            empty_output_policy=empty_output_policy,
        ),
        data=[
            SampleConfig(
                inputs=SampleInputsConfig(video_path="v.mp4"),
                output=SampleOutputConfig(out_dir="out"),
            )
        ],
        endpoints=EndpointsConfig(
            vlm=VlmEndpointConfig(url="http://vlm/v1", model="fake-vlm"),
            llm=LlmEndpointConfig(url="http://llm/v1", model="fake-llm"),
        ),
        super_resolution=SuperResolutionConfig(enabled=True, window_timeout=window_timeout),
    )


def _resolver(gpu_list: list[int] | None = None, use_multi_gpu: bool = False) -> SeedVR2Resolver:
    return SeedVR2Resolver(
        config=_config(use_multi_gpu=use_multi_gpu),
        logger=logging.getLogger("test_sr"),
        gpu_list=gpu_list if gpu_list is not None else [0],
    )


# ---------------------------------------------------------------------------
# GPU list parsing
# ---------------------------------------------------------------------------


def test_explicit_gpu_list_parsed() -> None:
    r = _resolver(gpu_list=[0, 1, 2])
    # Single-GPU mode by default: gpu_list trimmed to first GPU
    assert r._gpu_list == [0]


def test_single_gpu_parsed() -> None:
    r = _resolver(gpu_list=[2])
    assert r._gpu_list == [2]
    assert r._sp_size == 1


# ---------------------------------------------------------------------------
# sp_size derivation
# ---------------------------------------------------------------------------


def test_single_gpu_mode_uses_first_gpu() -> None:
    r = _resolver(gpu_list=[2, 3], use_multi_gpu=False)
    assert r._sp_size == 1
    assert r._gpu_list == [2]


def test_multi_gpu_mode_uses_all_gpus() -> None:
    r = _resolver(gpu_list=[0, 1, 2], use_multi_gpu=True)
    assert r._sp_size == 3
    assert r._gpu_list == [0, 1, 2]


def test_multi_gpu_single_id_sp_size_one() -> None:
    r = _resolver(gpu_list=[3], use_multi_gpu=True)
    assert r._sp_size == 1
    assert r._gpu_list == [3]


# ---------------------------------------------------------------------------
# torchrun command construction
# ---------------------------------------------------------------------------


def test_run_builds_torchrun_command(tmp_path: Path) -> None:
    """run() must build a torchrun command with the right flags."""
    r = _resolver(gpu_list=[1])
    (tmp_path / "out.mp4").write_bytes(b"sr_output")  # simulate success

    captured: list = []

    def fake_run_cmd(name, cmd, **kwargs):
        captured.append(cmd)

    with patch("sr_runner.seedvr2.run_cmd", side_effect=fake_run_cmd):
        r.run(tmp_path / "in.mp4", tmp_path / "out.mp4", log_dir=tmp_path)

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[0] == "torchrun"
    assert "--module" in cmd
    assert "--video_path" in cmd
    assert str(tmp_path / "in.mp4") in cmd
    assert "--output_path" in cmd
    assert str(tmp_path / "out.mp4") in cmd
    assert "--tmp_dir" in cmd
    assert str(tmp_path / "_work" / "sr" / "_tmp_window_segments") in cmd
    assert "--output_dir" in cmd
    assert str(tmp_path / "_work" / "sr" / "out") in cmd


def test_run_sets_cuda_visible_devices_single_gpu(tmp_path: Path) -> None:
    """Single-GPU mode: CUDA_VISIBLE_DEVICES uses only the first GPU."""
    r = _resolver(gpu_list=[2, 3], use_multi_gpu=False)
    (tmp_path / "out.mp4").write_bytes(b"sr_output")

    captured_env: dict = {}

    def fake_run_cmd(name, cmd, extra_env=None, **kwargs):
        if extra_env:
            captured_env.update(extra_env)

    with patch("sr_runner.seedvr2.run_cmd", side_effect=fake_run_cmd):
        r.run(tmp_path / "in.mp4", tmp_path / "out.mp4", log_dir=tmp_path)

    assert captured_env.get("CUDA_VISIBLE_DEVICES") == "2"
    assert captured_env.get("SEEDVR_ROOT") == str(r._seedvr_root)
    assert "SEEDVR_CKPTS_DIR" not in captured_env


def test_run_sets_cuda_visible_devices_multi_gpu(tmp_path: Path) -> None:
    """Multi-GPU mode: CUDA_VISIBLE_DEVICES includes all selected GPUs."""
    r = _resolver(gpu_list=[2, 3], use_multi_gpu=True)
    (tmp_path / "out.mp4").write_bytes(b"sr_output")

    captured_env: dict = {}

    def fake_run_cmd(name, cmd, extra_env=None, **kwargs):
        if extra_env:
            captured_env.update(extra_env)

    with patch("sr_runner.seedvr2.run_cmd", side_effect=fake_run_cmd):
        r.run(tmp_path / "in.mp4", tmp_path / "out.mp4", log_dir=tmp_path)

    assert captured_env.get("CUDA_VISIBLE_DEVICES") == "2,3"
    assert captured_env.get("SEEDVR_ROOT") == str(r._seedvr_root)
    assert "SEEDVR_CKPTS_DIR" not in captured_env


def test_image_input_uses_seedvr_torchrun_with_sp_size_one(tmp_path: Path) -> None:
    """Image SR should still run the SeedVR subprocess, but force sp_size=1."""
    r = _resolver(gpu_list=[2, 3], use_multi_gpu=True)
    (tmp_path / "out.png").write_bytes(b"sr_output")

    captured: dict = {}

    def fake_run_cmd(name, cmd, extra_env=None, **kwargs):
        captured["cmd"] = cmd
        captured["extra_env"] = extra_env or {}

    with patch("sr_runner.seedvr2.run_cmd", side_effect=fake_run_cmd):
        r.run(tmp_path / "in.png", tmp_path / "out.png", log_dir=tmp_path)

    cmd = captured["cmd"]
    assert cmd[0] == "torchrun"
    assert cmd[cmd.index("--sp_size") + 1] == "1"
    assert "--nproc-per-node=1" in cmd
    assert captured["extra_env"].get("CUDA_VISIBLE_DEVICES") == "2"


def test_run_passes_configured_window_timeout_to_subprocess(tmp_path: Path) -> None:
    r = SeedVR2Resolver(
        config=_config(window_timeout=123),
        logger=logging.getLogger("test_sr_timeout"),
        gpu_list=[0],
    )
    (tmp_path / "out.mp4").write_bytes(b"sr_output")
    captured: dict = {}

    def fake_run_cmd(name, cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)

    with patch("sr_runner.seedvr2.run_cmd", side_effect=fake_run_cmd):
        r.run(tmp_path / "in.mp4", tmp_path / "out.mp4", log_dir=tmp_path)

    cmd = captured["cmd"]
    assert "--window_timeout" in cmd
    assert cmd[cmd.index("--window_timeout") + 1] == "123"


def test_seedvr_runtime_root_points_to_code_and_pipeline_cache(tmp_path: Path) -> None:
    source_root = tmp_path / "seedvr"
    ckpts_dir = tmp_path / "cache" / "seedvr2"
    runtime_root = tmp_path / "cache" / "seedvr_runtime"
    source_root.mkdir()
    (source_root / "common").mkdir()
    (source_root / "configs_3b").mkdir()
    (source_root / "configs_7b").mkdir()
    (source_root / "data").mkdir()
    (source_root / "models").mkdir()
    (source_root / "projects").mkdir()
    (source_root / "pos_emb.pt").write_bytes(b"pos")
    (source_root / "neg_emb.pt").write_bytes(b"neg")
    runtime_root.mkdir(parents=True)
    stale_target = tmp_path / "stale_target"
    stale_target.mkdir()
    (runtime_root / "stale").symlink_to(stale_target, target_is_directory=True)

    out = _prepare_seedvr_runtime_root(
        source_root=source_root,
        runtime_root=runtime_root,
        ckpts_dir=ckpts_dir,
        logger=logging.getLogger("test_sr_symlink"),
    )

    assert out == runtime_root.resolve()
    assert (runtime_root / "projects").is_symlink()
    assert (runtime_root / "projects").resolve() == (source_root / "projects").resolve()
    assert (runtime_root / "common").is_symlink()
    assert (runtime_root / "models").is_symlink()
    assert (runtime_root / "pos_emb.pt").is_symlink()
    assert (runtime_root / "ckpts").is_symlink()
    assert (runtime_root / "ckpts").resolve() == ckpts_dir.resolve()
    assert not (runtime_root / "stale").exists()


# ---------------------------------------------------------------------------
# empty_output_policy
# ---------------------------------------------------------------------------


def test_missing_output_warns_by_default(tmp_path: Path) -> None:
    """Output not created after run → warning, does not raise when policy=warn."""
    r = _resolver(gpu_list=[0])
    # Do NOT create out.mp4 — simulates SR producing no output

    with patch("sr_runner.seedvr2.run_cmd"):  # succeeds but writes nothing
        # Should not raise
        r.run(tmp_path / "in.mp4", tmp_path / "out.mp4", log_dir=tmp_path)


def test_subprocess_decode_failure_does_not_warn_before_pipeline_fallback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    r = _resolver(gpu_list=[0])

    def fake_run_cmd(name, cmd, log_dir=None, **kwargs):
        assert log_dir is not None
        (Path(log_dir) / "sr.log").write_text(
            "h264_cuvid CUDA_ERROR_NOT_SUPPORTED cuvid decode callback error\n",
            encoding="utf-8",
        )
        raise SystemExit(1)

    with caplog.at_level(logging.WARNING), patch("sr_runner.seedvr2.run_cmd", side_effect=fake_run_cmd):
        r.run(tmp_path / "in.mp4", tmp_path / "out.mp4", log_dir=tmp_path)

    assert not caplog.records


def test_subprocess_mpeg4_decode_failure_does_not_warn_before_pipeline_fallback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    r = _resolver(gpu_list=[0])

    def fake_run_cmd(name, cmd, log_dir=None, **kwargs):
        assert log_dir is not None
        (Path(log_dir) / "sr.log").write_text(
            "mpeg4 error while decoding stream #0:0: Invalid data found when processing input\n",
            encoding="utf-8",
        )
        raise SystemExit(1)

    with caplog.at_level(logging.WARNING), patch("sr_runner.seedvr2.run_cmd", side_effect=fake_run_cmd):
        r.run(tmp_path / "in.mp4", tmp_path / "out.mp4", log_dir=tmp_path)

    assert not caplog.records


def test_missing_output_raises_when_fail_policy(tmp_path: Path) -> None:
    """Output not created + empty_output_policy=fail → SystemExit raised."""
    r = SeedVR2Resolver(
        config=_config(empty_output_policy="fail"),
        logger=logging.getLogger("test_sr_fail"),
        gpu_list=[0],
    )

    with patch("sr_runner.seedvr2.run_cmd"), pytest.raises(SystemExit):
        r.run(tmp_path / "in.mp4", tmp_path / "out.mp4", log_dir=tmp_path)
