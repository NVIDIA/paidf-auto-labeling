# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from core import ensure_scene_skeleton
from super_resolution.backends import seedvr2 as seedvr2_backend
from super_resolution.backends.seedvr2 import (
    CommandInvocation,
    CommandRunnerError,
    SeedVR2Resolver,
    prepare_seedvr_runtime_root,
    resolve_gpu_ids,
    validate_seedvr_source_root,
)
from super_resolution.config import SeedVR2Config


class _CaptureRunner:
    def __init__(self, *, create_output: bool = True) -> None:
        self.create_output = create_output
        self.invocations: list[CommandInvocation] = []

    def __call__(self, invocation: CommandInvocation) -> None:
        self.invocations.append(invocation)
        if self.create_output:
            output_path = Path(invocation.command[invocation.command.index("--output_path") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"sr")


class _FailingRunner:
    def __call__(self, invocation: CommandInvocation) -> None:
        raise CommandRunnerError(f"failed {invocation.name}")


class _BuggyRunner:
    def __call__(self, invocation: CommandInvocation) -> None:
        raise AssertionError(f"buggy {invocation.name}")


@pytest.fixture(autouse=True)
def _stub_video_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        seedvr2_backend,
        "prepare_video_decode",
        lambda _: SimpleNamespace(decoder_name="h264_cuvid"),
    )


def test_resolve_gpu_ids_parses_common_forms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("super_resolution.backends.seedvr2._all_gpu_ids", lambda: [0, 1])

    assert resolve_gpu_ids(0) == [0]
    assert resolve_gpu_ids("2, 3") == [2, 3]
    assert resolve_gpu_ids(None) == [0, 1]
    assert resolve_gpu_ids("") == [0, 1]
    assert resolve_gpu_ids(",") == [0, 1]
    with pytest.raises(ValueError, match="Invalid GPU id"):
        resolve_gpu_ids("0,nope")


def test_resolve_gpu_ids_fails_when_cuda_discovery_finds_no_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
        )
    )
    monkeypatch.setattr(seedvr2_backend.importlib, "import_module", lambda _: torch)

    with pytest.raises(RuntimeError, match="No CUDA devices"):
        resolve_gpu_ids(None)
    with pytest.raises(RuntimeError, match="No CUDA devices"):
        resolve_gpu_ids("all")
    with pytest.raises(RuntimeError, match="No CUDA devices"):
        resolve_gpu_ids("")


def test_seedvr2_subprocess_runner_requires_torchrun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("super_resolution.backends.seedvr2.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="requires torchrun"):
        SeedVR2Resolver(
            logger=logging.getLogger("test_seedvr2_missing_torchrun"),
            config=SeedVR2Config(
                seedvr_root=str(_seedvr_root(tmp_path)),
                model_cache_path=str(_cache_root(tmp_path)),
                gpu_ids="0",
            ),
        )


@pytest.mark.parametrize("decoder_name", ["h264_cuvid", "mpeg4"])
def test_seedvr2_builds_torchrun_command_with_explicit_seedvr_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decoder_name: str,
) -> None:
    monkeypatch.setattr(
        seedvr2_backend,
        "prepare_video_decode",
        lambda _: SimpleNamespace(decoder_name=decoder_name),
    )
    seedvr_root = _seedvr_root(tmp_path)
    cache_root = _cache_root(tmp_path)
    runner = _CaptureRunner()
    resolver = SeedVR2Resolver(
        logger=logging.getLogger("test_seedvr2"),
        config=SeedVR2Config(
            seedvr_root=str(seedvr_root),
            model_cache_path=str(cache_root),
            gpu_ids="2,3",
            keep_intermediates=True,
        ),
        command_runner=runner,
    )
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    scene_paths = ensure_scene_skeleton(tmp_path / "scene")

    result = resolver.run(media, scene_paths)

    assert result.success is True
    assert result.output_path == scene_paths.sidecars_dir / "sr_output.mp4"
    assert len(runner.invocations) == 1
    invocation = runner.invocations[0]
    assert invocation.command[0] == "torchrun"
    assert "--module" in invocation.command
    assert invocation.command[invocation.command.index("--module") + 1] == (
        "super_resolution.seedvr2_torchrun"
    )
    assert invocation.command[invocation.command.index("--video_path") + 1] == str(media)
    assert invocation.command[invocation.command.index("--output_path") + 1] == str(
        result.output_path
    )
    assert invocation.command[invocation.command.index("--sp_size") + 1] == "1"
    assert "--nproc-per-node=1" in invocation.command
    assert invocation.command[invocation.command.index("--video_decoder") + 1] == decoder_name
    assert invocation.env["CUDA_VISIBLE_DEVICES"] == "2"
    runtime_root = cache_root / "seedvr_runtime"
    assert invocation.cwd == runtime_root.resolve()
    assert invocation.env["SEEDVR_ROOT"] == str(runtime_root.resolve())
    assert invocation.env["SEEDVR_CKPT_DIR"] == str((cache_root / "seedvr2").resolve())
    assert str(runtime_root.resolve()) in invocation.env["PYTHONPATH"].split(":")
    assert (runtime_root / "projects").resolve() == (seedvr_root / "projects").resolve()
    assert (runtime_root / "ckpts").resolve() == (cache_root / "seedvr2").resolve()
    assert invocation.env["USER"]
    assert invocation.env["LOGNAME"] == invocation.env["USER"]
    assert invocation.env["HOME"].endswith("_runtime_cache/home")
    assert invocation.env["MPLCONFIGDIR"].endswith("_runtime_cache/xdg/matplotlib")
    assert invocation.env["TORCHINDUCTOR_CACHE_DIR"].endswith("_runtime_cache/xdg/torchinductor")


def test_seedvr2_multi_gpu_video_uses_all_selected_gpus(tmp_path: Path) -> None:
    seedvr_root = _seedvr_root(tmp_path)
    cache_root = _cache_root(tmp_path)
    runner = _CaptureRunner()
    resolver = SeedVR2Resolver(
        logger=logging.getLogger("test_seedvr2_multi"),
        config=SeedVR2Config(
            seedvr_root=str(seedvr_root),
            model_cache_path=str(cache_root),
            gpu_ids="0,1,2",
            use_multi_gpu=True,
        ),
        command_runner=runner,
    )
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")

    resolver.run(media, ensure_scene_skeleton(tmp_path / "scene"))

    invocation = runner.invocations[0]
    assert invocation.env["CUDA_VISIBLE_DEVICES"] == "0,1,2"
    assert invocation.command[invocation.command.index("--sp_size") + 1] == "3"
    assert "--nproc-per-node=3" in invocation.command


def test_seedvr2_mov_video_uses_mp4_output_suffix(tmp_path: Path) -> None:
    seedvr_root = _seedvr_root(tmp_path)
    cache_root = _cache_root(tmp_path)
    runner = _CaptureRunner()
    resolver = SeedVR2Resolver(
        logger=logging.getLogger("test_seedvr2_mov"),
        config=SeedVR2Config(
            seedvr_root=str(seedvr_root),
            model_cache_path=str(cache_root),
            gpu_ids="0",
        ),
        command_runner=runner,
    )
    media = tmp_path / "clip.mov"
    media.write_bytes(b"video")

    result = resolver.run(media, ensure_scene_skeleton(tmp_path / "scene"))

    assert result.output_path == tmp_path / "scene" / "sidecars" / "sr_output.mp4"
    invocation = runner.invocations[0]
    assert invocation.command[invocation.command.index("--output_path") + 1] == str(
        result.output_path
    )


def test_seedvr2_rejects_unsupported_video_extension(tmp_path: Path) -> None:
    seedvr_root = _seedvr_root(tmp_path)
    cache_root = _cache_root(tmp_path)
    resolver = SeedVR2Resolver(
        logger=logging.getLogger("test_seedvr2_unsupported_extension"),
        config=SeedVR2Config(
            seedvr_root=str(seedvr_root),
            model_cache_path=str(cache_root),
            gpu_ids="0",
        ),
        command_runner=_CaptureRunner(),
    )
    media = tmp_path / "clip.mkv"
    media.write_bytes(b"video")

    with pytest.raises(ValueError, match="Unsupported super-resolution media extension '.mkv'"):
        resolver.run(media, ensure_scene_skeleton(tmp_path / "scene"))


def test_seedvr2_image_forces_single_process_gpu(tmp_path: Path) -> None:
    seedvr_root = _seedvr_root(tmp_path)
    cache_root = _cache_root(tmp_path)
    runner = _CaptureRunner()
    resolver = SeedVR2Resolver(
        logger=logging.getLogger("test_seedvr2_image"),
        config=SeedVR2Config(
            seedvr_root=str(seedvr_root),
            model_cache_path=str(cache_root),
            gpu_ids="0,1",
            use_multi_gpu=True,
        ),
        command_runner=runner,
    )
    media = tmp_path / "frame.png"
    media.write_bytes(b"image")

    result = resolver.run(media, ensure_scene_skeleton(tmp_path / "scene"))

    invocation = runner.invocations[0]
    assert result.output_path == tmp_path / "scene" / "sidecars" / "sr_output.png"
    assert invocation.env["CUDA_VISIBLE_DEVICES"] == "0"
    assert invocation.command[invocation.command.index("--sp_size") + 1] == "1"
    assert "--nproc-per-node=1" in invocation.command


def test_seedvr2_missing_output_warns_or_raises(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    seedvr_root = _seedvr_root(tmp_path)
    cache_root = _cache_root(tmp_path)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    warn_logger = logging.getLogger("test_seedvr2_warn")
    caplog.set_level(logging.WARNING, logger=warn_logger.name)

    warn_resolver = SeedVR2Resolver(
        logger=warn_logger,
        config=SeedVR2Config(
            seedvr_root=str(seedvr_root),
            model_cache_path=str(cache_root),
            gpu_ids="0",
        ),
        command_runner=_CaptureRunner(create_output=False),
    )
    warn_result = warn_resolver.run(media, ensure_scene_skeleton(tmp_path / "warn"))
    assert warn_result.success is False
    assert warn_result.output_path is None
    assert any(
        record.name == warn_logger.name
        and record.levelno == logging.WARNING
        and "output was not created" in record.getMessage()
        for record in caplog.records
    )

    fail_resolver = SeedVR2Resolver(
        logger=logging.getLogger("test_seedvr2_fail"),
        config=SeedVR2Config(
            seedvr_root=str(seedvr_root),
            model_cache_path=str(cache_root),
            gpu_ids="0",
            empty_output_policy="fail",
        ),
        command_runner=_CaptureRunner(create_output=False),
    )
    with pytest.raises(RuntimeError, match="output was not created"):
        fail_resolver.run(media, ensure_scene_skeleton(tmp_path / "fail"))


def test_seedvr2_removes_stale_output_before_run(tmp_path: Path) -> None:
    seedvr_root = _seedvr_root(tmp_path)
    cache_root = _cache_root(tmp_path)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    scene_paths = ensure_scene_skeleton(tmp_path / "scene")
    stale_output = scene_paths.sidecars_dir / "sr_output.mp4"
    stale_output.write_bytes(b"old")

    resolver = SeedVR2Resolver(
        logger=logging.getLogger("test_seedvr2_stale_output"),
        config=SeedVR2Config(
            seedvr_root=str(seedvr_root),
            model_cache_path=str(cache_root),
            gpu_ids="0",
        ),
        command_runner=_CaptureRunner(create_output=False),
    )

    result = resolver.run(media, scene_paths)

    assert result.success is False
    assert not stale_output.exists()


def test_seedvr2_cleans_work_dir_after_command_failure(tmp_path: Path) -> None:
    seedvr_root = _seedvr_root(tmp_path)
    cache_root = _cache_root(tmp_path)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    scene_paths = ensure_scene_skeleton(tmp_path / "scene")

    resolver = SeedVR2Resolver(
        logger=logging.getLogger("test_seedvr2_command_failure_cleanup"),
        config=SeedVR2Config(
            seedvr_root=str(seedvr_root),
            model_cache_path=str(cache_root),
            gpu_ids="0",
        ),
        command_runner=_FailingRunner(),
    )

    result = resolver.run(media, scene_paths)

    assert result.success is False
    assert not (scene_paths.sidecars_dir / "_work" / "sr").exists()


def test_seedvr2_keeps_work_dir_after_failure_when_configured(tmp_path: Path) -> None:
    seedvr_root = _seedvr_root(tmp_path)
    cache_root = _cache_root(tmp_path)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    scene_paths = ensure_scene_skeleton(tmp_path / "scene")

    resolver = SeedVR2Resolver(
        logger=logging.getLogger("test_seedvr2_command_failure_keep_work"),
        config=SeedVR2Config(
            seedvr_root=str(seedvr_root),
            model_cache_path=str(cache_root),
            gpu_ids="0",
            keep_intermediates=True,
        ),
        command_runner=_FailingRunner(),
    )

    result = resolver.run(media, scene_paths)

    assert result.success is False
    assert (scene_paths.sidecars_dir / "_work" / "sr").exists()


def test_seedvr2_unexpected_runner_exception_bubbles(tmp_path: Path) -> None:
    seedvr_root = _seedvr_root(tmp_path)
    cache_root = _cache_root(tmp_path)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    scene_paths = ensure_scene_skeleton(tmp_path / "scene")

    resolver = SeedVR2Resolver(
        logger=logging.getLogger("test_seedvr2_unexpected_runner_exception"),
        config=SeedVR2Config(
            seedvr_root=str(seedvr_root),
            model_cache_path=str(cache_root),
            gpu_ids="0",
        ),
        command_runner=_BuggyRunner(),
    )

    with pytest.raises(AssertionError, match="buggy sr"):
        resolver.run(media, scene_paths)
    assert not (scene_paths.sidecars_dir / "_work" / "sr").exists()


def test_validate_seedvr_source_root_requires_seedvr_layout(tmp_path: Path) -> None:
    source_root = _seedvr_root(tmp_path)
    assert validate_seedvr_source_root(source_root=source_root) == source_root.resolve()

    missing_projects = tmp_path / "missing_projects"
    shutil.copytree(source_root, missing_projects)
    shutil.rmtree(missing_projects / "projects")

    with pytest.raises(RuntimeError, match="missing required path"):
        validate_seedvr_source_root(source_root=missing_projects)


def test_prepare_seedvr_runtime_root_symlinks_source_and_checkpoints(tmp_path: Path) -> None:
    source_root = _seedvr_root(tmp_path)
    cache_root = _cache_root(tmp_path)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "stale_file.py").write_text("old import target", encoding="utf-8")
    (runtime_root / "stale_dir").mkdir()
    (runtime_root / "stale_dir" / "module.py").write_text("old module", encoding="utf-8")
    (runtime_root / "stale_link").symlink_to(source_root / "common")

    prepared = prepare_seedvr_runtime_root(
        source_root=source_root,
        runtime_root=runtime_root,
        ckpts_dir=cache_root / "seedvr2",
        logger=logging.getLogger("test_seedvr_runtime_root"),
    )

    assert prepared == runtime_root.resolve()
    assert (runtime_root / "common").resolve() == (source_root / "common").resolve()
    assert (runtime_root / "projects").resolve() == (source_root / "projects").resolve()
    assert (runtime_root / "pos_emb.pt").resolve() == (source_root / "pos_emb.pt").resolve()
    assert (runtime_root / "ckpts").resolve() == (cache_root / "seedvr2").resolve()
    assert not (runtime_root / "stale_file.py").exists()
    assert not (runtime_root / "stale_dir").exists()
    assert not (runtime_root / "stale_link").exists()


def test_prepare_seedvr_runtime_root_tolerates_concurrent_stale_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = _seedvr_root(tmp_path)
    cache_root = _cache_root(tmp_path)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    stale_file = runtime_root / "stale_file.py"
    stale_file.write_text("old import target", encoding="utf-8")
    original_unlink = Path.unlink

    def unlink_with_concurrent_removal(path: Path, missing_ok: bool = False) -> None:
        original_unlink(path, missing_ok=missing_ok)
        if path == stale_file:
            raise FileNotFoundError(path)

    monkeypatch.setattr(Path, "unlink", unlink_with_concurrent_removal)

    prepare_seedvr_runtime_root(
        source_root=source_root,
        runtime_root=runtime_root,
        ckpts_dir=cache_root / "seedvr2",
        logger=logging.getLogger("test_seedvr_runtime_root_concurrent_cleanup"),
    )

    assert not stale_file.exists()


def test_prepare_seedvr_runtime_root_tolerates_concurrent_symlink_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = _seedvr_root(tmp_path)
    cache_root = _cache_root(tmp_path)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    common_link = runtime_root / "common"
    original_symlink_to = Path.symlink_to
    injected = False

    def symlink_to_with_concurrent_creation(
        path: Path,
        target: str | Path,
        target_is_directory: bool = False,
    ) -> None:
        nonlocal injected
        if path == common_link and not injected:
            injected = True
            original_symlink_to(path, target, target_is_directory=target_is_directory)
            raise FileExistsError(path)
        original_symlink_to(path, target, target_is_directory=target_is_directory)

    monkeypatch.setattr(Path, "symlink_to", symlink_to_with_concurrent_creation)

    prepare_seedvr_runtime_root(
        source_root=source_root,
        runtime_root=runtime_root,
        ckpts_dir=cache_root / "seedvr2",
        logger=logging.getLogger("test_seedvr_runtime_root_concurrent_symlink"),
    )

    assert common_link.resolve() == (source_root / "common").resolve()


def _seedvr_root(tmp_path: Path) -> Path:
    root = tmp_path / "seedvr"
    for name in (
        "common",
        "configs_3b",
        "configs_7b",
        "data",
        "models",
        "projects",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "pos_emb.pt").write_bytes(b"pos")
    (root / "neg_emb.pt").write_bytes(b"neg")
    return root


def _cache_root(tmp_path: Path) -> Path:
    root = tmp_path / "cache"
    ckpts = root / "seedvr2"
    ckpts.mkdir(parents=True, exist_ok=True)
    (ckpts / "ema_vae.pth").write_bytes(b"vae")
    (ckpts / "seedvr2_ema_3b.pth").write_bytes(b"model")
    return root
