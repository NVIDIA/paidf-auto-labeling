# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cli
import pytest


@pytest.fixture
def mock_dependencies(tmp_path):
    with (
        patch("cli.load_config_with_overrides") as load_cfg,
        patch("cli.validate_schema") as val_schema,
        patch("cli.run_pipeline") as run_pipe,
        patch("cli.setup_msc_config"),
        patch("cli.NVCFProgressTracker"),
        patch("cli.localize_path_to_dir") as localize,
        patch("cli.EndpointResolver"),
    ):
        # Default behaviors
        load_cfg.return_value = ({}, Path("."))
        run_pipe.return_value = 0
        localize.return_value = Path("local/video.mp4")

        # Mock validated schema structure
        mock_sample = MagicMock()
        mock_sample.inputs.video_path = "video.mp4"
        mock_sample.inputs.vlm_video_path = None
        mock_sample.inputs.metadata_json_path = None

        # model_copy should return a copy or self
        mock_sample.inputs.model_copy.return_value = mock_sample.inputs
        mock_sample.model_copy.return_value = mock_sample
        mock_sample.model_dump.return_value = {}

        mock_sample.output.out_dir = str(tmp_path / "out_dir")
        mock_sample.output.log_dir = None
        mock_sample.output.config_path = None

        mock_validated = MagicMock()
        mock_validated.data = [mock_sample]
        mock_validated.endpoints = None
        mock_validated.model_dump.return_value = {}
        # Disable all stage configs so no factories are called.
        mock_validated.super_resolution = None
        mock_validated.detection_and_tracking = None
        mock_validated.vlm_json = None
        mock_validated.mcq_generation = None

        val_schema.return_value = mock_validated

        yield {
            "load_cfg": load_cfg,
            "val_schema": val_schema,
            "run_pipe": run_pipe,
            "mock_validated": mock_validated,
            "mock_sample": mock_sample,
        }


def test_cli_main_success(mock_dependencies):
    argv = ["--config", "config.yaml"]
    rc = cli.main(argv)
    assert rc == 0
    mock_dependencies["run_pipe"].assert_called_once()


def test_cli_main_dry_run(mock_dependencies):
    argv = ["--config", "config.yaml", "--dry-run"]

    # In dry-run, run_pipeline IS called (with dry_run=True)
    # But _localize_into_dir is NOT called (filesystem logic skipped)
    with patch("cli.localize_path_to_dir") as localize:
        rc = cli.main(argv)
        assert rc == 0
        mock_dependencies["run_pipe"].assert_called_once()
        _, kwargs = mock_dependencies["run_pipe"].call_args
        assert kwargs["dry_run"] is True

        localize.assert_not_called()


def test_cli_dotlist_window_timeout_reaches_sr_runner(tmp_path):
    """Real CLI config loading applies super_resolution.window_timeout before SR init."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "pipeline_example.yaml"
    if not config_path.exists():
        pytest.skip(f"Config not found: {config_path}")

    captured: dict[str, int] = {}
    fake_sr_runner = MagicMock(name="sr_runner")

    def fake_create_sr_runner(validated, logger):  # noqa: ARG001
        captured["window_timeout"] = int(validated.super_resolution.window_timeout)
        return fake_sr_runner

    argv = [
        "--config",
        str(config_path),
        "--dry-run",
        f"data.0.inputs.video_path={tmp_path / 'video.mp4'}",
        f"data.0.output.out_dir={tmp_path / 'out'}",
        "pipeline.daft_validate=false",
        "super_resolution.enabled=true",
        "super_resolution.window_timeout=123",
        "detection_and_tracking.enabled=false",
        "vlm_json.enabled=false",
        "mcq_generation.enabled=false",
        "endpoints.vlm.url=http://127.0.0.1:1/v1",
        "endpoints.vlm.model=dummy-vlm",
        "endpoints.llm.url=http://127.0.0.1:1/v1",
        "endpoints.llm.model=dummy-llm",
    ]

    with (
        patch("cli.setup_msc_config"),
        patch("cli.NVCFProgressTracker", return_value=MagicMock(is_nvcf=False)),
        patch("cli.create_sr_runner", side_effect=fake_create_sr_runner),
        patch("cli.run_pipeline", return_value=0) as run_pipe,
    ):
        rc = cli.main(argv)

    assert rc == 0
    assert captured["window_timeout"] == 123
    _, kwargs = run_pipe.call_args
    assert kwargs["sr_runner"] is fake_sr_runner
    assert kwargs["dry_run"] is True


def test_cli_remote_staging_copies_raw_media(mock_dependencies):
    mock_dependencies["mock_sample"].inputs.video_path = "s3://bucket/video.mp4"

    rc = cli.main(["--config", "config.yaml"])

    assert rc == 0
    mock_dependencies["run_pipe"].assert_called_once()
    _, kwargs = mock_dependencies["run_pipe"].call_args
    assert kwargs["copy_raw"] is True


def test_cli_config_load_failure(mock_dependencies):
    # If load_config raises, main should catch or fail?
    # cli.py doesn't wrap load_config in try-except for loading errors specifically,
    # but the main try-except block at line 356 catches all Exceptions.

    mock_dependencies["load_cfg"].side_effect = RuntimeError("Load failed")

    argv = ["--config", "bad.yaml"]
    rc = cli.main(argv)
    assert rc == 1  # Exception caught, returns 1


def test_cli_schema_validation_failure(mock_dependencies):
    # validate_schema returns None on failure
    mock_dependencies["val_schema"].return_value = None

    argv = ["--config", "config.yaml"]
    rc = cli.main(argv)
    assert rc == 2  # Specific error code for validation fail


def test_cli_pipeline_failure(mock_dependencies):
    # run_pipeline returns non-zero
    mock_dependencies["run_pipe"].return_value = 1

    argv = ["--config", "config.yaml"]
    rc = cli.main(argv)
    assert rc == 1


def test_cli_pipeline_exception(mock_dependencies):
    # run_pipeline raises Exception
    mock_dependencies["run_pipe"].side_effect = RuntimeError("Pipeline boom")

    argv = ["--config", "config.yaml"]
    rc = cli.main(argv)
    assert rc == 1


def test_cli_endpoint_fallbacks(mock_dependencies, monkeypatch):
    # With env vars set and all stages disabled (fixture default), pipeline should succeed.
    monkeypatch.setenv("VLM_BASE_URL", "env_vlm_url")
    monkeypatch.setenv("VLM_MODEL", "env_vlm_model")

    rc = cli.main(["--config", "c.yaml"])
    # All stages are None (from mock_validated) so pipeline succeeds.
    assert rc == 0
    mock_dependencies["run_pipe"].assert_called_once()


def test_localize_into_dir_logic(tmp_path):
    # Unit test for helper _localize_into_dir
    # Case 1: local file exists
    src = tmp_path / "src.mp4"
    src.touch()
    dst_dir = tmp_path / "dst"

    out = cli.localize_path_to_dir(
        str(src),
        dst_dir=dst_dir,
        logger=MagicMock(),
        extensions=(".mp4",),
        config_dir=tmp_path,
        repo_root=tmp_path,
    )
    assert out.exists()
    assert out.resolve() == src.resolve()


def test_localize_into_dir_resolves_repo_root_first(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_dir = tmp_path / "configs"
    repo_root.mkdir()
    config_dir.mkdir()

    # Same relative path exists in BOTH places → prefer repo_root.
    (repo_root / "x.mp4").touch()
    (config_dir / "x.mp4").touch()

    out = cli.localize_path_to_dir(
        "x.mp4",
        dst_dir=tmp_path / "dst",
        logger=MagicMock(),
        extensions=(".mp4",),
        config_dir=config_dir,
        repo_root=repo_root,
    )
    assert out.resolve() == (repo_root / "x.mp4").resolve()


def test_localize_into_dir_falls_back_to_config_dir(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_dir = tmp_path / "configs"
    repo_root.mkdir()
    config_dir.mkdir()

    # Missing under repo_root, present under config_dir → fallback.
    (config_dir / "x.mp4").touch()

    out = cli.localize_path_to_dir(
        "x.mp4",
        dst_dir=tmp_path / "dst",
        logger=MagicMock(),
        extensions=(".mp4",),
        config_dir=config_dir,
        repo_root=repo_root,
    )
    assert out.resolve() == (config_dir / "x.mp4").resolve()


def test_localize_into_dir_dot_slash_forces_config_dir(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_dir = tmp_path / "configs"
    repo_root.mkdir()
    config_dir.mkdir()

    (repo_root / "x.mp4").touch()
    (config_dir / "x.mp4").touch()

    out = cli.localize_path_to_dir(
        "./x.mp4",
        dst_dir=tmp_path / "dst",
        logger=MagicMock(),
        extensions=(".mp4",),
        config_dir=config_dir,
        repo_root=repo_root,
    )
    assert out.resolve() == (config_dir / "x.mp4").resolve()


def test_localize_into_dir_remote(tmp_path):
    # Case 2: remote path triggers sync
    dst_dir = tmp_path / "dst"
    with patch("nvcf_msc_utils.sync_remote_to_local") as sync:
        # Simulate sync creating the file
        def side_effect(**kwargs):
            (dst_dir / "remote.mp4").touch()

        sync.side_effect = side_effect

        out = cli.localize_path_to_dir(
            "s3://bucket/remote.mp4",
            dst_dir=dst_dir,
            logger=MagicMock(),
            extensions=(".mp4",),
            config_dir=tmp_path,
            repo_root=tmp_path,
        )
        assert out.name == "remote.mp4"
        sync.assert_called_once()


def test_localize_into_dir_remote_requires_expected_filename(tmp_path):
    dst_dir = tmp_path / "dst"
    with patch("nvcf_msc_utils.sync_remote_to_local") as sync:
        # Simulate remote download producing a different filename.
        def side_effect(**kwargs):
            (dst_dir / "downloaded.mp4").touch()

        sync.side_effect = side_effect
        with pytest.raises(RuntimeError) as exc:
            cli.localize_path_to_dir(
                "s3://bucket/remote.mp4",
                dst_dir=dst_dir,
                logger=MagicMock(),
                extensions=(".mp4",),
                config_dir=tmp_path,
                repo_root=tmp_path,
            )
        assert "expected filename" in str(exc.value)


def test_cli_remote_upload(mock_dependencies):
    mock_sample = mock_dependencies["mock_sample"]
    # Set remote output dir
    mock_sample.output.out_dir = "s3://bucket/out"

    # Mock sync_local_to_remote to verify it is called
    with patch("cli.sync_local_to_remote") as sync_up:
        cli.main(["--config", "c.yaml"])

        # It should be called for out_dir
        # sync_local_to_remote(local_dir=..., remote_path='s3://bucket/out/', verbose=True)
        assert sync_up.called
        # Check call arguments
        # We might have multiple calls if log_dir is also remote, but here only out_dir

        # Verify the remote path argument in the call
        # call_args_list or any_call
        found = False
        for call_args in sync_up.call_args_list:
            # call(local_dir=..., remote_path=..., verbose=...)
            # kwargs might be used
            remote_path = call_args.kwargs.get("remote_path")
            if remote_path and remote_path.startswith("s3://bucket/out"):
                found = True
                break
        assert found, "sync_local_to_remote not called with correct remote path"


def test_cli_remote_out_dir_does_not_upload_colocated_logs_twice(mock_dependencies):
    mock_sample = mock_dependencies["mock_sample"]
    mock_sample.output.out_dir = "s3://bucket/out/"
    mock_sample.output.log_dir = "s3://bucket/out//logs"

    with patch("cli.sync_local_to_remote") as sync_up:
        rc = cli.main(["--config", "c.yaml"])

    assert rc == 0
    remote_paths = [c.kwargs.get("remote_path") for c in sync_up.call_args_list]
    assert remote_paths == ["s3://bucket/out/"]


def test_cli_uploads_separate_remote_log_dir(mock_dependencies):
    mock_sample = mock_dependencies["mock_sample"]
    mock_sample.output.out_dir = "s3://bucket/out"
    mock_sample.output.log_dir = "s3://bucket/logs"

    with patch("cli.sync_local_to_remote") as sync_up:
        rc = cli.main(["--config", "c.yaml"])

    assert rc == 0
    remote_paths = [c.kwargs.get("remote_path") for c in sync_up.call_args_list]
    assert remote_paths == ["s3://bucket/out/", "s3://bucket/logs"]


def test_run_daft_validate_skips_when_tao_daft_not_on_path(tmp_path):
    logger = MagicMock()
    with patch("cli.shutil.which", return_value=None) as which, patch("cli.subprocess.run") as run:
        cli._run_daft_validate(tmp_path, logger)
    which.assert_called_once_with("tao-daft")
    run.assert_not_called()
    logger.info.assert_called_once()
    assert "skipping validation" in logger.info.call_args.args[0]


def test_run_daft_validate_logs_ok_on_zero_exit(tmp_path):
    logger = MagicMock()
    scene = tmp_path / "scene"
    scene.mkdir()
    proc = MagicMock(returncode=0, stdout="", stderr="")
    with (
        patch("cli.shutil.which", return_value="/usr/bin/tao-daft"),
        patch("cli.subprocess.run", return_value=proc) as run,
    ):
        cli._run_daft_validate(scene, logger)
    args, kwargs = run.call_args
    cmd = args[0]
    assert cmd[:3] == ["tao-daft", "validate", "metropolis-v3.0"]
    assert "--path" in cmd and str(scene) in cmd
    assert "--raw" in cmd and "auto" in cmd
    assert "--strict" in cmd
    assert kwargs.get("check") is False
    logger.info.assert_called_once()
    logger.warning.assert_not_called()


def test_run_daft_validate_logs_warning_on_nonzero_exit(tmp_path):
    logger = MagicMock()
    scene = tmp_path / "scene"
    scene.mkdir()
    proc = MagicMock(returncode=1, stdout="some stdout", stderr="some stderr")
    with patch("cli.shutil.which", return_value="/usr/bin/tao-daft"), patch("cli.subprocess.run", return_value=proc):
        cli._run_daft_validate(scene, logger)
    logger.warning.assert_called_once()
    msg = logger.warning.call_args.args[0]
    assert "validator reported issues" in msg


def test_run_daft_validate_swallows_timeout(tmp_path):
    import subprocess

    logger = MagicMock()
    scene = tmp_path / "scene"
    scene.mkdir()
    with (
        patch("cli.shutil.which", return_value="/usr/bin/tao-daft"),
        patch("cli.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="tao-daft", timeout=1)),
    ):
        cli._run_daft_validate(scene, logger)
    logger.warning.assert_called_once()
    assert "timed out" in logger.warning.call_args.args[0]


def test_run_daft_validate_swallows_unexpected_exception(tmp_path):
    logger = MagicMock()
    scene = tmp_path / "scene"
    scene.mkdir()
    with (
        patch("cli.shutil.which", return_value="/usr/bin/tao-daft"),
        patch("cli.subprocess.run", side_effect=OSError("boom")),
    ):
        cli._run_daft_validate(scene, logger)
    logger.warning.assert_called_once()
    assert "invocation failed" in logger.warning.call_args.args[0]


def test_cli_invokes_daft_validate_when_enabled(mock_dependencies):
    mock_dependencies["mock_validated"].pipeline.daft_validate = True
    with patch("cli._run_daft_validate") as hook:
        rc = cli.main(["--config", "c.yaml"])
    assert rc == 0
    hook.assert_called_once()


def test_cli_skips_daft_validate_when_disabled(mock_dependencies):
    mock_dependencies["mock_validated"].pipeline.daft_validate = False
    with patch("cli._run_daft_validate") as hook:
        rc = cli.main(["--config", "c.yaml"])
    assert rc == 0
    hook.assert_not_called()


def test_cli_skips_daft_validate_on_dry_run(mock_dependencies):
    mock_dependencies["mock_validated"].pipeline.daft_validate = True
    with patch("cli._run_daft_validate") as hook:
        rc = cli.main(["--config", "c.yaml", "--dry-run"])
    assert rc == 0
    hook.assert_not_called()


def test_cli_skips_daft_validate_on_pipeline_failure(mock_dependencies):
    mock_dependencies["mock_validated"].pipeline.daft_validate = True
    mock_dependencies["run_pipe"].return_value = 1
    with patch("cli._run_daft_validate") as hook:
        rc = cli.main(["--config", "c.yaml"])
    assert rc == 1
    hook.assert_not_called()
