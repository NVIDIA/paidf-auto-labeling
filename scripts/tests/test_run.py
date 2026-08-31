# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/run.py"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from run import (
    DefaultArgs,
    Runnable,
    choose_runnable,
    discover_runnables,
    execute_runnable,
    load_env_file,
    main,
    parse_args,
    resolve_script_args,
)


@pytest.fixture
def two_runnables() -> list[Runnable]:
    return [
        Runnable(
            target="pkg-a:main",
            project_name="pkg-a",
            script_name="main",
            entrypoint="pkg_a.main:main",
        ),
        Runnable(
            target="pkg-b:train",
            project_name="pkg-b",
            script_name="train",
            entrypoint="pkg_b.train:run",
        ),
    ]


# --- parse_args ---


def test_parse_args_no_args() -> None:
    script, args = parse_args([])
    assert script is None
    assert args == []


def test_parse_args_positional() -> None:
    script, args = parse_args(["pkg-a:main"])
    assert script == "pkg-a:main"
    assert args == []


def test_parse_args_separator_splits_passthrough() -> None:
    script, args = parse_args(["pkg-a:main", "--", "--foo", "bar"])
    assert script == "pkg-a:main"
    assert args == ["--foo", "bar"]


def test_parse_args_separator_without_script() -> None:
    script, args = parse_args(["--", "--foo", "bar"])
    assert script is None
    assert args == ["--foo", "bar"]


def test_parse_args_rejects_extra_args_without_separator() -> None:
    with pytest.raises(ValueError, match="Pass script arguments after `--`"):
        parse_args(["pkg-a:main", "--extra"])


def test_parse_args_positional_target_with_input_file_arg() -> None:
    script, args = parse_args(["pkg-a:main", "--", "--input-file", "payload.json"])
    assert script == "pkg-a:main"
    assert args == ["--input-file", "payload.json"]


# --- choose_runnable ---


def test_choose_runnable_by_name(two_runnables: list[Runnable]) -> None:
    result = choose_runnable(two_runnables, "pkg-a:main")
    assert result.target == "pkg-a:main"


def test_choose_runnable_second_entry(two_runnables: list[Runnable]) -> None:
    result = choose_runnable(two_runnables, "pkg-b:train")
    assert result.project_name == "pkg-b"
    assert result.entrypoint == "pkg_b.train:run"


def test_choose_runnable_unknown_name_raises(two_runnables: list[Runnable]) -> None:
    with pytest.raises(ValueError, match="Unknown target"):
        choose_runnable(two_runnables, "nonexistent:script")


def test_choose_runnable_no_tty_raises(two_runnables: list[Runnable]) -> None:
    with patch("sys.stdin") as mock_stdin:
        mock_stdin.isatty.return_value = False
        with pytest.raises(ValueError, match="No target was specified"):
            choose_runnable(two_runnables, None)


def test_choose_runnable_interactive_selection(two_runnables: list[Runnable]) -> None:
    with patch("sys.stdin") as mock_stdin, patch("questionary.select") as mock_select:
        mock_stdin.isatty.return_value = True
        mock_select.return_value.ask.return_value = "pkg-b:train"
        result = choose_runnable(two_runnables, None)
        assert result.target == "pkg-b:train"


def test_choose_runnable_interactive_cancelled_raises(two_runnables: list[Runnable]) -> None:
    with patch("sys.stdin") as mock_stdin, patch("questionary.select") as mock_select:
        mock_stdin.isatty.return_value = True
        mock_select.return_value.ask.return_value = None
        with pytest.raises(KeyboardInterrupt):
            choose_runnable(two_runnables, None)


# --- execute_runnable ---


def test_execute_runnable_returns_exit_code() -> None:
    runnable = Runnable(
        target="pkg-a:main",
        project_name="pkg-a",
        script_name="main",
        entrypoint="pkg_a.main:main",
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        assert execute_runnable(runnable, []) == 0


def test_execute_runnable_nonzero_exit_code() -> None:
    runnable = Runnable(
        target="pkg-a:main",
        project_name="pkg-a",
        script_name="main",
        entrypoint="pkg_a.main:main",
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=2)
        assert execute_runnable(runnable, []) == 2


def test_execute_runnable_command_includes_package_and_entrypoint() -> None:
    runnable = Runnable(
        target="pkg-a:main",
        project_name="pkg-a",
        script_name="main",
        entrypoint="pkg_a.main:main",
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        execute_runnable(runnable, [])
        cmd: list[str] = mock_run.call_args[0][0]
        assert "--package" in cmd
        assert "pkg-a" in cmd
        assert "pkg_a.main:main" in cmd


def test_load_env_file_parses_dotenv_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "FOO=bar",
                "SPACED = value with spaces # trailing comment",
                "HASH=abc#def",
                "QUOTED='value # not a comment'",
                'DOUBLE="value\\nwith newline"',
                "export EXPORTED=yes",
                "EMPTY=",
            ]
        ),
        encoding="utf-8",
    )

    assert load_env_file(env_file) == {
        "FOO": "bar",
        "SPACED": "value with spaces",
        "HASH": "abc#def",
        "QUOTED": "value # not a comment",
        "DOUBLE": "value\nwith newline",
        "EXPORTED": "yes",
        "EMPTY": "",
    }


def test_execute_runnable_applies_env_file_when_present(tmp_path: Path) -> None:
    runnable = Runnable(
        target="pkg-a:main",
        project_name="pkg-a",
        script_name="main",
        entrypoint="pkg_a.main:main",
    )
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\n", encoding="utf-8")

    with patch("run.REPO_ROOT", tmp_path), patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        execute_runnable(runnable, [])
        cmd: list[str] = mock_run.call_args[0][0]
        env: dict[str, str] = mock_run.call_args.kwargs["env"]
        assert "--env-file" not in cmd
        assert env["FOO"] == "bar"


def test_execute_runnable_env_file_overrides_shell_environment(tmp_path: Path) -> None:
    runnable = Runnable(
        target="pkg-a:main",
        project_name="pkg-a",
        script_name="main",
        entrypoint="pkg_a.main:main",
    )
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=file\nBAR=from-file\n", encoding="utf-8")

    with (
        patch("run.REPO_ROOT", tmp_path),
        patch.dict("os.environ", {"FOO": "shell", "BAZ": "from-shell"}, clear=True),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        execute_runnable(runnable, [])
        env: dict[str, str] = mock_run.call_args.kwargs["env"]
        assert env["FOO"] == "file"
        assert env["BAR"] == "from-file"
        assert env["BAZ"] == "from-shell"


def test_execute_runnable_omits_env_file_when_missing(tmp_path: Path) -> None:
    runnable = Runnable(
        target="pkg-a:main",
        project_name="pkg-a",
        script_name="main",
        entrypoint="pkg_a.main:main",
    )

    with patch("run.REPO_ROOT", tmp_path), patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        execute_runnable(runnable, [])
        cmd: list[str] = mock_run.call_args[0][0]
        assert "--env-file" not in cmd


def test_execute_runnable_forwards_script_args() -> None:
    runnable = Runnable(
        target="pkg-a:main",
        project_name="pkg-a",
        script_name="main",
        entrypoint="pkg_a.main:main",
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        execute_runnable(runnable, ["--foo", "bar"])
        cmd: list[str] = mock_run.call_args[0][0]
        assert "--foo" in cmd
        assert "bar" in cmd


# --- discover_runnables ---


def _make_workspace(tmp_path: Path, members_toml: str) -> Path:
    root_pyproject = tmp_path / "pyproject.toml"
    root_pyproject.write_text(f"[tool.uv.workspace]\nmembers = {members_toml}\n")
    return root_pyproject


def test_discover_runnables_finds_scripts(tmp_path: Path) -> None:
    root_pyproject = _make_workspace(tmp_path, '["pkgs/*"]')
    pkg_dir = tmp_path / "pkgs" / "my-package"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "pyproject.toml").write_text(
        '[project]\nname = "my-package"\n[project.scripts]\nmain = "my_package.main:main"\n'
    )
    other_dir = tmp_path / "pkgs" / "other-package"
    other_dir.mkdir(parents=True)
    (other_dir / "pyproject.toml").write_text(
        '[project]\nname = "other-package"\n[project.scripts]\nmain = "other.main:main"\n'
    )

    with patch("run.REPO_ROOT", tmp_path), patch("run.ROOT_PYPROJECT", root_pyproject):
        runnables = discover_runnables()

    by_target = {r.target: r for r in runnables}
    assert by_target["my-package:main"].entrypoint == "my_package.main:main"
    assert by_target["other-package:main"].entrypoint == "other.main:main"


def test_discover_runnables_skips_package_without_scripts(tmp_path: Path) -> None:
    root_pyproject = _make_workspace(tmp_path, '["pkgs/*"]')
    pkg_dir = tmp_path / "pkgs" / "no-scripts"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "pyproject.toml").write_text('[project]\nname = "no-scripts"\n')

    with patch("run.REPO_ROOT", tmp_path), patch("run.ROOT_PYPROJECT", root_pyproject):
        runnables = discover_runnables()

    assert runnables == []


def test_discover_runnables_skips_dir_without_pyproject(tmp_path: Path) -> None:
    root_pyproject = _make_workspace(tmp_path, '["pkgs/*"]')
    (tmp_path / "pkgs" / "no-toml").mkdir(parents=True)

    with patch("run.REPO_ROOT", tmp_path), patch("run.ROOT_PYPROJECT", root_pyproject):
        runnables = discover_runnables()

    assert runnables == []


def test_discover_runnables_missing_workspace_key_raises(tmp_path: Path) -> None:
    root_pyproject = tmp_path / "pyproject.toml"
    root_pyproject.write_text('[project]\nname = "root"\n')

    with patch("run.REPO_ROOT", tmp_path), patch("run.ROOT_PYPROJECT", root_pyproject):
        with pytest.raises(ValueError, match="tool.uv.workspace.members"):
            discover_runnables()


def test_discover_runnables_members_not_list_raises(tmp_path: Path) -> None:
    root_pyproject = tmp_path / "pyproject.toml"
    root_pyproject.write_text('[tool.uv.workspace]\nmembers = "not-a-list"\n')

    with patch("run.REPO_ROOT", tmp_path), patch("run.ROOT_PYPROJECT", root_pyproject):
        with pytest.raises(ValueError, match="must be a list"):
            discover_runnables()


def test_discover_runnables_missing_project_name_raises(tmp_path: Path) -> None:
    root_pyproject = _make_workspace(tmp_path, '["pkgs/*"]')
    pkg_dir = tmp_path / "pkgs" / "bad-pkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "pyproject.toml").write_text('[project.scripts]\nmain = "bad.main:main"\n')

    with patch("run.REPO_ROOT", tmp_path), patch("run.ROOT_PYPROJECT", root_pyproject):
        with pytest.raises(ValueError, match="project.name"):
            discover_runnables()


def test_discover_runnables_returns_sorted(tmp_path: Path) -> None:
    root_pyproject = _make_workspace(tmp_path, '["pkgs/*"]')
    for name in ["zzz-pkg", "aaa-pkg"]:
        pkg_dir = tmp_path / "pkgs" / name
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "pyproject.toml").write_text(
            f"[project]\n"
            f'name = "{name}"\n'
            f"[project.scripts]\n"
            f'main = "{name.replace("-", "_")}.main:main"\n'
        )

    with patch("run.REPO_ROOT", tmp_path), patch("run.ROOT_PYPROJECT", root_pyproject):
        runnables = discover_runnables()

    targets = [r.target for r in runnables]
    assert targets == sorted(targets)


def test_discover_runnables_reads_default_args(tmp_path: Path) -> None:
    root_pyproject = _make_workspace(tmp_path, '["pkgs/*"]')
    pkg_dir = tmp_path / "pkgs" / "with-defaults"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "with-defaults"\n'
        "[project.scripts]\n"
        'main = "with_defaults.main:main"\n'
        'other = "with_defaults.other:run"\n'
        "[tool.run.default-args]\n"
        'main = "--log-level DEBUG --input-file data/input.jsonl"\n'
    )

    with patch("run.REPO_ROOT", tmp_path), patch("run.ROOT_PYPROJECT", root_pyproject):
        runnables = discover_runnables()

    by_target = {r.target: r for r in runnables}
    assert by_target["with-defaults:main"].default_args == (
        DefaultArgs(name="default", args="--log-level DEBUG --input-file data/input.jsonl"),
    )
    assert by_target["with-defaults:other"].default_args == ()


def test_discover_runnables_reads_default_arg_presets(tmp_path: Path) -> None:
    root_pyproject = _make_workspace(tmp_path, '["pkgs/*"]')
    pkg_dir = tmp_path / "pkgs" / "with-presets"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "with-presets"\n'
        "[project.scripts]\n"
        'main = "with_presets.main:main"\n'
        "[tool.run.default-args]\n"
        "main = [\n"
        '  { name = "simple", args = "--input-file payloads/simple.jsonl" },\n'
        '  { name = "remote", args = "--input-file payloads/remote.jsonl" },\n'
        "]\n"
    )

    with patch("run.REPO_ROOT", tmp_path), patch("run.ROOT_PYPROJECT", root_pyproject):
        runnables = discover_runnables()

    assert runnables[0].default_args == (
        DefaultArgs(name="simple", args="--input-file payloads/simple.jsonl"),
        DefaultArgs(name="remote", args="--input-file payloads/remote.jsonl"),
    )


def test_discover_runnables_invalid_default_args_table_raises(tmp_path: Path) -> None:
    root_pyproject = _make_workspace(tmp_path, '["pkgs/*"]')
    pkg_dir = tmp_path / "pkgs" / "bad-defaults"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "bad-defaults"\n'
        "[project.scripts]\n"
        'main = "bad_defaults.main:main"\n'
        "[tool.run]\n"
        'default-args = "not-a-table"\n'
    )

    with patch("run.REPO_ROOT", tmp_path), patch("run.ROOT_PYPROJECT", root_pyproject):
        with pytest.raises(ValueError, match="default-args"):
            discover_runnables()


def test_discover_runnables_invalid_default_args_value_raises(tmp_path: Path) -> None:
    root_pyproject = _make_workspace(tmp_path, '["pkgs/*"]')
    pkg_dir = tmp_path / "pkgs" / "bad-default-value"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "bad-default-value"\n'
        "[project.scripts]\n"
        'main = "bad_default_value.main:main"\n'
        "[tool.run.default-args]\n"
        "main = 42\n"
    )

    with patch("run.REPO_ROOT", tmp_path), patch("run.ROOT_PYPROJECT", root_pyproject):
        with pytest.raises(ValueError, match="must be a string"):
            discover_runnables()


def test_discover_runnables_invalid_default_args_preset_raises(tmp_path: Path) -> None:
    root_pyproject = _make_workspace(tmp_path, '["pkgs/*"]')
    pkg_dir = tmp_path / "pkgs" / "bad-default-preset"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "bad-default-preset"\n'
        "[project.scripts]\n"
        'main = "bad_default_preset.main:main"\n'
        "[tool.run.default-args]\n"
        'main = [{ name = "missing args" }]\n'
    )

    with patch("run.REPO_ROOT", tmp_path), patch("run.ROOT_PYPROJECT", root_pyproject):
        with pytest.raises(ValueError, match="must define an `args` string"):
            discover_runnables()


def test_discover_runnables_multiple_scripts_per_package(tmp_path: Path) -> None:
    root_pyproject = _make_workspace(tmp_path, '["pkgs/*"]')
    pkg_dir = tmp_path / "pkgs" / "multi"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "multi"\n'
        "[project.scripts]\n"
        'train = "multi.train:run"\n'
        'eval = "multi.eval:run"\n'
    )

    with patch("run.REPO_ROOT", tmp_path), patch("run.ROOT_PYPROJECT", root_pyproject):
        runnables = discover_runnables()

    assert len(runnables) == 2
    assert {r.script_name for r in runnables} == {"train", "eval"}


# --- resolve_script_args ---


def _runnable_with_defaults(default_args: str = "") -> Runnable:
    default_arg_presets = (DefaultArgs(name="default", args=default_args),) if default_args else ()
    return _runnable_with_default_presets(default_arg_presets)


def _runnable_with_default_presets(default_args: tuple[DefaultArgs, ...]) -> Runnable:
    return Runnable(
        target="pkg-a:main",
        project_name="pkg-a",
        script_name="main",
        entrypoint="pkg_a.main:main",
        default_args=default_args,
    )


def test_resolve_script_args_explicit_args_pass_through() -> None:
    runnable = _runnable_with_defaults("--foo bar")
    with patch("questionary.text") as mock_text:
        result = resolve_script_args(runnable, ["--given", "value"])
    assert result == ["--given", "value"]
    mock_text.assert_not_called()


def test_resolve_script_args_no_default_returns_empty() -> None:
    runnable = _runnable_with_defaults("")
    with patch("questionary.text") as mock_text:
        result = resolve_script_args(runnable, [])
    assert result == []
    mock_text.assert_not_called()


def test_resolve_script_args_non_tty_uses_default_directly() -> None:
    runnable = _runnable_with_defaults("--log-level DEBUG --input-file data/input.jsonl")
    with patch("sys.stdin") as mock_stdin, patch("questionary.text") as mock_text:
        mock_stdin.isatty.return_value = False
        result = resolve_script_args(runnable, [])
    assert result == ["--log-level", "DEBUG", "--input-file", "data/input.jsonl"]
    mock_text.assert_not_called()


def test_resolve_script_args_non_tty_uses_first_default_preset() -> None:
    runnable = _runnable_with_default_presets(
        (
            DefaultArgs(name="simple", args="--input-file payloads/simple.jsonl"),
            DefaultArgs(name="remote", args="--input-file payloads/remote.jsonl"),
        )
    )
    with (
        patch("sys.stdin") as mock_stdin,
        patch("questionary.select") as mock_select,
        patch("questionary.text") as mock_text,
    ):
        mock_stdin.isatty.return_value = False
        result = resolve_script_args(runnable, [])
    assert result == ["--input-file", "payloads/simple.jsonl"]
    mock_select.assert_not_called()
    mock_text.assert_not_called()


def test_resolve_script_args_tty_prompts_with_default_and_splits_answer() -> None:
    runnable = _runnable_with_defaults("--log-level DEBUG --input-file data/input.jsonl")
    with patch("sys.stdin") as mock_stdin, patch("questionary.text") as mock_text:
        mock_stdin.isatty.return_value = True
        mock_text.return_value.ask.return_value = "--log-level INFO --input-file data/other.jsonl"
        result = resolve_script_args(runnable, [])
    mock_text.assert_called_once_with(
        "Args:", default="--log-level DEBUG --input-file data/input.jsonl"
    )
    assert result == ["--log-level", "INFO", "--input-file", "data/other.jsonl"]


def test_resolve_script_args_tty_selects_default_preset_then_prompts() -> None:
    default_args = (
        DefaultArgs(name="simple", args="--input-file payloads/simple.jsonl"),
        DefaultArgs(name="remote", args="--input-file payloads/remote.jsonl"),
    )
    runnable = _runnable_with_default_presets(default_args)
    with (
        patch("sys.stdin") as mock_stdin,
        patch("questionary.select") as mock_select,
        patch("questionary.text") as mock_text,
    ):
        mock_stdin.isatty.return_value = True
        mock_select.return_value.ask.return_value = default_args[1]
        mock_text.return_value.ask.return_value = "--input-file payloads/modified.jsonl"
        result = resolve_script_args(runnable, [])
    mock_select.assert_called_once()
    mock_text.assert_called_once_with("Args:", default="--input-file payloads/remote.jsonl")
    assert result == ["--input-file", "payloads/modified.jsonl"]


def test_resolve_script_args_tty_default_preset_selection_cancelled_raises() -> None:
    runnable = _runnable_with_default_presets(
        (
            DefaultArgs(name="simple", args="--input-file payloads/simple.jsonl"),
            DefaultArgs(name="remote", args="--input-file payloads/remote.jsonl"),
        )
    )
    with (
        patch("sys.stdin") as mock_stdin,
        patch("questionary.select") as mock_select,
        patch("questionary.text") as mock_text,
    ):
        mock_stdin.isatty.return_value = True
        mock_select.return_value.ask.return_value = None
        with pytest.raises(KeyboardInterrupt):
            resolve_script_args(runnable, [])
    mock_text.assert_not_called()


def test_resolve_script_args_tty_cancelled_raises() -> None:
    runnable = _runnable_with_defaults("--foo bar")
    with patch("sys.stdin") as mock_stdin, patch("questionary.text") as mock_text:
        mock_stdin.isatty.return_value = True
        mock_text.return_value.ask.return_value = None
        with pytest.raises(KeyboardInterrupt):
            resolve_script_args(runnable, [])


def test_resolve_script_args_handles_quoted_values_in_answer() -> None:
    runnable = _runnable_with_defaults("--input '[]'")
    with patch("sys.stdin") as mock_stdin, patch("questionary.text") as mock_text:
        mock_stdin.isatty.return_value = True
        mock_text.return_value.ask.return_value = '--input \'[{"k": "v"}]\''
        result = resolve_script_args(runnable, [])
    assert result == ["--input", '[{"k": "v"}]']


# --- main ---


def test_main_success_returns_zero() -> None:
    runnable = Runnable(
        target="pkg-a:main",
        project_name="pkg-a",
        script_name="main",
        entrypoint="pkg_a.main:main",
    )
    with (
        patch("run.discover_runnables", return_value=[runnable]),
        patch("run.choose_runnable", return_value=runnable),
        patch("run.execute_runnable", return_value=0),
    ):
        assert main(["pkg-a:main"]) == 0


def test_main_propagates_script_exit_code() -> None:
    runnable = Runnable(
        target="pkg-a:main",
        project_name="pkg-a",
        script_name="main",
        entrypoint="pkg_a.main:main",
    )
    with (
        patch("run.discover_runnables", return_value=[runnable]),
        patch("run.choose_runnable", return_value=runnable),
        patch("run.execute_runnable", return_value=3),
    ):
        assert main(["pkg-a:main"]) == 3


def test_main_returns_one_on_value_error() -> None:
    with patch("run.discover_runnables", side_effect=ValueError("no scripts found")):
        assert main([]) == 1


def test_main_returns_one_when_no_runnables() -> None:
    with patch("run.discover_runnables", return_value=[]):
        assert main([]) == 1
