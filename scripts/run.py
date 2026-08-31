# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover and run workspace package scripts from one entrypoint.

This module powers the repo-level ``make run`` workflow. It scans the uv
workspace declared in the root ``pyproject.toml``, finds member packages that
define ``[project.scripts]``, and exposes those scripts as runnable targets in
the form ``project_name:script_name``.

Run a target directly with ``scripts/run.py project_name:script_name`` or choose
one interactively. Arguments after ``--`` are forwarded to the selected script.

The runner intentionally does not invoke the script name directly with
``uv run --package <project> <script>``. When multiple workspace packages
register the same script name, that approach can resolve to the wrong package.
Instead, this module reads the script entrypoint from ``project.scripts`` and
executes the target callable explicitly inside the selected package's
environment. That keeps script discovery standard-compliant while avoiding
cross-package name collisions.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import questionary

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"


@dataclass(frozen=True)
class DefaultArgs:
    """Describe one selectable default argument preset."""

    name: str
    args: str


@dataclass(frozen=True)
class Runnable:
    """Describe one script that can be launched from the workspace runner."""

    target: str
    project_name: str
    script_name: str
    entrypoint: str
    default_args: tuple[DefaultArgs, ...] = ()


ENTRYPOINT_RUNNER = """
from importlib import import_module
import sys

entrypoint = sys.argv[1]
module_name, separator, qualname = entrypoint.partition(":")
if not separator or not module_name or not qualname:
    raise SystemExit(f"Invalid script entrypoint: {entrypoint!r}")

target = import_module(module_name)
for attr in qualname.split("."):
    target = getattr(target, attr)

sys.argv = [entrypoint, *sys.argv[2:]]
result = target()
raise SystemExit(0 if result is None else result)
""".strip()


def discover_runnables() -> list[Runnable]:
    """Collect runnable script targets from all uv workspace member packages.

    The function reads the root workspace configuration, expands each member
    path, and inspects every package ``pyproject.toml``. Packages without a
    ``project.scripts`` table are skipped. Each script entry becomes a
    ``Runnable`` whose target is formatted as ``project_name:script_name``.

    Returns:
        A sorted list of runnable workspace script definitions.

    Raises:
        ValueError: If the root workspace configuration is invalid or a package
            defines malformed ``project`` or ``project.scripts`` metadata.
    """

    runnables: list[Runnable] = []
    with ROOT_PYPROJECT.open("rb") as pyproject_file:
        root_pyproject = tomllib.load(pyproject_file)

    try:
        member_patterns = root_pyproject["tool"]["uv"]["workspace"]["members"]
    except KeyError as error:
        raise ValueError("`pyproject.toml` must define `tool.uv.workspace.members`.") from error

    if not isinstance(member_patterns, list):
        raise ValueError("`tool.uv.workspace.members` must be a list.")

    member_dirs: list[Path] = []
    for pattern in member_patterns:
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("`tool.uv.workspace.members` must only contain strings.")
        member_dirs.extend(path for path in REPO_ROOT.glob(pattern) if path.is_dir())

    for package_dir in sorted(set(member_dirs)):
        pyproject_path = package_dir / "pyproject.toml"
        if not pyproject_path.exists():
            continue

        with pyproject_path.open("rb") as pyproject_file:
            pyproject = tomllib.load(pyproject_file)

        try:
            project_name = pyproject["project"]["name"]
        except KeyError as error:
            raise ValueError(f"`{pyproject_path}` must define `project.name`.") from error

        if not isinstance(project_name, str) or not project_name:
            raise ValueError(f"`{pyproject_path}` must define `project.name`.")

        scripts = pyproject["project"].get("scripts", {})
        if not isinstance(scripts, dict) or not scripts:
            continue

        default_args_table = pyproject.get("tool", {}).get("run", {}).get("default-args", {})
        if not isinstance(default_args_table, dict):
            raise ValueError(
                f"`{pyproject_path}` `tool.run.default-args` must be a table of "
                f"script name to argument presets."
            )

        for script_name, entrypoint in sorted(scripts.items()):
            if not isinstance(script_name, str) or not script_name:
                raise ValueError(f"`{pyproject_path}` has an invalid `project.scripts` entry.")
            if not isinstance(entrypoint, str) or not entrypoint:
                raise ValueError(f"`{pyproject_path}` has an invalid `project.scripts` entry.")

            default_args = parse_default_args(
                pyproject_path=pyproject_path,
                script_name=script_name,
                raw_default_args=default_args_table.get(script_name),
            )

            runnables.append(
                Runnable(
                    target=f"{project_name}:{script_name}",
                    project_name=project_name,
                    script_name=script_name,
                    entrypoint=entrypoint,
                    default_args=default_args,
                )
            )

    return runnables


def parse_default_args(
    pyproject_path: Path,
    script_name: str,
    raw_default_args: object,
) -> tuple[DefaultArgs, ...]:
    """Parse one script's default arg presets from package metadata.

    Preferred metadata is a list of inline tables:

    ``main = [{ name = "local", args = "--input-file payloads/simple.jsonl" }]``

    A single string is still accepted as one unnamed legacy preset so existing
    services continue to run while they migrate.
    """

    if raw_default_args is None or raw_default_args == "":
        return ()

    if isinstance(raw_default_args, str):
        return (DefaultArgs(name="default", args=raw_default_args),)

    if not isinstance(raw_default_args, list):
        raise ValueError(
            f"`{pyproject_path}` `tool.run.default-args.{script_name}` must be a string, "
            f"a list of strings, or a list of tables with `name` and `args` strings."
        )

    parsed_default_args: list[DefaultArgs] = []
    for index, default_args in enumerate(raw_default_args, start=1):
        if isinstance(default_args, str):
            parsed_default_args.append(
                DefaultArgs(name=default_args or f"args {index}", args=default_args)
            )
            continue

        if not isinstance(default_args, dict):
            raise ValueError(
                f"`{pyproject_path}` `tool.run.default-args.{script_name}` entry {index} "
                f"must be a string or a table with `name` and `args` strings."
            )

        name = default_args.get("name")
        args = default_args.get("args")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"`{pyproject_path}` `tool.run.default-args.{script_name}` entry {index} "
                f"must define a non-empty `name` string."
            )
        if not isinstance(args, str):
            raise ValueError(
                f"`{pyproject_path}` `tool.run.default-args.{script_name}` entry {index} "
                f"must define an `args` string."
            )

        parsed_default_args.append(DefaultArgs(name=name, args=args))

    return tuple(parsed_default_args)


def parse_args(argv: list[str]) -> tuple[str | None, list[str]]:
    """Parse ``[project:script] [-- script args...]``."""

    runner_args = argv
    script_args: list[str] = []

    if "--" in argv:
        separator_index = argv.index("--")
        runner_args = argv[:separator_index]
        script_args = argv[separator_index + 1 :]

    if len(runner_args) > 1:
        raise ValueError(
            "Expected at most one runner argument before `--`. Pass script arguments after `--`."
        )

    requested_script = runner_args[0] if runner_args else None
    return requested_script, script_args


def choose_runnable(
    runnables: list[Runnable],
    requested_name: str | None,
) -> Runnable:
    """Resolve the runnable target to execute.

    When a target is provided explicitly, this function validates and returns
    it. Otherwise it opens an interactive picker so the user can choose from
    the discovered workspace scripts.

    Args:
        runnables: Available runnable targets discovered from the workspace.
        requested_name: Optional ``project:script`` target supplied on the
            command line.

    Returns:
        The runnable selected explicitly or chosen interactively.

    Raises:
        ValueError: If the requested target is unknown or no interactive
            terminal is available for selection.
        KeyboardInterrupt: If the user cancels the interactive prompt.
    """

    if requested_name:
        for runnable in runnables:
            if requested_name == runnable.target:
                return runnable

        available = ", ".join(runnable.target for runnable in runnables)
        raise ValueError(
            f"Unknown target `{requested_name}`. Pass `project:script` before `--`. "
            f"Available targets: {available}."
        )

    if not sys.stdin.isatty():
        available = ", ".join(runnable.target for runnable in runnables)
        raise ValueError(
            "No target was specified and interactive selection is unavailable. "
            f"Pass `project:script` before `--`. Available targets: {available}."
        )

    selection = questionary.select(
        "Select a package script to run:",
        choices=[
            questionary.Choice(
                title=f"{runnable.target}",
                value=runnable.target,
            )
            for runnable in runnables
        ],
    ).ask()
    if selection is None:
        raise KeyboardInterrupt

    return next(runnable for runnable in runnables if runnable.target == selection)


def resolve_script_args(runnable: Runnable, script_args: list[str]) -> list[str]:
    """Resolve the final argv to forward to the selected script.

    When the user passed args explicitly (via ``--`` on the command line), those
    args win and are returned unchanged. Otherwise, the configured per-script
    defaults from ``[tool.run.default-args]`` are consulted: in an interactive
    terminal, the user picks a preset when multiple presets exist, then the
    selected args are shown in an editable prompt; in a non-interactive context,
    the first preset is used directly with no prompt.

    Args:
        runnable: The package script selected for execution.
        script_args: The arguments parsed from the command line (after ``--``).

    Returns:
        The argv list to forward to the script.

    Raises:
        KeyboardInterrupt: If the user cancels the interactive prompt.
    """

    if script_args:
        return script_args
    if not runnable.default_args:
        return []

    default_args = runnable.default_args[0]
    if not sys.stdin.isatty():
        return shlex.split(default_args.args)

    if len(runnable.default_args) > 1:
        selection = questionary.select(
            "Select args:",
            choices=[
                questionary.Choice(
                    title=format_default_args_choice(default_args),
                    value=default_args,
                )
                for default_args in runnable.default_args
            ],
        ).ask()
        if selection is None:
            raise KeyboardInterrupt
        default_args = selection

    answer = questionary.text("Args:", default=default_args.args).ask()
    if answer is None:
        raise KeyboardInterrupt
    return shlex.split(answer)


def format_default_args_choice(default_args: DefaultArgs) -> str:
    """Format one default-args preset for the interactive selector."""

    if not default_args.args or default_args.name == default_args.args:
        return default_args.name
    return f"{default_args.name}: {default_args.args}"


def load_env_file(path: Path) -> dict[str, str]:
    """Load dotenv-style key/value pairs from a file.

    The runner applies these values after copying the shell environment so
    repo-local configuration wins over ambient variables.
    """

    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").lstrip()

        key, separator, raw_value = stripped.partition("=")
        if not separator:
            raise ValueError(f"Invalid .env line {line_number}: missing `=`")

        key = key.strip()
        if not key:
            raise ValueError(f"Invalid .env line {line_number}: missing variable name")
        if any(char.isspace() for char in key):
            raise ValueError(f"Invalid .env line {line_number}: variable name contains whitespace")

        values[key] = _parse_env_value(raw_value)

    return values


def _parse_env_value(raw_value: str) -> str:
    value = _strip_env_comment(raw_value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        unquoted = value[1:-1]
        if value[0] == '"':
            return _unescape_double_quoted_env_value(unquoted)
        return unquoted
    return value


def _strip_env_comment(value: str) -> str:
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_double_quote:
            escaped = True
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if (
            char == "#"
            and not in_single_quote
            and not in_double_quote
            and (index == 0 or value[index - 1].isspace())
        ):
            return value[:index].rstrip()

    return value.strip()


def _unescape_double_quoted_env_value(value: str) -> str:
    replacements = {
        "\\": "\\",
        '"': '"',
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    result: list[str] = []
    escaped = False

    for char in value:
        if escaped:
            result.append(replacements.get(char, f"\\{char}"))
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        result.append(char)

    if escaped:
        result.append("\\")

    return "".join(result)


def execute_runnable(runnable: Runnable, script_args: list[str]) -> int:
    """Execute a selected package script inside its package environment.

    The script is launched by evaluating the registered ``project.scripts``
    entrypoint directly, rather than invoking the script name as a shell
    command. This avoids collisions when multiple packages register the same
    script name, such as several packages exposing a ``main`` entrypoint. When
    a repo-level ``.env`` file exists, its values are applied to the child
    process after the shell environment is copied so local configuration
    overrides ambient variables.

    Args:
        runnable: The package script selected for execution.
        script_args: Additional arguments to forward to the script.

    Returns:
        The subprocess exit code.
    """

    command = [
        "uv",
        "run",
    ]
    env_file = REPO_ROOT / ".env"
    env: dict[str, str] | None = None
    if env_file.exists():
        env = os.environ.copy()
        env.update(load_env_file(env_file))

    command.extend(
        [
            "--package",
            runnable.project_name,
            "python",
            "-c",
            ENTRYPOINT_RUNNER,
            runnable.entrypoint,
            *script_args,
        ]
    )
    print(f"Running `{runnable.target}`", flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False, env=env).returncode


def main(argv: list[str] | None = None) -> int:
    """Run the workspace script selection and dispatch flow.

    Args:
        argv: Optional command-line arguments. When omitted, arguments are read
            from ``sys.argv``.

    Returns:
        A process exit code where ``0`` indicates success, ``1`` indicates a
        user-facing configuration or validation error, and ``130`` indicates
        that the interactive selection was cancelled.
    """

    try:
        requested_script, script_args = parse_args(argv or sys.argv[1:])
        runnables = discover_runnables()
        if not runnables:
            raise ValueError("No runnable scripts were found in the workspace.")

        runnable = choose_runnable(runnables, requested_script)
        resolved_args = resolve_script_args(runnable, script_args)
        return execute_runnable(runnable, resolved_args)
    except KeyboardInterrupt:
        print("Service selection cancelled.", file=sys.stderr)
        return 130
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
