# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover and build Docker images registered by workspace packages.

This module powers the repo-level ``make build`` workflow. It scans the uv
workspace declared in the root ``pyproject.toml``, finds package-level
``[tool.build.images]`` registrations, and exposes those images as buildable
targets in the form ``project_name:image_name``.

Run a target directly with ``scripts/build.py project_name:image_name`` or
choose one interactively. Registered commands are split with ``shlex`` and
executed from the repository root. CI can list targets as JSON, replace the
registered Docker tag, and push the resulting image after a successful build.
"""

from __future__ import annotations

import importlib
import json
import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"

TEXT_FORMAT = "text"
JSON_FORMAT = "json"
DOCKER_REPO_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
DOCKER_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")

# Platforms to build for when a buildx command is pushed as a manifest list.
DEFAULT_BUILD_PLATFORMS = "linux/amd64,linux/arm64"


@dataclass(frozen=True)
class BuildableImage:
    """Describe one Docker image build target registered by a workspace package.

    Attributes:
        target: Fully qualified selector in ``project_name:image_name`` form.
        project_name: Name from the registering package's ``project.name`` metadata.
        version: Version from the registering package's ``project.version`` metadata.
        package_dir: Workspace member directory that registered the image.
        image_name: Key from the package's ``[tool.build.images]`` table.
        command: Docker build command to execute from the repository root.
        base_target: Optional ``project:image`` target of a shared base image that
            this image's builder stage starts from. When set, CI builds the base
            first and passes its pushed ref as the ``BUILDER_BASE_IMAGE`` build arg.
    """

    target: str
    project_name: str
    version: str
    package_dir: Path
    image_name: str
    command: str
    base_target: str | None = None


@dataclass(frozen=True)
class BuildOptions:
    """Command-line options for the build runner."""

    target: str | None
    list_images: bool
    gitlab_ci: bool
    output_format: str
    tag: str | None
    push: bool
    platforms: str
    build_args: tuple[str, ...] = ()


def _load_workspace_member_dirs() -> list[Path]:
    """Load and expand uv workspace member directories from the root project file.

    Returns:
        Sorted workspace member directories that currently exist in the repository.

    Raises:
        ValueError: If the root ``pyproject.toml`` does not define
            ``tool.uv.workspace.members`` as a list of non-empty strings.
    """

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

    return sorted(set(member_dirs))


def discover_buildable_images() -> list[BuildableImage]:
    """Collect Docker image build targets from all uv workspace member packages.

    Packages register images under ``[tool.build.images]`` in their
    ``pyproject.toml``. Each image entry is a command string whose key becomes
    the image name in the ``project:image`` target.

    Returns:
        A sorted list of buildable Docker image definitions.

    Raises:
        ValueError: If the root workspace configuration is invalid or a package
            defines malformed ``project`` or ``tool.build.images`` metadata.
    """

    images: list[BuildableImage] = []
    for package_dir in _load_workspace_member_dirs():
        pyproject_path = package_dir / "pyproject.toml"
        if not pyproject_path.exists():
            continue

        with pyproject_path.open("rb") as pyproject_file:
            pyproject = tomllib.load(pyproject_file)

        try:
            project_table = pyproject["project"]
        except KeyError as error:
            raise ValueError(f"`{pyproject_path}` must define `project.name`.") from error

        if not isinstance(project_table, dict):
            raise ValueError(f"`{pyproject_path}` `project` must be a table.")

        try:
            project_name = project_table["name"]
        except KeyError as error:
            raise ValueError(f"`{pyproject_path}` must define `project.name`.") from error

        if not isinstance(project_name, str) or not project_name:
            raise ValueError(f"`{pyproject_path}` must define `project.name`.")

        tool_table = pyproject.get("tool", {})
        if not isinstance(tool_table, dict):
            raise ValueError(f"`{pyproject_path}` `tool` must be a table.")

        build_table = tool_table.get("build", {})
        if not isinstance(build_table, dict):
            raise ValueError(f"`{pyproject_path}` `tool.build` must be a table.")

        images_table = build_table.get("images", {})
        if not images_table:
            continue
        if not isinstance(images_table, dict):
            raise ValueError(f"`{pyproject_path}` `tool.build.images` must be a table.")

        version = project_table.get("version")
        if not isinstance(version, str) or not version:
            raise ValueError(f"`{pyproject_path}` must define `project.version`.")

        bases_table = build_table.get("image-bases", {})
        if not isinstance(bases_table, dict):
            raise ValueError(f"`{pyproject_path}` `tool.build.image-bases` must be a table.")

        for image_name, command in sorted(images_table.items()):
            if not isinstance(image_name, str) or not image_name:
                raise ValueError(f"`{pyproject_path}` has an invalid `tool.build.images` entry.")
            if not isinstance(command, str) or not command:
                raise ValueError(
                    f"`{pyproject_path}` `tool.build.images.{image_name}` must be a string."
                )
            base_target = bases_table.get(image_name)
            if base_target is not None and (not isinstance(base_target, str) or not base_target):
                raise ValueError(
                    f"`{pyproject_path}` `tool.build.image-bases.{image_name}` must be a string."
                )
            images.append(
                BuildableImage(
                    target=f"{project_name}:{image_name}",
                    project_name=project_name,
                    version=version,
                    package_dir=package_dir,
                    image_name=image_name,
                    command=command,
                    base_target=base_target,
                )
            )

    return sorted(images, key=lambda image: image.target)


def parse_args(argv: list[str]) -> BuildOptions:
    """Parse build runner command-line arguments.

    Args:
        argv: Command-line arguments passed to the build runner, excluding the
            executable name.

    Returns:
        Parsed build runner options.

    Raises:
        ValueError: If the arguments are invalid.
    """

    target: str | None = None
    list_images = False
    gitlab_ci = False
    output_format = TEXT_FORMAT
    tag: str | None = None
    push = False
    platforms = DEFAULT_BUILD_PLATFORMS
    platform_flag_used = False
    build_args: list[str] = []

    def _validate_build_arg(value: str) -> str:
        key, separator, _ = value.partition("=")
        if not separator or not key.strip():
            raise ValueError("`--build-arg` must be in `KEY=VALUE` form.")
        return value

    positionals: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--list":
            list_images = True
        elif arg == "--gitlab-ci":
            gitlab_ci = True
        elif arg == "--push":
            push = True
        elif arg == "--platform":
            index += 1
            if index >= len(argv):
                raise ValueError("Expected a value after `--platform`.")
            platforms = argv[index].strip()
            if not platforms:
                raise ValueError("`--platform` must not be empty.")
            platform_flag_used = True
        elif arg.startswith("--platform="):
            platforms = arg.removeprefix("--platform=").strip()
            if not platforms:
                raise ValueError("`--platform` must not be empty.")
            platform_flag_used = True
        elif arg == "--format":
            index += 1
            if index >= len(argv):
                raise ValueError("Expected a value after `--format`.")
            output_format = argv[index]
        elif arg.startswith("--format="):
            output_format = arg.removeprefix("--format=")
        elif arg == "--tag":
            index += 1
            if index >= len(argv):
                raise ValueError("Expected a value after `--tag`.")
            tag = argv[index]
            if not tag:
                raise ValueError("`--tag` must not be empty.")
        elif arg.startswith("--tag="):
            tag = arg.removeprefix("--tag=")
            if not tag:
                raise ValueError("`--tag` must not be empty.")
        elif arg == "--build-arg":
            index += 1
            if index >= len(argv):
                raise ValueError("Expected a value after `--build-arg`.")
            build_args.append(_validate_build_arg(argv[index]))
        elif arg.startswith("--build-arg="):
            build_args.append(_validate_build_arg(arg.removeprefix("--build-arg=")))
        elif arg.startswith("-"):
            raise ValueError(f"Unknown option `{arg}`.")
        else:
            positionals.append(arg)
        index += 1

    if len(positionals) > 1:
        raise ValueError("Expected at most one build target.")

    if output_format not in {TEXT_FORMAT, JSON_FORMAT}:
        raise ValueError("`--format` must be one of: text, json.")

    platform_overridden = platform_flag_used

    target = positionals[0] if positionals else None
    if list_images and gitlab_ci:
        raise ValueError("Do not pass `--list` with `--gitlab-ci`.")
    if list_images and target:
        raise ValueError("Do not pass a build target with `--list`.")
    if list_images and tag:
        raise ValueError("Do not pass `--tag` with `--list`.")
    if list_images and push:
        raise ValueError("Do not pass `--push` with `--list`.")
    if list_images and platform_overridden:
        raise ValueError("Do not pass `--platform` with `--list`.")
    if gitlab_ci and target:
        raise ValueError("Do not pass a build target with `--gitlab-ci`.")
    if gitlab_ci and tag:
        raise ValueError("Do not pass `--tag` with `--gitlab-ci`.")
    if gitlab_ci and push:
        raise ValueError("Do not pass `--push` with `--gitlab-ci`.")
    if gitlab_ci and platform_overridden:
        raise ValueError("Do not pass `--platform` with `--gitlab-ci`.")
    if gitlab_ci and output_format != TEXT_FORMAT:
        raise ValueError("Do not pass `--format` with `--gitlab-ci`.")
    if not list_images and output_format != TEXT_FORMAT:
        raise ValueError("`--format` can only be used with `--list`.")
    if build_args and (list_images or gitlab_ci):
        raise ValueError("Do not pass `--build-arg` with `--list` or `--gitlab-ci`.")

    return BuildOptions(
        target=target,
        list_images=list_images,
        gitlab_ci=gitlab_ci,
        output_format=output_format,
        tag=tag,
        push=push,
        platforms=platforms,
        build_args=tuple(build_args),
    )


def choose_image(
    images: list[BuildableImage],
    requested_name: str | None,
) -> BuildableImage:
    """Resolve the Docker image target to build.

    If ``requested_name`` is supplied, it must exactly match a discovered
    ``project:image`` target. Otherwise, this function opens an interactive
    selector so the user can choose from the registered images.

    Args:
        images: Build targets discovered from workspace package metadata.
        requested_name: Optional ``project:image`` target supplied on the
            command line.

    Returns:
        The build target selected explicitly or chosen interactively.

    Raises:
        ValueError: If the requested target is unknown or interactive selection
            is unavailable.
        KeyboardInterrupt: If the user cancels the interactive prompt.
    """

    if requested_name:
        for image in images:
            if requested_name == image.target:
                return image

        available = ", ".join(image.target for image in images)
        raise ValueError(
            f"Unknown target `{requested_name}`. Pass `project:image`. "
            f"Available targets: {available}."
        )

    if not sys.stdin.isatty():
        available = ", ".join(image.target for image in images)
        raise ValueError(
            "No target was specified and interactive selection is unavailable. "
            f"Pass `project:image`. Available targets: {available}."
        )

    selection = _select_image_interactively(images)
    if selection is None:
        raise KeyboardInterrupt

    return next(image for image in images if image.target == selection)


def _select_image_interactively(images: list[BuildableImage]) -> str | None:
    """Choose an image with questionary when available, else a stdlib prompt."""

    try:
        questionary = cast(Any, importlib.import_module("questionary"))
    except ImportError:
        return _select_image_with_text_prompt(images)

    return cast(
        str | None,
        questionary.select(
            "Select a Docker image to build:",
            choices=[
                questionary.Choice(
                    title=f"{image.target} -> {image.command}",
                    value=image.target,
                )
                for image in images
            ],
        ).ask(),
    )


def _select_image_with_text_prompt(images: list[BuildableImage]) -> str | None:
    """Minimal interactive selector used when questionary is not installed."""

    print("Select a Docker image to build:", flush=True)
    for index, image in enumerate(images, start=1):
        print(f"  {index}. {image.target} -> {image.command}", flush=True)

    try:
        raw_selection = input("Enter selection number: ").strip()
    except EOFError as error:
        raise KeyboardInterrupt from error
    if not raw_selection:
        return None

    try:
        selection_index = int(raw_selection)
    except ValueError as error:
        raise ValueError(f"Selection must be a number, got `{raw_selection}`.") from error

    if selection_index < 1 or selection_index > len(images):
        raise ValueError(
            f"Selection index {selection_index} is outside the valid range 1-{len(images)}."
        )
    return images[selection_index - 1].target


def format_image_list(images: list[BuildableImage], output_format: str) -> str:
    """Format discovered build targets for human or CI consumption.

    Args:
        images: Build targets discovered from workspace package metadata.
        output_format: ``text`` for one target per line or ``json`` for a
            machine-readable list of image records.

    Returns:
        A formatted representation of the discovered image targets.

    Raises:
        ValueError: If ``output_format`` is not recognized.
    """

    if output_format == TEXT_FORMAT:
        return "\n".join(image.target for image in images)

    if output_format == JSON_FORMAT:
        payload: list[dict[str, str]] = []
        for image in images:
            package_path = image.package_dir.relative_to(REPO_ROOT)
            payload.append(
                {
                    "target": image.target,
                    "project_name": image.project_name,
                    "version": image.version,
                    "package_path": package_path.as_posix(),
                    "image_name": image.image_name,
                    "command": image.command,
                }
            )
        return json.dumps(payload, indent=2)

    raise ValueError("`output_format` must be one of: text, json.")


def _yaml_quote(value: str) -> str:
    """Quote a string as a YAML scalar using JSON-compatible quoting."""

    return json.dumps(value)


def sanitize_docker_repo(value: str) -> str:
    """Normalize a project name for use as a Docker repository path component."""

    normalized = re.sub(r"[^a-z0-9._-]+", "-", value.lower())
    normalized = re.sub(r"[._-]{2,}", "-", normalized).strip("._-")
    if not DOCKER_REPO_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"Docker repository value `{value}` cannot be normalized to match "
            f"`{DOCKER_REPO_PATTERN.pattern}`."
        )
    return normalized


def sanitize_docker_tag(value: str) -> str:
    """Normalize a value for use as a Docker tag component."""

    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).lstrip(".-")
    if len(normalized) > 128:
        raise ValueError("Docker tag values must be 128 characters or fewer after normalization.")
    if not DOCKER_TAG_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"Docker tag value `{value}` cannot be normalized to match "
            f"`{DOCKER_TAG_PATTERN.pattern}`."
        )
    return normalized


# Each image builds once per architecture on a native runner, then a merge job
# combines the two single-arch images into one manifest-list tag. arm64 builds
# run on the aarch64 pool (`arm-sdg`); amd64 stays on `sdg`.
_BUILD_ARCHES: tuple[str, ...] = ("amd64", "arm64")
_ARCH_RUNNER_TAGS: dict[str, str] = {"amd64": "sdg", "arm64": "arm-sdg"}

# Shell that derives the image tag label (`mr<iid>` on MRs, `main` otherwise).
_LABEL_SCRIPT_LINES: list[str] = [
    "    - |",
    '      if [ "$UPA_PARENT_PIPELINE_SOURCE" = "merge_request_event" ]; then',
    '        if [ -z "$UPA_PARENT_MERGE_REQUEST_IID" ]; then',
    '          echo "Error: UPA_PARENT_MERGE_REQUEST_IID is required for MR image tags"',
    "          exit 1",
    "        fi",
    '        LABEL="mr$UPA_PARENT_MERGE_REQUEST_IID"',
    "      else",
    '        LABEL="main"',
    "      fi",
]

# Shell that checks the required registry env vars and logs in to nvcr.io.
_LOGIN_SCRIPT_LINES: list[str] = [
    "    - |",
    '      if [ -z "$METRO_PERF_NGC_API_KEY" ]; then',
    '        echo "Error: METRO_PERF_NGC_API_KEY environment variable is not set"',
    "        exit 1",
    "      fi",
    '      if [ -z "$UPA_IMAGE_REGISTRY_PREFIX" ]; then',
    '        echo "Error: UPA_IMAGE_REGISTRY_PREFIX environment variable is not set"',
    "        exit 1",
    "      fi",
    (
        '    - echo "$METRO_PERF_NGC_API_KEY" | docker login nvcr.io '
        '-u "\\$oauthtoken" --password-stdin'
    ),
]


def _image_ref_script_lines(repository: str, image_name: str, image_version: str) -> list[str]:
    """Shell that assembles the final ``IMAGE_REF`` from the registry prefix and tag."""

    return [
        f'      IMAGE_REPOSITORY="$UPA_IMAGE_REGISTRY_PREFIX/{repository}"',
        f'      IMAGE_TAG="{image_name}-{image_version}-${{CI_COMMIT_SHORT_SHA}}.${{LABEL}}"',
        '      IMAGE_REF="${IMAGE_REPOSITORY}:${IMAGE_TAG}"',
    ]


def _render_arch_build_job(
    image: BuildableImage,
    by_target: dict[str, BuildableImage],
    base_targets: set[str],
    arch: str,
) -> list[str]:
    """Render one native single-arch build job that pushes a per-arch ``-<arch>`` tag."""

    job_name = f"build-publish:{image.target}:{arch}"
    image_repository = _registered_image_repository(image)
    image_name = sanitize_docker_tag(image.image_name)
    image_version = sanitize_docker_tag(image.version)

    # Shared base images run the full source compile, so give them more
    # headroom than the services that consume them.
    is_base = image.target in base_targets
    timeout_line = "  timeout: 5 hours" if is_base else "  timeout: 2 hours 30 minutes"

    needs_lines: list[str] = []
    base_ref_lines: list[str] = []
    build_command = (
        f"      python3 scripts/build.py {_yaml_quote(image.target)} "
        f'--tag "${{ARCH_REF}}" --platform linux/{arch} --push'
    )
    if image.base_target is not None:
        base = by_target.get(image.base_target)
        if base is None:
            raise ValueError(
                f"Image `{image.target}` references unknown base `{image.base_target}`."
            )
        base_repository = sanitize_docker_repo(base.project_name)
        base_name = sanitize_docker_tag(base.image_name)
        base_version = sanitize_docker_tag(base.version)
        # Depend on the base merge job so BUILDER_BASE_IMAGE is a manifest list
        # and this arch pulls its matching base variant.
        needs_lines = ["  needs:", f"    - {_yaml_quote('build-publish:' + base.target)}"]
        base_ref_lines = [
            (
                f'      BUILDER_BASE_IMAGE="$UPA_IMAGE_REGISTRY_PREFIX/{base_repository}:'
                f'{base_name}-{base_version}-${{CI_COMMIT_SHORT_SHA}}.${{LABEL}}"'
            ),
        ]
        build_command = (
            f"      python3 scripts/build.py {_yaml_quote(image.target)} "
            f'--tag "${{ARCH_REF}}" --platform linux/{arch} '
            '--build-arg "BUILDER_BASE_IMAGE=${BUILDER_BASE_IMAGE}" --push'
        )

    return [
        f"{_yaml_quote(job_name)}:",
        "  stage: build",
        "  image: docker:28.4.0",
        "  rules:",
        '    - if: $CI_PIPELINE_SOURCE == "parent_pipeline"',
        "      when: on_success",
        "  tags:",
        f"    - {_ARCH_RUNNER_TAGS[arch]}",
        # One in-flight build per arch pool; amd64 and arm64 still run in parallel.
        f"  resource_group: docker-build-{arch}",
        timeout_line,
        *needs_lines,
        "  services:",
        "    - docker:28.4.0-dind",
        "  before_script:",
        "    - apk add --no-cache python3",
        "    - docker --version",
        "    # Registry cache export needs the docker-container buildx driver.",
        "    - docker buildx create --name upa-builder --driver docker-container --use",
        "    - docker buildx inspect --bootstrap",
        *_LOGIN_SCRIPT_LINES,
        "  script:",
        *_LABEL_SCRIPT_LINES,
        *_image_ref_script_lines(image_repository, image_name, image_version),
        *base_ref_lines,
        f'      ARCH_REF="${{IMAGE_REF}}-{arch}"',
        f'      echo "Building and pushing {image.target} ({arch}) as ${{ARCH_REF}}"',
        build_command,
        "",
    ]


def _render_merge_job(image: BuildableImage) -> list[str]:
    """Render the job that combines the per-arch tags into one manifest-list tag."""

    job_name = f"build-publish:{image.target}"
    image_repository = _registered_image_repository(image)
    image_name = sanitize_docker_tag(image.image_name)
    image_version = sanitize_docker_tag(image.version)

    needs_lines = ["  needs:"]
    for arch in _BUILD_ARCHES:
        needs_lines.append(f"    - {_yaml_quote(f'build-publish:{image.target}:{arch}')}")

    manifest_sources = " ".join(f'"${{IMAGE_REF}}-{arch}"' for arch in _BUILD_ARCHES)

    return [
        f"{_yaml_quote(job_name)}:",
        "  stage: build",
        "  image: docker:28.4.0",
        "  rules:",
        '    - if: $CI_PIPELINE_SOURCE == "parent_pipeline"',
        "      when: on_success",
        *needs_lines,
        "  services:",
        "    - docker:28.4.0-dind",
        "  before_script:",
        "    - docker --version",
        *_LOGIN_SCRIPT_LINES,
        "  script:",
        *_LABEL_SCRIPT_LINES,
        *_image_ref_script_lines(image_repository, image_name, image_version),
        f'      echo "Merging {image.target} arch manifests into ${{IMAGE_REF}}"',
        f'      docker buildx imagetools create -t "${{IMAGE_REF}}" {manifest_sources}',
        "",
    ]


def _registered_image_repository(image: BuildableImage) -> str:
    """Return the unqualified repository declared by an image build command."""

    local_tag = _find_build_tag(shlex.split(image.command))
    if not local_tag:
        raise ValueError(
            f"Build command for {image.target} must include -t or --tag for CI publication."
        )
    if "/" in local_tag:
        raise ValueError(
            f"Build command for {image.target} must use an unqualified Docker repository "
            f"name for CI publication, got {local_tag}."
        )
    return sanitize_docker_repo(local_tag.split(":", maxsplit=1)[0])


def format_gitlab_ci_config(images: list[BuildableImage]) -> str:
    """Render a dynamic GitLab child pipeline that builds each image per arch and merges."""

    lines = [
        "stages:",
        "  - build",
        "",
        "default:",
        "  interruptible: true",
        "  tags:",
        "    - sdg",
        "",
    ]

    by_target = {image.target: image for image in images}
    base_targets = {image.base_target for image in images if image.base_target is not None}

    for image in images:
        for arch in _BUILD_ARCHES:
            lines.extend(_render_arch_build_job(image, by_target, base_targets, arch))
        lines.extend(_render_merge_job(image))

    return "\n".join(lines).rstrip() + "\n"


def _replace_build_tag(command: list[str], tag: str) -> list[str]:
    """Replace the first Docker build tag in ``command`` with ``tag``."""

    if not tag:
        raise ValueError("Docker image tag override must not be empty.")

    updated_command = list(command)
    for index, part in enumerate(updated_command):
        if part in {"-t", "--tag"}:
            tag_index = index + 1
            if tag_index >= len(updated_command):
                raise ValueError(f"`{part}` in Docker build command is missing a tag value.")
            updated_command[tag_index] = tag
            return updated_command
        if part.startswith("--tag="):
            updated_command[index] = f"--tag={tag}"
            return updated_command

    raise ValueError("Docker build command must include `-t` or `--tag` to override the tag.")


def _find_build_tag(command: list[str]) -> str | None:
    """Return the first Docker image tag from a build command."""

    for index, part in enumerate(command):
        if part in {"-t", "--tag"}:
            tag_index = index + 1
            if tag_index >= len(command):
                raise ValueError(f"`{part}` in Docker build command is missing a tag value.")
            return command[tag_index]
        if part.startswith("--tag="):
            return part.removeprefix("--tag=")

    return None


def _is_buildx_command(command: list[str]) -> bool:
    """Report whether a build command uses ``docker buildx build``."""

    return command[:3] == ["docker", "buildx", "build"]


def _single_platform_arch(platforms: str) -> str | None:
    """Return a cache-tag-safe arch key for a single-platform build, else ``None``.

    ``linux/arm64`` yields ``arm64``. ``linux/arm64/v8`` joins the variant with a
    hyphen (``arm64-v8``) so it stays a valid Docker tag rather than smuggling a
    slash in. A multi-platform list, a missing arch, or any other shape yields
    ``None`` so the cache ref keeps its legacy suffixless name.
    """

    entries = [entry.strip() for entry in platforms.split(",") if entry.strip()]
    if len(entries) != 1:
        return None
    parts = entries[0].split("/")
    if len(parts) == 2:
        _, arch = parts
        variant = ""
    elif len(parts) == 3:
        _, arch, variant = parts
    else:
        return None
    # Only alphanumeric arch/variant tokens keep the tag-safe guarantee; anything
    # else falls back to the legacy suffixless cache ref.
    if not arch.isalnum() or (variant and not variant.isalnum()):
        return None
    return f"{arch}-{variant}" if variant else arch


def _registry_cache_ref(image_ref: str, image: BuildableImage, *, arch: str | None = None) -> str:
    """Return a fixed ``buildcache-*`` tag on the same repo as ``image_ref``.

    The push tag carries the commit SHA, so it never repeats and can't seed a
    cache. We keep its repository path and pin a ``buildcache-<image>-<version>``
    tag instead, so the next build pulls the last run's layers with
    ``--cache-from`` rather than rebuilding the media stack.

    When ``arch`` is set, the tag gains an ``-<arch>`` suffix so the amd64 and
    arm64 legs of a split build keep separate caches instead of clobbering.

    We split on the last colon after the final ``/`` so a registry port like
    ``host:5000/team/pkg`` stays intact.
    """

    prefix, separator, remainder = image_ref.rpartition("/")
    repository_name = remainder.partition(":")[0]
    repository = f"{prefix}{separator}{repository_name}"
    arch_suffix = f"-{arch}" if arch else ""
    return f"{repository}:buildcache-{image.image_name}-{image.version}{arch_suffix}"


def _execute_buildx_build(
    command: list[str],
    image: BuildableImage,
    *,
    tag: str | None,
    push: bool,
    platforms: str,
) -> int:
    """Run a ``docker buildx build`` command with multi-architecture handling.

    A multi-architecture build produces an OCI manifest list, which cannot be
    loaded into the local Docker image store. Buildx therefore pushes the
    manifest list to the registry in a single step when ``push`` is set. A build
    without ``push`` cannot ``--load`` a manifest list, so it builds only the
    host architecture for local use.
    """

    if not platforms:
        raise ValueError("Build platforms must not be empty.")

    buildx_command = list(command)
    if push:
        push_ref = tag if tag is not None else _find_build_tag(buildx_command)
        if push_ref is None:
            raise ValueError(
                "Docker buildx build command must include `-t` or `--tag` when using `--push`."
            )
        cache_ref = _registry_cache_ref(push_ref, image, arch=_single_platform_arch(platforms))
        buildx_command += [
            "--platform",
            platforms,
            "--cache-from",
            f"type=registry,ref={cache_ref}",
            "--cache-to",
            f"type=registry,ref={cache_ref},mode=max",
            "--push",
        ]
    else:
        buildx_command += ["--load"]

    print(
        f"Building `{image.target}` with `{shlex.join(_redact_build_args(buildx_command))}`",
        flush=True,
    )
    return subprocess.run(buildx_command, cwd=REPO_ROOT, check=False).returncode


def _append_build_args(command: list[str], build_args: tuple[str, ...]) -> list[str]:
    """Append ``--build-arg KEY=VALUE`` pairs to a Docker build command."""

    updated = list(command)
    for build_arg in build_args:
        updated += ["--build-arg", build_arg]
    return updated


def _redact_build_args(command: list[str]) -> list[str]:
    """Mask ``--build-arg`` values for logging so a caller's value never leaks.

    ``--build-arg`` is a general CLI passthrough, so the value can be anything a
    caller supplies. Keep the key visible but replace the value; the unredacted
    command is still what runs.
    """

    def _mask(pair: str) -> str:
        key, separator, _ = pair.partition("=")
        return f"{key}=***" if separator else "***"

    redacted: list[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        if token == "--build-arg" and index + 1 < len(command):
            redacted += ["--build-arg", _mask(command[index + 1])]
            index += 2
            continue
        if token.startswith("--build-arg="):
            redacted.append(f"--build-arg={_mask(token.removeprefix('--build-arg='))}")
            index += 1
            continue
        redacted.append(token)
        index += 1
    return redacted


def execute_build(
    image: BuildableImage,
    *,
    tag: str | None = None,
    push: bool = False,
    platforms: str = DEFAULT_BUILD_PLATFORMS,
    build_args: tuple[str, ...] = (),
) -> int:
    """Execute a selected Docker image build command.

    The registered command string is parsed with ``shlex.split`` and executed
    without invoking a shell. The subprocess runs from the repository root, so
    registered Dockerfile and build context paths should be written relative to
    that location.

    ``docker buildx build`` commands build for ``platforms`` and, when ``push``
    is set, push the resulting manifest list directly (a manifest list cannot be
    loaded locally). Legacy ``docker build`` commands build a single image and,
    when ``push`` is set, push it in a separate ``docker push`` step.

    Args:
        image: Build target whose registered command should be executed.
        tag: Optional Docker image reference to use instead of the registered tag.
        push: Whether to push the built image after a successful build.
        platforms: Comma-separated target platforms for ``docker buildx`` builds.

    Returns:
        The Docker build or push subprocess exit code.

    Raises:
        ValueError: If tag replacement or push tag discovery fails.
    """

    command = shlex.split(image.command)
    if tag is not None:
        command = _replace_build_tag(command, tag)
    command = _append_build_args(command, build_args)

    if _is_buildx_command(command):
        return _execute_buildx_build(command, image, tag=tag, push=push, platforms=platforms)

    print(
        f"Building `{image.target}` with `{shlex.join(_redact_build_args(command))}`",
        flush=True,
    )
    build_result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if build_result.returncode != 0 or not push:
        return build_result.returncode

    push_tag = tag if tag is not None else _find_build_tag(command)
    if not push_tag:
        raise ValueError("Docker build command must include `-t` or `--tag` when using `--push`.")

    push_command = ["docker", "push", push_tag]
    print(f"Pushing `{push_tag}` with `{shlex.join(push_command)}`", flush=True)
    return subprocess.run(push_command, cwd=REPO_ROOT, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    """Run the workspace image selection and Docker build dispatch flow.

    Args:
        argv: Optional command-line arguments. When omitted, arguments are read
            from ``sys.argv``.

    Returns:
        A process exit code where ``0`` indicates success, ``1`` indicates a
        user-facing configuration or validation error, and ``130`` indicates
        that the interactive selection was cancelled.
    """

    try:
        options = parse_args(argv or sys.argv[1:])
        images = discover_buildable_images()
        if not images:
            raise ValueError("No Docker images were found in the workspace.")

        if options.list_images:
            print(format_image_list(images, options.output_format))
            return 0

        if options.gitlab_ci:
            print(format_gitlab_ci_config(images), end="")
            return 0

        image = choose_image(images, options.target)
        return execute_build(
            image,
            tag=options.tag,
            push=options.push,
            platforms=options.platforms,
            build_args=options.build_args,
        )
    except KeyboardInterrupt:
        print("Docker image selection cancelled.", file=sys.stderr)
        return 130
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
