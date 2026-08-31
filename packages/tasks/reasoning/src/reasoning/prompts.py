# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt registry for DAFT-export LLM aggregation stages.

Prompts live as YAML files on disk, loaded into a :class:`PromptRegistry`
at runtime. This is the "data, not code" pattern the project uses for
question banks (``mcq_generation/mcq/utils/bank.py``); here we apply it
to the LLM-aggregation prompts that drive ``msted.json``,
``temporal_localization.json``, and the reasoning-enrichment passes.

Design principles (the explicit use-case agnosticism the user asked for):

- **No prompts are hard-coded in Python.** The bundled defaults are YAML
  files in :data:`BUNDLED_PROMPT_DIR`; the registry constructor accepts
  any number of additional directories, and any YAML file matching
  ``<name>.yaml`` overrides the bundled variant of the same name.
- **Variants are first-class.** Switching from a generic prompt to a
  domain-specific one (traffic safety, warehouse safety, smart-city) is
  a single config knob (``prompt_variant: "av-surveillance"``) plus a
  single YAML drop-in. No code change.
- **Prompts are typed.** :class:`PromptVariant` declares the known
  fields (``name``, ``description``, ``system``, ``user_template``).
  Callers know exactly what to format; missing keys raise at load time
  rather than failing the LLM call later.
- **Templates are ``str.format`` strings.** No Jinja, no eval — just
  ``{key}`` placeholders the caller fills in. Format errors (unknown
  placeholder) raise clearly at render time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# All bundled prompt YAMLs live under ``data/prompts/`` next to this
# module. The directory is data-only (no Python __init__.py) and named
# ``data/`` to keep it visually separate from the ``prompts.py`` module
# that owns the loader — sidestepping the awkwardness of a sibling
# ``prompts/`` directory shadowing this very module name.
BUNDLED_PROMPT_DIR: Path = Path(__file__).resolve().parent / "data" / "prompts"


class PromptError(ValueError):
    """Raised when a prompt YAML fails to load or render.

    Distinct from :class:`reasoning.common.DaftConvertError` because
    prompt failures are configuration / authoring errors, not data
    conversion errors — they should fail loudly at startup or first
    LLM call rather than be swallowed by the per-scene best-effort
    error handling that wraps the converter calls."""


@dataclass(frozen=True)
class PromptVariant:
    """A named LLM prompt template loaded from a YAML file.

    ``name`` is the registry key (matches the YAML filename stem).
    ``system`` is sent verbatim as the ``role: system`` message.
    ``user_template`` is a ``str.format`` template; the caller passes
    domain-specific values (window captions, scene description, schema
    hint) via :meth:`render_user`.

    ``description`` is human-facing only — surfaced in error messages
    and ``--list-prompts`` style introspection."""

    name: str
    system: str
    user_template: str
    description: str = ""

    def render_user(self, **kwargs: Any) -> str:
        """Render ``user_template`` by substituting ``{key}`` placeholders.

        Wraps ``str.format`` so a typo in a placeholder produces a
        :class:`PromptError` pointing at the variant by name, instead of
        a bare ``KeyError`` that's hard to trace back to a YAML file."""
        try:
            return self.user_template.format(**kwargs)
        except KeyError as exc:
            raise PromptError(
                f"prompt variant {self.name!r} references unknown placeholder {exc.args[0]!r}; "
                f"caller passed keys: {sorted(kwargs)}"
            ) from exc
        except IndexError as exc:
            raise PromptError(
                f"prompt variant {self.name!r} has a malformed positional placeholder: {exc}"
            ) from exc
        except ValueError as exc:
            raise PromptError(
                f"prompt variant {self.name!r} has a malformed template: {exc}"
            ) from exc


def _load_variant(yaml_path: Path) -> PromptVariant:
    """Parse a single YAML file into a :class:`PromptVariant`.

    The schema is intentionally minimal: ``system`` and ``user_template``
    are required, ``name`` defaults to the filename stem, ``description``
    is optional. Anything else is rejected so typos in field names
    surface immediately."""
    try:
        text = yaml_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptError(f"cannot read prompt YAML {yaml_path}: {exc}") from exc

    try:
        obj = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PromptError(f"prompt YAML {yaml_path} is not valid YAML: {exc}") from exc

    if not isinstance(obj, dict):
        raise PromptError(
            f"prompt YAML {yaml_path} must be a mapping at the top level, got {type(obj).__name__}"
        )

    allowed_keys = {"name", "description", "system", "user_template"}
    extra = set(obj) - allowed_keys
    if extra:
        raise PromptError(
            f"prompt YAML {yaml_path} has unknown keys: {sorted(extra)}; "
            f"allowed keys are {sorted(allowed_keys)}"
        )

    name = str(obj.get("name") or yaml_path.stem).strip()
    if not name:
        raise PromptError(f"prompt YAML {yaml_path} has empty 'name'")

    system = obj.get("system")
    if not isinstance(system, str) or not system.strip():
        raise PromptError(f"prompt YAML {yaml_path} missing non-empty 'system' string")

    user_template = obj.get("user_template")
    if not isinstance(user_template, str) or not user_template.strip():
        raise PromptError(f"prompt YAML {yaml_path} missing non-empty 'user_template' string")

    description = str(obj.get("description") or "").strip()

    return PromptVariant(
        name=name,
        system=system,
        user_template=user_template,
        description=description,
    )


class PromptRegistry:
    """Filesystem-backed registry of named LLM prompt variants.

    Construction order (later entries win on name collision so callers
    can override the bundled defaults):

    1. The bundled directory at :data:`BUNDLED_PROMPT_DIR` (skipped when
       ``include_bundled=False``).
    2. Each directory in ``extra_dirs``, in order.

    Within a directory, every ``*.yaml`` and ``*.yml`` file is loaded;
    files starting with ``_`` are skipped (reserved for fragments /
    helpers).

    Use :meth:`get` to look up a variant by name. Use :meth:`names` for
    introspection (e.g. CLI help). The registry is frozen after
    construction — to add a variant, build a new registry."""

    def __init__(
        self,
        *,
        extra_dirs: Iterable[Path] | None = None,
        include_bundled: bool = True,
    ) -> None:
        variants: dict[str, PromptVariant] = {}
        sources: dict[str, Path] = {}

        dirs: list[Path] = []
        if include_bundled and BUNDLED_PROMPT_DIR.is_dir():
            dirs.append(BUNDLED_PROMPT_DIR)
        for d in extra_dirs or ():
            dp = Path(d).expanduser()
            if not dp.is_dir():
                raise PromptError(f"extra prompt directory does not exist: {dp}")
            dirs.append(dp)

        for d in dirs:
            for yaml_path in sorted(d.iterdir()):
                if not yaml_path.is_file():
                    continue
                if yaml_path.suffix.lower() not in (".yaml", ".yml"):
                    continue
                if yaml_path.name.startswith("_"):
                    continue
                variant = _load_variant(yaml_path)
                variants[variant.name] = variant
                sources[variant.name] = yaml_path

        self._variants: Mapping[str, PromptVariant] = variants
        self._sources: Mapping[str, Path] = sources

    def get(self, name: str) -> PromptVariant:
        """Return the variant registered under ``name`` or raise.

        The error message lists all known names, so a typo in
        ``prompt_variant`` config surfaces with actionable feedback."""
        v = self._variants.get(name)
        if v is None:
            raise PromptError(
                f"prompt variant {name!r} not found; known variants: {sorted(self._variants)}"
            )
        return v

    def names(self) -> list[str]:
        """Sorted list of registered variant names (introspection)."""
        return sorted(self._variants)

    def source(self, name: str) -> Path:
        """Path of the YAML file that defines ``name`` (introspection / errors)."""
        p = self._sources.get(name)
        if p is None:
            raise PromptError(f"prompt variant {name!r} not found")
        return p


__all__ = [
    "BUNDLED_PROMPT_DIR",
    "PromptError",
    "PromptRegistry",
    "PromptVariant",
]
