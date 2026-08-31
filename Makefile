# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
.PHONY: help sync lint lint-check mypy test test-e2e check run build clean

SCRIPT ?=
ARGS ?=
IMAGE ?=
PYTHON ?= python3
GIT ?= git
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
PYTEST_DISABLE_PLUGIN_AUTOLOAD ?= 1
PY_FILE_DIRS ?= packages services scripts tests
PY_FILES ?= $(shell \
	if command -v $(GIT) >/dev/null 2>&1; then \
		$(GIT) ls-files --cached --others --exclude-standard '*.py'; \
	else \
		find $(PY_FILE_DIRS) -type f -name '*.py'; \
	fi)

export UV_CACHE_DIR
export PYTEST_DISABLE_PLUGIN_AUTOLOAD

# Avoid leaking ambient PYTHONPATH, such as ROS site-packages, into uv commands.
# These paths can break pytest plugin discovery; CI usually has PYTHONPATH unset.
export PYTHONPATH :=

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  sync           Install all workspace packages"
	@echo "  lint           Fix lint issues and format code"
	@echo "  lint-check     Check lint and formatting without changes"
	@echo "  mypy           Run mypy type checks"
	@echo "  test           Run tests with coverage report"
	@echo "  test-e2e       Run end-to-end service tests (needs GPU runner + external model endpoints)"
	@echo "  check          Run lint, mypy, and test checks. Make sure this passes before submitting an MR."
	@echo "  run            Select and run a workspace script (optional ARGS=... for script argv)"
	@echo "  build          Select and build a registered Docker image"
	@echo "  clean          Remove build artifacts and .venv"

sync:
	uv sync --all-packages --all-extras

lint: sync
	uv run ruff format $(PY_FILES)
	uv run ruff check --fix $(PY_FILES)

lint-check: sync
	uv run ruff format --check $(PY_FILES)
	uv run ruff check $(PY_FILES)

mypy: sync
	uv run mypy $(PY_FILES)

test: sync
	uv run pytest -p pytest_cov --cov --cov-report=term --cov-report=xml:coverage_report.xml --junitxml=test_report.xml

# End-to-end tests run each service through its real CLI entrypoint. Tests self-skip
# when their prerequisites (external model endpoints, ffmpeg) are unavailable.
# --basetemp writes each test's scene outputs to e2e_artifacts/ so they can be
# inspected locally and uploaded as browsable CI artifacts.
test-e2e: sync
	uv run pytest -m e2e tests/e2e --basetemp=e2e_artifacts --junitxml=e2e_report.xml

check: sync lint mypy test

run: sync
	uv run --group dev python scripts/run.py $(SCRIPT) -- $(ARGS)

build:
	$(PYTHON) scripts/build.py $(IMAGE)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	rm -rf .venv .uv-cache dist build
