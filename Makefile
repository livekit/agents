.PHONY: help install format format-check lint lint-fix fix type-check check unit-tests \
        link-rtc link-rtc-local unlink-rtc status doctor

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
AGENTS_PROJECT := $(ROOT)/livekit-agents-next
PYTHON_SDKS := $(abspath $(ROOT)/../python-sdks)
RUST_SDKS := $(abspath $(ROOT)/../rust-sdks)
PYTHON_RTC := $(PYTHON_SDKS)/livekit-rtc
# The FFI crate livekit-rtc is built against: its own rust-sdks submodule.
RTC_RUST_SDK := $(PYTHON_RTC)/rust-sdks

OS := $(shell uname -s | tr A-Z a-z)
ARCH := $(shell uname -m)
FFI_OS := $(if $(filter darwin,$(OS)),macos,$(OS))
FFI_ARCH := $(if $(filter aarch64,$(ARCH)),arm64,$(ARCH))
FFI_LIB := $(if $(filter darwin,$(OS)),liblivekit_ffi.dylib,liblivekit_ffi.so)

.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install: ## Install all dependencies with dev extras
	uv sync --all-extras --dev

format: ## Format code with ruff
	uv run ruff format .

format-check: ## Check code formatting without changing files
	uv run ruff format --check .

lint: ## Run ruff linter
	uv run ruff check .

lint-fix: ## Run ruff linter and fix issues
	uv run ruff check --fix .

fix: format lint-fix ## Format and fix lint issues

type-check: ## Run strict mypy and pyright (scripts/check_types.py)
	uv run python scripts/check_types.py

check: format-check lint type-check ## Run all checks (format-check, lint, type-check)

unit-tests: ## Run the unit and audio_eot test categories (no cloud accounts)
	PYTHONPATH="$(ROOT)" uv run pytest --unit --audio_eot $(PYTEST_ARGS)

link-rtc: ## Link ../python-sdks/livekit-rtc with downloaded FFI artifacts
	cd $(PYTHON_RTC) && python3 rust-sdks/download_ffi.py --platform $(FFI_OS) --arch $(FFI_ARCH) \
		--output livekit/rtc/resources
	uv add --package livekit-agents-next --editable $(PYTHON_RTC)
	uv sync --all-extras --dev

link-rtc-local: ## Build the FFI from livekit-rtc's rust-sdks and link ../python-sdks/livekit-rtc
	cd $(RTC_RUST_SDK) && cargo build --release -p livekit-ffi
	cd $(PYTHON_RTC) && ./generate_proto.sh
	uv add --package livekit-agents-next --editable $(PYTHON_RTC)
	uv sync --all-extras --dev
	@echo "Use the local FFI with: export LIVEKIT_LIB_PATH=$(RTC_RUST_SDK)/target/release/$(FFI_LIB)"

unlink-rtc: ## Restore the pinned PyPI livekit
	python3 -c 'import re, sys; p = sys.argv[1]; t = open(p).read(); \
		t = re.sub(r"^livekit = \{ path = .*\n", "", t, flags=re.M); \
		t = re.sub(r"\n*\[tool\.uv\.sources\]\n*(?=\[|\Z)", "\n", t); open(p, "w").write(t)' \
		$(AGENTS_PROJECT)/pyproject.toml
	uv sync --all-extras --dev

status: ## Show where livekit (rtc) and its FFI come from
	@info=$$(uv pip show livekit 2>/dev/null); \
	version=$$(echo "$$info" | awk '/^Version:/ {print $$2}'); \
	location=$$(echo "$$info" | sed -n 's/^Editable project location: //p'); \
	if [ -z "$$info" ]; then echo "livekit: not installed"; \
	elif [ -n "$$location" ]; then echo "livekit: local editable v$$version ($$location)"; \
	else echo "livekit: PyPI v$$version"; fi; \
	if [ -n "$$LIVEKIT_LIB_PATH" ]; then echo "FFI: local build ($$LIVEKIT_LIB_PATH)"; \
	elif [ -n "$$location" ]; then echo "FFI: downloaded artifacts ($$location/livekit/rtc/resources)"; \
	else echo "FFI: bundled with the PyPI wheel"; fi

doctor: ## Check the development environment
	@issues=0; \
	for tool in uv python3 cargo git; do \
		if command -v $$tool >/dev/null 2>&1; then echo "ok      $$tool"; \
		else echo "MISSING $$tool"; issues=$$((issues + 1)); fi; \
	done; \
	for dir in $(PYTHON_SDKS) $(RUST_SDKS); do \
		if [ -d "$$dir" ]; then echo "ok      $$dir"; \
		else echo "MISSING $$dir (checkout beside this repository)"; issues=$$((issues + 1)); fi; \
	done; \
	if [ -d "$(ROOT)/.venv" ]; then echo "ok      .venv"; \
	else echo "MISSING .venv (run 'make install')"; issues=$$((issues + 1)); fi; \
	if [ -n "$$LIVEKIT_LIB_PATH" ] && [ ! -f "$$LIVEKIT_LIB_PATH" ]; then \
		echo "MISSING LIVEKIT_LIB_PATH file $$LIVEKIT_LIB_PATH"; issues=$$((issues + 1)); fi; \
	[ $$issues -eq 0 ] || { echo "$$issues issue(s)"; exit 1; }
