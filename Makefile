.PHONY: help install format format-check lint lint-fix fix type-check check test-e2e \
        link-rtc link-rtc-local unlink-rtc status doctor build build-debug rust-check \
        test bench bench-startup bench-host bench-sentences bench-sync

ROOT := $(CURDIR)
AGENTS_PROJECT := $(ROOT)/python/livekit-agents
PYTHON_SDKS := $(abspath $(ROOT)/../python-sdks)
RUST_SDKS := $(abspath $(ROOT)/../rust-sdks)
PYTHON_RTC := $(PYTHON_SDKS)/livekit-rtc
# The FFI crate livekit-rtc is built against: its own rust-sdks submodule.
RTC_RUST_SDK := $(PYTHON_RTC)/rust-sdks
ifeq ($(OS),Windows_NT)
HOST_OS := windows
ARCH := $(if $(filter ARM64,$(PROCESSOR_ARCHITECTURE)),aarch64,x86_64)
else
HOST_OS := $(shell uname -s | tr A-Z a-z)
ARCH := $(shell uname -m)
endif
FFI_OS := $(if $(filter darwin,$(HOST_OS)),macos,$(HOST_OS))
FFI_ARCH := $(if $(filter aarch64,$(ARCH)),arm64,$(ARCH))
FFI_LIB := $(if $(filter windows,$(HOST_OS)),livekit_ffi.dll,$(if $(filter darwin,$(HOST_OS)),liblivekit_ffi.dylib,liblivekit_ffi.so))

.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install: ## Install all dependencies with dev extras
	uv sync --all-extras --dev

build: ## Rebuild the Python extension in place with the fast dev profile
	$(MAKE) -C rust python

build-debug: ## Rebuild the Python extension with native debug symbols
	$(MAKE) -C rust python-debug

rust-check: ## Check all Rust crates and test targets, including FFmpeg
	$(MAKE) -C rust check

format: ## Format code with ruff
	uv run ruff format .

format-check: ## Check code formatting without changing files
	uv run ruff format --check .

lint: ## Run ruff linter
	uv run ruff check .

lint-fix: ## Run ruff linter and fix issues
	uv run ruff check --fix .

fix: format lint-fix ## Format and fix lint issues

type-check: ## Run strict mypy and pyright
	uv run python .github/check_types.py $(TYPE_CHECK_ARGS)

check: format-check lint type-check ## Run all checks (format-check, lint, type-check)

test: ## Run the hermetic Python suites (host, plugins, smoke); pass PYTEST_ARGS
	uv run pytest tests $(PYTEST_ARGS)

test-e2e: ## Run tests/e2e against a LiveKit project (LIVEKIT_URL, _API_KEY, _API_SECRET)
	uv run pytest tests/e2e --e2e $(PYTEST_ARGS)

bench: ## Run the IPC benchmark; pass BENCH_ARGS (--help lists them)
	uv run python -m bench.ipc $(BENCH_ARGS)

bench-startup: ## Time start, dev and console from launch to ready, cold caches; pass BENCH_ARGS
	uv run --no-sync python -m bench.startup $(BENCH_ARGS)

bench-host: ## Time the Rust-to-Python host boundary (calls, context, streams, cancellation)
	uv run --no-sync cargo test --release --manifest-path rust/Cargo.toml -p lk-agents-py \
		--features ffmpeg --test host_bench -- --ignored --nocapture

bench-sentences: ## Time the sentence splitter's pushes on the experiment corpus and adversarial replies
	cargo test --release --manifest-path rust/Cargo.toml -p lk-agents-runtime --test sentence_bench \
		-- --ignored --nocapture --test-threads=1

LK_SYNC_CORPUS ?= $(HOME)/livekit/sync-corpus
bench-sync: ## Transcript sync accuracy over a recorded corpus (LK_SYNC_CORPUS); appends to its history.tsv
	LK_SYNC_CORPUS=$(LK_SYNC_CORPUS) LK_SYNC_COMMIT=$$(git describe --always --dirty) \
		cargo test --release --manifest-path rust/Cargo.toml -p lk-agents-voice --lib \
		transcript_sync_accuracy -- --ignored --nocapture

link-rtc: ## Link ../python-sdks/livekit-rtc with downloaded FFI artifacts
	cd $(PYTHON_RTC) && python3 rust-sdks/download_ffi.py --platform $(FFI_OS) --arch $(FFI_ARCH) \
		--output livekit/rtc/resources
	uv add --package livekit-agents --editable $(PYTHON_RTC)
	uv sync --all-extras --dev

link-rtc-local: ## Build the FFI from livekit-rtc's rust-sdks and link ../python-sdks/livekit-rtc
	cd $(RTC_RUST_SDK) && cargo build --release -p livekit-ffi
	cd $(PYTHON_RTC) && ./generate_proto.sh
	uv add --package livekit-agents --editable $(PYTHON_RTC)
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
