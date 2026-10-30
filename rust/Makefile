.DEFAULT_GOAL := help

.PHONY: help require-cargo-runner build check python python-debug format format-check clippy test-checks

ROOT := $(CURDIR)/..
# An executable, e.g. cargo, mbx, or an absolute path to either. No global shim required.
CARGO_RUNNER ?= mbx
export CARGO_RUNNER
MATURIN ?= maturin
VIRTUAL_ENV ?= $(ROOT)/.venv
ifeq ($(OS),Windows_NT)
PYO3_PYTHON ?= $(VIRTUAL_ENV)/Scripts/python.exe
else
PYO3_PYTHON ?= $(VIRTUAL_ENV)/bin/python
endif
export PYO3_PYTHON VIRTUAL_ENV
BUILD_ARGS ?=

help:
	@printf '%s\n' \
		'make build         Build every workspace crate, including FFmpeg' \
		'make check         Check compilation, Clippy, boundaries and codegen' \
		'make python        Rebuild the Python extension in place (run make install first)' \
		'make python-debug  Rebuild the Python extension with native debug symbols' \
		'make format        Format the Rust workspace' \
		'make format-check  Check Rust formatting' \
		'make clippy        Run the shared purity policy on ABI, core, voice and media' \
		'make test-checks   Test the checkers and code generator' \
		'' \
		'Rust commands use mbx by default; pass CARGO_RUNNER=cargo to bypass it.'

# Check only targets that invoke this runner; help and formatting need no cache tool.
require-cargo-runner:
	@command -v "$(CARGO_RUNNER)" >/dev/null 2>&1 || { \
		printf "error: Rust command '%s' was not found.\n" "$(CARGO_RUNNER)" >&2; \
		printf '%s\n' \
			'Install mbx: cargo install --locked mbx' \
			'Or rerun make with CARGO_RUNNER=cargo.' >&2; \
		exit 1; \
	}

build check clippy test-checks: | require-cargo-runner

build:
	"$(CARGO_RUNNER)" build --locked --workspace --features lk-agents-py/ffmpeg $(BUILD_ARGS)

check:
	bash architecture/check.sh

python:
	cd "$(ROOT)/livekit-agents-next" && "$(MATURIN)" develop --skip-install --locked $(BUILD_ARGS)

# Keep Maturin's configured ABI/media features; --features would replace them.
# RUSTFLAGS restores the platform debug format but must retain the simulation cfg.
python-debug: export RUSTFLAGS = --cfg tokio_unstable
python-debug: export LK_AGENTS_NATIVE_DEBUG = 1
python-debug:
	cd "$(ROOT)/livekit-agents-next" && "$(MATURIN)" develop --skip-install --locked \
		--config 'profile.dev.package.lk-agents-py.debug=1' $(BUILD_ARGS)

format:
	cargo fmt --all

format-check:
	cargo fmt --all -- --check

# One config for this explicit package group. Other policies get separate invocations.
# This is a diagnostic pass; architecture also validates exemptions and baselines.
clippy:
	CLIPPY_CONF_DIR="$(CURDIR)/architecture" "$(CARGO_RUNNER)" clippy \
		--locked --no-deps --all-targets \
		-p lk-agents-abi -p lk-agents-core -p lk-agents-voice -p lk-agents-media

test-checks:
	bash architecture/check.sh --tests
