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
		'make check         Run every automatic rule: fmt, Clippy, rust/check and codegen drift' \
		'make python        Rebuild the Python extension in place (run make install first)' \
		'make python-debug  Rebuild the Python extension with native debug symbols' \
		'make format        Format the Rust workspace' \
		'make format-check  Check Rust formatting' \
		'make clippy        Run Clippy with the purity, I/O and base policies, denying warnings' \
		'make test-checks   Test rust/check and the code generator' \
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

# AGENTS.md "Rust architecture rules": every [check] and [lint] rule.
check: format-check clippy
	"$(CARGO_RUNNER)" run --locked -q -p lk-agents-check
	"$(CARGO_RUNNER)" run --locked -q -p lk-agents-codegen -- --check
	python3 ai_slop/lk-agents-codegen/thread_api.py --check

python:
	cd "$(ROOT)/python/livekit-agents" && "$(MATURIN)" develop --skip-install --locked $(BUILD_ARGS)

# Keep Maturin's configured ABI/media features; --features would replace them.
# RUSTFLAGS restores the platform debug format but must retain the simulation cfg.
python-debug: export RUSTFLAGS = --cfg tokio_unstable
python-debug: export LK_AGENTS_NATIVE_DEBUG = 1
python-debug:
	cd "$(ROOT)/python/livekit-agents" && "$(MATURIN)" develop --skip-install --locked \
		--config 'profile.dev.package.lk-agents-py.debug=1' $(BUILD_ARGS)

format:
	cargo fmt

format-check:
	cargo fmt -- --check

PURITY := -p lk-agents-abi -p lk-agents-core -p lk-agents-voice -p lk-agents-media
IO := -p lk-agents-runtime -p lk-agents-worker -p lk-agents-net -p lk-agents-inference \
	-p lk-agents-storage -p lk-agents-py -p lk-agents-cli -p lk-agents-telemetry
BASE := -p lk-agents-fuzz -p lk-agents-codegen -p lk-agents-bindgen -p lk-agents-host-derive \
	-p lk-agents-check
CLIPPY := "$(CARGO_RUNNER)" clippy --locked --no-deps --all-targets --all-features

# One Clippy policy per invocation: rust/check/purity, rust/check/io, then rust/clippy.toml.
clippy:
	CLIPPY_CONF_DIR="$(CURDIR)/check/purity" $(CLIPPY) $(PURITY) -- -D warnings
	CLIPPY_CONF_DIR="$(CURDIR)/check/io" $(CLIPPY) $(IO) -- -D warnings
	$(CLIPPY) $(BASE) -- -D warnings

test-checks:
	"$(CARGO_RUNNER)" test --locked -q -p lk-agents-check -p lk-agents-codegen --bins
