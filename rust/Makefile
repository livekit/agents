.DEFAULT_GOAL := help

.PHONY: help format format-check clippy architecture check

help:
	@printf '%s\n' \
		'make format        Format the Rust workspace' \
		'make format-check  Check Rust formatting' \
		'make clippy        Run the shared purity policy on ABI, core, voice and media' \
		'make architecture  Run the full architecture gate, fixtures and codegen checks' \
		'make check         Check formatting and architecture'

format:
	cargo fmt --all

format-check:
	cargo fmt --all -- --check

# One config for this explicit package group. Other policies get separate invocations.
# This is a diagnostic pass; architecture also validates exemptions and baselines.
clippy:
	CLIPPY_CONF_DIR="$(CURDIR)/architecture" cargo clippy \
		--locked --no-deps --all-targets \
		-p lk-agents-abi -p lk-agents-core -p lk-agents-voice -p lk-agents-media

architecture:
	bash architecture/check.sh

check: format-check architecture
