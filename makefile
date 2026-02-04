.PHONY: help install format format-check lint lint-fix check type-check test test-unit test-docker clean build \
        link-rtc link-rtc-local link-rtc-version unlink-rtc status doctor

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m
BOLD := \033[1m

# Paths (computed as absolute paths)
MAKEFILE_DIR := $(shell pwd)
AGENTS_PROJECT := $(MAKEFILE_DIR)/livekit-agents
PYTHON_RTC := $(MAKEFILE_DIR)/../python-sdks/livekit-rtc
RUST_SUBMODULE := $(MAKEFILE_DIR)/../python-sdks/livekit-rtc/rust-sdks
PACKAGE_NAME := livekit

# Platform and architecture auto-detection
ARCH := $(shell uname -m)
OS := $(shell uname -s | tr A-Z a-z)

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "$(BOLD)$(CYAN)Available targets:$(RESET)"
	@echo ""
	@echo "$(BOLD)Development Workflows:$(RESET)"
	@grep -E '^(link-rtc|link-rtc-local|link-rtc-version|unlink-rtc|status|doctor):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Code Quality:$(RESET)"
	@grep -E '^(format|format-check|lint|lint-fix|type-check|check):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)Other:$(RESET)"
	@grep -E '^(install|clean|build):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'

install: ## Install all dependencies with dev extras
	@echo "$(BOLD)$(CYAN)Installing dependencies...$(RESET)"
	@uv sync --all-extras --dev
	@echo "$(BOLD)$(GREEN)✓ Dependencies installed$(RESET)"

format: ## Format code with ruff
	@echo "$(BOLD)$(CYAN)Formatting code...$(RESET)"
	@uv run ruff format .
	@echo "$(BOLD)$(GREEN)✓ Code formatted$(RESET)"

format-check: ## Check code formatting without making changes
	@echo "$(BOLD)$(CYAN)Checking code formatting...$(RESET)"
	@if uv run ruff format --check .; then \
		echo "$(BOLD)$(GREEN)✓ Code formatting is correct$(RESET)"; \
	else \
		echo "$(BOLD)$(RED)✗ Code formatting issues found. Run 'make format' to fix.$(RESET)"; \
		exit 1; \
	fi

lint: ## Run ruff linter
	@echo "$(BOLD)$(CYAN)Running linter...$(RESET)"
	@if uv run ruff check .; then \
		echo "$(BOLD)$(GREEN)✓ No linting issues found$(RESET)"; \
	else \
		echo "$(BOLD)$(RED)✗ Linting issues found. Run 'make fix' to automatically fix them.$(RESET)"; \
		exit 1; \
	fi

lint-fix: ## Run ruff linter and fix issues automatically
	@echo "$(BOLD)$(CYAN)Running linter with auto-fix...$(RESET)"
	@uv run ruff check --fix .
	@echo "$(BOLD)$(GREEN)✓ Linting complete$(RESET)"

type-check: ## Run mypy type checker
	@echo "$(BOLD)$(CYAN)Running type checker...$(RESET)"
	@if uv run python scripts/check_types.py; then \
		echo "$(BOLD)$(GREEN)✓ Type checking passed$(RESET)"; \
	else \
		echo "$(BOLD)$(RED)✗ Type checking failed$(RESET)"; \
		exit 1; \
	fi

check: format-check lint type-check ## Run all checks (format, lint, type-check)
	@echo "$(BOLD)$(GREEN)✓ All checks passed!$(RESET)"

fix: format lint-fix ## Run format and lint checks and fix issues automatically (format, lint)

# ============================================
# Development Workflows
# ============================================

link-rtc: ## Link to local python-rtc (default FFI version)
	@echo "$(BOLD)$(CYAN)🐍 Linking local python-rtc with default FFI version...$(RESET)"
	@set -e; \
	DETECTED_ARCH="$(ARCH)"; \
	DETECTED_OS="$(OS)"; \
	if [ "$$DETECTED_ARCH" = "aarch64" ]; then \
		PLATFORM_ARCH="arm64"; \
	else \
		PLATFORM_ARCH="$$DETECTED_ARCH"; \
	fi; \
	if [ "$$DETECTED_OS" = "darwin" ]; then \
		PLATFORM_OS="macos"; \
	else \
		PLATFORM_OS="$$DETECTED_OS"; \
	fi; \
	echo "$(CYAN)📦 Downloading FFI artifacts for $$PLATFORM_OS-$$PLATFORM_ARCH...$(RESET)"; \
	cd $(PYTHON_RTC) && python rust-sdks/download_ffi.py --platform "$$PLATFORM_OS" --arch "$$PLATFORM_ARCH" --output livekit/rtc/resources; \
	echo "$(CYAN)🔗 Adding local python-rtc to agents...$(RESET)"; \
	cd $(AGENTS_PROJECT) && uv add --editable "../../python-sdks/livekit-rtc" && uv sync; \
	echo "$(BOLD)$(GREEN)✅ Linked to local python-rtc (with default FFI version)$(RESET)"

link-rtc-local: ## Build and link local rust SDK from source
	@echo "$(BOLD)$(CYAN)🦀 Building and linking local rust SDK...$(RESET)"
	@set -e; \
	echo "$(CYAN)🦀 Building livekit-ffi...$(RESET)"; \
	cd $(RUST_SUBMODULE) && cargo build --release -p livekit-ffi; \
	echo "$(CYAN)📝 Generating protobuf FFI protocol...$(RESET)"; \
	cd $(PYTHON_RTC) && ./generate_proto.sh; \
	RUST_LIB_DIR="$$(cd $(RUST_SUBMODULE) && pwd)/target/release"; \
	if [ "$(OS)" = "darwin" ]; then \
		RUST_LIB_PATH="$$RUST_LIB_DIR/liblivekit_ffi.dylib"; \
	elif [ "$(OS)" = "linux" ]; then \
		RUST_LIB_PATH="$$RUST_LIB_DIR/liblivekit_ffi.so"; \
	else \
		RUST_LIB_PATH="$$RUST_LIB_DIR/livekit_ffi.dll"; \
	fi; \
	echo "$(CYAN)   LIVEKIT_LIB_PATH=$$RUST_LIB_PATH$(RESET)"; \
	echo "$(CYAN)🔗 Adding local python-rtc to agents...$(RESET)"; \
	cd $(AGENTS_PROJECT) && uv add --editable "../../python-sdks/livekit-rtc" && uv sync; \
	echo "$(BOLD)$(GREEN)✅ Linked to local rust-sdk + python-rtc$(RESET)"; \
	echo ""; \
	echo "$(BOLD)$(YELLOW)📋 To use the local rust lib in your terminal, run:$(RESET)"; \
	echo "$(BOLD)   export LIVEKIT_LIB_PATH=$$RUST_LIB_PATH$(RESET)"

link-rtc-version: ## Install specific livekit version from PyPI (e.g., make link-rtc-version VERSION=0.18.0)
ifndef VERSION
	@echo "$(BOLD)$(RED)Error: VERSION not specified. Use: make link-rtc-version VERSION=0.18.0$(RESET)"
	@exit 1
endif
	@echo "$(BOLD)$(CYAN)📦 Installing livekit version $(VERSION)...$(RESET)"
	@set -e; \
	if grep -q "^livekit = { path = " pyproject.toml 2>/dev/null; then \
		echo "$(CYAN)   Removing workspace root livekit source override...$(RESET)"; \
		sed -i.bak '/^livekit = { path = /d' pyproject.toml && rm -f pyproject.toml.bak; \
	fi; \
	cd $(AGENTS_PROJECT); \
	if grep -q "\[tool.uv.sources\]" pyproject.toml && grep -q "^livekit = " pyproject.toml; then \
		echo "$(CYAN)   Removing [tool.uv.sources] override...$(RESET)"; \
		sed -i.bak '/^livekit = /d' pyproject.toml && rm -f pyproject.toml.bak; \
	fi; \
	uv remove $(PACKAGE_NAME) 2>/dev/null || true; \
	uv add "$(PACKAGE_NAME)==$(VERSION)" && uv sync; \
	echo "$(BOLD)$(GREEN)✅ Linked to livekit $(VERSION)$(RESET)"

unlink-rtc: ## Unlink local and restore PyPI version
	@echo "$(BOLD)$(CYAN)🔓 Unlinking local python-rtc...$(RESET)"
	@set -e; \
	if grep -q "^livekit = { path = " pyproject.toml 2>/dev/null; then \
		echo "$(CYAN)   Removing workspace root livekit source override...$(RESET)"; \
		sed -i.bak '/^livekit = { path = /d' pyproject.toml && rm -f pyproject.toml.bak; \
	fi; \
	cd $(AGENTS_PROJECT); \
	if grep -q "\[tool.uv.sources\]" pyproject.toml; then \
		echo "$(CYAN)   Removing [tool.uv.sources] override...$(RESET)"; \
		sed -i.bak '/\[tool.uv.sources\]/,/^livekit = /d' pyproject.toml && rm -f pyproject.toml.bak; \
	fi; \
	uv remove --dev $(PACKAGE_NAME) 2>/dev/null || true; \
	uv add --upgrade-package $(PACKAGE_NAME) $(PACKAGE_NAME) && uv sync; \
	echo "$(BOLD)$(GREEN)✅ Restored PyPI version$(RESET)"

status: ## Show current linking status
	@echo "$(BOLD)$(CYAN)📍 Current status:$(RESET)"
	@echo ""
	@set -e; \
	RUST_SUBMODULE_DIR="$$(cd $(RUST_SUBMODULE) 2>/dev/null && pwd || echo "")"; \
	cd $(AGENTS_PROJECT); \
	SHOW_OUTPUT=$$(uv pip show livekit 2>/dev/null || echo ""); \
	IS_LOCAL_EDITABLE=false; \
	if [ -z "$$SHOW_OUTPUT" ]; then \
		echo "   livekit: NOT INSTALLED"; \
	elif echo "$$SHOW_OUTPUT" | grep -q "Editable project location:"; then \
		IS_LOCAL_EDITABLE=true; \
		VERSION=$$(echo "$$SHOW_OUTPUT" | grep "^Version:" | awk '{print $$2}'); \
		LOCATION=$$(echo "$$SHOW_OUTPUT" | grep "Editable project location:" | cut -d' ' -f4-); \
		echo "   livekit: LOCAL (editable) v$$VERSION"; \
		echo "   path:    $$LOCATION"; \
	else \
		VERSION=$$(echo "$$SHOW_OUTPUT" | grep "^Version:" | awk '{print $$2}'); \
		echo "   livekit: PyPI (v$$VERSION)"; \
	fi; \
	echo ""; \
	if [ -n "$$LIVEKIT_LIB_PATH" ]; then \
		SUBMODULE_COMMIT=$$([ -n "$$RUST_SUBMODULE_DIR" ] && cd "$$RUST_SUBMODULE_DIR" 2>/dev/null && git rev-parse --short HEAD || echo 'unknown'); \
		echo "   FFI: LOCAL BUILD (rust-sdks @ $$SUBMODULE_COMMIT)"; \
		echo "   path: $$LIVEKIT_LIB_PATH"; \
	elif [ "$$IS_LOCAL_EDITABLE" = "true" ]; then \
		FFI_PATH="$$(cd $(PYTHON_RTC) 2>/dev/null && pwd || echo "")/livekit/rtc/resources"; \
		if [ -d "$$FFI_PATH" ] && { [ -f "$$FFI_PATH/liblivekit_ffi.dylib" ] || [ -f "$$FFI_PATH/liblivekit_ffi.so" ] || [ -f "$$FFI_PATH/livekit_ffi.dll" ]; }; then \
			CARGO_TOML_PATH="$$RUST_SUBMODULE_DIR/livekit-ffi/Cargo.toml"; \
			if [ -f "$$CARGO_TOML_PATH" ]; then \
				FFI_VERSION=$$(grep '^version = ' "$$CARGO_TOML_PATH" | head -1 | sed 's/.*"\(.*\)".*/\1/'); \
				echo "   FFI: PRE-BUILT ARTIFACTS (v$$FFI_VERSION)"; \
			else \
				echo "   FFI: PRE-BUILT ARTIFACTS"; \
			fi; \
		else \
			echo "   FFI: NOT AVAILABLE"; \
		fi; \
	else \
		echo "   FFI: BUNDLED (PyPI package)"; \
	fi

doctor: ## Check development environment health
	@echo "$(BOLD)$(CYAN)🏥 Running diagnostics...$(RESET)"
	@echo ""
	@ISSUES=0; \
	echo "$(BOLD)📦 Required Tools:$(RESET)"; \
	if command -v uv &> /dev/null; then \
		UV_VERSION=$$(uv --version 2>&1 | head -1); \
		echo "   ✓ uv: $$UV_VERSION"; \
	else \
		echo "   ✗ uv: NOT FOUND"; \
		ISSUES=$$((ISSUES + 1)); \
	fi; \
	if command -v python &> /dev/null; then \
		PYTHON_VERSION=$$(python --version 2>&1); \
		echo "   ✓ python: $$PYTHON_VERSION"; \
	else \
		echo "   ✗ python: NOT FOUND"; \
		ISSUES=$$((ISSUES + 1)); \
	fi; \
	if command -v cargo &> /dev/null; then \
		CARGO_VERSION=$$(cargo --version 2>&1); \
		echo "   ✓ cargo: $$CARGO_VERSION"; \
	else \
		echo "   ⚠ cargo: NOT FOUND (required for 'make link-rtc-local')"; \
	fi; \
	if command -v git &> /dev/null; then \
		GIT_VERSION=$$(git --version 2>&1); \
		echo "   ✓ git: $$GIT_VERSION"; \
	else \
		echo "   ✗ git: NOT FOUND"; \
		ISSUES=$$((ISSUES + 1)); \
	fi; \
	echo ""; \
	echo "$(BOLD)📂 Repository Structure:$(RESET)"; \
	if [ -d "$(PYTHON_RTC)" ]; then \
		echo "   ✓ python-rtc: $$(cd $(PYTHON_RTC) && pwd)"; \
		PYTHON_RTC_BRANCH=$$(cd $(PYTHON_RTC) && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo ""); \
		if [ -n "$$PYTHON_RTC_BRANCH" ]; then \
			PYTHON_RTC_COMMIT=$$(cd $(PYTHON_RTC) && git rev-parse --short HEAD 2>/dev/null || echo "unknown"); \
			echo "      Branch: $$PYTHON_RTC_BRANCH @ $$PYTHON_RTC_COMMIT"; \
		fi; \
	else \
		echo "   ✗ python-rtc: NOT FOUND at $(PYTHON_RTC)"; \
		ISSUES=$$((ISSUES + 1)); \
	fi; \
	if [ -d "$(RUST_SUBMODULE)" ]; then \
		echo "   ✓ rust-submodule: $$(cd $(RUST_SUBMODULE) && pwd)"; \
		if [ -e "$(RUST_SUBMODULE)/.git" ]; then \
			RUST_BRANCH=$$(cd $(RUST_SUBMODULE) && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown"); \
			RUST_COMMIT=$$(cd $(RUST_SUBMODULE) && git rev-parse --short HEAD 2>/dev/null || echo "unknown"); \
			RUST_TAG=$$(cd $(RUST_SUBMODULE) && git describe --exact-match --tags 2>/dev/null || echo ""); \
			if [ -n "$$RUST_TAG" ]; then \
				echo "      Branch: $$RUST_BRANCH @ $$RUST_COMMIT (tag: $$RUST_TAG)"; \
			else \
				echo "      Branch: $$RUST_BRANCH @ $$RUST_COMMIT"; \
			fi; \
		else \
			echo "      ⚠ Not a git repository"; \
		fi; \
	else \
		echo "   ✗ rust-submodule: NOT FOUND at $(RUST_SUBMODULE)"; \
		ISSUES=$$((ISSUES + 1)); \
	fi; \
	if [ -d "$(AGENTS_PROJECT)" ]; then \
		AGENTS_ABS_PATH=$$(cd $(AGENTS_PROJECT) && pwd); \
		echo "   ✓ agents-project: $$AGENTS_ABS_PATH"; \
		AGENTS_REPO_ROOT=$$(cd $(MAKEFILE_DIR) && git rev-parse --show-toplevel 2>/dev/null || echo ""); \
		if [ -n "$$AGENTS_REPO_ROOT" ]; then \
			AGENTS_BRANCH=$$(cd $(MAKEFILE_DIR) && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown"); \
			AGENTS_COMMIT=$$(cd $(MAKEFILE_DIR) && git rev-parse --short HEAD 2>/dev/null || echo "unknown"); \
			echo "      Branch: $$AGENTS_BRANCH @ $$AGENTS_COMMIT"; \
		fi; \
	else \
		echo "   ✗ agents-project: NOT FOUND at $(AGENTS_PROJECT)"; \
		ISSUES=$$((ISSUES + 1)); \
	fi; \
	echo ""; \
	echo "$(BOLD)🔍 Current Configuration:$(RESET)"; \
	if [ -d ".venv" ]; then \
		echo "   ✓ Virtual environment: .venv exists"; \
	else \
		echo "   ⚠ Virtual environment: .venv not found (run 'make install')"; \
	fi; \
	cd $(AGENTS_PROJECT); \
	SHOW_OUTPUT=$$(uv pip show livekit 2>/dev/null || echo ""); \
	if [ -z "$$SHOW_OUTPUT" ]; then \
		echo "   ✗ livekit: NOT INSTALLED"; \
		echo "      Run 'make link-rtc' to set up"; \
		ISSUES=$$((ISSUES + 1)); \
	elif echo "$$SHOW_OUTPUT" | grep -q "Editable project location:"; then \
		VERSION=$$(echo "$$SHOW_OUTPUT" | grep "^Version:" | awk '{print $$2}'); \
		echo "   ✓ livekit: LOCAL (editable) v$$VERSION"; \
	else \
		VERSION=$$(echo "$$SHOW_OUTPUT" | grep "^Version:" | awk '{print $$2}'); \
		echo "   ✓ livekit: PyPI v$$VERSION"; \
	fi; \
	if [ -n "$$LIVEKIT_LIB_PATH" ]; then \
		if [ -f "$$LIVEKIT_LIB_PATH" ]; then \
			echo "   ✓ LIVEKIT_LIB_PATH: $$LIVEKIT_LIB_PATH"; \
		else \
			echo "   ✗ LIVEKIT_LIB_PATH set but file not found: $$LIVEKIT_LIB_PATH"; \
			ISSUES=$$((ISSUES + 1)); \
		fi; \
	else \
		echo "   ⚠ LIVEKIT_LIB_PATH: not set (using bundled FFI)"; \
	fi; \
	echo ""; \
	if [ $$ISSUES -eq 0 ]; then \
		echo "$(BOLD)$(GREEN)✅ All checks passed! Environment is healthy.$(RESET)"; \
	else \
		echo "$(BOLD)$(RED)⚠️  Found $$ISSUES issue(s). Please fix the errors above.$(RESET)"; \
		exit 1; \
	fi