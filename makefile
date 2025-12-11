.PHONY: help install format format-check lint check type-check test test-unit test-docker clean build

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m
BOLD := \033[1m

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "$(BOLD)$(CYAN)Available targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'

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
		echo "$(BOLD)$(RED)✗ Linting issues found$(RESET)"; \
		exit 1; \
	fi

lint-fix: ## Run ruff linter and fix issues automatically
	@echo "$(BOLD)$(CYAN)Running linter with auto-fix...$(RESET)"
	@uv run ruff check --fix .
	@echo "$(BOLD)$(GREEN)✓ Linting complete$(RESET)"

type-check: ## Run mypy type checker
	@echo "$(BOLD)$(CYAN)Running type checker...$(RESET)"
	@uv pip install pip 2>/dev/null || true
	@if uv run mypy --install-types --non-interactive \
		-p livekit.agents \
		-p livekit.plugins.openai \
		-p livekit.plugins.anthropic \
		-p livekit.plugins.mistralai \
		-p livekit.plugins.assemblyai \
		-p livekit.plugins.aws \
		-p livekit.plugins.azure \
		-p livekit.plugins.bey \
		-p livekit.plugins.bithuman \
		-p livekit.plugins.cartesia \
		-p livekit.plugins.clova \
		-p livekit.plugins.deepgram \
		-p livekit.plugins.elevenlabs \
		-p livekit.plugins.fal \
		-p livekit.plugins.gladia \
		-p livekit.plugins.google \
		-p livekit.plugins.groq \
		-p livekit.plugins.hume \
		-p livekit.plugins.minimal \
		-p livekit.plugins.neuphonic \
		-p livekit.plugins.nltk \
		-p livekit.plugins.resemble \
		-p livekit.plugins.rime \
		-p livekit.plugins.silero \
		-p livekit.plugins.speechify \
		-p livekit.plugins.speechmatics \
		-p livekit.plugins.tavus \
		-p livekit.plugins.turn_detector \
		-p livekit.plugins.hedra \
		-p livekit.plugins.langchain \
		-p livekit.plugins.baseten \
		-p livekit.plugins.sarvam \
		-p livekit.plugins.inworld \
		-p livekit.plugins.simli \
		-p livekit.plugins.anam \
		-p livekit.plugins.ultravox \
		-p livekit.plugins.fireworksai \
		-p livekit.plugins.minimax; then \
		echo "$(BOLD)$(GREEN)✓ Type checking passed$(RESET)"; \
	else \
		echo "$(BOLD)$(RED)✗ Type checking failed$(RESET)"; \
		exit 1; \
	fi

check: format-check lint type-check ## Run all checks (format, lint, type-check)
	@echo "$(BOLD)$(GREEN)✓ All checks passed!$(RESET)"