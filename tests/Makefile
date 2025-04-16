.PHONY: test up down

up:
	docker compose build
	docker compose up -d

down:
	docker compose down

test: up
	until curl -sf http://127.0.0.1:8474/proxies; do \
	  echo "Waiting for toxiproxy..."; \
	  sleep 1; \
	done

	docker compose exec app uv sync --all-extras --dev
	docker compose exec app uv run pytest -s --tb=short tests/test_tts.py --show-capture=all
	$(MAKE) down
