.PHONY: test up down

up:
	@if [ -f ../.env ]; then \
	  echo "Found .env file. Using it..."; \
	  docker compose --env-file ../.env build; \
	  docker compose --env-file ../.env up -d; \
	else \
	  echo "No .env file found. Running without it..."; \
	  docker compose build; \
	  docker compose up -d; \
	fi

down:
	docker compose down

test: up
	@docker compose exec app bash -c "\
	  until curl -sf http://toxiproxy:8474/proxies; do \
	    echo 'Waiting for toxiproxy...'; \
	    sleep 1; \
	  done"
	echo 'Toxiproxy is ready'

	docker compose exec app uv sync --all-extras --dev
	docker compose exec app uv run pytest -s --color=yes --tb=short tests/test_tts.py --show-capture=all
	$(MAKE) down
