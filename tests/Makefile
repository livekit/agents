PLUGIN ?=
PYTEST_ARGS ?=

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
	docker compose exec -e PLUGIN="$(PLUGIN)" app uv run pytest -s --color=yes --tb=short --log-cli-level=DEBUG tests/test_tts.py --show-capture=all $(PYTEST_ARGS)
	$(MAKE) down

unit-tests:
	uv run pytest tests/ --ignore=tests/test_realtime --ignore=tests/test_stt.py --ignore=tests/test_tts.py -x -v $(PYTEST_ARGS)

realtime-tests:
	uv run pytest tests/test_realtime/ -v $(PYTEST_ARGS)
