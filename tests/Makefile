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
	cd .. && PYTHONPATH="$$PWD" uv run pytest \
		tests/test_agent_session.py \
		tests/test_aio.py \
		tests/test_audio_decoder.py \
		tests/test_chat_ctx.py \
		tests/test_config.py \
		tests/test_connection_pool.py \
		tests/test_debounce.py \
		tests/test_google_thought_signatures.py \
		tests/test_inference_stt_fallback.py \
		tests/test_inference_tts_fallback.py \
		tests/test_ipc.py \
		tests/test_ivr_activity.py \
		tests/test_langgraph.py \
		tests/test_plugin_google_stt.py \
		tests/test_schema_gemini.py \
		tests/test_tts_fallback.py \
		tests/test_stt_fallback.py \
		tests/test_recording.py \
		tests/test_tokenizer.py \
		tests/test_transcription_filter.py \
		tests/test_tools.py