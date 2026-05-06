# Makefile — common developer commands for SEC Filing Intelligence
# Usage: make <target>

.PHONY: api ui ingest ingest-local stats test test-unit test-cov \
        eval gen-testset lint format docker-up docker-down docker-logs help

# ── Development ───────────────────────────────────────────────────────────────
api:          ## Start the FastAPI backend on port 8000 (with hot reload)
	uvicorn src.api.app:app --reload --port 8000

ui:           ## Start the Streamlit UI on port 8501
	streamlit run main.py

# ── Data pipeline ─────────────────────────────────────────────────────────────
ingest:       ## Ingest all 5 default tickers using Jina embedder
	python scripts/ingest.py full --tickers AAPL MSFT GOOGL AMZN NVDA

ingest-local: ## Ingest using the free local (no API key) embedder
	python scripts/ingest.py full --tickers AAPL MSFT GOOGL AMZN NVDA --use-local

stats:        ## Show current vector store statistics
	python scripts/ingest.py stats

# ── Testing ───────────────────────────────────────────────────────────────────
test:         ## Run unit + integration tests (e2e excluded)
	pytest tests/unit/ tests/integration/ -v

test-unit:    ## Run unit tests only
	pytest tests/unit/ -v

test-e2e:     ## Run end-to-end tests (requires docker compose up)
	pytest tests/e2e/ -m e2e -v

test-cov:     ## Run tests with coverage report (fail under 70%)
	pytest tests/unit/ tests/integration/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=70

worker:       ## Start the RQ ingestion worker (requires REDIS_URL in .env)
	python scripts/worker.py

# ── Evaluation ────────────────────────────────────────────────────────────────
eval:         ## Run RAGAS evaluation (Groq provider, 50 samples)
	python evals/run_ragas.py --provider groq --sample-size 50

gen-testset:  ## Generate synthetic Q&A test set for evaluation
	python evals/generate_testset.py

# ── Code quality ──────────────────────────────────────────────────────────────
lint:         ## Run ruff linter across all source directories
	ruff check src/ tests/ evals/ scripts/

format:       ## Auto-format code with ruff
	ruff format src/ tests/ evals/ scripts/

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:    ## Build and start API (8000) + UI (8501) containers
	docker compose up --build

docker-down:  ## Stop and remove all containers
	docker compose down

docker-logs:  ## Follow live logs from the API container
	docker compose logs -f api

# ── Help ──────────────────────────────────────────────────────────────────────
help:         ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
