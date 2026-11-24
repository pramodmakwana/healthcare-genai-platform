.PHONY: infra-check install data ingest api test lint eval agent-eval judge calibrate dagster up down

install:
	pip install -e ".[faiss,orchestration,dev]"

data:
	python scripts/generate_sample_data.py --patients 40 --seed 7
	python scripts/generate_agent_cases.py

ingest:
	python -m hcai.ingestion.pipeline

api:
	uvicorn hcai.api.main:app --reload --port 8000

test:
	pytest -q

lint:
	ruff check src tests scripts dags orchestration

eval:
	python -m hcai.eval.run_eval

agent-eval:
	python -m hcai.eval.run_agent_eval

judge:  # needs ANTHROPIC_API_KEY
	LLM_PROVIDER=anthropic python -m hcai.eval.run_eval --judge --experiments baseline,improved

dagster:
	dagster dev -f orchestration/dagster_defs.py

calibrate:
	python -m hcai.eval.calibrate

up:
	docker compose up -d --build

down:
	docker compose down

infra-check:  # needs terraform or tofu, and checkov (pip install checkov)
	cd infra/terraform && (command -v tofu >/dev/null && TF=tofu || TF=terraform; \
	  $$TF fmt -check -recursive && $$TF init -backend=false -input=false >/dev/null && $$TF validate && $$TF test)
	checkov -d infra/terraform --framework terraform --quiet --compact --skip-download
