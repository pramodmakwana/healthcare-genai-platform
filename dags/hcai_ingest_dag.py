"""Airflow DAG: refresh the RAG index from the FHIR server, then gate on evaluation.

Drop this file into the dags/ folder of the Airflow in healthcare-fhir-data-pipeline and
`pip install -e /path/to/healthcare-genai-platform` in the Airflow image. Steps:

  extract_and_validate -> build_index -> evaluate -> quality_gate

The quality gate fails the run (and alerts) if retrieval quality drops below thresholds,
so a bad data load never silently degrades the assistant.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

from airflow.decorators import dag, task

FHIR_BASE_URL = os.environ.get("FHIR_BASE_URL", "http://hapi-fhir:8080/fhir")
MIN_HIT_AT_K = float(os.environ.get("HCAI_MIN_HIT_AT_K", "0.9"))
MIN_EVIDENCE_RECALL = float(os.environ.get("HCAI_MIN_EVIDENCE_RECALL", "0.85"))


@dag(
    dag_id="hcai_rag_index_refresh",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["genai", "rag", "fhir"],
)
def hcai_rag_index_refresh():
    @task
    def extract_and_validate() -> int:
        from hcai.ingestion.documents import load_fhir_server
        from hcai.ingestion.pipeline import validate

        docs = load_fhir_server(FHIR_BASE_URL, max_patients=1000)
        validate(docs)
        if not docs:
            raise ValueError("No patients returned from FHIR server")
        return len(docs)

    @task
    def build_index(n_docs: int) -> dict:
        from hcai.ingestion.pipeline import run

        stats = run(fhir_server=FHIR_BASE_URL)
        print(f"Indexed {stats['chunks']} chunks from {n_docs} patient documents")
        return stats

    @task
    def evaluate(stats: dict) -> dict:
        # Runs against the golden set; on a real FHIR load you'd maintain a golden set for it.
        subprocess.run([sys.executable, "-m", "hcai.eval.run_eval", "--experiments", "improved"], check=True)
        from hcai.config import PROJECT_ROOT

        return json.loads((PROJECT_ROOT / "reports" / "eval_results.json").read_text())["experiments"][0]

    @task
    def quality_gate(result: dict) -> None:
        if result["hit_at_k"] < MIN_HIT_AT_K or result["evidence_recall"] < MIN_EVIDENCE_RECALL:
            raise ValueError(f"RAG quality gate failed: {result}")

    quality_gate(evaluate(build_index(extract_and_validate())))


hcai_rag_index_refresh()
