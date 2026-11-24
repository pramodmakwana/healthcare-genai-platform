"""Dagster version of the ingestion + evaluation pipeline (asset-based).

    source_documents ──► rag_index ──► rag_quality ──(asset check: quality gate)
                                  └──► agent_quality

Airflow (dags/) and Dagster do the same job here on purpose: Airflow fits the existing
healthcare-fhir-data-pipeline stack; Dagster shows the asset/lineage model, where every
artefact (documents, index, eval results) is a versioned asset with metadata and checks.

Run locally:   dagster dev -f orchestration/dagster_defs.py   (UI on http://localhost:3000)
"""


import tempfile

from dagster import (
    AssetCheckResult,
    AssetExecutionContext,
    Definitions,
    MetadataValue,
    ScheduleDefinition,
    asset,
    asset_check,
    define_asset_job,
)

from hcai.config import PROJECT_ROOT, get_settings

MIN_HIT_AT_K = 0.9
MIN_EVIDENCE_RECALL = 0.85


@asset(group_name="rag", description="FHIR patient records + guideline/policy documents, validated")
def source_documents(context: AssetExecutionContext) -> dict:
    from hcai.ingestion.pipeline import load_all, validate

    docs = load_all(get_settings())
    validate(docs)
    counts = {"documents": len(docs),
              "patient_records": sum(d.source_type == "patient_record" for d in docs),
              "guidelines": sum(d.source_type == "guideline" for d in docs)}
    context.add_output_metadata({k: MetadataValue.int(v) for k, v in counts.items()})
    return counts


@asset(group_name="rag", description="Chunked, embedded, indexed corpus in the configured vector store")
def rag_index(context: AssetExecutionContext, source_documents: dict) -> dict:
    from hcai.ingestion.pipeline import run

    stats = run(get_settings())
    context.add_output_metadata({"chunks": MetadataValue.int(stats["chunks"]),
                                 "vector_store": stats["vector_store"],
                                 "embedding_provider": stats["embedding_provider"]})
    return stats


@asset(group_name="quality", description="RAG evaluation on the golden set (improved configuration)")
def rag_quality(context: AssetExecutionContext, rag_index: dict) -> dict:
    import json
    from dataclasses import asdict

    from hcai.eval.run_eval import EXPERIMENTS, run_experiment

    rows = [json.loads(x) for x in (PROJECT_ROOT / "data/eval/golden.jsonl").read_text().splitlines() if x.strip()]
    result, _ = run_experiment("improved", EXPERIMENTS["improved"], rows, get_settings())
    out = asdict(result)
    context.add_output_metadata({k: MetadataValue.float(float(out[k])) for k in
                                 ("hit_at_k", "evidence_recall", "answer_accuracy", "correct_refusal_rate",
                                  "cost_per_1k_queries_usd")})
    return out


@asset(group_name="quality", description="Prior-auth and discharge-summary agent evaluation")
def agent_quality(context: AssetExecutionContext, rag_index: dict) -> dict:
    from hcai.eval.run_agent_eval import eval_discharge, eval_prior_auth
    from hcai.rag.factory import build_rag_chain

    with tempfile.TemporaryDirectory() as tmp:
        rag = build_rag_chain(get_settings(vector_store="memory", index_path=tmp))
        pa, ds = eval_prior_auth(rag), eval_discharge(rag.llm)
    context.add_output_metadata({"prior_auth_status_accuracy": MetadataValue.float(pa["status_accuracy"]),
                                 "discharge_issue_recall": MetadataValue.float(ds["issue_recall"])})
    return {"prior_auth": pa, "discharge": ds}


@asset_check(asset=rag_quality, description="Block promotion if retrieval quality regresses")
def rag_quality_gate(rag_quality: dict) -> AssetCheckResult:
    ok = rag_quality["hit_at_k"] >= MIN_HIT_AT_K and rag_quality["evidence_recall"] >= MIN_EVIDENCE_RECALL
    return AssetCheckResult(passed=ok, metadata={"hit_at_k": rag_quality["hit_at_k"],
                                                 "evidence_recall": rag_quality["evidence_recall"]})


refresh_job = define_asset_job("rag_refresh", selection="*")

defs = Definitions(
    assets=[source_documents, rag_index, rag_quality, agent_quality],
    asset_checks=[rag_quality_gate],
    jobs=[refresh_job],
    schedules=[ScheduleDefinition(job=refresh_job, cron_schedule="0 2 * * *")],
)
