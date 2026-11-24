# Healthcare GenAI Platform

Retrieval-Augmented Generation (RAG) over **FHIR patient records** and **clinical guidelines / operational policies**, plus two **human-in-the-loop LangGraph agents** (prior-authorization triage and discharge-summary drafting). It includes clinical guardrails, provider-agnostic LLMs, pluggable vector stores, Airflow and Dagster pipelines, an evaluation harness that measures every change, and **AWS infrastructure as code** (ECS Fargate, RDS pgvector, Bedrock-ready) with OIDC-based continuous deployment and CloudWatch monitoring.

Built as the AI layer on top of [`healthcare-fhir-data-pipeline`](https://github.com/pramodmakwana/healthcare-fhir-data-pipeline) (Synthea → HAPI FHIR → Postgres → Airflow).

> All patient data in this repo is synthetic. The guideline documents are illustrative summaries written for this project and are **not clinical guidance**.

---

## What it does

```
            ┌──────────────── Ingestion (Airflow DAG / CLI) ────────────────┐
FHIR R4  ──►│ load bundles / HAPI $everything → render per-patient summary  │
Guidelines─►│ section-aware chunking → embeddings → vector store + BM25     │
            └───────────────────────────────────────────────────────────────┘
                                         │
 POST /ask ─► input guardrails ─► patient-scoped hybrid retrieval ─► abstain gate
              (injection,          (BM25 + dense, RRF fusion;          (low relevance →
               emergency,           one patient's record only)          no LLM call)
               personal advice)                │
                                               ▼
                         model routing (Haiku for lookups, Sonnet otherwise)
                                               │
                                               ▼
                    output checks: citation validity + groundedness score
                                               │
                                               ▼
                answer + citations + flags + tokens + cost + latency (PHI-redacted logs)
```

| Area | Implementation |
|---|---|
| LLM providers | Claude (Anthropic API, prompt caching), OpenAI, Azure OpenAI, AWS Bedrock (Converse API), Google Gemini, offline mock. One interface: `hcai.llm` |
| Embeddings | hashing (offline/CI), fastembed `bge-small` (local), Voyage AI, OpenAI |
| Vector stores | in-memory (numpy), FAISS, ChromaDB, **pgvector** (HNSW), Pinecone |
| Retrieval | dense, BM25, hybrid (Reciprocal Rank Fusion), patient-scoped retrieval, relevance-gated abstention |
| Guardrails | prompt-injection / bulk-export blocking, emergency detection, personal dosing/diagnosis refusal, citation validation, groundedness scoring, PHI redaction in logs |
| Healthcare data | FHIR R4 (Patient, Condition, Observation, MedicationRequest, Encounter) with SNOMED CT, ICD-10-CM, LOINC, RxNorm codes kept in the text |
| Agents (LangGraph) | **Prior-auth triage**: patient lookup → drug classification (brand-name normalisation) → documentation checklist → SLA routing → policy citation via RAG → drafted prescriber note → **pharmacist review** (interrupt/resume). **Discharge summary**: drafter → independent reviewer (SOP checklist, medication reconciliation) → auto-revise loop → **consultant sign-off** (interrupt/resume). Full audit trail per run |
| API | FastAPI: `/ask`, `/search`, `/ingest` (admin token), `/health`, `/metrics`, `/agents/prior-auth`, `/agents/discharge-summary` (+ review/sign-off endpoints) |
| Evaluation | RAG: golden set (95 questions) + safety set (25 prompts). Agents: 38 prior-auth requests, 12 admissions with planted documentation gaps. **Claude grader** (claim-level faithfulness) as an option. Markdown + JSON reports |
| Orchestration | Airflow DAG (fits the existing FHIR pipeline) and a Dagster asset graph (`source_documents → rag_index → rag_quality / agent_quality`) with an asset-check quality gate |
| Ops | Docker (multi-stage, non-root, read-only root filesystem on ECS), docker-compose (pgvector + API, memory-capped for 8GB/WSL2), GitHub Actions (lint, tests incl. Postgres service, RAG + agent quality gates, image smoke test) |
| Cloud (AWS, Terraform) | VPC, ECS Fargate + ALB + autoscaling, RDS PostgreSQL 16 + pgvector (private, encrypted, TLS-only, RDS-managed password), Secrets Manager, KMS, IAM least privilege (Bedrock access only in Bedrock mode), GitHub OIDC deploy role. Validated with `tofu validate`, **7 plan-time security tests** on a mocked AWS provider, and **Checkov (170 passed, 0 failed, 13 documented skips)** |
| Deployment | `deploy.yml`: OIDC credentials → image tagged by git SHA → task definition → ingestion task → rolling update with automatic rollback → smoke test. `infra.yml`: fmt / validate / test / Checkov on every infra change |
| Observability | JSON logs with a request ID on every line (`X-Request-ID`), CloudWatch Embedded Metric Format (latency, tokens, cost, guardrail blocks, abstentions, groundedness failures, agent runs), dashboard, 7 alarms, Logs Insights trace query |
| Agent state | LangGraph checkpoints in Postgres: paused pharmacist and consultant reviews survive restarts and deploys, and any replica can resume them (tested by restarting the API mid-review) |

---

## Results

`python -m hcai.eval.run_eval` compares four configurations on the same questions. Each row adds one change, so you can see what each one contributes.

| Configuration | What changes |
|---|---|
| `baseline` | fixed 120-word chunks, vector search only, top-8, one model, no abstention |
| `section_chunking` | + split on document sections, title/section header in every chunk |
| `hybrid_retrieval` | + BM25 and dense fused with RRF, top-5 |
| `improved` | + patient-scoped retrieval, abstain on low relevance, route simple lookups to Haiku |

Latest run (mock LLM, hashing embeddings, synthetic data; full table in [`reports/eval_report.md`](reports/eval_report.md)):

| Metric | baseline | improved | Change |
|---|---|---|---|
| Retrieval hit@k | 0.788 | 1.000 | +27% |
| Evidence recall (answer text actually retrieved) | 0.788 | 1.000 | +27% |
| Answer accuracy | 0.694 | 0.941 | +36% |
| Correct refusals on out-of-scope questions | 0.600 | 1.000 | +67% |
| Avg input tokens / query | 1,274 | 409 | −68% |
| Projected cost per 1k queries (Claude pricing) | $3.14 | $0.20 | −94% |
| Guardrails: harmful prompts caught / benign wrongly blocked | — | 100% / 0% | |

**How to read these numbers honestly**

- These come from a **mock LLM** (deterministic and extractive) so the pipeline and CI run without API keys. Retrieval, token and abstention metrics are real. Answer accuracy and groundedness reflect the mock, not Claude. Run with `LLM_PROVIDER=anthropic` to get measured numbers (see below).
- Most of the cost reduction comes from **routing** to Haiku, which only holds up if Haiku's answers stay accurate. Confirm that with a real-model run before quoting it.
- Latency in mock mode is sub-millisecond noise. Measure real latency with a real provider.
- Synthetic data is cleaner than real records, so retrieval at 1.000 will not carry over as-is.
- The abstain threshold was calibrated on the same question set it's evaluated on. For a claim you'll defend in an interview, calibrate on a separate set (`python -m hcai.eval.calibrate`).
- Citation precision falls from 0.884 (`hybrid_retrieval`) to 0.757 (`improved`) because the mock also cites the guideline chunk that patient-scoped retrieval adds. Check whether this holds with Claude.

### Agents

`python -m hcai.eval.run_agent_eval` ([`reports/agent_eval_report.md`](reports/agent_eval_report.md)):

| Agent | Metric | Result |
|---|---|---|
| Prior-auth triage | Triage status correct (38 requests) | 100% |
| | Missing-document list exactly right | 100% |
| | Checks completed automatically per request (lab dates, BMI, prior medicines, eGFR, justification) | 2.9 |
| Discharge summary | Planted documentation gaps caught (recall / precision) | 100% / 100% |
| | Gaps fixed automatically from the FHIR record (ICD-10 codes) | 100% |
| | Send-to-GP deadline correct (48h rule, Friday rule) | 100% |

**What these numbers do and don't show.** The test cases and the agents' rules both come from the same policy documents, so 100% mainly shows the agents follow the rules they were built from. The useful signal came from the **harder cases**: the first version scored 92% because it didn't recognise brand names ("Ozempic", "Mounjaro") or free text ("semaglutide 1mg pen"). It treated them as unknown drugs and passed requests as complete with documents missing. The fix was brand/free-text normalisation, plus a fail-safe so an unrecognised drug always pends for a pharmacist. The real test is clinicians' own cases and real request volumes. Time saved per request is not measured here; that needs a pilot with timing data.

**Design choices worth knowing:**
- **The model never decides.** Rules decide what is required, what is missing and the deadlines. The LLM drafts text. A pharmacist or consultant makes every final call through LangGraph `interrupt()`, and nothing is final until they act.
- **The reviewer only fixes what it can prove.** A missing ICD-10 code that exists in the patient's FHIR conditions is filled in automatically. A missing reason for a medication change, the follow-up plan or the allergy status becomes a question for the consultant. The agent never invents these, and a summary with open questions can't be signed off.
- **Structured data comes from FHIR, not from text.** Lab values, dates and medicines are read from the FHIR resources, and policy wording comes from cited RAG answers.

### Measured run with Claude

```bash
export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-...
python -m hcai.eval.run_eval            # ~380 Claude calls across 4 configs (mostly Haiku)
python -m hcai.eval.run_eval --limit 20 # cheap smoke run first
python -m hcai.eval.run_eval --judge --experiments baseline,improved   # + Claude grader (JUDGE_MODEL, default Sonnet)
python -m hcai.eval.run_agent_eval      # agents with Claude drafting the notes and hospital course
```

The grader breaks each answer into individual claims and checks each one against the retrieved sources (it never sees the expected answer). The report then gains *Judge faithfulness* and *Judge hallucination rate* rows. Those, not the mock's word-overlap score, are the numbers to quote for hallucination.

---

## Quickstart (Windows + WSL2, 8GB RAM)

Run these in the **WSL** terminal (`user@host:~$`), not PowerShell.

```bash
git clone https://github.com/pramodmakwana/healthcare-genai-platform.git
cd healthcare-genai-platform
python3 -m venv .venv && source .venv/bin/activate
make install                 # pip install -e ".[faiss,dev]"
cp .env.example .env         # defaults run fully offline

make test                    # 35 tests (Postgres ones need TEST_PG_DSN)
make eval                    # writes reports/eval_report.md
make api                     # http://localhost:8000/docs
```

Ask a question:

```bash
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "At what eGFR should metformin be stopped?"}' | python3 -m json.tool
```

Run the agents (each pauses for a human, then resumes):

```bash
# 1. Prior-auth triage -> returns thread_id, checklist, deadlines, drafted note
curl -s -X POST localhost:8000/agents/prior-auth -H 'Content-Type: application/json' -d '{
  "mrn": "MRN100017", "drug": "Ozempic", "indication": "type 2 diabetes",
  "prescriber_justification": "HbA1c above target despite metformin; BMI over 30."}'
# 2. Pharmacist decision
curl -s -X POST localhost:8000/agents/prior-auth/<thread_id>/review -H 'Content-Type: application/json' \
  -d '{"action": "approve_note", "reviewer": "pharmacist.demo"}'

# Discharge summary: draft + open queries, then consultant sign-off with answers
curl -s -X POST localhost:8000/agents/discharge-summary -H 'Content-Type: application/json' -d '{"admission_id": "ADM2601"}'
```

Dagster UI: `make dagster` → http://localhost:3000 (assets, lineage, check results).

### Deploy to AWS

See **[docs/RUNBOOK.md](docs/RUNBOOK.md)**: setup, first apply, connecting GitHub Actions, monitoring, incident checklist and teardown. Roughly **USD 65–85/month** while running; `terraform destroy` removes it all.

```bash
make infra-check   # fmt + validate + mocked security tests + Checkov (no AWS account needed)
```

### With pgvector (Docker)

```bash
docker compose up -d --build     # Postgres+pgvector on :5433, API on :8000
```

Port 5433 avoids a clash with the Postgres in `healthcare-fhir-data-pipeline`, so both stacks can run side by side. Containers are memory-capped (512MB + 1GB).

### Using your FHIR pipeline's data

```bash
# Synthea bundles: copy them into data/raw/fhir/ and re-ingest
python -m hcai.ingestion.pipeline
# or pull straight from the HAPI FHIR server
python -m hcai.ingestion.pipeline --fhir-server http://localhost:8080/fhir
```

`dags/hcai_ingest_dag.py` does the same thing on a schedule inside your existing Airflow, then runs the evaluation and **fails the DAG if retrieval quality drops** below thresholds.

### Switching components

All configuration is in `.env`:

| Want | Set |
|---|---|
| Claude | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY` |
| Azure OpenAI / Bedrock / Gemini | `LLM_PROVIDER=azure_openai` / `bedrock` / `gemini` + that provider's keys (`pip install -e ".[openai]"` etc.) |
| Real local embeddings | `EMBEDDING_PROVIDER=fastembed`, `pip install fastembed`, then **re-run `make calibrate`** and update `ABSTAIN_MIN_SCORE` |
| pgvector / Chroma / FAISS / Pinecone | `VECTOR_STORE=pgvector` / `chroma` / `faiss` / `pinecone` |

---

## Project layout

```
src/hcai/
  llm/            provider interface, 5 providers + mock, cost estimation
  embeddings/     hashing, fastembed, Voyage, OpenAI
  ingestion/      FHIR + markdown loaders, chunking, pipeline (validate → chunk → embed → index)
  vectorstores/   memory, FAISS, Chroma, pgvector, Pinecone
  retrieval/      dense / BM25 / hybrid RRF, patient scoping, confidence for abstention
  guardrails/     input checks, citation + groundedness checks, PHI redaction
  rag/            prompt, routing, the RAG chain, factory
  agents/         prior_auth.py, discharge.py (LangGraph), tools.py (FHIR patient repository, SLA dates), checkpoint.py
  observability.py  JSON logging, request IDs, CloudWatch EMF metrics
  api/            FastAPI app
  eval/           RAG + agent evaluation, Claude grader, threshold calibration
dags/             Airflow DAG with quality gate
orchestration/    Dagster asset definitions with asset check
infra/terraform/  AWS infrastructure (+ tests/security.tftest.hcl)
docs/RUNBOOK.md   deploy, operate, monitor, tear down
scripts/          synthetic FHIR, eval-set and agent-case generators
data/raw/         synthetic FHIR bundles, guideline/policy documents, admissions
data/eval/        golden (95), safety (25), prior_auth_cases (38), discharge_cases (12)
config/           pricing.yaml (Claude prices, checked 2026-11-21)
reports/          evaluation outputs
```

---

## Design decisions

- **Patient-scoped retrieval.** When a question names one patient (MRN or full name), only that patient's chunks are ranked, plus non-patient guidance. In testing, this fixed questions like "eGFR for MRN100002" where short, header-heavy chunks from *other* patients outranked the right observation chunk. It also enforces "minimum necessary" access: other patients' data never enters the prompt.
- **Abstain before generating.** If the best retrieval similarity is below a calibrated threshold, the API answers "not enough information" without calling the LLM. That's safer than letting the model improvise, and cheaper.
- **Hybrid retrieval.** Clinical questions hinge on exact tokens (MRNs, LOINC codes, drug names), which BM25 handles well, and on paraphrase ("kidney function" vs "eGFR"), which embeddings handle. RRF merges the two without calibrating their score scales.
- **Guardrails are deterministic and tested.** They're regex/rule-based, so they're fast, auditable and covered by unit tests, and the safety set includes benign controls to measure false positives. An LLM-based classifier can be layered on top in Phase 2.
- **Everything runs offline.** The mock provider and hashing embeddings mean CI needs no secrets. Real providers are switched on through configuration.


## License

MIT
