# Evaluation report

- LLM provider: `mock`  |  Embeddings: `hashing`  |  Questions: 95
- Generated: 2026-11-22 06:32
- **Mock LLM run.** Token counts are estimates and cost is projected at Claude prices (config/pricing.yaml). Answer-quality metrics reflect retrieval + the extractive mock, not a real model. Re-run with `LLM_PROVIDER=anthropic` for measured numbers.

## Configurations

- **baseline**: `{"chunking": "fixed", "retrieval_mode": "dense", "top_k": 8, "llm_routing": false, "abstain_min_score": 0.0, "patient_scoping": false}`
- **section_chunking**: `{"chunking": "section", "retrieval_mode": "dense", "top_k": 8, "llm_routing": false, "abstain_min_score": 0.0, "patient_scoping": false}`
- **hybrid_retrieval**: `{"chunking": "section", "retrieval_mode": "hybrid", "top_k": 5, "llm_routing": false, "abstain_min_score": 0.0, "patient_scoping": false}`
- **improved**: `{"chunking": "section", "retrieval_mode": "hybrid", "top_k": 5, "llm_routing": true, "patient_scoping": true}`

## Results

| Metric | baseline | section_chunking | hybrid_retrieval | improved | Change (baseline → improved) |
|---|---|---|---|---|---|
| Retrieval hit@k | 0.788 | 0.941 | 1.000 | 1.000 | +27% (better) |
| Retrieval MRR | 0.782 | 0.827 | 0.994 | 1.000 | +28% (better) |
| Recall@k | 0.788 | 0.935 | 0.994 | 0.994 | +26% (better) |
| Evidence recall (answer text retrieved) | 0.788 | 0.812 | 0.847 | 1.000 | +27% (better) |
| Answer accuracy | 0.694 | 0.729 | 0.729 | 0.941 | +36% (better) |
| Citation rate | 1.000 | 0.977 | 1.000 | 1.000 | +0% (same) |
| Citation precision | 0.582 | 0.731 | 0.884 | 0.757 | +30% (better) |
| Mean groundedness | 1.000 | 1.000 | 1.000 | 1.000 | +0% (same) |
| Ungrounded-answer rate | 0.000 | 0.000 | 0.000 | 0.000 | n/a |
| Correct refusals (unanswerable) | 0.600 | 0.600 | 0.400 | 1.000 | +67% (better) |
| Avg input tokens / query | 1273.6 | 677.7 | 465.5 | 409.0 | -68% (better) |
| Latency p50 (ms) | 1.000 | 0.700 | 0.800 | 2.200 | +120% (worse) |
| Latency p95 (ms) | 1.200 | 0.900 | 1.000 | 2.800 | +133% (worse) |
| Cost per 1k queries (USD) | 3.138 | 1.897 | 1.435 | 0.200 | -94% (better) |

## Guardrails

25 prompts (injection, emergency, personal-advice, plus benign controls):

- Category accuracy: **100%**
- Harmful prompts caught: **100%**
- Benign prompts wrongly blocked: **0%**
