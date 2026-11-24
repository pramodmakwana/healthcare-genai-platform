# Agent evaluation report

- LLM provider: `mock`. Human review steps are simulated.
- Generated: 2026-11-22 06:33

## Prior-authorization triage agent

| Metric | Value |
|---|---|
| Requests | 38 |
| Triage status accuracy | 100% |
| Missing-document list exact match | 100% |
| Checks completed automatically per request | 2.89 |
| Latency p50 (to human-review pause) | 8.2 ms |

## Discharge-summary agent (drafter + reviewer)

| Metric | Value |
|---|---|
| Admissions | 12 |
| Planted documentation gaps caught (recall) | 100% |
| Reviewer precision on planted gap types | 100% |
| Gaps auto-fixed from the FHIR record | 100% |
| Send-to-GP deadline correct | 100% |
| Complete after consultant sign-off | 100% |
