"""Structured logging, request tracing and CloudWatch metrics.

- Every request gets an ID (taken from X-Request-ID if the caller sends one) that appears
  on every log line for that request and in the response header, so one question can be
  traced through guardrails, retrieval, the LLM call and the agents.
- LOG_FORMAT=json writes one JSON object per line (what CloudWatch Logs Insights queries).
- In JSON mode, per-request metrics are also written in CloudWatch Embedded Metric Format
  (EMF): CloudWatch turns those log lines into metrics automatically - no agent, no SDK,
  no extra API calls. The Terraform dashboard and alarms are built on them.
- Nothing here logs question text; the RAG chain logs only PHI-redacted questions.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

SERVICE = os.environ.get("SERVICE_NAME", "hcai-api")
NAMESPACE = os.environ.get("METRICS_NAMESPACE", "HCAI")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "service": SERVICE,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_RequestIdFilter())
    if json_mode():
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"))
    root.addHandler(handler)
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def json_mode() -> bool:
    return os.environ.get("LOG_FORMAT", "text").lower() == "json"


UNITS = {"LatencyMs": "Milliseconds", "RetrievalMs": "Milliseconds", "InputTokens": "Count",
         "OutputTokens": "Count", "CostUSD": "None", "GuardrailBlocked": "Count", "Abstained": "Count",
         "LowGroundedness": "Count", "Requests": "Count", "Errors": "Count", "AgentRuns": "Count"}


def emf_record(endpoint: str, metrics: dict[str, float], **properties) -> dict:
    """Build a CloudWatch Embedded Metric Format record (one metric set, two dimensions)."""
    return {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": NAMESPACE,
                "Dimensions": [["Service", "Endpoint"]],
                "Metrics": [{"Name": k, "Unit": UNITS.get(k, "None")} for k in metrics],
            }],
        },
        "Service": SERVICE,
        "Endpoint": endpoint,
        "request_id": request_id_var.get(),
        **{k: float(v) for k, v in metrics.items()},
        **properties,
    }


def emit_metrics(endpoint: str, metrics: dict[str, float], **properties) -> None:
    if json_mode():
        sys.stdout.write(json.dumps(emf_record(endpoint, metrics, **properties)) + "\n")
        sys.stdout.flush()
