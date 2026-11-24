"""Where paused agent runs are stored.

A prior-auth or discharge run can wait hours for a pharmacist or consultant. With the
in-memory saver that state dies on every restart or deploy; with Postgres it survives,
and any API replica can resume any run (needed once ECS runs more than one task).

AGENT_CHECKPOINTER=memory   (default, local dev and tests)
AGENT_CHECKPOINTER=postgres (uses AGENT_DB_DSN, falling back to PG_DSN)
"""

from __future__ import annotations

import logging
import os

from langgraph.checkpoint.memory import InMemorySaver

log = logging.getLogger("hcai.agents")


def get_checkpointer(kind: str | None = None, dsn: str | None = None):
    kind = (kind or os.environ.get("AGENT_CHECKPOINTER", "memory")).lower()
    if kind == "memory":
        return InMemorySaver()
    if kind == "postgres":
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        dsn = dsn or os.environ.get("AGENT_DB_DSN") or os.environ.get("PG_DSN") or ""
        if not dsn and not os.environ.get("PGHOST"):
            raise ValueError("AGENT_CHECKPOINTER=postgres needs AGENT_DB_DSN, PG_DSN or PGHOST")
        pool = ConnectionPool(dsn, min_size=1, max_size=5, open=True,
                              kwargs={"autocommit": True, "prepare_threshold": 0})
        saver = PostgresSaver(pool)
        saver.setup()  # creates checkpoint tables if missing (idempotent)
        log.info("Agent checkpoints stored in Postgres")
        return saver
    raise ValueError(f"Unknown AGENT_CHECKPOINTER: {kind}")
