"""Suggest ABSTAIN_MIN_SCORE for the current embedder.

Similarity scales differ a lot between embedders (hashing vs bge-small vs Voyage), so the
abstain threshold must be re-calibrated whenever EMBEDDING_PROVIDER changes. This prints the
top dense score for answerable vs unanswerable golden questions and suggests a midpoint.

Caveat: calibrating and evaluating on the same questions flatters the result. For a claim you
put in front of others, calibrate on a separate question set.
"""

from __future__ import annotations

import json
import statistics
import tempfile

from hcai.config import PROJECT_ROOT, get_settings
from hcai.embeddings.embedders import get_embedder
from hcai.ingestion import pipeline
from hcai.retrieval.retriever import Retriever
from hcai.vectorstores.stores import get_vector_store


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        settings = get_settings(vector_store="memory", index_path=tmp)
        pipeline.run(settings)
        emb = get_embedder(settings)
        retriever = Retriever(emb, get_vector_store(settings, emb.dim), "dense")
        rows = [json.loads(x) for x in (PROJECT_ROOT / "data/eval/golden.jsonl").read_text().splitlines() if x.strip()]
        ans, unans = [], []
        for r in rows:
            retriever.retrieve(r["question"], 1)
            (ans if r["expected_doc_ids"] else unans).append(retriever.last_confidence)
    lo_ans, hi_unans = min(ans), max(unans)
    print(f"answerable   : min={lo_ans:.3f} median={statistics.median(ans):.3f}")
    print(f"unanswerable : max={hi_unans:.3f} median={statistics.median(unans):.3f}")
    if lo_ans > hi_unans:
        print(f"Separable. Suggested ABSTAIN_MIN_SCORE={round((lo_ans + hi_unans) / 2, 3)}")
    else:
        print("Overlapping distributions: a threshold will cost some answerable questions. "
              f"Conservative choice: ABSTAIN_MIN_SCORE={round(hi_unans * 0.9, 3)} or 0 to disable.")


if __name__ == "__main__":
    main()
