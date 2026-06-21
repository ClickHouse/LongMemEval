# Loom on LongMemEval-S — Accuracy

[Loom](https://github.com/ClickHouse/loom) is a ClickHouse-backed memory service.
This benchmark plugs Loom into LongMemEval-S at the **indexing + retrieval** stages;
the reader (answerer) and judge (grader) are LLMs — the standard measurement
apparatus, not part of Loom.

## Setup

- **Dataset:** LongMemEval-S, 500 questions (491 answered; 9 dropped to reader API errors).
- **Indexing + retrieval:** Loom — `memory.set_from_messages` per session, then
  `memory.search` at `top_k=200`, `search_mode=rrf`, no reranker (Loom's product default).
- **Reader:** gpt-5, the official fact-extraction prompt (`run_loom.py:_ANSWER_PROMPT`).
- **Judge:** gpt-5 semantic judge (the instrument managed memory platforms grade under),
  with gpt-4o as a reference grader.
- **Embeddings:** OpenAI `text-embedding-3-small`.

## Accuracy

| Metric | Loom |
|---|---|
| **Accuracy — gpt-5 reader + gpt-5 judge** | **88.4%** |
| Accuracy — gpt-5 reader + gpt-4o judge (reference) | 92.1% |

Per-category (gpt-5 judge):

| Category | Accuracy |
|---|---|
| single-session-user | 98.6% |
| single-session-assistant | 96.4% |
| temporal-reasoning | 89.3% |
| knowledge-update | 87.0% |
| multi-session | 83.5% |
| single-session-preference | 70.0% |

Retrieval recall (Loom's own retrieval quality, independent of the reader):

| Recall | Loom |
|---|---|
| Evidence session present in top-k | 99.6% |
| *Every* gold session present in top-k | 97.1% |
| Gold answer string present in a retrieved excerpt | 48.1% |

## How to read these numbers

**Accuracy is reader/judge-dominated, not retrieval-dominated.** Recall@200 is 99.6% —
Loom surfaces a memory from the gold evidence session on virtually every question. The
88.4% is what the gpt-5 reader, *given that context*, writes as a correct answer. On
identical Loom retrieval, swapping the reader gpt-4o→gpt-5 moves accuracy +6–8pt, and
swapping the judge gpt-4o→gpt-5 moves it ~−4pt (the gpt-5 judge is stricter, almost
entirely on the open-ended preference rubric). So the headline is as much a property of
the reader and judge as of the memory.

## Latency

Loom's default retrieval runs LLM-in-loop work on the read path — query planning, and a
HyDE recall-rescue when the top hit is weak. That helps on paraphrase-heavy or
sparse-memory workloads, but on LongMemEval (recall already 99.6%) it does **not** change
which memories are retrieved. Running retrieval at `--retrieval-budget fast` (pure vector
path, no read-path LLM) holds accuracy and recall while cutting latency ~7×:

| retrieval budget | accuracy (gpt-5 judge) | fact recall | search p50 |
|---|---|---|---|
| default | 88.9% | 47/99 | ~1,000 ms |
| **fast** (pure vector) | **90.9%** | 47/99 | **~140 ms** |

Paired: one ingest, the same 99 questions, only the retrieval budget differs. Recall is
identical (differs on 0 questions) and the accuracy gap is within n=99 noise — the point
is **fast loses nothing.** So for QA-style workloads `--retrieval-budget fast` is the
latency-optimal setting; the LLM-in-loop default buys recall robustness this workload
does not need.

## Context: other published accuracy numbers

LongMemEval-S accuracy is published by other systems under *their own* reader+judge, so
the figures are not directly comparable without matching the instrument:

- mem0: 91 (open source) / 94.4 (managed platform), gpt-5 reader + gpt-5 judge.
- Zep: 90.2 (blog, methodology undisclosed); 71.2 (reproducible paper, gpt-4o + official judge).

On the closest matched instrument (gpt-5 reader + gpt-5 judge), **Loom is 88.4% — about 3
points under mem0's open-source number.** A blind re-adjudication of the 23 questions where
the gpt-4o and gpt-5 judges disagreed found 18 were gpt-5 judge over-strictness (mostly the
preference rubric) and 5 genuine errors, which would put Loom's honestly-graded accuracy
nearer ~92%; but a fair use of that requires the same re-adjudication on the other systems'
answers, which has not been done. **The honest matched number is 88.4%.**

> The latency above and the context-token / HyDE-rate figures from
> `run_loom.py --measure-latency --metrics-out` are Loom's own operational
> measurements on this hardware. They are **not** comparable to other systems'
> published latency/token numbers (different harness, hardware, read path, and
> tokenizer), so no cross-system latency/token ranking is claimed here — only the
> Loom default-vs-fast comparison above.

## Reproduce

```bash
python loom/run_loom.py --base-url http://127.0.0.1:7777 \
  --dataset data/longmemeval_s_cleaned.json \
  --out loom/hyp.jsonl --top-k 200 --answer-model gpt-5 --ingest-concurrency 8
python src/evaluation/evaluate_qa.py gpt-5 loom/hyp.jsonl data/longmemeval_s_cleaned.json
```
