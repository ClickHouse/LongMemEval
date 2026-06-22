# Loom on LongMemEval-S — Benchmark Results

[Loom](https://github.com/ClickHouse/loom) is a ClickHouse-backed memory service.
This benchmark plugs Loom into LongMemEval-S at the **indexing + retrieval** stages;
the *reader* (answerer) and *judge* (grader) are LLMs — the standard measurement
apparatus, not part of Loom. The sections below report every dimension a memory
service is judged on: accuracy (across reader/judge choices), retrieval recall,
latency, token efficiency, and the HyDE fallback rate.

## Setup

- **Dataset:** LongMemEval-S, 500 questions (491 answered; a few dropped to reader API timeouts).
- **Indexing + retrieval:** Loom — `memory.set_from_messages` per session, then
  `memory.search` at `top_k=200`, `search_mode=rrf`, no reranker (product default).
- **Embeddings:** OpenAI `text-embedding-3-small`. **Extraction:** `gpt-4o-mini`.
- **Reader / judge:** OpenAI `gpt-4o` and `gpt-5` (varied below to show their effect).
- Single-node Loom + ClickHouse.

## 1. Accuracy — reader × judge

The end-to-end score depends as much on the **reader** and **judge** as on the memory.
Full-500, identical Loom retrieval, varying only the answerer and grader:

| reader ↓ \ judge → | gpt-4o judge | gpt-5 judge |
|---|---|---|
| **gpt-4o reader** | 84.9% | 82.2% |
| **gpt-5 reader** | **92.9%** | **88.2%** |

- **Reader effect:** gpt-4o → gpt-5 on the *same* retrieval = **+6 to +8pt**. The memory is
  identical; the answerer is the lever.
- **Judge effect:** gpt-5 judge is **stricter** (~3–5pt lower), almost entirely on the
  open-ended single-session-preference rubric.
- **Matched instrument** (gpt-5 reader + gpt-5 judge, what memory platforms publish under):
  **88.2%** (independently reproduced at 88.4% on a second full-500 run).
- **Judge adjudication:** a blind 3-rater re-grade of the 23 questions where the gpt-4o and
  gpt-5 judges disagreed found **18 were gpt-5 over-strictness** (mostly the preference
  rubric) and **5 genuine errors** — implying honestly-graded accuracy nearer **~92%**. That
  is only usable as a cross-system claim if the other systems' answers are re-adjudicated
  the same way, which has not been done; the matched number stays **88.2%**.

### Per-category (gpt-5 reader + gpt-5 judge)

| Category | Accuracy |
|---|---|
| single-session-user | 98.6% |
| single-session-assistant | 96.4% |
| temporal-reasoning | 89.3% |
| knowledge-update | 87.0% |
| multi-session | 83.5% |
| single-session-preference | 70.0% |

## 2. Retrieval recall (Loom's own quality, reader-independent)

| Recall metric | Loom |
|---|---|
| Evidence session present in top-k | **99.6%** |
| *Every* gold session present in top-k | 97.1% |
| Gold answer string present in a retrieved excerpt | 48.1% |

Recall@200 is 99.6% — Loom surfaces a memory from the gold evidence session on nearly every
question. This is *why* accuracy is reader/judge-dominated: the facts are in the context; the
score is what the reader makes of them.

## 3. Latency — by retrieval budget

Loom's default retrieval runs LLM-in-loop work on the read path (query planning, plus a HyDE
recall-rescue on a weak top hit). Paired A/B — one ingest, the same 99 questions, gpt-5
reader+judge, only the retrieval budget differs:

| retrieval budget | accuracy | fact recall | search p50 | search p95 |
|---|---|---|---|---|
| default (LLM-in-loop) | 88.9% | 47/99 | ~1,000 ms | ~5,200 ms |
| **`fast`** (pure vector) | **90.9%** | 47/99 | **~140 ms** | ~620 ms |

The `fast` budget **holds accuracy** (within n=99 noise) and **recall** (identical — differs
on 0 questions) at **~7× lower latency**. On this workload (recall already 99.6%) the
LLM-in-loop work does not change *what* is retrieved, so it is latency without benefit —
`--retrieval-budget fast` is the latency-optimal setting for QA workloads. (Floor for a
simple well-matched query is ~290 ms; the default path's p50 ranges ~1.0–1.9s depending on
query mix and load.)

## 4. Token efficiency — by top_k

Context handed to the reader (median, ~4 chars/token), measured on a populated namespace:

| top_k | memories served | ~tokens |
|---|---|---|
| 20 | 20 | ~1,927 |
| 50 | ~48 | ~4,177 |
| 200 | ~119–188 | ~11,290 |

Token cost is a **recall/cost knob**: the 88–92% accuracy above uses `top_k=200`. Smaller `k`
serves far less context but lowers recall and accuracy — the low token count and the high
accuracy do not co-exist at the same `k`.

## 5. HyDE fallback

The HyDE recall-rescue (an LLM that rewrites a weak query to an answer-shape and re-searches)
**fired on ~10% of queries** and, in a 60-question A/B, **changed which answer was retrieved on
0 of them** — it fires partly on abstention/preference questions it cannot help. On a
high-recall workload there is little to rescue, so it is mostly latency; it is left enabled
(a knob, not removed) because it can help paraphrase-heavy or sparse-memory workloads.

## How to read these numbers

- **Accuracy is reader/judge-dominated, not retrieval-dominated** (recall@200 = 99.6%).
- **Latency and token figures are Loom's own operational measurements on this hardware.**
  They are **not** comparable to other systems' published latency/token numbers (different
  harness, hardware, read path, tokenizer) — no cross-system latency/token ranking is claimed.
- Other systems' published *accuracy*: mem0 91 (OSS) / 94.4 (managed), Zep 90.2 (blog) /
  71.2 (reproducible paper). On the matched gpt-5 reader+judge instrument Loom is 88.2% —
  ~3pt under mem0's open-source number.

## Reproduce

```bash
# Accuracy (matched instrument) + retrieval metrics:
python loom/run_loom.py --base-url http://127.0.0.1:7777 \
  --dataset data/longmemeval_s_cleaned.json \
  --out loom/hyp.jsonl --metrics-out loom/metrics.json \
  --top-k 200 --answer-model gpt-5 --ingest-concurrency 8 --measure-latency
python src/evaluation/evaluate_qa.py gpt-5 loom/hyp.jsonl data/longmemeval_s_cleaned.json

# Latency-optimal (fast retrieval budget):
python loom/run_loom.py ... --retrieval-budget fast --measure-latency
```
