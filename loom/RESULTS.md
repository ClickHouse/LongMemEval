# Loom on LongMemEval-S — Benchmark Results

[Loom](https://github.com/ClickHouse/loom) is a ClickHouse-backed memory service.
This benchmark plugs Loom into LongMemEval-S at the **indexing + retrieval** stages;
the *reader* (answerer) and *judge* (grader) are LLMs — the standard measurement
apparatus, not part of Loom. The sections below report every dimension a memory
service is judged on: accuracy (across reader/judge choices), retrieval recall,
latency, token efficiency, and the HyDE fallback rate.

## Setup

- **Dataset:** LongMemEval-S, 500 questions; **491 scored**. The other ~9 hit
  reader-API timeouts (after retries) and were **excluded from this run's
  denominator**. The harness now records a placeholder empty answer on such
  failures — scored **incorrect**, not dropped — so later runs keep the full
  500 denominator; this ~9-question exclusion is specific to this earlier run.
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
- **gpt-5 reader + gpt-5 judge:** **88.2%** (independently reproduced at 88.4% on a second
  full-500 run). This is the headline.
- **Judge adjudication:** a blind 3-rater re-grade of the 23 questions where the gpt-4o and
  gpt-5 judges disagreed found **18 were gpt-5 over-strictness** (mostly the preference
  rubric) and **5 genuine errors** — implying honestly-graded accuracy nearer **~92%** once
  the over-strictness is removed. The headline reported here stays the un-adjudicated **88.2%**.
- **LLM-free read path confirm (full-500, gpt-5 reader):** with the read-path LLM legs
  default-off (see §3/§5), accuracy is **87.2% (gpt-5 judge) / 91.2% (gpt-4o judge)** —
  within ~1pt of the matrix, confirming the LLM-free read path holds accuracy.

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

## 3. Latency — read path is LLM-free by default

Loom's read path is now **LLM-free by default** (query planning + HyDE + query-rewrite all
default-off — see §5). `memory.search` wall-clock, measured over the **full-500** run on
real question-form queries (each search includes the remote `text-embedding-3-small` query
embedding + the ClickHouse query; no read-path LLM):

| read path | search p50 | search p95 |
|---|---|---|
| LLM-in-loop (prior default, full-500 instrumented) | ~1,920 ms | ~5,740 ms |
| **LLM-free (current default, full-500 clean)** | **~680 ms** | **~1,880 ms** |

Removing the read-path LLM legs cuts p50 ~2.8× (≈1,920 → 680 ms) at equal accuracy (§1). On
this workload (recall already 99.6%) the LLM-in-loop work does not change *what* is retrieved,
so it is latency without benefit. The ~680 ms p50 is dominated by the **remote embedding
round-trip + the CH query**, not Loom compute (a co-located/local embedder removes the embed
RTT; warm floor ~130 ms on short queries). The LLM legs stay available via
`--retrieval-budget deep` for paraphrase-heavy / sparse-memory workloads.

A smaller paired A/B (n=99, gpt-5 reader+judge) agreed in direction — `fast` budget held
accuracy (90.9% vs 88.9%) and recall (differs on 0 questions) — but its ~140 ms p50 was a
short-query lower bound, not the full-500 serving figure above.

## 4. Token efficiency — by top_k

Context handed to the reader (median, ~4 chars/token), measured on a populated namespace:

| top_k | memories served | ~tokens |
|---|---|---|
| 20 | 20 | ~1,927 |
| 50 | ~48 | ~4,177 |
| 200 | ~120–190 | ~11,500 |

Token cost is a **recall/cost knob**: the 88–92% accuracy above uses `top_k=200`. Smaller `k`
serves far less context but lowers recall and accuracy — the low token count and the high
accuracy do not co-exist at the same `k`.

## 5. HyDE fallback

The HyDE recall-rescue (an LLM that rewrites a weak query to an answer-shape and re-searches)
**fired on ~10% of queries** and, in a 60-question A/B, **changed which answer was retrieved on
0 of them** — it fires partly on abstention/preference questions it cannot help. On a
high-recall workload there is little to rescue, so it is mostly latency. It is now
**default-off** (along with the other two read-path LLM legs — §3), remaining available as an
opt-in knob (`--retrieval-budget deep`) for paraphrase-heavy or sparse-memory workloads.
The full-500 runs above report `hyde_fired_pct = 0.0` (LLM-free default).

## How to read these numbers

- **Accuracy is reader/judge-dominated, not retrieval-dominated** (recall@200 = 99.6%): the
  facts are in the retrieved context; the score is what the reader and judge make of them.
- **Latency and token figures are Loom's own operational measurements on this hardware.**
  They are setup-specific (harness, hardware, read path, and tokenizer all affect them), so
  treat them as Loom-vs-Loom (e.g. the budget comparison above), not as a portable ranking.

## Reproduce

```bash
# Accuracy (gpt-5 reader + gpt-5 judge) + retrieval metrics:
python loom/run_loom.py --base-url http://127.0.0.1:7777 \
  --dataset data/longmemeval_s_cleaned.json \
  --out loom/hyp.jsonl --metrics-out loom/metrics.json \
  --top-k 200 --answer-model gpt-5 --ingest-concurrency 8 --measure-latency
python src/evaluation/evaluate_qa.py gpt-5 loom/hyp.jsonl data/longmemeval_s_cleaned.json

# Latency-optimal (fast retrieval budget):
python loom/run_loom.py ... --retrieval-budget fast --measure-latency
```
