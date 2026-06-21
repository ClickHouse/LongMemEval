# Loom on LongMemEval-S — Results

[Loom](https://github.com/ClickHouse/loom) is a ClickHouse-backed memory service.
This benchmark plugs Loom into LongMemEval-S at the **indexing + retrieval** stages;
the reader (answerer) and judge (grader) are LLMs — the standard measurement
apparatus, not part of Loom. It reports the four dimensions a memory service is
actually judged on: **answer accuracy, retrieval recall, token efficiency, and
search latency.**

## Setup

- **Dataset:** LongMemEval-S, 500 questions (491 answered; 9 dropped to reader API errors).
- **Indexing + retrieval:** Loom — `memory.set_from_messages` per session, then
  `memory.search` at `top_k=200`, `search_mode=rrf`, no reranker (Loom's product default).
- **Reader:** gpt-5, the official fact-extraction prompt (`run_loom.py:_ANSWER_PROMPT`).
- **Judge:** gpt-5 semantic judge (matched to how managed memory platforms grade),
  with gpt-4o as a reference grader.
- **Embeddings:** OpenAI `text-embedding-3-small`. Single-node Loom + ClickHouse.

## Results

| Metric | Loom |
|---|---|
| **Accuracy** — gpt-5 reader + gpt-5 judge | **88.4%** |
| Accuracy — gpt-5 reader + gpt-4o judge (reference) | 92.1% |
| Recall — evidence session in top-k | 99.6% |
| Recall — *every* gold session in top-k | 97.1% |
| Recall — gold answer string present in a retrieved excerpt | 48.1% |
| Context served to reader (median) @ top_k=200 | ~11,290 tokens |
| Context served to reader (median) @ top_k=50 | ~4,177 tokens |
| Search latency — clean p50 / p95 / floor | 1,920 / 5,742 / 290 ms |
| HyDE recall-fallback fired | 10% of queries |

Per-category accuracy (gpt-5 judge): single-session-user 98.6, single-session-assistant
96.4, temporal-reasoning 89.3, knowledge-update 87.0, multi-session 83.5,
single-session-preference 70.0.

## How to read these numbers

**Accuracy is reader-dominated, not retrieval-dominated.** Recall@200 is 99.6% —
Loom surfaces a memory from the gold evidence session on virtually every question.
The 88.4% is what the gpt-5 reader, *given that context*, writes as a correct answer.
On identical Loom retrieval, swapping the reader gpt-4o→gpt-5 moves accuracy +6–8pt,
and swapping the judge gpt-4o→gpt-5 moves it ~−4pt (the gpt-5 judge is stricter,
almost entirely on the open-ended preference rubric). So the headline is as much a
property of the reader and judge as of the memory.

**Token efficiency is a recall/cost knob, not a single number.** At `top_k=200`
(the setting that yields the accuracy above) Loom serves ~11,290 tokens of context.
At `top_k=50` it serves ~4,177 — but recall, and therefore accuracy, drops. The low
token count and the high accuracy do not co-exist at the same `k`.

**Latency is LLM-in-the-loop.** The ~290ms floor is the embedding RTT + a ClickHouse
vector read. The ~1,920ms p50 is because ~32% of queries (aggregational / multi-hop)
trigger a query-planning LLM call and ~10% trigger a HyDE recall-rescue LLM call, both
on the search critical path. They buy recall; they cost latency. A local embedder
removes the embedding RTT, and gating the planner/HyDE on simple queries would cut the
median (at a possible recall cost) — neither is applied in these numbers. Latency was
measured one query at a time on a quiesced server (no concurrent ingest); the in-run
under-load figure is higher and not reported here.

## Measurement scope (and why latency/tokens are not cross-system comparable)

The latency and token figures above are **Loom's own measurements on this hardware**,
reported to characterize Loom — not to rank it against other systems:

- **Latency** is the wall-clock of the `memory.search` call (client-side), which
  *includes* Loom's read-path query-planning LLM (~32% of queries), HyDE LLM (10%),
  and the remote embedding RTT. A graph-read memory store with no read-time LLM and
  local/cached embeddings is measuring a different operation — so published search
  latencies (e.g. ~100ms figures) are **not** like-for-like with this number.
- **Tokens** is `chars/4` of the retrieved context at `top_k=200`; other systems
  publish a real tokenizer count over a curated ~20-item context. Different tokenizer
  and different retrieval breadth.

A genuine cross-system latency/token comparison requires running every system through
one harness on one machine, timing the search call identically and tokenizing each
context the same way. That has not been done here.

## Context: other published accuracy numbers

LongMemEval-S accuracy is published by other systems under *their own* reader+judge,
so the figures are not directly comparable without matching the instrument:

- mem0: 91 (open source) / 94.4 (managed platform), gpt-5 reader + gpt-5 judge.
- Zep: 90.2 (blog, methodology undisclosed); 71.2 (reproducible paper, gpt-4o + official judge).

On the closest matched instrument (gpt-5 reader + gpt-5 judge), **Loom is 88.4% —
about 3 points under mem0's open-source number.** A blind re-adjudication of the 23
questions where the gpt-4o and gpt-5 judges disagreed found 18 were gpt-5 judge
over-strictness (mostly the preference rubric) and 5 genuine errors, which would put
Loom's honestly-graded accuracy nearer ~92%; but a fair use of that requires the same
re-adjudication on the other systems' answers, which has not been done. **The honest
matched number is 88.4%.**

## Reproduce

```bash
python loom/run_loom.py --base-url http://127.0.0.1:7777 \
  --dataset data/longmemeval_s_cleaned.json \
  --out loom/hyp.jsonl --metrics-out loom/metrics.json \
  --top-k 200 --answer-model gpt-5 --ingest-concurrency 8 --measure-latency
python src/evaluation/evaluate_qa.py gpt-5 loom/hyp.jsonl data/longmemeval_s_cleaned.json
```
