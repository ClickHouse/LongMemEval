# Benchmarking Loom on LongMemEval

[Loom](https://github.com/ClickHouse/loom) is a ClickHouse-backed memory service.
This integration plugs it into LongMemEval at the **indexing + retrieval** stages
and reuses the repo's official **reader** and **judge**, so the resulting QA
number is comparable to other published systems.

| stage | who does it |
|-------|-------------|
| indexing + retrieval | **Loom** (`loom/run_loom.py` — ingest via `memory.set_from_messages`, retrieve via `memory.search`) |
| reading (answer generation) | the official `src/generation/run_generation.py` prompt, replicated in `run_loom.py` (facts variant, step-by-step) |
| judging | the official `src/evaluation/evaluate_qa.py`, run unchanged on the hypotheses file |

Only the ingest+retrieve stage is Loom's; the reader and judge are the standard
ones. (The official `src/retrieval/run_retrieval.py` is built around in-process
retrievers — BM25 / Contriever / Stella / GTE over a flat corpus — and has no
hook for an external memory *service*, which is why this adapter exists.)

## Prerequisites

1. A running Loom server and a bearer token with write access. See the
   [Loom repo](https://github.com/ClickHouse/loom) for `make dev` and
   `mint-token`.
2. `OPENAI_API_KEY` in the environment (used by the reader model, default
   `gpt-4o`, and by the judge).
3. Install the adapter dep (everything else is already in the repo's requirements):

   ```bash
   pip install -r loom/requirements.txt
   ```

4. The dataset (LongMemEval-S, the variant other systems report on):

   ```bash
   wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json -O data/longmemeval_s_cleaned.json
   ```

## Run

```bash
export LOOM_TOKEN="<your loom token>"
export OPENAI_API_KEY="<your key>"

# 1) Ingest + retrieve with Loom, generate answers with the official reader,
#    write a hypotheses file. (Omit --limit for the full 500.)
python loom/run_loom.py \
  --base-url http://127.0.0.1:7777 \
  --dataset data/longmemeval_s_cleaned.json \
  --out loom/loom_hyp.jsonl \
  --limit 40 --shuffle   # omit --limit for the full 500; --shuffle gives a mixed sample

# 2) Grade with the OFFICIAL judge (gpt-4o, per-question-type prompts).
python src/evaluation/evaluate_qa.py gpt-4o loom/loom_hyp.jsonl data/longmemeval_s_cleaned.json
```

`run_loom.py` prints **evidence-session recall@k** (Loom's own retrieval metric);
`evaluate_qa.py` prints the **QA accuracy** (overall + per question type).

## Notes

- `--search-mode rrf` (default) lets Loom's query planner self-route; it never
  sees the gold `question_type`.
- Indexing is one `set_from_messages` per session (the natural unit), run
  concurrently (`--ingest-concurrency`) because each call does LLM extraction
  server-side; a -S question has ~50 sessions.
- `run_loom.py` creates a fresh namespace per question, so questions don't leak
  into each other.
