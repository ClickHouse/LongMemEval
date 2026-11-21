# Benchmarking Loom on LongMemEval

[Loom](https://github.com/ClickHouse/loom) is a ClickHouse-backed memory service.
This integration plugs it into LongMemEval at the **indexing + retrieval** stages,
reads with an adapted facts/CoT prompt, and grades QA accuracy with a modified
semantic judge (`src/evaluation/evaluate_qa.py`).

| stage | who does it |
|-------|-------------|
| indexing + retrieval | **Loom** (`loom/run_loom.py` — ingest via `memory.set_from_messages`, retrieve via `memory.search`) |
| reading (answer generation) | adapted from the official `src/generation/run_generation.py` facts/CoT template |
| judging | modified `src/evaluation/evaluate_qa.py` — semantic judge (gpt-4o), per question type |

The reader adds temporal precedence, counting distinct actual instances,
coreference, date arithmetic and optional server-computed aggregate guidance.
For non-reasoning models, only the temperature and output cap match the official
reader settings (`temperature=0`, `max_tokens=800`); the prompt is adapted.
The judge has rephrased per-question-type prompts, reasoning-model handling,
organization forwarding and file/reporting changes. It retains the base
judge's retry/backoff and yes-substring decision rule. These modified stages can
affect scores independently of Loom's indexing and retrieval.

The official retrieval script supports in-process retrievers — BM25, Contriever,
Stella and GTE over a flat corpus. This adapter supplies a separate path for an
external memory service.

## Prerequisites

1. A running Loom server and a bearer token with write access. See the
   [Loom repo](https://github.com/ClickHouse/loom) for `make dev` and
   `mint-token`.
   The adapter uses org `dev`, agent `lme-loom` and a random `lme-...`
   namespace per question. An isolated test deployment must authorize those
   scopes for extraction and search; the adapter does not provision grants.
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

# 1) Ingest + retrieve with Loom, generate answers with the adapted reader,
#    write a hypotheses file. (Omit --limit for the full 500.)
python loom/run_loom.py \
  --base-url http://127.0.0.1:7777 \
  --dataset data/longmemeval_s_cleaned.json \
  --out loom/loom_hyp.jsonl \
  --limit 40 --shuffle   # omit --limit for the full 500; --shuffle gives a mixed sample

# 2) Grade QA accuracy (modified gpt-4o judge, per question type).
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
- The harness requests inline extraction with `defer: false`. A deferred or
  queued response stops the run before retrieval; background extraction is not
  supported by this harness.
- Failed questions remain in QA and recall denominators. Loaded-latency
  percentiles use only recorded samples, with measured/skipped counts and null
  percentiles when no sample exists. They do not count every HTTP attempt.
- Context size estimates count rendered memory history and structured aggregate
  operands at four characters per token. They are not provider token counts or
  total prompt or cost measurements.
- `run_loom.py` creates a fresh namespace per question, so questions don't leak
  into each other.
