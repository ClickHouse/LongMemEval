"""Benchmark the Loom memory service on LongMemEval.

Loom (https://github.com/ClickHouse/loom) is a ClickHouse-backed memory service
exposing an HTTP API (`memory.set_from_messages` to index, `memory.search` to
retrieve). The official LongMemEval retrieval script (`src/retrieval/run_retrieval.py`)
is built around in-process retrievers (BM25 / Contriever / Stella / GTE) over a
flat corpus, so it cannot drive an external memory *service*. This adapter plugs
Loom in at the INDEXING + RETRIEVAL stages; the downstream READER and JUDGE stay
the official ones:

  indexing + retrieval : Loom  (this script)
  reading              : the official run_generation.py prompt (replicated here,
                         "facts extracted from history chats" + step-by-step)
  judging              : the official src/evaluation/evaluate_qa.py (run separately
                         on the hypotheses file this script writes)

Pipeline, per question:
  1. Create a fresh Loom namespace.
  2. Ingest every haystack session via memory.set_from_messages (one call per
     session, run concurrently), forwarding the session date as observation_date.
  3. memory.search the question -> top-k memories.
  4. Generate an answer from those memories with the official reader prompt.
  5. Record the answer (hypothesis) + evidence-session recall@k.

Outputs a hypotheses JSONL ({"question_id", "hypothesis"}) to grade with the
official judge:

    python loom/run_loom.py --base-url http://127.0.0.1:7777 --token "$LOOM_TOKEN" \
        --dataset data/longmemeval_s_cleaned.json --out loom/loom_hyp.jsonl
    python src/evaluation/evaluate_qa.py gpt-4o loom/loom_hyp.jsonl \
        data/longmemeval_s_cleaned.json

Requires: a running Loom server + token, OPENAI_API_KEY (for the reader model),
and `httpx` (see loom/requirements.txt).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import httpx

_HAYSTACK_DATE = re.compile(r"^(\d{4})/(\d{2})/(\d{2})")

# The official reader prompt for a fact-retrieval system, replicated verbatim
# from src/generation/run_generation.py (the "facts extracted from history
# chats" variant with step-by-step reasoning, cot=True). A single user message,
# no system prompt; the full completion is the hypothesis the judge grades.
_ANSWER_PROMPT = (
    "I will give you several facts extracted from history chats between you "
    "and a user. Please answer the question based on the relevant facts. "
    "Answer the question step by step: first extract all the relevant "
    "information, and then reason over the information to get the answer."
    "\n\n\nHistory Chats:\n\n{history}\n\nCurrent Date: {date}\nQuestion: "
    "{question}\nAnswer (step by step):"
)


def _iso(longmemeval_date: str) -> str:
    """'2023/04/10 (Mon) 17:50' -> '2023-04-10' (the ISO prefix Loom's
    observation_date accepts); '' if unparseable."""
    m = _HAYSTACK_DATE.match(longmemeval_date or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


async def _post(client: httpx.AsyncClient, url: str, body: dict, token: str,
                *, retries: int = 3) -> dict:
    """POST with retry-on-5xx + backoff. A dropped index/search would silently
    corrupt recall, so transient ClickHouse write contention must be ridden out."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last: httpx.Response | None = None
    for attempt in range(retries):
        r = await client.post(url, json=body, headers=headers, timeout=120.0)
        if r.status_code < 500:
            r.raise_for_status()
            return r.json()
        last = r
        await asyncio.sleep(0.5 * (2 ** attempt))
    assert last is not None
    last.raise_for_status()
    raise RuntimeError("unreachable")


def _history_block(hits: list[dict]) -> str:
    """Render retrieved memories as dated blocks, oldest first (mirrors the
    official run_generation.py per-session formatting)."""
    def date_of(h: dict) -> str:
        d = (h.get("valid_at") or h.get("temporal_anchor") or "").strip()
        return "" if d.startswith("1970-01-01") else d[:10]  # epoch sentinel = undated

    blocks = []
    for i, h in enumerate(sorted(hits, key=lambda h: date_of(h) or "9999")):
        content = (h.get("content_excerpt") or "").strip()
        blocks.append(f"### Memory {i + 1}:\nDate: {date_of(h) or 'unknown'}\n"
                      f"Content:\n{content}\n")
    return "\n".join(blocks) or "(no facts retrieved)"


async def _answer(client: httpx.AsyncClient, question: str, hits: list[dict],
                  question_date: str, model: str, api_key: str) -> str:
    body = {
        "model": model,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": _ANSWER_PROMPT.format(
            history=_history_block(hits),
            date=question_date or "unknown",
            question=question,
        )}],
    }
    r = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body, timeout=60.0,
    )
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


async def _run_item(client: httpx.AsyncClient, base_url: str, token: str, item: dict,
                    *, top_k: int, search_mode: str, ingest_conc: int,
                    model: str, api_key: str, item_sem: asyncio.Semaphore) -> dict:
    async with item_sem:
        ns = f"lme-{uuid.uuid4().hex[:10]}"
        identity = {"org": "dev", "namespace": ns, "agent": "lme-loom", "user_id": "-"}
        sessions = item.get("haystack_sessions", []) or []
        dates = item.get("haystack_dates", []) or []
        sids = item.get("haystack_session_ids", []) or []
        key_to_sessions: dict[str, set[str]] = {}

        # 1-2) Index every session (concurrently, bounded). One memory.set_from_messages
        # per session is the natural indexing unit and the only tractable granularity
        # on -S (~50 sessions/question). Each call runs LLM extraction server-side.
        ingest_sem = asyncio.Semaphore(max(1, ingest_conc))

        async def ingest(i: int, session: list) -> None:
            if not session:
                return
            sid = str(sids[i]) if i < len(sids) and sids[i] else ""
            body: dict = {**identity, "messages": session, "max_tokens": 4096}
            if sid:
                body["session_id"] = sid
            if i < len(dates) and _iso(dates[i]):
                body["observation_date"] = _iso(dates[i])
            async with ingest_sem:
                resp = await _post(client, base_url + "/v1/memory.set_from_messages", body, token)
            for w in resp.get("written") or []:
                key = str(w.get("memory_key") or "")
                if key:
                    key_to_sessions.setdefault(key, set()).add(sid)

        await asyncio.gather(*(ingest(i, s) for i, s in enumerate(sessions)))

        # 3) Retrieve. search_mode=rrf lets Loom's planner self-route (it never
        # sees the gold question_type). Built-in reranker enabled per request.
        search_body: dict = {
            **identity, "query": str(item["question"]), "top_k": top_k,
            "search_mode": search_mode, "alpha": 0.5, "include_top_n_unmatched": 120,
            "rerank": "builtin:openai",
        }
        q_iso = _iso(str(item.get("question_date", "")))
        if q_iso:
            search_body["observation_date"] = q_iso
        resp = await _post(client, base_url + "/v1/memory.search", search_body, token)
        hits = resp.get("results", [])

        # Evidence-session recall@k: did retrieval surface a memory from any
        # labelled gold evidence session? (the standard LongMemEval retrieval metric)
        answer_sessions = {str(s) for s in (item.get("answer_session_ids") or []) if s}
        retrieved = set()
        for h in hits[:top_k]:
            retrieved |= key_to_sessions.get(str(h.get("memory_key") or ""), set())
        retrieved.discard("")
        recalled = bool(answer_sessions & retrieved) if answer_sessions else False

        # 4) Read: generate an answer with the official reader prompt.
        hypothesis = await _answer(client, str(item["question"]), hits,
                                   str(item.get("question_date", "")), model, api_key)
        return {
            "question_id": str(item["question_id"]),
            "question_type": str(item.get("question_type", "")),
            "hypothesis": hypothesis,
            "recalled": recalled,
        }


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:7777", help="Loom server URL")
    p.add_argument("--token", default=os.environ.get("LOOM_TOKEN", ""), help="Loom bearer token")
    p.add_argument("--dataset", default="data/longmemeval_s_cleaned.json")
    p.add_argument("--out", default="loom/loom_hyp.jsonl", help="hypotheses JSONL for evaluate_qa.py")
    p.add_argument("--limit", type=int, default=0, help="cap questions (0 = all 500)")
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--search-mode", default="rrf",
                   help="Loom search mode; 'rrf' = let Loom's planner self-route")
    p.add_argument("--concurrency", type=int, default=4, help="questions in flight")
    p.add_argument("--ingest-concurrency", type=int, default=8,
                   help="concurrent index calls per question")
    p.add_argument("--answer-model", default="gpt-4o", help="reader model (OpenAI)")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("OPENAI_API_KEY is required (reader model).", file=sys.stderr)
        return 2
    ds_path = Path(args.dataset)
    if not ds_path.exists():
        print(f"missing dataset: {ds_path}\n  wget https://huggingface.co/datasets/"
              "xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json "
              f"-O {ds_path}", file=sys.stderr)
        return 2
    dataset = json.loads(ds_path.read_text())
    if args.limit > 0:
        dataset = dataset[: args.limit]
    print(f"loom-longmemeval: {len(dataset)} questions, top_k={args.top_k}, "
          f"search_mode={args.search_mode}, model={args.answer_model}", flush=True)

    item_sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        h = await client.get(args.base_url + "/v1/health", timeout=5.0)
        if h.status_code != 200:
            print(f"Loom server unhealthy: {h.status_code}", file=sys.stderr)
            return 2

        async def runner(item: dict) -> None:
            try:
                r = await _run_item(client, args.base_url, args.token, item,
                                    top_k=args.top_k, search_mode=args.search_mode,
                                    ingest_conc=args.ingest_concurrency,
                                    model=args.answer_model, api_key=api_key,
                                    item_sem=item_sem)
                results.append(r)
                print(f"  {'✓' if r['recalled'] else '✗'} {r['question_id']} "
                      f"[{r['question_type']}]", flush=True)
            except (httpx.HTTPError, KeyError) as e:
                print(f"  ! {item.get('question_id', '?')} ERROR: {e}", file=sys.stderr, flush=True)

        await asyncio.gather(*(runner(it) for it in dataset))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps({"question_id": r["question_id"], "hypothesis": r["hypothesis"]}) + "\n")

    # Retrieval recall@k by question type (Loom's own metric; the QA score comes
    # from evaluate_qa.py on the hypotheses file).
    by_type: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r["recalled"])
    print(f"\nrecall@{args.top_k} by question_type:")
    for qt in sorted(by_type):
        v = by_type[qt]
        print(f"  {qt:28} {sum(v)}/{len(v)} ({sum(v) / len(v) * 100:.1f}%)")
    tot = [r["recalled"] for r in results]
    print(f"  {'OVERALL':28} {sum(tot)}/{len(tot)} "
          f"({sum(tot) / len(tot) * 100:.1f}%)" if tot else "  (no results)")
    print(f"\nwrote {out_path}\nNow grade with the official judge:\n"
          f"  python src/evaluation/evaluate_qa.py gpt-4o {out_path} {args.dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
