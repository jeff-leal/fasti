from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import openai
from pydantic import BaseModel, ValidationError, create_model

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional
    def tqdm(x, **k):
        return x

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
IRT = DATA / "irt"
GPT = DATA / "gpt_topics"
SCALING = DATA / "gpt_scaling"                   # where 40D wrote the shared sample
QUEUE_DIR = GPT / "queue24h"                     # uploaded requests and raw responses
PROMPTS = REPO / "code" / "prompts"
KEY_PATH = REPO.parent / "misc" / "key.txt"      # shared with the other GPT stages
# This machine inspects TLS, so the OpenAI SDK's bundled certifi roots reject the
# connection ("unable to get local issuer certificate").  The Windows trust store
# does hold the inspection root, so it is exported once to a PEM bundle and handed
# to httpx.  Falls back to a PowerShell HTTP client (OPENAI_USE_POWERSHELL_HTTP=1),
# which uses the Windows store natively but spawns a process per request.
CA_BUNDLE = Path(os.getenv("OPENAI_CA_BUNDLE")
                 or REPO.parent / "misc" / "win_ca_bundle.pem")

# The codebook loader.  Numbered stages cannot be imported by name, but this one
# is an unnumbered library, so a plain import is enough.
sys.path.insert(0, str(REPO / "code"))
import topic_codes  # noqa: E402

CODES = tuple(topic_codes.CODES)
TopicCode = Literal[CODES]
CODE_DEF = "TopicCode"           # the $defs name the batch keys reference
ENUM_LIMIT = 1000                # provider cap on enum values in one schema

# A namespace of its own: a topic answer and a left--right score are different
# payloads, so no checkpoint written by 40E can ever be read here, or the other
# way round.  The record carries the prompt hash and the hash of the batch's chunk
# ids, so a record is reused on resume only when it was coded under this prompt
# AND holds the sentences the batch holds now.
SCHEMA_VERSION = "topic_code_v01"
SEED = 20260721
BATCH_SIZE = 50
WORKERS = 6                                      # concurrent requests, live timing

# --------------------------------------------------------- 24h queue timing --- #
QUEUE_ENDPOINT = "/v1/chat/completions"
QUEUE_WINDOW = "24h"
REQUESTS_PER_JOB = 1000                          # requests per uploaded input file
QUEUE_FILE_LIMIT_BYTES = 190 * 1024 * 1024       # provider caps the input file at 200 MB
CUSTOM_ID_LIMIT = 64                             # provider caps custom_id length
POLL_SECONDS = 60
MAX_WAIT_HOURS = 26.0                            # the window is 24h; leave a margin
TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}
# A gap this long between two coded batches is an interruption, not slow coding:
# the widest gap ever observed inside a live run is a few seconds, even on the
# slowest corpus.  Used to keep dead time out of the running time the appendix
# reports when a run has to be resumed.
IDLE_GAP_SECONDS = 60

# ------------------------------------------------------------------ corpora --- #
# The same two sentence-chunk feathers the other stages read, keyed on `doc_id`,
# which equals the document id of stages 04 and 40D.
PROMPT_VERSIONS = {"v01": "topic_{c}_v01.txt"}
DEFAULT_PROMPT_VERSION = "v01"
SCOPES = ("sample", "full")
DEFAULT_SCOPE = "sample"

CORPORA = {
    "us": dict(text=DATA / "us" / "campaignview_chunks_sent.feather"),
    "br": dict(text=DATA / "br" / "br_manifestos_chunks_sent.feather"),
}
for _c, _cfg in CORPORA.items():
    _cfg["prompts"] = {v: PROMPTS / f.format(c=_c) for v, f in PROMPT_VERSIONS.items()}
    _cfg["sample"] = SCALING / f"gpt_sample_{_c}.csv"
    _cfg["matrices"] = IRT / f"irt_matrices_{_c}.npz"

# ------------------------------------------------------------------- models --- #
# The same two models as the ask-and-average baseline, so the two LLM stages are
# comparable.  Reasoning models (gpt-5 family) reject temperature != 1 unless the
# reasoning effort is "none", which is also what a cheap, near-deterministic
# classification wants.  Prices are USD per 1M tokens; verify before quoting.
MODELS = {
    "gpt4omini": dict(id="gpt-4o-mini", supports_temperature=True, supports_seed=True,
                      reasoning_effort=None,
                      price=dict(input=0.15, cached_input=0.075, output=0.60)),
    "gpt54mini": dict(id="gpt-5.4-mini", supports_temperature=True, supports_seed=True,
                      reasoning_effort="none",
                      price=dict(input=0.75, cached_input=0.075, output=4.50)),
}
PRICE_SOURCE = {"observed_date": "2026-07-22",
                "urls": ["https://developers.openai.com/api/docs/pricing",
                         "https://platform.openai.com/docs/guides/batch"],
                "note": "USD per 1M tokens. Batch API estimates use the 50% 24h discount."}


# ------------------------------------------------------------------ schema --- #
# pydantic 1.10.8 under anaconda, 2.x under the venv, and this stage has to run
# the same either way.
PYDANTIC_V2 = hasattr(BaseModel, "model_validate")


if PYDANTIC_V2:
    from pydantic import ConfigDict

    class StrictBatchBase(BaseModel):
        model_config = ConfigDict(extra="forbid")
else:
    class StrictBatchBase(BaseModel):
        class Config:
            extra = "forbid"


def _json_schema(model) -> dict[str, Any]:
    return model.model_json_schema() if PYDANTIC_V2 else model.schema()


def _parse_obj(model, obj):
    return (model.model_validate(obj) if PYDANTIC_V2 else model.parse_obj(obj))


def _to_dict(obj):
    return obj.model_dump() if PYDANTIC_V2 else obj.dict()


def batch_keys(n: int) -> list[str]:
    return [f"r{i:03d}" for i in range(1, n + 1)]


def make_batch_model(keys: list[str]):
    fields = {k: (TopicCode, ...) for k in keys}
    return create_model("BatchTopicCodes", __base__=StrictBatchBase, **fields)


def verify_schema(keys: list[str]) -> dict[str, Any]:
    """The strict schema for one batch, checked before every call.

    pydantic writes the code enum out once per key, and the provider caps a
    schema at 1000 enum values in total: 50 keys x 35 codes is 1750 and the call
    is rejected before it costs anything.  So the enum is hoisted into a single
    `$defs` entry that all 50 keys reference -- 35 values in the whole schema,
    with the same meaning.  Both forms are asserted here, the inline one pydantic
    produced and the hoisted one actually sent, so the hoist cannot quietly widen
    what the model is allowed to answer.
    """
    schema = _json_schema(make_batch_model(keys))
    props = schema.get("properties", {})
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise RuntimeError("schema must be object with additionalProperties=false")
    if set(props) != set(keys) or set(schema.get("required", [])) != set(keys):
        raise RuntimeError("schema keys do not exactly match local batch ids")
    for k in keys:
        enum = props[k].get("enum")
        if props[k].get("type") != "string" or enum is None:
            raise RuntimeError(f"{k}: topic schema must be a string enum")
        if list(enum) != list(CODES):
            raise RuntimeError(f"{k}: enum is not exactly the {len(CODES)} codebook codes")

    ref = f"#/$defs/{CODE_DEF}"
    schema = {**schema,
              "$defs": {CODE_DEF: {"type": "string", "enum": list(CODES)}},
              "properties": {k: {"$ref": ref} for k in keys}}
    if set(schema["properties"]) != set(keys) or set(schema["required"]) != set(keys):
        raise RuntimeError("hoisted schema keys do not exactly match local batch ids")
    if any(p != {"$ref": ref} for p in schema["properties"].values()):
        raise RuntimeError("hoisted schema: a key does not reference the code enum")
    if schema["$defs"][CODE_DEF]["enum"] != list(CODES):
        raise RuntimeError(f"hoisted schema: $defs enum is not the {len(CODES)} codes")
    n_enum = len(schema["$defs"][CODE_DEF]["enum"])
    if n_enum > ENUM_LIMIT:
        raise RuntimeError(f"{n_enum} codes exceeds the provider's {ENUM_LIMIT}-value "
                           f"limit on a single schema")
    return schema


def response_format(keys: list[str]) -> dict[str, Any]:
    return {"type": "json_schema",
            "json_schema": {"name": f"topic_codes_{len(keys)}",
                            "strict": True,
                            "schema": verify_schema(keys)}}


def format_batch(items: pd.DataFrame, keys: list[str]) -> str:
    """The user message: local ids and sentence text, and nothing else."""
    payload = [{"id": k, "text": "" if pd.isna(t) else str(t)}
               for k, t in zip(keys, items.chunk_text)]
    return json.dumps({"items": payload}, ensure_ascii=False)


def build_prompt(corpus: str, version: str) -> tuple[str, Path]:
    """The system prompt: the template on disk, with the codebook read into it.

    The codebook is not copied into the prompt file.  It is the author's workbook,
    it is edited there, and a copy on disk would be one more thing to keep in
    step.  Because the hash is taken over the assembled text, editing the workbook
    changes `prompt_sha256` and every stored answer coded under the old codebook
    is re-coded rather than reused.
    """
    path = CORPORA[corpus]["prompts"][version]
    template = path.read_text(encoding="utf-8").strip()
    if "{codebook}" not in template:
        raise SystemExit(f"{path.name}: no {{codebook}} placeholder to fill")
    return template.replace("{codebook}", topic_codes.codebook_block()), path


# ------------------------------------------------------------------- inputs --- #
def load_chunks(corpus: str, scope: str, limit_chunks: int | None,
                doc_ids: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sentence-chunks of the selected documents, in a stable, reproducible order."""
    cfg = CORPORA[corpus]
    if scope == "sample":
        keep = set(pd.read_csv(cfg["sample"], dtype=str).doc_id)
        source = cfg["sample"].name
    else:
        keep = set(map(str, np.load(cfg["matrices"],
                                    allow_pickle=True)["docs"].tolist()))
        source = cfg["matrices"].name
    df = pd.read_feather(cfg["text"], columns=["chunk_id", "doc_id", "chunk_text"])
    df["doc_id"] = df.doc_id.astype(str)
    df["chunk_id"] = df.chunk_id.astype(str)
    if not doc_ids:
        df = df[df.doc_id.isin(keep)].copy()
        got = df.doc_id.nunique()
        if got < len(keep):
            print(f"  WARNING [{corpus}]: {len(keep) - got:,} selected documents have "
                  f"no text in {cfg['text'].name}")
    else:
        got = len(keep)
    # Stable order so batch boundaries (and the checkpoint) are reproducible.
    df = df.sort_values(["doc_id", "chunk_id"], kind="stable").reset_index(drop=True)
    full_info = {"scope": scope, "scope_source": source,
                 "full_sample_docs": int(got), "full_sample_chunks": int(len(df))}
    # Named documents.  A diagnostic on chosen documents, not the estimate, so it
    # reaches the whole corpus rather than only the selection -- such a run is
    # always run-tagged, so it can never touch the production codes.
    if doc_ids:
        missing = [d for d in doc_ids if d not in set(df.doc_id)]
        if missing:
            raise SystemExit(f"[{corpus}] not in the corpus: {missing}")
        df = df[df.doc_id.isin(doc_ids)].copy().reset_index(drop=True)
        outside = [d for d in doc_ids if d not in keep]
        full_info["doc_ids"] = list(doc_ids)
        full_info["doc_ids_outside_scope"] = outside
        full_info["full_sample_chunks"] = int(len(df))
        if outside:
            print(f"  note [{corpus}]: {len(outside)} of {len(doc_ids)} named documents "
                  f"are outside the {scope} selection")
    if limit_chunks:
        df = df.head(limit_chunks).copy()
    full_info.update({"selected_docs": int(df.doc_id.nunique()),
                      "selected_chunks": int(len(df))})
    print(f"[{corpus}/{scope}] {full_info['selected_docs']:,} documents -> "
          f"{full_info['selected_chunks']:,} sentence-chunks to code")
    return df, full_info


# --------------------------------------------------------------- api + cost --- #
def call_kwargs(model: dict, seed: int) -> dict:
    kw: dict = {}
    if model["supports_temperature"]:
        kw["temperature"] = 0
    if model["supports_seed"]:
        kw["seed"] = seed
    if model["reasoning_effort"]:
        kw["reasoning_effort"] = model["reasoning_effort"]
    return kw


def request_body(model: dict, prompt: str, items: pd.DataFrame,
                 keys: list[str], seed: int) -> dict:
    """The one request both timings send, so neither can drift from the other."""
    return dict(model=model["id"],
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": format_batch(items, keys)}],
                response_format=response_format(keys),
                **call_kwargs(model, seed))


def zero_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
            "total_tokens": 0}


def usage_to_dict(usage) -> dict:
    if usage is None:
        return zero_usage()
    u = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    det = u.get("prompt_tokens_details") or {}
    return {"prompt_tokens": int(u.get("prompt_tokens") or 0),
            "completion_tokens": int(u.get("completion_tokens") or 0),
            "cached_tokens": int((det or {}).get("cached_tokens") or 0),
            "total_tokens": int(u.get("total_tokens") or 0)}


def estimate_cost(usage: dict, price: dict) -> float:
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int(usage.get("cached_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    uncached = max(prompt - cached, 0)
    return (uncached * price["input"]
            + cached * price["cached_input"]
            + completion * price["output"]) / 1_000_000


def proxy_usage(chunks: pd.DataFrame, prompt: str, n_batches: int) -> dict:
    """A cost proxy measured on the text actually selected, not a fixed constant.

    The system prompt carries the whole codebook and is re-sent with every batch,
    so it is amortized over 50 sentences here rather than assumed away.  Four
    characters to the token is the usual rough rule for both languages; caching is
    NOT modelled, so a real run comes in under this.
    """
    n = len(chunks)
    chars = float(chunks.chunk_text.astype(str).str.len().sum())
    sys_tokens = len(prompt) / 4
    return {"prompt_tokens": int(n_batches * sys_tokens + chars / 4 + n * 12),
            "completion_tokens": int(n * 8),      # {"r001":"MAC", ...}
            "cached_tokens": 0}


_ps_seq = itertools.count()


def ps_chat_completion(request: dict) -> dict:
    tmp = GPT / "_tmp_openai"
    tmp.mkdir(parents=True, exist_ok=True)
    stem = f"chat_{os.getpid()}_{threading.get_ident()}_{next(_ps_seq):06d}"
    body_path = tmp / f"{stem}_request.json"
    out_path = tmp / f"{stem}_response.json"
    body_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    ps = (
        "$ErrorActionPreference='Stop'; "
        "$key = if ($env:OPENAI_API_KEY) { $env:OPENAI_API_KEY } "
        f"else {{ (Get-Content '{KEY_PATH}' -Raw).Trim() }}; "
        "$headers = @{Authorization = \"Bearer $key\"}; "
        f"$r = Invoke-RestMethod -Uri 'https://api.openai.com/v1/chat/completions' "
        f"-Method Post -Headers $headers -ContentType 'application/json' -InFile '{body_path}'; "
        f"$r | ConvertTo-Json -Depth 100 | Set-Content -Path '{out_path}' -Encoding UTF8"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    payload = json.loads(out_path.read_text(encoding="utf-8-sig"))
    for p in (body_path, out_path):          # keep the pair only when it failed
        p.unlink(missing_ok=True)
    return payload


def _field(obj, name):
    """One accessor for both the SDK objects and the plain dicts of a batch line."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def parse_completion(completion, keys: list[str], pairs: list[tuple[str, str]],
                     batch_id: str) -> dict:
    """Validate one chat completion into topic records keyed by the local ids.

    Raises unless the answer carries exactly the local ids of this batch, each
    with one code of the codebook.  The class label is attached here, locally,
    from `topic_codes.py`, and the document and chunk ids from the batch manifest
    -- the model returned only the code and never saw the ids.
    """
    BatchModel = make_batch_model(keys)
    choice = (_field(completion, "choices") or [None])[0]
    if choice is None:
        raise RuntimeError("no choices in response")
    if _field(choice, "finish_reason") == "length":
        raise RuntimeError("finish_reason=length")
    if _field(choice, "finish_reason") == "content_filter":
        raise RuntimeError("finish_reason=content_filter")
    msg = _field(choice, "message")
    refusal = _field(msg, "refusal")
    if refusal:
        raise RuntimeError(f"refusal: {refusal}")
    content = _field(msg, "content")
    try:
        parsed = _to_dict(_parse_obj(BatchModel, json.loads(content)))
    except (TypeError, json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(f"invalid structured output: {exc}") from exc
    if set(parsed.keys()) != set(keys):
        raise RuntimeError("key mismatch: response keys do not exactly match local ids")

    recs = []
    for k, (cid, did) in zip(keys, pairs):
        code = parsed[k]
        recs.append({"batch_id": batch_id, "id": k, "chunk_id": cid, "doc_id": did,
                     "code": code, "label": topic_codes.LABEL[code]})
    return {"items": recs,
            "usage": usage_to_dict(_field(completion, "usage")),
            "returned_model": _field(completion, "model"),
            "fingerprint": _field(completion, "system_fingerprint")}


def call_api(client, model: dict, prompt: str, items: pd.DataFrame,
             keys: list[str], seed: int, batch_id: str) -> dict:
    request = request_body(model, prompt, items, keys, seed)
    completion = (ps_chat_completion(request) if client == "powershell"
                  else client.chat.completions.create(**request))
    pairs = list(zip(items.chunk_id.astype(str), items.doc_id.astype(str)))
    return parse_completion(completion, keys, pairs, batch_id)


# -------------------------------------------------------------- checkpoints --- #
def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_checkpoint(path: Path, prompt_sha: str, chunk_sha: dict[str, str]):
    """Records this run may reuse.

    A batch id is positional (``_b0001``), so it alone cannot say whether a stored
    record holds the sentences that batch holds now: change the scope, the
    document selection or the corpus file and position 1 is a different batch of
    text.  Every record therefore carries a hash of its chunk ids, and one that
    does not match the batch now standing at that position is re-coded rather than
    reused.  Without this the mismatch is invisible, because the (batch_id, id)
    pairs still line up and every validation check still passes.
    """
    done, items, usage = set(), [], zero_usage()
    stale_prompt = stale_content = 0
    if not path.exists():
        return done, items, usage
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # An interrupted run can leave one truncated trailing record; the
                # batch simply counts as not done and is coded again on resume.
                print(f"  skipping unreadable checkpoint line {ln} in {path.name}")
                continue
            if rec.get("schema_version") != SCHEMA_VERSION:
                stale_prompt += 1
                continue
            # A record coded under a different prompt -- or a different codebook,
            # which the hash covers -- is not this run's data.
            if rec.get("prompt_sha256") not in (None, prompt_sha):
                stale_prompt += 1
                continue
            bid = rec.get("batch_id", str(rec.get("batch_idx")))
            if rec.get("chunk_sha256") != chunk_sha.get(bid):
                stale_content += 1
                continue
            done.add(bid)
            items.extend(rec["items"])
            for k in usage:
                usage[k] += int((rec.get("usage") or {}).get(k) or 0)
    if stale_prompt:
        print(f"  ignoring {stale_prompt:,} checkpoint record(s) in {path.name} from "
              f"an earlier schema version, prompt or codebook; they are re-coded, "
              f"not reused")
    if stale_content:
        print(f"  ignoring {stale_content:,} checkpoint record(s) in {path.name} whose "
              f"sentences differ from the batch now at that position; re-coded")
    return done, items, usage


def checkpoint_stamps(path: Path, prompt_sha: str) -> list[datetime]:
    """When this run's batches were coded, in order.

    Only this run's own records count: a file that also holds records from an
    earlier schema version or an earlier prompt would otherwise report the span
    between two unrelated runs.
    """
    stamps = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if (rec.get("schema_version") != SCHEMA_VERSION
                            or rec.get("prompt_sha256") not in (None, prompt_sha)):
                        continue
                    stamps.append(datetime.fromisoformat(rec["created_at"]))
                except Exception:
                    continue
    return sorted(stamps)


def checkpoint_span(path: Path, prompt_sha: str) -> float | None:
    """Seconds from the first to the last coded batch, across every invocation."""
    s = checkpoint_stamps(path, prompt_sha)
    return round((s[-1] - s[0]).total_seconds(), 1) if len(s) > 1 else None


def checkpoint_idle(path: Path, prompt_sha: str) -> float:
    """Seconds inside the span during which nothing was being coded.

    A run that is interrupted and resumed leaves one long gap between consecutive
    records -- the time it sat dead -- while coding itself never pauses for more
    than a few seconds even at the slowest observed rate.  Anything longer than
    IDLE_GAP_SECONDS is therefore an interruption, not work, and the appendix must
    not count it as running time.
    """
    s = checkpoint_stamps(path, prompt_sha)
    return round(sum(g for g in ((s[i + 1] - s[i]).total_seconds()
                                 for i in range(len(s) - 1))
                     if g > IDLE_GAP_SECONDS), 1)


def save_checkpoint(path: Path, batch_idx: int, batch_id: str, result: dict,
                    prompt_sha: str, batch_timing: str, chunk_sha: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"schema_version": SCHEMA_VERSION, "batch_idx": batch_idx,
                            "batch_id": batch_id, "prompt_sha256": prompt_sha,
                            "chunk_sha256": chunk_sha,
                            "batch_timing": batch_timing,
                            "items": result["items"], "usage": result["usage"],
                            "returned_model": result["returned_model"],
                            "fingerprint": result["fingerprint"], "created_at": now()},
                           ensure_ascii=False) + "\n")


def build_manifest(run_key: str, batches: list[pd.DataFrame]) -> list[dict]:
    """The local id -> chunk_id/doc_id map, the only place the ids are joined."""
    out = []
    for bi, batch in enumerate(batches):
        bid = f"{run_key}_b{bi + 1:04d}"
        keys = batch_keys(len(batch))
        cids = [str(c) for c in batch.chunk_id]
        out.append({"batch_idx": bi, "batch_id": bid, "keys": keys,
                    # Identifies the sentences themselves, so a stored record can be
                    # checked against the batch now standing at this position.
                    "chunk_sha256": hashlib.sha256(
                        "\n".join(cids).encode("utf-8")).hexdigest(),
                    "pairs": [(c, str(d)) for c, d in zip(cids, batch.doc_id)],
                    "items": [{"id": k, "chunk_id": cid, "doc_id": str(did)}
                              for k, cid, did in zip(keys, cids, batch.doc_id)]})
    return out


# ------------------------------------------------------------ queue24h jobs --- #
def jobs_path(run_key: str) -> Path:
    return GPT / f"queue24h_jobs_{run_key}.jsonl"


def load_jobs(path: Path) -> dict[str, dict]:
    """The job ledger, folded by job id so the last record wins."""
    jobs: dict[str, dict] = {}
    if not path.exists():
        return jobs
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            jid = rec.get("job_id")
            if jid:
                jobs[jid] = {**jobs.get(jid, {}), **rec}
    return jobs


def append_job(path: Path, rec: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_queue_input(path: Path, model: dict, prompt: str,
                      group: list[tuple[int, str, pd.DataFrame]], seed: int) -> int:
    with path.open("w", encoding="utf-8") as f:
        for _, bid, batch in group:
            if len(bid) > CUSTOM_ID_LIMIT:
                raise RuntimeError(f"custom_id '{bid}' exceeds {CUSTOM_ID_LIMIT} "
                                   f"characters; shorten --run-tag")
            keys = batch_keys(len(batch))
            f.write(json.dumps({"custom_id": bid, "method": "POST",
                                "url": QUEUE_ENDPOINT,
                                "body": request_body(model, prompt, batch, keys, seed)},
                               ensure_ascii=False) + "\n")
    return path.stat().st_size


def submit_jobs(client, run_key: str, model: dict, prompt: str, prompt_sha: str,
                pending: list[tuple[int, str, pd.DataFrame]], args, jpath: Path):
    """Upload the pending batches and open one 24h job per input file.

    The ledger record is written the moment the provider returns a job id, before
    anything is awaited, so an interrupted invocation can always find the job
    again instead of opening a second one for the same sentences.
    """
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    groups = [pending[i:i + args.requests_per_job]
              for i in range(0, len(pending), args.requests_per_job)]
    print(f"  submitting {len(pending)} batch(es) as {len(groups)} job(s) "
          f"on the {QUEUE_WINDOW} queue")
    for group in groups:
        stem = f"{run_key}_b{group[0][0] + 1:04d}-{group[-1][0] + 1:04d}"
        in_path = QUEUE_DIR / f"queue_in_{stem}.jsonl"
        size = write_queue_input(in_path, model, prompt, group, args.seed)
        if size > QUEUE_FILE_LIMIT_BYTES:
            raise RuntimeError(f"{in_path.name} is {size / 1e6:.0f} MB, over the "
                               f"provider's input-file limit; lower --requests-per-job")
        with in_path.open("rb") as fh:
            up = client.files.create(file=fh, purpose="batch")
        job = client.batches.create(input_file_id=up.id, endpoint=QUEUE_ENDPOINT,
                                    completion_window=QUEUE_WINDOW,
                                    metadata={"run": run_key,
                                              "schema_version": SCHEMA_VERSION})
        append_job(jpath, {"schema_version": SCHEMA_VERSION, "job_id": job.id,
                           "input_file_id": up.id, "input_file": str(in_path),
                           "input_file_bytes": int(size),
                           "n_requests": len(group),
                           "batch_idx": [bi for bi, _, _ in group],
                           "batch_ids": [bid for _, bid, _ in group],
                           "model_id": model["id"], "prompt_sha256": prompt_sha,
                           "submitted_at": now(),
                           "created_at_unix": _field(job, "created_at"),
                           "status": _field(job, "status")})
        print(f"    job {job.id}  ({len(group)} requests, {size / 1e6:.1f} MB, "
              f"status {_field(job, 'status')})")


def poll_jobs(client, job_ids: list[str], args) -> tuple[dict[str, Any], set[str]]:
    """Wait until every job reaches a terminal state, or the wait limit expires.

    A status read is a GET, never a re-send of a classification request, so a
    transient failure here is logged and polled again rather than abandoning a
    job that is already paid for and running.
    """
    deadline = time.time() + args.max_wait_hours * 3600
    states: dict[str, Any] = {}
    finished: set[str] = set()
    while len(finished) < len(job_ids):
        for jid in job_ids:
            if jid in finished:
                continue
            try:
                job = client.batches.retrieve(jid)
            except Exception as exc:
                print(f"  status read failed for {jid} ({exc}); retrying at the "
                      f"next poll")
                continue
            states[jid] = job
            if _field(job, "status") in TERMINAL_STATES:
                finished.add(jid)
        if len(finished) >= len(job_ids):
            break
        if time.time() > deadline:
            print(f"  wait limit of {args.max_wait_hours} h reached and "
                  f"{len(job_ids) - len(finished)} job(s) are still running. "
                  f"Nothing is lost: re-run this command later to collect them.")
            break
        for jid in job_ids:
            job = states.get(jid)
            if job is None or jid in finished:
                continue
            rc = _field(job, "request_counts")
            counts = (f"{_field(rc, 'completed') or 0}/{_field(rc, 'total') or 0} done, "
                      f"{_field(rc, 'failed') or 0} failed") if rc else ""
            print(f"    {jid}: {_field(job, 'status')}  {counts}", flush=True)
        time.sleep(args.poll_seconds)
    return states, finished


def fetch_file(client, file_id: str, path: Path) -> str:
    """Download one batch result file and keep it beside the data, for the record."""
    content = client.files.content(file_id)
    text = content.text if hasattr(content, "text") else content.read().decode("utf-8")
    path.write_text(text, encoding="utf-8")
    return text


def harvest_job(client, job, by_id: dict[str, dict], done: set[str], ckpt: Path,
                prompt_sha: str) -> tuple[list[dict], dict, list[dict]]:
    """Turn one finished job into checkpoint records.  Never re-sends anything."""
    jid = _field(job, "id")
    items, usage, failed = [], zero_usage(), []
    status = _field(job, "status")
    if status != "completed":
        print(f"    job {jid} ended as {status}")

    out_id = _field(job, "output_file_id")
    if out_id:
        text = fetch_file(client, out_id, QUEUE_DIR / f"queue_out_{jid}.jsonl")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            bid = rec.get("custom_id")
            entry = by_id.get(bid)
            if entry is None or bid in done:
                continue                       # not this run's, or already coded
            resp = rec.get("response") or {}
            if rec.get("error") or resp.get("status_code") != 200:
                failed.append({"batch_idx": entry["batch_idx"], "batch_id": bid,
                               "error": json.dumps(rec.get("error")
                                                   or resp.get("body"))[:500]})
                continue
            try:
                result = parse_completion(resp.get("body"), entry["keys"],
                                          entry["pairs"], bid)
            except Exception as exc:
                failed.append({"batch_idx": entry["batch_idx"], "batch_id": bid,
                               "error": str(exc)})
                continue
            save_checkpoint(ckpt, entry["batch_idx"], bid, result, prompt_sha,
                            "queue24h", entry["chunk_sha256"])
            done.add(bid)
            items.extend(result["items"])
            for k in usage:
                usage[k] += result["usage"][k]

    err_id = _field(job, "error_file_id")
    if err_id:
        text = fetch_file(client, err_id, QUEUE_DIR / f"queue_err_{jid}.jsonl")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            entry = by_id.get(rec.get("custom_id"))
            failed.append({"batch_idx": entry["batch_idx"] if entry else None,
                           "batch_id": rec.get("custom_id"),
                           "error": json.dumps(rec.get("error"))[:500]})
    return items, usage, failed


def queue_wall_seconds(jobs: dict[str, dict]) -> float | None:
    """Provider-clocked time from the first job opening to the last one finishing."""
    starts = [j["created_at_unix"] for j in jobs.values() if j.get("created_at_unix")]
    ends = [j["completed_at_unix"] for j in jobs.values() if j.get("completed_at_unix")]
    if not starts or not ends:
        return None
    return round(float(max(ends) - min(starts)), 1)


def run_queue24h(client, run_key: str, model: dict, prompt: str, prompt_sha: str,
                 batches: list[pd.DataFrame], manifest: list[dict],
                 done: set[str], ckpt: Path, args):
    """Submit, wait, and collect on the 24h queue.  Resumable, and never retries."""
    jpath = jobs_path(run_key)
    all_jobs = load_jobs(jpath)
    # A job opened under a different prompt belongs to a different run and is left
    # alone: neither counted as covering these batches nor collected into this
    # checkpoint.  Prompt versions never mix in one output file.
    jobs = {jid: j for jid, j in all_jobs.items()
            if j.get("prompt_sha256") in (None, prompt_sha)}
    if len(all_jobs) > len(jobs):
        print(f"  ignoring {len(all_jobs) - len(jobs)} job(s) in {jpath.name} opened "
              f"under a different prompt")
    by_id = {m["batch_id"]: m for m in manifest}
    submitted = {bid for j in jobs.values() for bid in (j.get("batch_ids") or [])}

    # A batch already sitting in a job is never sent again on its own.  One that
    # came back failed stays failed unless the author asks for it explicitly.
    eligible = submitted if args.resubmit_failed else set()
    pending = [(m["batch_idx"], m["batch_id"], batches[m["batch_idx"]])
               for m in manifest
               if m["batch_id"] not in done
               and (m["batch_id"] not in submitted or m["batch_id"] in eligible)]
    if pending:
        submit_jobs(client, run_key, model, prompt, prompt_sha, pending, args, jpath)
        jobs = {jid: j for jid, j in load_jobs(jpath).items()
                if j.get("prompt_sha256") in (None, prompt_sha)}
    elif not jobs:
        print("  nothing to submit and no open jobs")
    else:
        print(f"  nothing new to submit; {len(submitted)} batch(es) already sent")

    live = [jid for jid, j in jobs.items() if not j.get("harvested_at")]
    if args.submit_only:
        print(f"  --submit-only: {len(live)} job(s) open on the {QUEUE_WINDOW} queue. "
              f"Re-run the same command without --submit-only to collect them.")
        return [], zero_usage(), [], jobs, True
    if not live:
        return [], zero_usage(), [], jobs, False

    print(f"  waiting on {len(live)} job(s); polling every {args.poll_seconds}s")
    states, finished = poll_jobs(client, live, args)

    items, usage, failed = [], zero_usage(), []
    for jid in finished:
        job = states[jid]
        got, u, fail = harvest_job(client, job, by_id, done, ckpt, prompt_sha)
        items.extend(got)
        failed.extend(fail)
        for k in usage:
            usage[k] += u[k]
        rec = dict(jobs.get(jid, {}))
        rec.update({"job_id": jid, "status": _field(job, "status"),
                    "harvested_at": now(),
                    "created_at_unix": _field(job, "created_at") or rec.get("created_at_unix"),
                    "completed_at_unix": (_field(job, "completed_at")
                                          or _field(job, "failed_at")
                                          or _field(job, "expired_at")),
                    "output_file_id": _field(job, "output_file_id"),
                    "error_file_id": _field(job, "error_file_id"),
                    "n_coded": len(got), "n_failed": len(fail)})
        append_job(jpath, rec)
        jobs[jid] = rec
        print(f"    job {jid}: {len(got):,} sentences coded, {len(fail)} failed")
    incomplete = len(live) - len(finished)
    return items, usage, failed, jobs, incomplete > 0


# --------------------------------------------------------------------- run --- #
def run_live(client, model: dict, prompt: str, prompt_sha: str,
             batches: list[pd.DataFrame], manifest: list[dict],
             done: set[str], ckpt: Path, args, desc: str = "coding"):
    """The concurrent path: one chat completion per batch, answered immediately."""
    # Requests run concurrently -- the wall clock is dominated by API latency, not
    # by local work.  Only this thread appends to the checkpoint and accumulates
    # usage, so the file stays one-record-per-line and needs no lock.  A batch that
    # raises is recorded and skipped; it is never re-sent automatically, and a later
    # re-run of this script picks it up through the ordinary resume path.
    items, usage, failed = [], zero_usage(), []
    pending = [m for m in manifest if m["batch_id"] not in done]

    def work(m: dict):
        return m, call_api(client, model, prompt, batches[m["batch_idx"]],
                           m["keys"], args.seed, m["batch_id"])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, m): m for m in pending}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=desc, unit="batch"):
            m = futures[fut]
            try:
                m, result = fut.result()
            except Exception as exc:
                print(f"  FAILED batch {m['batch_idx'] + 1}/{len(batches)}: {exc}")
                failed.append({"batch_idx": m["batch_idx"], "batch_id": m["batch_id"],
                               "error": str(exc)})
                continue
            save_checkpoint(ckpt, m["batch_idx"], m["batch_id"], result, prompt_sha,
                            "live", m["chunk_sha256"])
            done.add(m["batch_id"])
            items.extend(result["items"])
            for k in usage:
                usage[k] += result["usage"][k]
    return items, usage, failed


def run_one(client, model_key: str, corpus: str, args) -> dict:
    model = MODELS[model_key]
    prompt, prompt_file = build_prompt(corpus, args.prompt_version)
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    chunks, chunk_info = load_chunks(corpus, args.scope, args.limit_chunks, args.doc_ids)
    batches = [chunks.iloc[i:i + args.batch_size]
               for i in range(0, len(chunks), args.batch_size)]
    run_suffix = f"_{args.run_tag}" if args.run_tag else ""
    # The scope is part of the run key, so a sample run and a full run never share
    # a checkpoint, a job ledger, an output file or a batch id.
    run_key = f"{model_key}_{corpus}_{args.scope}{run_suffix}"
    manifest = build_manifest(run_key, batches)

    ckpt = GPT / f"checkpoint_{run_key}.jsonl"
    done, all_items, usage = load_checkpoint(
        ckpt, prompt_sha, {m["batch_id"]: m["chunk_sha256"] for m in manifest})
    if done:
        print(f"  resuming {run_key}: {len(done)}/{len(batches)} batches done")

    if args.dry_run or client is None:
        for b in batches:
            verify_schema(batch_keys(len(b)))
        n = len(chunks)
        proxy = proxy_usage(chunks, prompt, len(batches))
        est = estimate_cost(proxy, model["price"])
        print(f"  DRY RUN {run_key}: {n:,} chunks, {len(batches)} batches, "
              f"rough cost estimate ${est:.2f} standard / ${0.5 * est:.2f} on the "
              f"{QUEUE_WINDOW} queue (measured text, {len(prompt) / 4:.0f}-token "
              f"codebook prompt per batch, caching not modelled)")
        return {"model": model_key, "corpus": corpus, "scope": args.scope,
                "dry_run": True, "n_chunks": int(n), "n_batches": len(batches),
                "prompt_sha256": prompt_sha,
                "rough_cost_usd": round(est, 4),
                "rough_batch24h_cost_usd": round(0.5 * est, 4)}

    jobs: dict[str, dict] = {}
    still_open = False
    started = time.time()
    if args.batch_timing == "queue24h":
        new_items, new_usage, failed, jobs, still_open = run_queue24h(
            client, run_key, model, prompt, prompt_sha, batches, manifest, done,
            ckpt, args)
    else:
        new_items, new_usage, failed = run_live(
            client, model, prompt, prompt_sha, batches, manifest, done, ckpt, args,
            desc=run_key)
    python_elapsed = time.time() - started
    all_items.extend(new_items)
    for k in usage:
        usage[k] += new_usage[k]

    if args.submit_only and args.batch_timing == "queue24h":
        print(f"  {run_key}: jobs submitted; no codes written yet.")
        return {"model": model_key, "corpus": corpus, "scope": args.scope,
                "submitted_only": True, "jobs": sorted(jobs)}

    out_cols = ["batch_id", "id", "chunk_id", "doc_id", "code", "label"]
    codes = pd.DataFrame(all_items, columns=out_cols)
    if len(codes):
        codes = (codes
                 .drop_duplicates(subset=["chunk_id"], keep="last")
                 .sort_values(["doc_id", "chunk_id"]))
    out_csv = GPT / f"topics_{run_key}.csv"
    codes.to_csv(out_csv, index=False, encoding="utf-8")

    direct_cost = estimate_cost(usage, model["price"])
    batch_cost = 0.5 * direct_cost
    factor = chunk_info["full_sample_chunks"] / max(chunk_info["selected_chunks"], 1)
    expected = {(m["batch_id"], k) for m in manifest for k in m["keys"]}
    observed = set(zip(codes.get("batch_id", []), codes.get("id", [])))
    n_residual = int(codes.code.isin(topic_codes.RESIDUAL).sum()) if len(codes) else 0
    validation = {"valid_observations": int(len(codes)),
                  "expected_observations": int(len(expected)),
                  "exactly_expected_observations": len(codes) == len(expected),
                  "every_expected_local_id_once": observed == expected and not codes.duplicated(["batch_id", "id"]).any(),
                  "codes_in_codebook": bool(codes.code.isin(CODES).all()),
                  # The label is joined from topic_codes.py, never returned by the
                  # model, so this can only fail if the join itself broke.
                  "labels_match_codes": bool(
                      (codes.label == codes.code.map(topic_codes.LABEL)).all()),
                  "n_classes_used": int(codes.code.nunique()),
                  "n_classes_available": len(CODES)}

    # The appendix reports cost and running time from exactly these fields, so the
    # number here must always be the time the coding actually took.
    #   queue24h -- the provider's own clock, first job opened to last job finished,
    #               so a run collected in a later invocation still reports the work,
    #               not the time spent polling;
    #   live     -- this invocation's wall clock, when the run was never interrupted;
    #   resumed  -- a run killed and restarted was never wholly inside one
    #               invocation, and neither piece is the running time: this
    #               invocation saw only the tail, and the checkpoint span includes
    #               the dead time in between.  Coding time is the span minus that
    #               dead time, which is invariant to how many invocations it took;
    #   neither  -- when an invocation codes nothing new (a pure re-read of the
    #               checkpoint) its own zero seconds would erase the recorded time,
    #               so the previous run's figure is carried forward instead.
    meta_path = GPT / f"meta_{run_key}.json"
    prior_elapsed = None
    if meta_path.exists():
        try:
            prior_elapsed = (json.loads(meta_path.read_text(encoding="utf-8"))
                             .get("timing", {}).get("elapsed_seconds_this_invocation"))
        except Exception:
            prior_elapsed = None
    n_new_batches = len({it["batch_id"] for it in new_items})
    provider_wall = queue_wall_seconds(jobs) if args.batch_timing == "queue24h" else None
    ckpt_span = checkpoint_span(ckpt, prompt_sha)
    idle = checkpoint_idle(ckpt, prompt_sha)
    if provider_wall is not None:
        elapsed = provider_wall
        basis = "provider job clock, first job created to last job completed"
    elif idle and ckpt_span:
        elapsed = ckpt_span - idle
        basis = (f"checkpoint span less {idle:.0f} s of interruption; the run was "
                 f"resumed, so no single invocation covered the whole of it")
    elif n_new_batches or not prior_elapsed:
        elapsed = python_elapsed
        basis = "wall clock of this invocation"
    else:
        elapsed = prior_elapsed
        basis = "carried forward: this invocation coded nothing new"
    # `transport` keeps the meaning it has in 40E: which HTTP client carried the
    # requests.  WHEN they were answered is `batch_timing`.
    transport = "powershell" if client == "powershell" else "httpx"
    meta = {"created_at": now(), "schema_version": SCHEMA_VERSION,
            "model_key": model_key, "model_id": model["id"], "corpus": corpus,
            "scope": args.scope,
            "run_tag": args.run_tag, "seed": args.seed, "batch_size": args.batch_size,
            "workers": args.workers,
            "timing": {"batches_total": len(batches),
                       "batches_this_invocation": n_new_batches,
                       "elapsed_seconds_this_invocation": round(elapsed, 1),
                       "wall_seconds_all_invocations": ckpt_span,
                       "idle_seconds_between_invocations": idle,
                       "python_elapsed_seconds_this_invocation": round(python_elapsed, 1),
                       "transport": transport,
                       "batch_timing": args.batch_timing,
                       "basis": basis},
            "n_chunks": int(len(codes)),
            "n_documents": int(codes.doc_id.nunique()) if len(codes) else 0,
            "chunk_info": chunk_info,
            "usage": usage, "direct_cost_usd": direct_cost,
            "batch24h_cost_usd": batch_cost,
            "billed_cost_usd": batch_cost if args.batch_timing == "queue24h" else direct_cost,
            "billed_cost_basis": ("24h queue, half the standard prices"
                                  if args.batch_timing == "queue24h" else "standard prices"),
            "projected_full_direct_usd": factor * direct_cost,
            "projected_full_batch24h_usd": 0.5 * factor * direct_cost,
            "price_usd_per_1m": model["price"], "price_source": PRICE_SOURCE,
            "prompt_sha256": prompt_sha,
            "prompt_version": args.prompt_version,
            "prompt_file": str(prompt_file),
            "codebook": {"file": str(topic_codes.BOOK),
                         "sha256": topic_codes.BOOK_SHA256,
                         "n_classes": len(CODES),
                         "codes": list(CODES)},
            "class_counts": (codes.code.value_counts().to_dict() if len(codes) else {}),
            "n_residual": n_residual,
            "openai_sdk_version": openai.__version__,
            "validation": validation,
            "files": {"topics": str(out_csv), "checkpoint": str(ckpt),
                      "metadata": str(meta_path)},
            "failed_batches": failed}
    if args.batch_timing == "queue24h":
        meta["queue24h"] = {
            "window": QUEUE_WINDOW, "endpoint": QUEUE_ENDPOINT,
            "requests_per_job": args.requests_per_job,
            "jobs_ledger": str(jobs_path(run_key)),
            "jobs": [{"job_id": j.get("job_id"), "n_requests": j.get("n_requests"),
                      "status": j.get("status"),
                      "created_at_unix": j.get("created_at_unix"),
                      "completed_at_unix": j.get("completed_at_unix"),
                      "seconds": (None if not (j.get("completed_at_unix")
                                               and j.get("created_at_unix"))
                                  else float(j["completed_at_unix"] - j["created_at_unix"])),
                      "n_coded": j.get("n_coded"), "n_failed": j.get("n_failed")}
                     for j in jobs.values()],
            "jobs_still_open": bool(still_open)}
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  {run_key}: {len(codes):,} chunks coded into "
          f"{validation['n_classes_used']} of {len(CODES)} classes "
          f"({n_residual:,} residual), billed ${meta['billed_cost_usd']:.4f} "
          f"(${meta['direct_cost_usd']:.4f} at standard prices) -> {out_csv.name}")
    if failed:
        print(f"  {len(failed)} batch(es) failed; nothing was re-sent. Re-run to "
              f"fill them: {[b['batch_idx'] for b in failed]}")
    if still_open:
        print(f"  some jobs are still running; re-run the same command to collect them")
    return meta


def get_client(dry_run: bool):
    if dry_run:
        return None
    if os.getenv("OPENAI_USE_POWERSHELL_HTTP") == "1":
        return "powershell"
    key = os.getenv("OPENAI_API_KEY")
    if not key and KEY_PATH.exists():
        key = KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"no OpenAI key ($OPENAI_API_KEY or {KEY_PATH})")
    from openai import OpenAI
    # max_retries=0: a failed batch is recorded, never silently re-sent.
    kw: dict = {"max_retries": 0, "timeout": 180.0}
    if CA_BUNDLE.exists():
        import httpx
        kw["http_client"] = httpx.Client(verify=str(CA_BUNDLE), timeout=180.0)
        print(f"  TLS: verifying against {CA_BUNDLE.name} (Windows trust store)")
    return OpenAI(api_key=key, **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--corpora", nargs="*", default=list(CORPORA), choices=list(CORPORA))
    ap.add_argument("--scope", choices=SCOPES, default=DEFAULT_SCOPE,
                    help="which documents to code. sample: the 1,000-document "
                         "per-corpus sample of 40D, shared with ask-and-average "
                         "(default); full: the whole estimation sample")
    ap.add_argument("--batch-timing", choices=["queue24h", "live"], default="queue24h",
                    help="when the batches are answered, not what is in them. "
                         "queue24h: the provider's 24h queue at half price (default); "
                         "live: answered immediately, concurrently")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--requests-per-job", type=int, default=REQUESTS_PER_JOB,
                    help="queue24h only: batches per uploaded job file")
    ap.add_argument("--poll-seconds", type=int, default=POLL_SECONDS,
                    help="queue24h only: seconds between job status reads")
    ap.add_argument("--max-wait-hours", type=float, default=MAX_WAIT_HOURS,
                    help="queue24h only: stop waiting after this long; the jobs "
                         "keep running and a later re-run collects them")
    ap.add_argument("--submit-only", action="store_true",
                    help="queue24h only: open the jobs and exit without waiting")
    ap.add_argument("--resubmit-failed", action="store_true",
                    help="queue24h only: send batches that came back failed. Off "
                         "by default -- nothing is ever re-sent automatically")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help="live only: concurrent requests; the provider's rate "
                         "limit is the ceiling")
    ap.add_argument("--prompt-version", choices=sorted(PROMPT_VERSIONS),
                    default=DEFAULT_PROMPT_VERSION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--run-tag", default="",
                    help="suffix for isolated smoke/test outputs")
    ap.add_argument("--limit-chunks", type=int, default=0,
                    help="code only the first N sentence chunks")
    ap.add_argument("--doc-ids", nargs="*", default=None,
                    help="code only these documents. They must already be in the "
                         "selected scope; this can only narrow a run")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate inputs/schema and estimate cost, no API calls")
    args = ap.parse_args()
    args.run_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_tag.strip())
    args.limit_chunks = args.limit_chunks or None
    if args.batch_size != BATCH_SIZE:
        raise SystemExit(f"batch size is fixed at {BATCH_SIZE} for this project")
    args.workers = max(1, args.workers)
    args.requests_per_job = max(1, args.requests_per_job)

    GPT.mkdir(parents=True, exist_ok=True)
    client = get_client(args.dry_run)
    if args.batch_timing == "queue24h" and client == "powershell":
        raise SystemExit("the 24h queue needs the native client; unset "
                         "OPENAI_USE_POWERSHELL_HTTP or pass --batch-timing live")
    print(f"  codebook: {len(CODES)} classes from {topic_codes.BOOK.name} "
          f"(sha256 {topic_codes.BOOK_SHA256[:12]})")
    print(f"  prompt version: {args.prompt_version}   scope: {args.scope}")
    if not args.dry_run:
        print(f"  batch timing: {args.batch_timing}"
              + (f" ({QUEUE_WINDOW} queue, half price)"
                 if args.batch_timing == "queue24h"
                 else f" (answered now, {args.workers} concurrent requests)"))
    summary = []
    for mk in args.models:
        for c in args.corpora:
            print(f"\n=== {MODELS[mk]['id']} · {c.upper()} · {args.scope} ===")
            summary.append(run_one(client, mk, c, args))
    if args.dry_run:
        tot = sum(m.get("rough_cost_usd", 0) for m in summary)
        print(f"\nRough total across all runs: ${tot:.2f} standard / "
              f"${0.5 * tot:.2f} on the {QUEUE_WINDOW} queue")
    elif not args.submit_only:
        total = sum(m.get("billed_cost_usd", 0) for m in summary)
        print(f"\nTotal billed across all runs: ${total:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
