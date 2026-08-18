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
from typing import Any

import numpy as np
import pandas as pd
import openai
from pydantic import BaseModel, ValidationError

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional
    def tqdm(x, **k):
        return x

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = Path(os.getenv("TOPIC2IRT_ROOT", HERE.parent))
DATA = REPO / "data"
TOPICS = DATA / "topics"
STANCE = DATA / "stance"
OUT = DATA / "topic_labels"
PROMPTS = REPO / "code" / "prompts"
KEY_PATH = REPO.parent / "misc" / "key.txt"      # shared with the other GPT stages
# This machine inspects TLS, so the OpenAI SDK's bundled certifi roots reject the
# connection.  The Windows trust store does hold the inspection root, so it is
# exported once to a PEM bundle and handed to httpx.  Falls back to a PowerShell
# HTTP client (OPENAI_USE_POWERSHELL_HTTP=1), which uses the Windows store
# natively but spawns a process per request.
CA_BUNDLE = Path(os.getenv("OPENAI_CA_BUNDLE")
                 or REPO.parent / "misc" / "win_ca_bundle.pem")

# A namespace of its own: a topic label and a sentence code are different
# payloads, so no checkpoint written by 40E or 40P can ever be read here.
SCHEMA_VERSION = "topic_label_v01"
SEED = 20260812
WORKERS = 4                        # concurrent requests; one topic each

# The sample the model sees.  Ten per stance overfits the particular sentences
# drawn; twenty is where the answer stops following them.  Fifty is the setting,
# for margin on the heterogeneous topics: the whole corpus costs about a dollar
# either way, and the error on which subject dominates falls like 1/sqrt(n) --
# roughly +/-6 points at 60 sentences against +/-3 at 150.  Past this the limit
# stops being the sample and becomes attention over a long flat list.
N_PER_STANCE = 50
STANCES = ("Left", "Neutral", "Right")     # the values 03F writes, as 04 reads them
N_KEY_TERMS = 10                           # leading c-TF-IDF terms shown with the sample

# ------------------------------------------------------- how short is short --- #
# The prompt asks for two or three words; the schema enforces it.  Strict
# structured output constrains decoding to the schema, so a `pattern` on the
# string is a wall the model cannot step over -- `maxLength` is NOT among the
# keywords strict mode accepts and would be rejected with a 400 before anything
# is billed, so the length limit is expressed as a regex like everything else.
#
# The grammar: a word starts with a capital, a digit or "(" (so `2nd Amendment`
# and `Public Servants (Hiring/Career)` pass) and runs to at most TOKEN_MAX
# characters.  A word may be followed by up to three more, each joined by a
# space, by "&", or by a lower-case connector, and a comma may close any word but
# the last.  A lower-case opening, a fifth word or a whole sentence cannot be
# produced at all.  Checked against the author's own 261 labels: every one passes.
#
# WORD_CAP is four, not the three the prompt asks for, and the difference is
# deliberate.  Under a three-word ceiling the first production run could not write
# "Public Lands & National Parks", so constrained decoding satisfied the grammar by
# gluing the words together -- "Public Lands & NationalParks", thirteen times in
# 263 labels.  A grammar that cannot be met honestly is met dishonestly.  So the
# wall stands one word further out than the request, and `check_label` flags the
# fourth word instead of forbidding it.
#
# The label must also END in a word.  Ending on a joiner was reachable before, and
# the run produced "Health Workforce Training &" and "Public Employee Pay &".
#
# The comma is allowed on purpose.  A three-way enumeration ("Medicines,
# Telehealth & Vaccines") is sometimes the honest name of a cluster, and where it
# is instead the model dodging the choice, the enumeration is itself the evidence
# that the topic is mixed -- so it is flagged and read, not forbidden and lost.
MAX_WORDS = 3                     # flagged above this; "&" and connectors are not words
WORD_CAP = 4                      # and refused above this
TOKEN_MAX = 20                    # characters in one word (longest of his own: 17)
CONNECTORS = ("of", "in", "for", "and", "the", "to", "on", "at", "vs")
_W = rf"[A-Z0-9(][A-Za-z0-9'./()-]{{0,{TOKEN_MAX - 1}}}"
_JOINED = rf",? (?:&|{'|'.join(CONNECTORS)}) {_W}"      # " & National", " of Law"
_PLAIN = rf",? {_W}"                                     # " Parks", ", Telehealth"
LABEL_PATTERN = rf"^{_W}(?:{_JOINED}|{_PLAIN}){{0,{WORD_CAP - 1}}}$"
LABEL_RX = re.compile(LABEL_PATTERN)
# Four words of TOKEN_MAX, each of the three joins costing a comma, a space, the
# longest connector and another space.
HARD_CEILING_CHARS = WORD_CAP * TOKEN_MAX + (WORD_CAP - 1) * 6
# The implied hard ceiling is 4 x TOKEN_MAX + 3 = 83 characters.  The author's
# longest label is 31, so LABEL_MAX_CHARS is that plus slack: a label above it is
# legal but flagged for him to look at.
LABEL_MAX_CHARS = 46
# A second wall, in tokens rather than characters: a three-word label is about a
# dozen output tokens, so a truncated answer here means something went wrong.  It
# raises (finish_reason=length) rather than being written down short.
MAX_OUTPUT_TOKENS = 64

# Mirrors the banned list in the prompt files.  A label carrying one of these is
# flagged in the audit workbook, never silently rewritten and never re-asked.
FILLER = frozenset({"miscellaneous", "misc", "various", "general", "generic",
                    "other", "others", "assorted", "mixed", "issues", "topics",
                    "boilerplate", "related", "stuff", "things"})

PROMPT_VERSIONS = {"v01": "topic_label_{c}_v01.txt"}
DEDUP_PROMPT_VERSIONS = {"v01": "topic_label_dedup_{c}_v01.txt"}
DEFAULT_PROMPT_VERSION = "v01"

CORPORA = {
    "us": dict(doc_col="doc_id"),
    "br": dict(doc_col="platform_id"),     # the document id 04 scales BR on
}
for _c, _cfg in CORPORA.items():
    _cfg["chunks"] = TOPICS / f"topic_model_{_c}" / "chunk_topics.feather"
    _cfg["terms"] = TOPICS / f"topic_model_{_c}" / "topic_info_before_after.xlsx"
    _cfg["stance"] = STANCE / f"stance_pred_{_c}.csv"
    _cfg["prompts"] = {v: PROMPTS / f.format(c=_c) for v, f in PROMPT_VERSIONS.items()}
    _cfg["dedup_prompts"] = {v: PROMPTS / f.format(c=_c)
                             for v, f in DEDUP_PROMPT_VERSIONS.items()}

# The model the author asked for.  Reasoning models reject temperature != 1
# unless the reasoning effort is "none", which is also what a cheap, near
# deterministic naming task wants.  Prices are USD per 1M tokens.
MODELS = {
    "gpt54mini": dict(id="gpt-5.4-mini", supports_temperature=True, supports_seed=True,
                      reasoning_effort="none",
                      price=dict(input=0.75, cached_input=0.075, output=4.50)),
}
PRICE_SOURCE = {"observed_date": "2026-07-22",
                "urls": ["https://developers.openai.com/api/docs/pricing"],
                "note": "USD per 1M tokens. Verify before quoting."}


# ------------------------------------------------------------------ schema --- #
# pydantic 1.10.8 under anaconda, 2.x under the venv, and this stage has to run
# the same either way.
PYDANTIC_V2 = hasattr(BaseModel, "model_validate")


if PYDANTIC_V2:
    from pydantic import ConfigDict

    class TopicLabel(BaseModel):
        model_config = ConfigDict(extra="forbid")
        label: str
else:
    class TopicLabel(BaseModel):
        label: str

        class Config:
            extra = "forbid"


def _json_schema(model) -> dict[str, Any]:
    return model.model_json_schema() if PYDANTIC_V2 else model.schema()


def _parse_obj(model, obj):
    return model.model_validate(obj) if PYDANTIC_V2 else model.parse_obj(obj)


def _to_dict(obj):
    return obj.model_dump() if PYDANTIC_V2 else obj.dict()


def verify_schema(use_pattern: bool = True) -> dict[str, Any]:
    """The strict schema, checked before every call: one short string, nothing else.

    pydantic writes the string out with no constraint on it, so the length grammar
    is injected here and asserted, exactly as `40P` hoists its enum: the schema
    that is sent is the schema that was checked.
    """
    schema = _json_schema(TopicLabel)
    props = schema.get("properties", {})
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise RuntimeError("schema must be object with additionalProperties=false")
    if set(props) != {"label"} or set(schema.get("required", [])) != {"label"}:
        raise RuntimeError("schema must hold exactly the required key 'label'")
    if props["label"].get("type") != "string":
        raise RuntimeError("label must be a string")
    if not use_pattern:
        return schema
    schema = {**schema,
              "properties": {"label": {**props["label"], "pattern": LABEL_PATTERN}}}
    if schema["properties"]["label"]["pattern"] != LABEL_PATTERN:
        raise RuntimeError("the length grammar did not reach the schema")
    return schema


def response_format(use_pattern: bool = True) -> dict[str, Any]:
    return {"type": "json_schema",
            "json_schema": {"name": "topic_label", "strict": True,
                            "schema": verify_schema(use_pattern)}}


def dedup_key(i: int) -> str:
    """The id a topic carries inside a collision request: r001, r002, ...

    Positional, as in `40E` and `40P`, and never the topic id: the model is given
    no identifier of its own that it could read a name off.
    """
    return f"r{i + 1:03d}"


def verify_dedup_schema(keys, use_pattern: bool = True) -> dict[str, Any]:
    """One required string per topic of the group, and nothing else.

    Built here rather than from a pydantic model because the key set changes with
    the group; the same assertions run on it that `verify_schema` runs on the
    single-label schema, so what is sent is what was checked.
    """
    keys = list(keys)
    prop = {"type": "string"}
    if use_pattern:
        prop["pattern"] = LABEL_PATTERN
    schema = {"type": "object", "additionalProperties": False,
              "properties": {k: dict(prop) for k in keys},
              "required": keys}
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate key in the collision schema")
    if set(schema["properties"]) != set(schema["required"]):
        raise RuntimeError("every key of the collision schema must be required")
    if use_pattern and any(v.get("pattern") != LABEL_PATTERN
                           for v in schema["properties"].values()):
        raise RuntimeError("the length grammar did not reach the collision schema")
    return schema


def dedup_response_format(keys, use_pattern: bool = True) -> dict[str, Any]:
    return {"type": "json_schema",
            "json_schema": {"name": "topic_labels_distinct", "strict": True,
                            "schema": verify_dedup_schema(keys, use_pattern)}}


# ------------------------------------------------------------------- inputs --- #
def load_key_terms(corpus: str) -> dict[int, str]:
    """topic id -> its leading key terms, from the topic model's own workbook.

    The same file 04 reads for the key terms the paper prints, so the label and
    those key terms always describe the same topic.
    """
    d = pd.read_excel(CORPORA[corpus]["terms"])

    def terms(row) -> str:
        kw = str(row.get("keywords_after", "") or "")
        if not kw or kw.lower() == "nan":
            kw = str(row.get("keywords_before", "") or "")
        if not kw or kw.lower() == "nan":
            return ""
        return ", ".join(kw.split(", ")[:N_KEY_TERMS])

    return {int(r.topic): terms(r) for _, r in d.iterrows()}


def load_chunks(corpus: str) -> pd.DataFrame:
    """Every assigned sentence-chunk of the corpus with its topic and its stance.

    The topic comes from the topic model's own assignment, the stance from the
    03F classifier, joined on `chunk_id` -- the same join 04 makes to build the
    matrices, so the topics named here are the columns that are scaled.
    """
    cfg = CORPORA[corpus]
    for path in (cfg["chunks"], cfg["stance"]):
        if not path.exists():
            raise SystemExit(f"[{corpus}] missing input: {path}")
    doc_col = cfg["doc_col"]
    ct = pd.read_feather(cfg["chunks"],
                         columns=["chunk_id", doc_col, "topic", "chunk_text"])
    ct = ct.rename(columns={doc_col: "doc_id"})
    sp = pd.read_csv(cfg["stance"], usecols=["chunk_id", "stance"])
    df = ct.merge(sp, on="chunk_id", how="inner")
    df = df[df.topic >= 0].copy()
    df["chunk_id"] = df.chunk_id.astype(str)
    df["doc_id"] = df.doc_id.astype(str)
    df["topic"] = df.topic.astype(int)
    df["stance"] = df.stance.astype(str)
    unknown = sorted(set(df.stance) - set(STANCES))
    if unknown:
        raise SystemExit(f"[{corpus}] stance values outside {list(STANCES)}: {unknown}")
    # A stable order, so the draw depends on the data and not on how the feather
    # happened to be written.
    df = df.sort_values("chunk_id", kind="stable").reset_index(drop=True)
    print(f"[{corpus}] {len(df):,} sentence-chunks in {df.topic.nunique()} topics, "
          f"{df.doc_id.nunique():,} documents")
    return df


def topic_seed(corpus: str, topic: int) -> int:
    """A per-topic seed that does not move when another topic changes."""
    h = hashlib.sha256(f"{SEED}:{corpus}:{topic}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def build_samples(corpus: str, df: pd.DataFrame, per_stance: int) -> list[dict]:
    """For each topic, up to `per_stance` sentences per stance, reshuffled together."""
    groups = df.groupby(["topic", "stance"]).indices     # positional, cheap
    samples = []
    for topic in sorted(df.topic.unique()):
        rng = np.random.default_rng(topic_seed(corpus, int(topic)))
        picked, per = [], {}
        for stance in STANCES:
            pos = groups.get((topic, stance))
            pos = np.sort(pos) if pos is not None else np.array([], dtype=int)
            take = min(per_stance, len(pos))
            per[stance] = int(take)
            if take:
                picked.append(rng.choice(pos, size=take, replace=False))
        if not picked:
            continue
        pos = np.concatenate(picked)
        rng.shuffle(pos)                                  # the three stances interleave
        items = df.iloc[pos]
        chunk_ids = [str(c) for c in items.chunk_id]
        samples.append({
            "topic": int(topic),
            "n_chunks": int((df.topic == topic).sum()),
            "n_docs": int(df.loc[df.topic == topic, "doc_id"].nunique()),
            "n_sent": int(len(items)),
            "per_stance": per,
            "sample_sha256": hashlib.sha256("\n".join(chunk_ids).encode("utf-8")).hexdigest(),
            "items": [{"chunk_id": c, "doc_id": str(d), "stance": str(s),
                       "text": "" if pd.isna(t) else str(t)}
                      for c, d, s, t in zip(chunk_ids, items.doc_id, items.stance,
                                            items.chunk_text)],
        })
    return samples


def build_prompt(corpus: str, version: str, which: str = "prompts") -> tuple[str, Path]:
    path = CORPORA[corpus][which][version]
    return path.read_text(encoding="utf-8").strip(), path


def format_topic(sample: dict, key_terms: str) -> str:
    """The user message: the key terms and the sentences, and nothing else.

    The topic id, the chunk ids and the document ids stay local; the model sees
    no identifier it could name the topic after.
    """
    return json.dumps({"key_terms": key_terms,
                       "sentences": [it["text"] for it in sample["items"]]},
                      ensure_ascii=False)


def format_group(group: dict, key_terms: dict[int, str]) -> str:
    """The user message of a collision request: the whole set, side by side.

    Each member carries the positional id the schema requires back and exactly
    what the first pass saw for it, so the rename is read off the same evidence
    as the name it replaces.
    """
    return json.dumps(
        {"shared_label": group["label"],
         "topics": [{"id": dedup_key(i),
                     "key_terms": key_terms.get(s["topic"], ""),
                     "sentences": [it["text"] for it in s["items"]]}
                    for i, s in enumerate(group["samples"])]},
        ensure_ascii=False)


def collision_groups(labels_by_topic: dict[int, str],
                     samples_by_topic: dict[int, dict]) -> list[dict]:
    """The sets of topics that came out of the first pass sharing a name.

    Matched case- and space-insensitively, so "Public Safety" and "public  safety"
    are one collision.  Each group carries its members in topic order, which makes
    the request, and therefore the group hash, independent of dict ordering.
    """
    by_key: dict[str, list[int]] = {}
    for topic, label in labels_by_topic.items():
        by_key.setdefault(" ".join(str(label).split()).casefold(), []).append(topic)
    groups = []
    for key, topics in sorted(by_key.items()):
        if len(topics) < 2:
            continue
        topics = sorted(topics)
        members = [samples_by_topic[t] for t in topics]
        stamp = "|".join(f"{t}:{samples_by_topic[t]['sample_sha256']}" for t in topics)
        groups.append({
            "label": labels_by_topic[topics[0]],
            "topics": topics,
            "samples": members,
            "group_sha256": hashlib.sha256(
                f"{key}\n{stamp}".encode("utf-8")).hexdigest(),
        })
    return groups


# ---------------------------------------------------------------- the label --- #
def check_label(raw: str) -> tuple[str, list[str]]:
    """Clean the answer and flag it.  Never rewrites it, never re-asks.

    The schema already makes most of this unreachable; it is checked anyway,
    because a run with `--no-pattern` has no wall, and because a flag the author
    can read is worth more than a guarantee he has to take on trust.
    """
    label = " ".join(str(raw or "").split()).strip(' "\'')
    flags = []
    words = [w for w in label.split()
             if w != "&" and w.lower() not in CONNECTORS]
    if not label:
        flags.append("empty")
    if len(words) > MAX_WORDS:
        flags.append(f"{len(words)}_words")
    if len(label) > LABEL_MAX_CHARS:
        flags.append(f"{len(label)}_chars")
    if any(re.sub(r"[^A-Za-z]", "", w).lower() in FILLER for w in words):
        flags.append("filler")
    # Not an error: a list of three things can be the right name.  But it is also
    # what a model writes when it will not choose, so the author sees which it is.
    if "," in label:
        flags.append("enumeration")
    if not LABEL_RX.match(label):
        flags.append("off_pattern")
    return label, flags


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


def request_body(model: dict, prompt: str, sample: dict, key_terms: str,
                 seed: int, use_pattern: bool = True) -> dict:
    return dict(model=model["id"],
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": format_topic(sample, key_terms)}],
                response_format=response_format(use_pattern),
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                **call_kwargs(model, seed))


def dedup_request_body(model: dict, prompt: str, group: dict,
                       key_terms: dict[int, str], seed: int,
                       use_pattern: bool = True) -> dict:
    keys = [dedup_key(i) for i in range(len(group["topics"]))]
    return dict(model=model["id"],
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": format_group(group, key_terms)}],
                response_format=dedup_response_format(keys, use_pattern),
                # One label's worth of tokens per member, plus the braces and keys.
                max_completion_tokens=32 + MAX_OUTPUT_TOKENS * len(keys),
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


def proxy_usage(samples: list[dict], prompt: str, key_terms: dict[int, str]) -> dict:
    """A cost proxy measured on the sentences actually drawn, not a constant.

    The prompt is re-sent with every topic, so it is counted once per topic here
    rather than assumed away.  Four characters to the token is the usual rough
    rule for both languages; caching is NOT modelled, so a real run comes in
    under this.
    """
    chars = sum(len(format_topic(s, key_terms.get(s["topic"], ""))) for s in samples)
    return {"prompt_tokens": int(len(samples) * len(prompt) / 4 + chars / 4),
            "completion_tokens": int(len(samples) * 12),
            "cached_tokens": 0}


_ps_seq = itertools.count()


def ps_chat_completion(request: dict) -> dict:
    tmp = OUT / "_tmp_openai"
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
    """One accessor for both the SDK objects and plain dicts."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def parse_completion(completion) -> dict:
    choice = (_field(completion, "choices") or [None])[0]
    if choice is None:
        raise RuntimeError("no choices in response")
    if _field(choice, "finish_reason") in ("length", "content_filter"):
        raise RuntimeError(f"finish_reason={_field(choice, 'finish_reason')}")
    msg = _field(choice, "message")
    refusal = _field(msg, "refusal")
    if refusal:
        raise RuntimeError(f"refusal: {refusal}")
    try:
        parsed = _to_dict(_parse_obj(TopicLabel, json.loads(_field(msg, "content"))))
    except (TypeError, json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(f"invalid structured output: {exc}") from exc
    label, flags = check_label(parsed["label"])
    return {"label": label, "raw_label": parsed["label"], "flags": flags,
            "usage": usage_to_dict(_field(completion, "usage")),
            "returned_model": _field(completion, "model"),
            "fingerprint": _field(completion, "system_fingerprint")}


def call_api(client, model: dict, prompt: str, sample: dict, key_terms: str,
             seed: int, use_pattern: bool = True) -> dict:
    request = request_body(model, prompt, sample, key_terms, seed, use_pattern)
    t0 = time.perf_counter()
    completion = (ps_chat_completion(request) if client == "powershell"
                  else client.chat.completions.create(**request))
    seconds = time.perf_counter() - t0
    return {**parse_completion(completion), "seconds": round(seconds, 3)}


def parse_dedup_completion(completion, group: dict) -> dict:
    """One label per member, keyed positionally, mapped back to the topic ids."""
    choice = (_field(completion, "choices") or [None])[0]
    if choice is None:
        raise RuntimeError("no choices in response")
    if _field(choice, "finish_reason") in ("length", "content_filter"):
        raise RuntimeError(f"finish_reason={_field(choice, 'finish_reason')}")
    msg = _field(choice, "message")
    refusal = _field(msg, "refusal")
    if refusal:
        raise RuntimeError(f"refusal: {refusal}")
    try:
        parsed = json.loads(_field(msg, "content"))
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid structured output: {exc}") from exc
    keys = [dedup_key(i) for i in range(len(group["topics"]))]
    if not isinstance(parsed, dict) or set(parsed) != set(keys):
        raise RuntimeError(f"expected exactly the keys {keys}, got {sorted(parsed)}")
    labels, raw, flags = {}, {}, {}
    for i, topic in enumerate(group["topics"]):
        label, fl = check_label(parsed[keys[i]])
        labels[topic], raw[topic], flags[topic] = label, parsed[keys[i]], fl
    return {"labels": labels, "raw_labels": raw, "flags": flags,
            "usage": usage_to_dict(_field(completion, "usage")),
            "returned_model": _field(completion, "model"),
            "fingerprint": _field(completion, "system_fingerprint")}


def call_dedup_api(client, model: dict, prompt: str, group: dict,
                   key_terms: dict[int, str], seed: int,
                   use_pattern: bool = True) -> dict:
    request = dedup_request_body(model, prompt, group, key_terms, seed, use_pattern)
    t0 = time.perf_counter()
    completion = (ps_chat_completion(request) if client == "powershell"
                  else client.chat.completions.create(**request))
    seconds = time.perf_counter() - t0
    return {**parse_dedup_completion(completion, group),
            "seconds": round(seconds, 3)}


# -------------------------------------------------------------- checkpoints --- #
def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_checkpoint(path: Path, prompt_sha: str, sample_sha: dict[int, str]):
    """Records this run may reuse.

    A topic id alone cannot say whether a stored label was read off the sentences
    that topic holds now: refit the topic model, or change the draw, and topic 7
    is a different set of sentences.  Every record therefore carries the hash of
    the sample that produced it, and one that does not match is re-asked rather
    than reused.
    """
    done, items, usage = {}, [], zero_usage()
    stale_prompt = stale_sample = 0
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
                # An interrupted run can leave one truncated trailing record; that
                # topic simply counts as not done and is asked again on resume.
                print(f"  skipping unreadable checkpoint line {ln} in {path.name}")
                continue
            if rec.get("schema_version") != SCHEMA_VERSION:
                stale_prompt += 1
                continue
            if rec.get("prompt_sha256") not in (None, prompt_sha):
                stale_prompt += 1
                continue
            topic = int(rec["topic"])
            if rec.get("sample_sha256") != sample_sha.get(topic):
                stale_sample += 1
                continue
            done[topic] = rec
            items.append(rec)
            for k in usage:
                usage[k] += int((rec.get("usage") or {}).get(k) or 0)
    if stale_prompt:
        print(f"  ignoring {stale_prompt:,} record(s) in {path.name} written under an "
              f"earlier schema version or prompt; those topics are asked again")
    if stale_sample:
        print(f"  ignoring {stale_sample:,} record(s) in {path.name} whose sentences "
              f"differ from the sample drawn now; asked again")
    return done, items, usage


def save_checkpoint(path: Path, sample: dict, result: dict, prompt_sha: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"schema_version": SCHEMA_VERSION,
                            "topic": sample["topic"],
                            "prompt_sha256": prompt_sha,
                            "sample_sha256": sample["sample_sha256"],
                            "label": result["label"], "raw_label": result["raw_label"],
                            "flags": result["flags"], "usage": result["usage"],
                            "seconds": result["seconds"],
                            "returned_model": result["returned_model"],
                            "fingerprint": result["fingerprint"],
                            "created_at": now()}, ensure_ascii=False) + "\n")


def load_dedup_checkpoint(path: Path, prompt_sha: str, group_sha: dict[str, str]):
    """Collision groups this run may reuse.

    Guarded the way the naming checkpoint is, one level up: the record carries the
    hash of the whole group, its members and their samples together, so a group
    that gained or lost a member, or whose members were renamed, is asked again
    rather than reused.
    """
    done, usage = {}, zero_usage()
    stale = 0
    if not path.exists():
        return done, usage
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"  skipping unreadable checkpoint line {ln} in {path.name}")
                continue
            if (rec.get("schema_version") != SCHEMA_VERSION
                    or rec.get("prompt_sha256") not in (None, prompt_sha)
                    or rec.get("group_sha256") not in group_sha):
                stale += 1
                continue
            done[rec["group_sha256"]] = rec
            for k in usage:
                usage[k] += int((rec.get("usage") or {}).get(k) or 0)
    if stale:
        print(f"  ignoring {stale:,} collision record(s) in {path.name} written under "
              f"an earlier prompt or a different group; those groups are asked again")
    return done, usage


def save_dedup_checkpoint(path: Path, group: dict, result: dict, prompt_sha: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"schema_version": SCHEMA_VERSION,
                            "group_sha256": group["group_sha256"],
                            "prompt_sha256": prompt_sha,
                            "shared_label": group["label"],
                            "topics": group["topics"],
                            "labels": {str(t): result["labels"][t]
                                       for t in group["topics"]},
                            "raw_labels": {str(t): result["raw_labels"][t]
                                           for t in group["topics"]},
                            "flags": {str(t): result["flags"][t]
                                      for t in group["topics"]},
                            "usage": result["usage"], "seconds": result["seconds"],
                            "returned_model": result["returned_model"],
                            "fingerprint": result["fingerprint"],
                            "created_at": now()}, ensure_ascii=False) + "\n")


def checkpoint_stamps(path: Path, prompt_sha: str) -> list[datetime]:
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


# ------------------------------------------------------------------- output --- #
def audit_workbook(path: Path, labels: pd.DataFrame, sentences: pd.DataFrame):
    """The human check: the labels on one sheet, the sentences sent on the other.

    Both sheets carry the topic id, so a doubtful label is read against the exact
    sentences that produced it.  `Approved Label` and `Note` are left empty for
    the reader.
    """
    from openpyxl.styles import Alignment

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        labels.to_excel(xl, sheet_name="labels", index=False)
        sentences.to_excel(xl, sheet_name="sentences", index=False)
        widths = {"labels": {"Corpus": 8, "Topic": 7, "Label": 30,
                             "First Pass Label": 26, "Words": 7,
                             "Chars": 7, "Flags": 22, "N Chunks": 11, "N Docs": 9,
                             "N Sent": 8, "N Left": 8, "N Neutral": 11, "N Right": 9,
                             "Key Terms": 70, "Seconds": 9, "Prompt Tokens": 15,
                             "Completion Tokens": 18, "Cost USD": 11,
                             "Approved Label": 26, "Note": 30},
                  "sentences": {"Corpus": 8, "Topic": 7, "Label": 26, "Order": 7,
                                "Stance": 10, "Chunk Id": 26, "Doc Id": 30,
                                "Sentence": 110}}
        wrap = {"labels": ("Key Terms",), "sentences": ("Sentence",)}
        for sheet, frame in (("labels", labels), ("sentences", sentences)):
            ws = xl.sheets[sheet]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for i, col in enumerate(frame.columns, start=1):
                letter = ws.cell(row=1, column=i).column_letter
                ws.column_dimensions[letter].width = widths[sheet].get(col, 14)
                if col in wrap[sheet]:
                    for row in range(2, len(frame) + 2):
                        ws.cell(row=row, column=i).alignment = Alignment(
                            wrap_text=True, vertical="top")


def run_one(client, model_key: str, corpus: str, args) -> dict:
    model = MODELS[model_key]
    prompt, prompt_file = build_prompt(corpus, args.prompt_version)
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    dedup_prompt, dedup_file = build_prompt(corpus, args.prompt_version, "dedup_prompts")
    dedup_sha = hashlib.sha256(dedup_prompt.encode("utf-8")).hexdigest()

    t0 = time.time()
    df = load_chunks(corpus)
    key_terms = load_key_terms(corpus)
    t_load = time.time() - t0

    t0 = time.time()
    samples = build_samples(corpus, df, args.per_stance)
    if args.topics:
        keep = {int(t) for t in args.topics}
        missing = sorted(keep - {s["topic"] for s in samples})
        if missing:
            raise SystemExit(f"[{corpus}] no sentences for topic(s) {missing}")
        samples = [s for s in samples if s["topic"] in keep]
    if args.limit_topics:
        samples = samples[:args.limit_topics]
    t_sample = time.time() - t0
    thin = [s for s in samples if min(s["per_stance"].values()) < args.per_stance]
    print(f"[{corpus}] {len(samples)} topics to name, "
          f"{sum(s['n_sent'] for s in samples):,} sentences sampled"
          + (f"; {len(thin)} topic(s) hold fewer than {args.per_stance} sentences in "
             f"some stance: {[s['topic'] for s in thin]}" if thin else ""))

    run_suffix = f"_{args.run_tag}" if args.run_tag else ""
    run_key = f"{model_key}_{corpus}{run_suffix}"
    ckpt = OUT / f"checkpoint_{run_key}.jsonl"
    sample_sha = {s["topic"]: s["sample_sha256"] for s in samples}
    done, records, usage = load_checkpoint(ckpt, prompt_sha, sample_sha)
    if done:
        print(f"  resuming {run_key}: {len(done)}/{len(samples)} topics already named")

    if args.dry_run or client is None:
        verify_schema(not args.no_pattern)
        verify_dedup_schema([dedup_key(i) for i in range(2)], not args.no_pattern)
        proxy = proxy_usage(samples, prompt, key_terms)
        est = estimate_cost(proxy, model["price"])
        print(f"  DRY RUN {run_key}: {len(samples)} requests, one per topic, "
              f"rough cost estimate ${est:.2f} (measured sentences, "
              f"{len(prompt) / 4:.0f}-token prompt per topic, caching not modelled)")
        print("  the collision pass is sized by the names the first pass returns, "
              "so it cannot be counted here; it is one request per group of topics "
              "that share a name")
        return {"model": model_key, "corpus": corpus, "dry_run": True,
                "n_topics": len(samples), "prompt_sha256": prompt_sha,
                "dedup_prompt_sha256": dedup_sha,
                "rough_cost_usd": round(est, 4)}

    # One topic per request, answered live.  Requests run concurrently -- the wall
    # clock is API latency, not local work -- but only this thread appends to the
    # checkpoint, so the file stays one record per line and needs no lock.  A topic
    # that raises is recorded and skipped; it is never re-sent automatically, and a
    # later re-run picks it up through the ordinary resume path.
    t0 = time.time()
    failed, new = [], []
    pending = [s for s in samples if s["topic"] not in done]

    def work(sample: dict):
        return sample, call_api(client, model, prompt, sample,
                                key_terms.get(sample["topic"], ""), args.seed,
                                not args.no_pattern)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, s): s for s in pending}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=run_key, unit="topic"):
            s = futures[fut]
            try:
                s, result = fut.result()
            except Exception as exc:
                print(f"  FAILED topic {s['topic']}: {exc}")
                failed.append({"topic": s["topic"], "error": str(exc)})
                continue
            save_checkpoint(ckpt, s, result, prompt_sha)
            rec = {"topic": s["topic"], "label": result["label"],
                   "raw_label": result["raw_label"], "flags": result["flags"],
                   "usage": result["usage"], "seconds": result["seconds"]}
            done[s["topic"]] = rec
            records.append(rec)
            new.append(rec)
            for k in usage:
                usage[k] += result["usage"][k]
    t_label = time.time() - t0

    # -------------------------------------------------------- collisions --- #
    # The topics the first pass gave the same name go back together, one request
    # per group, and are renamed against one another.  This is the only place the
    # answers are allowed to depend on each other, and it is confined to the sets
    # where that dependence is the point.  A group that raises keeps its
    # first-pass names and is picked up by a later re-run through the resume path.
    t0 = time.time()
    by_topic = {int(r["topic"]): r for r in records}
    first_pass = {t: r["label"] for t, r in by_topic.items()}
    resolved, renamed, dedup_flags = dict(first_pass), {}, {}
    dedup_usage, dedup_failed = zero_usage(), []
    dedup_seconds = []
    groups = ([] if args.no_dedup
              else collision_groups(first_pass, {s["topic"]: s for s in samples}))
    if groups:
        dckpt = OUT / f"checkpoint_dedup_{run_key}.jsonl"
        gsha = {g["group_sha256"]: g["label"] for g in groups}
        dedup_done, dedup_usage = load_dedup_checkpoint(dckpt, dedup_sha, gsha)
        print(f"  {len(groups)} collision group(s) covering "
              f"{sum(len(g['topics']) for g in groups)} topics: "
              + "; ".join(f"{g['label']} x{len(g['topics'])}" for g in groups))

        def rename(group: dict):
            return group, call_dedup_api(client, model, dedup_prompt, group,
                                         key_terms, args.seed, not args.no_pattern)

        pending_groups = [g for g in groups if g["group_sha256"] not in dedup_done]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(rename, g): g for g in pending_groups}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{run_key} collisions", unit="group"):
                g = futures[fut]
                try:
                    g, result = fut.result()
                except Exception as exc:
                    print(f"  FAILED collision group '{g['label']}' "
                          f"(topics {g['topics']}): {exc}")
                    dedup_failed.append({"label": g["label"], "topics": g["topics"],
                                         "error": str(exc)})
                    continue
                save_dedup_checkpoint(dckpt, g, result, dedup_sha)
                dedup_done[g["group_sha256"]] = {
                    "labels": {str(t): result["labels"][t] for t in g["topics"]},
                    "flags": {str(t): result["flags"][t] for t in g["topics"]},
                    "seconds": result["seconds"]}
                for k in dedup_usage:
                    dedup_usage[k] += result["usage"][k]

        for g in groups:                      # apply cached and fresh alike
            rec = dedup_done.get(g["group_sha256"])
            if not rec:
                continue
            dedup_seconds.append(rec.get("seconds"))
            for t in g["topics"]:
                new_label = rec["labels"][str(t)]
                resolved[t] = new_label
                dedup_flags[t] = list(rec["flags"].get(str(t), []))
                if new_label != first_pass[t]:
                    renamed[t] = new_label
        print(f"  renamed {len(renamed)} of {sum(len(g['topics']) for g in groups)} "
              f"topics in the collision pass")
    t_dedup = time.time() - t0

    # A name that still collides after the pass is flagged, never asked a third time.
    counts: dict[str, int] = {}
    for lab in resolved.values():
        counts[" ".join(str(lab).split()).casefold()] = (
            counts.get(" ".join(str(lab).split()).casefold(), 0) + 1)
    still_duplicate = {t for t, lab in resolved.items()
                       if counts[" ".join(str(lab).split()).casefold()] > 1}

    # ------------------------------------------------------------- write out --- #
    t0 = time.time()
    named = [s for s in samples if s["topic"] in by_topic]

    def row(s: dict) -> dict:
        """One topic: the label, what it was read off, and what it cost."""
        rec = by_topic[s["topic"]]
        u = rec.get("usage") or zero_usage()
        label = resolved[s["topic"]]
        flags = list(dedup_flags.get(s["topic"], rec["flags"]))
        if s["topic"] in renamed:
            flags.append("renamed")
        if s["topic"] in still_duplicate:
            flags.append("duplicate")
        return {
            "Corpus": corpus.upper(), "Topic": s["topic"], "Label": label,
            "First Pass Label": ("" if s["topic"] not in renamed
                                 else first_pass[s["topic"]]),
            "Words": len([w for w in label.split()
                          if w != "&" and w.lower() not in CONNECTORS]),
            "Chars": len(label), "Flags": ", ".join(flags),
            "N Chunks": s["n_chunks"], "N Docs": s["n_docs"], "N Sent": s["n_sent"],
            "N Left": s["per_stance"]["Left"], "N Neutral": s["per_stance"]["Neutral"],
            "N Right": s["per_stance"]["Right"],
            "Key Terms": key_terms.get(s["topic"], ""),
            "Seconds": rec.get("seconds"),
            "Prompt Tokens": u["prompt_tokens"],
            "Completion Tokens": u["completion_tokens"],
            "Cost USD": round(estimate_cost(u, model["price"]), 6),
            "Approved Label": "", "Note": "",
        }

    labels = pd.DataFrame([row(s) for s in named])
    sentences = pd.DataFrame([{
        "Corpus": corpus.upper(), "Topic": s["topic"],
        "Label": resolved[s["topic"]], "Order": i + 1,
        "Stance": it["stance"], "Chunk Id": it["chunk_id"], "Doc Id": it["doc_id"],
        "Sentence": it["text"],
    } for s in named for i, it in enumerate(s["items"])])

    out_csv = OUT / f"topic_labels_{run_key}.csv"
    labels.drop(columns=["Approved Label", "Note"]).to_csv(
        out_csv, index=False, encoding="utf-8")
    audit_xlsx = OUT / f"audit_{run_key}.xlsx"
    audit_workbook(audit_xlsx, labels, sentences)
    t_audit = time.time() - t0

    flagged = labels[labels.Flags != ""] if len(labels) else labels
    validation = {"topics_named": int(len(labels)),
                  "topics_expected": len(samples),
                  "every_topic_named": len(labels) == len(samples),
                  "labels_unique": bool(labels.Label.nunique() == len(labels))
                                   if len(labels) else True,
                  "labels_unique_before_collision_pass": bool(
                      len(set(first_pass.values())) == len(first_pass)),
                  "collision_groups": len(groups),
                  "topics_in_collision_groups": sum(len(g["topics"]) for g in groups),
                  "topics_renamed": len(renamed),
                  "collisions_remaining": len(still_duplicate),
                  "n_flagged": int(len(flagged)),
                  "flags": (flagged.Flags.value_counts().to_dict()
                            if len(flagged) else {}),
                  "words_max": int(labels.Words.max()) if len(labels) else 0,
                  "chars_max": int(labels.Chars.max()) if len(labels) else 0,
                  "sentences_sent": int(len(sentences))}

    per_topic = labels.Seconds.dropna() if len(labels) else pd.Series(dtype=float)
    stamps = checkpoint_stamps(ckpt, prompt_sha)
    meta_path = OUT / f"meta_{run_key}.json"
    meta = {"created_at": now(), "schema_version": SCHEMA_VERSION,
            "model_key": model_key, "model_id": model["id"], "corpus": corpus,
            "run_tag": args.run_tag, "seed": args.seed,
            "per_stance": args.per_stance, "stances": list(STANCES),
            "n_key_terms": N_KEY_TERMS, "workers": args.workers,
            "timing": {"load_seconds": round(t_load, 1),
                       "sample_seconds": round(t_sample, 1),
                       "label_seconds": round(t_label, 1),
                       "dedup_seconds": round(t_dedup, 1),
                       "audit_seconds": round(t_audit, 1),
                       "topics_this_invocation": len(new),
                       "per_topic_seconds": (
                           {"mean": round(float(per_topic.mean()), 2),
                            "median": round(float(per_topic.median()), 2),
                            "max": round(float(per_topic.max()), 2)}
                           if len(per_topic) else None),
                       "wall_seconds_all_invocations": (
                           round((stamps[-1] - stamps[0]).total_seconds(), 1)
                           if len(stamps) > 1 else None)},
            # What the model was allowed to answer, recorded with the answers.
            "output_constraints": {
                "max_words": MAX_WORDS, "word_cap": WORD_CAP,
                "token_max_chars": TOKEN_MAX,
                "label_pattern": None if args.no_pattern else LABEL_PATTERN,
                "pattern_hard_ceiling_chars": None if args.no_pattern
                                              else HARD_CEILING_CHARS,
                "flag_above_chars": LABEL_MAX_CHARS,
                "max_completion_tokens": MAX_OUTPUT_TOKENS},
            "n_topics": len(labels), "n_sentences": int(len(sentences)),
            "usage": usage, "cost_usd": estimate_cost(usage, model["price"])
                                        + estimate_cost(dedup_usage, model["price"]),
            "cost_usd_naming": estimate_cost(usage, model["price"]),
            "cost_usd_per_topic": (round(estimate_cost(usage, model["price"])
                                         / len(labels), 6) if len(labels) else None),
            # The collision pass, recorded apart: it is a different request shape,
            # a different prompt, and the only one that sees more than one topic.
            "dedup": {"enabled": not args.no_dedup,
                      "prompt_sha256": dedup_sha, "prompt_file": str(dedup_file),
                      "n_groups": len(groups),
                      "groups": [{"shared_label": g["label"], "topics": g["topics"],
                                  "labels": [resolved[t] for t in g["topics"]]}
                                 for g in groups],
                      "n_renamed": len(renamed),
                      "renamed": {str(t): {"from": first_pass[t], "to": lab}
                                  for t, lab in sorted(renamed.items())},
                      "collisions_remaining": sorted(still_duplicate),
                      "usage": dedup_usage,
                      "cost_usd": estimate_cost(dedup_usage, model["price"]),
                      "seconds": [s for s in dedup_seconds if s is not None],
                      "failed_groups": dedup_failed},
            "price_usd_per_1m": model["price"], "price_source": PRICE_SOURCE,
            "prompt_sha256": prompt_sha, "prompt_version": args.prompt_version,
            "prompt_file": str(prompt_file),
            "inputs": {k: str(CORPORA[corpus][k]) for k in ("chunks", "stance", "terms")},
            "openai_sdk_version": openai.__version__,
            "validation": validation,
            "files": {"labels": str(out_csv), "audit": str(audit_xlsx),
                      "checkpoint": str(ckpt),
                      "checkpoint_dedup": str(OUT / f"checkpoint_dedup_{run_key}.jsonl"),
                      "metadata": str(meta_path)},
            "failed_topics": failed}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    print(f"  {run_key}: {len(labels)} topics named, {len(renamed)} renamed against a "
          f"collision, {validation['collisions_remaining']} name(s) still shared, "
          f"{validation['n_flagged']} flagged, "
          f"${meta['cost_usd']:.4f} -> {out_csv.name} + {audit_xlsx.name}")
    if dedup_failed:
        print(f"  {len(dedup_failed)} collision group(s) failed; those topics keep "
              f"their first-pass names. Re-run to fill them.")
    if failed:
        print(f"  {len(failed)} topic(s) failed; nothing was re-sent. Re-run to fill "
              f"them: {[f['topic'] for f in failed]}")
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
    # max_retries=0: a failed topic is recorded, never silently re-sent.
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
    ap.add_argument("--per-stance", type=int, default=N_PER_STANCE,
                    help="sentences drawn per stance. Twenty is where the label "
                         "stops following the particular sentences drawn; the "
                         "default is fifty, for margin on the mixed topics")
    ap.add_argument("--prompt-version", choices=sorted(PROMPT_VERSIONS),
                    default=DEFAULT_PROMPT_VERSION)
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help="concurrent requests, one topic each")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--run-tag", default="",
                    help="suffix for isolated smoke/test outputs")
    ap.add_argument("--topics", nargs="*", type=int, default=None,
                    help="name only these topic ids")
    ap.add_argument("--limit-topics", type=int, default=0,
                    help="name only the first N topics")
    ap.add_argument("--no-dedup", action="store_true",
                    help="skip the collision pass. The topics that came out of the "
                         "naming pass sharing a name then keep it, flagged")
    ap.add_argument("--no-pattern", action="store_true",
                    help="drop the length grammar from the response schema. Only "
                         "if the provider ever rejects it; the label is then held "
                         "short by the prompt alone and the flags still fire")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the samples, verify the schema and estimate cost, "
                         "no API calls")
    args = ap.parse_args()
    args.run_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_tag.strip())
    args.limit_topics = args.limit_topics or None
    args.workers = max(1, args.workers)
    if args.per_stance < 1:
        raise SystemExit("--per-stance must be at least 1")

    OUT.mkdir(parents=True, exist_ok=True)
    client = get_client(args.dry_run)
    print(f"  prompt version: {args.prompt_version}   "
          f"{args.per_stance} sentences per stance ({', '.join(STANCES)}), reshuffled")
    limit = ("prompt only (--no-pattern)" if args.no_pattern else
             f"schema grammar, at most {WORD_CAP} words (flagged above "
             f"{MAX_WORDS}), ending in a word, {HARD_CEILING_CHARS} characters")
    print(f"  label limit: {limit}; {MAX_OUTPUT_TOKENS} output tokens")

    started = datetime.now()
    summary, timing, scale = [], {}, {}
    for mk in args.models:
        for c in args.corpora:
            print(f"\n=== {MODELS[mk]['id']} · {c.upper()} ===")
            meta = run_one(client, mk, c, args)
            summary.append(meta)
            if not meta.get("dry_run"):
                t = meta["timing"]
                timing[c] = {"1 load": t["load_seconds"], "2 sample": t["sample_seconds"],
                             "3 label": t["label_seconds"],
                             "4 collisions": t["dedup_seconds"],
                             "5 audit": t["audit_seconds"]}
                scale[c] = {"n_docs": meta["n_topics"], "n_chunks": meta["n_sentences"]}
    if args.dry_run:
        tot = sum(m.get("rough_cost_usd", 0) for m in summary)
        print(f"\nRough total across all runs: ${tot:.2f}")
    else:
        print(f"\nTotal billed across all runs: "
              f"${sum(m.get('cost_usd', 0) for m in summary):.2f}")
        if timing and not args.run_tag:
            sys.path.insert(0, str(REPO / "code"))
            from timing import log_run
            log_run("02B topic labels", timing, scale=scale, started=started,
                    note=f"{', '.join(args.models)}; {args.per_stance} sentences per "
                         f"stance; n_docs = topics named, n_chunks = sentences sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
