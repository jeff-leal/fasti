from __future__ import annotations

from pathlib import Path
from typing import Literal, Union
import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone

import httpx
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, create_model
from tqdm import tqdm

try:
    from sklearn.metrics import classification_report, f1_score
except Exception:  # pragma: no cover - only used as a runtime fallback
    classification_report = None
    f1_score = None


PYDANTIC_V2 = hasattr(BaseModel, "model_validate")

if PYDANTIC_V2:
    FORBID_EXTRA_CONFIG = ConfigDict(extra="forbid")
else:
    class ForbidExtraConfig:
        extra = "forbid"

    FORBID_EXTRA_CONFIG = ForbidExtraConfig


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "topic2irt" / "data" / "processed" / "stratified_stance_sample.csv"
DEFAULT_PROMPT = ROOT / "topic2irt" / "code" / "prompts" / "stance_dimension_prompt_v01.txt"
DEFAULT_OUT_DIR = ROOT / "topic2irt" / "data" / "cmp_stance_dim"
API_KEY_PATH = ROOT / "misc" / "key.txt"

SCHEMA_VERSION = "stance_dimension_v02"
DEFAULT_SEED = 14605

DIM_LABELS = {
    "D01": "Economy",
    "D02": "Society",
}

STANCE_LABELS = {
    "S01": "State",
    "S02": "Market",
    "S03": "Progressive",
    "S04": "Conservative",
    "S05": "Neutral",
}

GOLD_TAG_MAP = {
    ("Economy", "Left"): ("D01", "S01"),
    ("Economy", "Right"): ("D01", "S02"),
    ("Economy", "Neutral"): ("D01", "S05"),
    ("Society", "Left"): ("D02", "S03"),
    ("Society", "Right"): ("D02", "S04"),
    ("Society", "Neutral"): ("D02", "S05"),
}

COMBINED_LABELS = [
    "D01/S01",
    "D01/S02",
    "D01/S05",
    "D02/S03",
    "D02/S04",
    "D02/S05",
]

MODEL_PRICES_USD_PER_1M = {
    "gpt-4.1-mini": {
        "standard": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
        "batch": {"input": 0.20, "cached_input": None, "output": 0.80},
    },
    "gpt-4.1": {
        "standard": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
        "batch": {"input": 1.00, "cached_input": None, "output": 4.00},
    },
}

PRICE_SOURCE = {
    "url": "https://developers.openai.com/api/docs/pricing",
    "observed_date": "2026-06-26",
    "basis": "Raw official pricing page source; prices are USD per 1M tokens.",
}

PROXY_PROMPT_TOKENS_PER_ROW = 140
PROXY_COMPLETION_TOKENS_PER_ROW = 40


class D01Code(BaseModel):
    if PYDANTIC_V2:
        model_config = FORBID_EXTRA_CONFIG
    else:
        class Config:
            extra = "forbid"

    dim: Literal["D01"]
    st: Literal["S01", "S02", "S05"]


class D02Code(BaseModel):
    if PYDANTIC_V2:
        model_config = FORBID_EXTRA_CONFIG
    else:
        class Config:
            extra = "forbid"

    dim: Literal["D02"]
    st: Literal["S03", "S04", "S05"]


SentenceCode = Union[D01Code, D02Code]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify manifesto sentences into dimension/stance tags."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("STANCE_DIM_MODEL", "gpt-4.1-mini"),
        help="OpenAI model. Default: gpt-4.1-mini.",
    )
    parser.add_argument(
        "--input",
        default=os.getenv("STANCE_DIM_INPUT", str(DEFAULT_INPUT)),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--prompt",
        default=os.getenv("STANCE_DIM_PROMPT", str(DEFAULT_PROMPT)),
        help="Prompt text file.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.getenv("STANCE_DIM_OUT_DIR", str(DEFAULT_OUT_DIR)),
        help="Output directory.",
    )
    parser.add_argument(
        "--scope",
        choices=["smoke", "validation", "full"],
        default=os.getenv("STANCE_DIM_SCOPE", "smoke"),
        help="Rows to classify.",
    )
    parser.add_argument(
        "--run-tag",
        default=os.getenv("STANCE_DIM_RUN_TAG", ""),
        help="Suffix for output/checkpoint names.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("STANCE_DIM_BATCH_SIZE", "50")),
        help="Rows per API call.",
    )
    parser.add_argument(
        "--validation-per-class",
        type=int,
        default=int(os.getenv("STANCE_DIM_VALIDATION_PER_CLASS", "100")),
        help="Rows sampled per dim_stance class for validation scope.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("STANCE_DIM_SEED", str(DEFAULT_SEED))),
        help="Random seed for smoke/validation row selection and API seed.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=int(os.getenv("STANCE_DIM_MAX_ROWS", "0")),
        help="Optional cap after scope selection. 0 means no cap.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs/schema and print estimates without API calls or writes.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def json_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def model_dump(obj):
    return obj.model_dump() if PYDANTIC_V2 else obj.dict()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def read_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sid": str})
    required = {
        "sid",
        "manifesto_id",
        "party",
        "partyname",
        "countryname",
        "country_group",
        "language",
        "date",
        "cmp_code",
        "dimension",
        "stance",
        "dim_stance",
        "n_words",
        "text",
        "text_en",
        "gpt_dimension",
        "gpt_stance",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Input is missing required columns: {missing}")
    if df["sid"].duplicated().any():
        dupes = df.loc[df["sid"].duplicated(), "sid"].head(10).tolist()
        raise RuntimeError(f"Duplicate sid values found: {dupes}")
    return df


def select_scope(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.scope == "full":
        selected = df.copy()
    elif args.scope == "smoke":
        n = min(args.batch_size, len(df))
        selected = df.sample(n=n, random_state=args.seed).sort_index()
    else:
        parts = []
        for _, group in df.groupby("dim_stance", sort=True):
            n = min(args.validation_per_class, len(group))
            parts.append(group.sample(n=n, random_state=args.seed))
        selected = pd.concat(parts).sort_index()

    if args.max_rows > 0:
        selected = selected.iloc[: args.max_rows].copy()
    return selected.reset_index(drop=True)


def row_keys(rows: pd.DataFrame) -> list[str]:
    return [f"r{i:03d}" for i in range(1, len(rows) + 1)]


def make_batch_model(keys: list[str]):
    fields = {key: (SentenceCode, ...) for key in keys}
    return create_model(
        "BatchStanceDimensionCodes",
        __config__=FORBID_EXTRA_CONFIG,
        **fields,
    )


def format_batch(rows: pd.DataFrame) -> str:
    items = []
    for key, row in zip(row_keys(rows), rows.itertuples(index=False)):
        items.append(
            {
                "id": key,
                "text": "" if pd.isna(row.text) else str(row.text),
            }
        )
    return json.dumps(
        {
            "task": (
                "Classify every item. Use exactly the input id as the output object key. "
                "Each object must contain only dim and st."
            ),
            "items": items,
        },
        ensure_ascii=False,
    )


def classify_failure(error: str) -> str:
    error_l = error.lower()
    if "finish_reason=length" in error_l:
        return "length"
    if "content_filter" in error_l:
        return "content_filter"
    if "refusal" in error_l:
        return "refusal"
    if "key mismatch" in error_l:
        return "key_mismatch"
    if "empty parsed" in error_l:
        return "empty_parsed"
    if "validation" in error_l or "literal" in error_l or "value error" in error_l:
        return "schema_validation"
    return "other"


def compatible(dim: str, st: str) -> bool:
    return (
        (dim == "D01" and st in {"S01", "S02", "S05"})
        or (dim == "D02" and st in {"S03", "S04", "S05"})
    )


def usage_to_dict(usage) -> dict:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}

    usage_dict = (
        usage.model_dump()
        if hasattr(usage, "model_dump")
        else usage.dict()
        if hasattr(usage, "dict")
        else {}
    )
    prompt_tokens = int(usage_dict.get("prompt_tokens") or 0)
    completion_tokens = int(usage_dict.get("completion_tokens") or 0)
    details = usage_dict.get("prompt_tokens_details") or {}
    cached_tokens = int(details.get("cached_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
    }


def pricing_for_model(model: str, mode: str = "standard") -> dict:
    for prefix, price_modes in MODEL_PRICES_USD_PER_1M.items():
        if model == prefix or model.startswith(prefix + "-"):
            return price_modes[mode]
    return MODEL_PRICES_USD_PER_1M["gpt-4.1-mini"][mode]


def estimate_cost(usage: dict, model: str, mode: str = "standard") -> float:
    prices = pricing_for_model(model, mode)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cached_tokens = int(usage.get("cached_tokens") or 0)
    uncached_tokens = max(prompt_tokens - cached_tokens, 0)
    cached_rate = prices["cached_input"]
    cached_cost = 0.0 if cached_rate is None else cached_tokens * cached_rate
    return (
        uncached_tokens * prices["input"]
        + cached_cost
        + completion_tokens * prices["output"]
    ) / 1_000_000


def proxy_usage(n_rows: int) -> dict:
    return {
        "prompt_tokens": int(n_rows * PROXY_PROMPT_TOKENS_PER_ROW),
        "completion_tokens": int(n_rows * PROXY_COMPLETION_TOKENS_PER_ROW),
        "cached_tokens": 0,
    }


def scale_usage(usage: dict, observed_rows: int, target_rows: int) -> dict:
    if observed_rows <= 0:
        return proxy_usage(target_rows)
    scale = target_rows / observed_rows
    return {
        "prompt_tokens": int(round((usage.get("prompt_tokens") or 0) * scale)),
        "completion_tokens": int(round((usage.get("completion_tokens") or 0) * scale)),
        "cached_tokens": int(round((usage.get("cached_tokens") or 0) * scale)),
    }


def call_api(client: OpenAI, rows: pd.DataFrame, prompt: str, model: str, seed: int) -> dict:
    expected_keys = row_keys(rows)
    sid_by_key = {
        key: str(row.sid)
        for key, row in zip(expected_keys, rows.itertuples(index=False))
    }
    BatchModel = make_batch_model(expected_keys)

    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": format_batch(rows)},
        ],
        response_format=BatchModel,
        temperature=0,
        seed=seed,
    )

    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError("finish_reason=length")
    if choice.finish_reason == "content_filter":
        raise RuntimeError("finish_reason=content_filter")

    message = choice.message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Refusal: {message.refusal}")
    if not message.parsed:
        raise RuntimeError("Empty parsed response")

    parsed = model_dump(message.parsed)
    got_keys = list(parsed.keys())
    if got_keys != expected_keys:
        raise RuntimeError(f"key mismatch: expected={expected_keys}, got={got_keys}")

    codes = []
    for key in expected_keys:
        code = parsed[key]
        if not compatible(code["dim"], code["st"]):
            raise RuntimeError(f"incompatible schema result for {key}: {code}")
        codes.append(
            {
                "id": sid_by_key[key],
                "batch_key": key,
                "dim": code["dim"],
                "st": code["st"],
            }
        )

    return {
        "codes": codes,
        "fingerprint": getattr(completion, "system_fingerprint", None),
        "returned_model": getattr(completion, "model", None),
        "usage": usage_to_dict(getattr(completion, "usage", None)),
    }


def checkpoint_paths(out_dir: Path, run_tag: str) -> dict:
    tag = f"_{run_tag}" if run_tag else ""
    return {
        "selected_input": out_dir / f"stance_dim_selected_input{tag}.csv",
        "labels": out_dir / f"stance_dim_labels{tag}.csv",
        "merged": out_dir / f"stance_dim_merged{tag}.csv",
        "checkpoint": out_dir / f"stance_dim_checkpoint{tag}.jsonl",
        "meta": out_dir / f"stance_dim_run_meta{tag}.json",
        "report": out_dir / f"stance_dim_eval_report{tag}.md",
        "cmp_diagnostics": out_dir / f"stance_dim_cmp_diagnostics{tag}.csv",
    }


def load_checkpoint(path: Path) -> tuple[set[int], list[dict], dict, set, set, int]:
    base_usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    if not path.exists():
        return set(), [], base_usage, set(), set(), 0

    done: set[int] = set()
    codes: list[dict] = []
    usage = dict(base_usage)
    fingerprints = set()
    returned_models = set()
    incompatible = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("schema_version") != SCHEMA_VERSION:
                incompatible += 1
                continue
            done.add(int(rec["batch_idx"]))
            codes.extend(rec["codes"])
            rec_usage = rec.get("usage") or {}
            for key in usage:
                usage[key] += int(rec_usage.get(key) or 0)
            fingerprints.add(rec.get("fingerprint"))
            returned_models.add(rec.get("returned_model"))
    return done, codes, usage, fingerprints, returned_models, incompatible


def save_checkpoint(path: Path, batch_idx: int, result: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "batch_idx": batch_idx,
                    "codes": result["codes"],
                    "fingerprint": result.get("fingerprint"),
                    "returned_model": result.get("returned_model"),
                    "usage": result.get("usage"),
                    "created_at": json_now(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def decode_labels(labels: pd.DataFrame) -> pd.DataFrame:
    decoded = labels.copy()
    decoded["gpt_dimension"] = decoded["dim"].map(DIM_LABELS)
    decoded["gpt_stance"] = decoded["st"].map(STANCE_LABELS)
    decoded["gpt_dim_stance_tag"] = decoded["dim"] + "/" + decoded["st"]
    decoded["gpt_dim_stance"] = decoded["gpt_dimension"] + "/" + decoded["gpt_stance"]
    return decoded


def merge_predictions(source: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    decoded = decode_labels(labels)[["id", "gpt_dimension", "gpt_stance", "gpt_dim_stance_tag"]]
    # Drop the empty placeholder prediction columns from the input first, so the
    # GPT predictions land in clean column names (no "_pred" suffix collision).
    src = source.drop(columns=["gpt_dimension", "gpt_stance"], errors="ignore")
    merged = src.merge(decoded, left_on="sid", right_on="id", how="left")
    if "id" in merged.columns:
        merged = merged.drop(columns=["id"])
    return merged


def add_gold_tags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    gold_dim = []
    gold_st = []
    for row in out.itertuples(index=False):
        dim, st = GOLD_TAG_MAP.get((str(row.dimension), str(row.stance)), (None, None))
        gold_dim.append(dim)
        gold_st.append(st)
    out["gold_dim"] = gold_dim
    out["gold_st"] = gold_st
    out["gold_combined"] = [
        f"{d}/{s}" if d and s else None for d, s in zip(out["gold_dim"], out["gold_st"])
    ]
    return out


def metric_summary(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    if f1_score is None:
        return {}
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "class_f1": {
            label: float(f1_score(y_true, y_pred, labels=[label], average="macro", zero_division=0))
            for label in labels
        },
    }


def write_evaluation(
    selected: pd.DataFrame,
    labels: pd.DataFrame,
    paths: dict,
    model: str,
    usage: dict,
) -> dict:
    if labels.empty:
        return {}

    pred = labels.copy()
    pred["pred_combined"] = pred["dim"] + "/" + pred["st"]
    scored = add_gold_tags(selected).merge(pred, left_on="sid", right_on="id", how="inner")
    scored = scored[scored["gold_combined"].notna()].copy()
    if scored.empty:
        return {}

    y_true = scored["gold_combined"].tolist()
    y_pred = scored["pred_combined"].tolist()
    combined_metrics = metric_summary(y_true, y_pred, COMBINED_LABELS)
    dim_metrics = metric_summary(
        scored["gold_dim"].tolist(),
        scored["dim"].tolist(),
        ["D01", "D02"],
    )
    stance_metrics = metric_summary(
        scored["gold_st"].tolist(),
        scored["st"].tolist(),
        ["S01", "S02", "S03", "S04", "S05"],
    )

    scored["agree"] = scored["gold_combined"] == scored["pred_combined"]
    min_n = 20 if len(scored) >= 1000 else 5
    diagnostics = []
    for cmp_code, group in scored.groupby("cmp_code", dropna=False):
        n = len(group)
        if n < min_n:
            continue
        disagreements = group[~group["agree"]]
        disagree_n = len(disagreements)
        if disagree_n:
            top_alt = disagreements["pred_combined"].value_counts().idxmax()
            top_alt_n = int((disagreements["pred_combined"] == top_alt).sum())
        else:
            top_alt = ""
            top_alt_n = 0
        disagree_rate = disagree_n / n if n else 0
        top_alt_share = top_alt_n / disagree_n if disagree_n else 0
        if disagree_rate >= 0.5 or top_alt_share >= 0.7:
            diagnostics.append(
                {
                    "cmp_code": cmp_code,
                    "n": n,
                    "disagree_n": disagree_n,
                    "disagree_rate": disagree_rate,
                    "top_gpt_alt": top_alt,
                    "top_gpt_alt_n": top_alt_n,
                    "top_gpt_alt_share_among_disagreements": top_alt_share,
                    "gold_distribution": group["gold_combined"].value_counts().to_dict(),
                    "pred_distribution": group["pred_combined"].value_counts().to_dict(),
                }
            )

    diag_df = pd.DataFrame(diagnostics).sort_values(
        ["disagree_rate", "n"], ascending=[False, False]
    ) if diagnostics else pd.DataFrame()
    diag_df.to_csv(paths["cmp_diagnostics"], index=False)

    report_lines = [
        "# Stance Dimension Evaluation",
        "",
        f"- Model: `{model}`",
        f"- Rows evaluated: {len(scored)}",
        f"- Usage: `{json.dumps(usage, sort_keys=True)}`",
        f"- Estimated standard cost: ${estimate_cost(usage, model):.6f}",
        "",
        "## Summary Metrics",
        "",
        f"- Combined macro F1: {combined_metrics.get('macro_f1', float('nan')):.3f}",
        f"- Dimension macro F1: {dim_metrics.get('macro_f1', float('nan')):.3f}",
        f"- Stance macro F1: {stance_metrics.get('macro_f1', float('nan')):.3f}",
        "",
        "## Combined Class F1",
        "",
    ]
    for label in COMBINED_LABELS:
        score = combined_metrics.get("class_f1", {}).get(label, float("nan"))
        report_lines.append(f"- `{label}`: {score:.3f}")

    if classification_report is not None:
        report_lines.extend(
            [
                "",
                "## Classification Report",
                "",
                "```text",
                classification_report(
                    y_true,
                    y_pred,
                    labels=COMBINED_LABELS,
                    zero_division=0,
                ),
                "```",
            ]
        )

    report_lines.extend(
        [
            "",
            "## CMP Code Diagnostics",
            "",
            f"- Flagged CMP codes: {len(diag_df)}",
            f"- Diagnostics CSV: `{paths['cmp_diagnostics']}`",
        ]
    )
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "n_evaluated": int(len(scored)),
        "combined": combined_metrics,
        "dimension": dim_metrics,
        "stance": stance_metrics,
        "cmp_diagnostics_rows": int(len(diag_df)),
        "report": str(paths["report"]),
        "cmp_diagnostics": str(paths["cmp_diagnostics"]),
    }


def build_meta(
    args: argparse.Namespace,
    input_path: Path,
    prompt_path: Path,
    out_dir: Path,
    prompt: str,
    selected: pd.DataFrame,
    full_n: int,
    total_usage: dict,
    failed_batches: list[dict],
    fingerprints: set,
    returned_models: set,
    evaluation: dict,
) -> dict:
    observed_projection = None
    if total_usage["prompt_tokens"] or total_usage["completion_tokens"]:
        projected_usage = scale_usage(total_usage, len(selected) - sum(b["n_rows"] for b in failed_batches), full_n)
        observed_projection = {
            "full_sample_rows": full_n,
            "projected_usage": projected_usage,
            "projected_standard_cost_usd": estimate_cost(projected_usage, args.model),
            "projected_gpt4_1_standard_cost_usd": estimate_cost(projected_usage, "gpt-4.1"),
        }

    initial_usage = proxy_usage(full_n)
    return {
        "created_at": json_now(),
        "schema_version": SCHEMA_VERSION,
        "model_alias": args.model,
        "model_prices_usd_per_1m": MODEL_PRICES_USD_PER_1M,
        "price_source": PRICE_SOURCE,
        "openai_sdk_version": openai_version(),
        "pydantic_version": pydantic_version(),
        "input": str(input_path),
        "prompt": str(prompt_path),
        "prompt_sha256": prompt_hash(prompt),
        "out_dir": str(out_dir),
        "scope": args.scope,
        "run_tag": args.run_tag,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_selected": int(len(selected)),
        "full_sample_rows": int(full_n),
        "tag_mapping": {
            "dimension": DIM_LABELS,
            "stance": STANCE_LABELS,
        },
        "gold_mapping": {f"{k[0]}/{k[1]}": f"{v[0]}/{v[1]}" for k, v in GOLD_TAG_MAP.items()},
        "usage": total_usage,
        "estimated_standard_cost_usd": estimate_cost(total_usage, args.model),
        "initial_proxy_estimate": {
            "usage": initial_usage,
            "standard_cost_usd": estimate_cost(initial_usage, args.model),
            "gpt4_1_standard_cost_usd": estimate_cost(initial_usage, "gpt-4.1"),
            "prompt_tokens_per_row": PROXY_PROMPT_TOKENS_PER_ROW,
            "completion_tokens_per_row": PROXY_COMPLETION_TOKENS_PER_ROW,
        },
        "observed_projection": observed_projection,
        "fingerprints": sorted(str(x) for x in fingerprints),
        "returned_models": sorted(str(x) for x in returned_models),
        "failed_batches": failed_batches,
        "evaluation": evaluation,
    }


def openai_version() -> str:
    try:
        import openai

        return getattr(openai, "__version__", "unknown")
    except Exception:
        return "unknown"


def pydantic_version() -> str:
    try:
        import pydantic

        return getattr(pydantic, "__version__", "unknown")
    except Exception:
        return "unknown"


def main():
    args = parse_args()
    input_path = resolve_path(args.input)
    prompt_path = resolve_path(args.prompt)
    out_dir = resolve_path(args.out_dir)
    paths = checkpoint_paths(out_dir, args.run_tag or args.scope)

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    df = read_input(input_path)
    selected = select_scope(df, args)
    full_n = len(df)
    batches = [
        selected.iloc[i : i + args.batch_size]
        for i in range(0, len(selected), args.batch_size)
    ]

    if args.dry_run:
        print(f"Scope: {args.scope}")
        print(f"Selected rows: {len(selected)}")
        print(f"Batches: {len(batches)}")
        print(f"Prompt SHA256: {prompt_hash(prompt)}")
        print(f"Output dir: {out_dir}")
        print(f"Initial full-sample mini estimate: ${estimate_cost(proxy_usage(full_n), args.model):.4f}")
        print(f"Initial full-sample gpt-4.1 estimate: ${estimate_cost(proxy_usage(full_n), 'gpt-4.1'):.4f}")
        if batches:
            make_batch_model(row_keys(batches[0]))
            print("First batch schema constructed OK.")
        return

    if not API_KEY_PATH.exists():
        raise RuntimeError(f"API key file not found: {API_KEY_PATH}")

    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(paths["selected_input"], index=False)

    done_batches, all_codes, total_usage, fingerprints, returned_models, incompatible = load_checkpoint(
        paths["checkpoint"]
    )
    if done_batches:
        print(f"Resuming: {len(done_batches)}/{len(batches)} batches already done.")
    if incompatible:
        print(f"Ignoring {incompatible} checkpoint records from older schema versions.")

    failed_batches: list[dict] = []
    client = OpenAI(
        api_key=API_KEY_PATH.read_text(encoding="utf-8").strip(),
        http_client=httpx.Client(verify=False),
    )

    for batch_idx, batch in enumerate(tqdm(batches, desc=f"Coding {args.scope}")):
        if batch_idx in done_batches:
            continue
        try:
            result = call_api(client, batch, prompt, args.model, args.seed)
        except Exception as exc:
            tqdm.write(f"FAILED batch {batch_idx + 1}/{len(batches)}: {exc}")
            failed_batches.append(
                {
                    "batch_idx": batch_idx,
                    "n_rows": int(len(batch)),
                    "error_type": classify_failure(str(exc)),
                    "error": str(exc),
                }
            )
            continue

        save_checkpoint(paths["checkpoint"], batch_idx, result)
        all_codes.extend(result["codes"])
        for key in total_usage:
            total_usage[key] += int((result.get("usage") or {}).get(key) or 0)
        fingerprints.add(result.get("fingerprint"))
        returned_models.add(result.get("returned_model"))
        tqdm.write(
            f"batch {batch_idx + 1}/{len(batches)} | rows={len(batch)} | "
            f"tokens={result.get('usage')}"
        )

    labels = pd.DataFrame(all_codes).drop_duplicates(subset=["id"], keep="last")
    if not labels.empty:
        labels = labels.sort_values("id")
    labels.to_csv(paths["labels"], index=False)

    merged = merge_predictions(selected, labels)
    merged.to_csv(paths["merged"], index=False)

    evaluation = write_evaluation(selected, labels, paths, args.model, total_usage)
    meta = build_meta(
        args=args,
        input_path=input_path,
        prompt_path=prompt_path,
        out_dir=out_dir,
        prompt=prompt,
        selected=selected,
        full_n=full_n,
        total_usage=total_usage,
        failed_batches=failed_batches,
        fingerprints=fingerprints,
        returned_models=returned_models,
        evaluation=evaluation,
    )
    paths["meta"].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone: {len(labels)}/{len(selected)} rows labelled.")
    print(f"Usage: {total_usage}")
    print(f"Estimated standard cost: ${estimate_cost(total_usage, args.model):.6f}")
    if meta["observed_projection"]:
        print(
            "Projected full-sample standard cost from observed usage: "
            f"${meta['observed_projection']['projected_standard_cost_usd']:.4f}"
        )
        print(
            "Projected full-sample gpt-4.1 standard cost from observed usage: "
            f"${meta['observed_projection']['projected_gpt4_1_standard_cost_usd']:.4f}"
        )
    if failed_batches:
        print(f"Failed batches: {[b['batch_idx'] for b in failed_batches]}")
    print(f"Labels: {paths['labels']}")
    print(f"Merged: {paths['merged']}")
    print(f"Meta: {paths['meta']}")
    if evaluation:
        print(f"Report: {paths['report']}")


if __name__ == "__main__":
    main()
