from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GPT = REPO / "data" / "gpt_scaling"

# The classifier owns the batching and the schema; import it so this test cannot
# drift from production (the leading digit blocks a plain import).
_spec = importlib.util.spec_from_file_location("gpt40e", HERE / "40E_gpt_classify.py")
g = importlib.util.module_from_spec(_spec)
sys.modules["gpt40e"] = g
_spec.loader.exec_module(g)


def pick_sentences(corpus: str, n: int, seed: int) -> pd.DataFrame:
    """A reproducible spread of real sentences from the scored sample."""
    keep = set(pd.read_csv(GPT / f"gpt_sample_{corpus}.csv", dtype=str).doc_id)
    df = pd.read_feather(g.CORPORA[corpus]["text"],
                         columns=["chunk_id", "doc_id", "chunk_text"])
    df["doc_id"] = df.doc_id.astype(str)
    df["chunk_id"] = df.chunk_id.astype(str)
    df = df[df.doc_id.isin(keep)]
    # Skip fragments: a two-word line cannot be placed left or right by anyone.
    df = df[df.chunk_text.astype(str).str.len() >= 60]
    df = df.sort_values(["doc_id", "chunk_id"], kind="stable")
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def score_once(client, model: dict, prompt: str, sents: pd.DataFrame,
               batch_size: int, temperature: float | None, seed: int) -> list:
    """One full pass over the sentences, in batches, returning one row each."""
    rows = []
    for start in range(0, len(sents), batch_size):
        batch = sents.iloc[start:start + batch_size]
        keys = g.batch_keys(len(batch))
        BatchModel = g.make_batch_model(keys)
        kw = {"seed": seed}
        if temperature is not None:
            kw["temperature"] = temperature
        if model["reasoning_effort"]:
            kw["reasoning_effort"] = model["reasoning_effort"]
        completion = client.chat.completions.create(
            model=model["id"],
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": g.format_batch(batch, keys)}],
            response_format=g.response_format(keys), **kw)
        parsed = BatchModel.parse_obj(
            json.loads(completion.choices[0].message.content)).dict()
        for k, (cid, txt) in zip(keys, zip(batch.chunk_id, batch.chunk_text)):
            v = parsed[k]
            rows.append({"chunk_id": cid, "score": None if v == "NA" else int(v),
                         "raw": "NA" if v == "NA" else int(v), "text": txt})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt54mini", choices=list(g.MODELS))
    ap.add_argument("--corpus", default="br", choices=list(g.CORPORA))
    ap.add_argument("--n-sentences", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-temperature", action="store_true",
                    help="omit the parameter entirely (the pre-2026-07-22 behaviour)")
    ap.add_argument("--seed", type=int, default=g.SEED)
    args = ap.parse_args()

    model = g.MODELS[args.model]
    temp = None if args.no_temperature else args.temperature
    prompt = g.CORPORA[args.corpus]["prompt"].read_text(encoding="utf-8").strip()
    sents = pick_sentences(args.corpus, args.n_sentences, args.seed)
    n_batches = (len(sents) + args.batch_size - 1) // args.batch_size

    print(f"=== reproducibility · {model['id']} · {args.corpus.upper()} ===")
    print(f"  {len(sents)} sentences x {args.replicates} replicates "
          f"= {n_batches * args.replicates} calls of {args.batch_size} sentences")
    print(f"  temperature = {'OMITTED' if temp is None else temp}"
          f", reasoning_effort = {model['reasoning_effort']}, seed = {args.seed}")
    print(f"  prompt sha256 = {g.hashlib.sha256(prompt.encode()).hexdigest()[:16]}")

    client = g.get_client(dry_run=False)
    if client == "powershell":
        raise SystemExit("this test needs the native client; unset OPENAI_USE_POWERSHELL_HTTP")

    recs = []
    for r in range(args.replicates):
        try:
            rows = score_once(client, model, prompt, sents, args.batch_size, temp, args.seed)
        except Exception as exc:
            msg = str(exc).replace("\n", " ")
            print(f"\n  REPLICATE {r + 1} FAILED: {msg[:400]}")
            if temp is not None and ("temperature" in msg.lower()
                                     or "unsupported" in msg.lower()):
                print("\n  >>> The API REJECTED temperature for this model. <<<")
            return 1
        for row in rows:
            recs.append({"replicate": r + 1, **row})
        print(f"  replicate {r + 1}/{args.replicates} done", flush=True)

    d = pd.DataFrame(recs)
    wide = d.pivot(index="chunk_id", columns="replicate", values="raw")
    wide = wide.reindex(sents.chunk_id)                     # keep the sampled order
    text = sents.set_index("chunk_id").chunk_text

    # ---- per-sentence mode and agreement -------------------------------- #
    summary = []
    for cid, row in wide.iterrows():
        vals = [v for v in row.tolist() if v is not None and not pd.isna(v)]
        cnt = Counter(vals)
        mode, hits = (cnt.most_common(1)[0] if cnt else (None, 0))
        summary.append({"chunk_id": cid, "mode": mode,
                        "agreement": hits / len(vals) if vals else np.nan,
                        "distinct": len(cnt), "identical": len(cnt) == 1,
                        "values": ",".join(str(v) for v in row.tolist()),
                        "text": str(text.loc[cid])[:90]})
    S = pd.DataFrame(summary)

    # ---- between-replicate correlation ---------------------------------- #
    num = wide.apply(pd.to_numeric, errors="coerce")
    pear, spear = [], []
    for a, b in itertools.combinations(num.columns, 2):
        m = num[a].notna() & num[b].notna()
        if m.sum() > 2 and num.loc[m, a].nunique() > 1 and num.loc[m, b].nunique() > 1:
            pear.append(pearsonr(num.loc[m, a], num.loc[m, b])[0])
            spear.append(spearmanr(num.loc[m, a], num.loc[m, b])[0])

    exact = float(S.identical.mean())
    mean_agree = float(S.agreement.mean())
    print(f"\n  sentences identical across all {args.replicates} replicates: "
          f"{int(S.identical.sum())}/{len(S)}  ({exact:.1%})")
    print(f"  mean per-sentence agreement with the mode : {mean_agree:.3f}")
    if pear:
        print(f"  between-replicate Pearson  r: mean {np.mean(pear):+.4f}  "
              f"min {np.min(pear):+.4f}  ({len(pear)} pairs)")
        print(f"  between-replicate Spearman r: mean {np.mean(spear):+.4f}  "
              f"min {np.min(spear):+.4f}")
    print("\n  per-sentence detail (scores across replicates):")
    for _, r in S.iterrows():
        flag = "  " if r.identical else "<<"
        print(f"   {flag} mode {str(r['mode']):>4}  agree {r.agreement:.2f}  "
              f"[{r['values']}]  {r.text[:60]}")

    tag = f"{args.model}_{args.corpus}"
    S.to_csv(GPT / f"reproducibility_{tag}.csv", index=False, encoding="utf-8")
    md = [f"# Reproducibility of the ask-and-average scores — {model['id']}, "
          f"{args.corpus.upper()}", "",
          f"- temperature: {'omitted' if temp is None else temp}",
          f"- reasoning effort: {model['reasoning_effort']}",
          f"- seed: {args.seed}", f"- sentences: {len(S)}",
          f"- replicates: {args.replicates}  (batches of {args.batch_size})",
          f"- sentences identical across every replicate: {int(S.identical.sum())}/{len(S)} "
          f"({exact:.1%})",
          f"- mean per-sentence agreement with the modal score: {mean_agree:.3f}"]
    if pear:
        md += [f"- between-replicate Pearson r: mean {np.mean(pear):+.4f}, "
               f"min {np.min(pear):+.4f} over {len(pear)} pairs",
               f"- between-replicate Spearman r: mean {np.mean(spear):+.4f}, "
               f"min {np.min(spear):+.4f}"]
    md += ["", f"_generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_"]
    (GPT / f"REPRODUCIBILITY_{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n  wrote reproducibility_{tag}.csv and REPRODUCIBILITY_{tag}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
