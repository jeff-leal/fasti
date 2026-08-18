from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
IRT = REPO / "data" / "irt"
GPT = REPO / "data" / "gpt_scaling"

# Stage 07 owns the benchmarks and the party metadata, aligned to doc_id.
_spec = importlib.util.spec_from_file_location("irt_data07", HERE / "04_irt_data.py")
irt_data = importlib.util.module_from_spec(_spec)
sys.modules["irt_data07"] = irt_data
_spec.loader.exec_module(irt_data)

MODELS = {"gpt4omini": "GPT-4o mini", "gpt54mini": "GPT-5.4 mini"}
CORPORA = ["us", "br"]


def doc_theta(model_key: str, corpus: str) -> pd.DataFrame | None:
    """Document ideal point = mean of the numeric (political) sentence scores."""
    p = GPT / f"scores_{model_key}_{corpus}.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p, dtype={"doc_id": str, "chunk_id": str})
    pol = s[s.political == True].copy()                              # noqa: E712
    pol["score"] = pd.to_numeric(pol["score"], errors="coerce")
    pol = pol[np.isfinite(pol["score"])]
    theta = pol.groupby("doc_id")["score"].mean().rename("theta").reset_index()
    n_docs = s.doc_id.nunique()
    dropped = n_docs - theta.doc_id.nunique()
    if dropped:
        print(f"  [{model_key}/{corpus}] {dropped:,} of {n_docs:,} documents had no "
              f"political sentence and get no score")
    return theta


def bench(corpus: str) -> dict:
    """Benchmarks and party keyed by doc_id, from the estimation-sample cache."""
    D = irt_data.load_matrices(corpus, verbose=False)
    docs = pd.Index(D["docs"].astype(str))
    out = {"docs": docs, "party": pd.Series(D["party"], index=docs)}
    if corpus == "us":
        out["cfscore"] = pd.Series(D["cfscore"], index=docs)
        out["nominate"] = pd.Series(D["nominate"], index=docs)
    else:
        out["ext"] = pd.Series(D["ext"], index=docs)
    return out


def _r(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return pearsonr(a[m], b[m])[0] if m.sum() > 2 else np.nan


def report_us(theta: pd.DataFrame, B: dict) -> list[str]:
    t = theta.set_index("doc_id")["theta"].reindex(B["docs"]).to_numpy(float)
    cf, nom = B["cfscore"].to_numpy(float), B["nominate"].to_numpy(float)
    party = B["party"].to_numpy()
    dem, rep = party == "Democrat", party == "Republican"

    def rr(bench_arr, mask=None):
        a, b = t.copy(), bench_arr.copy()
        if mask is not None:
            a, b = np.where(mask, a, np.nan), np.where(mask, b, np.nan)
        return _r(a, b)

    n = int(np.isfinite(t).sum())
    rows = [f"- Documents scored (with a benchmark counterpart): {n:,}",
            f"- CFScore    r = {rr(cf):+.3f}  (Dem {rr(cf, dem):+.3f}, Rep {rr(cf, rep):+.3f})",
            f"- DW-NOMINATE r = {rr(nom):+.3f}  (Dem {rr(nom, dem):+.3f}, Rep {rr(nom, rep):+.3f})"]
    for line in rows:
        print("  " + line)
    return rows


def report_br(theta: pd.DataFrame, B: dict) -> list[str]:
    t = theta.set_index("doc_id")["theta"].reindex(B["docs"])
    df = pd.DataFrame({"theta": t.to_numpy(float),
                       "party": B["party"].to_numpy(),
                       "ext": B["ext"].to_numpy(float)}, index=B["docs"])
    df = df[np.isfinite(df.theta) & np.isfinite(df.ext)]
    grp = df.groupby("party").agg(n=("theta", "size"), theta=("theta", "mean"),
                                  ext=("ext", "first")).reset_index()
    r = _r(grp.theta.to_numpy(), grp.ext.to_numpy())
    rows = [f"- Documents scored (party has an expert score): {len(df):,}",
            f"- Parties covered: {len(grp)}",
            f"- Party-level expert placement r = {r:+.3f}"]
    for line in rows:
        print("  " + line)
    return rows, grp.sort_values("ext")


def main() -> int:
    GPT.mkdir(parents=True, exist_ok=True)
    B = {c: bench(c) for c in CORPORA}
    report = ["# GPT ask-and-average — ideal points and benchmark correlations", "",
              "Document ideal point = mean of the sentence scores the model marked "
              "political (non-political sentences dropped), following Le Mens & Gallego "
              "(2025). Scores run 0 (extremely left) to 100 (extremely right), so a "
              "positive correlation is the expected sign.", ""]
    any_scores = False

    for model_key, model_name in MODELS.items():
        for corpus in CORPORA:
            theta = doc_theta(model_key, corpus)
            if theta is None:
                continue
            any_scores = True
            out = IRT / f"{model_key}_theta_{corpus}.csv"
            theta.to_csv(out, index=False, encoding="utf-8")
            print(f"\n{model_name} · {corpus.upper()}  ({len(theta):,} documents) -> {out.name}")
            report += [f"## {model_name} — {corpus.upper()}", ""]
            if corpus == "us":
                report += report_us(theta, B["us"]) + [""]
            else:
                rows, grp = report_br(theta, B["br"])
                grp.to_csv(GPT / f"party_theta_{model_key}_br.csv", index=False, encoding="utf-8")
                report += rows + [""]

    if not any_scores:
        print("No scores_*.csv found yet. Run code/40E_gpt_classify.py first "
              "(it needs an OpenAI key and costs money).")
        return 0

    (GPT / "IDEAL_POINTS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nwrote {(GPT / 'IDEAL_POINTS_REPORT.md').relative_to(REPO)}")
    print("The theta CSVs are read by code/50F_tables_paper.py as two comparison-table rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
