from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import pearsonr, spearmanr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = Path(os.getenv("TOPIC2IRT_ROOT", HERE.parents[0]))
DATA = REPO / "data"
IRT = DATA / "irt"
IRT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE))
import macro_topics as mt  # noqa: E402
import topic_codes  # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


irt_data = _by_path("irt_data07", HERE / "04_irt_data.py")
jirt = _by_path("joint_irt", HERE / "joint_irt.py")

MODELS = {"gpt54mini": "GPT-5.4 mini", "gpt4omini": "GPT-4o mini"}
#: the two stance runs: the original prompt, and the one that tells the model to
#: ignore the names of parties, politicians and places.
PROMPTS = ("v01", "nonames")
PROMPT = "nonames"
MIN_CHUNKS = irt_data.MIN_CHUNKS
DROP_CODES = (mt.BOILERPLATE_CODE,)          # keep OTH, the "none of the above" class


def design_weights(corpus, docs, party):
    """w_i = N_h / n_h for the party stratum the document was drawn from."""
    D = irt_data.load_matrices(corpus, verbose=False)
    avail = pd.Series(D["party"]).value_counts()          # N_h in the estimation sample
    drawn = pd.Series(party).value_counts()               # n_h actually scored
    w = np.array([avail.get(p, np.nan) / drawn.get(p, np.nan) for p in party], float)
    if not np.isfinite(w).all():
        bad = sorted(set(np.asarray(party)[~np.isfinite(w)]))
        raise ValueError(f"{corpus}: no stratum size for {bad}")
    return w


def load_frame(model, corpus, prompt=PROMPT):
    """One row per sentence: doc, GPT class code, and GPT stance in [-1, 1]."""
    tp = pd.read_csv(DATA / "gpt_topics" / f"topics_{model}_{corpus}_sample.csv",
                     usecols=["chunk_id", "doc_id", "code"])
    tag = "" if prompt == "v01" else f"_{prompt}"     # v01 files carry no suffix
    sc = pd.read_csv(DATA / "gpt_scaling" / f"scores_{model}_{corpus}{tag}.csv",
                     usecols=["chunk_id", "political", "score"])
    df = tp.merge(sc, on="chunk_id", how="inner")
    n_all = len(df)

    # A non-political sentence takes the neutral value, exactly the role the
    # classifier's Neutral class plays in the unsupervised leg. A sentence the
    # model returned as NA carries no stance and is dropped.
    pol = df["political"].astype(str).str.lower().isin({"true", "1", "yes"})
    df["sval"] = np.where(pol, 2.0 * df["score"].astype(float) / 100.0 - 1.0, 0.0)
    na = pol & ~np.isfinite(df["score"].astype(float))
    df = df[~na].copy()
    print(f"[{model}/{prompt}/{corpus}] {n_all:,} sentences coded and scored, "
          f"{int(na.sum())} returned no score")
    return df


def build(model, corpus, prompt=PROMPT):
    df = load_frame(model, corpus, prompt)

    keep_codes = [c for c in topic_codes.CODES if c not in DROP_CODES]
    dropped = df["code"].isin(DROP_CODES)
    print(f"[{model}/{corpus}] {int(dropped.sum()):,} sentences in "
          f"{', '.join(DROP_CODES)} dropped before estimating; "
          f"{int((df['code'] == mt.RESIDUAL_CODE).sum()):,} in "
          f"{mt.RESIDUAL_CODE} kept")
    df = df[~dropped]

    size = df.groupby("doc_id")["doc_id"].transform("size")
    short = size < MIN_CHUNKS
    if short.any():
        print(f"[{model}/{corpus}] {df.loc[short, 'doc_id'].nunique()} document(s) "
              f"left with fewer than {MIN_CHUNKS} sentences and dropped")
    df = df[~short]

    docs, d_ix = np.unique(df.doc_id.to_numpy(), return_inverse=True)
    codes = [c for c in keep_codes if c in set(df["code"])]
    col = {c: j for j, c in enumerate(codes)}
    t_ix = df["code"].map(col).to_numpy()
    N, K = len(docs), len(codes)
    Y = coo_matrix((np.ones(len(d_ix)), (d_ix, t_ix)), shape=(N, K)).tocsr()
    Y.sum_duplicates()
    S = coo_matrix((df["sval"].to_numpy(float), (d_ix, t_ix)), shape=(N, K)).tocsr()
    S.sum_duplicates()
    Y.sort_indices()
    S.sort_indices()

    D = irt_data.load_matrices(corpus, verbose=False)
    ix = pd.Series(np.arange(len(D["docs"])), index=np.asarray(D["docs"]).astype(str))
    pos = ix.reindex(docs).to_numpy()
    if not np.isfinite(pos).all():
        raise KeyError(f"{corpus}: GPT documents absent from the estimation sample")
    pos = pos.astype(int)
    party = np.asarray(D["party"])[pos]
    out = dict(Y=Y, S=S, codes=np.array(codes, dtype=object), docs=docs, party=party,
               w=design_weights(corpus, docs, party), kind=D["kind"],
               model=model, corpus=corpus, prompt=prompt)
    if D["kind"] == "us":
        out["cfscore"] = D["cfscore"][pos]
        out["nominate"] = D["nominate"][pos]
    else:
        out["ext"] = D["ext"][pos]
    # the per-document ask-and-average score, on the same sentences
    out["askavg"] = (df.groupby("doc_id")["sval"].mean().reindex(docs).to_numpy(float))
    print(f"[{model}/{corpus}] N={N:,} K={K} cells={Y.nnz:,} "
          f"weights {out['w'].min():.2f}-{out['w'].max():.2f}")
    return out


def _orient_flip(theta, P):
    if P["kind"] == "us":
        ok = np.isfinite(P["cfscore"])
        return pearsonr(theta[ok], P["cfscore"][ok])[0] < 0
    ext, party = P["ext"], P["party"]
    ok = np.isfinite(ext)
    parties = sorted(set(party[ok]))
    pm = np.array([theta[(party == p) & ok].mean() for p in parties])
    pe = np.array([ext[(party == p) & ok][0] for p in parties])
    return spearmanr(pm, pe).correlation < 0


def fit_one(model, corpus, prompt=PROMPT, refit=False):
    npz = IRT / f"joint_irt_fit_gpt_{model}_{prompt}_{corpus}_godambe.npz"
    sepz = IRT / f"joint_irt_se_gpt_{model}_{prompt}_{corpus}.npz"
    csvp = IRT / f"gpt_askavg_{model}_{corpus}_{prompt}.csv"
    if npz.exists() and sepz.exists() and csvp.exists() and not refit:
        print(f"[{model}/{corpus}] cached fit reused")
        return

    P = build(model, corpus, prompt)
    pd.DataFrame({"doc_id": P["docs"], "party": P["party"], "w": P["w"],
                  "theta": P["askavg"]}).to_csv(csvp, index=False)

    t0 = time.perf_counter()
    fit = jirt.fit_joint_irt(P["Y"], P["S"], sample_weight=P["w"], orientation=1.0,
                             use_godambe=True, verbose=False)
    secs = time.perf_counter() - t0
    theta = np.asarray(fit.theta, float)
    beta = np.asarray(fit.beta, float)
    rho = np.asarray(fit.ownership, float)
    if _orient_flip(theta, P):
        theta, beta, rho = -theta, -beta, -rho
        print(f"[{model}/{corpus}] axis reflected so that + = right")

    np.savez(npz, theta=theta, alpha=np.asarray(fit.alpha, float), beta=beta,
             ownership=rho, curvature=np.asarray(fit.curvature, float),
             sigma2=fit.sigma2, lam=np.asarray(fit.lam, float),
             kappa_dm=fit.kappa_dm, c_stance=fit.c_stance, c_count=fit.c_count,
             godambe_factor=fit.godambe_factor, tau2_curv_std=fit.tau2_curv_std,
             columns=P["codes"].astype(str), docs=P["docs"].astype(str),
             party=P["party"].astype(str), w=P["w"], prompt=prompt, model=model,
             dropped_columns=np.asarray(DROP_CODES, dtype=str),
             oriented=True, fit_time=secs, joint_conv=fit.joint_converged,
             n_docs=int(P["Y"].shape[0]), n_topics=int(P["Y"].shape[1]))

    Yc = jirt._canonical_csr(P["Y"])
    s = jirt._prepare_stance(Yc, P["S"])
    yplus = np.asarray(Yc.sum(axis=1)).ravel().astype(float)
    kap = float(fit.kappa_dm)
    d = (np.ones_like(yplus) if not np.isfinite(kap)
         else np.maximum((yplus + kap) / (1.0 + kap), 1.0))
    sand = jirt._joint_sandwich(
        theta, Yc, s, P["w"], np.asarray(fit.alpha, float), beta, float(fit.sigma2),
        np.asarray(fit.lam, float), rho, np.asarray(fit.curvature, float), 1.0 / d,
        float(fit.c_stance), float(fit.c_count), jirt._active_rows(Yc),
        Yc.indices.astype(np.int64), Yc.data, yplus, 32768, float(fit.tau2_curv_std))
    np.savez(sepz, beta_se=sand["beta_se"], ownership_se=sand["ownership_se"])
    print(f"[{model}/{corpus}] {secs:.0f}s converged={fit.joint_converged} -> {npz.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="*", default=["us", "br"])
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--prompts", nargs="*", default=list(PROMPTS))
    ap.add_argument("--refit", action="store_true")
    a = ap.parse_args()
    for m in a.models:
        for pr in a.prompts:
            for c in [x.lower() for x in a.corpora]:
                fit_one(m, c, prompt=pr, refit=a.refit)


if __name__ == "__main__":
    main()
