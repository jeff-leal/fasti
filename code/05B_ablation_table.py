from __future__ import annotations
import sys
import time
import importlib.util
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console defaults to cp1252
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]                          # .../topic2irt   (code/ is one level down)
DATA = REPO / "data"
OUT = DATA / "irt"

SVAL = {"Left": -1.0, "Right": 1.0, "Neutral": 0.0}

# ---- load the estimator classes (module name starts with a digit -> importlib) ---------- #
_spec = importlib.util.spec_from_file_location("est07a", HERE / "ideal_point_estimators.py")
est = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(est)
EmbeddingPCA, StanceIRT, TopicIRT, TwoStageIRT = est.EmbeddingPCA, est.StanceIRT, est.TopicIRT, est.TwoStageIRT


# =============================== data loading =============================== #
def _emb_join(chunk_ids: np.ndarray, doc_ix: np.ndarray, ids_path: Path):
    """Map each modeled chunk to its row in the embedding matrix (inner join on chunk_id)."""
    emb_ids = np.load(ids_path, allow_pickle=True)
    emb_map = pd.DataFrame({"chunk_id": emb_ids.astype(str), "emb_row": np.arange(len(emb_ids), dtype=np.int64)})
    j = pd.DataFrame({"chunk_id": chunk_ids.astype(str), "doc_ix": doc_ix.astype(np.int64)})
    j = j.merge(emb_map, on="chunk_id", how="inner")
    return j.emb_row.to_numpy(np.int64), j.doc_ix.to_numpy(np.int64)


MIN_CHUNKS = 10   # must match code/04_irt_data.py: identical estimation sample


def _drop_nonplatform(df, corpus):
    """Remove the same non-platform filings 04_irt_data.py removes.

    The ids are taken from that stage's own output rather than recomputed here, so the
    two samples cannot drift apart when the screen is retuned.
    """
    z = np.load(DATA / "irt" / f"irt_matrices_{corpus}.npz", allow_pickle=True)
    bad = set(z["nonplatform"].astype(str)) if "nonplatform" in z.files else set()
    if not bad:
        return df
    keep = ~df.doc_id.astype(str).isin(bad)
    print(f"[{corpus}] non-platform filings dropped: "
          f"{df.loc[~keep, 'doc_id'].nunique():,} of {df.doc_id.nunique():,}")
    return df[keep].copy()


def _drop_short(df, corpus):
    df = _drop_nonplatform(df, corpus)
    keep = df.groupby("doc_id")["doc_id"].transform("size") >= MIN_CHUNKS
    print(f"[{corpus}] documents with < {MIN_CHUNKS} chunks dropped: "
          f"{df.loc[~keep, 'doc_id'].nunique():,} of {df.doc_id.nunique():,}")
    return df[keep].copy()


def _to_main_columns(df, corpus):
    """Put every row on the paper's main column scheme.

    The ablation asks what each block of the model contributes, so every row of
    the table has to be estimated on the same columns the joint fit uses: the
    fine topics grouped into the codebook's classes, with the boilerplate class
    out.  Leaving the ablations on the fine topics would confound the block
    being removed with the column scheme.
    """
    sys.path.insert(0, str(HERE))
    import macro_topics as mt
    cmap = mt.crosswalk(corpus)
    df = df.copy()
    df["topic"] = df["topic"].astype(int).map(cmap)
    drop = df["topic"] == mt.BOILERPLATE_CODE
    print(f"[{corpus}] {int(drop.sum()):,} boilerplate sentences dropped; "
          f"{df.loc[~drop, 'topic'].nunique()} macro classes")
    df = df[~drop]
    keep = df.groupby("doc_id")["doc_id"].transform("size") > 0
    return df[keep].copy()


def load_us() -> dict:
    sp = pd.read_csv(DATA / "stance" / "stance_pred_us.csv", usecols=["chunk_id", "stance"])
    ct = pd.read_feather(DATA / "topics" / "topic_model_us" / "chunk_topics.feather",
                         columns=["chunk_id", "doc_id", "topic"])
    df = ct.merge(sp, on="chunk_id", how="inner")
    df = df[df.topic >= 0].copy()
    df["party"] = df.doc_id.str.split("|").str[3]
    df = _drop_short(df, "us")
    df = _to_main_columns(df, "us")
    df["sval"] = df.stance.map(SVAL)

    docs, d_ix = np.unique(df.doc_id.to_numpy(), return_inverse=True)
    tops, t_ix = np.unique(df.topic.to_numpy().astype(str), return_inverse=True)
    N, T = len(docs), len(tops)
    Y = coo_matrix((np.ones(len(d_ix)), (d_ix, t_ix)), shape=(N, T)).tocsr(); Y.sum_duplicates(); Y.sort_indices()
    S = coo_matrix((df.sval.to_numpy(float), (d_ix, t_ix)), shape=(N, T)).tocsr(); S.sum_duplicates(); S.sort_indices()
    party = pd.Series(df.party.to_numpy(), index=df.doc_id.to_numpy()).groupby(level=0).first().reindex(docs).to_numpy()

    sc = pd.read_csv(DATA / "us" / "campaignview_with_scores.csv", dtype={"cd": str, "year": str})
    sc["doc_id"] = (sc.candidate_webname.astype(str) + "|" + sc.state_postal.astype(str) + "|" + sc.cd.astype(str)
                    + "|" + sc.cand_party.astype(str) + "|" + sc.year.astype(str))
    ext = sc[["doc_id", "cfscore", "nominate_dim1"]].drop_duplicates("doc_id").set_index("doc_id")
    cfscore = ext.cfscore.reindex(docs).to_numpy(float)
    nominate = ext.nominate_dim1.reindex(docs).to_numpy(float)

    emb_rows, emb_docix = _emb_join(df.chunk_id.to_numpy(), d_ix, DATA / "us" / "emb_minilm_chunk_ids.npy")
    return dict(corpus="us", docs=docs, party=party, Y=Y, S=S, N=N, T=T,
                externals={"CFScore": cfscore, "DW-NOMINATE": nominate},
                emb_path=DATA / "us" / "emb_minilm.npy", emb_rows=emb_rows, emb_docix=emb_docix, ESC=None)


def load_br() -> dict:
    exp = pd.read_csv(DATA / "raw" / "party_scores_bolognesi2023.csv")[["SG_PARTIDO", "party_mean_expert"]]
    ESC = dict(zip(exp.SG_PARTIDO.str.upper(), exp.party_mean_expert.astype(float)))
    sp = pd.read_csv(DATA / "stance" / "stance_pred_br.csv", usecols=["chunk_id", "stance"])
    ct = pd.read_feather(DATA / "topics" / "topic_model_br" / "chunk_topics.feather",
                         columns=["chunk_id", "platform_id", "topic"])
    df = ct.merge(sp, on="chunk_id", how="inner")
    df = df[df.topic >= 0].rename(columns={"platform_id": "doc_id"})
    # LEFT join: every platform is scaled; party only supplies the benchmark.
    pmap = pd.read_feather(DATA / "br" / "platform_party_map.feather")[["platform_id", "party"]]
    pmap["party"] = pmap.party.astype(str).str.upper()
    df = df.merge(pmap.rename(columns={"platform_id": "doc_id"}), on="doc_id", how="left")
    df["party"] = df.party.fillna("UNKNOWN")
    df = _drop_short(df, "br")
    df = _to_main_columns(df, "br")
    df["sval"] = df.stance.map(SVAL)

    docs, d_ix = np.unique(df.doc_id.to_numpy(), return_inverse=True)
    tops, t_ix = np.unique(df.topic.to_numpy().astype(str), return_inverse=True)
    N, T = len(docs), len(tops)
    Y = coo_matrix((np.ones(len(d_ix)), (d_ix, t_ix)), shape=(N, T)).tocsr(); Y.sum_duplicates(); Y.sort_indices()
    S = coo_matrix((df.sval.to_numpy(float), (d_ix, t_ix)), shape=(N, T)).tocsr(); S.sum_duplicates(); S.sort_indices()
    party = pd.Series(df.party.to_numpy(), index=df.doc_id.to_numpy()).groupby(level=0).first().reindex(docs).to_numpy().astype(str)

    emb_rows, emb_docix = _emb_join(df.chunk_id.to_numpy(), d_ix, DATA / "br" / "emb_minilm_multi_chunk_ids.npy")
    return dict(corpus="br", docs=docs, party=party, Y=Y, S=S, N=N, T=T,
                externals={}, ESC=ESC, party_ext=np.array([ESC.get(p, np.nan) for p in party]),
                emb_path=DATA / "br" / "emb_minilm_multi.npy", emb_rows=emb_rows, emb_docix=emb_docix)


# =============================== metrics =============================== #
def pcorr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan


def within_one_party(theta, ext, party, which) -> float:
    """Pearson r of theta vs ext among candidates of a single party (e.g. only Democrats)."""
    m = np.isfinite(ext) & (np.asarray(party) == which)
    return pcorr(np.asarray(theta, float)[m], np.asarray(ext, float)[m])


def party_level_pcorr(theta, party, ESC) -> float:
    """BR: correlate party-mean theta with the party expert score (external is party level)."""
    pm = pd.Series(np.asarray(theta, float)).groupby(np.asarray(party)).mean()
    pe = pd.Series(ESC)
    common = pm.index.intersection(pe.index)
    return pcorr(pm.loc[common].to_numpy(), pe.loc[common].to_numpy())


def coverage(D) -> dict:
    """Observations entering each correlation (same for every estimator: theta is always finite)."""
    if D["corpus"] == "us":
        cf, nm, p = D["externals"]["CFScore"], D["externals"]["DW-NOMINATE"], D["party"]
        f = np.isfinite
        return {"CFScore_overall": int(f(cf).sum()),
                "CFScore_Dem": int((f(cf) & (p == "Democrat")).sum()),
                "CFScore_Rep": int((f(cf) & (p == "Republican")).sum()),
                "NOMINATE_overall": int(f(nm).sum()),
                "NOMINATE_Dem": int((f(nm) & (p == "Democrat")).sum()),
                "NOMINATE_Rep": int((f(nm) & (p == "Republican")).sum())}
    common = [q for q in pd.unique(D["party"]) if q in D["ESC"]]
    return {"BR_overall": len(common)}


# =============================== fit + evaluate =============================== #
def make_estimators() -> list:
    return [EmbeddingPCA(), StanceIRT(), TopicIRT(), TwoStageIRT()]


def fit_theta(estimator, D):
    """Dispatch the right inputs to each estimator's fit()."""
    name = estimator.name
    if isinstance(estimator, EmbeddingPCA):
        emb = np.load(D["emb_path"], mmap_mode="r")
        estimator.fit(emb, D["emb_rows"], D["emb_docix"], D["N"])
    elif isinstance(estimator, (StanceIRT, TwoStageIRT)):
        estimator.fit(D["Y"], D["S"])
    elif isinstance(estimator, TopicIRT):
        estimator.fit(D["Y"])
    else:
        raise TypeError(name)
    return np.asarray(estimator.theta_, float)


def orient(theta, D):
    """Flip theta so it correlates positively with the primary benchmark (CFScore for US, expert for BR)."""
    if D["corpus"] == "us":
        ref = pcorr(theta, D["externals"]["CFScore"])
    else:
        ref = party_level_pcorr(theta, D["party"], D["ESC"])
    return -theta if (np.isfinite(ref) and ref < 0) else theta


def evaluate(D) -> pd.DataFrame:
    rows = []
    for estr in make_estimators():
        t0 = time.perf_counter()
        theta = orient(fit_theta(estr, D), D)
        dt = time.perf_counter() - t0
        if D["corpus"] == "us":
            party = D["party"]; cf = D["externals"]["CFScore"]; nm = D["externals"]["DW-NOMINATE"]
            rec = {
                "estimator": estr.name,
                "CFScore_overall": pcorr(theta, cf),
                "CFScore_Dem": within_one_party(theta, cf, party, "Democrat"),
                "CFScore_Rep": within_one_party(theta, cf, party, "Republican"),
                "NOMINATE_overall": pcorr(theta, nm),
                "NOMINATE_Dem": within_one_party(theta, nm, party, "Democrat"),
                "NOMINATE_Rep": within_one_party(theta, nm, party, "Republican"),
            }
        else:
            rec = {
                "estimator": estr.name,
                "BR_overall": party_level_pcorr(theta, D["party"], D["ESC"]),
                "BR_overall_candidate": pcorr(theta, D["party_ext"]),
            }
        print(f"  [{D['corpus']}] {estr.name:<18} fit {dt:5.1f}s  "
              + "  ".join(f"{k}={v:+.3f}" for k, v in rec.items() if k != "estimator"))
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    OUT.mkdir(parents=True, exist_ok=True)

    res, cov = {}, {}
    if which in ("us", "both"):
        print("Loading US ..."); D = load_us(); res["us"] = evaluate(D); cov.update(coverage(D))
    if which in ("br", "both"):
        print("Loading BR ..."); D = load_br(); res["br"] = evaluate(D); cov.update(coverage(D))

    # merge into one wide table keyed by estimator, in the canonical row order
    order = ["Embeddings + PCA", "Stance IRT", "Topic IRT", "2-stage IRT"]
    wide = pd.DataFrame({"estimator": order})
    if "us" in res:
        wide = wide.merge(res["us"], on="estimator", how="left")
    if "br" in res:
        wide = wide.merge(res["br"][["estimator", "BR_overall"]], on="estimator", how="left")

    cols = [c for c in ["CFScore_overall", "CFScore_Dem", "CFScore_Rep",
                        "NOMINATE_overall", "NOMINATE_Dem", "NOMINATE_Rep", "BR_overall"]
            if c in wide.columns]
    hdr = {"CFScore_overall": "CF Overall", "CFScore_Dem": "CF w-Dem", "CFScore_Rep": "CF w-Rep",
           "NOMINATE_overall": "NOM Overall", "NOMINATE_Dem": "NOM w-Dem", "NOMINATE_Rep": "NOM w-Rep",
           "BR_overall": "BR Overall"}

    W = 112
    print("\n" + "=" * W)
    print("ABLATION: Pearson r of estimated theta with external ideology")
    print("US: Overall = all candidates; w-Dem / w-Rep = within Democrats / within Republicans.")
    print("BR: Overall = party level (the expert score exists only per party).")
    print("=" * W)
    print(f"{'Estimator':<18}" + "".join(f"{hdr[c]:>13}" for c in cols))
    print(f"{'n (obs)':<18}" + "".join(f"{cov.get(c, ''):>13}" for c in cols))
    print("-" * W)
    for _, r in wide.iterrows():
        line = f"{r['estimator']:<18}"
        for c in cols:
            line += f"{r[c]:>13.2f}" if pd.notna(r[c]) else f"{'--':>13}"
        print(line)
    print("=" * W)

    wide.to_csv(OUT / "ablation_table.csv", index=False, encoding="utf-8")
    tidy = wide.melt(id_vars="estimator", var_name="metric", value_name="pearson_r").dropna()
    tidy["n"] = tidy["metric"].map(cov)
    tidy.to_csv(OUT / "ablation_metrics.csv", index=False, encoding="utf-8")
    pd.Series(cov, name="n").to_csv(OUT / "ablation_coverage.csv", encoding="utf-8")
    print(f"\nSaved -> {OUT / 'ablation_table.csv'}")
    print(f"Saved -> {OUT / 'ablation_metrics.csv'}")
    print(f"Saved -> {OUT / 'ablation_coverage.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
