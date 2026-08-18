from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
DATA = REPO / "data"
PROC = DATA / "processed"
OUT = DATA / "irt"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 14605 - 2025 - 4        # the project's seed, as in every other stage

# ---- Doc2Vec hyperparameters (Case's settings, unchanged) ------------------- #
# Rare words are masked into a single bucket, never dropped, which is why
# Doc2Vec below runs with min_count=1: gensim must not prune on top of this.
#
# US keeps Case's rule exactly: corpus frequency > 10. The Brazilian corpus is an
# order of magnitude longer (41.5M token occurrences against 5.1M), so the same
# absolute frequency admits a far thinner word; there it takes a corpus frequency
# of at least 30 AND a document frequency of at least 10 to be kept.
OOV_RULE = {
    "us": dict(min_freq=11, min_docfreq=1),
    "br": dict(min_freq=30, min_docfreq=10),
}
OOV_TOKEN = "__oov__"
VECTOR_SIZE = 200
WINDOW = 15
NEGATIVE = 15
SAMPLE = 1e-4
DM_MODE = 1                # PV-DM
DBOW_WORDS = 1 - DM_MODE
FIXED_EPOCHS = 15
MIN_EPOCHS = 3
DRIFT_DELTA = 0.01         # early-stop when |change in median cosine| < this
DRIFT_SAMPLE = 30_000      # monitor drift on a sample, not the whole corpus

CPU_IDLE, MAX_THREADS = 0, 32
CPU = os.cpu_count() or 1
N_THREADS = min(max(1, CPU - CPU_IDLE), MAX_THREADS)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS)

import gensim  # noqa: E402
from gensim.models.doc2vec import TaggedDocument, Doc2Vec  # noqa: E402
from gensim.models.callbacks import CallbackAny2Vec  # noqa: E402


class CosineDriftLogger(CallbackAny2Vec):
    """Cosine drift between successive epochs; early-stops on a plateau.

    Ported verbatim from the project's Doc2Vec stage. The per-epoch medians are
    the fit's convergence trace and are persisted alongside the vectors.
    """

    def __init__(self, doc_tags, delta_threshold=DRIFT_DELTA, min_epochs=MIN_EPOCHS):
        self.doc_tags = doc_tags
        self.prev_vecs = None
        self.means, self.medians = [], []
        self.delta_threshold = delta_threshold
        self.min_epochs = min_epochs
        self._last_median = None
        self.epoch = 0
        self.stopped_early = False

    def on_train_begin(self, model):
        self.prev_vecs = np.vstack([model.dv[t] for t in self.doc_tags])

    def on_epoch_end(self, model):
        self.epoch += 1
        cur = np.vstack([model.dv[t] for t in self.doc_tags])
        if self.prev_vecs is not None:
            sims = np.einsum("ij,ij->i", cur, self.prev_vecs) / (
                np.linalg.norm(cur, axis=1) * np.linalg.norm(self.prev_vecs, axis=1))
            sims = np.clip(sims, -1, 1)
            m, md = float(sims.mean()), float(np.median(sims))
            self.means.append(m)
            self.medians.append(md)
            print(f"    epoch {self.epoch:2d} | cos-mean {m:.4f}  cos-median {md:.4f}")
            if self._last_median is not None:
                d = abs(md - self._last_median)
                if self.epoch > self.min_epochs and d < self.delta_threshold:
                    print(f"    -> early stop (change in median {d:.4f} "
                          f"< {self.delta_threshold})")
                    self.stopped_early = True
                    raise KeyboardInterrupt
            self._last_median = md
        self.prev_vecs = cur.copy()


# =============================== externals =============================== #
def externals_us(docs: np.ndarray) -> pd.DataFrame:
    sc = pd.read_csv(DATA / "us" / "campaignview_with_scores.csv",
                     dtype={"cd": str, "year": str})
    sc["doc_id"] = (sc.candidate_webname.astype(str) + "|" + sc.state_postal.astype(str)
                    + "|" + sc.cd.astype(str) + "|" + sc.cand_party.astype(str)
                    + "|" + sc.year.astype(str))
    sc = sc[["doc_id", "cfscore", "nominate_dim1"]].drop_duplicates("doc_id")
    out = pd.DataFrame({"doc_id": docs})
    out["party"] = [str(d).split("|")[3] if str(d).count("|") >= 3 else "" for d in docs]
    return out.merge(sc, on="doc_id", how="left")


def externals_br(docs: np.ndarray) -> pd.DataFrame:
    pmap = pd.read_feather(DATA / "br" / "platform_party_map.feather")[["platform_id", "party"]]
    pmap["platform_id"] = pmap.platform_id.astype(str)
    pmap["party"] = pmap.party.astype(str).str.upper()
    exp = pd.read_csv(DATA / "raw" / "party_scores_bolognesi2023.csv")
    exp = exp[["SG_PARTIDO", "party_mean_expert"]].rename(
        columns={"SG_PARTIDO": "party", "party_mean_expert": "expert"})
    exp["party"] = exp.party.str.upper()
    out = pd.DataFrame({"doc_id": np.asarray(docs, dtype=str)})
    out = out.merge(pmap.rename(columns={"platform_id": "doc_id"}), on="doc_id", how="left")
    return out.merge(exp, on="party", how="left")


def pcorr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan


def party_level_r(theta, party, expert):
    d = pd.DataFrame({"theta": theta, "party": party, "expert": expert}).dropna(
        subset=["party", "expert"])
    g = d.groupby("party").agg(mt=("theta", "mean"), expert=("expert", "first"))
    return pcorr(g.mt.to_numpy(), g.expert.to_numpy()), len(g)


# =============================== the fit =============================== #
def fit_one(corpus: str, variant: str) -> dict:
    suf = "_uni" if variant == "unigram" else ""
    tag = f"{corpus}/{variant}"
    print(f"\n{'=' * 70}\n[{tag}]\n{'=' * 70}")
    t0 = time.time()

    df = pd.read_feather(PROC / f"wf_tokens_{corpus}{suf}.feather")
    docs = df.doc_id.astype(str).to_numpy()
    toks = [t.split() for t in df.tokens.to_list()]
    n_occ = sum(len(t) for t in toks)
    print(f"  documents: {len(toks):,}  token occurrences: {n_occ:,} "
          f"({n_occ / max(len(toks), 1):,.0f} per document)")

    # ---- rare words -> a single OOV bucket (kept, never dropped) ------------
    rule = OOV_RULE[corpus]
    freq, docfreq = Counter(), Counter()
    for t in toks:
        freq.update(t)
        docfreq.update(set(t))
    keep = frozenset(w for w, c in freq.items()
                     if c >= rule["min_freq"] and docfreq[w] >= rule["min_docfreq"])
    toks = [[w if w in keep else OOV_TOKEN for w in t] for t in toks]
    print(f"  vocabulary (distinct types) {len(freq):,} -> kept at corpus "
          f"freq >= {rule['min_freq']} and doc freq >= {rule['min_docfreq']}: "
          f"{len(keep):,}; folded to {OOV_TOKEN}: {len(freq) - len(keep):,}")

    # ---- Doc2Vec: one tagged document per platform -------------------------
    tagged = [TaggedDocument(w, [i]) for i, w in enumerate(toks)]
    rng = np.random.default_rng(RANDOM_STATE)
    drift_tags = sorted(rng.choice(len(tagged),
                                   size=min(DRIFT_SAMPLE, len(tagged)),
                                   replace=False).tolist())
    logger = CosineDriftLogger(drift_tags)

    model = Doc2Vec(dm=DM_MODE, dbow_words=DBOW_WORDS, vector_size=VECTOR_SIZE,
                    window=WINDOW, min_count=1, negative=NEGATIVE, sample=SAMPLE,
                    epochs=FIXED_EPOCHS, workers=CPU, seed=RANDOM_STATE,
                    compute_loss=False)
    model.build_vocab(tagged)
    print(f"  vocabulary in model: {len(model.wv):,}; training PV-DM "
          f"{VECTOR_SIZE}d, window {WINDOW}, negative {NEGATIVE}, "
          f"up to {FIXED_EPOCHS} epochs on {CPU} workers")
    tw = time.time()
    try:
        model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs,
                    callbacks=[logger])
    except KeyboardInterrupt:
        pass
    fit_s = time.time() - tw

    V = np.vstack([model.dv[i] for i in range(len(toks))]).astype(np.float64)
    np.save(OUT / f"doc2vec_vectors_{corpus}{suf}.npy", V.astype(np.float32))
    pd.DataFrame({"epoch": np.arange(1, len(logger.medians) + 1),
                  "cos_mean": logger.means, "cos_median": logger.medians}).to_csv(
        OUT / f"doc2vec_drift_{corpus}{suf}.csv", index=False)

    # ---- PCA: first principal component of the document vectors ------------
    Xc = V - V.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    theta = U[:, 0] * S[0]
    evr = float(S[0] ** 2 / (S ** 2).sum())
    print(f"  PCA: PC1 explains {100 * evr:.1f}% of the variance in the "
          f"document vectors")

    ext = externals_us(docs) if corpus == "us" else externals_br(docs)
    assert list(ext.doc_id) == list(docs), "external join reordered the documents"

    # Orient once, globally, against the primary benchmark of the corpus.
    ref = (pcorr(theta, ext.cfscore.to_numpy()) if corpus == "us"
           else party_level_r(theta, ext.party.to_numpy(), ext.expert.to_numpy())[0])
    if np.isfinite(ref) and ref < 0:
        theta = -theta
    pd.DataFrame({"doc_id": docs, "theta": theta}).to_csv(
        OUT / f"doc2vec_theta_{corpus}{suf}.csv", index=False)

    name = f"Doc2Vec + PCA ({'unigrams' if variant == 'unigram' else 'with bigrams'})"
    row = {"estimator": name, "corpus": corpus, "variant": variant}
    if corpus == "us":
        p = ext.party.to_numpy()
        cf, nm = ext.cfscore.to_numpy(), ext.nominate_dim1.to_numpy()
        dem, rep = p == "Democrat", p == "Republican"
        row.update(CFScore_overall=pcorr(theta, cf),
                   CFScore_Dem=pcorr(theta[dem], cf[dem]),
                   CFScore_Rep=pcorr(theta[rep], cf[rep]),
                   NOMINATE_overall=pcorr(theta, nm),
                   NOMINATE_Dem=pcorr(theta[dem], nm[dem]),
                   NOMINATE_Rep=pcorr(theta[rep], nm[rep]))
        print(f"  -> {name} | CF {row['CFScore_overall']:.3f} "
              f"(D {row['CFScore_Dem']:.3f}, R {row['CFScore_Rep']:.3f}) | "
              f"NOM {row['NOMINATE_overall']:.3f} (D {row['NOMINATE_Dem']:.3f}, "
              f"R {row['NOMINATE_Rep']:.3f})")
    else:
        r, npar = party_level_r(theta, ext.party.to_numpy(), ext.expert.to_numpy())
        row["BR_overall"] = r
        print(f"  -> {name} | BR party-level r = {r:.3f} over {npar} parties")

    meta = {"estimator": name, "corpus": corpus, "variant": variant,
            "ndoc": len(docs), "token_occurrences": n_occ,
            "full_vocab": len(freq), "kept_vocab": len(keep), "vocab": len(model.wv),
            "min_freq": rule["min_freq"], "min_docfreq": rule["min_docfreq"],
            "vector_size": VECTOR_SIZE,
            "window": WINDOW, "negative": NEGATIVE, "sample": SAMPLE,
            "dm": DM_MODE, "epochs_max": FIXED_EPOCHS,
            "epochs_run": logger.epoch, "stopped_early": logger.stopped_early,
            "drift_delta": DRIFT_DELTA, "drift_sample": len(drift_tags),
            "pc1_explained_var": evr, "seed": RANDOM_STATE,
            "gensim": gensim.__version__, "fit_seconds": fit_s}
    print(f"  [{tag}] done in {time.time() - t0:.1f}s "
          f"({logger.epoch} epochs, {fit_s:.0f}s training)")
    return {"row": row, "meta": meta}


COLS = ["CFScore_overall", "CFScore_Dem", "CFScore_Rep",
        "NOMINATE_overall", "NOMINATE_Dem", "NOMINATE_Rep", "BR_overall"]
ORDER = ["Doc2Vec + PCA (unigrams)", "Doc2Vec + PCA (with bigrams)"]


def main() -> int:
    which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    todo = ["us", "br"] if which == "both" else [which]
    t0 = time.time()

    rows, metas = [], []
    for corpus in todo:
        for variant in ("unigram", "bigram"):
            res = fit_one(corpus, variant)
            rows.append(res["row"])
            metas.append(res["meta"])

    long = pd.DataFrame(rows)
    for c in COLS:
        if c not in long.columns:
            long[c] = np.nan

    def merge_keep(path, new):
        """Running one corpus must not discard the other corpus's rows."""
        if path.exists():
            old = pd.read_csv(path)
            if "corpus" in old.columns:
                new = pd.concat([old[~old.corpus.isin(todo)], new], ignore_index=True)
        return new.sort_values(["corpus", "variant"]).reset_index(drop=True)

    long = merge_keep(OUT / "doc2vec_metrics_long.csv", long)
    long.to_csv(OUT / "doc2vec_metrics_long.csv", index=False, encoding="utf-8")
    merge_keep(OUT / "doc2vec_meta.csv", pd.DataFrame(metas)).to_csv(
        OUT / "doc2vec_meta.csv", index=False, encoding="utf-8")

    # One row per estimator: the US fit supplies the US columns, BR the BR column.
    wide = long.groupby("estimator")[COLS].first().reindex(
        [e for e in ORDER if e in set(long.estimator)]).reset_index()
    wide.to_csv(OUT / "doc2vec_metrics.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 96)
    print("MODEL COMPARISON ROWS (Pearson r with the external benchmarks)")
    print("=" * 96)
    print(wide.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
    print(f"\nwrote {OUT / 'doc2vec_metrics.csv'}")
    print(f"wrote {OUT / 'doc2vec_meta.csv'}")
    print(f"TOTAL {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
