import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import sklearn
import tomotopy as tp
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = DATA / "irt"

SEED = 14601
K = {"us": 108, "br": 158}
MIN_TF = 21        # complement of the 40B OOV fold: corpus freq <= 20 folds
MIN_DF = 11        # complement of the 40B OOV fold: doc freq <= 10 folds
TOP_N = 25         # terms persisted per topic (metrics use the first 10)
LDA_ITERS = 1000
NMF_MAX_ITER = 400
# leave two cores for the concurrently running STM fit
WORKERS = max(1, (os.cpu_count() or 8) - 2)


def build_matrix(corpus: str):
    t0 = time.time()
    df = pd.read_feather(DATA / "processed" / f"wf_tokens_{corpus}.feather")
    texts = df.tokens.tolist()
    cv_all = CountVectorizer(analyzer=str.split)
    X_all = cv_all.fit_transform(texts)
    tf = np.asarray(X_all.sum(0)).ravel()
    dfreq = np.asarray((X_all > 0).sum(0)).ravel()
    keep = (tf >= MIN_TF) & (dfreq >= MIN_DF)
    vocab = np.array(cv_all.get_feature_names_out())[keep]
    vpath = OUT / f"topic_quality_vocab_{corpus}.txt"
    vpath.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    X = X_all[:, keep]
    print(f"[{corpus}] docs={X.shape[0]:,} vocab {X_all.shape[1]:,} -> "
          f"{X.shape[1]:,} (tf>={MIN_TF}, df>={MIN_DF}); nnz={X.nnz:,} "
          f"({time.time()-t0:.1f}s) -> {vpath.name}")
    return X, vocab, texts


def top_terms(components: np.ndarray, vocab: np.ndarray) -> list[list[str]]:
    order = np.argsort(-components, axis=1)[:, :TOP_N]
    return [[vocab[j] for j in row] for row in order]


def _write(corpus: str, rows: list[dict], meta: list[dict]) -> None:
    new = pd.DataFrame(rows)
    tpath = OUT / f"topic_quality_top_{corpus}.csv"
    if tpath.exists():
        old = pd.read_csv(tpath)
        old = old[~old.model.isin(new.model.unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(tpath, index=False, encoding="utf-8")
    mpath = OUT / "topic_quality_fit_meta.csv"
    md = pd.DataFrame(meta)
    if mpath.exists():
        oldm = pd.read_csv(mpath)
        oldm = oldm[~((oldm.corpus == corpus) & oldm.model.isin(md.model))]
        md = pd.concat([oldm, md], ignore_index=True)
    md.to_csv(mpath, index=False, encoding="utf-8")
    print(f"[{corpus}] wrote topic_quality_top_{corpus}.csv + meta")


def fit(corpus: str, models: list[str]) -> None:
    print(f"\n{'='*74}\n{corpus.upper()}  (K={K[corpus]}, models={'+'.join(models)})\n{'='*74}")
    X, vocab, texts = build_matrix(corpus)
    k = K[corpus]
    rows, meta = [], []

    if "lda" in models:
        t0 = time.time()
        vs = set(vocab.tolist())
        mdl = tp.LDAModel(k=k, alpha=50.0 / k, eta=0.01, seed=SEED)
        mdl.optim_interval = 0    # keep the stated priors fixed during sampling
        skipped = 0
        for doc in texts:
            toks = [w for w in doc.split() if w in vs]
            if toks:
                mdl.add_doc(toks)
            else:
                skipped += 1
        mdl.train(LDA_ITERS, workers=WORKERS)
        s = time.time() - t0
        print(f"[{corpus}] LDA: {LDA_ITERS} Gibbs iterations, workers={WORKERS}, "
              f"{skipped} empty docs skipped, {s:.0f}s")
        for t in range(k):
            terms = [w for w, _ in mdl.get_topic_words(t, top_n=TOP_N)]
            rows += [dict(corpus=corpus, model="lda", topic=t, rank=r + 1, term=w)
                     for r, w in enumerate(terms)]
        meta.append(dict(
            corpus=corpus, model="lda",
            implementation=f"tomotopy {tp.__version__} LDAModel",
            K=k, vocab=len(vocab), docs=len(mdl.docs),
            config=(f"parallel collapsed Gibbs sampling; fixed alpha=50/K, "
                    f"eta=0.01; {LDA_ITERS} iterations; workers={WORKERS}; "
                    f"seed={SEED}"),
            iterations=LDA_ITERS, seconds=round(s, 1)))

    if "nmf" not in models:
        _write(corpus, rows, meta)
        return

    t0 = time.time()
    tfidf = TfidfTransformer().fit_transform(X)
    nmf = NMF(n_components=k, init="nndsvd", beta_loss="frobenius",
              max_iter=NMF_MAX_ITER, random_state=SEED)
    nmf.fit(tfidf)
    s = time.time() - t0
    print(f"[{corpus}] NMF: {nmf.n_iter_} iterations, {s:.0f}s")
    for t, terms in enumerate(top_terms(nmf.components_, vocab)):
        rows += [dict(corpus=corpus, model="nmf", topic=t, rank=r + 1, term=w)
                 for r, w in enumerate(terms)]
    meta.append(dict(
        corpus=corpus, model="nmf",
        implementation=f"scikit-learn {sklearn.__version__} NMF",
        K=k, vocab=len(vocab), docs=X.shape[0],
        config=(f"tf-idf weighting; NNDSVD init; Frobenius loss; coordinate "
                f"descent; max_iter={NMF_MAX_ITER}; seed={SEED}"),
        iterations=nmf.n_iter_, seconds=round(s, 1)))

    _write(corpus, rows, meta)


def main() -> int:
    which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    sel = (sys.argv[2] if len(sys.argv) > 2 else "all").lower()
    models = ["lda", "nmf"] if sel == "all" else [sel]
    for corpus in (["us", "br"] if which == "both" else [which]):
        fit(corpus, models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
