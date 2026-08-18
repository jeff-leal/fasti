import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import tomotopy as tp
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = DATA / "irt"

SEED = 14601
K = {"us": 108, "br": 158}
TOP_N = 25
LDA_ITERS = 1000
LDA_CHUNK = 50
NMF_RUNGS = [10, 25, 50, 100, 200, 300, 400]
WORKERS = 6


def log(corpus, msg):
    print(f"{time.strftime('%H:%M:%S')} [{corpus}] {msg}", flush=True)


def write_trace(rows):
    path = OUT / "topic_quality_trace_lda_nmf.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        drop = old.set_index(["model", "corpus"]).index.isin(
            new.set_index(["model", "corpus"]).index)
        new = pd.concat([old[~drop], new], ignore_index=True)
    new.to_csv(path, index=False, encoding="utf-8")


def load_texts(corpus):
    vocab = [w for w in (OUT / f"topic_quality_vocab_{corpus}.txt")
             .read_text(encoding="utf-8").splitlines() if w]
    texts = pd.read_feather(
        DATA / "processed" / f"wf_tokens_{corpus}.feather").tokens.tolist()
    return vocab, texts


def lda_trace(corpus):
    t0 = time.time()
    vocab, texts = load_texts(corpus)
    vs = set(vocab)
    k = K[corpus]
    mdl = tp.LDAModel(k=k, alpha=50.0 / k, eta=0.01, seed=SEED)
    mdl.optim_interval = 0    # keep the stated priors fixed during sampling
    skipped = 0
    for doc in texts:
        toks = [w for w in doc.split() if w in vs]
        if toks:
            mdl.add_doc(toks)
        else:
            skipped += 1
    log(corpus, f"LDA: docs={len(mdl.docs):,} ({skipped} empty skipped), "
                f"K={k} [prep {time.time()-t0:.0f}s]")
    tf0 = time.time()
    trace = []
    for done in range(LDA_CHUNK, LDA_ITERS + 1, LDA_CHUNK):
        mdl.train(LDA_CHUNK, workers=WORKERS)
        trace.append(dict(model="lda", corpus=corpus, iteration=done,
                          objective="ll_per_word", value=mdl.ll_per_word))
        if done % 200 == 0:
            log(corpus, f"LDA progress: {done}/{LDA_ITERS}, "
                        f"ll/word={mdl.ll_per_word:.3f}, "
                        f"{(time.time()-tf0)/60:.1f} min")
    secs = time.time() - tf0
    log(corpus, f"LDA: {LDA_ITERS} iterations, {secs:.0f}s")

    rows = []
    for t in range(k):
        for r, (w, _) in enumerate(mdl.get_topic_words(t, top_n=TOP_N)):
            rows.append(dict(corpus=corpus, model="lda", topic=t,
                             rank=r + 1, term=w))
    new = pd.DataFrame(rows)
    tpath = OUT / f"topic_quality_top_{corpus}.csv"
    old = pd.read_csv(tpath)
    pd.concat([old[old.model != "lda"], new], ignore_index=True) \
        .to_csv(tpath, index=False, encoding="utf-8")
    meta = dict(
        corpus=corpus, model="lda",
        implementation=f"tomotopy {tp.__version__} LDAModel",
        K=k, vocab=len(vocab), docs=len(mdl.docs),
        config=(f"parallel collapsed Gibbs sampling; fixed alpha=50/K, "
                f"eta=0.01; {LDA_ITERS} iterations; workers={WORKERS}; "
                f"seed={SEED}"),
        iterations=LDA_ITERS, seconds=round(secs, 1))
    mpath = OUT / "topic_quality_fit_meta.csv"
    oldm = pd.read_csv(mpath)
    oldm = oldm[~((oldm.corpus == corpus) & (oldm.model == "lda"))]
    pd.concat([oldm, pd.DataFrame([meta])], ignore_index=True) \
        .to_csv(mpath, index=False, encoding="utf-8")
    log(corpus, "LDA: canonical top terms + meta replaced")
    return trace


def nmf_trace(corpus):
    t0 = time.time()
    vocab, texts = load_texts(corpus)
    X = CountVectorizer(analyzer=str.split, vocabulary=vocab).fit_transform(texts)
    tfidf = TfidfTransformer().fit_transform(X)
    x_norm2 = float((tfidf.data ** 2).sum())
    log(corpus, f"NMF: matrix {tfidf.shape[0]:,} x {tfidf.shape[1]:,}, "
                f"nnz={tfidf.nnz:,} [prep {time.time()-t0:.0f}s]")
    trace = []
    for rung in NMF_RUNGS:
        t1 = time.time()
        nmf = NMF(n_components=K[corpus], init="nndsvd",
                  beta_loss="frobenius", max_iter=rung, random_state=SEED)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # expected: not converged at low caps
            W = nmf.fit_transform(tfidf)
        H = nmf.components_
        cross = float((tfidf.T @ W * H.T).sum())
        wh_norm2 = float(np.trace((W.T @ W) @ (H @ H.T)))
        rel = np.sqrt(max(x_norm2 - 2 * cross + wh_norm2, 0.0)) / np.sqrt(x_norm2)
        it = int(nmf.n_iter_)
        trace.append(dict(model="nmf", corpus=corpus, iteration=it,
                          objective="rel_frobenius", value=rel))
        log(corpus, f"NMF rung {rung}: n_iter={it}, rel loss={rel:.4f} "
                    f"({time.time()-t1:.0f}s)")
        if it < rung:
            log(corpus, f"NMF converged at {it} iterations")
            break
    return trace


def main() -> int:
    which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    for corpus in (["us", "br"] if which == "both" else [which]):
        rows = lda_trace(corpus) + nmf_trace(corpus)
        write_trace(rows)
        log(corpus, "trace rows written")
    print("40M done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
