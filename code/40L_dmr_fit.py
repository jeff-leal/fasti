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

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = DATA / "irt"

SEED = 14601
K = {"us": 108, "br": 158}
TOP_N = 25
DMR_ITERS = 1000            # same sweep count as the LDA row
WORKERS = 6                 # same as the LDA row (cpu - 2)
OPTIM = {"us": 50, "br": 10}        # lambda-optimization interval (see header)
BUDGET_MIN = {"us": 25, "br": 38}   # wall-clock budget per corpus, minutes
CHUNK_CAP = {"us": 50, "br": 25}    # smaller chunks -> denser convergence trace


def log(corpus, msg):
    print(f"{time.strftime('%H:%M:%S')} [{corpus}] {msg}", flush=True)


def covariate_labels(corpus: str, doc_ids: list[str]) -> tuple[list[list[str]], str]:
    if corpus == "us":
        labels = []
        for d in doc_ids:
            p = d.split("|")
            labels.append([f"author={p[0]}", f"state={p[1]}", f"party={p[3]}"])
        return labels, "candidate + state + party"
    m = pd.read_feather(DATA / "br" / "platform_party_map.feather")
    lut = {str(pid): (str(st), str(pt).upper())
           for pid, st, pt in zip(m.platform_id, m.state, m.party)}
    labels = []
    for d in doc_ids:
        st, pt = lut[str(d)]
        labels.append([f"state={st}", f"party={pt}"])
    return labels, "state + party"


def fit(corpus: str, n_sub: int | None, iters: int,
        budget_min: float | None = None) -> None:
    t0 = time.time()
    budget = 60.0 * (budget_min if budget_min is not None else BUDGET_MIN[corpus])
    k = K[corpus]
    vocab = [w for w in (OUT / f"topic_quality_vocab_{corpus}.txt")
             .read_text(encoding="utf-8").splitlines() if w]
    vs = set(vocab)
    df = pd.read_feather(DATA / "processed" / f"wf_tokens_{corpus}.feather")
    if n_sub is not None:
        df = df.sample(n=n_sub, random_state=SEED).sort_index()
        log(corpus, f"TEST MODE: {n_sub}-doc subsample, {iters} iterations")
    labels, cov_desc = covariate_labels(corpus, df.doc_id.astype(str).tolist())

    mdl = tp.DMRModel(k=k, alpha=50.0 / k, eta=0.01, seed=SEED)
    mdl.optim_interval = OPTIM[corpus]
    skipped = 0
    n_tok = 0
    for doc, lab in zip(df.tokens.tolist(), labels):
        toks = [w for w in doc.split() if w in vs]
        if toks:
            mdl.add_doc(toks, multi_metadata=lab)
            n_tok += len(toks)
        else:
            skipped += 1
    log(corpus, f"docs={len(mdl.docs):,} ({skipped} empty skipped), "
                f"tokens={n_tok:,}, K={k}, covariates: {cov_desc} "
                f"({len(mdl.multi_metadata_dict):,} labels) "
                f"[prep {time.time()-t0:.0f}s]")

    def write_out(done: int, secs: float) -> None:
        rows = []
        for t in range(k):
            for r, (w, _) in enumerate(mdl.get_topic_words(t, top_n=TOP_N)):
                rows.append(dict(corpus=corpus, model="dmr", topic=t,
                                 rank=r + 1, term=w))
        new = pd.DataFrame(rows)
        if n_sub is not None:
            new.to_csv(OUT / f"topic_quality_top_dmr_{corpus}_test.csv",
                       index=False, encoding="utf-8")
            return
        meta = dict(
            corpus=corpus, model="dmr",
            implementation=f"tomotopy {tp.__version__} DMRModel",
            K=k, vocab=len(vocab), docs=len(mdl.docs),
            config=(f"Dirichlet-multinomial regression on topic prevalence; "
                    f"covariates {cov_desc} as multi-hot labels; parallel "
                    f"collapsed Gibbs with lambda optimization every "
                    f"{mdl.optim_interval} iterations; alpha=50/K, eta=0.01, "
                    f"sigma=1; {done} iterations; workers={WORKERS}; "
                    f"seed={SEED}"),
            iterations=done, seconds=round(secs, 1))
        tpath = OUT / f"topic_quality_top_{corpus}.csv"
        old = pd.read_csv(tpath)
        pd.concat([old[old.model != "dmr"], new], ignore_index=True) \
            .to_csv(tpath, index=False, encoding="utf-8")
        mpath = OUT / "topic_quality_fit_meta.csv"
        oldm = pd.read_csv(mpath)
        oldm = oldm[~((oldm.corpus == corpus) & (oldm.model == "dmr"))]
        pd.concat([oldm, pd.DataFrame([meta])], ignore_index=True) \
            .to_csv(mpath, index=False, encoding="utf-8")

    tf0 = time.time()
    done = 0
    trace = []
    while done < iters:
        left = budget - (time.time() - t0)
        if done == 0:
            step = 25
        else:
            rate = (time.time() - tf0) / done
            feasible = int((left * 0.92) / rate) // 25 * 25
            step = min(CHUNK_CAP[corpus], feasible, iters - done)
            if step < 25:
                log(corpus, f"budget reached: {done} iterations in "
                            f"{(time.time()-t0)/60:.1f} min")
                break
        mdl.train(step, workers=WORKERS)
        done += step
        secs = time.time() - tf0
        trace.append(dict(model="dmr", corpus=corpus, iteration=done,
                          objective="ll_per_word", value=mdl.ll_per_word))
        log(corpus, f"progress: {done}/{iters} iterations, {secs/done:.2f} s/iter, "
                    f"ll/word={mdl.ll_per_word:.3f}, {(time.time()-t0)/60:.1f} min elapsed")
        write_out(done, secs)
    secs = time.time() - tf0

    if n_sub is None:
        tr_path = OUT / "topic_quality_trace_dmr.csv"
        new_tr = pd.DataFrame(trace)
        if tr_path.exists():
            old_tr = pd.read_csv(tr_path)
            new_tr = pd.concat([old_tr[old_tr.corpus != corpus], new_tr],
                               ignore_index=True)
        new_tr.to_csv(tr_path, index=False, encoding="utf-8")

    if n_sub is not None:
        log(corpus, f"TEST done: wrote topic_quality_top_dmr_{corpus}_test.csv | "
                    f"fit {secs:.0f}s | TOTAL {time.time()-t0:.0f}s")
        return
    log(corpus, f"wrote topic_quality_top_{corpus}.csv + fit_meta | "
                f"{done} iterations | TOTAL {time.time()-t0:.0f}s")


def main() -> int:
    which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    sub = sys.argv[2].lower() if len(sys.argv) > 2 else "full"
    n_sub = None if sub == "full" else int(sub)
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else DMR_ITERS
    budget_min = float(sys.argv[4]) if len(sys.argv) > 4 else None
    for corpus in (["us", "br"] if which == "both" else [which]):
        fit(corpus, n_sub, iters, budget_min)
    return 0


if __name__ == "__main__":
    sys.exit(main())
