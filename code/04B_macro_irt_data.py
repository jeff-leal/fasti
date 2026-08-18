from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = Path(os.getenv("TOPIC2IRT_ROOT", HERE.parents[0]))
IRT = REPO / "data" / "irt"
IRT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE))
import macro_topics as mt  # noqa: E402

_spec = importlib.util.spec_from_file_location("irt_data07", HERE / "04_irt_data.py")
irt_data = importlib.util.module_from_spec(_spec)
sys.modules["irt_data07"] = irt_data
_spec.loader.exec_module(irt_data)

LEVELS = ("macro", "micro_nobp")


def cache_path(level: str, corpus: str) -> Path:
    return IRT / f"irt_matrices_{level}_{corpus}.npz"


def build_macro(D, corpus):
    """Fine columns summed within each macro class, boilerplate included."""
    Y, S, codes = mt.aggregate(D["Y"], D["S"], D["tops"], corpus)
    labels = mt.macro_labels(codes)
    blocks = np.array([mt.BLOCK[str(c)] for c in codes], dtype=object)
    n_fine = np.array([sum(1 for v in mt.crosswalk(corpus).values() if v == c)
                       for c in codes], int)
    return dict(Y=Y, S=S, tops=np.asarray(codes, dtype=object), kw=labels,
                block=blocks, n_fine=n_fine)


def build_micro_nobp(D, corpus):
    """Fine columns with the boilerplate topics removed."""
    tops = np.asarray(D["tops"]).astype(int)
    keep = ~np.isin(tops, sorted(mt.boilerplate_topics(corpus)))
    Y, S = csr_matrix(D["Y"][:, keep]), csr_matrix(D["S"][:, keep])
    Y.eliminate_zeros()
    Y.sort_indices()
    S.sort_indices()
    labels = mt.topic_labels(corpus)
    return dict(Y=Y, S=S, tops=tops[keep],
                kw=np.array([labels.get(int(t), f"Topic {t}") for t in tops[keep]],
                            dtype=object),
                block=np.array([mt.LABEL[mt.crosswalk(corpus)[int(t)]]
                                for t in tops[keep]], dtype=object),
                n_fine=np.ones(int(keep.sum()), int))


BUILDERS = {"macro": build_macro, "micro_nobp": build_micro_nobp}


def build(level: str, corpus: str, verbose: bool = True) -> dict:
    D = irt_data.load_matrices(corpus, verbose=False)
    P = BUILDERS[level](D, corpus)
    Y, S = P["Y"], P["S"]
    out = dict(D)                       # documents, party, benchmarks all carry over
    out.update(Y=Y, S=S, tops=P["tops"], kw=P["kw"], block=P["block"],
               n_fine=P["n_fine"], level=level,
               docfreq=np.asarray((Y > 0).sum(0)).ravel().astype(int),
               T=Y.shape[1], ncells=int(Y.nnz))
    if verbose:
        print(f"[{corpus}/{level}] N={out['N']:,} K={out['T']} cells={out['ncells']:,} "
              f"(fine: T={D['T']} cells={D['Y'].nnz:,})")
    return out


def _save(path: Path, D: dict) -> None:
    Y, S = D["Y"], D["S"]
    save = dict(
        Y_data=Y.data.astype(np.int32), Y_indices=Y.indices.astype(np.int32),
        Y_indptr=Y.indptr.astype(np.int64),
        S_data=S.data.astype(np.int32), S_indices=S.indices.astype(np.int32),
        S_indptr=S.indptr.astype(np.int64),
        shape=np.array(Y.shape), kind=np.array(D["kind"]), level=np.array(D["level"]),
        tops=np.asarray(D["tops"]).astype(str), kw=np.asarray(D["kw"]).astype(str),
        block=np.asarray(D["block"]).astype(str), n_fine=D["n_fine"],
        party=D["party"], docfreq=D["docfreq"], docs=np.asarray(D["docs"]).astype(str),
        N=D["N"], T=D["T"], ncells=D["ncells"],
    )
    if D["kind"] == "br":
        save["ext"] = D["ext"]
        save["ESC_keys"] = np.array(list(D["ESC"].keys()))
        save["ESC_vals"] = np.array(list(D["ESC"].values()), float)
    else:
        save["cfscore"] = D["cfscore"]
        save["nominate"] = D["nominate"]
    np.savez_compressed(path, **save)


def _load(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    sh = tuple(z["shape"])
    Y = csr_matrix((z["Y_data"].astype(np.float64), z["Y_indices"].astype(np.int32),
                    z["Y_indptr"].astype(np.int32)), shape=sh)
    S = csr_matrix((z["S_data"].astype(np.float64), z["S_indices"].astype(np.int32),
                    z["S_indptr"].astype(np.int32)), shape=sh)
    kind = str(z["kind"])
    D = dict(Y=Y, S=S, tops=z["tops"], kw=z["kw"], block=z["block"],
             n_fine=z["n_fine"], party=z["party"], docfreq=z["docfreq"],
             docs=z["docs"], N=int(z["N"]), T=int(z["T"]), ncells=int(z["ncells"]),
             kind=kind, level=str(z["level"]))
    if kind == "br":
        D["ext"] = z["ext"]
        D["ESC"] = dict(zip(z["ESC_keys"], z["ESC_vals"].astype(float)))
    else:
        D["cfscore"] = z["cfscore"]
        D["nominate"] = z["nominate"]
    return D


def load_matrices(level: str, corpus: str, force: bool = False,
                  verbose: bool = True) -> dict:
    """The (Y, S) matrices for one column scheme, cached beside stage 04's."""
    level, corpus = level.lower(), corpus.lower()
    if level == "micro":
        return irt_data.load_matrices(corpus, verbose=verbose)
    if level not in BUILDERS:
        raise KeyError(f"unknown level {level!r}; known: {('micro',) + LEVELS}")
    path = cache_path(level, corpus)
    if path.exists() and not force:
        D = _load(path)
        if verbose:
            print(f"[{corpus}/{level}] cached (Y,S): N={D['N']:,} K={D['T']} "
                  f"cells={D['ncells']:,}")
        return D
    D = build(level, corpus, verbose=verbose)
    _save(path, D)
    if verbose:
        print(f"[{corpus}/{level}] wrote {path.relative_to(REPO)} "
              f"({path.stat().st_size/1e6:.1f} MB)")
    return D


#: the paper's main column scheme. Every table, figure and reported number
#: in the body reads this one, so there is a single place to change it.
MAIN_LEVEL = "macro"


def load_main(corpus: str, variant: str = "godambe"):
    """The main fit and its matrices, aligned to each other.

    Dropping the boilerplate column empties a handful of documents, so the fit
    is shorter than the matrices it was built from.  This returns ``(F, D)``
    with ``D`` cut to the documents the fit kept and its columns ordered as the
    fit ordered them, which is what every consumer needs and none should
    reimplement.
    """
    D = dict(load_matrices(MAIN_LEVEL, corpus, verbose=False))
    F = np.load(IRT / f"joint_irt_fit_{MAIN_LEVEL}_{corpus}_{variant}.npz",
                allow_pickle=True)
    if "docs" in F.files:
        keep = np.isin(np.asarray(D["docs"]).astype(str), F["docs"].astype(str))
        n = len(D["docs"])
        for k, v in list(D.items()):
            if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
                D[k] = v[keep]
        D["Y"], D["S"] = D["Y"][keep].tocsr(), D["S"][keep].tocsr()
        D["N"] = int(keep.sum())
    if "columns" in F.files:
        cols = np.asarray(F["columns"]).astype(str)
        have = np.asarray(D["tops"]).astype(str)
        take = np.array([np.flatnonzero(have == c)[0] for c in cols])
        for k in ("tops", "kw", "block", "n_fine", "docfreq"):
            if k in D and isinstance(D[k], np.ndarray) and len(D[k]) == len(have):
                D[k] = np.asarray(D[k])[take]
        D["Y"], D["S"] = D["Y"][:, take].tocsr(), D["S"][:, take].tocsr()
        D["T"] = len(cols)
    D["docfreq"] = np.asarray((D["Y"] > 0).sum(0)).ravel().astype(int)
    D["level"] = MAIN_LEVEL
    return F, D


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="*", default=["us", "br"])
    ap.add_argument("--levels", nargs="*", default=list(LEVELS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from datetime import datetime
    from timing import log_run
    import time as _time

    started, timing, scale = datetime.now(), {}, {}
    for c in [x.lower() for x in a.corpora]:
        timing[c] = {}
        for lv in a.levels:
            _t0 = _time.perf_counter()
            D = load_matrices(lv, c, force=a.force)
            timing[c][f"1 {lv}"] = _time.perf_counter() - _t0
            scale[c] = {"n_docs": int(D["N"])}
            if lv == "macro":
                bp = [i for i, t in enumerate(D["tops"])
                      if str(t) == mt.BOILERPLATE_CODE]
                share = (float(D["Y"][:, bp].sum()) / float(D["Y"].sum()) * 100
                         if bp else 0.0)
                print(f"    boilerplate column present: {bool(bp)}  "
                      f"({share:.1f}% of sentences)")

    print(log_run("04B macro_irt_data", timing, scale=scale, started=started,
                  note=f"levels: {', '.join(a.levels)}. Grouping from topic_macro_crosswalk.xlsx."))
