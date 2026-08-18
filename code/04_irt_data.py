from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = DATA / "irt"
OUT.mkdir(parents=True, exist_ok=True)

SVAL = {"Left": -1.0, "Right": 1.0, "Neutral": 0.0}
MIN_CHUNKS = 10
BR_LEFT_CUT = 4.77        # Bolognesi expert score of REDE; at or below = left bloc

# Non-platform screen (Brazil).  A minority of the filings recovered from the electoral
# authority are not platforms at all: a power of attorney, a court filing, a clearance
# certificate, a filing receipt, a consent form, party convention minutes, a campaign
# workbook, a printout of the candidate's registry page, a cover page whose body never
# arrived.  The exclusion is the list below, and every document on it was read whole and
# confirmed; the screen in code/nonplatform.py found them to be read.
NONPLATFORM_FILE = DATA / "br" / "nonplatform_br.csv"

# A chunk this short or shorter carries no usable signal on its own, whatever topic the
# model gave it, and is reassigned to this sentinel fine topic instead - grouped into the
# BOI macro class by the crosswalk like any other topic (code/prompts/topic_macro_crosswalk.xlsx).
SHORT_CHUNK_MAX_WORDS = 6
SHORT_CHUNK_TOPIC = -2
SHORT_CHUNK_LABEL = f"short chunk (<= {SHORT_CHUNK_MAX_WORDS} words)"

SENT_FILE = {"us": DATA / "us" / "campaignview_chunks_sent.feather",
             "br": DATA / "br" / "br_manifestos_chunks_sent.feather"}


def _reassign_short(df, corpus):
    """Route every chunk at or under the word floor to the SHORT_CHUNK_TOPIC sentinel,
    regardless of its original topic - including chunks the topic model called noise."""
    wc = _rd(pd.read_feather, SENT_FILE[corpus], columns=["chunk_id", "word_count"])
    df = df.merge(wc, on="chunk_id", how="left")
    missing = df["word_count"].isna().sum()
    if missing:
        raise ValueError(f"{corpus}: {missing} chunk_id(s) absent from {SENT_FILE[corpus].name}")
    short = df["word_count"] <= SHORT_CHUNK_MAX_WORDS
    n_short = int(short.sum())
    df.loc[short, "topic"] = SHORT_CHUNK_TOPIC
    print(f"[{corpus}] chunks reassigned to boilerplate (<= {SHORT_CHUNK_MAX_WORDS} words): "
          f"{n_short:,} of {len(df):,} ({100*n_short/len(df):.2f}%)")
    return df.drop(columns="word_count")


def _rd(fn, path, **kw):
    """Read with retries: Google Drive occasionally reports a transient lock."""
    last = None
    for _ in range(8):
        try:
            return fn(path, **kw)
        except (FileNotFoundError, OSError) as exc:
            last = exc
            time.sleep(1.5)
    raise last


def _labels(corpus: str) -> dict:
    """topic id -> its four leading key terms."""
    ba = _rd(pd.read_excel,
             DATA / "topics" / f"topic_model_{corpus}" / "topic_info_before_after.xlsx")

    def lab(row):
        kw = str(row.get("keywords_after", "") or "")
        if not kw or kw.lower() == "nan":
            kw = str(row.get("keywords_before", "") or "")
        return ", ".join(kw.split(", ")[:4]) if kw and kw.lower() != "nan" else "?"

    return {int(r.topic): lab(r) for _, r in ba.iterrows()}


def _drop_nonplatform(df, corpus):
    """Exclude the filings that are not campaign platforms.

    Returns the surviving rows and the excluded document ids, which the funnel table
    reports as their own line.  The excluded ids are those on the confirmed list that
    are still in the corpus at this point; most of the list is already gone, because a
    filing is short and the corpus has been through the chunking stages.
    """
    conf = _rd(pd.read_csv, NONPLATFORM_FILE, dtype={"doc": str})
    listed = set(conf.doc)
    here = set(df.doc_id.astype(str))
    bad = sorted(listed & here)
    # How many of them the length floor would NOT have caught anyway, which is the
    # number of documents the exclusion actually keeps out of the estimation sample.
    size = df.doc_id.astype(str).value_counts()
    scalable = [d for d in bad if size.get(d, 0) >= MIN_CHUNKS]
    print(f"[{corpus}] documents excluded as non-platform filings: {len(bad):,} "
          f"of {df.doc_id.nunique():,} here, {len(listed):,} confirmed in all; "
          f"{len(scalable):,} of them had {MIN_CHUNKS}+ chunks and would otherwise have been scaled")
    return (df[~df.doc_id.isin(bad)], np.array(bad, dtype=str),
            np.array(scalable, dtype=str))


def _drop_short(df, corpus):
    size = df.groupby("doc_id")["doc_id"].transform("size")
    keep = size >= MIN_CHUNKS
    n_drop = df.loc[~keep, "doc_id"].nunique()
    print(f"[{corpus}] documents with < {MIN_CHUNKS} chunks dropped: {n_drop:,} "
          f"of {df.doc_id.nunique():,}")
    return df[keep].copy()


def _assemble(df, corpus):
    """Shared (Y, S) assembly once df has doc_id, topic, stance, party."""
    df["sval"] = df.stance.map(SVAL)
    docs, d_ix = np.unique(df.doc_id.to_numpy(), return_inverse=True)
    tops, t_ix = np.unique(df.topic.to_numpy(), return_inverse=True)
    N, T = len(docs), len(tops)
    Y = coo_matrix((np.ones(len(d_ix)), (d_ix, t_ix)), shape=(N, T)).tocsr()
    Y.sum_duplicates()
    S = coo_matrix((df.sval.to_numpy(float), (d_ix, t_ix)), shape=(N, T)).tocsr()
    S.sum_duplicates()
    Y.sort_indices()
    S.sort_indices()
    party = (df.groupby("doc_id")["party"].first().reindex(docs).to_numpy().astype(str))
    docfreq = np.asarray((Y > 0).sum(0)).ravel().astype(int)
    labels = _labels(corpus)
    kw = np.array([labels.get(int(t), SHORT_CHUNK_LABEL if t == SHORT_CHUNK_TOPIC else "?")
                   for t in tops])
    return dict(Y=Y, S=S, tops=tops, kw=kw, party=party, docfreq=docfreq,
                N=N, T=T, ncells=int(Y.nnz), docs=docs)


def build_us():
    sp = _rd(pd.read_csv, DATA / "stance" / "stance_pred_us.csv",
             usecols=["chunk_id", "stance"])
    ct = _rd(pd.read_feather, DATA / "topics" / "topic_model_us" / "chunk_topics.feather",
             columns=["chunk_id", "doc_id", "topic"])
    df = ct.merge(sp, on="chunk_id", how="inner")
    df = _reassign_short(df, "us")
    df = df[(df.topic >= 0) | (df.topic == SHORT_CHUNK_TOPIC)].copy()
    df["party"] = df.doc_id.str.split("|").str[3]
    df = _drop_short(df, "us")

    D = _assemble(df, "us")
    sc = _rd(pd.read_csv, DATA / "us" / "campaignview_with_scores.csv",
             dtype={"cd": str, "year": str})
    sc["doc_id"] = (sc.candidate_webname.astype(str) + "|" + sc.state_postal.astype(str)
                    + "|" + sc.cd.astype(str) + "|" + sc.cand_party.astype(str)
                    + "|" + sc.year.astype(str))
    e = sc[["doc_id", "cfscore", "nominate_dim1"]].drop_duplicates("doc_id").set_index("doc_id")
    D["cfscore"] = e.cfscore.reindex(D["docs"]).to_numpy(float)
    D["nominate"] = e.nominate_dim1.reindex(D["docs"]).to_numpy(float)
    D["kind"] = "us"
    print(f"[us] documents {D['N']:,} | topics {D['T']} | cells {D['ncells']:,} | "
          f"cfscore {np.isfinite(D['cfscore']).sum():,} | "
          f"nominate {np.isfinite(D['nominate']).sum():,}")
    return D


def build_br():
    exp = _rd(pd.read_csv, DATA / "raw" / "party_scores_bolognesi2023.csv")
    ESC = dict(zip(exp.SG_PARTIDO.str.upper(), exp.party_mean_expert.astype(float)))
    sp = _rd(pd.read_csv, DATA / "stance" / "stance_pred_br.csv",
             usecols=["chunk_id", "stance"])
    ct = _rd(pd.read_feather, DATA / "topics" / "topic_model_br" / "chunk_topics.feather",
             columns=["chunk_id", "platform_id", "topic"])
    df = ct.merge(sp, on="chunk_id", how="inner")
    df = _reassign_short(df, "br")
    df = df[(df.topic >= 0) | (df.topic == SHORT_CHUNK_TOPIC)].rename(
        columns={"platform_id": "doc_id"})
    df, nonplatform, np_scalable = _drop_nonplatform(df, "br")

    # LEFT join: a platform with no party mapping is still scaled, it merely has
    # no expert benchmark.  Party never selects the estimation sample.
    pmap = _rd(pd.read_feather, DATA / "br" / "platform_party_map.feather",
               columns=["platform_id", "party"])
    pmap["party"] = pmap.party.astype(str).str.upper()
    df = df.merge(pmap.rename(columns={"platform_id": "doc_id"}), on="doc_id", how="left")
    df["party"] = df.party.fillna("UNKNOWN")
    df = _drop_short(df, "br")

    D = _assemble(df, "br")
    D["ext"] = np.array([ESC.get(p, np.nan) for p in D["party"]])   # NaN where absent
    D["ESC"] = ESC
    D["nonplatform"] = nonplatform
    D["nonplatform_scalable"] = np_scalable
    D["kind"] = "br"
    print(f"[br] documents {D['N']:,} | topics {D['T']} | cells {D['ncells']:,} | "
          f"with expert score {int(np.isfinite(D['ext']).sum()):,} | "
          f"parties {len(set(D['party']))}")
    return D


BUILDERS = {"us": build_us, "br": build_br}


def cache_path(corpus):
    return OUT / f"irt_matrices_{corpus}.npz"


def _save(path, D):
    Y, S = D["Y"], D["S"]
    save = dict(
        Y_data=Y.data.astype(np.int16), Y_indices=Y.indices.astype(np.int16), Y_indptr=Y.indptr,
        S_data=S.data.astype(np.int16), S_indices=S.indices.astype(np.int16), S_indptr=S.indptr,
        shape=np.array(Y.shape), kind=np.array(D["kind"]),
        tops=D["tops"], kw=D["kw"], party=D["party"], docfreq=D["docfreq"],
        docs=D["docs"].astype(str),
        N=D["N"], T=D["T"], ncells=D["ncells"], min_chunks=MIN_CHUNKS,
    )
    if D["kind"] == "br":
        save["ext"] = D["ext"]
        save["ESC_keys"] = np.array(list(D["ESC"].keys()))
        save["ESC_vals"] = np.array(list(D["ESC"].values()), float)
        save["nonplatform"] = D["nonplatform"]
        save["nonplatform_scalable"] = D["nonplatform_scalable"]
    else:
        save["cfscore"] = D["cfscore"]
        save["nominate"] = D["nominate"]
    np.savez_compressed(path, **save)


def _load(path):
    z = np.load(path, allow_pickle=True)
    sh = tuple(z["shape"])
    Y = csr_matrix((z["Y_data"].astype(np.float64), z["Y_indices"].astype(np.int32),
                    z["Y_indptr"].astype(np.int32)), shape=sh)
    S = csr_matrix((z["S_data"].astype(np.float64), z["S_indices"].astype(np.int32),
                    z["S_indptr"].astype(np.int32)), shape=sh)
    kind = str(z["kind"])
    D = dict(Y=Y, S=S, tops=z["tops"], kw=z["kw"], party=z["party"],
             docfreq=z["docfreq"], docs=z["docs"], N=int(z["N"]), T=int(z["T"]),
             ncells=int(z["ncells"]), kind=kind)
    if kind == "br":
        D["ext"] = z["ext"]
        D["ESC"] = dict(zip(z["ESC_keys"], z["ESC_vals"].astype(float)))
        D["nonplatform"] = (z["nonplatform"] if "nonplatform" in z.files
                            else np.array([], dtype=str))
        D["nonplatform_scalable"] = (z["nonplatform_scalable"]
                                     if "nonplatform_scalable" in z.files
                                     else np.array([], dtype=str))
    else:
        D["cfscore"] = z["cfscore"]
        D["nominate"] = z["nominate"]
    return D


def load_matrices(corpus: str, force: bool = False, verbose: bool = True):
    corpus = corpus.lower()
    if corpus not in BUILDERS:
        raise KeyError(f"unknown corpus {corpus!r}; known: {sorted(BUILDERS)}")
    path = cache_path(corpus)
    if path.exists() and not force:
        D = _load(path)
        if verbose:
            print(f"[{corpus}] cached (Y,S): N={D['N']:,} T={D['T']} "
                  f"cells={D['ncells']:,}")
        return D
    D = BUILDERS[corpus]()
    _save(path, D)
    if verbose:
        print(f"[{corpus}] wrote {path.relative_to(REPO)} "
              f"({path.stat().st_size/1e6:.1f} MB)")
    return D


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="*", default=["us", "br"])
    ap.add_argument("--force", action="store_true", help="rebuild from per-sentence files")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from datetime import datetime
    from timing import log_run

    started, timing, scale = datetime.now(), {}, {}
    for c in a.corpora:
        c = c.lower()
        t0 = time.perf_counter()
        D = load_matrices(c, force=a.force)
        timing[c] = {"1 build": time.perf_counter() - t0}
        scale[c] = {"n_docs": int(D["N"])}       # n_chunks is not a quantity this stage carries
    print(log_run("04 irt_data", timing, scale=scale, started=started,
                  note=("Rebuild from the per-sentence files." if a.force else
                        "Cached matrices reused where present; pass --force to rebuild.")))
