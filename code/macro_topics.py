from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
REPO = Path(os.getenv("TOPIC2IRT_ROOT", HERE.parents[0]))

sys.path.insert(0, str(HERE))
import topic_codes  # noqa: E402

BOOK = REPO / "code" / "prompts" / "topic_macro_crosswalk.xlsx"
BOILERPLATE_CODE = "BOI"
RESIDUAL_CODE = "OTH"

_REQUIRED = ["Corpus", "Topic", "Topic Label", "Macro Code", "Macro Label",
             "Boilerplate", "Key Terms"]


def _truthy(s) -> bool:
    return str(s).strip().lower() in {"true", "1", "yes", "y", "x"}


def _read() -> pd.DataFrame:
    if not BOOK.exists():
        raise FileNotFoundError(
            f"{BOOK} is missing; it is the crosswalk's source of truth")
    d = pd.read_excel(BOOK)
    d.columns = [str(c).strip() for c in d.columns]
    missing = [c for c in _REQUIRED if c not in d.columns]
    if missing:
        raise KeyError(f"{BOOK.name}: missing column(s) {missing}")
    d["Corpus"] = d["Corpus"].astype(str).str.strip().str.upper()
    d["Topic"] = d["Topic"].astype(int)
    d["Macro Code"] = d["Macro Code"].astype(str).str.strip()
    d["Macro Label"] = d["Macro Label"].astype(str).str.strip()
    d["Topic Label"] = d["Topic Label"].astype(str).str.strip()
    d["Boilerplate"] = d["Boilerplate"].map(_truthy)

    # The loader refuses rather than guesses: an unknown class, a duplicate
    # topic, or a topic flagged boilerplate outside the boilerplate class each
    # mean the workbook and the codebook have drifted apart.
    unknown = sorted(set(d["Macro Code"]) - set(topic_codes.CODES))
    if unknown:
        raise KeyError(f"{BOOK.name}: macro code(s) not in the codebook: {unknown}")
    dup = d.duplicated(["Corpus", "Topic"])
    if dup.any():
        raise ValueError(f"{BOOK.name}: duplicate (Corpus, Topic) rows:\n{d[dup]}")
    stray = d[d["Boilerplate"] & (d["Macro Code"] != BOILERPLATE_CODE)]
    if len(stray):
        raise ValueError(f"{BOOK.name}: flagged boilerplate outside "
                         f"{BOILERPLATE_CODE}:\n{stray[['Corpus', 'Topic', 'Macro Code']]}")
    return d


CROSSWALK = _read()

#: codebook order, so every corpus lays its classes out the same way
CLASS_ORDER = list(topic_codes.CODES)
LABEL = dict(topic_codes.LABEL)
PROMPT_LABEL = dict(topic_codes.PROMPT_LABEL)
BLOCK = dict(topic_codes.BLOCK)

CORPORA = ("us", "br")


def _slice(corpus: str) -> pd.DataFrame:
    return CROSSWALK[CROSSWALK.Corpus == corpus.upper()]


def crosswalk(corpus: str) -> dict:
    """fine topic id -> macro class code."""
    s = _slice(corpus)
    if s.empty:
        raise KeyError(f"{BOOK.name}: no rows for corpus {corpus!r}")
    return dict(zip(s["Topic"].astype(int), s["Macro Code"]))


def topic_labels(corpus: str) -> dict:
    """fine topic id -> the label the author assigned it."""
    s = _slice(corpus)
    return dict(zip(s["Topic"].astype(int), s["Topic Label"]))


def key_terms(corpus: str) -> dict:
    """fine topic id -> its c-TF-IDF key terms, as one comma-separated string."""
    s = _slice(corpus)
    return dict(zip(s["Topic"].astype(int), s["Key Terms"].astype(str)))


def boilerplate_topics(corpus: str) -> frozenset:
    """The fine topics belonging to the boilerplate class.

    Used at the *fine* level, where dropping them is its own decision, taken
    independently of the macro one.
    """
    s = _slice(corpus)
    return frozenset(int(t) for t, c in zip(s["Topic"], s["Macro Code"])
                     if c == BOILERPLATE_CODE)


def classes_present(corpus: str, tops=None) -> list:
    """Macro codes carrying at least one fine topic, in codebook order.

    Boilerplate is a class like any other here.  Dropping it is a fit-time
    decision, not a property of the crosswalk.
    """
    cmap = crosswalk(corpus)
    have = set(cmap[int(t)] for t in (cmap if tops is None else tops))
    return [c for c in CLASS_ORDER if c in have]


def aggregate(Y, S, tops, corpus: str):
    """Sum the columns of (Y, S) within each macro class.

    Every fine topic is grouped, boilerplate included, so the returned matrices
    carry a boilerplate column wherever the corpus has one.  Returns
    ``(Y, S, codes)``; the documents are untouched.
    """
    cmap = crosswalk(corpus)
    tops = np.asarray(tops).astype(int)
    missing = sorted(set(tops.tolist()) - set(cmap))
    if missing:
        raise KeyError(f"{corpus}: fine topics absent from the crosswalk: {missing}")

    codes = classes_present(corpus, tops)
    col = {c: j for j, c in enumerate(codes)}
    A = csr_matrix((np.ones(len(tops)),
                    (np.arange(len(tops)), [col[cmap[int(t)]] for t in tops])),
                   shape=(len(tops), len(codes)))
    Ya = csr_matrix(Y @ A)
    Sa = csr_matrix(S @ A)
    Ya.eliminate_zeros()
    Ya.sort_indices()
    Sa.sort_indices()
    return Ya, Sa, np.array(codes, dtype=object)


def drop_columns(Y, S, codes, drop=(BOILERPLATE_CODE,)):
    """Remove whole macro columns before a fit, and say which went.

    This is the fit-time decision the module deliberately keeps separate from
    the grouping above: the paper's main fit drops the boilerplate column, but
    the macro matrices themselves are built with it.
    """
    codes = np.asarray(codes, dtype=object)
    drop = {str(c) for c in drop}
    keep = np.array([str(c) not in drop for c in codes])
    Yk, Sk = csr_matrix(Y[:, keep]), csr_matrix(S[:, keep])
    Yk.eliminate_zeros()
    Yk.sort_indices()
    Sk.sort_indices()
    return Yk, Sk, codes[keep], [str(c) for c in codes[~keep]]


def macro_labels(codes) -> np.ndarray:
    """Display labels for a list of macro codes."""
    return np.array([LABEL[str(c)] for c in codes], dtype=object)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(f"{BOOK.name}: {len(CROSSWALK)} fine topics over "
          f"{len(set(CROSSWALK['Macro Code']))} classes\n")
    for c in CORPORA:
        s = _slice(c)
        present = classes_present(c)
        print(f"[{c}] {len(s)} fine topics -> {len(present)} classes, "
              f"boilerplate among them ({len(boilerplate_topics(c))} fine topics)")
        for code in present:
            n = int((s["Macro Code"] == code).sum())
            print(f"    {code}  {LABEL[code]:<32} {n:>3} topics   [{BLOCK[code]}]")
        print()
