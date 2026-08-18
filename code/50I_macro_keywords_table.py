from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = Path(os.getenv("TOPIC2IRT_ROOT", HERE.parents[0]))
DATA = REPO / "data"
OUT = REPO / "paper" / "sections" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE))
import macro_topics as mt  # noqa: E402

_spec = importlib.util.spec_from_file_location("irt_data50i", HERE / "04_irt_data.py")
irt_data = importlib.util.module_from_spec(_spec)
sys.modules["irt_data50i"] = irt_data
_spec.loader.exec_module(irt_data)

CORPUS_NAME = {"us": "United States", "br": "Brazil"}
TOP_N = 10
MAX_FEATURES = 50_000


def _tex(s: str) -> str:
    return (str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#"))


def _stopwords(corpus: str) -> frozenset:
    import nltk
    for pkg in ("stopwords",):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords as sw
    words = set(sw.words("english"))
    if corpus == "br":
        words |= set(sw.words("portuguese"))
    return frozenset(words)


def _stemmer(corpus: str):
    from nltk.stem.snowball import SnowballStemmer
    return SnowballStemmer("portuguese" if corpus == "br" else "english")


def _load_chunks(corpus: str) -> pd.DataFrame:
    """chunk_id, topic, chunk_text for exactly the estimation-sample chunks,
    with the short-chunk rule of 04_irt_data.py already applied."""
    sp = irt_data._rd(pd.read_csv, DATA / "stance" / f"stance_pred_{corpus}.csv",
                      usecols=["chunk_id"])
    idcol = "doc_id" if corpus == "us" else "platform_id"
    ct = irt_data._rd(pd.read_feather,
                      DATA / "topics" / f"topic_model_{corpus}" / "chunk_topics.feather",
                      columns=["chunk_id", idcol, "topic"])
    df = ct.merge(sp, on="chunk_id", how="inner")
    df = irt_data._reassign_short(df, corpus)
    df = df[(df.topic >= 0) | (df.topic == irt_data.SHORT_CHUNK_TOPIC)].copy()

    sent = irt_data._rd(pd.read_feather, irt_data.SENT_FILE[corpus],
                        columns=["chunk_id", "chunk_text"])
    df = df.merge(sent, on="chunk_id", how="left")
    missing = df["chunk_text"].isna().sum()
    if missing:
        raise ValueError(f"{corpus}: {missing} chunk(s) missing chunk_text")
    return df[["chunk_id", "topic", "chunk_text"]]


def _ctfidf(df: pd.DataFrame, corpus: str, codes: list) -> dict:
    """Top TOP_N c-TF-IDF terms per supertopic code, {code: [(word, weight), ...]}.

    Ranking runs on stems, so "land" and "lands" contribute to one score instead of
    splitting it across two weaker entries; the word shown for a stem is whichever of
    its surface forms is most frequent *in that class*, so the display is always a
    real word, never the stem itself, which is often not one (Porter/Snowball strip
    to a root, not to a dictionary form).
    """
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import CountVectorizer

    cmap = mt.crosswalk(corpus)
    df = df.copy()
    df["macro"] = df["topic"].astype(int).map(cmap)
    missing = df["macro"].isna().sum()
    if missing:
        raise KeyError(f"{corpus}: {missing} chunks with a topic absent from the crosswalk")

    cv = CountVectorizer(lowercase=True, token_pattern=r"\b[^\W\d_]{3,}\b",
                         stop_words=list(_stopwords(corpus)), max_features=MAX_FEATURES)
    X = cv.fit_transform(df["chunk_text"].astype(str))
    vocab = np.array(cv.get_feature_names_out())

    code_ix = {c: i for i, c in enumerate(codes)}
    rows = df["macro"].map(code_ix).to_numpy()
    keep = ~pd.isna(rows)
    rows = rows[keep].astype(int)
    Xk = X[np.flatnonzero(keep)]

    G = csr_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))),
                   shape=(len(codes), Xk.shape[0]))
    word_counts = np.asarray((G @ Xk).todense())            # classes x words

    stemmer = _stemmer(corpus)
    stems = np.array([stemmer.stem(w) for w in vocab])
    uniq_stems, stem_ix = np.unique(stems, return_inverse=True)
    n_stems = len(uniq_stems)
    M = csr_matrix((np.ones(len(stem_ix)), (np.arange(len(stem_ix)), stem_ix)),
                   shape=(len(vocab), n_stems))
    class_counts = np.asarray((csr_matrix(word_counts) @ M).todense())   # classes x stems
    tf_t = class_counts.sum(axis=0)                          # stems
    class_words = word_counts.sum(axis=1)                    # classes (unaffected by stemming)
    A = tf_t.sum() / len(codes)

    stem_members = {}
    for s in range(n_stems):
        stem_members[s] = np.flatnonzero(stem_ix == s)

    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        idf = np.log(1.0 + A / np.where(tf_t > 0, tf_t, np.nan))
    for c, i in code_ix.items():
        cw = class_words[i]
        tf = class_counts[i] / cw if cw > 0 else class_counts[i]
        w = tf * idf
        w = np.nan_to_num(w, nan=0.0)
        top = np.argsort(-w)[:TOP_N]
        terms = []
        for s in top:
            if w[s] <= 0:
                continue
            members = stem_members[s]
            rep = members[np.argmax(word_counts[i, members])]
            terms.append((vocab[rep], float(w[s])))
        out[c] = terms
    return out


def build(corpus: str) -> str:
    df = _load_chunks(corpus)
    codes = mt.classes_present(corpus)
    kw = _ctfidf(df, corpus, codes)

    blocks: dict = {}
    for code in codes:
        blocks.setdefault(mt.BLOCK[code], []).append(code)

    body = []
    first = True
    for block, block_codes in blocks.items():
        if not first:
            body.append(r"\addlinespace")
        first = False
        body.append(rf"\multicolumn{{2}}{{@{{}}l}}{{\emph{{{_tex(block)}}}}} \\*")
        for code in block_codes:
            name = _tex(mt.LABEL[code])
            if code == mt.BOILERPLATE_CODE:
                name += r"$^{\dagger}$"
            terms = ", ".join(t for t, _ in kw.get(code, []))
            body.append(rf"{name} & {_tex(terms)} \\")

    note = (
        rf"\item The {TOP_N} strongest class-based TF-IDF terms (c-TF-IDF) of each "
        rf"supertopic in the {CORPUS_NAME[corpus]} corpus, computed directly over the "
        rf"supertopic groups on the estimation-sample sentences -- the same sentences and the "
        rf"same short-sentence rule that feed the main fit -- rather than pooled from the "
        rf"subtopics' own key terms. Unigrams of at least three letters, no digits, "
        rf"stop words removed, ranked by stem so a plural or other inflected form does "
        rf"not split a term's score; the word shown is the most frequent surface form of "
        rf"its stem. $^{{\dagger}}$Boilerplate pools every format and register subtopic "
        rf"together with every sentence of five words or fewer, whatever its original "
        rf"topic, so its own terms are dominated by document structure rather than "
        rf"policy content."
    )

    head = [rf"\caption{{Top {TOP_N} Keywords per Supertopic, {CORPUS_NAME[corpus]}}}",
            rf"\label{{tab:macro-keywords-{corpus}}} \\",
            r"\toprule", r"Supertopic & Top keywords \\", r"\midrule",
            r"\endfirsthead",
            rf"\caption[]{{Top {TOP_N} Keywords per Supertopic, "
            rf"{CORPUS_NAME[corpus]} \emph{{(continued)}}}} \\",
            r"\toprule", r"Supertopic & Top keywords \\", r"\midrule",
            r"\endhead",
            r"\midrule",
            rf"\multicolumn{{2}}{{@{{}}p{{\dimexpr\linewidth-8pt\relax}}@{{}}}}"
            rf"{{\scriptsize {note[6:]}}} \\",
            r"\endlastfoot"]
    return "\n".join([
        r"% GENERATED by code/50I_macro_keywords_table.py -- do not edit by hand.",
        r"\begingroup", r"\scriptsize", r"\singlespacing",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\setlength{\LTcapwidth}{\textwidth}",
        r"\begin{longtable}{@{}p{3.2cm}p{11.3cm}@{}}",
        *head, *body,
        r"\end{longtable}", r"\endgroup", "",
    ])


def main():
    for c in ("us", "br"):
        p = OUT / f"macro_keywords_{c}.tex"
        p.write_text(build(c), encoding="utf-8")
        print(f"wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
