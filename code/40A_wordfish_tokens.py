import re
import sys
import time
import unicodedata
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from gensim.models import Phrases
from gensim.models.phrases import Phraser

nltk.download("stopwords", quiet=True)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
DATA = REPO / "data"
PROC = DATA / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# The sentinel has to survive TOKEN_RE, so it is a bare word rather than a
# "__eos__"-style marker: the regex would strip the underscores and silently
# degrade the marker to an ordinary token that bigrams could glue across.
EOS = "eos"
TARGET_BIGRAMS = 2000
TRAIN_MIN_COUNT = 50          # a pair must occur >= this many times
TRAIN_NPMI = 0.40             # permissive training floor; raised at apply time
TOKEN_RE = re.compile(r"[a-zà-ÿ][a-zà-ÿ]+")
ACR_RE = re.compile(r"\b[A-Z]{2,6}\b")

CORPORA = {
    "us": dict(text=DATA / "us" / "campaignview_chunks_sent.feather",
               doc_col="doc_id", lang="english", mayor_only=False),
    "br": dict(text=DATA / "br" / "br_manifestos_chunks_sent.feather",
               doc_col="doc_id", lang="portuguese", mayor_only=True),
}

# Brazil keeps MAYORAL platforms only. The stage-07 estimation sample also carries
# vice-mayors, city councillors, supplementary-election mayors and a handful of
# presidential references; those are a different office and do not belong in a
# scale of mayoral platforms.
MAP_BR = DATA / "br" / "platform_party_map.feather"

# Negation seeds per language. Negations survive as unigrams but are barred from
# collocations: "not raise taxes" must not collapse into a phrase that reads as
# its opposite.
NEG_SEEDS = {
    "portuguese": {"não", "nao", "nem", "nunca", "jamais", "sem", "tampouco",
                   "nenhum", "nenhuma", "nenhuns", "nenhumas",
                   "ninguém", "ninguem", "nada"},
    "english": {"not", "no", "nor", "never", "none", "neither", "nothing",
                "nobody", "nowhere", "cannot", "cant", "dont", "doesnt",
                "didnt", "wont", "wouldnt", "shouldnt", "couldnt", "isnt",
                "arent", "wasnt", "werent", "hasnt", "havent", "hadnt",
                "aint", "without"},
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def stopword_sets(lang: str):
    """Return (non-negation stopwords to drop, tokens barred from bigrams)."""
    sw = sorted(set(nltk_stopwords.words(lang)))
    seeds = NEG_SEEDS[lang]
    folded = {strip_accents(w) for w in seeds}
    negation = set(seeds) | folded
    for w in sw:
        if strip_accents(w) in folded:
            negation.add(w)
            negation.add(strip_accents(w))
    non_neg = {w for w in sw if w not in negation}
    return sw, negation, non_neg, negation | {EOS}


def sample_docs(corpus: str) -> set:
    """Estimation sample of stage 07, restricted to mayoral platforms in Brazil."""
    npz = np.load(DATA / "irt" / f"irt_matrices_{corpus}.npz", allow_pickle=True)
    docs = set(npz["docs"].astype(str).tolist())
    if not CORPORA[corpus]["mayor_only"]:
        return docs
    m = pd.read_feather(MAP_BR, columns=["platform_id", "valid_mayor_platform"])
    mayor = set(m.loc[m.valid_mayor_platform == True, "platform_id"].astype(str))  # noqa: E712
    keep = docs & mayor
    print(f"Mayoral restriction: {len(docs):,} estimation-sample documents -> "
          f"{len(keep):,} mayoral ({len(docs - mayor):,} dropped as a different office)")
    return keep


def build(corpus: str) -> None:
    cfg = CORPORA[corpus]
    t0 = time.time()
    print(f"\n{'=' * 78}\n{corpus.upper()}\n{'=' * 78}")

    sw, negation, non_neg, block = stopword_sets(cfg["lang"])
    print(f"Stopwords: NLTK {cfg['lang']} ({len(sw)}); negations kept out of "
          f"bigrams ({len(negation)}); non-negation removed ({len(non_neg)})")
    print("Negation set:", ", ".join(sorted(negation)))

    keep_docs = sample_docs(corpus)
    df = pd.read_feather(cfg["text"], columns=[cfg["doc_col"], "chunk_text"])
    df = df.rename(columns={cfg["doc_col"]: "doc_id"})
    df["doc_id"] = df.doc_id.astype(str)
    df = df[df.doc_id.isin(keep_docs)]
    n_have = df.doc_id.nunique()
    print(f"Estimation sample: {len(keep_docs):,} docs; text found for {n_have:,} "
          f"({len(df):,} sentences, {time.time() - t0:.1f}s)")
    if n_have < len(keep_docs):
        print(f"  WARNING: {len(keep_docs) - n_have:,} estimation documents have no "
              f"text in {cfg['text'].name}")

    # ---- acronyms, harvested from the raw text before any lowercasing --------
    acr = set()
    for chunk in df.chunk_text.astype(str):
        acr.update(ACR_RE.findall(chunk))
    acr = sorted(acr)
    (PROC / f"wf_acronyms_{corpus}.txt").write_text("\n".join(acr) + "\n",
                                                    encoding="utf-8")
    print(f"Acronyms: {len(acr):,} all-caps terms -> wf_acronyms_{corpus}.txt")

    # ---- sentences -> one document string with sentinels (vectorised join) ---
    t1 = time.time()
    doc_text = (df.assign(chunk_text=df.chunk_text.astype(str).str.lower())
                  .groupby("doc_id", sort=False)["chunk_text"]
                  .agg((" " + EOS + " ").join))
    print(f"Joined to {len(doc_text):,} documents ({time.time() - t1:.1f}s)")

    # ---- tokenize each document once ----------------------------------------
    t1 = time.time()
    doc_ids = doc_text.index.to_list()
    docs = [[t for t in TOKEN_RE.findall(txt) if t == EOS or t not in non_neg]
            for txt in doc_text.to_list()]
    print(f"Tokenized {len(docs):,} documents ({time.time() - t1:.1f}s)")

    # ---- train bigram Phrases (single process), block negations + sentinel ---
    t1 = time.time()
    phrases = Phrases(docs, min_count=TRAIN_MIN_COUNT, threshold=TRAIN_NPMI,
                      scoring="npmi", connector_words=frozenset())
    export = {p: s for p, s in phrases.export_phrases().items()
              if not any(tok in block for tok in p.split("_"))}
    pairs = sorted(export.items(), key=lambda x: -x[1])
    cutoff = pairs[TARGET_BIGRAMS - 1][1] if len(pairs) > TARGET_BIGRAMS else TRAIN_NPMI
    keep = [(p, s) for p, s in pairs if s >= cutoff]
    keep_set = {p for p, _ in keep}
    print(f"Trained Phrases ({time.time() - t1:.1f}s); candidate bigrams="
          f"{len(pairs):,}; NPMI cutoff={cutoff:.3f} -> {len(keep):,} kept")
    if len(pairs) <= TARGET_BIGRAMS:
        print(f"  NOTE: fewer than {TARGET_BIGRAMS:,} candidates cleared "
              f"min_count={TRAIN_MIN_COUNT} and NPMI={TRAIN_NPMI}; all are kept")

    phrases.threshold = cutoff
    phraser = Phraser(phrases)
    phraser.phrasegrams = {p: sc for p, sc in phraser.phrasegrams.items()
                           if p in keep_set}

    # ---- apply, drop the sentinel, write both variants -----------------------
    t1 = time.time()
    big, uni, n_glued = [], [], 0
    for d in docs:
        glued = [t for t in phraser[d] if t != EOS]
        n_glued += sum("_" in t for t in glued)
        big.append(" ".join(glued))
        uni.append(" ".join(t.replace("_", " ") for t in glued))
    pd.DataFrame({"doc_id": doc_ids, "tokens": big}).to_feather(
        PROC / f"wf_tokens_{corpus}.feather")
    pd.DataFrame({"doc_id": doc_ids, "tokens": uni}).to_feather(
        PROC / f"wf_tokens_{corpus}_uni.feather")
    print(f"Applied + saved both streams ({time.time() - t1:.1f}s): "
          f"{n_glued:,} glued collocation tokens in the bigram variant")
    print(f"  -> wf_tokens_{corpus}.feather / wf_tokens_{corpus}_uni.feather")

    col = pd.DataFrame(keep, columns=["bigram", "npmi"])
    col["token1"] = col.bigram.str.split("_").str[0]
    col["token2"] = col.bigram.str.split("_").str[1]
    col.to_csv(PROC / f"wf_collocations_{corpus}.csv", index=False, encoding="utf-8")
    print(f"Saved {len(col):,} bigrams -> wf_collocations_{corpus}.csv")
    print(f"\nTop 20 bigrams by NPMI [{corpus}]:")
    print(col.head(20).to_string(index=False))
    print(f"\n[{corpus}] WALL TIME: {time.time() - t0:.1f}s")


def main() -> int:
    which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    todo = ["us", "br"] if which == "both" else [which]
    t0 = time.time()
    for corpus in todo:
        build(corpus)
    print(f"\nTOTAL WALL TIME: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
