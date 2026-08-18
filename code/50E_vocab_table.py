from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
PROC = REPO / "data" / "processed"
IRT = REPO / "data" / "irt"
OUT = REPO / "paper" / "sections" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

CORPUS_LABEL = {"us": "United States", "br": "Brazil"}
VARIANT_LABEL = {"unigram": "Unigrams", "bigram": "With bigrams"}


def stream_stats(corpus: str, variant: str) -> tuple[int, int]:
    suf = "_uni" if variant == "unigram" else ""
    df = pd.read_feather(PROC / f"wf_tokens_{corpus}{suf}.feather")
    freq = Counter()
    n_occ = 0
    for t in df.tokens:
        toks = t.split()
        n_occ += len(toks)
        freq.update(toks)
    return n_occ, len(freq)


def main() -> int:
    wf = pd.read_csv(IRT / "wordfish_ca_meta.csv")
    d2v = pd.read_csv(IRT / "doc2vec_meta.csv")

    rows = []
    for corpus in ("us", "br"):
        for variant in ("unigram", "bigram"):
            occ, types = stream_stats(corpus, variant)
            w = wf[(wf.corpus == corpus) & (wf.variant == variant)
                   & (wf.model == "wordfish")]
            d = d2v[(d2v.corpus == corpus) & (d2v.variant == variant)]
            rows.append(dict(
                corpus=corpus, variant=variant, occ=occ, types=types,
                scaling=int(w.full_vocab.iloc[0]) if len(w) else None,
                fitted=int(w.vocab.iloc[0]) if len(w) else None,
                emb=int(d.kept_vocab.iloc[0]) if len(d) else None,
                ndoc=int(w.ndoc.iloc[0]) if len(w) else None))
            print(f"  {corpus} {variant:8s} occ={occ:>12,} types={types:>8,} "
                  f"scaling={rows[-1]['scaling']} fitted={rows[-1]['fitted']} "
                  f"emb={rows[-1]['emb']}")

    def num(v):
        return f"{v:,}" if v is not None else "---"

    lines = [
        # "!" keeps the float near its subsection instead of being deferred to
        # the end of the document once the note is counted in.
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Vocabulary Size Under Each Specification}",
        r"\label{tab:vocab-sizes}",
        r"\begin{threeparttable}",
        r"\small",
        # Trim the column padding rather than the type: at the default 6pt the
        # seven columns run into the margin.
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r" & & & & \multicolumn{2}{c}{Word-count scaling} & Embeddings \\",
        r"\cmidrule(lr){5-6}\cmidrule(l){7-7}",
        r"Corpus & Variant & Tokens (millions) & Types & Features & Fitted & Types \\",
        r"\midrule",
    ]
    last = None
    for r in rows:
        c = CORPUS_LABEL[r["corpus"]] if r["corpus"] != last else ""
        if r["corpus"] != last and last is not None:
            lines.append(r"\addlinespace[2pt]")
        last = r["corpus"]
        lines.append(f"{c} & {VARIANT_LABEL[r['variant']]} & {r['occ'] / 1e6:.1f} & "
                     f"{num(r['types'])} & {num(r['scaling'])} & "
                     f"{num(r['fitted'])} & {num(r['emb'])} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Tokens counts running words after non-negation stopwords are "
        r"removed; Types counts the distinct tokens in that stream. Features is "
        r"the vocabulary the two word-count models see after stemming, "
        r"bucketing, and folding rare types into a single out-of-vocabulary "
        r"bucket (Appendix~\ref{app:unsup-counts}); correspondence analysis "
        r"always fits this column. Fitted reports the features that enter "
        r"Wordfish, reduced further only where the full vocabulary returns no "
        r"finite estimates. The embedding column reports the unstemmed types "
        r"Doc2Vec retains (Appendix~\ref{app:unsup-d2v}).",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    (OUT / "vocab_sizes.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT / 'vocab_sizes.tex'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
