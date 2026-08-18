import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = DATA / "irt"
TABLES = REPO / "paper" / "sections" / "tables"

TOPK = 10
MODELS = ["paper", "lda", "nmf", "stm", "dmr"]
LABELS = {"paper": "BERTopic",
          "lda": "LDA", "nmf": "NMF", "stm": "STM",
          "dmr": "DMR"}
HARDWARE = {"paper": "T4/L4 GPU",
            "lda": "6-core CPU", "nmf": "4-core CPU",
            "stm": "1-core CPU", "dmr": "6-core CPU"}


def paper_topics(corpus: str) -> list[list[str]]:
    npz = np.load(OUT / f"irt_matrices_{corpus}.npz", allow_pickle=True)
    kept = set(int(t) for t in npz["tops"])
    ba = pd.read_excel(DATA / "topics" / f"topic_model_{corpus}" /
                       "topic_info_before_after.xlsx")
    ba = ba[ba.topic.astype(int).isin(kept)].sort_values("topic")
    return [str(s).split(", ")[:TOPK] for s in ba.keywords_after]


def csv_topics(path: Path, model: str) -> list[list[str]] | None:
    if not path.exists():
        print(f"  MISSING: {path.name} -- {model} left out of this run")
        return None
    df = pd.read_csv(path)
    df = df[(df.model == model) & (df["rank"] <= TOPK)]
    return [g.sort_values("rank").term.tolist()
            for _, g in df.groupby("topic", sort=True)]


def run(corpus: str, only: list[str] | None = None) -> list[dict]:
    t0 = time.time()
    # sys.intern: one shared str object per type, not one per occurrence --
    # the BR stream (39M tokens) must fit beside the running R fits
    texts = [[sys.intern(w) for w in s.split()] for s in
             pd.read_feather(DATA / "processed" / f"wf_tokens_{corpus}.feather").tokens]
    vocab = {w for t in texts for w in t}
    print(f"[{corpus}] reference corpus: {len(texts):,} documents, "
          f"{sum(len(t) for t in texts):,} tokens, {len(vocab):,} types "
          f"({time.time()-t0:.0f}s)")

    raw = {"paper": paper_topics(corpus),
           "lda": csv_topics(OUT / f"topic_quality_top_{corpus}.csv", "lda"),
           "nmf": csv_topics(OUT / f"topic_quality_top_{corpus}.csv", "nmf"),
           "dmr": csv_topics(OUT / f"topic_quality_top_{corpus}.csv", "dmr"),
           "stm": csv_topics(OUT / f"topic_quality_top_stm_{corpus}.csv", "stm")}
    raw = {m: t for m, t in raw.items()
           if t is not None and (only is None or m in only)}

    # processes=1 is the OCTIS default; >1 spawns workers that each copy the
    # reference corpus, which does not fit in RAM beside the running R fits
    cm = Coherence(texts=texts, topk=TOPK, measure="c_npmi", processes=1)
    td = TopicDiversity(topk=TOPK)

    rows = []
    for m in [m for m in MODELS if m in raw]:
        tps = raw[m]
        in_dict = [[w for w in t if w in vocab] for t in tps]
        scored = [t for t in in_dict if len(t) >= 2]
        n_terms = sum(len(t) for t in tps)
        n_kept = sum(len(t) for t in in_dict)
        t1 = time.time()
        npmi = cm.score({"topics": scored})
        div = td.score({"topics": tps})
        print(f"[{corpus}] {m}: K={len(tps)} scored={len(scored)} "
              f"coverage={n_kept}/{n_terms} npmi={npmi:.4f} "
              f"diversity={div:.3f} ({time.time()-t1:.0f}s)")
        rows.append(dict(corpus=corpus, model=m, K=len(tps),
                         npmi=float(npmi), diversity=float(div),
                         coverage=n_kept / n_terms,
                         topics_scored=len(scored)))
    return rows


def paper_fit_meta() -> pd.DataFrame:
    """BERTopic wall clock = sentence embedding (the tqdm totals logged by
    01_embed.ipynb) plus the topic-model fit (the 'fit {N}s' prints of
    02_topic_model.ipynb), both from the notebooks' saved GPU-run outputs."""
    import json
    import re

    def outputs(name):
        cells = json.loads((REPO / "code" / name)
                           .read_text(encoding="utf-8"))["cells"]
        return ["".join("".join(o.get("text", [])) for o in c.get("outputs", [])
                        if o.get("output_type") == "stream") for c in cells]

    def tqdm_secs(text, tag):
        last = None
        for m in re.finditer(rf"\b{tag}: 100%\|[^\[]*\[(\d+(?::\d+)+)<", text):
            last = m
        if last is None:
            return None
        secs = 0
        for p in last.group(1).split(":"):
            secs = secs * 60 + int(p)
        return secs

    embed_out = "\n".join(outputs("01_embed.ipynb"))
    embed = {"us": tqdm_secs(embed_out, "us"),
             "br": tqdm_secs(embed_out, "br encode")}

    rows = []
    for out in outputs("02_topic_model.ipynb"):
        m = re.search(r"^fit (\d+)s \|", out, flags=re.M)
        if m:
            corpus = "us" if "US PIPELINE" in out else \
                     "br" if "BR" in out else None
            if corpus and embed[corpus] is not None:
                rows.append(dict(corpus=corpus, model="paper",
                                 embed_seconds=embed[corpus],
                                 fit_seconds=int(m.group(1)),
                                 seconds=embed[corpus] + int(m.group(1))))
    return pd.DataFrame(rows)


def load_meta() -> pd.DataFrame:
    metas = [OUT / "topic_quality_fit_meta.csv"] + \
            [OUT / f"topic_quality_stm_meta_{c}.csv" for c in ["us", "br"]]
    return pd.concat([pd.read_csv(p) for p in metas if p.exists()] +
                     [paper_fit_meta()], ignore_index=True)


def write_latex(mt: pd.DataFrame) -> None:
    meta = load_meta()

    def cell(c, m, col, nd):
        v = mt.loc[(mt.corpus == c) & (mt.model == m), col]
        return f"${v.iloc[0]:.{nd}f}$" if len(v) else "--"

    def tcell(c, m):
        v = meta.loc[(meta.corpus == c) & (meta.model == m), "seconds"]
        return f"${max(1, round(v.iloc[0] / 60))}$" if len(v) else "--"

    def icell(c, m):
        r = meta.loc[(meta.corpus == c) & (meta.model == m)]
        if not len(r) or "iterations" not in r.columns or pd.isna(r.iloc[0]["iterations"]):
            return "--"
        return f"${int(r.iloc[0]['iterations'])}$"

    kus = int(mt.loc[mt.corpus == "us", "K"].iloc[0])
    kbr = int(mt.loc[mt.corpus == "br", "K"].iloc[0])
    lines = [
        "%% auto-generated by code/40J_topic_quality_metrics.py -- do not edit",
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Topic Quality and Runtime Against Alternative Topic Models}",
        r"\label{tab:topic-quality}", r"\begin{threeparttable}", r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
        r" & \multicolumn{4}{c}{\shortstack{US Congress\\Candidates}} & "
        r"\multicolumn{4}{c}{\shortstack{Brazil Mayoral\\Candidates}} & \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"Model & Coherence & Diversity & \shortstack{Time\\(min.)} & Iter. & "
        r"Coherence & Diversity & \shortstack{Time\\(min.)} & Iter. & Hardware \\",
        r"\midrule",
    ]
    for m in [m for m in MODELS if (mt.model == m).any()]:
        lines.append(f"{LABELS[m]} & {cell('us', m, 'npmi', 3)} & "
                     f"{cell('us', m, 'diversity', 2)} & {tcell('us', m)} & "
                     f"{icell('us', m)} & "
                     f"{cell('br', m, 'npmi', 3)} & "
                     f"{cell('br', m, 'diversity', 2)} & {tcell('br', m)} & "
                     f"{icell('br', m)} & "
                     f"{HARDWARE[m]} \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]", r"\footnotesize",
        r"\item Coherence is the mean normalized pointwise mutual information over each "
        r"topic's top ten terms and diversity the share of unique terms across all "
        r"topics' top ten \citep{grootendorst2022bertopic}. All models are scored on "
        r"the same reference corpus, the bigram token stream of Appendix~\ref{app:unsup-prep}, at the "
        "corpus's final topic count (%d topics in the United States, %d in Brazil). "
        % (kus, kbr) +
        r"BERTopic (UMAP + HDBSCAN) is the topic model of Section~\ref{sec:topics}; "
        r"the key terms are the unigrams of Equation~\ref{eq:ctfidf}. Time is one "
        r"fit's wall clock in minutes on the listed hardware; the BERTopic time is "
        r"sentence embedding (T4) plus the topic-model fit (L4). Iter.\ is the "
        r"fit's number of training sweeps.",
        r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}",
    ]
    (TABLES / "topic_quality.tex").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")
    print("wrote paper/sections/tables/topic_quality.tex")


def write_numbers(mt: pd.DataFrame) -> None:
    meta = load_meta()
    up = {"us": "US", "br": "Br"}
    lines = ["%% auto-generated by code/40J_topic_quality_metrics.py -- do not edit"]
    for c in ["us", "br"]:
        v = meta.loc[meta.corpus == c, "vocab"]
        if len(v):
            lines.append(rf"\providecommand{{\TQvocab{up[c]}}}{{{int(v.iloc[0]):,}}}")
        cov = mt.loc[(mt.corpus == c) & (mt.model == "paper"), "coverage"]
        if len(cov):
            lines.append(rf"\providecommand{{\TQcov{up[c]}}}{{{100*cov.iloc[0]:.1f}}}")
        pr = meta.loc[(meta.corpus == c) & (meta.model == "paper")]
        if len(pr) and "embed_seconds" in pr.columns and pd.notna(pr.iloc[0]["embed_seconds"]):
            share = 100 * pr.iloc[0]["embed_seconds"] / pr.iloc[0]["seconds"]
            lines.append(rf"\providecommand{{\TQembShare{up[c]}}}{{{share:.1f}}}")
        keys = {"paper": "Bert", "lda": "Lda", "nmf": "Nmf",
                "stm": "Stm", "dmr": "Dmr"}
        for _, r in mt[mt.corpus == c].iterrows():
            k = keys.get(r["model"])
            if k:
                lines.append(rf"\providecommand{{\TQnpmi{k}{up[c]}}}{{{r['npmi']:.3f}}}")
                lines.append(rf"\providecommand{{\TQdiv{k}{up[c]}}}{{{r['diversity']:.2f}}}")
        for _, r in meta[meta.corpus == c].iterrows():
            if "iterations" not in meta.columns or pd.isna(r["iterations"]):
                continue
            key = f"{r['model'].upper().replace('LDA','Lda').replace('NMF','Nmf').replace('STM','Stm').replace('DMR','Dmr')}"
            lines.append(rf"\providecommand{{\TQits{key}{up[c]}}}{{{int(r['iterations'])}}}")
    (TABLES / "topic_quality_numbers.tex").write_text("\n".join(lines) + "\n",
                                                      encoding="utf-8")
    print("wrote paper/sections/tables/topic_quality_numbers.tex")


def main() -> int:
    if "--tables-only" in sys.argv:
        mt = pd.read_csv(OUT / "topic_quality_metrics.csv")
        write_latex(mt)
        write_numbers(mt)
        return 0
    only = None
    if "--add" in sys.argv:
        only = sys.argv[sys.argv.index("--add") + 1].split(",")
    rows = []
    for corpus in ["us", "br"]:
        rows += run(corpus, only)
    mt = pd.DataFrame(rows)
    if only is not None:
        old = pd.read_csv(OUT / "topic_quality_metrics.csv")
        mt = pd.concat([old[~old.model.isin(mt.model.unique())], mt],
                       ignore_index=True)
    mt.to_csv(OUT / "topic_quality_metrics.csv", index=False, encoding="utf-8")
    print(mt.to_string(index=False))
    write_latex(mt)
    write_numbers(mt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
