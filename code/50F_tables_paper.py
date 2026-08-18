from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
IRT = REPO / "data" / "irt"
OUT = REPO / "paper" / "sections" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location("irt_data07", HERE / "04_irt_data.py")
irt_data = importlib.util.module_from_spec(_spec)
sys.modules["irt_data07"] = irt_data
_spec.loader.exec_module(irt_data)

_ms = importlib.util.spec_from_file_location("macro_irt_data", HERE / "04B_macro_irt_data.py")
macro_data = importlib.util.module_from_spec(_ms)
sys.modules["macro_irt_data"] = macro_data
_ms.loader.exec_module(macro_data)


def load(corpus):
    """The main fit: the supertopic columns, boilerplate dropped."""
    F, D = macro_data.load_main(corpus)
    return D, F["theta"].astype(float)


# --------------------------------------------------------------------------- #
def positions_by_group():
    """Group-level summary of the ideal points, both corpora."""
    rows = []
    DU, thU = load("us")
    for g, m in [("Democrats", DU["party"] == "Democrat"),
                 ("Republicans", DU["party"] == "Republican")]:
        rows.append(("United States", g, int(m.sum()), thU[m]))
    DB, thB = load("br")
    ext = DB["ext"]
    ok = np.isfinite(ext)
    # With the mayoral-platform mask every Brazilian party carries an expert
    # score, so the "no expert score" group is empty and must not be emitted
    # (an empty group prints a nan row). Only append it if it has documents.
    has_unscored = bool((~ok).any())
    br_groups = [("Left Parties", ok & (ext <= irt_data.BR_LEFT_CUT)),
                 ("Center \\& Right Parties", ok & (ext > irt_data.BR_LEFT_CUT))]
    if has_unscored:
        br_groups.append(("Parties without an expert score", ~ok))
    for g, m in br_groups:
        rows.append(("Brazil", g, int(m.sum()), thB[m]))

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Estimated Ideal Points by Party Group}",
        r"\label{tab:positions-by-group}",
        r"\begin{threeparttable}",
        r"\small",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"\multicolumn{1}{c}{Corpus} & \multicolumn{1}{c}{Group} & "
        r"\multicolumn{1}{c}{$n$} & \multicolumn{1}{c}{Mean} & "
        r"\multicolumn{1}{c}{Median} & \multicolumn{1}{c}{SD} \\",
        r"\midrule",
    ]
    last = None
    for corpus, g, n, v in rows:
        c = corpus if corpus != last else ""
        if corpus != last and last is not None:
            lines.append(r"\midrule")
        last = corpus
        lines.append(f"{c} & {g} & {n:,} & {v.mean():+.3f} & "
                     f"{np.median(v):+.3f} & {v.std():.3f} \\\\")
    sdU, sdB = float(thU.std()), float(thB.std())
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Summary of the fitted ideal points $\theta_i$ within each group of "
        r"documents. The corpus-wide standard deviation of $\theta$ is "
        f"{sdU:.3f} in the United States and {sdB:.3f} in Brazil. "
        r"Footnote~\ref{fn:brblocs} lists the parties in each Brazilian bloc"
        + (r"; parties the expert survey does not cover are scaled with all "
           r"others and reported as their own row" if has_unscored else "")
        + r". Group means are tied by the centering constraint of "
        r"Section~\ref{sec:model}.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    (OUT / "positions_by_group.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote positions_by_group.tex")


# --------------------------------------------------------------------------- #
NOTABLES = [
    # (doc_id substring, cycle, display name, district)
    ("Pramila Jayapal", "2022", "Pramila Jayapal (D)", "WA-7"),
    ("Alexandria Ocasio-Cortez", "2022", "Alexandria Ocasio-Cortez (D)", "NY-14"),
    ("Ilhan Omar", "2022", "Ilhan Omar (D)", "MN-5"),
    ("Jared Golden", "2022", "Jared Golden (D)", "ME-2"),
    ("Henry Cuellar", "2020", "Henry Cuellar (D)", "TX-28"),
    ("Brian Fitzpatrick", "2020", "Brian Fitzpatrick (R)", "PA-1"),
    ("Liz Cheney", "2022", "Liz Cheney (R)", "WY-AL"),
    ("Elise Stefanik", "2018", "Elise Stefanik (R)", "NY-21"),
    ("Matt Gaetz", "2022", "Matt Gaetz (R)", "FL-1"),
    ("Lauren Boebert", "2022", "Lauren Boebert (R)", "CO-3"),
    ("Majorie Taylor Greene", "2022", "Marjorie Taylor Greene (R)", "GA-14"),
]


def notable_candidates_us():
    D, th = load("us")
    docs = D["docs"]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Estimated Positions of Notable US Candidates}",
        r"\label{tab:notable-us}",
        r"\begin{threeparttable}",
        r"\small",
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Candidate & District & Cycle & $\theta$ \\",
        r"\midrule",
    ]
    dem = th[D["party"] == "Democrat"].mean()
    rep = th[D["party"] == "Republican"].mean()
    for pat, year, name, dist in NOTABLES:
        hit = [i for i, d in enumerate(docs)
               if pat.lower() in str(d).lower() and str(d).endswith(year)]
        if len(hit) != 1:
            print(f"  WARNING: {pat} {year} matched {len(hit)} documents; skipped")
            continue
        lines.append(f"{name} & {dist} & {year} & ${th[hit[0]]:+.2f}$ \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Fitted ideal point of the candidate's platform in the listed primary "
        r"cycle; each candidate--cycle platform is scaled as its own document. "
        f"For reference, the Democratic mean is ${dem:+.2f}$, the Republican mean "
        f"${rep:+.2f}$, and the corpus standard deviation ${th.std():.2f}$. "
        r"Candidates were selected before inspecting their estimates; the "
        r"accompanying text discusses the exceptions.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    (OUT / "notable_candidates_us.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote notable_candidates_us.tex")


def positions_by_party_br():
    D, th = load("br")
    party, ext = D["party"], D["ext"]
    ok = np.isfinite(ext)
    rows = []
    for p in sorted(set(party[ok])):
        m = party == p
        rows.append(dict(party=p, expert=float(ext[m][0]), n=int(m.sum()),
                         mean=float(th[m].mean()), median=float(np.median(th[m])),
                         sd=float(th[m].std())))
    df = pd.DataFrame(rows).sort_values("expert").reset_index(drop=True)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Estimated Positions of Brazilian Parties, Ordered by Expert Placement}",
        r"\label{tab:positions-by-party-br}",
        r"\begin{threeparttable}",
        r"\small",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Party & Expert & $n$ & Mean $\theta$ & Median $\theta$ & SD $\theta$ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(f"{r.party} & {r.expert:.2f} & {int(r.n):,} & "
                     f"${r['mean']:+.3f}$ & ${r['median']:+.3f}$ & {r.sd:.3f} \\\\")
    pm, pe = df["mean"].to_numpy(), df["expert"].to_numpy()
    r_all = pearsonr(pm, pe)[0]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Parties are ordered by the expert left--right placement of "
        r"\citet{bolognesi2023new}, on which higher means further right. $n$ is "
        r"the number of the party's platforms in the estimation sample; mean, "
        r"median, and SD summarize their ideal points $\theta_i$. Across the "
        f"{len(df)} parties, party-mean $\\theta$ and the expert score correlate "
        f"at $r={r_all:.2f}$. Parties without an expert placement are estimated "
        r"with the rest but omitted from this table.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    (OUT / "positions_by_party_br.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote positions_by_party_br.tex ({len(df)} parties, r={r_all:.3f})")


# --------------------------------------------------------------------------- #
# Unsupervised baselines, fit by 40B_wordfish_ca_fit.R and 40C_doc2vec_pca.py.
# Those stages persist a document score per fit; the correlations are computed
# HERE, from the persisted scores and the stage-07 metadata, so every row of the
# table is validated against exactly the same benchmark and the same documents.
# Each entry is (row label in the paper, file stem of the persisted scores).
COLS = ["CFScore_overall", "CFScore_Dem", "CFScore_Rep",
        "NOMINATE_overall", "NOMINATE_Dem", "NOMINATE_Rep", "BR_overall"]

BASELINES = [
    ("Correspondence analysis, unigrams", "ca_theta_{c}_uni"),
    ("Correspondence analysis, bigrams",  "ca_theta_{c}"),
    ("Wordfish, unigrams",                "wordfish_theta_{c}_uni"),
    ("Wordfish, bigrams",                 "wordfish_theta_{c}"),
    ("Doc2Vec + PCA, unigrams",           "doc2vec_theta_{c}_uni"),
    ("Doc2Vec + PCA, bigrams",            "doc2vec_theta_{c}"),
    # LLM ask-and-average (Le Mens & Gallego 2025), scored by stages 23--25 on a
    # 1,000-document sample per corpus; the row is omitted until those scores exist.
    ("Ask and average, GPT-4o mini",      "gpt4omini_theta_{c}"),
    ("Ask and average, GPT-5.4 mini",     "gpt54mini_theta_{c}"),
]


def _baseline_ndoc(corpus):
    """Documents the unsupervised baselines were fit on, from the fit metadata."""
    p = IRT / "wordfish_ca_meta.csv"
    if not p.exists():
        return None
    m = pd.read_csv(p)
    m = m[m.corpus == corpus]
    return int(m.ndoc.iloc[0]) if len(m) else None


def _sample_sentence(DU, DB):
    """State the estimation sample, and say plainly when it is not yet common.

    The baselines are fit on mayoral platforms only in Brazil. Until the IRT
    stages are refit under the same restriction the Brazilian column mixes two
    samples, and a table note that claimed otherwise would be false.
    """
    nb = _baseline_ndoc("br")
    if nb is None or nb == DB["N"]:
        return (r"Every model is fit on the same estimation sample: "
                f"{DU['N']:,} US platforms and {DB['N']:,} Brazilian platforms. "
                + _llm_sample_sentence(DU))
    return (f"In the United States every model is fit on the same {DU['N']:,} "
            r"platforms; in Brazil the unsupervised baselines are fit on the "
            f"{nb:,} mayoral platforms and the remaining models on the "
            f"{DB['N']:,} platforms of the estimation sample. "
            + _llm_sample_sentence(DU))


def _llm_sample_n(DU):
    """US documents behind the ask-and-average correlations, benchmark by
    benchmark. Far below the $n$ row, because those rows read a subsample."""
    stems = [s for lab, s in BASELINES if lab.startswith("Ask and average")]
    if not stems:
        return None
    th = _theta_for(stems[0], "us", DU["docs"])
    if th is None:
        return None
    return (int((np.isfinite(th) & np.isfinite(DU["cfscore"])).sum()),
            int((np.isfinite(th) & np.isfinite(DU["nominate"])).sum()))


def _llm_sample_sentence(DU):
    """The one exception to the common sample: the LLM rows read a subsample.

    Scoring every sentence of both corpora with a commercial model is priced per
    token, so the baseline runs on a stratified draw of documents. Saying so is
    not optional: the two rows are not computed on the same documents as the
    rest of the table, and the $n$ row therefore does not describe them.
    """
    ns = {}
    for c in ("us", "br"):
        p = GPT / f"gpt_sample_{c}.csv"
        if p.exists():
            ns[c] = len(pd.read_csv(p, dtype=str))
    stems = [s for lab, s in BASELINES if lab.startswith("Ask and average")]
    if len(ns) != 2 or not stems:
        return ""
    same = f"{ns['us']:,}" if ns["us"] == ns["br"] else None
    n = (f"a sample of {same} platforms per corpus" if same else
         f"a sample of {ns['us']:,} US and {ns['br']:,} Brazilian platforms")
    out = (r"The two ask-and-average rows are computed on " + n +
           r", stratified by party (Section~\ref{sec:gpt})")
    # Their US correlations rest on far fewer documents than the $n$ row
    # reports, which the reader has to be told.
    n = _llm_sample_n(DU)
    if n is not None:
        n_cf, n_nom = n
        out += (rf", so their US correlations rest on {n_cf:,} platforms "
                rf"carrying a CFScore and {n_nom:,} carrying a DW-NOMINATE "
                r"score")
    return out + ". "


def _theta_for(stem, corpus, docs):
    """Persisted score aligned to `docs`; NaN where the fit is missing."""
    p = IRT / f"{stem.format(c=corpus)}.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p).drop_duplicates("doc_id").set_index("doc_id").theta
    th = s.reindex(pd.Index(docs).astype(str)).to_numpy(float)
    if np.isfinite(th).sum() < 3:
        print(f"  WARNING: {p.name} covers none of the estimation sample")
        return None
    if not np.isfinite(th).all():
        print(f"  WARNING: {p.name} misses "
              f"{int((~np.isfinite(th)).sum()):,} of {len(docs):,} documents")
    return th


def _load_baselines(DU, DB):
    """Row label -> metrics Series, computed from the persisted scores."""
    cf, nom, partyU = DU["cfscore"], DU["nominate"], DU["party"]
    dem, rep = partyU == "Democrat", partyU == "Republican"
    extB, partyB = DB["ext"], DB["party"]
    okb = np.isfinite(extB)
    parties = sorted(set(partyB[okb]))

    def r(a, b, m=None):
        m = np.isfinite(a) & np.isfinite(b) & (True if m is None else m)
        return pearsonr(a[m], b[m])[0] if m.sum() > 2 else np.nan

    def party_r(th):
        keep = [p for p in parties if np.isfinite(th[partyB == p]).any()]
        pm = np.array([np.nanmean(th[partyB == p]) for p in keep])
        pe = np.array([extB[partyB == p][0] for p in keep])
        return pearsonr(pm, pe)[0] if len(keep) > 2 else np.nan

    out = []
    for label, stem in BASELINES:
        thU = _theta_for(stem, "us", DU["docs"])
        thB = _theta_for(stem, "br", DB["docs"])
        if thU is None and thB is None:
            print(f"  WARNING: no persisted scores for '{label}'; row omitted")
            continue
        v = {c: np.nan for c in COLS}
        if thU is not None:
            # Orient once, globally, against the primary US benchmark.
            if r(thU, cf) < 0:
                thU = -thU
            v.update(CFScore_overall=r(thU, cf), CFScore_Dem=r(thU, cf, dem),
                     CFScore_Rep=r(thU, cf, rep), NOMINATE_overall=r(thU, nom),
                     NOMINATE_Dem=r(thU, nom, dem), NOMINATE_Rep=r(thU, nom, rep))
        if thB is not None:
            if party_r(thB) < 0:
                thB = -thB
            v["BR_overall"] = party_r(thB)
        out.append((label, pd.Series(v)))
    return out


def validation_table():
    ab = pd.read_csv(IRT / "ablation_table.csv").set_index("estimator")

    # the joint model is fit by 08, not by the ablation script
    DU, thU = load("us")
    DB, thB = load("br")
    cf, nom, party = DU["cfscore"], DU["nominate"], DU["party"]

    def r(a, b, m=None):
        m = np.isfinite(b) if m is None else (m & np.isfinite(b))
        return pearsonr(a[m], b[m])[0]

    dem, rep = party == "Democrat", party == "Republican"
    ext, partyB = DB["ext"], DB["party"]
    okb = np.isfinite(ext)
    parties = sorted(set(partyB[okb]))
    pmB = np.array([thB[partyB == p].mean() for p in parties])
    peB = np.array([ext[partyB == p][0] for p in parties])

    joint = dict(CFScore_overall=r(thU, cf), CFScore_Dem=r(thU, cf, dem),
                 CFScore_Rep=r(thU, cf, rep), NOMINATE_overall=r(thU, nom),
                 NOMINATE_Dem=r(thU, nom, dem), NOMINATE_Rep=r(thU, nom, rep),
                 BR_overall=pearsonr(pmB, peB)[0])

    # Panels: the unsupervised word-count baselines, then the embedding
    # baselines, then the models of this paper.
    # Group by label, not by list position, so a missing baseline (e.g. the LLM
    # rows before stage 24 has been run) simply drops out of its panel.
    base = dict(_load_baselines(DU, DB))

    def pick(labels):
        return [(lab, base[lab]) for lab in labels if lab in base]

    panels = [
        ("Unsupervised scaling, bag-of-word",
         pick(["Correspondence analysis, unigrams", "Correspondence analysis, bigrams",
               "Wordfish, unigrams", "Wordfish, bigrams"])),
        ("Document embeddings",
         pick(["Doc2Vec + PCA, unigrams", "Doc2Vec + PCA, bigrams"])
         # The ablation frame keys this row on the estimator's own name; the
         # table says where the vectors come from, so it reads parallel to the
         # corpus-trained rows above it.
         + [("Pre-trained Embeddings + PCA", ab.loc["Embeddings + PCA"])]),
        ("Large language models",
         pick(["Ask and average, GPT-4o mini", "Ask and average, GPT-5.4 mini"])),
        ("This paper", [("Fasti", pd.Series(joint))]),
    ]
    panels = [(t, rows) for t, rows in panels if rows]
    # The decomposition into the two blocks travels in its own float, so the
    # baseline comparison and the component comparison are read separately.
    component_rows = [("Topic IRT", ab.loc["Topic IRT"]),
                      ("Stance IRT", ab.loc["Stance IRT"]),
                      ("Fasti", pd.Series(joint))]
    cols = COLS
    # Every row of both floats, for the macro block at the end of this function.
    # The joint model's macros come from the component row, since the baseline
    # float labels the same estimates with the method's name.
    order = [r for _, rows in panels for r in rows] + component_rows

    n_cf = int(np.isfinite(cf).sum())
    n_dem = int((np.isfinite(cf) & dem).sum())
    n_rep = int((np.isfinite(cf) & rep).sum())
    n_nom = int(np.isfinite(nom).sum())
    n_nomd = int((np.isfinite(nom) & dem).sum())
    n_nomr = int((np.isfinite(nom) & rep).sum())

    n_row = (rf"$n$ & {n_cf:,} & {n_dem:,} & {n_rep:,} & {n_nom:,} & {n_nomd:,} "
             rf"& {n_nomr:,} & {len(parties)} \\")

    def _float(caption, label, panels_, note, bold=True):
        """One benchmark float: the shared header, the panels, and its own note.

        A panel whose title is None prints its rows without an italic header,
        which is what the single-panel component table wants.  "Best in column"
        is computed within this float alone, so bold always means the best of
        the models the reader can see side by side.  The component float sets
        ``bold=False``: its rows are blocks of one model, not rival models.
        """
        rows_here = [r for _, rs in panels_ for r in rs]
        best = {c: max(v[c] for _, v in rows_here if pd.notna(v.get(c)))
                for c in cols} if bold else {}
        out = [
            r"\begin{table}[ht]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\begin{threeparttable}",
            r"\small",
            # Descriptive row labels need the column padding trimmed, not the
            # type shrunk: at the default 6pt the tabular runs into the margin.
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{@{}lcccccc c@{}}",
            r"\toprule",
            r" & \multicolumn{6}{c}{United States} & Brazil \\",
            r"\cmidrule(lr){2-7}\cmidrule(l){8-8}",
            r" & \multicolumn{3}{c}{CFScore} & \multicolumn{3}{c}{DW-NOMINATE} & Expert \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-8}",
            r"Model & All & Dem. & Rep. & All & Dem. & Rep. & Party level \\",
            r"\midrule",
        ]
        for pi, (title, rows) in enumerate(panels_):
            if pi:
                out.append(r"\addlinespace[2pt]")
            if title:
                out.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{{title}}}}} \\")
            for name, v in rows:
                cells = []
                for c in cols:
                    if pd.isna(v.get(c)):
                        cells.append(r"---")
                        continue
                    s = f"{v[c]:.2f}"
                    cells.append(rf"$\mathbf{{{s}}}$"
                                 if bold and abs(v[c] - best[c]) < 5e-3
                                 else f"${s}$")
                lead = r"\quad " if title else ""
                out.append(rf"{lead}{name} & " + " & ".join(cells) + r" \\")
        out += [
            r"\midrule",
            n_row,
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item " + note,
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
        return out

    bold_note = (r"$n$ counts the documents (for Brazil, the parties) behind each "
                 r"correlation; bold marks the best model in each column.")
    brazil_note = (r"The Brazilian column compares party-mean $\theta$ with the "
                   r"expert placements of \citet{bolognesi2023new}. ")

    baselines = _float(
        "Correlation of Estimated Ideal Points With External Benchmarks",
        "tab:validation", panels,
        r"Appendix~\ref{app:wordfish} details the unsupervised and embedding "
        r"baselines. Pearson correlation between the estimated ideal point "
        r"$\theta$ and each external benchmark. "
        + _sample_sentence(DU, DB) + brazil_note + bold_note)
    (OUT / "validation.tex").write_text("\n".join(baselines) + "\n", encoding="utf-8")
    print("wrote validation.tex")

    components = _float(
        "Correlation With External Benchmarks by Model Component",
        "tab:validation-components", [(None, component_rows)],
        r"Pearson correlation between the estimated ideal point $\theta$ and each "
        r"external benchmark. Topic IRT and Stance IRT fit Equations~\ref{eq:counts} "
        r"and~\ref{eq:stance} alone, and Fasti fits both. " + brazil_note +
        r"$n$ counts the documents (for Brazil, the parties) behind each correlation.",
        bold=False)
    (OUT / "validation_components.tex").write_text("\n".join(components) + "\n",
                                                   encoding="utf-8")
    print("wrote validation_components.tex")

    # Macros so the prose never hand-types a correlation the table also reports.
    MACRO = {
        "Correspondence analysis, unigrams": "CaUni",
        "Correspondence analysis, bigrams": "CaBi",
        "Wordfish, unigrams": "WfUni",
        "Wordfish, bigrams": "WfBi",
        "Doc2Vec + PCA, unigrams": "DtvUni",
        "Doc2Vec + PCA, bigrams": "DtvBi",
        "Ask and average, GPT-4o mini": "AskAvgFouro",
        "Ask and average, GPT-5.4 mini": "AskAvgFive",
        "Pre-trained Embeddings + PCA": "Emb",
        "Topic IRT": "Topic",
        "Stance IRT": "Stance",
        "Fasti": "Joint",
    }
    SHORT = {"CFScore_overall": "Cf", "CFScore_Dem": "CfDem", "CFScore_Rep": "CfRep",
             "NOMINATE_overall": "Nom", "NOMINATE_Dem": "NomDem",
             "NOMINATE_Rep": "NomRep", "BR_overall": "Br"}
    # \providecommand, not \newcommand: the introduction quotes the headline
    # correlations and so \input this file before Section 6 does, and a macro
    # defined twice with \newcommand aborts the compile.
    mac = [r"%% auto-generated by code/50F_tables_paper.py -- do not edit"]
    for name, v in order:
        if name not in MACRO:
            continue
        for c in cols:
            if pd.notna(v.get(c)):
                mac.append(rf"\providecommand{{\r{MACRO[name]}{SHORT[c]}}}{{{v[c]:.2f}}}")
    mac.append(rf"\providecommand{{\nBrParties}}{{{len(parties)}}}")
    # Sample sizes, so the prose can say how many documents a correlation rests
    # on without hand-typing it either.
    mac.append(rf"\providecommand{{\nCfAll}}{{{n_cf:,}}}")
    mac.append(rf"\providecommand{{\nNomAll}}{{{n_nom:,}}}")
    nl = _llm_sample_n(DU)
    if nl is not None:
        mac.append(rf"\providecommand{{\nLlmCf}}{{{nl[0]:,}}}")
        mac.append(rf"\providecommand{{\nLlmNom}}{{{nl[1]:,}}}")
    (OUT / "validation_numbers.tex").write_text("\n".join(mac) + "\n", encoding="utf-8")
    print(f"wrote validation_numbers.tex ({len(mac) - 1} macros)")
    for name, v in order:
        print(f"  {name:<18} " + "  ".join(f"{c.split('_')[0][:3]}{c.split('_')[1][:3]}={v[c]:+.3f}"
                                           for c in cols))


GPT = REPO / "data" / "gpt_scaling"
GPT_RUNS = [("gpt4omini", "GPT-4o mini"), ("gpt54mini", "GPT-5.4 mini")]
GPT_CORPORA = [("us", "US"), ("br", "Brazil")]
# The same sentence-chunk files the classifier reads, so the projection below
# counts exactly the units that would be billed.
CHUNKS = {"us": REPO / "data" / "us" / "campaignview_chunks_sent.feather",
          "br": REPO / "data" / "br" / "br_manifestos_chunks_sent.feather"}


def _full_corpus_chunks(corpus):
    """Sentence-chunks in the whole estimation sample, not just the GPT sample."""
    docs = set(map(str, np.load(IRT / f"irt_matrices_{corpus}.npz",
                                allow_pickle=True)["docs"].tolist()))
    df = pd.read_feather(CHUNKS[corpus], columns=["doc_id"])
    return int(df.doc_id.astype(str).isin(docs).sum()), len(docs)


def gpt_cost_table():
    """What the ask-and-average baseline cost and how long it took.

    Every figure is read from the run metadata the classifier writes, so the
    table reports the run that actually happened and never a projection.  The
    provider's 24h batch tier would halve the price, but the runs were made
    synchronously and that discount was not realized, so it is not reported.
    """
    # Validation correlation per run, computed exactly as in the validation
    # table: candidate-level vs CFScores in the US, party-level vs the expert
    # placements in Brazil, oriented so the correlation is positive.
    DU, _ = load("us")
    DB, _ = load("br")
    cf = DU["cfscore"]
    extB, partyB = DB["ext"], DB["party"]
    okb = np.isfinite(extB)
    parties = sorted(set(partyB[okb]))

    def _val_r(key, c):
        th = _theta_for(f"{key}_theta_{{c}}", c, (DU if c == "us" else DB)["docs"])
        if th is None:
            return np.nan
        if c == "us":
            m = np.isfinite(th) & np.isfinite(cf)
            return abs(pearsonr(th[m], cf[m])[0])
        keep = [p for p in parties if np.isfinite(th[partyB == p]).any()]
        pm = np.array([np.nanmean(th[partyB == p]) for p in keep])
        pe = np.array([extB[partyB == p][0] for p in keep])
        return abs(pearsonr(pm, pe)[0])

    rows, full = [], {}
    for key, label in GPT_RUNS:
        for c, corpus_label in GPT_CORPORA:
            p = GPT / f"meta_{key}_{c}.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text(encoding="utf-8"))
            u = m["usage"]
            if c not in full:
                full[c] = _full_corpus_chunks(c)
            n_full, n_docs_full = full[c]
            minutes = m["timing"]["elapsed_seconds_this_invocation"] / 60
            # Linear in the number of sentences, at the same concurrency. It is a
            # scale-up of the run that happened, not a run anyone made.
            factor = n_full / max(m["n_chunks"], 1)
            rows.append({"model": label, "corpus": corpus_label,
                         "r_val": _val_r(key, c),
                         "docs": m["n_documents"], "obs": m["n_chunks"],
                         "prompt": u["prompt_tokens"], "cached": u["cached_tokens"],
                         "completion": u["completion_tokens"],
                         "total": u["total_tokens"],
                         "minutes": minutes, "cost": m["direct_cost_usd"],
                         "proj_hours": minutes * factor / 60,
                         "proj_cost": m["direct_cost_usd"] * factor,
                         "full_chunks": n_full, "full_docs": n_docs_full,
                         "failed": len(m["failed_batches"]),
                         "workers": m["workers"], "price": m["price_usd_per_1m"],
                         "observed": m["price_source"]["observed_date"]})
    if not rows:
        print("  no GPT run metadata found; skipping gpt_cost.tex")
        return

    M = 1_000_000
    tok_total = sum(x["total"] for x in rows)
    prompt_share = 100 * sum(x["prompt"] for x in rows) / tok_total
    compl_share = 100 * sum(x["completion"] for x in rows) / tok_total
    cache_share = {lab: 100 * sum(x["cached"] for x in rows if x["model"] == lab)
                   / sum(x["prompt"] for x in rows if x["model"] == lab)
                   for lab in dict.fromkeys(x["model"] for x in rows)}
    cache_txt = " and ".join(f"{v:.0f}\\% for {lab}" for lab, v in cache_share.items())
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Cost, Running Time, and Validation of the Ask-and-Average Baseline}",
        r"\label{tab:gpt-cost}",
        r"\begin{threeparttable}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r" & & & & \multicolumn{3}{c}{\shortstack{Sample\\(realized)}} "
        r"& \multicolumn{2}{c}{\shortstack{Full corpus\\(projected)}} \\",
        r"\cmidrule(lr){5-7}\cmidrule(l){8-9}",
        r"Model & Corpus & Sentences & \shortstack{Tokens\\(millions)} "
        r"& $r$\tnote{a}~ & Min. & Cost\tnote{b}~ & Hours & Cost\tnote{b} \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            rf"{r['model']} & {r['corpus']} & {r['obs']:,} & "
            rf"{r['total'] / M:.2f} & "
            rf"{r['r_val']:.2f} & "
            rf"{r['minutes']:.1f} & \${r['cost']:.2f} & "
            rf"{r['proj_hours']:.1f} & \${r['proj_cost']:.0f} \\")
    prices = "; ".join(
        f"{lab} at \\${p['input']:.2f} per million prompt tokens, "
        f"\\${p['cached_input']:.3f} cached, \\${p['output']:.2f} completion"
        for lab, p in {r["model"]: r["price"] for r in rows}.items())
    # One sample size per corpus, so the stated scale-up factor cannot drift
    # from the rows above.
    samp = {c: next(r["obs"] for r in rows if r["corpus"] == lab)
            for c, lab in GPT_CORPORA}
    ndocs = {c: next(r["docs"] for r in rows if r["corpus"] == lab)
             for c, lab in GPT_CORPORA}
    br_sample = GPT / "gpt_sample_br.csv"
    n_br_parties = (pd.read_csv(br_sample, dtype=str).party.nunique()
                    if br_sample.exists() else None)
    scale = "; ".join(
        f"{lab} {full[c][0] / samp[c]:.1f} times the sample"
        for c, lab in GPT_CORPORA)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item Each row is one run of the ask-and-average baseline over one "
        r"corpus. Sentences is the number of sentences scored; each run "
        rf"scored every sentence of the same {rows[0]['docs']:,} sampled "
        r"platforms. Cost is the amount billed at the provider's prices "
        rf"published on {rows[0]['observed']}, and Min.\ is wall-clock time "
        rf"with {rows[0]['workers']} concurrent requests; the four runs were "
        r"executed concurrently, so their times are not additive. The last two "
        r"columns scale each run linearly by the number of sentences to the "
        rf"full estimation sample ({scale}); that extrapolation was not run.",
        r"\item[a] $r$ is the validation correlation of Table~\ref{tab:validation}, "
        r"computed on the run's own sample: candidate-level against CFScores in "
        r"the United States and party-level against the expert placements in "
        r"Brazil.",
        r"\item[b] The provider also offers a 24-hour batch tier at half these "
        r"prices; the runs here were synchronous, so every dollar amount in the "
        r"table is full price.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    (OUT / "gpt_cost.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote gpt_cost.tex ({len(rows)} runs, "
          f"${sum(r['cost'] for r in rows):.2f} realized)")

    # The prose quotes the totals; generate them too, so no cost or running time
    # is ever hand-typed into the section.
    longest = max(r["minutes"] for r in rows)
    mac = [r"%% auto-generated by code/50F_tables_paper.py -- do not edit",
           rf"\newcommand{{\gptCostTotal}}{{{sum(r['cost'] for r in rows):.2f}}}",
           rf"\newcommand{{\gptProjCostTotal}}{{{sum(r['proj_cost'] for r in rows):.0f}}}",
           rf"\newcommand{{\gptWallLongest}}{{{longest:.0f}}}",
           rf"\newcommand{{\gptProjHoursMax}}{{{max(r['proj_hours'] for r in rows):.0f}}}",
           rf"\newcommand{{\gptSentUs}}{{{samp['us']:,}}}",
           rf"\newcommand{{\gptSentBr}}{{{samp['br']:,}}}",
           rf"\newcommand{{\gptSampleDocs}}{{{rows[0]['docs']:,}}}"]
    if n_br_parties is not None:
        # Section 5 names the party count before validation_numbers.tex (which
        # defines \nBrParties) is \input in Section 6; a separate macro avoids the
        # double-definition that sharing the name would cause.
        mac.append(rf"\newcommand{{\gptBrParties}}{{{n_br_parties}}}")
    (OUT / "gpt_numbers.tex").write_text("\n".join(mac) + "\n", encoding="utf-8")
    print(f"wrote gpt_numbers.tex ({len(mac) - 1} macros)")


# --------------------------------------------------------------------------- #
def _topic_labels(corpus, tops):
    """Labels aligned to the fitted columns.

    A macro column is a class of the codebook and carries its own label; a fine
    column is a topic and carries the label the author assigned it.
    """
    sys.path.insert(0, str(HERE))
    import macro_topics as mt
    if all(str(t) in mt.LABEL for t in tops):
        out = [mt.LABEL[str(t)] for t in tops]
    else:
        lut = mt.topic_labels(corpus)
        out = [lut.get(int(t), f"Topic {int(t)}") for t in tops]
    return [s.replace("&", r"\&") for s in out]


def top_topics_by_group(topn=15):
    """The ten most covered topics of each corpus, with each party group's
    coverage share and mean stance."""
    spec = {
        "us": ("United States",
               [("Democrats", lambda D: D["party"] == "Democrat"),
                ("Republicans", lambda D: D["party"] == "Republican")], ""),
        "br": ("Brazil",
               [("Left Parties",
                 lambda D: np.isfinite(D["ext"]) & (D["ext"] <= irt_data.BR_LEFT_CUT)),
                ("Center \\& Right Parties",
                 lambda D: np.isfinite(D["ext"]) & (D["ext"] > irt_data.BR_LEFT_CUT))],
               r" Footnote~\ref{fn:brblocs} lists the parties in each bloc."),
    }
    for corpus, (cname, gdefs, gnote) in spec.items():
        D, _th = load(corpus)
        Y, S = D["Y"], D["S"]
        labs = _topic_labels(corpus, D["tops"])
        n = np.asarray(Y.sum(0)).ravel()
        cols = []
        for gname, gmask in gdefs:
            m = gmask(D).nonzero()[0]
            gn = np.asarray(Y[m].sum(0)).ravel()
            gs = np.asarray(S[m].sum(0)).ravel()
            rank = np.empty(len(gn), int)
            rank[np.argsort(-gn)] = np.arange(1, len(gn) + 1)
            cols.append((gname, rank, gn, gs))
        order = np.argsort(-n)[:topn]

        def _hdr(g):
            # Full proper nouns only, never abbreviated; long names break onto
            # a second header line instead.
            if len(g) <= 14:
                return g
            words = g.split()
            return (r"\shortstack{" + " ".join(words[:-1]) + r"\\"
                    + words[-1] + "}")

        heads = " & ".join(rf"\multicolumn{{3}}{{c}}{{{_hdr(g)}}}" for g, *_ in cols)
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            rf"\caption{{Most Covered Topics, {cname}}}",
            rf"\label{{tab:top-topics-{corpus}}}",
            r"\begin{threeparttable}",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            rf"Topic & {heads} \\",
            r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
            "& " + " & ".join(["Rank & Share & Stance"] * len(cols)) + r" \\",
            r"\midrule",
        ]
        for j in order:
            cells = " & ".join(
                f"{rk[j]} & {100 * gn[j] / gn.sum():.1f}\\% & ${gs[j] / gn[j]:+.2f}$"
                for _, rk, gn, gs in cols)
            lines.append(f"{labs[j]} & {cells} \\\\")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item The fifteen most covered topics of the corpus, ordered by overall "
            r"coverage. Rank is the topic's position in the group's own coverage "
            r"ranking; Share is the percent of the group's sentences assigned to "
            r"the topic; Stance is the mean of the group's sentence codes on the "
            r"topic, on the $[-1,1]$ scale of Section~\ref{sec:stance}. Topics "
            r"carry their assigned labels." + gnote,
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
        (OUT / f"top_topics_{corpus}.tex").write_text("\n".join(lines) + "\n",
                                                      encoding="utf-8")
        print(f"wrote top_topics_{corpus}.tex")


if __name__ == "__main__":
    positions_by_group()
    notable_candidates_us()
    positions_by_party_br()
    validation_table()
    gpt_cost_table()
    top_topics_by_group()
    print(f"tables -> {OUT}")
