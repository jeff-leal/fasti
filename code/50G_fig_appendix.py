from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = Path(os.getenv("TOPIC2IRT_ROOT", HERE.parents[0]))
IRT = REPO / "data" / "irt"
OUT = REPO / "paper" / "Figures"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE))
import paper_style  # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


irt_data = _by_path("irt_data07", HERE / "04_irt_data.py")
macro_data = _by_path("macro_irt_data", HERE / "04B_macro_irt_data.py")

BLUE, RED, INK = "#4C72B0", "#C44E52", "#222222"
TITLES = {"us": "US Congress Candidates", "br": "Brazil Mayoral Candidates"}
BR_LEFT_CUT = irt_data.BR_LEFT_CUT

#: the language model whose topic coding the appendix reports. GPT-5.4 mini
#: uses the boilerplate class as the codebook defines it (8.6% of Brazilian and
#: 1.3% of US sentences, close to the topic model's own share), where GPT-4o
#: mini almost never applies it and pushes that material into the policy
#: classes; the two are within a few hundredths on external validity.
GPT_MODEL = "gpt54mini"
#: the stance run the appendix prose quotes
GPT_PROMPT = "nonames"
GPT_NAME = {"gpt54mini": "GPT-5.4 mini", "gpt4omini": "GPT-4o mini"}
#: the two stance scoring runs, original prompt and no-party-cue prompt
PROMPTS = ("v01", "nonames")
CUE = {"v01": "", "nonames": ", no cues"}


def _z(v, w=None):
    v = np.asarray(v, float)
    w = np.ones_like(v) if w is None else np.asarray(w, float)
    m = np.average(v, weights=w)
    return (v - m) / np.sqrt(np.average((v - m) ** 2, weights=w))


def wcorr(x, y, w=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[ok], y[ok], w[ok]
    cx, cy = x - np.average(x, weights=w), y - np.average(y, weights=w)
    return float(np.average(cx * cy, weights=w)
                 / np.sqrt(np.average(cx ** 2, weights=w)
                           * np.average(cy ** 2, weights=w)))


def series(corpus):
    """Every scale on one index, the document id, with its design weight."""
    D = irt_data.load_matrices(corpus, verbose=False)
    docs_all = np.asarray(D["docs"]).astype(str)
    out = {}

    for key, level in (("macro", "macro"), ("micro", "micro_nobp")):
        F = np.load(IRT / f"joint_irt_fit_{level}_{corpus}_godambe.npz",
                    allow_pickle=True)
        d = F["docs"].astype(str) if "docs" in F.files else docs_all
        out[key] = pd.Series(F["theta"].astype(float), index=d)

    w = None
    for m in GPT_NAME:
        for pr in PROMPTS:
            F = np.load(IRT / f"joint_irt_fit_gpt_{m}_{pr}_{corpus}_godambe.npz",
                        allow_pickle=True)
            out[f"gpt_{m}_{pr}"] = pd.Series(F["theta"].astype(float),
                                             index=F["docs"].astype(str))
            a = pd.read_csv(IRT / f"gpt_askavg_{m}_{corpus}_{pr}.csv")
            out[f"avg_{m}_{pr}"] = pd.Series(a["theta"].to_numpy(float),
                                             index=a["doc_id"].astype(str))
            if w is None:
                w = pd.Series(a["w"].to_numpy(float), index=a["doc_id"].astype(str))

    party = pd.Series(np.asarray(D["party"]).astype(str), index=docs_all)
    if corpus == "us":
        grp = party.map({"Democrat": "Democrats", "Republican": "Republicans"})
    else:
        ext = pd.Series(np.asarray(D["ext"], float), index=docs_all)
        grp = pd.Series(np.where(ext <= BR_LEFT_CUT, "Left Parties",
                                 "Center & Right Parties"), index=docs_all)
        grp[~np.isfinite(ext)] = np.nan
    return out, grp, w


def _panel(ax, x, y, grp, w, xlabel, ylabel, title, weighted):
    order = (["Democrats", "Republicans"] if "Democrats" in set(grp.dropna())
             else ["Left Parties", "Center & Right Parties"])
    for name, col in zip(order, (BLUE, RED)):
        m = (grp == name).to_numpy()
        ax.scatter(x[m], y[m], s=4, alpha=.35, color=col, lw=0, label=name)
    r = wcorr(x, y, w)
    lo = min(np.nanmin(x), np.nanmin(y))
    hi = max(np.nanmax(x), np.nanmax(y))
    ax.plot([lo, hi], [lo, hi], color="0.35", lw=.9, ls="--", dashes=(6, 4), zorder=5)
    ax.text(0.045, 0.955, f"r = {r:.2f}", transform=ax.transAxes, va="top",
            ha="left", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, pad=4)
    ax.grid(alpha=.18)
    ax.margins(0.04)
    # Three panels on one article-width row leave little horizontal room, so the
    # tick count is capped rather than the type shrunk.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    return r, int(np.isfinite(x).sum()), weighted


def fig_fit_agreement():
    """One figure: the macro-topic against the fine-topic estimate, both corpora.

    The language-model comparisons are not repeated here; the agreement matrix
    reports every pair of scales at once.
    """
    paper_style.apply(base=10)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))
    reported = []
    for ax, corpus in zip(axes, ("us", "br")):
        S, grp, _ = series(corpus)
        ix = S["macro"].index.intersection(S["micro"].index)
        x = _z(S["macro"].reindex(ix).to_numpy())
        y = _z(S["micro"].reindex(ix).to_numpy())
        r, n, _ = _panel(ax, x, y, grp.reindex(ix), None,
                         r"Supertopic $\theta$ (SD units)",
                         r"Subtopic $\theta$ (SD units)", TITLES[corpus], False)
        reported.append((corpus, r, n))
        _legend_under(ax, ncol=2, frameon=False, fontsize=8, markerscale=2.4,
                      handletextpad=.3, columnspacing=1.6)
    fig.tight_layout(w_pad=2.4)
    _check_ticks(fig, axes, "fit agreement")
    name = "fig_fit_agreement.pdf"
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {name}   " + "  ".join(f"{c} {r:+.3f} (n={n:,})"
                                         for c, r, n in reported))


def _legend_under(ax, gap_pt=4, **kw):
    """The colour key sits just under the axis label, measured not guessed."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    y0 = ax.xaxis.label.get_window_extent(rend).y0 - gap_pt / 72.0 * fig.dpi
    y = ax.transAxes.inverted().transform((0, y0))[1]
    kw.setdefault("borderpad", 0.1)
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), **kw)


#: every scale the agreement matrix carries: the main estimate, then each
#: language model crossed with the two ways of turning its output into a position
#: (the joint model, and averaging its sentence scores) and the two stance prompts.
HEAT_ORDER = ([("macro", "Joint IRT, supertopics")]
              + [(f"gpt_{m}_{pr}", f"Joint IRT, {GPT_NAME[m]}{CUE[pr]}")
                 for m in GPT_NAME for pr in PROMPTS]
              + [(f"avg_{m}_{pr}", f"Ask-and-average, {GPT_NAME[m]}{CUE[pr]}")
                 for m in GPT_NAME for pr in PROMPTS])


def fig_model_agreement():
    """One Pearson matrix per corpus over every estimated scale.

    Nine scales leave no room for a written row label beside a readable cell, so
    both axes carry the number and the key sits to the left of the panels.
    """
    paper_style.apply(base=10)
    keys = [k for k, _ in HEAT_ORDER]
    n = len(keys)
    # The canvas is the text width exactly, so nothing is scaled down on the
    # page, and the key sits under the panels rather than beside them: every
    # inch the key would take on the left is an inch off the cells.
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.3))
    fig.subplots_adjust(left=0.045, right=0.90, top=0.93, bottom=0.30,
                        wspace=0.10)
    for ax, corpus in zip(axes, ("us", "br")):
        S, _, w = series(corpus)
        ix = S[keys[0]].index
        for k in keys[1:]:
            ix = ix.intersection(S[k].index)
        ww = w.reindex(ix).to_numpy()
        M = np.array([[wcorr(S[a].reindex(ix).to_numpy(),
                             S[b].reindex(ix).to_numpy(), ww)
                       for b in keys] for a in keys])
        # White-to-light-blue ramp: every cell stays light enough that the
        # black annotations read at any correlation, and the ramp keeps the
        # palette of the rest of the paper.
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("corr_light",
                                                 ["#FFFFFF", "#89ABD4"])
        im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap=cmap)
        ax.set_xticks(range(n), [str(i + 1) for i in range(n)], fontsize=7.5)
        ax.set_yticks(range(n), [str(i + 1) for i in range(n)], fontsize=7.5)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.6, color="black")
        ax.set_title(f"{TITLES[corpus]}  (n = {len(ix):,})", fontsize=9, pad=5)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)

    # The key to the numbering, in three columns under the panels.
    labs = [f"{i + 1}. {lab}" for i, (_, lab) in enumerate(HEAT_ORDER)]
    per = 3
    for c in range(3):
        fig.text(0.045 + c * 0.30, 0.20, "\n".join(labs[c * per:(c + 1) * per]),
                 va="top", ha="left", fontsize=7, linespacing=1.6)
    fig.colorbar(im, ax=axes, fraction=0.028, pad=0.02,
                 label="Pearson correlation").outline.set_visible(False)
    _check_ticks(fig, axes, "model agreement")
    name = "fig_model_agreement.pdf"
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {name}   {n} scales")


def _ticks_in_view(ticks, labels, lim):
    """A locator can emit ticks outside the view; their labels are never drawn,
    so they must not be counted as collisions."""
    lo, hi = sorted(lim)
    return [b for t, b in zip(ticks, labels) if lo - 1e-9 <= t <= hi + 1e-9]


def _check_ticks(fig, axes, what, gap=1.0):
    """Every drawn string must clear every other one, measured by the renderer."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = []
    for ax in axes:
        arts = (_ticks_in_view(ax.get_yticks(), ax.get_yticklabels(), ax.get_ylim())
                + _ticks_in_view(ax.get_xticks(), ax.get_xticklabels(), ax.get_xlim())
                + list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label])
        for a in arts:
            if not a.get_visible() or not str(a.get_text()).strip():
                continue
            b = a.get_window_extent(rend)
            if b.width and b.height:
                boxes.append((b, str(a.get_text())[:24]))
    bad = [f"{boxes[i][1]!r}/{boxes[j][1]!r}"
           for i in range(len(boxes)) for j in range(i + 1, len(boxes))
           if boxes[i][0].padded(-gap).overlaps(boxes[j][0].padded(-gap))]
    print(f"  [{'OK' if not bad else 'CRAMPED'}] {what}: {len(bad)} text collisions"
          + (f" -> {'; '.join(bad[:4])}" if bad else ""))
    return len(bad)


TAB = REPO / "paper" / "sections" / "tables"


def write_numbers():
    """Every number the alternative-resolution appendix quotes, as macros.

    The prose never carries a literal, so the text cannot drift from the fits.
    """
    import pandas as pd
    sys.path.insert(0, str(HERE))
    import macro_topics as mt
    import topic_codes

    m = {}
    for c in ("us", "br"):
        U = c.upper()
        S = mt.CROSSWALK[mt.CROSSWALK.Corpus == U]
        m[f"nMacro{U.title()}"] = str(len(mt.classes_present(c)) - 1)   # boilerplate out
        m[f"nFine{U.title()}"] = str(len(S))

        D = macro_data.load_matrices("macro", c, verbose=False)
        cols = [str(t) for t in D["tops"]]
        n = np.asarray(D["Y"].sum(0)).ravel()
        j = cols.index(mt.BOILERPLATE_CODE)
        m[f"pctBoiler{U.title()}"] = f"{100.0 * n[j] / n.sum():.1f}"

        Sr, _, w = series(c)
        ix = Sr["macro"].index.intersection(Sr["micro"].index)
        m[f"rMacroMicro{U.title()}"] = (
            f"{wcorr(Sr['macro'].reindex(ix), Sr['micro'].reindex(ix)):.2f}")
        key = f"gpt_{GPT_MODEL}_{GPT_PROMPT}"
        ix = Sr["macro"].index.intersection(Sr[key].index)
        m[f"rMacroGpt{U.title()}"] = (
            f"{wcorr(Sr['macro'].reindex(ix), Sr[key].reindex(ix), w.reindex(ix)):.2f}")

        F = np.load(IRT / f"joint_irt_fit_gpt_{GPT_MODEL}_{GPT_PROMPT}_{c}_godambe.npz",
                    allow_pickle=True)
        m[f"nGptDrop{U.title()}"] = str(1000 - int(F["n_docs"]))

        t = pd.read_csv(REPO / "data" / "gpt_topics"
                        / f"topics_{GPT_MODEL}_{c}_sample.csv", usecols=["code"])
        sh = t.code.value_counts() / len(t) * 100.0
        m[f"pctGptBoiler{U.title()}"] = f"{sh.get(mt.BOILERPLATE_CODE, 0.0):.1f}"
        if c == "br":
            m["pctGptIntlBr"] = f"{sh.get('INT', 0.0):.2f}"
            m["pctGptImmBr"] = f"{sh.get('IMM', 0.0):.2f}"

    m["gptModelName"] = GPT_NAME[GPT_MODEL]
    lines = ["%% auto-generated by code/50G_fig_appendix.py -- do not edit"]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in m.items()]
    (TAB / "alt_resolution_numbers.tex").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")
    print(f"wrote alt_resolution_numbers.tex ({len(m)} macros)")


def main():
    fig_fit_agreement()
    fig_model_agreement()
    write_numbers()
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
