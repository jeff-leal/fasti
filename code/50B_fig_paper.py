from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
IRT = REPO / "data" / "irt"
OUT = REPO / "paper" / "Figures"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE))
import paper_style  # noqa: E402
import macro_topics as mt  # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


irt_data = _by_path("irt_data07", HERE / "04_irt_data.py")
macro_data = _by_path("macro_irt_data", HERE / "04B_macro_irt_data.py")

#: the main column scheme; every main-text figure is drawn on it
MAIN_LEVEL = "macro"
#: the subtopic scheme the appendix gap figures use
FINE_LEVEL = "micro_nobp"

BLUE, RED, GRAY, INK, TEAL = "#4C72B0", "#C44E52", "#C9C9C9", "#222222", "#3E7C8B"
REF_GRAY = "#555555"        # solid dark-gray reference line at parity / zero
BR_LEFT_CUT = irt_data.BR_LEFT_CUT
MIN_DOCFREQ = 50            # topics too rare to interpret are not plotted
N_BARS = 10
Z95 = 1.959964              # normal quantile for a 95% CI whisker

# Format/register ("boilerplate") topics: document structure, naming, and
# templated framing rather than policy. They are grouped into the boilerplate
# class like any other topic, and both fits drawn here drop that class before
# estimating (see 05A_joint_irt_fit.py and 04B_macro_irt_data.py), so nothing is
# excluded at drawing time. See sections/04_data for the rationale (CampaignView
# is curated; the BR corpus is an admin record).

# Gap-figure type at article scale (the poster design, article type sizes).
FS_LBL, FS_VAL = 9, 8

TITLES = {"us": "US Congress Candidates", "br": "Brazil Mayoral Candidates"}

# Long labels eat the width the bars need. Abbreviated for display only (never
# truncated, and the codebook label itself is unchanged).  Same map as the poster.
LABEL_SHORT = {"Governance Principles (Efficiency)": "Governance (Efficiency)",
               "Governance Principles (Planning)": "Governance (Planning)",
               "Section Headers & Boilerplate": "Sec. Headers & Boilerplate",
               "Candidate Registration/Cover": "Cand. Registration/Cover",
               "Federal Education Programs": "Federal Educ. Programs",
               "Second Amendment": "2nd Amendment",
               "Violence Against Women": "Violence Ag. Women",
               "Community Health Teams": "Community Health Teams",
               "Table Of Contents": "Table of Contents"}


def fit_path(level, corpus, variant="godambe"):
    return IRT / (f"joint_irt_fit_{corpus}_{variant}.npz" if level == "micro"
                  else f"joint_irt_fit_{level}_{corpus}_{variant}.npz")


def se_path(level, corpus):
    return IRT / (f"joint_irt_se_{corpus}.npz" if level == "micro"
                  else f"joint_irt_se_{level}_{corpus}.npz")


def load(corpus, level=MAIN_LEVEL):
    """Parameters of the Godambe-calibrated fit; already oriented so + = right.

    Dropping a column can empty a document, so the returned ``D`` is aligned to
    the documents the fit actually kept, and its columns to the columns the fit
    actually used.
    """
    D = dict(macro_data.load_matrices(level, corpus, verbose=False))
    F = np.load(fit_path(level, corpus), allow_pickle=True)
    if not bool(F["oriented"]):
        raise RuntimeError(f"{corpus}: fit is not oriented; rerun 05A_joint_irt_fit.py")

    if "docs" in F.files:                       # align documents to the fit
        keep = np.isin(np.asarray(D["docs"]).astype(str), F["docs"].astype(str))
        n = len(D["docs"])
        for k, v in list(D.items()):
            if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
                D[k] = v[keep]
        D["Y"], D["S"] = D["Y"][keep].tocsr(), D["S"][keep].tocsr()
        D["N"] = int(keep.sum())
    if "columns" in F.files:                    # align columns to the fit
        cols = np.asarray(F["columns"]).astype(str)
        have = np.asarray(D["tops"]).astype(str)
        take = np.array([np.flatnonzero(have == c)[0] for c in cols])
        for k in ("tops", "kw", "block", "n_fine", "docfreq"):
            if k in D and isinstance(D[k], np.ndarray) and len(D[k]) == len(have):
                D[k] = np.asarray(D[k])[take]
        D["Y"], D["S"] = D["Y"][:, take].tocsr(), D["S"][:, take].tocsr()
        D["T"] = len(cols)
    D["level"] = level
    D["docfreq"] = np.asarray((D["Y"] > 0).sum(0)).ravel().astype(int)
    return (F["theta"].astype(float), F["alpha"].astype(float),
            F["beta"].astype(float), F["ownership"].astype(float),
            F["curvature"].astype(float), D)


def _round10(n):
    """Group sizes in a legend read in round tens, never in raw counts."""
    return max(10, int(round(float(n) / 10.0)) * 10)


def groups(corpus, D, th):
    if corpus == "us":
        p = D["party"]
        return [("Democrats", th[p == "Democrat"], BLUE),
                ("Republicans", th[p == "Republican"], RED)]
    ext = D["ext"]
    ok = np.isfinite(ext)
    return [("Left Parties", th[ok & (ext <= BR_LEFT_CUT)], BLUE),
            ("Center & Right Parties", th[ok & (ext > BR_LEFT_CUT)], RED)]


# --------------------------------------------------------------------------- #
# Ideal-point distributions: percent histograms per group, as on the poster
# (theta_dist_us_br), at article scale.  No kernel densities.
# --------------------------------------------------------------------------- #
def _fmt(t):
    return rf"${t:.0f}$" if abs(t - round(t)) < 1e-9 else rf"${t:.1f}$"


def _hist_panel(ax, groups_z, title, bounds, xlabel, ylabel=True):
    """Overlaid percent histograms with clipped edge bins, poster style.

    `groups_z` is a list of (name, values-in-SD-units, color).
    """
    lo, hi = bounds
    nb = 36
    w = (hi - lo) / nb
    edges = np.concatenate([[lo - w], np.linspace(lo, hi, nb + 1), [hi + w]])
    for name, gz, col in groups_z:
        gz = np.clip(gz, lo - w / 2, hi + w / 2)
        ax.hist(gz, bins=edges, weights=np.full(len(gz), 100.0 / len(gz)),
                color=col, alpha=.6, label=name)
    ax.axvline(0, color="0.35", ls="--", lw=.8, zorder=0)
    ax.set_xlim(lo - w, hi + w)

    # Edge ticks read "<=lo" / ">=hi": interior ticks keep clear of them.
    guard = 0.16 * ((hi - lo) + 2 * w)
    interior = [t for t in MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
                if (t - lo) > guard and (hi - t) > guard]

    def edge(sym, t):
        return rf"${sym}\!{t:.0f}$" if abs(t - round(t)) < 1e-9 else rf"${sym}\!{t:.1f}$"

    ax.set_xticks([lo - w / 2] + interior + [hi + w / 2])
    ax.set_xticklabels([edge(r"\leq", lo)] + [_fmt(t) for t in interior]
                       + [edge(r"\geq", hi)])
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel("Frequency (%)")
    ax.set_title(title, fontsize=10, pad=4)
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)


def fig_theta_dist(data):
    paper_style.apply(base=10)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, corpus, bounds, ylab in [(axes[0], "us", (-2.5, 2.5), True),
                                     (axes[1], "br", (-4.0, 4.0), False)]:
        th, *_, D = data[(corpus, MAIN_LEVEL)]
        gz = [(name, (g - th.mean()) / th.std(), col)
              for name, g, col in groups(corpus, D, th)]
        _hist_panel(ax, gz, TITLES[corpus], bounds,
                    r"Estimated Ideal Point ($\theta$, SD units)", ylab)
    fig.tight_layout()
    for ax in axes:                    # color key sits under each panel's axis label
        _legend_under(ax, frameon=False, ncol=2, fontsize=8, columnspacing=1.6,
                      handletextpad=0.5, handlelength=1.3)
    _check_legend_clear(fig, axes, "theta dist")
    fig.savefig(OUT / "fig_theta_dist.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote fig_theta_dist.pdf")


def fig_benchmark_dist(data):
    """US positions by party under the two external benchmarks, in the exact
    format of fig_theta_dist (appendix companion to that figure)."""
    paper_style.apply(base=10)
    th, *_, D = data[("us", MAIN_LEVEL)]
    party = D["party"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    panels = [("nominate", "DW-NOMINATE", True), ("cfscore", "CFScores", False)]
    for ax, (col, ttl, ylab) in zip(axes, panels):
        v = D[col].astype(float)
        ok = np.isfinite(v)
        z = (v - v[ok].mean()) / v[ok].std()
        gz = [("Democrats", z[ok & (party == "Democrat")], BLUE),
              ("Republicans", z[ok & (party == "Republican")], RED)]
        _hist_panel(ax, gz, ttl, (-2.5, 2.5), f"{ttl} (SD units)", ylab)
    fig.tight_layout()
    for ax in axes:
        _legend_under(ax, frameon=False, ncol=2, fontsize=8, columnspacing=1.6,
                      handletextpad=0.5, handlelength=1.3)
    _check_legend_clear(fig, axes, "benchmark dist")
    fig.savefig(OUT / "fig_benchmark_dist.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote fig_benchmark_dist.pdf")


# --------------------------------------------------------------------------- #
# Stance composition, one corpus per figure.
# --------------------------------------------------------------------------- #
def _figlegend_under(fig, axes, handles, gap_pt=4, **kw):
    """One shared key, 4pt under the lowest axis label."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    y0 = min(ax.xaxis.label.get_window_extent(rend).y0 for ax in axes)
    y = (y0 - gap_pt / 72.0 * fig.dpi) / fig.get_window_extent().height
    kw.setdefault("borderpad", 0.1)
    return fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), **kw)


def fig_stance_by_topic(data):
    """100%-stacked Left/Neutral/Right composition per topic, sorted by left
    share: both corpora side by side in one figure, as on the poster."""
    paper_style.apply(base=10)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, corpus in zip(axes, ("us", "br")):
        th, alpha, beta, rho, curv, D = data[(corpus, MAIN_LEVEL)]
        Y, S = D["Y"], D["S"]
        n = np.asarray(Y.sum(0)).ravel()             # sentences per topic
        s = np.asarray(S.sum(0)).ravel()             # signed sum = n_right - n_left
        absS = np.asarray(abs(S).sum(0)).ravel()     # n_right + n_left
        right = (absS + s) / 2.0
        left = (absS - s) / 2.0
        neutral = n - absS
        keep = n >= 200
        L, N_, R = left[keep] / n[keep], neutral[keep] / n[keep], right[keep] / n[keep]
        o = np.argsort(-L)
        y = np.arange(int(keep.sum()))

        ax.barh(y, L[o], color=BLUE, height=1.0)
        ax.barh(y, N_[o], left=L[o], color=GRAY, height=1.0)
        ax.barh(y, R[o], left=L[o] + N_[o], color=RED, height=1.0)
        ax.axvline(0.5, color=INK, ls=":", lw=1.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(y) - 0.5)
        ax.invert_yaxis()
        ax.set_yticks([])
        ax.set_xlabel("Share of sentences")
        if corpus == "us":
            ax.set_ylabel("Topics")
        ax.set_title(TITLES[corpus], fontsize=10, pad=4)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    _figlegend_under(fig, axes, [Patch(color=BLUE, label="Left"),
                                 Patch(color=GRAY, label="Neutral"),
                                 Patch(color=RED, label="Right")],
                     ncol=3, frameon=False, fontsize=8, columnspacing=1.6,
                     handlelength=1.2, handletextpad=0.5)
    fig.savefig(OUT / "fig_stance_by_topic.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote fig_stance_by_topic.pdf")


# --------------------------------------------------------------------------- #
# Stance-gap and salience-gap figures: the poster formatting (fig_poster.py),
# one corpus per figure.  Assigned topic labels, sandwich-SE whiskers, value
# annotations; loadings expressed per SD of theta.
# --------------------------------------------------------------------------- #
def topic_labels(corpus, tops, level=MAIN_LEVEL):
    """Display labels for the fitted columns.

    At the macro level a column is a class of the codebook and carries its own
    label; at the fine level it is a topic and carries the label the author
    assigned it by hand.
    """
    if level == MAIN_LEVEL:
        return np.array([LABEL_SHORT.get(mt.LABEL[str(c)], mt.LABEL[str(c)])
                         for c in tops])
    lut = mt.topic_labels(corpus)
    out = []
    for t in tops:
        lab = str(lut.get(int(t), "")).strip()
        lab = lab if lab and lab.lower() != "nan" else f"Topic {int(t)}"
        out.append(LABEL_SHORT.get(lab, lab))
    return np.array(out)


def loadings_se(corpus, D, level=MAIN_LEVEL):
    """Godambe sandwich SEs for beta_t and rho_t, cached next to the fit."""
    cache = se_path(level, corpus)
    if cache.exists():
        z = np.load(cache)
        return z["beta_se"].astype(float), z["ownership_se"].astype(float)

    import joint_irt as jirt
    F = np.load(fit_path(level, corpus), allow_pickle=True)
    Yc = jirt._canonical_csr(D["Y"])
    s = jirt._prepare_stance(Yc, D["S"])
    rows = jirt._active_rows(Yc)
    cols = Yc.indices.astype(np.int64)
    y = Yc.data
    yplus = np.asarray(Yc.sum(axis=1)).ravel().astype(float)
    kap = float(F["kappa_dm"])
    d = (np.ones_like(yplus) if not np.isfinite(kap)
         else np.maximum((yplus + kap) / (1.0 + kap), 1.0))
    tau2 = float(F["tau2_curv_std"]) if "tau2_curv_std" in F.files else np.inf
    sand = jirt._joint_sandwich(
        F["theta"].astype(float), Yc, s, np.ones(Yc.shape[0]), F["alpha"].astype(float),
        F["beta"].astype(float), float(F["sigma2"]), F["lam"].astype(float),
        F["ownership"].astype(float), F["curvature"].astype(float), 1.0 / d,
        float(F["c_stance"]), float(F["c_count"]), rows, cols, y, yplus, 32768, tau2)
    np.savez(cache, beta_se=sand["beta_se"], ownership_se=sand["ownership_se"])
    print(f"  computed sandwich SEs -> {cache.name}")
    return sand["beta_se"].astype(float), sand["ownership_se"].astype(float)


def ranked(corpus, D, metric, topN=N_BARS, level=MAIN_LEVEL):
    """Top columns on a loading, with labels, SEs and the annotation string.

    Loadings are expressed per SD of theta, so |beta_t| sigma_theta is the stance
    change and e^{|rho_t| sigma_theta} the coverage multiplier for a candidate one
    SD toward the side that emphasizes the topic.  `val` keeps the signed loading
    (the sign picks the panel and the colour); `annot` carries the magnitude.
    Boilerplate is already out of the fit, so nothing is excluded here beyond the
    fine-level coverage floor, which the macro classes never come near.
    """
    F = np.load(fit_path(level, corpus), allow_pickle=True)
    sd = float(F["theta"].astype(float).std())
    beta = F["beta"].astype(float) * sd
    rho = F["ownership"].astype(float) * sd
    se_b, se_r = loadings_se(corpus, D, level)
    se_b, se_r = se_b * sd, se_r * sd

    tops = np.asarray(D["tops"])
    keep = (np.asarray(D["docfreq"]) >= (MIN_DOCFREQ if level != MAIN_LEVEL else 0))
    df = pd.DataFrame({"topic": tops, "label": topic_labels(corpus, tops, level),
                       "beta": beta, "rho": rho, "se_beta": se_b,
                       "se_rho": se_r})[keep]
    if metric in ("beta", "beta_low"):
        asc = metric == "beta_low"
        r = df.reindex(df.beta.abs().sort_values(ascending=asc).index).head(topN).copy()
        r["val"] = r.beta.abs()
        r["se"] = r.se_beta.fillna(0.0)
        r["annot"] = r.val.map(lambda b: f"{b:.2f}")
        r["color"] = TEAL
    else:
        right = df[df.rho > 0].sort_values("rho", ascending=False).head(topN)
        left = df[df.rho < 0].sort_values("rho").head(topN)
        r = pd.concat([right, left]).sort_values("rho", ascending=False).copy()
        r["val"] = r.rho
        r["se"] = r.se_rho.fillna(0.0)
        # A topic the left emphasizes has a gap below one, so printing e^{rho_t
        # sigma_theta} there runs the numbers backwards against the bars: the
        # strongest topic gets the smallest number.  Print e^{|rho_t| sigma_theta}
        # instead, read per SD toward that panel's own side.  On the log axis
        # that is the exact mirror of the right panel, so in both panels the
        # multiplier is at least one and is monotone in bar length.
        r["annot"] = np.exp(r.rho.abs()).map(lambda e: f"×{e:.2f}")
        r["color"] = np.where(r.rho >= 0, RED, BLUE)
    return r.reset_index(drop=True)


def _legend_under(ax, gap_pt=4, **kw):
    """Put the key just under the axis label — a 4pt gap, measured, not guessed."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    y0 = ax.xaxis.label.get_window_extent(rend).y0 - gap_pt / 72.0 * fig.dpi
    y = ax.transAxes.inverted().transform((0, y0))[1]
    kw.setdefault("borderpad", 0.1)
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), **kw)


def _ticks_in_view(ticks, labels, lim):
    """A locator can emit ticks outside the view; their labels are never drawn,
    so they must not be counted as collisions."""
    lo, hi = sorted(lim)
    return [b for t, b in zip(ticks, labels) if lo - 1e-9 <= t <= hi + 1e-9]


def _check_ticks(fig, axes, what, gap=1.0):
    """Every drawn string must clear every other one.

    Measured, not eyeballed: the renderer's own bounding box for each visible
    text artist, compared pairwise within a panel and across panels.  Row labels
    that touch, a title sitting on a bar, an annotation running into a
    neighbouring row all surface here.
    """
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
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][0].padded(-gap).overlaps(boxes[j][0].padded(-gap)):
                bad.append(f"{boxes[i][1]!r}/{boxes[j][1]!r}")
    print(f"  [{'OK' if not bad else 'CRAMPED'}] {what}: {len(bad)} text collisions"
          + (f" -> {'; '.join(bad[:4])}" if bad else ""))
    return len(bad)


def _check_legend_clear(fig, axes, what, gap=2):
    """The colour key must not touch the axis labels or the panel."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bad = 0
    for ax in axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(rend)
        for other in (ax.xaxis.label, ax.yaxis.label, ax.title):
            ob = other.get_window_extent(rend)
            if ob.width and lb.expanded(1.0, 1.0).overlaps(ob.padded(gap)):
                bad += 1
        if lb.overlaps(ax.get_window_extent(rend)):
            bad += 1
    print(f"  [{'OK' if not bad else 'OVERLAP'}] {what}: {bad} legend clashes")
    return bad


def draw_coefs(ax, r, metric):
    n = len(r)
    y = np.arange(n)[::-1]
    ekw = dict(ecolor="#111111", elinewidth=1.2, capsize=2.5, capthick=1.2, zorder=6)
    val, se = r.val.values, r.se.values
    ax.barh(y, val, xerr=se, color=r.color, alpha=0.55, height=0.8, zorder=3,
            error_kw=ekw)
    pad = (np.abs(val) + se).max() * 0.02
    for yi, v, s, a in zip(y, val, se, r.annot):
        ax.text(v + s + pad, yi, a, va="center", ha="left", fontsize=FS_VAL, color=INK)
    ax.set_xlim(0, (val + se).max() * 1.20)
    ax.set_yticks(y)
    ax.set_yticklabels(r.label, fontsize=FS_LBL)
    ax.set_ylim(-0.65, n - 0.35)
    ax.set_xlabel(r"Stance gap  $(|\beta_t|)$", fontsize=FS_LBL + 1)
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.tick_params(length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


# Salience-figure type: 11pt labels on a 12pt row pitch (bars nearly touching),
# so 15 rows per side stay legible without the figure growing long.
SAL_LBL, SAL_VAL = 11, 10
SAL_PITCH_IN = 12.0 / 72.0


def _fit_annots(fig, axes, stretch, what, pad_pt=2.0, gap_pt=6.0, tol_px=0.25,
                tries=12):
    """Widen both panels until every printed value clears the topic labels.

    A long confidence interval pushes its annotation past the axis and into the
    topic label on its row.  The type never shrinks to fix that: the axis grows
    instead, by the shortfall the renderer actually measures.  `stretch(k)`
    reapplies the shared limit stretched by the factor k, so the two sides stay on
    one scale.  Two clearances are enforced — `pad_pt` inside the axis edge, and
    `gap_pt` of real white space from the tick label on the same row, which is
    what a reader sees.  Widening moves the axis edge and the annotation apart at
    slightly different rates, so the clearance approaches its target from below
    instead of crossing it; `tol_px` is the sub-pixel slack that stops the loop.
    """
    for _ in range(tries):
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        over, width = 0.0, 1.0
        for ax in axes:
            ab = ax.get_window_extent(rend)
            width = max(width, ab.width)
            labs = [t.get_window_extent(rend) for t in ax.get_yticklabels()]
            for t in ax.texts:
                tb = t.get_window_extent(rend)
                over = max(over, ab.x0 + pad_pt - tb.x0, tb.x1 + pad_pt - ab.x1)
                for lb in labs:
                    if tb.y1 <= lb.y0 or lb.y1 <= tb.y0:
                        continue                      # a different row
                    if lb.x1 <= ab.x0 + 1.0:          # labels sit left of the panel
                        over = max(over, lb.x1 + gap_pt - tb.x0)
                    elif lb.x0 >= ab.x1 - 1.0:        # labels sit right of the panel
                        over = max(over, tb.x1 + gap_pt - lb.x0)
        if over <= tol_px:
            print(f"  [OK] {what}: every value clear of its label")
            return True
        stretch(1.0 + over / width)
    print(f"  [CROWDED] {what}: a value still crowds its label")
    return False


def _salience_panel(ax, sub, color, side, M, n):
    """One column of the two-column salience figure: a diverging bar from parity
    to each topic's salience gap, with a 95% CI cap, on a shared multiplicative
    (log) axis, anchored by a solid dark-gray reference at parity (no salience
    difference), strongest topic on top.  The bar is drawn at the signed gap; the
    printed multiplier is its magnitude, per SD toward this panel's own side."""
    xv = np.exp(sub.val.values)
    lo = np.exp(sub.val.values - Z95 * sub.se.values)
    hi = np.exp(sub.val.values + Z95 * sub.se.values)
    y = np.arange(n - 1, n - 1 - len(sub), -1)
    ax.set_xscale("log")
    ax.barh(y, xv - 1.0, left=1.0, color=color, alpha=0.55, height=0.62,
            edgecolor="none", zorder=3)
    ax.errorbar(xv, y, xerr=[xv - lo, hi - xv], fmt="none", ecolor="#333333",
                elinewidth=1.1, capsize=2.2, capthick=1.0, zorder=5)
    ax.axvline(1.0, color=REF_GRAY, lw=1.1, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(sub.label, fontsize=SAL_LBL)
    ax.set_ylim(-0.6, n - 0.4)
    if side == "left":
        ax.set_xlim(1.0 / M, 1.0)
        for yi, l, a in zip(y, lo, sub.annot):
            ax.text(l / 1.10, yi, a, va="center", ha="right",
                    fontsize=SAL_VAL, color=INK)
        ax.set_title("Emphasized by the left", fontsize=SAL_LBL, pad=4)
    else:
        ax.set_xlim(1.0, M)
        ax.yaxis.tick_right()
        for yi, h, a in zip(y, hi, sub.annot):
            ax.text(h * 1.10, yi, a, va="center", ha="left",
                    fontsize=SAL_VAL, color=INK)
        ax.set_title("Emphasized by the right", fontsize=SAL_LBL, pad=4)
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.tick_params(length=0)
    for s_ in ("top", "right", "left", "bottom"):
        ax.spines[s_].set_visible(False)


def fig_salience_gap(data, topN=15):
    """Two-column salience figure: the topics the left emphasizes most in a left
    panel (bars extending left of parity), those the right emphasizes most in a
    right panel (extending right), on one shared multiplicative axis.

    At the macro level the whole taxonomy fits on the page, so every class is
    drawn and nothing is clipped; at the fine level the panels keep the fifteen
    strongest on each side.
    """
    for corpus in ("us", "br"):
        for suffix, level in (("", MAIN_LEVEL), ("_micro", FINE_LEVEL)):
            paper_style.apply(base=10)
            th, alpha, beta, rho, curv, D = data[(corpus, level)]
            n_show = 10 ** 6 if level == MAIN_LEVEL else topN
            r = ranked(corpus, D, "rho", n_show, level=level)
            left = r[r.val < 0].sort_values("val").reset_index(drop=True)
            right = r[r.val >= 0].sort_values("val", ascending=False).reset_index(drop=True)
            n = max(len(left), len(right))
            xl = np.exp(left.val.values - Z95 * left.se.values)
            xr = np.exp(right.val.values + Z95 * right.se.values)
            M = max(xr.max() if len(xr) else 1.0,
                    (1.0 / np.maximum(xl, 1e-3)).max() if len(xl) else 1.0) * 1.4
            fig, (axL, axR) = plt.subplots(1, 2,
                                           figsize=(6.5, SAL_PITCH_IN * n + 0.75),
                                           gridspec_kw={"wspace": 0.08})
            _salience_panel(axL, left, BLUE, "left", M, n)
            _salience_panel(axR, right, RED, "right", M, n)
            fig.supxlabel(r"Salience gap  $(e^{|\rho_t| \sigma_\theta})$",
                          fontsize=SAL_LBL)
            fig.tight_layout()

            box = [M]                     # stretching is multiplicative in log M
            def stretch(k, box=box, axL=axL, axR=axR):
                box[0] = float(np.exp(np.log(box[0]) * k))
                axL.set_xlim(1.0 / box[0], 1.0)
                axR.set_xlim(1.0, box[0])
            _fit_annots(fig, (axL, axR), stretch, f"salience {corpus}{suffix or ''}")
            _check_ticks(fig, (axL, axR), f"salience {corpus}{suffix or ''}")
            name = f"fig_salience_gap_{corpus}{suffix}.pdf"
            fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            print(f"wrote {name}")


def _stancegap_panel(ax, sub, side, M, n):
    """One column of the two-column stance-gap figure: a diverging bar from zero
    to each topic's stance gap, with a 95% CI cap, anchored by a solid dark-gray
    zero line, the panel's extreme topic at the top.  Teal in both panels: the
    gap is a magnitude, not an ideological direction."""
    xv, se = sub.val.values, sub.se.values
    y = np.arange(n - 1, n - 1 - len(sub), -1)
    ax.barh(y, xv, left=0.0, color=TEAL, alpha=0.55, height=0.62,
            edgecolor="none", zorder=3)
    ax.errorbar(xv, y, xerr=Z95 * se, fmt="none", ecolor="#333333",
                elinewidth=1.1, capsize=2.2, capthick=1.0, zorder=5)
    ax.axvline(0.0, color=REF_GRAY, lw=1.1, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(sub.label, fontsize=SAL_LBL)
    ax.set_ylim(-0.6, n - 0.4)
    if side == "left":
        ax.set_xlim(M, 0.0)          # inverted axis: dots sit to the left
        for yi, x, s, a in zip(y, xv, se, sub.annot):
            ax.text(x + Z95 * s + M * 0.02, yi, a, va="center", ha="right",
                    fontsize=SAL_VAL, color=INK)
        ax.set_title("Most divisive", fontsize=SAL_LBL, pad=4)
    else:
        ax.set_xlim(0.0, M)
        ax.yaxis.tick_right()
        for yi, x, s, a in zip(y, xv, se, sub.annot):
            ax.text(x + Z95 * s + M * 0.02, yi, a, va="center", ha="left",
                    fontsize=SAL_VAL, color=INK)
        ax.set_title("Least divisive", fontsize=SAL_LBL, pad=4)
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.tick_params(length=0)
    for s_ in ("top", "right", "left", "bottom"):
        ax.spines[s_].set_visible(False)


def fig_topic_coefs(data, topN=15):
    """Per-corpus stance-gap figures in the two-column salience layout: the
    largest stance gaps in a left panel (bars extending left of zero, largest at
    the top), the smallest in a right panel (extending right, smallest at the
    top), on one shared magnitude axis.

    At the macro level the two panels partition the whole taxonomy, so every
    class appears exactly once and none is clipped; at the fine level they show
    the fifteen strongest and the fifteen weakest.
    """
    for corpus in ("us", "br"):
        for suffix, level in (("", MAIN_LEVEL), ("_micro", FINE_LEVEL)):
            paper_style.apply(base=10)
            th, alpha, beta, rho, curv, D = data[(corpus, level)]
            if level == MAIN_LEVEL:
                K = int(len(ranked(corpus, D, "beta", 10 ** 6, level=level)))
                n_hi, n_lo = (K + 1) // 2, K // 2
            else:
                n_hi = n_lo = topN
            hi = ranked(corpus, D, "beta", n_hi, level=level)
            lo = ranked(corpus, D, "beta_low", n_lo, level=level)
            n = max(len(hi), len(lo))
            M = max((hi.val + Z95 * hi.se).max(), (lo.val + Z95 * lo.se).max()) * 1.25
            fig, (axL, axR) = plt.subplots(1, 2,
                                           figsize=(6.5, SAL_PITCH_IN * n + 0.75),
                                           gridspec_kw={"wspace": 0.08})
            _stancegap_panel(axL, hi, "left", M, n)
            _stancegap_panel(axR, lo, "right", M, n)
            fig.supxlabel(r"Stance gap  $(|\beta_t|)$", fontsize=SAL_LBL)
            fig.tight_layout()

            box = [M]                     # the magnitude axis stretches linearly
            def stretch(k, box=box, axL=axL, axR=axR):
                box[0] *= k
                axL.set_xlim(box[0], 0.0)     # inverted: bars run leftward
                axR.set_xlim(0.0, box[0])
            _fit_annots(fig, (axL, axR), stretch, f"stance gap {corpus}{suffix or ''}")
            _check_ticks(fig, (axL, axR), f"stance gap {corpus}{suffix or ''}")
            name = f"fig_stance_gap_{corpus}{suffix}.pdf"
            fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            print(f"wrote {name}")
    fig_salience_gap(data, 15)


# --------------------------------------------------------------------------- #
# External-validity scatters: the poster's validation panel (validation_us_br)
# at article scale.  US: candidate theta against DW-NOMINATE, colored by party.
# BR: party-mean theta against the expert scores, marker area = platforms per
# party, acronyms routed clear of the dots by the poster's deterministic
# placer.  Ideal points are z-scored for display only (correlations are
# scale-free).  Axis labels carry no poster reference numbers; the sources are
# cited in the LaTeX figure note.
# --------------------------------------------------------------------------- #
def fig_validation_scatter(data):
    from scipy.stats import pearsonr
    sys.path.insert(0, str(HERE / "report"))
    import fig_poster as poster

    paper_style.apply(base=10)
    thU_r, _, _, _, _, DU = data[("us", MAIN_LEVEL)]
    thU = (thU_r - thU_r.mean()) / thU_r.std()
    nom = DU["nominate"].astype(float)
    ok = np.isfinite(nom)
    rU = pearsonr(thU[ok], nom[ok])[0]

    thB_r, _, _, _, _, DB = data[("br", MAIN_LEVEL)]
    thB = (thB_r - thB_r.mean()) / thB_r.std()
    ext, partyB = DB["ext"], DB["party"]
    okb = np.isfinite(ext)
    pm = (pd.DataFrame({"party": partyB[okb], "theta": thB[okb]})
          .groupby("party").theta.agg(["mean", "size"]))
    parties = pm.index.to_numpy()
    xB = np.array([ext[partyB == p][0] for p in parties], float)
    yB = pm["mean"].to_numpy()
    nB = pm["size"].to_numpy()
    rB = pearsonr(xB, yB)[0]

    def msize(n):
        # Same size map as the poster, halved for the article canvas.
        return 15.0 + 2.0 * np.sqrt(n)

    fig, (axU, axB) = plt.subplots(1, 2, figsize=(6.5, 3.1))

    party = DU["party"]
    for name, col in [("Democrat", BLUE), ("Republican", RED)]:
        m = ok & (party == name)
        axU.scatter(nom[m], thU[m], s=4, alpha=.35, color=col, label=name, lw=0)
    b1, b0 = np.polyfit(nom[ok], thU[ok], 1)
    xs = np.array([np.nanmin(nom), np.nanmax(nom)])
    axU.plot(xs, b1 * xs + b0, color="k", lw=1.0, ls="--", dashes=(6, 4), zorder=5)
    axU.set_xlabel("DW-NOMINATE")
    axU.set_ylabel(r"Estimated Ideal Point ($\theta$)")
    axU.set_title(TITLES["us"], fontsize=10, pad=4)
    axU.text(0.045, 0.955, f"r = {rU:.2f}", transform=axU.transAxes, va="top",
             ha="left", fontsize=9)
    axU.margins(0.03)
    axU.grid(alpha=.18)

    axB.scatter(xB, yB, s=msize(nB), color=BLUE, edgecolor="k", lw=.5, zorder=3)
    b1, b0 = np.polyfit(xB, yB, 1)
    xs = np.array([xB.min(), xB.max()])
    axB.plot(xs, b1 * xs + b0, color="k", lw=1.0, ls="--", dashes=(6, 4), zorder=5)
    axB.set_xlabel("Expert Scores")
    axB.set_ylabel(r"Party-Mean Ideal Point ($\theta$)")
    axB.set_title(TITLES["br"], fontsize=10, pad=4)
    axB.text(0.045, 0.955, f"r = {rB:.2f}", transform=axB.transAxes, va="top",
             ha="left", fontsize=9)
    axB.grid(alpha=.18)
    axB.margins(0.06)
    axB.autoscale_view()
    x0, x1 = axB.get_xlim()
    axB.set_xlim(x0 - 0.04 * (x1 - x0), x1 + 0.24 * (x1 - x0))
    y0, y1 = axB.get_ylim()
    axB.set_ylim(y0 - 0.14 * (y1 - y0), y1 + 0.14 * (y1 - y0))

    # The size key is a reading aid, not a count of any particular party, so it
    # is printed in round tens rather than in the exact sizes it happens to sit on.
    ticks = sorted({_round10(v) for v in (nB.min(), np.median(nB), nB.max())})
    fig.tight_layout(w_pad=2.0)
    _legend_under(axU, ncol=2, frameon=False, markerscale=2.2,
                  handletextpad=.3, columnspacing=1.8, fontsize=8)
    leg = _legend_under(axB,
                        handles=[plt.Line2D([], [], ls="none", marker="o", color=BLUE,
                                            markeredgecolor="k", markeredgewidth=.5,
                                            markersize=np.sqrt(msize(n)), label=f"{n:,}")
                                 for n in ticks],
                        title="Platforms per party", ncol=3, frameon=False,
                        fontsize=7, title_fontsize=7,
                        columnspacing=1.4, handletextpad=0.6)
    texts = poster._place_labels(fig, axB, xB, yB, msize(nB),
                                 [poster.BR_LABEL_ABBR.get(p, p) for p in parties],
                                 avoid=[leg], fontsize=6.5)
    poster._report_overlaps(fig, axB, texts, "BR party labels", avoid=[leg],
                            marks=poster._marker_boxes(axB, (xB, yB), msize(nB)))
    _check_legend_clear(fig, [axU, axB], "fig_validation_scatter")
    name = "fig_validation_scatter.pdf"
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {name}   US r={rU:+.3f} (n={int(ok.sum()):,})  "
          f"BR party-level r={rB:+.3f} (n={len(pm)})")


def main():
    data = {(c, lv): load(c, lv) for c in ("us", "br")
            for lv in (MAIN_LEVEL, FINE_LEVEL)}
    fig_validation_scatter(data)
    fig_theta_dist(data)
    fig_benchmark_dist(data)
    fig_stance_by_topic(data)
    fig_topic_coefs(data)
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
