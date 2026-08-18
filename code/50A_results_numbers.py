from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
IRT = REPO / "data" / "irt"

_spec = importlib.util.spec_from_file_location("irt_data07", HERE / "04_irt_data.py")
irt_data = importlib.util.module_from_spec(_spec)
sys.modules["irt_data07"] = irt_data
_spec.loader.exec_module(irt_data)

_ms = importlib.util.spec_from_file_location("macro_irt_data", HERE / "04B_macro_irt_data.py")
macro_data = importlib.util.module_from_spec(_ms)
sys.modules["macro_irt_data"] = macro_data
_ms.loader.exec_module(macro_data)

BR_LEFT_CUT = irt_data.BR_LEFT_CUT
MIN_DOCFREQ = 50          # interpretation floor for the coefficient rankings

_LINES: list[str] = []


def say(msg=""):
    print(msg)
    _LINES.append(msg)


def _labels(corpus, tops):
    """Labels aligned to the fitted columns.

    A macro column is a class of the codebook and carries its own label; a fine
    column is a topic and carries the label the author assigned it.
    """
    sys.path.insert(0, str(HERE))
    import macro_topics as mt
    if all(str(t) in mt.LABEL for t in tops):
        return [mt.LABEL[str(t)] for t in tops]
    lut = mt.topic_labels(corpus)
    return [lut.get(int(t), f"Topic {int(t)}") for t in tops]


def kappa_std(theta, curv):
    x = np.asarray(theta, float)
    x2 = x * x
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, x2, rcond=None)
    return np.asarray(curv, float) * float(np.std(x2 - A @ coef))


def overlap(a, b):
    return (100 * float((a > np.median(b)).mean()),
            100 * float((b < np.median(a)).mean()))


def rank(sel, key, n, reverse=False):
    o = sel[np.argsort(-key[sel] if reverse else key[sel])]
    return o[:n]


def report(corpus):
    F, D = macro_data.load_main(corpus)
    th, alpha, beta = F["theta"], F["alpha"], F["beta"]
    rho, curv = F["ownership"], F["curvature"]
    tops, kw, party, docfreq = D["tops"], D["kw"], D["party"], D["docfreq"]
    Y, S = D["Y"], D["S"]
    sd = float(th.std())

    say("=" * 78)
    say(f"{corpus.upper()}   N={len(th):,}  topics={len(beta)}  active cells={Y.nnz:,}")
    say(f"  sigma2={float(F['sigma2']):.4f}  kappa_dm={float(F['kappa_dm']):.2f}  "
        f"Godambe factor={float(F['godambe_factor']):.3f}   sd(theta)={sd:.4f}")

    # ---- ideal-point distribution by group ----
    if corpus == "us":
        g1, g2 = "Democrats", "Republicans"
        m1, m2 = party == "Democrat", party == "Republican"
    else:
        ext = D["ext"]
        ok = np.isfinite(ext)
        g1, g2 = "Left Parties", "Center & Right Parties"
        m1, m2 = ok & (ext <= BR_LEFT_CUT), ok & (ext > BR_LEFT_CUT)
    o1, o2 = overlap(th[m1], th[m2])
    say(f"  {g1:<14} mean {th[m1].mean():+.3f}  sd {th[m1].std():.3f}  n={m1.sum():,}")
    say(f"  {g2:<14} mean {th[m2].mean():+.3f}  sd {th[m2].std():.3f}  n={m2.sum():,}")
    gap = th[m2].mean() - th[m1].mean()
    say(f"  gap of means {gap:+.3f}  ({gap/sd:.2f} sd of theta)")
    say(f"  OVERLAP: {o1:.1f}% of {g1} above the {g2} median; "
        f"{o2:.1f}% of {g2} below the {g1} median")
    say(f"  WRONG SIDE OF ZERO: {100 * float((th[m1] > 0).mean()):.1f}% of {g1} "
        f"right of zero; {100 * float((th[m2] < 0).mean()):.1f}% of {g2} left of zero")
    say(f"  centering identity: n1*m1 + n2*m2 = {m1.sum()*th[m1].mean() + m2.sum()*th[m2].mean():+.3f}")

    # ---- agenda overlap between the two groups ----
    labs = _labels(corpus, tops)
    i1, i2 = np.flatnonzero(m1), np.flatnonzero(m2)
    n1 = np.asarray(Y[i1].sum(0)).ravel()
    n2 = np.asarray(Y[i2].sum(0)).ravel()
    st1 = np.asarray(S[i1].sum(0)).ravel() / np.maximum(n1, 1)
    st2 = np.asarray(S[i2].sum(0)).ravel() / np.maximum(n2, 1)
    sh1, sh2 = n1 / n1.sum(), n2 / n2.sum()
    nall = np.asarray(Y.sum(0)).ravel()
    say(f"  AGENDA: Spearman of coverage shares, {g1} vs {g2}: "
        f"{spearmanr(sh1, sh2)[0]:.2f} over {len(sh1)} topics")
    for k in (10, 15):
        t1, t2 = set(np.argsort(-n1)[:k]), set(np.argsort(-n2)[:k])
        say(f"    top-{k} of each group: {len(t1 & t2)} shared")
        say(f"      only {g1}: " + "; ".join(sorted(labs[j] for j in t1 - t2)))
        say(f"      only {g2}: " + "; ".join(sorted(labs[j] for j in t2 - t1)))
        top = np.argsort(-nall)[:k]
        ratio = np.maximum(sh1[top], sh2[top]) / np.minimum(sh1[top], sh2[top])
        d = np.abs(st1[top] - st2[top])
        flips = int(((st1[top] * st2[top]) < 0).sum())
        say(f"      overall top-{k}: stance mean|diff|={d.mean():.2f} "
            f"max={d.max():.2f} sign flips={flips} | salience share-ratio "
            f"mean={ratio.mean():.2f} max={ratio.max():.2f} "
            f"n>1.5x={int((ratio > 1.5).sum())}")
        say(f"      cumulative share of overall top-{k}: "
            f"{g1} {100 * sh1[top].sum():.1f}%; {g2} {100 * sh2[top].sum():.1f}%")

    # ---- stance composition of topics ----
    n = np.asarray(Y.sum(0)).ravel()
    s = np.asarray(S.sum(0)).ravel()
    ms = s / np.maximum(n, 1)
    say(f"  topic mean stance: left(<-0.1) {100*np.mean(ms<-0.1):.0f}%  "
        f"neutral {100*np.mean(np.abs(ms)<=0.1):.0f}%  right(>0.1) {100*np.mean(ms>0.1):.0f}%"
        f"   corpus mean stance {s.sum()/n.sum():+.3f}")

    sel = np.flatnonzero(docfreq >= MIN_DOCFREQ)
    ab = np.abs(beta)
    say(f"  [rankings restrict to the {len(sel)} of {len(beta)} topics in >= {MIN_DOCFREQ} documents]")

    say("  LARGEST stance gap |beta_t| (divisive):")
    for j in rank(sel, ab, 12, reverse=True):
        say(f"    beta {beta[j]:+.3f}   {str(tops[j]):<4} {kw[j]}")
    say("  SMALLEST stance gap (consensus):")
    for j in rank(sel, ab, 8):
        say(f"    beta {beta[j]:+.3f}   {str(tops[j]):<4} {kw[j]}")

    say("  most RIGHT-emphasized (coverage multiplier per +1 SD right):")
    for j in rank(sel, rho, 10, reverse=True):
        say(f"    rho {rho[j]:+.3f}  x{np.exp(rho[j]*sd):5.2f}   {str(tops[j]):<4} {kw[j]}")
    say("  most LEFT-emphasized:")
    for j in rank(sel, rho, 10):
        say(f"    rho {rho[j]:+.3f}  x{np.exp(rho[j]*sd):5.2f}   {str(tops[j]):<4} {kw[j]}")

    ks = kappa_std(th, curv)
    say("  largest |standardized curvature| (non-monotone attention):")
    for j in rank(sel, np.abs(ks), 6, reverse=True):
        say(f"    kappa~ {ks[j]:+.3f}   {str(tops[j]):<4} {kw[j]}")

    say(f"  expected stance S_it = alpha_t + beta_t * theta at the group means:")
    for j in rank(sel, ab, 8, reverse=True):
        say(f"    {str(tops[j]):<4} alpha {alpha[j]:+.2f} beta {beta[j]:+.2f}  "
            f"| {g1} {alpha[j]+beta[j]*th[m1].mean():+.2f}  "
            f"| {g2} {alpha[j]+beta[j]*th[m2].mean():+.2f}   {kw[j][:36]}")

    # ---- validation ----
    if corpus == "us":
        for name, lab in [("cfscore", "CFscore"), ("nominate", "DW-NOMINATE")]:
            b = D[name]
            ok = np.isfinite(b)
            say(f"  VALIDATION {lab}: n={ok.sum():,}  Pearson {pearsonr(th[ok], b[ok])[0]:+.3f}  "
                f"Spearman {spearmanr(th[ok], b[ok]).correlation:+.3f}")
            for g, m in [(g1, party == "Democrat"), (g2, party == "Republican")]:
                mm = ok & m
                say(f"     within {g}: n={mm.sum():,}  Pearson {pearsonr(th[mm], b[mm])[0]:+.3f}")
        # named candidates
        docs = D["docs"]
        say("  named candidates:")
        for pat in ["Ocasio-Cortez", "Ilhan Omar", "Jayapal", "Stefanik", "Taylor Greene",
                    "Liz Cheney", "Fitzpatrick", "Matt Gaetz", "Boebert", "Henry Cuellar",
                    "Jared Golden"]:
            for i, d in enumerate(docs):
                if pat.lower() in str(d).lower():
                    say(f"    {th[i]:+.3f}  {d}")
    else:
        ext = D["ext"]
        ok = np.isfinite(ext)
        parties = sorted(set(party[ok]))
        pm = np.array([th[party == p].mean() for p in parties])
        pe = np.array([ext[party == p][0] for p in parties])
        r, p = pearsonr(pm, pe)
        say(f"  VALIDATION expert scores, PARTY LEVEL: n_parties={len(parties)}  "
            f"Pearson {r:+.3f} (p={p:.1e})  Spearman {spearmanr(pm, pe).correlation:+.3f}")
        say(f"     documents with an expert score: {ok.sum():,} of {len(th):,}")
        wsd = np.mean([th[party == p].std() for p in parties])
        say(f"     mean within-party sd {wsd:.3f}  vs  between-party sd of means {pm.std():.3f}")
        say("     party means (sorted by theta):")
        for i in np.argsort(pm):
            say(f"       {parties[i]:<15} theta {pm[i]:+.3f}  expert {pe[i]:.2f}  "
                f"n={int((party == parties[i]).sum()):,}")
    say()


if __name__ == "__main__":
    for c in [a.lower() for a in sys.argv[1:]] or ["us", "br"]:
        _LINES.clear()
        report(c)
        (IRT / f"results_numbers_{c}.txt").write_text("\n".join(_LINES), encoding="utf-8")
