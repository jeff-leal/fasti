from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]
IRT = REPO / "data" / "irt"
IRT.mkdir(parents=True, exist_ok=True)

# sibling modules whose names begin with a digit must be loaded by path
sys.path.insert(0, str(HERE))
import macro_topics as mt  # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


irt_data = _by_path("irt_data07", HERE / "04_irt_data.py")
macro_data = _by_path("macro_irt_data", HERE / "04B_macro_irt_data.py")
jirt = _by_path("joint_irt", HERE / "joint_irt.py")
fit_joint_irt = jirt.fit_joint_irt

VARIANTS = [("godambe", True), ("nogodambe", False)]
LEVELS = ("macro", "micro", "micro_nobp")

#: the paper's main fit drops the boilerplate class from the macro fit. This is
#: the fit-time decision, made here and nowhere else; the macro matrices
#: themselves carry the column (see 04B_macro_irt_data.py).
DEFAULT_DROP = {"macro": (mt.BOILERPLATE_CODE,), "micro": (), "micro_nobp": ()}


def fit_path(level, corpus, variant, kind="fit", ext="npz"):
    stem = f"joint_irt_{kind}_{corpus}_{variant}" if level == "micro" else \
           f"joint_irt_{kind}_{level}_{corpus}_{variant}"
    return IRT / f"{stem}.{ext}"


def se_path(level, corpus):
    return IRT / (f"joint_irt_se_{corpus}.npz" if level == "micro"
                  else f"joint_irt_se_{level}_{corpus}.npz")


def orientation_flip(theta, D):
    """True when the fitted axis must be reflected so that + = right."""
    if D["kind"] == "us":
        b = D["cfscore"]
        ok = np.isfinite(b)
        return pearsonr(theta[ok], b[ok])[0] < 0
    ext, party = D["ext"], D["party"]
    ok = np.isfinite(ext)
    parties = sorted(set(party[ok]))
    pm = np.array([theta[party == p].mean() for p in parties])
    pe = np.array([ext[party == p][0] for p in parties])
    return spearmanr(pm, pe).correlation < 0


def prepare(D, level, drop):
    """Apply the fit-time column drop, then remove documents it empties.

    Returns ``(Y, S, columns, dropped_columns, keep_documents)``.
    """
    Y, S, cols = D["Y"], D["S"], np.asarray(D["tops"], dtype=object)
    gone: list = []
    if drop:
        Y, S, cols, gone = mt.drop_columns(Y, S, cols, drop)
    keep = np.asarray(Y.sum(axis=1)).ravel() > 0
    if not keep.all():
        Y, S = Y[keep].tocsr(), S[keep].tocsr()
        why = f"once {', '.join(gone)} was dropped" if gone else "in this column scheme"
        print(f"    {int((~keep).sum())} document(s) left with no sentences {why}, "
              f"and dropped")
    return Y, S, cols, gone, keep


def fit_one(corpus, variant, use_godambe, D, level="micro", drop=(), refit=False):
    npz = fit_path(level, corpus, variant)
    csv = fit_path(level, corpus, variant, kind="loss", ext="csv")
    if npz.exists() and csv.exists() and not refit:
        print(f"[{corpus}/{level}/{variant}] cached fit reused")
        return

    Y, S, cols, gone, keep = prepare(D, level, drop)
    Dk = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape
              else v) for k, v in D.items()}
    Dk["Y"], Dk["S"] = Y, S

    print(f"[{corpus}/{level}/{variant}] fitting N={Y.shape[0]:,} K={Y.shape[1]} "
          f"(use_godambe={use_godambe}) ...", flush=True)
    t0 = time.perf_counter()
    fit = fit_joint_irt(Y, S, orientation=1.0,
                        use_godambe=use_godambe, verbose=True)
    secs = time.perf_counter() - t0

    theta = np.asarray(fit.theta, float)
    beta = np.asarray(fit.beta, float)
    rho = np.asarray(fit.ownership, float)
    curv = np.asarray(fit.curvature, float)
    alpha = np.asarray(fit.alpha, float)
    if orientation_flip(theta, Dk):
        theta, beta, rho = -theta, -beta, -rho
        print(f"[{corpus}/{level}/{variant}] axis reflected so that + = right")

    np.savez(
        npz,
        theta=theta, theta_stance=fit.theta_stance, alpha=alpha, beta=beta,
        sigma2=fit.sigma2, lam=fit.lam, ownership=rho, curvature=curv,
        kappa_dm=fit.kappa_dm, c_stance=fit.c_stance, c_count=fit.c_count,
        godambe_factor=fit.godambe_factor, use_godambe=fit.use_godambe,
        tau2_curv_std=fit.tau2_curv_std, eb_conv=fit.eb_converged,
        doc_intercept=fit.doc_intercept, dm_status=str(fit.dm_status),
        s1_conv=fit.stage1_converged, joint_conv=fit.joint_converged,
        fallback=fit.theta_fallback_documents, fit_time=secs,
        oriented=True, n_docs=int(Y.shape[0]), n_topics=int(Y.shape[1]),
        level=level, columns=np.asarray(cols).astype(str),
        dropped_columns=np.asarray(gone, dtype=str),
        docs=np.asarray(Dk["docs"]).astype(str),
    )

    frames = []
    for stage, hist, key in [("stage1", fit.stage1_history, "sse"),
                             ("count_init", fit.count_init_history, "nll"),
                             ("curvature_eb", fit.eb_history, "tau2_curv_std"),
                             ("joint", fit.joint_history, "frozen_calibrated_nll_after")]:
        if not hist:
            continue
        d = pd.DataFrame(hist)
        d.insert(0, "stage", stage)
        d["loss"] = d[key]
        frames.append(d)
    pd.concat(frames, ignore_index=True).to_csv(csv, index=False)

    print(f"[{corpus}/{level}/{variant}] {secs:.0f}s converged={fit.joint_converged} "
          f"fallback={fit.theta_fallback_documents} -> {npz.name}")


def standard_errors(level, corpus, D, drop=(), refit=False):
    """Godambe sandwich SEs for beta_t and rho_t, cached beside the fit."""
    cache = se_path(level, corpus)
    fitfile = fit_path(level, corpus, "godambe")
    if (cache.exists() and not refit
            and cache.stat().st_mtime >= fitfile.stat().st_mtime):
        return
    Y, S, _, _, _ = prepare(D, level, drop)
    F = np.load(fitfile, allow_pickle=True)
    Yc = jirt._canonical_csr(Y)
    s = jirt._prepare_stance(Yc, S)
    rows = jirt._active_rows(Yc)
    cols = Yc.indices.astype(np.int64)
    yplus = np.asarray(Yc.sum(axis=1)).ravel().astype(float)
    kap = float(F["kappa_dm"])
    d = (np.ones_like(yplus) if not np.isfinite(kap)
         else np.maximum((yplus + kap) / (1.0 + kap), 1.0))
    tau2 = float(F["tau2_curv_std"]) if "tau2_curv_std" in F.files else np.inf
    sand = jirt._joint_sandwich(
        F["theta"].astype(float), Yc, s, np.ones(Yc.shape[0]), F["alpha"].astype(float),
        F["beta"].astype(float), float(F["sigma2"]), F["lam"].astype(float),
        F["ownership"].astype(float), F["curvature"].astype(float), 1.0 / d,
        float(F["c_stance"]), float(F["c_count"]), rows, cols, Yc.data, yplus,
        32768, tau2)
    np.savez(cache, beta_se=sand["beta_se"], ownership_se=sand["ownership_se"])
    print(f"[{corpus}/{level}] sandwich SEs -> {cache.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="*", default=["us", "br"])
    ap.add_argument("--levels", nargs="*", default=list(LEVELS), choices=LEVELS)
    ap.add_argument("--variants", nargs="*", default=[v for v, _ in VARIANTS])
    ap.add_argument("--drop-classes", nargs="*", default=None,
                    help="macro columns to drop before fitting "
                         f"(default {DEFAULT_DROP['macro']} for the macro level)")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--rebuild-data", action="store_true",
                    help="rebuild the (Y,S) cache from the per-sentence files first")
    a = ap.parse_args()
    for c in [x.lower() for x in a.corpora]:
        for level in a.levels:
            drop = tuple(a.drop_classes) if a.drop_classes is not None \
                else DEFAULT_DROP[level]
            D = (irt_data.load_matrices(c, force=a.rebuild_data) if level == "micro"
                 else macro_data.load_matrices(level, c, force=a.rebuild_data))
            for variant, ug in VARIANTS:
                if variant not in a.variants:
                    continue
                fit_one(c, variant, ug, D, level=level, drop=drop, refit=a.refit)
            if "godambe" in a.variants:
                standard_errors(level, c, D, drop=drop, refit=a.refit)


if __name__ == "__main__":
    main()
