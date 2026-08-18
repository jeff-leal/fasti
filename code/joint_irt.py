from __future__ import annotations


from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.special import logsumexp

_TINY = np.finfo(np.float64).tiny
_EPS = np.finfo(np.float64).eps


# ---------------------------------------------------------------------
# Sparse input preparation
# ---------------------------------------------------------------------
def _canonical_csr(Y):
    if not issparse(Y):
        raise TypeError("Y must be a scipy sparse matrix (CSR is preferred).")
    Y = Y.tocsr().astype(np.float64, copy=True)
    Y.sum_duplicates()
    Y.eliminate_zeros()
    Y.sort_indices()
    if Y.nnz == 0:
        raise ValueError("Y has no positive stored cells.")
    if np.any(~np.isfinite(Y.data)) or np.any(Y.data <= 0.0):
        raise ValueError("Stored Y entries must be finite and strictly positive.")
    return Y


def _active_rows(Y):
    return np.repeat(np.arange(Y.shape[0], dtype=np.int64), np.diff(Y.indptr))


def _aligned_stance_from_sum(Y, stance_sum):
    Z = stance_sum.tocsr().astype(np.float64, copy=True)
    if Z.shape != Y.shape:
        raise ValueError("stance_sum and Y must have identical shapes.")
    Z.sum_duplicates()
    Z.eliminate_zeros()
    Z.sort_indices()

    out = np.zeros(Y.nnz, dtype=np.float64)
    for i in range(Y.shape[0]):
        ya, yb = Y.indptr[i], Y.indptr[i + 1]
        za, zb = Z.indptr[i], Z.indptr[i + 1]
        if ya == yb or za == zb:
            continue
        ycols = Y.indices[ya:yb]
        zcols = Z.indices[za:zb]
        take = np.searchsorted(zcols, ycols)
        valid = take < zcols.size
        if np.any(valid):
            ii = np.flatnonzero(valid)
            valid[ii] &= zcols[take[ii]] == ycols[ii]
        if np.any(valid):
            out[ya + np.flatnonzero(valid)] = Z.data[za:zb][take[valid]]
    return out / Y.data


def _prepare_stance(Y, stance):
    if issparse(stance):
        s = _aligned_stance_from_sum(Y, stance)
    else:
        s = np.asarray(stance, dtype=np.float64).reshape(-1)
        if s.size != Y.nnz:
            raise ValueError(
                "A dense stance vector must be aligned with canonical Y.data "
                "and contain exactly Y.nnz entries."
            )
    if np.any(~np.isfinite(s)) or np.any(s < -1.0 - 1e-12) or np.any(s > 1.0 + 1e-12):
        raise ValueError("Stance means must be finite and lie in [-1, 1].")
    return np.clip(s, -1.0, 1.0)


def _prepare_sample_weight(sample_weight, N):
    if sample_weight is None:
        return np.ones(N, dtype=np.float64)
    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if w.size != N or np.any(~np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("sample_weight must be finite, strictly positive, and length N.")
    return w


# ---------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------
def _weighted_standardize(theta, w):
    W = float(w.sum())
    mean = float(np.dot(w, theta) / W)
    centered = theta - mean
    scale = float(np.sqrt(np.dot(w, centered * centered) / W))
    if not np.isfinite(scale) or scale <= 64.0 * _EPS:
        raise FloatingPointError("The latent scale has effectively zero weighted variance.")
    return centered / scale, mean, scale


def _reidentify_stance(theta, alpha, beta, w):
    theta_new, mean, scale = _weighted_standardize(theta, w)
    return theta_new, alpha + beta * mean, beta * scale


def _reidentify_joint(theta, alpha, beta, lam, ownership, curvature, w):
    """Center theta and set RMS(beta)=1 while preserving fitted predictors.

    Stage-2's N(0,1) theta MAP term is nonconstant only when the latent scale
    is not simultaneously fixed by Var_w(theta)=1.  This deterministic item
    scale constraint supplies the required identification without changing any
    fitted stance or count predictor.
    """
    W = float(w.sum())
    mean = float(np.dot(w, theta) / W)
    beta_rms = float(np.sqrt(np.mean(beta * beta)))
    if not np.isfinite(beta_rms) or beta_rms <= 64.0 * _EPS:
        raise FloatingPointError("Stage-2 stance discriminations have effectively zero RMS.")

    # old theta = mean + scale * theta_new; beta_new = beta * scale.
    scale = 1.0 / beta_rms
    theta_new = (theta - mean) / scale

    alpha_new = alpha + beta * mean
    beta_new = beta * scale

    ownership_new = scale * (ownership + 2.0 * curvature * mean)
    curvature_new = (scale * scale) * curvature
    lam_new = lam + ownership * mean + curvature * (mean * mean + scale * scale - 1.0)

    # Softmax gauges remove document-common components without changing pi_it.
    lam_new -= lam_new.mean()
    ownership_new -= ownership_new.mean()
    curvature_new -= curvature_new.mean()
    return theta_new, alpha_new, beta_new, lam_new, ownership_new, curvature_new


def _loss_tail_ratio(improvement, largest_previous_improvement):
    return float(improvement / max(1.0, largest_previous_improvement))


def _sd_quadratic(theta):
    """SD of theta^2 residualized on [1, theta].

    This is the scale factor that turns a raw curvature into the scale-invariant
    standardized curvature kappa~_t = curvature_t * sd_q: under theta -> s*theta
    the raw curvature scales as 1/s^2 while sd_q scales as s^2, so their product
    is invariant.  It is the deterministic bridge that lets a single frozen
    curvature-prior variance stay coherent across the reidentification.
    """
    x = np.asarray(theta, dtype=np.float64)
    x2 = x * x
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, x2, rcond=None)
    sd_q = float(np.std(x2 - A @ coef))
    return max(sd_q, 64.0 * _EPS)


def _effective_curvature_tau2(theta, tau2_curv_std):
    """Raw-curvature prior variance implied by the invariant standardized prior.

    The prior is kappa~_t = curvature_t * sd_q ~ N(0, tau2_curv_std); expressed on
    the raw curvature it is curvature_t ~ N(0, tau2_curv_std / sd_q^2).  sd_q is
    recomputed from the current theta so the penalty tracks the latent scale.
    """
    if not np.isfinite(tau2_curv_std):
        return np.inf
    sd_q = _sd_quadratic(theta)
    return tau2_curv_std / (sd_q * sd_q)


# ---------------------------------------------------------------------
# Stage 1: sparse stance ALS
# ---------------------------------------------------------------------
def _stance_sse(theta, alpha, beta, rows, cols, s, cell_w):
    resid = s - alpha[cols] - beta[cols] * theta[rows]
    return float(np.dot(cell_w, resid * resid))


def _stage1_initialize_theta(rows, y, s, w, N, orientation):
    cell_w = w[rows] * y
    den = np.bincount(rows, weights=cell_w, minlength=N)
    raw = np.divide(
        np.bincount(rows, weights=cell_w * s, minlength=N),
        den,
        out=np.zeros(N, dtype=np.float64),
        where=den > 0.0,
    )
    theta, _, _ = _weighted_standardize(raw, w)
    return theta * float(orientation)


def _stance_update_items(theta, rows, cols, s, cell_w, T):
    th = theta[rows]
    sw = np.bincount(cols, weights=cell_w, minlength=T)
    swt = np.bincount(cols, weights=cell_w * th, minlength=T)
    swtt = np.bincount(cols, weights=cell_w * th * th, minlength=T)
    sws = np.bincount(cols, weights=cell_w * s, minlength=T)
    swts = np.bincount(cols, weights=cell_w * th * s, minlength=T)

    mean_t = np.divide(swt, sw, out=np.zeros(T), where=sw > 0.0)
    mean_s = np.divide(sws, sw, out=np.zeros(T), where=sw > 0.0)
    sxx = swtt - swt * mean_t
    sxs = swts - swt * mean_s
    threshold = 64.0 * _EPS * np.maximum.reduce(
        (np.abs(swtt), np.abs(swt), sw, np.ones(T))
    )
    nonsingular = sxx > threshold
    beta = np.divide(sxs, sxx, out=np.zeros(T), where=nonsingular)
    alpha = mean_s - beta * mean_t
    beta[~nonsingular] = 0.0
    alpha[~np.isfinite(alpha)] = 0.0
    beta[~np.isfinite(beta)] = 0.0
    return alpha, beta


def _stance_update_theta(alpha, beta, rows, cols, s, cell_w, N):
    b = beta[cols]
    num = np.bincount(rows, weights=cell_w * b * (s - alpha[cols]), minlength=N)
    den = np.bincount(rows, weights=cell_w * b * b, minlength=N)
    threshold = 64.0 * _EPS * np.maximum(den.max(initial=0.0), 1.0)
    return np.divide(num, den, out=np.zeros(N), where=den > threshold)


def _estimate_sigma2(theta, alpha, beta, rows, cols, s, y, w):
    resid = s - alpha[cols] - beta[cols] * theta[rows]
    numerator = float(np.dot(w[rows] * y, resid * resid))
    denom = float(w[rows].sum())
    return max(numerator / max(denom, _TINY), _TINY)


def _fit_stage1(rows, cols, y, s, w, N, T, orientation, max_sweeps,
                tol_loss_tail, patience, verbose):
    cell_w = w[rows] * y
    theta = _stage1_initialize_theta(rows, y, s, w, N, orientation)
    alpha0 = np.divide(
        np.bincount(cols, weights=cell_w * s, minlength=T),
        np.bincount(cols, weights=cell_w, minlength=T),
        out=np.zeros(T),
        where=np.bincount(cols, weights=cell_w, minlength=T) > 0.0,
    )
    beta0 = np.zeros(T)
    sse_prev = _stance_sse(theta, alpha0, beta0, rows, cols, s, cell_w)
    alpha, beta = alpha0, beta0
    history, largest_improvement = [], 0.0
    stable = 0

    for sweep in range(1, max_sweeps + 1):
        alpha, beta = _stance_update_items(theta, rows, cols, s, cell_w, T)
        theta_raw = _stance_update_theta(alpha, beta, rows, cols, s, cell_w, N)
        theta, alpha, beta = _reidentify_stance(theta_raw, alpha, beta, w)
        sse_now = _stance_sse(theta, alpha, beta, rows, cols, s, cell_w)
        improvement = max(0.0, sse_prev - sse_now)
        tail = _loss_tail_ratio(improvement, largest_improvement)
        largest_improvement = max(largest_improvement, improvement)
        sigma2 = _estimate_sigma2(theta, alpha, beta, rows, cols, s, y, w)
        history.append(dict(
            sweep=float(sweep), sse=float(sse_now), sigma2=float(sigma2),
            improvement=float(improvement), tail_fraction=float(tail),
        ))
        if verbose:
            print(f"stage 1 | sweep={sweep:3d} sse={sse_now:.8g} tail={tail:.2e}")
        stable = stable + 1 if (sweep > 1 and tail <= tol_loss_tail) else 0
        if stable >= patience:
            return theta, alpha, beta, sigma2, history, True
        sse_prev = sse_now
    return theta, alpha, beta, sigma2, history, False


# ---------------------------------------------------------------------
# Count block: profiled multinomial quasi-likelihood
# ---------------------------------------------------------------------
def _solve_batched_3x3(H, g):
    Hs = H.copy()
    scale = np.maximum(np.trace(Hs, axis1=1, axis2=2) / 3.0, 1.0)
    Hs[:, np.arange(3), np.arange(3)] += (1e-12 * scale)[:, None]
    try:
        return np.linalg.solve(Hs, g[..., None])[..., 0]
    except np.linalg.LinAlgError:
        out = np.zeros_like(g)
        for t in range(Hs.shape[0]):
            out[t] = np.linalg.lstsq(Hs[t], g[t], rcond=None)[0]
        return out


def _count_observed_statistics(rows, cols, y, w, inv_d, theta, h, T):
    wy = w[rows] * inv_d[rows] * y
    return (
        np.bincount(cols, weights=wy, minlength=T),
        np.bincount(cols, weights=wy * theta[rows], minlength=T),
        np.bincount(cols, weights=wy * h[rows], minlength=T),
    )


def _count_nll(lam, ownership, curvature, theta, h, inv_d, w, yplus,
               obs0, obs1, obs2, chunk_size, tau2_curv=np.inf):
    total = 0.0
    N = theta.size
    for lo in range(0, N, chunk_size):
        hi = min(lo + chunk_size, N)
        r = (
            lam[None, :]
            + theta[lo:hi, None] * ownership[None, :]
            + h[lo:hi, None] * curvature[None, :]
        )
        total += float(np.dot(w[lo:hi] * inv_d[lo:hi] * yplus[lo:hi],
                              logsumexp(r, axis=1)))
    observed = float(np.dot(lam, obs0) + np.dot(ownership, obs1) + np.dot(curvature, obs2))
    penalty = (
        0.5 * float(np.dot(curvature, curvature)) / tau2_curv
        if np.isfinite(tau2_curv) else 0.0
    )
    return total - observed + penalty


def _count_gradient_majorizer(lam, ownership, curvature, theta, h, inv_d, w,
                              yplus, obs0, obs1, obs2, chunk_size,
                              tau2_curv=np.inf):
    T = lam.size
    e0 = np.zeros(T)
    e1 = np.zeros(T)
    e2 = np.zeros(T)
    e11 = np.zeros(T)
    e12 = np.zeros(T)
    e22 = np.zeros(T)

    N = theta.size
    for lo in range(0, N, chunk_size):
        hi = min(lo + chunk_size, N)
        th = theta[lo:hi]
        hh = h[lo:hi]
        r = lam[None, :] + th[:, None] * ownership[None, :] + hh[:, None] * curvature[None, :]
        p = np.exp(r - logsumexp(r, axis=1, keepdims=True))
        mu = (w[lo:hi] * inv_d[lo:hi] * yplus[lo:hi])[:, None] * p
        e0 += mu.sum(axis=0)
        e1 += (mu * th[:, None]).sum(axis=0)
        e2 += (mu * hh[:, None]).sum(axis=0)
        e11 += (mu * (th[:, None] * th[:, None])).sum(axis=0)
        e12 += (mu * (th[:, None] * hh[:, None])).sum(axis=0)
        e22 += (mu * (hh[:, None] * hh[:, None])).sum(axis=0)

    g = np.column_stack((e0 - obs0, e1 - obs1, e2 - obs2))
    H = np.empty((T, 3, 3))
    H[:, 0, 0] = e0
    H[:, 0, 1] = H[:, 1, 0] = e1
    H[:, 0, 2] = H[:, 2, 0] = e2
    H[:, 1, 1] = e11
    H[:, 1, 2] = H[:, 2, 1] = e12
    H[:, 2, 2] = e22
    if np.isfinite(tau2_curv):
        # Gaussian curvature prior curvature_t ~ N(0, tau2_curv): exact gradient
        # and Hessian contributions to the (curvature) coordinate only.
        g[:, 2] += curvature / tau2_curv
        H[:, 2, 2] += 1.0 / tau2_curv
    return g, H


def _fit_count_items(theta, yplus, rows, cols, y, w, inv_d, T, *,
                     lam0=None, ownership0=None, curvature0=None,
                     max_sweeps=25, max_backtracks=20, tol_loss_tail=1e-4,
                     patience=3, chunk_size=32768, verbose=False, label="count",
                     tau2_curv=np.inf):
    h = theta * theta - 1.0
    obs0, obs1, obs2 = _count_observed_statistics(rows, cols, y, w, inv_d, theta, h, T)

    if lam0 is None:
        lam = np.log(np.maximum(obs0, _TINY))
        lam -= lam.mean()
    else:
        lam = np.asarray(lam0, float).copy()
        lam -= lam.mean()
    ownership = np.zeros(T) if ownership0 is None else np.asarray(ownership0, float).copy()
    curvature = np.zeros(T) if curvature0 is None else np.asarray(curvature0, float).copy()
    ownership -= ownership.mean()
    curvature -= curvature.mean()

    nll_prev = _count_nll(
        lam, ownership, curvature, theta, h, inv_d, w, yplus, obs0, obs1, obs2,
        chunk_size, tau2_curv,
    )
    history, largest_improvement = [], 0.0
    stable = 0

    for sweep in range(1, max_sweeps + 1):
        g, H = _count_gradient_majorizer(
            lam, ownership, curvature, theta, h, inv_d, w, yplus,
            obs0, obs1, obs2, chunk_size, tau2_curv,
        )
        delta = _solve_batched_3x3(H, g)

        accepted = False
        step_used = 0.0
        nll_now = nll_prev
        for bt in range(max_backtracks):
            step = 0.5 ** bt
            lam_try = lam - step * delta[:, 0]
            ownership_try = ownership - step * delta[:, 1]
            curvature_try = curvature - step * delta[:, 2]
            lam_try -= lam_try.mean()
            ownership_try -= ownership_try.mean()
            curvature_try -= curvature_try.mean()
            nll_try = _count_nll(
                lam_try, ownership_try, curvature_try, theta, h, inv_d, w,
                yplus, obs0, obs1, obs2, chunk_size, tau2_curv,
            )
            if np.isfinite(nll_try) and nll_try <= nll_prev + 1e-12 * (1.0 + abs(nll_prev)):
                lam, ownership, curvature = lam_try, ownership_try, curvature_try
                nll_now, step_used, accepted = nll_try, step, True
                break

        improvement = max(0.0, nll_prev - nll_now)
        tail = _loss_tail_ratio(improvement, largest_improvement)
        largest_improvement = max(largest_improvement, improvement)
        history.append(dict(
            sweep=float(sweep), nll=float(nll_now), improvement=float(improvement),
            tail_fraction=float(tail), newton_step_max=float(np.max(np.abs(delta))),
            accepted_step=float(step_used), step_accepted=float(accepted),
        ))
        if verbose:
            print(f"{label} | sweep={sweep:3d} nll={nll_now:.8g} tail={tail:.2e} step={step_used:.3g}")
        stable = stable + 1 if (sweep > 1 and tail <= tol_loss_tail) else 0
        if stable >= patience:
            return lam, ownership, curvature, history, True
        if not accepted:
            return lam, ownership, curvature, history, bool(tail <= tol_loss_tail)
        nll_prev = nll_now

    return lam, ownership, curvature, history, False


# ---------------------------------------------------------------------
# Curvature empirical Bayes (marginal-likelihood / EM shrinkage)
# ---------------------------------------------------------------------
def _curvature_posterior_var(lam, ownership, curvature, theta, h, inv_d, w,
                             yplus, tau2_curv, chunk_size):
    """Per-topic posterior variance of curvature_t under the N(0, tau2_curv)
    prior.  It is the (curvature, curvature) entry of the inverse penalized
    3x3 count information at the current optimum -- the EM E-step quantity."""
    T = lam.size
    zeros = np.zeros(T)
    _, H = _count_gradient_majorizer(
        lam, ownership, curvature, theta, h, inv_d, w, yplus,
        zeros, zeros, zeros, chunk_size, tau2_curv,
    )
    scale = np.maximum(np.trace(H, axis1=1, axis2=2) / 3.0, 1.0)
    H[:, np.arange(3), np.arange(3)] += (1e-10 * scale)[:, None]
    inv = np.linalg.inv(H)
    return np.maximum(inv[:, 2, 2], 0.0)


def _eb_curvature(theta, yplus, rows, cols, y, w, inv_d, T, lam0, ownership0,
                  curvature0, *, max_backtracks, tol_loss_tail, patience,
                  chunk_size, count_max_sweeps, eb_max_iters, eb_tol, verbose,
                  label):
    """Empirical-Bayes shrinkage of the item curvatures.

    Alternates (i) a penalized count-item refit at the current prior variance
    with (ii) the marginal-likelihood/EM update tau2 = mean_t(curvature_t^2 +
    posterior_var_t).  theta is fixed here (the Stage-1 pilot scale), so the raw
    prior variance is well defined; the returned value is converted to the
    scale-invariant standardized-curvature variance tau2_curv_std for freezing.
    """
    h = theta * theta - 1.0
    lam = np.asarray(lam0, float).copy()
    ownership = np.asarray(ownership0, float).copy()
    curvature = np.asarray(curvature0, float).copy()
    sd_q = _sd_quadratic(theta)

    tau2_raw = max(float(np.mean(curvature * curvature)), 1e-12)
    history = []
    converged = False
    for it in range(1, eb_max_iters + 1):
        lam, ownership, curvature, _, _ = _fit_count_items(
            theta, yplus, rows, cols, y, w, inv_d, T,
            lam0=lam, ownership0=ownership, curvature0=curvature,
            max_sweeps=count_max_sweeps, max_backtracks=max_backtracks,
            tol_loss_tail=tol_loss_tail, patience=patience, chunk_size=chunk_size,
            verbose=False, label=label, tau2_curv=tau2_raw,
        )
        post_var = _curvature_posterior_var(
            lam, ownership, curvature, theta, h, inv_d, w, yplus, tau2_raw, chunk_size
        )
        tau2_new = max(float(np.mean(curvature * curvature + post_var)), 1e-14)
        rel = abs(tau2_new - tau2_raw) / max(tau2_raw, 1e-14)
        history.append(dict(
            iter=float(it), tau2_raw=float(tau2_new),
            tau2_curv_std=float(tau2_new * sd_q * sd_q), rel=float(rel),
            curv_absmax=float(np.max(np.abs(curvature))),
        ))
        if verbose:
            print(f"{label} EB | iter={it:2d} tau2_raw={tau2_new:.4g} "
                  f"tau2_std={tau2_new * sd_q * sd_q:.4g} rel={rel:.2e} "
                  f"|curv|max={np.max(np.abs(curvature)):.4g}")
        prev = tau2_raw
        tau2_raw = tau2_new
        if it >= 2 and rel <= eb_tol:
            converged = True
            break

    tau2_curv_std = tau2_raw * sd_q * sd_q
    return lam, ownership, curvature, float(tau2_curv_std), history, converged


# ---------------------------------------------------------------------
# Dirichlet-multinomial effective information
# ---------------------------------------------------------------------
def _pearson_composition_stat(theta, lam, ownership, curvature, Y, rows, cols, y,
                              yplus, chunk_size):
    N = theta.size
    h = theta * theta - 1.0
    R = np.empty(N)

    for lo in range(0, N, chunk_size):
        hi = min(lo + chunk_size, N)
        th = theta[lo:hi]
        hh = h[lo:hi]
        r = lam[None, :] + th[:, None] * ownership[None, :] + hh[:, None] * curvature[None, :]
        p = np.exp(r - logsumexp(r, axis=1, keepdims=True))
        a, b = Y.indptr[lo], Y.indptr[hi]
        lr = rows[a:b] - lo
        cc = cols[a:b]
        yy = y[a:b]
        denom = yplus[lo:hi][lr] * np.maximum(p[lr, cc], _TINY)
        ss = np.bincount(lr, weights=(yy * yy) / denom, minlength=hi - lo)
        R[lo:hi] = ss - yplus[lo:hi]
    return R


def _estimate_dm_kappa(theta, lam, ownership, curvature, Y, rows, cols, y, yplus,
                       w, chunk_size):
    R = _pearson_composition_stat(
        theta, lam, ownership, curvature, Y, rows, cols, y, yplus, chunk_size
    )
    W = float(w.sum())
    rbar = float(np.dot(w, R) / W)
    mbar = float(np.dot(w, yplus) / W)
    B = float(Y.shape[1] - 1)

    if B <= 0.0 or mbar <= 1.0 + 64.0 * _EPS:
        kappa = np.inf
        status = "not_identified_all_documents_length_one"
    elif rbar <= B * (1.0 + 64.0 * _EPS):
        kappa = np.inf
        status = "multinomial_boundary"
    elif rbar >= B * mbar * (1.0 - 64.0 * _EPS):
        kappa = 0.0
        status = "max_overdispersion_boundary"
    else:
        kappa = (B * mbar - rbar) / (rbar - B)
        kappa = float(max(kappa, 0.0))
        status = "interior"

    if np.isinf(kappa):
        d = np.ones_like(yplus, dtype=np.float64)
    else:
        d = (yplus + kappa) / (1.0 + kappa)
        d = np.maximum(d, 1.0)
    return kappa, d, R, status


# ---------------------------------------------------------------------
# Godambe calibration and document update
# ---------------------------------------------------------------------
def _doc_score_curvature(theta, Y, s, w, alpha, beta, sigma2, lam, ownership,
                         curvature, inv_d, rows, cols, y, yplus, chunk_size,
                         theta_map=False):
    """Return unweighted per-document score/working-curvature contributions."""
    N = theta.size
    uS = np.empty(N)
    hS = np.empty(N)
    uY = np.empty(N)
    hY = np.empty(N)

    for lo in range(0, N, chunk_size):
        hi = min(lo + chunk_size, N)
        th = theta[lo:hi]
        hh = th * th - 1.0
        r = lam[None, :] + th[:, None] * ownership[None, :] + hh[:, None] * curvature[None, :]
        rp = ownership[None, :] + 2.0 * th[:, None] * curvature[None, :]
        lse = logsumexp(r, axis=1)
        p = np.exp(r - lse[:, None])
        erp = (p * rp).sum(axis=1)
        varrp = np.maximum((p * rp * rp).sum(axis=1) - erp * erp, 0.0)

        a, b = Y.indptr[lo], Y.indptr[hi]
        lr = rows[a:b] - lo
        cc = cols[a:b]
        yy = y[a:b]
        q = alpha[cc] + beta[cc] * th[lr]
        resid = s[a:b] - q
        stance_u = np.bincount(lr, weights=yy * beta[cc] * resid, minlength=hi - lo) / sigma2
        stance_h = np.bincount(lr, weights=yy * beta[cc] * beta[cc], minlength=hi - lo) / sigma2

        obs_rp = np.bincount(lr, weights=yy * rp[lr, cc], minlength=hi - lo)
        count_u = inv_d[lo:hi] * (obs_rp - yplus[lo:hi] * erp)
        count_h = inv_d[lo:hi] * yplus[lo:hi] * varrp

        if theta_map:
            # Score and working curvature of log N(theta_i; 0, 1).
            stance_u -= th
            stance_h += 1.0

        uS[lo:hi], hS[lo:hi] = stance_u, stance_h
        uY[lo:hi], hY[lo:hi] = count_u, count_h
    return uS, hS, uY, hY


def _godambe_factor(u, h, w):
    H = float(np.dot(w, h))
    if not np.isfinite(H) or H <= 64.0 * _EPS:
        return 1.0, 0.0, H
    ubar = float(np.dot(w, u) / w.sum())
    J = float(np.dot(w, (u - ubar) ** 2))
    if not np.isfinite(J) or J <= 64.0 * _EPS * max(H, 1.0):
        return 1.0, J, H
    return float(J / H), J, H


def _calibrated_doc_nll(theta, Y, s, alpha, beta, sigma2, lam, ownership,
                        curvature, inv_d, cS, cY, rows, cols, y, yplus,
                        chunk_size):
    N = theta.size
    out = np.empty(N)
    for lo in range(0, N, chunk_size):
        hi = min(lo + chunk_size, N)
        th = theta[lo:hi]
        hh = th * th - 1.0
        r = lam[None, :] + th[:, None] * ownership[None, :] + hh[:, None] * curvature[None, :]
        lse = logsumexp(r, axis=1)
        a, b = Y.indptr[lo], Y.indptr[hi]
        lr = rows[a:b] - lo
        cc = cols[a:b]
        yy = y[a:b]
        q = alpha[cc] + beta[cc] * th[lr]
        stance_sse = np.bincount(
            lr, weights=yy * (s[a:b] - q) ** 2, minlength=hi - lo
        )
        obs_r = np.bincount(lr, weights=yy * r[lr, cc], minlength=hi - lo)
        out[lo:hi] = (
            0.5 * stance_sse / (sigma2 * cS)
            + inv_d[lo:hi] * (yplus[lo:hi] * lse - obs_r) / cY
            + 0.5 * th * th
        )
    return out


def _joint_objective_nll(theta, Y, s, w, alpha, beta, sigma2, lam, ownership,
                         curvature, inv_d, cS, cY, rows, cols, y, yplus,
                         chunk_size, tau2_curv_std=np.inf):
    doc_nll = _calibrated_doc_nll(
        theta, Y, s, alpha, beta, sigma2, lam, ownership, curvature, inv_d,
        cS, cY, rows, cols, y, yplus, chunk_size
    )
    total = float(np.dot(w, doc_nll))
    # Scale-invariant curvature prior, added on the same (1/cY) footing as the
    # count block it regularizes.  The effective raw variance tracks theta.
    tau2_eff = _effective_curvature_tau2(theta, tau2_curv_std)
    if np.isfinite(tau2_eff):
        total += 0.5 * float(np.dot(curvature, curvature)) / tau2_eff / cY
    return total


def _fallback_rescore_document(i, grid_lo, grid_hi, grid_size, max_backtracks,
                               tol_step, Y, s, alpha, beta, sigma2, lam,
                               ownership, curvature, inv_d, cS, cY):
    a, b = Y.indptr[i], Y.indptr[i + 1]
    cc = Y.indices[a:b]
    yy = Y.data[a:b]
    ss = s[a:b]
    M = float(yy.sum())
    inv = float(inv_d[i])

    def nll_grad_hess(theta):
        h = theta * theta - 1.0
        r = lam + ownership * theta + curvature * h
        rp = ownership + 2.0 * curvature * theta
        lse = float(logsumexp(r))
        p = np.exp(r - lse)
        erp = float(np.dot(p, rp))
        varrp = max(float(np.dot(p, rp * rp) - erp * erp), 0.0)

        q = alpha[cc] + beta[cc] * theta
        resid = ss - q
        stance_nll = 0.5 * float(np.dot(yy, resid * resid)) / (sigma2 * cS)
        stance_u = float(np.dot(yy, beta[cc] * resid)) / (sigma2 * cS)
        stance_h = float(np.dot(yy, beta[cc] * beta[cc])) / (sigma2 * cS)

        count_nll = inv * (M * lse - float(np.dot(yy, r[cc]))) / cY
        count_u = inv * (float(np.dot(yy, rp[cc])) - M * erp) / cY
        count_h = inv * M * varrp / cY
        return (
            stance_nll + count_nll + 0.5 * theta * theta,
            stance_u + count_u - theta,
            stance_h + count_h + 1.0,
        )

    grid = np.linspace(grid_lo, grid_hi, grid_size)
    vals = np.array([nll_grad_hess(z)[0] for z in grid])
    theta = float(grid[int(np.argmin(vals))])
    for _ in range(30):
        nll, score, fisher = nll_grad_hess(theta)
        if not np.isfinite(fisher) or fisher <= 1e-14:
            break
        direction = score / fisher
        if abs(direction) <= tol_step * (1.0 + abs(theta)):
            break
        accepted = False
        for bt in range(max_backtracks):
            cand = theta + (0.5 ** bt) * direction
            cand_nll = nll_grad_hess(cand)[0]
            if np.isfinite(cand_nll) and cand_nll <= nll + 1e-12 * (1.0 + abs(nll)):
                theta, accepted = cand, True
                break
        if not accepted:
            break
    return theta


def _subset_nll(th, K, lr, cc, yy, ss, alpha, beta, sigma2, lam, ownership,
                curvature, inv_idx, yp_idx, cS, cY):
    """Calibrated per-document NLL for K active documents (dense over topics,
    sparse over that subset's active cells).  Same arithmetic as
    _calibrated_doc_nll restricted to the given documents."""
    hh = th * th - 1.0
    r = lam[None, :] + th[:, None] * ownership[None, :] + hh[:, None] * curvature[None, :]
    lse = logsumexp(r, axis=1)
    q = alpha[cc] + beta[cc] * th[lr]
    resid = ss - q
    stance_sse = np.bincount(lr, weights=yy * resid * resid, minlength=K)
    obs_r = np.bincount(lr, weights=yy * r[lr, cc], minlength=K)
    return 0.5 * stance_sse / (sigma2 * cS) + inv_idx * (yp_idx * lse - obs_r) / cY + 0.5 * th * th


def _subset_score_fisher(th, K, lr, cc, yy, ss, alpha, beta, sigma2, lam,
                         ownership, curvature, inv_idx, yp_idx, cS, cY):
    """Per-document score and working Fisher (incl. N(0,1) MAP) for K active
    documents.  Same arithmetic as _doc_score_curvature(theta_map=True) with
    unit weights, restricted to the given documents."""
    hh = th * th - 1.0
    r = lam[None, :] + th[:, None] * ownership[None, :] + hh[:, None] * curvature[None, :]
    lse = logsumexp(r, axis=1)
    p = np.exp(r - lse[:, None])
    rp = ownership[None, :] + 2.0 * th[:, None] * curvature[None, :]
    erp = np.sum(p * rp, axis=1)
    varrp = np.maximum(np.sum(p * rp * rp, axis=1) - erp * erp, 0.0)
    q = alpha[cc] + beta[cc] * th[lr]
    resid = ss - q
    stance_u = np.bincount(lr, weights=yy * beta[cc] * resid, minlength=K) / sigma2 - th
    stance_h = np.bincount(lr, weights=yy * beta[cc] * beta[cc], minlength=K) / sigma2 + 1.0
    obs_rp = np.bincount(lr, weights=yy * rp[lr, cc], minlength=K)
    count_u = inv_idx * (obs_rp - yp_idx * erp)
    count_h = inv_idx * yp_idx * varrp
    return stance_u / cS + count_u / cY, stance_h / cS + count_h / cY


def _update_theta_joint(theta, Y, s, alpha, beta, sigma2, lam, ownership,
                        curvature, inv_d, cS, cY, rows, cols, y, yplus,
                        chunk_size, max_iter, max_backtracks, tol_step,
                        grid_size, verbose=False):
    # Per-document Newton with safeguarded per-document backtracking.  Each
    # document's theta_i is an independent scalar problem, so every dense/sparse
    # evaluation is restricted to the ACTIVE (not-yet-converged, not-failed)
    # documents.  This is bit-for-bit identical to evaluating all N documents
    # every step -- the per-document accept/step decisions depend only on that
    # document's own quantities -- but the line search no longer recomputes
    # documents that already converged.
    N = theta.size
    theta = theta.copy()
    converged = np.zeros(N, dtype=bool)
    failed = np.zeros(N, dtype=bool)

    for iteration in range(1, max_iter + 1):
        idx = np.flatnonzero(~(converged | failed))
        K = idx.size
        if K == 0:
            break

        # Gather this iteration's active cells once (reused across backtracks).
        loc = np.full(N, -1, dtype=np.int64)
        loc[idx] = np.arange(K)
        cmask = loc[rows] >= 0
        lr = loc[rows[cmask]]
        cc = cols[cmask]
        yy = y[cmask]
        ss = s[cmask]
        inv_idx = inv_d[idx]
        yp_idx = yplus[idx]
        th = theta[idx].copy()

        score, fisher = _subset_score_fisher(
            th, K, lr, cc, yy, ss, alpha, beta, sigma2, lam, ownership,
            curvature, inv_idx, yp_idx, cS, cY,
        )
        direction = np.divide(score, fisher, out=np.zeros_like(score),
                              where=np.isfinite(fisher) & (fisher > 1e-14))
        small = np.abs(direction) <= tol_step * (1.0 + np.abs(th))
        converged[idx[small]] = True
        todo = ~small
        if not np.any(todo):
            break

        current_nll = _subset_nll(
            th, K, lr, cc, yy, ss, alpha, beta, sigma2, lam, ownership,
            curvature, inv_idx, yp_idx, cS, cY,
        )
        accepted = np.zeros(K, dtype=bool)
        remaining = todo.copy()
        step = np.ones(K)

        for _ in range(max_backtracks):
            if not np.any(remaining):
                break
            th_try = th.copy()
            th_try[remaining] = th[remaining] + step[remaining] * direction[remaining]
            cand_nll = _subset_nll(
                th_try, K, lr, cc, yy, ss, alpha, beta, sigma2, lam, ownership,
                curvature, inv_idx, yp_idx, cS, cY,
            )
            ok = (
                remaining
                & np.isfinite(cand_nll)
                & (cand_nll <= current_nll + 1e-12 * (1.0 + np.abs(current_nll)))
            )
            if np.any(ok):
                th[ok] = th_try[ok]
                accepted[ok] = True
                converged[idx[ok]] |= np.abs(step[ok] * direction[ok]) <= tol_step * (
                    1.0 + np.abs(th[ok])
                )
            remaining &= ~ok
            step[remaining] *= 0.5

        theta[idx] = th
        failed[idx[remaining]] = True
        if verbose:
            print(
                f"joint theta | iter={iteration:2d} accepted={int(accepted.sum()):,} "
                f"converged={int(converged.sum()):,} fallback_pending={int(failed.sum()):,}"
            )
        if not np.any(accepted) and np.any(todo):
            break

    fallback = (~converged) | failed
    if np.any(fallback):
        grid_lo, grid_hi = float(theta.min()), float(theta.max())
        if grid_hi - grid_lo <= 64.0 * _EPS:
            grid_lo, grid_hi = -3.0, 3.0
        common = (Y, s, alpha, beta, sigma2, lam, ownership, curvature, inv_d, cS, cY)
        for i in np.flatnonzero(fallback):
            theta[i] = _fallback_rescore_document(
                int(i), grid_lo, grid_hi, grid_size, max_backtracks, tol_step, *common
            )
    return theta, int(fallback.sum())


def _profile_doc_intercepts(theta, yplus, lam, ownership, curvature, chunk_size):
    N = theta.size
    h = theta * theta - 1.0
    out = np.empty(N)
    for lo in range(0, N, chunk_size):
        hi = min(lo + chunk_size, N)
        r = lam[None, :] + theta[lo:hi, None] * ownership[None, :] + h[lo:hi, None] * curvature[None, :]
        out[lo:hi] = np.log(yplus[lo:hi]) - logsumexp(r, axis=1)
    return out



# ---------------------------------------------------------------------
# Stage-2 cluster-robust sandwich
# ---------------------------------------------------------------------
def _joint_sandwich(theta, Y, s, w, alpha, beta, sigma2, lam, ownership,
                    curvature, inv_d, cS, cY, rows, cols, y, yplus,
                    chunk_size, tau2_curv_std=np.inf):
    """Profiled document-cluster Godambe covariance for the Stage-2 item block.

    The calculation treats the frozen Stage-1 calibration (sigma2, kappa_dm,
    d_i, c_count) as the fixed working calibration that defines Stage 2.  It
    profiles the N document scores from the sandwich rather than constructing a
    dense (N + 5T)-square covariance matrix.
    """
    N, T = Y.shape
    P = 5 * T
    score = np.zeros((N, P), dtype=np.float64)
    h_theta = np.zeros(N, dtype=np.float64)
    h_theta_item = np.zeros((N, P), dtype=np.float64)
    theta_score = np.zeros(N, dtype=np.float64)
    H = np.zeros((P, P), dtype=np.float64)

    # Stance scores and working curvature.
    th_active = theta[rows]
    resid = s - alpha[cols] - beta[cols] * th_active
    sb = w[rows] * y / (sigma2 * cS)
    stance_alpha_score = sb * resid
    stance_beta_score = stance_alpha_score * th_active
    np.add.at(score, (rows, cols), stance_alpha_score)
    np.add.at(score, (rows, T + cols), stance_beta_score)
    np.add.at(theta_score, rows, sb * beta[cols] * resid)
    np.add.at(h_theta, rows, sb * beta[cols] * beta[cols])
    np.add.at(h_theta_item, (rows, cols), sb * beta[cols])
    np.add.at(h_theta_item, (rows, T + cols), sb * beta[cols] * th_active)

    h_aa = np.bincount(cols, weights=sb, minlength=T)
    h_ab = np.bincount(cols, weights=sb * th_active, minlength=T)
    h_bb = np.bincount(cols, weights=sb * th_active * th_active, minlength=T)
    ix = np.arange(T)
    H[ix, ix] += h_aa
    H[ix, T + ix] += h_ab
    H[T + ix, ix] += h_ab
    H[T + ix, T + ix] += h_bb

    # Count scores and multinomial working curvature, chunked over documents.
    H_count = np.zeros((3 * T, 3 * T), dtype=np.float64)
    for lo in range(0, N, chunk_size):
        hi = min(lo + chunk_size, N)
        B = hi - lo
        th = theta[lo:hi]
        hh = th * th - 1.0
        r = lam[None, :] + th[:, None] * ownership[None, :] + hh[:, None] * curvature[None, :]
        p = np.exp(r - logsumexp(r, axis=1, keepdims=True))
        rp = ownership[None, :] + 2.0 * th[:, None] * curvature[None, :]
        erp = np.sum(p * rp, axis=1)
        varrp = np.maximum(np.sum(p * rp * rp, axis=1) - erp * erp, 0.0)

        a, b = Y.indptr[lo], Y.indptr[hi]
        lr = rows[a:b] - lo
        cc = cols[a:b]
        yy = y[a:b]
        eff = w[lo:hi] * inv_d[lo:hi] / cY
        mass = eff * yplus[lo:hi]

        e = -mass[:, None] * p
        if yy.size:
            np.add.at(e, (lr, cc), eff[lr] * yy)
        score[lo:hi, 2 * T:3 * T] += e
        score[lo:hi, 3 * T:4 * T] += e * th[:, None]
        score[lo:hi, 4 * T:5 * T] += e * hh[:, None]

        obs_rp = np.bincount(
            lr, weights=eff[lr] * yy * rp[lr, cc], minlength=B
        )
        theta_score[lo:hi] += obs_rp - mass * erp
        h_theta[lo:hi] += mass * varrp

        base = mass[:, None] * p * (rp - erp[:, None])
        h_theta_item[lo:hi, 2 * T:3 * T] += base
        h_theta_item[lo:hi, 3 * T:4 * T] += base * th[:, None]
        h_theta_item[lo:hi, 4 * T:5 * T] += base * hh[:, None]

        x = (np.ones(B, dtype=np.float64), th, hh)
        for q in range(3):
            for r_ in range(3):
                wt = mass * x[q] * x[r_]
                diag = np.sum(wt[:, None] * p, axis=0)
                block = slice(q * T, (q + 1) * T)
                block2 = slice(r_ * T, (r_ + 1) * T)
                H_count[block, block2] += np.diag(diag)
                H_count[block, block2] -= p.T @ (wt[:, None] * p)

    H[2 * T:5 * T, 2 * T:5 * T] += H_count

    # N(0, 1) MAP contribution.  It contributes to the bread, not the meat.
    theta_score -= w * theta
    h_theta += w

    # Curvature EB prior: a MAP penalty, so it enters the bread (item block)
    # only, on the (1/cY) count footing.  Its effective raw variance is fixed
    # at the reported theta scale.
    tau2_eff = _effective_curvature_tau2(theta, tau2_curv_std)
    if np.isfinite(tau2_eff):
        ix_c = np.arange(T)
        H[4 * T + ix_c, 4 * T + ix_c] += 1.0 / (tau2_eff * cY)

    good = np.isfinite(h_theta) & (h_theta > 64.0 * _EPS)
    if not np.all(good):
        raise FloatingPointError("Non-positive profiled theta curvature in sandwich calculation.")

    ratio = theta_score / h_theta
    profile_score = score - h_theta_item * ratio[:, None]
    bread = H - h_theta_item.T @ (h_theta_item / h_theta[:, None])
    meat = profile_score.T @ profile_score

    # The raw parameterization contains the three softmax gauges and the
    # Stage-2 beta-norm identification direction.  The Moore-Penrose inverse
    # is precisely the inverse on the estimable tangent space.
    bread_inv = np.linalg.pinv(bread, rcond=1e-10)
    covariance = bread_inv @ meat @ bread_inv.T
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.maximum(np.diag(covariance), 0.0)

    return {
        "alpha_se": np.sqrt(diagonal[:T]),
        "beta_se": np.sqrt(diagonal[T:2 * T]),
        "lam_se": np.sqrt(diagonal[2 * T:3 * T]),
        "ownership_se": np.sqrt(diagonal[3 * T:4 * T]),
        "curvature_se": np.sqrt(diagonal[4 * T:5 * T]),
        "covariance": covariance,
        "bread": bread,
        "meat": meat,
        "conditional_on_calibration": True,
    }


# ---------------------------------------------------------------------
# Full staged joint fit
# ---------------------------------------------------------------------
@dataclass
class JointIRTFit:
    theta_stance: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    sigma2: float
    lam: np.ndarray
    ownership: np.ndarray
    curvature: np.ndarray
    kappa_dm: float
    dm_scale: np.ndarray
    dm_status: str
    c_stance: float
    c_count: float
    godambe_factor: float
    use_godambe: bool
    tau2_curv_std: float
    theta: np.ndarray
    doc_intercept: np.ndarray
    stage1_history: list
    count_init_history: list
    eb_history: list
    joint_history: list
    stage1_converged: bool
    count_init_converged: bool
    eb_converged: bool
    joint_converged: bool
    theta_fallback_documents: int


def fit_joint_irt(
    Y,
    stance,
    *,
    sample_weight=None,
    orientation=1.0,
    chunk_size=32768,
    stage1_max_sweeps=100,
    count_init_max_sweeps=25,
    joint_max_sweeps=30,
    count_item_max_sweeps=8,
    max_backtracks=20,
    tol_loss_tail=1e-4,
    joint_tol_flat=5e-3,
    joint_patience=2,
    patience=3,
    theta_max_iter=20,
    theta_tol_step=1e-6,
    theta_grid_size=33,
    use_godambe=True,
    eb_curvature=True,
    eb_max_iters=25,
    eb_tol=1e-3,
    verbose=True,
    compute_se=False,
):
    """Fit sparse staged JointIRT with fixed Stage-1 calibration."""
    use_godambe = bool(use_godambe)
    Y = _canonical_csr(Y)
    N, T = Y.shape
    if np.any(np.diff(Y.indptr) == 0):
        raise ValueError("Y contains an empty document; remove it before fitting.")
    if np.any(np.asarray(Y.sum(axis=0)).ravel() <= 0.0):
        raise ValueError("Y contains an empty topic; remove it before fitting.")

    s = _prepare_stance(Y, stance)
    rows = _active_rows(Y)
    cols = Y.indices.astype(np.int64, copy=False)
    y = Y.data
    yplus = np.asarray(Y.sum(axis=1)).ravel().astype(np.float64)
    w = _prepare_sample_weight(sample_weight, N)

    if verbose:
        print(
            f"sparse joint stance-topic IRT | N={N:,} T={T:,} "
            f"active cells={Y.nnz:,}"
        )

    # Stage 1: stance pilot.
    theta_stance, alpha, beta, sigma2, stage1_history, stage1_converged = _fit_stage1(
        rows, cols, y, s, w, N, T, orientation, stage1_max_sweeps,
        tol_loss_tail, patience, verbose,
    )

    # Stage-1 count pilot and one fixed DM calibration.
    inv_d = np.ones(N)
    lam, ownership, curvature, count_init_history, count_init_converged = _fit_count_items(
        theta_stance, yplus, rows, cols, y, w, inv_d, T,
        max_sweeps=count_init_max_sweeps,
        max_backtracks=max_backtracks,
        tol_loss_tail=tol_loss_tail,
        patience=patience,
        chunk_size=chunk_size,
        verbose=verbose,
        label="count init",
    )
    kappa_dm, d, _, dm_status = _estimate_dm_kappa(
        theta_stance, lam, ownership, curvature, Y, rows, cols, y, yplus, w, chunk_size
    )
    inv_d = 1.0 / d
    lam, ownership, curvature, count_dm_history, _ = _fit_count_items(
        theta_stance, yplus, rows, cols, y, w, inv_d, T,
        lam0=lam, ownership0=ownership, curvature0=curvature,
        max_sweeps=count_init_max_sweeps,
        max_backtracks=max_backtracks,
        tol_loss_tail=tol_loss_tail,
        patience=patience,
        chunk_size=chunk_size,
        verbose=False,
        label="count dm init",
    )
    count_init_history.extend(count_dm_history)

    # Empirical-Bayes curvature shrinkage: calibrate the standardized-curvature
    # prior variance once (at the Stage-1 pilot scale) and freeze it.  This is
    # the genuine shrinkage of the overfit-prone item curvatures.
    if eb_curvature:
        lam, ownership, curvature, tau2_curv_std, eb_history, eb_converged = _eb_curvature(
            theta_stance, yplus, rows, cols, y, w, inv_d, T,
            lam, ownership, curvature,
            max_backtracks=max_backtracks, tol_loss_tail=tol_loss_tail,
            patience=patience, chunk_size=chunk_size,
            count_max_sweeps=count_init_max_sweeps, eb_max_iters=eb_max_iters,
            eb_tol=eb_tol, verbose=verbose, label="count eb",
        )
    else:
        tau2_curv_std = np.inf
        eb_history = []
        eb_converged = True

    # These calibration quantities define the Stage-2 quasi-objective and are
    # deliberately frozen for every Stage-2 sweep and in the returned fit.
    sigma2 = _estimate_sigma2(theta_stance, alpha, beta, rows, cols, s, y, w)
    _, _, uY_stance, hY_stance = _doc_score_curvature(
        theta_stance, Y, s, w, alpha, beta, sigma2, lam, ownership, curvature,
        inv_d, rows, cols, y, yplus, chunk_size,
    )
    cS = 1.0
    godambe_factor, JY_cal, HY_cal = _godambe_factor(uY_stance, hY_stance, w)
    cY = godambe_factor if use_godambe else 1.0
    JS_cal = HS_cal = np.nan

    # Reparameterize once into the Stage-2 MAP identification.  This exactly
    # preserves all fitted predictors from the pilot.
    theta, alpha, beta, lam, ownership, curvature = _reidentify_joint(
        theta_stance.copy(), alpha, beta, lam, ownership, curvature, w
    )

    joint_history = []
    largest_improvement = 0.0
    cumulative_improvement = 0.0
    stable = 0
    theta_fallback_documents = 0
    joint_converged = False

    for sweep in range(1, joint_max_sweeps + 1):
        nll_before = _joint_objective_nll(
            theta, Y, s, w, alpha, beta, sigma2, lam, ownership, curvature,
            inv_d, cS, cY, rows, cols, y, yplus, chunk_size, tau2_curv_std,
        )
        # Effective raw curvature-prior variance at the current latent scale
        # (theta is fixed during the item-block update within this sweep).
        tau2_eff = _effective_curvature_tau2(theta, tau2_curv_std)

        old_theta = theta.copy()
        old_alpha = alpha.copy()
        old_beta = beta.copy()
        old_lam = lam.copy()
        old_ownership = ownership.copy()
        old_curvature = curvature.copy()

        # Fixed-calibration block updates.
        alpha_new, beta_new = _stance_update_items(theta, rows, cols, s, w[rows] * y, T)
        lam_new, ownership_new, curvature_new, count_history, _ = _fit_count_items(
            theta, yplus, rows, cols, y, w, inv_d, T,
            lam0=lam,
            ownership0=ownership,
            curvature0=curvature,
            max_sweeps=count_item_max_sweeps,
            max_backtracks=max_backtracks,
            tol_loss_tail=tol_loss_tail,
            patience=patience,
            chunk_size=chunk_size,
            verbose=False,
            label="joint count",
            tau2_curv=tau2_eff,
        )
        theta_raw, theta_fallback_documents = _update_theta_joint(
            theta, Y, s, alpha_new, beta_new, sigma2, lam_new, ownership_new,
            curvature_new, inv_d, cS, cY, rows, cols, y, yplus, chunk_size,
            theta_max_iter, max_backtracks, theta_tol_step, theta_grid_size,
            verbose=verbose,
        )
        cand = _reidentify_joint(
            theta_raw, alpha_new, beta_new, lam_new, ownership_new, curvature_new, w
        )

        # The internal block solves are descent steps conditional on their
        # complements.  A single global backtrack preserves monotone descent
        # after the exact Stage-2 MAP reidentification.
        theta, alpha, beta, lam, ownership, curvature = cand
        nll_after = _joint_objective_nll(
            theta, Y, s, w, alpha, beta, sigma2, lam, ownership, curvature,
            inv_d, cS, cY, rows, cols, y, yplus, chunk_size, tau2_curv_std,
        )
        global_step = 1.0
        accepted_global = np.isfinite(nll_after) and (
            nll_after <= nll_before + 1e-12 * (1.0 + abs(nll_before))
        )

        if not accepted_global:
            accepted_global = False
            for bt in range(1, max_backtracks + 1):
                step = 0.5 ** bt
                try:
                    trial = _reidentify_joint(
                        old_theta + step * (theta - old_theta),
                        old_alpha + step * (alpha - old_alpha),
                        old_beta + step * (beta - old_beta),
                        old_lam + step * (lam - old_lam),
                        old_ownership + step * (ownership - old_ownership),
                        old_curvature + step * (curvature - old_curvature),
                        w,
                    )
                except FloatingPointError:
                    continue
                trial_nll = _joint_objective_nll(
                    trial[0], Y, s, w, trial[1], trial[2], sigma2, trial[3], trial[4],
                    trial[5], inv_d, cS, cY, rows, cols, y, yplus, chunk_size, tau2_curv_std,
                )
                if np.isfinite(trial_nll) and trial_nll <= nll_before + 1e-12 * (1.0 + abs(nll_before)):
                    theta, alpha, beta, lam, ownership, curvature = trial
                    nll_after = trial_nll
                    global_step = step
                    accepted_global = True
                    break

        if not accepted_global:
            theta, alpha, beta = old_theta, old_alpha, old_beta
            lam, ownership, curvature = old_lam, old_ownership, old_curvature
            nll_after = nll_before
            global_step = 0.0

        improvement = max(0.0, nll_before - nll_after)
        tail = _loss_tail_ratio(improvement, largest_improvement)
        rel_improvement = improvement / max(1.0, abs(nll_before))
        largest_improvement = max(largest_improvement, improvement)
        # Flattening measure: this sweep's drop as a fraction of the TOTAL loss
        # reduction achieved so far.  Almost all of the calibrated NLL is an
        # irreducible baseline (the multinomial log-normalizer), so the descent
        # only moves the loss by a few hundred out of ~1e6; the meaningful scale
        # is the achieved descent, not |nll|.
        cumulative_improvement += improvement
        flat_fraction = improvement / max(1.0, cumulative_improvement)

        row = dict(
            sweep=float(sweep),
            frozen_calibrated_nll_before=float(nll_before),
            frozen_calibrated_nll_after=float(nll_after),
            improvement=float(improvement),
            relative_improvement=float(rel_improvement),
            tail_fraction=float(tail),
            flat_fraction=float(flat_fraction),
            sigma2=float(sigma2),
            kappa_dm=float(kappa_dm),
            dm_status=dm_status,
            c_stance=float(cS),
            c_count=float(cY),
            godambe_factor=float(godambe_factor),
            use_godambe=bool(use_godambe),
            tau2_curv_std=float(tau2_curv_std),
            tau2_curv_effective=float(tau2_eff),
            stance_J_calibration=float(JS_cal),
            stance_H_calibration=float(HS_cal),
            count_J_calibration=float(JY_cal),
            count_H_calibration=float(HY_cal),
            count_item_sweeps=float(len(count_history)),
            theta_fallback_documents=float(theta_fallback_documents),
            global_step=float(global_step),
        )
        joint_history.append(row)

        if verbose:
            kd = "inf" if np.isinf(kappa_dm) else f"{kappa_dm:.4g}"
            print(
                f"joint | sweep={sweep:3d} nll={nll_after:.8g} "
                f"flat={flat_fraction:.2e} rel={rel_improvement:.2e} "
                f"kappa={kd} cS={cS:.3g} cY={cY:.3g} godambe={use_godambe}"
            )

        # Converge as soon as the loss FLATTENS: a sweep that adds less than
        # joint_tol_flat (default 0.5%) of the total achieved descent, for
        # joint_patience consecutive sweeps.  Yit/Sit are noisy measurements;
        # chasing loss changes below this is precision the data does not carry.
        stable = stable + 1 if (sweep > 1 and flat_fraction <= joint_tol_flat) else 0
        if stable >= joint_patience:
            joint_converged = True
            break
        if global_step == 0.0:
            # No feasible descent direction remains for the fixed objective.
            joint_converged = True
            break

    doc_intercept = _profile_doc_intercepts(theta, yplus, lam, ownership, curvature, chunk_size)

    fit = JointIRTFit(
        theta_stance=theta_stance,
        alpha=alpha,
        beta=beta,
        sigma2=float(sigma2),
        lam=lam,
        ownership=ownership,
        curvature=curvature,
        kappa_dm=float(kappa_dm),
        dm_scale=d,
        dm_status=dm_status,
        c_stance=float(cS),
        c_count=float(cY),
        godambe_factor=float(godambe_factor),
        use_godambe=bool(use_godambe),
        tau2_curv_std=float(tau2_curv_std),
        theta=theta,
        doc_intercept=doc_intercept,
        stage1_history=stage1_history,
        count_init_history=count_init_history,
        eb_history=eb_history,
        joint_history=joint_history,
        stage1_converged=stage1_converged,
        count_init_converged=count_init_converged,
        eb_converged=eb_converged,
        joint_converged=joint_converged,
        theta_fallback_documents=int(theta_fallback_documents),
    )
    fit.sandwich_ = (
        _joint_sandwich(
            theta, Y, s, w, alpha, beta, sigma2, lam, ownership, curvature,
            inv_d, cS, cY, rows, cols, y, yplus, chunk_size, tau2_curv_std,
        )
        if compute_se else None
    )
    return fit


class JointIRT:
    """Sparse staged joint stance-topic IRT."""

    name = "Joint IRT"

    def __init__(self, *, use_godambe=True, **fit_kwargs):
        fit_kwargs.setdefault("verbose", False)
        if "use_godambe" in fit_kwargs:
            raise TypeError(
                "Pass use_godambe directly to JointIRT(...), not through fit_kwargs."
            )
        self.use_godambe = bool(use_godambe)
        self.fit_kwargs = dict(fit_kwargs)

    def fit(self, Y, S, sample_weight=None):
        Yc = Y if issparse(Y) else csr_matrix(np.asarray(Y, dtype=np.float64))
        Sc = S if issparse(S) else csr_matrix(np.asarray(S, dtype=np.float64))
        fit = fit_joint_irt(
            Yc,
            Sc,
            sample_weight=sample_weight,
            use_godambe=self.use_godambe,
            **self.fit_kwargs,
        )
        self.fit_ = fit
        self.theta_ = np.asarray(fit.theta, dtype=np.float64).ravel()
        return self
