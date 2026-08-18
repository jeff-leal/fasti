from __future__ import annotations


import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.special import logsumexp

REPO = Path(__file__).resolve().parents[1]          # .../topic2irt

# Constants ported verbatim from irt_local_poisson.py
RANDOM_STATE = 14605 - 2025 - 4
EPS = 1e-12

# Constants ported verbatim from two_stage_stance_anchored_sparse.py
_TINY = np.finfo(np.float64).tiny
_EPS = np.finfo(np.float64).eps


# =====================================================================
# Base
# =====================================================================
class IdealPointEstimator:
    """Common base: subclasses set ``self.theta_`` in ``fit`` and return self."""

    name: str = "base"

    def fit(self, *args, **kwargs):          # pragma: no cover - interface only
        raise NotImplementedError


# =====================================================================
# Embeddings + PCA  (written fresh; memory-safe streaming)
# =====================================================================
class EmbeddingPCA(IdealPointEstimator):
    """Doc ideal point from sentence embeddings.

    Pipeline:
        1. L2-normalize each per-chunk sentence embedding to unit norm.
        2. Mean-pool the unit embeddings within each document -> one vector/doc.
        3. PCA on the doc-level mean-embedding matrix (center columns; SVD).
        4. theta = first principal-component score per doc.

    The chunk universe ``emb`` may be a huge ``np.memmap`` (e.g. 3.19M x 384
    float32). The row-gather + normalize + per-doc sum are streamed in batches of
    ``batch`` rows so the full stack of chunk vectors is never materialised; only
    the small (n_docs x d) doc-mean matrix and the PCA on it are held in memory.
    """

    name = "Embeddings + PCA"

    def __init__(self, batch: int = 200_000):
        self.batch = int(batch)

    def fit(self, emb, chunk_rows, doc_index, n_docs):
        # emb: (N_total x d) ndarray or np.memmap float32 (full chunk universe)
        # chunk_rows: int64 row index into emb for each modeled chunk
        # doc_index:  int64 doc index (0..n_docs-1) for each modeled chunk
        chunk_rows = np.asarray(chunk_rows, dtype=np.int64)
        doc_index = np.asarray(doc_index, dtype=np.int64)
        if chunk_rows.shape[0] != doc_index.shape[0]:
            raise ValueError("chunk_rows and doc_index must have equal length.")
        n_docs = int(n_docs)
        d = int(emb.shape[1])
        n = chunk_rows.shape[0]

        doc_sum = np.zeros((n_docs, d), dtype=np.float64)
        doc_cnt = np.zeros(n_docs, dtype=np.float64)

        for lo in range(0, n, self.batch):
            hi = min(lo + self.batch, n)
            rows = chunk_rows[lo:hi]
            docs = doc_index[lo:hi]
            # gather this batch of chunk vectors (materialises only the batch)
            X = np.asarray(emb[rows], dtype=np.float64)
            norms = np.sqrt((X * X).sum(axis=1))
            norms = np.where(norms > 0.0, norms, 1.0)          # guard zero vectors
            X /= norms[:, None]                                # L2-normalize rows
            np.add.at(doc_sum, docs, X)                        # per-doc running sum
            np.add.at(doc_cnt, docs, 1.0)

        cnt = np.where(doc_cnt > 0.0, doc_cnt, 1.0)
        doc_mean = doc_sum / cnt[:, None]                       # (n_docs, d)

        # PCA: center columns, SVD, first-PC score
        mu = doc_mean.mean(axis=0, keepdims=True)
        Xc = doc_mean - mu
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        theta = U[:, 0] * S[0]                                  # first-PC score per doc

        self.doc_count_ = doc_cnt
        self.theta_ = np.asarray(theta, dtype=np.float64).ravel()
        return self


# =====================================================================
# Stance IRT  (ported verbatim from us_meanstance_model.fit_meanstance)
# =====================================================================
def _fit_meanstance(x, w, iters=400, tol=1e-8, ridge=1.0):
    """z = alpha_t + beta_t theta_i, weighted ALS; theta standardized each sweep.

    Verbatim port of ``us_meanstance_model.fit_meanstance``. Returns
    (theta, beta, psi/alpha, n_iters). NO external orientation flip is applied
    here (the original did that outside the function).
    """
    J = x.shape[1]
    th = (w * x).sum(1) / np.maximum(w.sum(1), 1e-9)
    th = (th - th.mean()) / (th.std() + 1e-9)
    beta = np.zeros(J)
    psi = np.zeros(J)
    prev = None
    for it in range(iters):
        Sw = w.sum(0)
        Swt = (w * th[:, None]).sum(0)
        Swtt = (w * th[:, None] ** 2).sum(0)
        Swx = (w * x).sum(0)
        Swtx = (w * th[:, None] * x).sum(0)
        det = Sw * Swtt - Swt ** 2 + 1e-9
        psi = (Swtt * Swx - Swt * Swtx) / det
        beta = (Sw * Swtx - Swt * Swx) / det
        num = (w * beta[None, :] * (x - psi[None, :])).sum(1)
        den = (w * beta[None, :] ** 2).sum(1) + ridge
        th = num / den
        th = (th - th.mean()) / (th.std() + 1e-9)
        if prev is not None and np.max(np.abs(th - prev)) < tol:
            break
        prev = th.copy()
    return th, beta, psi, it + 1


class StanceIRT(IdealPointEstimator):
    """Weighted rank-1 mean-stance factor model.

        z_it = alpha_t + beta_t * theta_i + eps       (weighted ALS)

    where z_it is doc i's mean signed stance on topic t and the cell weight is the
    sentence count Y_it. theta_i is standardized to unit variance each sweep.
    Ported from ``us_meanstance_model.fit_meanstance``.
    """

    name = "Stance IRT"

    def __init__(self, iters: int = 400, tol: float = 1e-8, ridge: float = 1.0):
        self.iters = int(iters)
        self.tol = float(tol)
        self.ridge = float(ridge)

    def fit(self, Y, S):
        # Y: doc x topic sentence-COUNT matrix (CSR or dense)
        # S: doc x topic signed-stance-SUM matrix (same shape/support as Y)
        N = Y.toarray() if issparse(Y) else np.asarray(Y, dtype=np.float64)
        Ssum = S.toarray() if issparse(S) else np.asarray(S, dtype=np.float64)
        N = np.asarray(N, dtype=np.float64)
        Ssum = np.asarray(Ssum, dtype=np.float64)
        if N.shape != Ssum.shape:
            raise ValueError("Y and S must have identical shape.")
        # mean stance per doc-topic on active cells (0 elsewhere); weights = counts
        Z = np.divide(Ssum, N, out=np.zeros_like(N), where=N > 0)
        theta, beta, alpha, nit = _fit_meanstance(
            Z, N, iters=self.iters, tol=self.tol, ridge=self.ridge
        )
        self.beta_ = beta
        self.alpha_ = alpha
        self.n_iter_ = nit
        self.theta_ = np.asarray(theta, dtype=np.float64).ravel()
        return self


# =====================================================================
# Topic IRT  (PoissonScaleEM ported verbatim from irt_local_poisson.py)
# =====================================================================
def _varimax(B, iters=100, tol=1e-8):
    """Plain varimax rotation of a loading matrix B (returns rotated B and R)."""
    p, k = B.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(iters):
        L = B @ R
        u, s, vt = np.linalg.svd(
            B.T @ (L ** 3 - L @ np.diag((L ** 2).mean(0))))
        R = u @ vt
        d2 = s.sum()
        if d != 0 and d2 / d < 1 + tol:
            break
        d = d2
    return B @ R, R


class PoissonScaleEM:
    """Poisson topic-scaling IRT in D dimensions, numpy CPU, relative convergence.

        y_ij ~ Poisson(lambda_ij)
        log lambda_ij = alpha_i + psi_j + sum_d beta_jd * theta_id

    Block coordinate ascent with an exact Newton step per block, deterministic SVD
    init, location + PCA-rotation identification with the N(0,1) theta prior
    setting the scale, and a scale-free relative-logpost stop criterion. Ported
    verbatim from ``irt_local_poisson.PoissonScaleEM``.
    """

    def __init__(self, dims=1, rtol=1e-6, max_iter=500, ridge_theta=1.0,
                 ridge_beta=0.1, beta_prior="gaussian", eps_rel=1e-6,
                 mix_init="data", eps_mode="fixed", eps_c=1e-3,
                 dirichlet_a=2.0, pi_floor=0.02, vm_a=2.0, vm_b=1.0,
                 hess_jitter=1e-8, verbose=True):
        self.dims = int(dims)
        self.eps_mode = str(eps_mode)
        self.eps_c = float(eps_c)
        self.dirichlet_a = float(dirichlet_a)
        self.pi_floor = float(pi_floor)
        self.vm_a = float(vm_a)
        self.vm_b = float(vm_b)
        self.mix_init = str(mix_init)
        self.beta_prior = str(beta_prior)
        self.eps_rel = float(eps_rel)
        self.ridge_theta = float(ridge_theta)
        self.ridge_beta = float(ridge_beta)
        self.rtol = float(rtol)
        self.max_iter = int(max_iter)
        self.hess_jitter = float(hess_jitter)
        self.verbose = verbose

    # ----- internals ------------------------------------------------- #
    def _loglik_var(self, Y, alpha, psi, Theta, Beta):
        eta = alpha[:, None] + psi[None, :] + Theta @ Beta.T
        return float(np.sum(Y * eta - np.exp(np.clip(eta, -30.0, 30.0))))

    @staticmethod
    def _newton_block(Y, off_row, off_col, coef_design, params, prior_prec=None):
        eta = off_row[:, None] + off_col[None, :] + params @ coef_design.T
        lam = np.exp(np.clip(eta, -30.0, 30.0))
        resid = Y - lam
        grad = resid @ coef_design
        XtLX = np.einsum("op,ro,oq->rpq", coef_design, lam, coef_design,
                         optimize=True)
        p = coef_design.shape[1]
        XtLX[:, np.arange(p), np.arange(p)] += 1e-8
        if prior_prec is not None:
            pr = np.asarray(prior_prec, float)
            if pr.ndim == 1:
                pr = pr[None, :]
            grad = grad - pr * params
            XtLX[:, np.arange(p), np.arange(p)] += pr
        delta = np.linalg.solve(XtLX, grad[..., None])[..., 0]
        return params + delta

    def _identify(self, alpha, psi, Theta, Beta):
        N = Theta.shape[0]
        mu = Theta.mean(0)
        psi = psi + Beta @ mu
        Theta = Theta - mu
        cov = (Theta.T @ Theta) / N
        w, V = np.linalg.eigh(cov)
        V = V[:, np.argsort(w)[::-1]]
        Theta = Theta @ V
        Beta = Beta @ V
        ra, rb = self.ridge_theta, self.ridge_beta
        if ra > 0 and rb > 0:
            num = rb * np.sum(Beta ** 2, axis=0)
            den = ra * np.sum(Theta ** 2, axis=0) + EPS
            c = (num / den) ** 0.25
            Theta = Theta * c
            Beta = Beta / c
        for d in range(Beta.shape[1]):
            j = np.argmax(np.abs(Beta[:, d]))
            if Beta[j, d] < 0:
                Beta[:, d] *= -1.0
                Theta[:, d] *= -1.0
        return alpha, psi, Theta, Beta

    def _logpost(self, Y, alpha, psi, Theta, Beta):
        return (self._loglik_var(Y, alpha, psi, Theta, Beta)
                - 0.5 * self.ridge_theta * float(np.sum(Theta ** 2))
                - 0.5 * self.ridge_beta * float(np.sum(Beta ** 2)))

    # ----- spike-and-slab mixture prior on the loadings (2-D) -------- #
    def _mix_variances(self, mp):
        e = mp["eps2"]
        return np.array([[e, e],
                         [mp["t1"], e],
                         [e, mp["t2"]],
                         [mp["t12a"], mp["t12b"]]])

    def _mix_estep(self, Beta, mp):
        V = self._mix_variances(mp)
        pi = mp["pi"]
        b2 = Beta ** 2
        logN = (-0.5 * np.log(2 * np.pi * V)[None, :, :]
                - b2[:, None, :] / (2 * V)[None, :, :]).sum(2)
        logr = np.log(pi + EPS)[None, :] + logN
        logr -= logr.max(1, keepdims=True)
        r = np.exp(logr)
        r /= r.sum(1, keepdims=True)
        Omega = r @ (1.0 / V)
        return r, Omega

    def _mix_mstep(self, Beta, r, mp):
        J = Beta.shape[0]
        b2 = Beta ** 2
        nk = r.sum(0) + EPS
        a = self.dirichlet_a
        pi = (r.sum(0) + (a - 1.0)) / (J + 4.0 * (a - 1.0))
        if self.pi_floor > 0:
            pi = np.maximum(pi, self.pi_floor)
            pi = pi / pi.sum()
        if self.eps_mode == "fixed":
            eps2 = self.eps_c * float(np.median(b2.mean(0))) + 1e-12
        else:
            spike_ss = (r[:, 0] * (b2[:, 0] + b2[:, 1])).sum() + \
                       (r[:, 1] * b2[:, 1]).sum() + (r[:, 2] * b2[:, 0]).sum()
            spike_n = 2 * r[:, 0].sum() + r[:, 1].sum() + r[:, 2].sum() + EPS
            eps2 = max(float(spike_ss / spike_n), 1e-8)
        floor = 4.0 * eps2
        t1 = max(float((r[:, 1] * b2[:, 0]).sum() / nk[1]), floor)
        t2 = max(float((r[:, 2] * b2[:, 1]).sum() / nk[2]), floor)
        t12a = max(float((r[:, 3] * b2[:, 0]).sum() / nk[3]), floor)
        t12b = max(float((r[:, 3] * b2[:, 1]).sum() / nk[3]), floor)
        return {"pi": pi, "t1": t1, "t2": t2, "t12a": t12a, "t12b": t12b, "eps2": eps2}

    def _mix_logprior(self, Beta, mp):
        V = self._mix_variances(mp)
        pi = mp["pi"]
        b2 = Beta ** 2
        logN = (-0.5 * np.log(2 * np.pi * V)[None, :, :]
                - b2[:, None, :] / (2 * V)[None, :, :]).sum(2)
        logmix = logN + np.log(pi + EPS)[None, :]
        m = logmix.max(1, keepdims=True)
        return float((m[:, 0] + np.log(np.exp(logmix - m).sum(1))).sum())

    def _identify_scale(self, alpha, psi, Theta, Beta):
        N = Theta.shape[0]
        mu = Theta.mean(0)
        psi = psi + Beta @ mu
        Theta = Theta - mu
        sd = Theta.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Theta = Theta / sd
        Beta = Beta * sd
        for d in range(Beta.shape[1]):
            j = np.argmax(np.abs(Beta[:, d]))
            if Beta[j, d] < 0:
                Beta[:, d] *= -1.0
                Theta[:, d] *= -1.0
        return alpha, psi, Theta, Beta

    # ----- Alternative 1: global MDL/BIC support selection ----------- #
    _MDL_SUPPORTS = ([], [0], [1], [0, 1])
    _MDL_CODES = (0, 1, 2, 12)
    _MDL_DF = (0, 1, 1, 2)

    def _fit_regime_block(self, Y, alpha, Theta, S, psi0, Beta0,
                          n_inner=50, tol=1e-7):
        N, J = Y.shape
        k = len(S)
        Xfea = np.hstack([np.ones((N, 1)), Theta[:, S]]) if k else np.ones((N, 1))
        params = np.hstack([psi0[:, None], Beta0[:, S]]) if k else psi0[:, None].copy()
        prior = np.concatenate([[0.0], np.full(k, 1e-6)])
        prev = None
        for _ in range(n_inner):
            params = self._newton_block(Y.T, np.zeros(J), alpha, Xfea, params,
                                        prior_prec=prior)
            if prev is not None and np.max(np.abs(params - prev)) < tol:
                break
            prev = params.copy()
        psi = params[:, 0]
        Beta_full = np.zeros((J, self.dims))
        for ii, d in enumerate(S):
            Beta_full[:, d] = params[:, 1 + ii]
        eta = alpha[:, None] + psi[None, :] + Theta @ Beta_full.T
        LL_j = (Y * eta - np.exp(np.clip(eta, -30.0, 30.0))).sum(0)
        return psi, Beta_full, LL_j

    def _select_supports(self, Y, alpha, Theta, psi, Beta, n_eff):
        pen = 0.5 * np.log(n_eff)
        psis, Betas, LLs = [], [], []
        for S in self._MDL_SUPPORTS:
            ps, Bf, LLj = self._fit_regime_block(Y, alpha, Theta, S, psi, Beta)
            psis.append(ps)
            Betas.append(Bf)
            LLs.append(LLj)
        LLs = np.array(LLs)
        df = np.array(self._MDL_DF)[:, None]
        Q = -LLs + pen * df
        best = Q.argmin(0)
        J = Y.shape[1]
        psi_new = np.empty(J)
        Beta_new = np.zeros((J, self.dims))
        LL_sel = np.empty(J)
        for z in range(4):
            m = best == z
            if m.any():
                psi_new[m] = psis[z][m]
                Beta_new[m] = Betas[z][m]
                LL_sel[m] = LLs[z][m]
        codes = np.array(self._MDL_CODES)[best]
        df_beta = int(np.array(self._MDL_DF)[best].sum())
        return psi_new, Beta_new, codes, float(LL_sel.sum()), df_beta

    # ----- Alternative 2: von Mises axis-angle shrinkage ------------- #
    def _vm_logprior(self, phi, kappa, a, b):
        J = phi.shape[0]
        S = float(np.cos(4.0 * phi).sum())
        return (kappa * S - J * float(np.log(np.i0(kappa)))
                + (a - 1.0) * np.log(kappa) - b * kappa)

    def _vm_update_kappa(self, phi, a, b):
        J = phi.shape[0]
        S = float(np.cos(4.0 * phi).sum())
        ks = np.logspace(-2, np.log10(80.0), 400)
        g = ks * S - J * np.log(np.i0(ks)) + (a - 1.0) * np.log(ks) - b * ks
        return float(ks[int(np.argmax(g))])

    def _vm_objj(self, Y, alpha, psi, Theta, r, phi, kappa):
        c, s = np.cos(phi)[None, :], np.sin(phi)[None, :]
        U = c * Theta[:, 0][:, None] + s * Theta[:, 1][:, None]
        eta = alpha[:, None] + psi[None, :] + r[None, :] * U
        LLj = (Y * eta - np.exp(np.clip(eta, -30.0, 30.0))).sum(0)
        return LLj + kappa * np.cos(4.0 * phi)

    def _vm_feature_step(self, Y, alpha, psi, Theta, r, phi, kappa, a, b,
                         n_inner=5, max_dphi=0.5, n_back=25):
        N, J = Y.shape
        t1, t2 = Theta[:, 0][:, None], Theta[:, 1][:, None]
        for _ in range(n_inner):
            c, s = np.cos(phi)[None, :], np.sin(phi)[None, :]
            U = c * t1 + s * t2
            V = -s * t1 + c * t2
            eta = alpha[:, None] + psi[None, :] + r[None, :] * U
            lam = np.exp(np.clip(eta, -30.0, 30.0))
            resid = Y - lam
            rV = r[None, :] * V
            g_psi = resid.sum(0)
            g_r = (resid * U).sum(0)
            g_phi = (resid * rV).sum(0) - 4.0 * kappa * np.sin(4.0 * phi)
            G = np.stack([g_psi, g_r, g_phi], axis=1)
            Hpp = -lam.sum(0)
            Hpr = -(lam * U).sum(0)
            Hpphi = -(lam * rV).sum(0)
            Hrr = -(lam * U * U).sum(0) - 1e-6
            Hrphi = -(lam * U * rV).sum(0) + (resid * V).sum(0)
            Hphiphi = (-(lam * rV * rV).sum(0)
                       - (resid * r[None, :] * U).sum(0)
                       - 16.0 * kappa * np.cos(4.0 * phi))
            H = np.empty((J, 3, 3))
            H[:, 0, 0] = Hpp
            H[:, 0, 1] = H[:, 1, 0] = Hpr
            H[:, 0, 2] = H[:, 2, 0] = Hpphi
            H[:, 1, 1] = Hrr
            H[:, 1, 2] = H[:, 2, 1] = Hrphi
            H[:, 2, 2] = Hphiphi
            A = -H
            A[:, np.arange(3), np.arange(3)] += 1e-6
            delta = np.linalg.solve(A, G[..., None])[..., 0]
            delta[:, 2] = np.clip(delta[:, 2], -max_dphi, max_dphi)
            base = self._vm_objj(Y, alpha, psi, Theta, r, phi, kappa)
            step = np.ones(J)
            psi_n, r_n, phi_n = psi.copy(), r.copy(), phi.copy()
            for _bk in range(n_back):
                psi_t = psi + step * delta[:, 0]
                r_t = r + step * delta[:, 1]
                phi_t = np.mod(phi + step * delta[:, 2], 2.0 * np.pi)
                obj_t = self._vm_objj(Y, alpha, psi_t, Theta, r_t, phi_t, kappa)
                ok = obj_t > base + 1e-9
                psi_n = np.where(ok, psi_t, psi_n)
                r_n = np.where(ok, r_t, r_n)
                phi_n = np.where(ok, phi_t, phi_n)
                base = np.where(ok, obj_t, base)
                step = np.where(ok, step, step * 0.5)
                if ok.all():
                    break
            psi, r, phi = psi_n, r_n, phi_n
        return psi, r, phi

    def _identify_polar(self, alpha, psi, Theta, r, phi):
        N, D = Theta.shape
        Beta = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)
        mu = Theta.mean(0)
        psi = psi + Beta @ mu
        Theta = Theta - mu
        c = np.sqrt(float((Theta ** 2).sum()) / (N * D)) + EPS
        Theta = Theta / c
        r = r * c
        return alpha, psi, Theta, r, phi

    def _standard_errors(self, Y, alpha, psi, Theta, Beta, beta_prec=None):
        N, J = Y.shape
        D = self.dims
        p = 1 + D
        lam = np.exp(alpha[:, None] + psi[None, :] + Theta @ Beta.T)
        Xdoc = np.hstack([np.ones((J, 1)), Beta])
        Hdoc = np.einsum("jp,ij,jq->ipq", Xdoc, lam, Xdoc, optimize=True)
        pr = np.concatenate([[0.0], np.full(D, self.ridge_theta)])
        Hdoc[:, np.arange(p), np.arange(p)] += pr + 1e-8
        cov_doc = np.linalg.inv(Hdoc)
        se_theta = np.sqrt(np.clip(np.diagonal(cov_doc, axis1=1, axis2=2)[:, 1:], 0, None))
        Xfea = np.hstack([np.ones((N, 1)), Theta])
        Hfea = np.einsum("ip,ij,iq->jpq", Xfea, lam, Xfea, optimize=True)
        if beta_prec is None:
            Hfea[:, np.arange(p), np.arange(p)] += \
                np.concatenate([[0.0], np.full(D, self.ridge_beta)]) + 1e-8
        else:
            bp = np.hstack([np.zeros((J, 1)), np.asarray(beta_prec, float)])
            Hfea[:, np.arange(p), np.arange(p)] += bp + 1e-8
        cov_fea = np.linalg.inv(Hfea)
        se_beta = np.sqrt(np.clip(np.diagonal(cov_fea, axis1=1, axis2=2)[:, 1:], 0, None))
        self.cov_beta_ = cov_fea[:, 1:, 1:]
        return se_theta, se_beta

    def _beta_lik_curvature(self, Y, alpha, psi, Theta, Beta):
        N, J = Y.shape
        lam = np.exp(alpha[:, None] + psi[None, :] + Theta @ Beta.T)
        return (lam.T @ (Theta ** 2))

    # ----- public ---------------------------------------------------- #
    def fit(self, Y):
        Y = np.asarray(Y, dtype=np.float64)
        N, J = Y.shape
        D = self.dims
        rng = np.random.default_rng(RANDOM_STATE)

        M = np.log(Y + 0.5)
        rm, cm, gm = M.mean(1, keepdims=True), M.mean(0, keepdims=True), M.mean()
        R = M - rm - cm + gm
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        alpha = np.log(Y.sum(1) + 1.0)
        psi = np.log(Y.sum(0) + 1.0) - np.log(Y.sum() + 1.0)
        Theta = U[:, :D] * S[:D]
        Beta = Vt[:D, :].T
        if not np.all(np.isfinite(Theta)):
            Theta = rng.standard_normal((N, D)) * 0.1
            Beta = rng.standard_normal((J, D)) * 0.1
        mixture = (self.beta_prior == "mixture")
        free2d = (self.beta_prior == "free2d")
        mdl = (self.beta_prior == "mdl")
        vonmises = (self.beta_prior == "vonmises")
        if (mixture or free2d or mdl or vonmises) and D != 2:
            raise ValueError("mixture/free2d/mdl/vonmises are defined for dims=2 only.")
        self.mix_trace_ = []
        if mixture or free2d or mdl or vonmises:
            alpha, psi, Theta, Beta = self._identify_scale(alpha, psi, Theta, Beta)
            if self.mix_init == "varimax":
                Bv, Rrot = _varimax(Beta)
                Beta = Bv
                Theta = Theta @ Rrot
                alpha, psi, Theta, Beta = self._identify_scale(alpha, psi, Theta, Beta)
            elif self.mix_init == "random":
                rr0 = np.random.default_rng(RANDOM_STATE + 7)
                ang = rr0.uniform(0, 2 * np.pi)
                Rrot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
                Beta = Beta @ Rrot
                Theta = Theta @ Rrot
                alpha, psi, Theta, Beta = self._identify_scale(alpha, psi, Theta, Beta)
            if mdl:
                self.n_eff_ = float(N)
                self._pen = 0.5 * np.log(self.n_eff_)
                psi, Beta, support, _, dfb = self._select_supports(
                    Y, alpha, Theta, psi, Beta, self.n_eff_)
                alpha, psi, Theta, Beta = self._identify_scale(alpha, psi, Theta, Beta)
                self.support_, self.df_beta_ = support, dfb
                ll_old = (self._loglik_var(Y, alpha, psi, Theta, Beta)
                          - 0.5 * self.ridge_theta * float(np.sum(Theta ** 2))
                          - self._pen * dfb)
                ll_hist = [ll_old]
                stop_reason = "max_iter"
                theta_prec = np.concatenate([[0.0], np.full(D, self.ridge_theta)])
                self.mix_trace_ = []
                for it in range(1, self.max_iter + 1):
                    Xdoc = np.hstack([np.ones((J, 1)), Beta])
                    pdoc = np.hstack([alpha[:, None], Theta])
                    pdoc = self._newton_block(Y, np.zeros(N), psi, Xdoc, pdoc,
                                              prior_prec=theta_prec)
                    alpha, Theta = pdoc[:, 0], pdoc[:, 1:]
                    alpha, psi, Theta, Beta = self._identify_scale(alpha, psi, Theta, Beta)
                    psi, Beta, support, _, dfb = self._select_supports(
                        Y, alpha, Theta, psi, Beta, self.n_eff_)
                    alpha, psi, Theta, Beta = self._identify_scale(alpha, psi, Theta, Beta)
                    self.support_, self.df_beta_ = support, dfb
                    ll_new = (self._loglik_var(Y, alpha, psi, Theta, Beta)
                              - 0.5 * self.ridge_theta * float(np.sum(Theta ** 2))
                              - self._pen * dfb)
                    rel = (ll_new - ll_old) / (max(abs(ll_new), abs(ll_old)) + EPS)
                    ll_hist.append(ll_new)
                    if self.verbose and (it <= 3 or it % 10 == 0):
                        print(f"    iter {it:4d}  obj={ll_new:,.1f}  rel={rel:.2e}  df_beta={dfb}")
                    if abs(rel) < self.rtol:
                        stop_reason = f"rtol ({self.rtol:g}) met"
                        ll_old = ll_new
                        break
                    ll_old = ll_new
                self.alpha_, self.psi_, self.theta_, self.beta_ = alpha, psi, Theta, Beta
                self.logpost_ = self.obj_ = ll_old
                self.loglik_ = self._loglik_var(Y, alpha, psi, Theta, Beta)
                free = np.zeros((J, D), bool)
                free[:, 0] = np.isin(support, [1, 12])
                free[:, 1] = np.isin(support, [2, 12])
                bp = np.where(free, 0.0, 1e8)
                self.se_theta_, self.se_beta_ = self._standard_errors(
                    Y, alpha, psi, Theta, Beta, beta_prec=bp)
                self.n_iter_ = it
                self.stop_reason_ = stop_reason
                self.ll_hist_ = np.array(ll_hist)
                if self.verbose:
                    print(f"    -> stopped: {stop_reason} | iters={it} | "
                          f"obj={ll_old:,.1f} | df_beta={dfb}")
                return self
            if vonmises:
                a_vm, b_vm = self.vm_a, self.vm_b
                r = np.hypot(Beta[:, 0], Beta[:, 1])
                phi = np.arctan2(Beta[:, 1], Beta[:, 0])
                kappa = 1.0
                theta_prec = np.concatenate([[0.0], np.full(D, self.ridge_theta)])

                def _vm_obj(al, ps, Th, rr, ph, kp):
                    Bc = np.stack([rr * np.cos(ph), rr * np.sin(ph)], axis=1)
                    return (self._loglik_var(Y, al, ps, Th, Bc)
                            - 0.5 * self.ridge_theta * float(np.sum(Th ** 2))
                            + self._vm_logprior(ph, kp, a_vm, b_vm))

                ll_old = _vm_obj(alpha, psi, Theta, r, phi, kappa)
                ll_hist = [ll_old]
                stop_reason = "max_iter"
                for it in range(1, self.max_iter + 1):
                    Beta = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)
                    Xdoc = np.hstack([np.ones((J, 1)), Beta])
                    pdoc = np.hstack([alpha[:, None], Theta])
                    pdoc = self._newton_block(Y, np.zeros(N), psi, Xdoc, pdoc,
                                              prior_prec=theta_prec)
                    alpha, Theta = pdoc[:, 0], pdoc[:, 1:]
                    alpha, psi, Theta, r, phi = self._identify_polar(alpha, psi, Theta, r, phi)
                    psi, r, phi = self._vm_feature_step(Y, alpha, psi, Theta, r, phi,
                                                        kappa, a_vm, b_vm)
                    alpha, psi, Theta, r, phi = self._identify_polar(alpha, psi, Theta, r, phi)
                    kappa = self._vm_update_kappa(phi, a_vm, b_vm)
                    ll_new = _vm_obj(alpha, psi, Theta, r, phi, kappa)
                    rel = (ll_new - ll_old) / (max(abs(ll_new), abs(ll_old)) + EPS)
                    ll_hist.append(ll_new)
                    if self.verbose and (it <= 3 or it % 10 == 0):
                        print(f"    iter {it:4d}  obj={ll_new:,.1f}  rel={rel:.2e}  kappa={kappa:.2f}")
                    if abs(rel) < self.rtol:
                        stop_reason = f"rtol ({self.rtol:g}) met"
                        ll_old = ll_new
                        break
                    ll_old = ll_new
                Beta = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)
                self.alpha_, self.psi_, self.theta_, self.beta_ = alpha, psi, Theta, Beta
                self.r_vm_, self.phi_vm_, self.kappa_ = r, phi, kappa
                self.logpost_ = self.obj_ = ll_old
                self.loglik_ = self._loglik_var(Y, alpha, psi, Theta, Beta)
                self.se_theta_, self.se_beta_ = self._standard_errors(
                    Y, alpha, psi, Theta, Beta,
                    beta_prec=np.full((J, D), 1e-6))
                self.n_iter_ = it
                self.stop_reason_ = stop_reason
                self.ll_hist_ = np.array(ll_hist)
                if self.verbose:
                    print(f"    -> stopped: {stop_reason} | iters={it} | "
                          f"obj={ll_old:,.1f} | kappa={kappa:.2f}")
                return self
            v0 = float(np.var(Beta))
            mp = {"pi": np.full(4, 0.25), "t1": v0, "t2": v0, "t12a": v0,
                  "t12b": v0, "eps2": self.eps_rel * v0 + 1e-12}
            if self.mix_init == "simple":
                mp["pi"] = np.array([0.2, 0.3, 0.3, 0.2])
                mp["eps2"] = 0.01 * v0
            elif self.mix_init == "smallcross":
                mp["t12a"] = mp["t12b"] = 0.04 * v0
                mp["pi"] = np.array([0.25, 0.35, 0.35, 0.05])
            elif self.mix_init == "random":
                rr = np.random.default_rng(RANDOM_STATE)
                mp["pi"] = rr.dirichlet(np.ones(4))
                mp["t1"], mp["t2"], mp["t12a"], mp["t12b"] = (v0 * rr.uniform(0.2, 2, 4))
            ll_old = (self._loglik_var(Y, alpha, psi, Theta, Beta)
                      - 0.5 * self.ridge_theta * float(np.sum(Theta ** 2))
                      + (self._mix_logprior(Beta, mp) if mixture else 0.0))
        else:
            alpha, psi, Theta, Beta = self._identify(alpha, psi, Theta, Beta)
            ll_old = self._logpost(Y, alpha, psi, Theta, Beta)
        ll_hist = [ll_old]
        stop_reason = "max_iter"
        theta_prec = np.concatenate([[0.0], np.full(D, self.ridge_theta)])
        beta_prec = np.concatenate([[0.0], np.full(D, self.ridge_beta)])
        for it in range(1, self.max_iter + 1):
            Xdoc = np.hstack([np.ones((J, 1)), Beta])
            pdoc = np.hstack([alpha[:, None], Theta])
            pdoc = self._newton_block(Y, np.zeros(N), psi, Xdoc, pdoc,
                                      prior_prec=theta_prec)
            alpha, Theta = pdoc[:, 0], pdoc[:, 1:]
            Xfea = np.hstack([np.ones((N, 1)), Theta])
            pfea = np.hstack([psi[:, None], Beta])
            if mixture:
                r, Omega = self._mix_estep(Beta, mp)
                bprec = np.hstack([np.zeros((J, 1)), Omega])
                pfea = self._newton_block(Y.T, np.zeros(J), alpha, Xfea, pfea,
                                          prior_prec=bprec)
                psi, Beta = pfea[:, 0], pfea[:, 1:]
                alpha, psi, Theta, Beta = self._identify_scale(alpha, psi, Theta, Beta)
                r, Omega = self._mix_estep(Beta, mp)
                mp = self._mix_mstep(Beta, r, mp)
                self.mix_trace_.append({"it": it, "eps2": mp["eps2"], "t1": mp["t1"],
                    "t2": mp["t2"], "t12a": mp["t12a"], "t12b": mp["t12b"],
                    "pi0": mp["pi"][0], "pi1": mp["pi"][1], "pi2": mp["pi"][2],
                    "pi12": mp["pi"][3]})
                ll_new = (self._loglik_var(Y, alpha, psi, Theta, Beta)
                          - 0.5 * self.ridge_theta * float(np.sum(Theta ** 2))
                          + self._mix_logprior(Beta, mp))
            elif free2d:
                pfea = self._newton_block(Y.T, np.zeros(J), alpha, Xfea, pfea,
                                          prior_prec=np.array([0., 1e-6, 1e-6]))
                psi, Beta = pfea[:, 0], pfea[:, 1:]
                alpha, psi, Theta, Beta = self._identify_scale(alpha, psi, Theta, Beta)
                ll_new = (self._loglik_var(Y, alpha, psi, Theta, Beta)
                          - 0.5 * self.ridge_theta * float(np.sum(Theta ** 2)))
            else:
                pfea = self._newton_block(Y.T, np.zeros(J), alpha, Xfea, pfea,
                                          prior_prec=beta_prec)
                psi, Beta = pfea[:, 0], pfea[:, 1:]
                alpha, psi, Theta, Beta = self._identify(alpha, psi, Theta, Beta)
                ll_new = self._logpost(Y, alpha, psi, Theta, Beta)
            rel = (ll_new - ll_old) / (max(abs(ll_new), abs(ll_old)) + EPS)
            ll_hist.append(ll_new)
            if self.verbose and (it <= 3 or it % 10 == 0):
                print(f"    iter {it:4d}  logpost={ll_new:,.1f}  rel={rel:.2e}")
            if abs(rel) < self.rtol:
                stop_reason = f"rtol ({self.rtol:g}) met"
                ll_old = ll_new
                break
            ll_old = ll_new

        self.alpha_, self.psi_, self.theta_, self.beta_ = alpha, psi, Theta, Beta
        self.logpost_ = ll_old
        self.loglik_ = self._loglik_var(Y, alpha, psi, Theta, Beta)
        if mixture:
            self.mix_ = mp
            self.r_, self.omega_ = self._mix_estep(Beta, mp)
            self.se_theta_, self.se_beta_ = self._standard_errors(
                Y, alpha, psi, Theta, Beta, beta_prec=self.omega_)
        else:
            self.se_theta_, self.se_beta_ = self._standard_errors(Y, alpha, psi, Theta, Beta)
        self.n_iter_ = it
        self.stop_reason_ = stop_reason
        self.ll_hist_ = np.array(ll_hist)
        if self.verbose:
            print(f"    -> stopped: {stop_reason} | iters={it} | "
                  f"ll={ll_old:,.1f}")
        return self


class TopicIRT(IdealPointEstimator):
    """1-D Poisson topic-scaling IRT wrapper around ``PoissonScaleEM``.

        log lambda_ij = alpha_i + psi_j + beta_j * theta_i

    theta = the single latent document position column. Ported from
    ``irt_local_poisson``; the estimator runs with ``dims=1``.
    """

    name = "Topic IRT"

    def __init__(self, ridge_theta: float = 1.0, ridge_beta: float = 0.1,
                 rtol: float = 1e-6, max_iter: int = 500):
        self.ridge_theta = float(ridge_theta)
        self.ridge_beta = float(ridge_beta)
        self.rtol = float(rtol)
        self.max_iter = int(max_iter)

    def fit(self, Y):
        # Y: doc x topic count matrix (CSR or dense). PoissonScaleEM wants dense.
        Yd = Y.toarray() if issparse(Y) else np.asarray(Y, dtype=np.float64)
        model = PoissonScaleEM(
            dims=1, rtol=self.rtol, max_iter=self.max_iter,
            ridge_theta=self.ridge_theta, ridge_beta=self.ridge_beta,
            verbose=False,
        ).fit(Yd)
        self.model_ = model
        self.theta_ = np.asarray(model.theta_[:, 0], dtype=np.float64).ravel()
        return self


# =====================================================================
# 2-stage IRT  (ported verbatim from two_stage_stance_anchored_sparse.py
#               + two_stage_linear_fit.fit_linear)
# =====================================================================
class TwoStageStanceAnchoredFit:
    """Fitted parameters and compact convergence diagnostics.

    Plain-class port of the original dataclass container (no numerics)."""

    def __init__(self, theta_stance, alpha, beta, sigma2, psi, rho, kappa,
                 theta, doc_intercept, stage1_history, stage2_history,
                 stage1_converged, stage2_converged, stage3_fallback_documents):
        self.theta_stance = theta_stance
        self.alpha = alpha
        self.beta = beta
        self.sigma2 = sigma2
        self.psi = psi
        self.rho = rho
        self.kappa = kappa
        self.theta = theta
        self.doc_intercept = doc_intercept
        self.stage1_history = stage1_history
        self.stage2_history = stage2_history
        self.stage1_converged = stage1_converged
        self.stage2_converged = stage2_converged
        self.stage3_fallback_documents = stage3_fallback_documents


def _canonical_csr(Y):
    """Return a positive, sorted, duplicate-free float64 CSR count matrix."""
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
    """Row index for each canonical CSR data entry, aligned with Y.data."""
    return np.repeat(np.arange(Y.shape[0], dtype=np.int64), np.diff(Y.indptr))


def _aligned_stance_from_sum(Y, stance_sum):
    """Extract a sparse signed-stance sum at Y's cells and divide by Y.data."""
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
        if not np.any(valid):
            continue
        valid_idx = np.flatnonzero(valid)
        valid[valid_idx] &= zcols[take[valid_idx]] == ycols[valid_idx]
        if np.any(valid):
            pos = ya + np.flatnonzero(valid)
            out[pos] = Z.data[za:zb][take[valid]]

    return out / Y.data


def _prepare_stance(Y, stance):
    """Accept active-cell means or a sparse signed-stance-sum matrix."""
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


def _weighted_standardize(theta, w):
    """Return exact weighted mean-zero, unit-variance scores and old moments."""
    W = float(w.sum())
    mean = float(np.dot(w, theta) / W)
    centered = theta - mean
    scale = float(np.sqrt(np.dot(w, centered * centered) / W))
    if not np.isfinite(scale) or scale <= 64.0 * _EPS:
        raise FloatingPointError("The stance scale has effectively zero weighted variance.")
    return centered / scale, mean, scale


def _reidentify_stage1(theta, alpha, beta, w):
    """Reidentify theta exactly while preserving alpha_t + beta_t theta_i."""
    theta_new, mean, scale = _weighted_standardize(theta, w)
    return theta_new, alpha + beta * mean, beta * scale


def _loss_tail_ratio(improvement, largest_previous_improvement):
    """Plateau diagnostic for Stage 1/2 convergence."""
    return float(improvement / max(1.0, largest_previous_improvement))


# ----- Stage 1: sparse linear weighted ALS ----------------------------
def _stage1_sse(theta, alpha, beta, rows, cols, s, cell_w):
    resid = s - alpha[cols] - beta[cols] * theta[rows]
    return float(np.dot(cell_w, resid * resid))


def _stage1_initialize_theta(rows, y, s, w, N, orientation):
    """Document-level count-weighted mean stance, externally oriented."""
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


def _stage1_update_items(theta, rows, cols, s, cell_w, T):
    """Vectorized weighted 2x2 normal-equation update for (alpha_t, beta_t)."""
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

    threshold = 64.0 * _EPS * np.maximum.reduce((np.abs(swtt), np.abs(swt), sw, np.ones(T)))
    nonsingular = sxx > threshold
    beta = np.divide(sxs, sxx, out=np.zeros(T, dtype=np.float64), where=nonsingular)
    alpha = mean_s - beta * mean_t
    beta[~nonsingular] = 0.0
    alpha[~np.isfinite(alpha)] = 0.0
    beta[~np.isfinite(beta)] = 0.0
    return alpha, beta


def _stage1_update_theta(alpha, beta, rows, cols, s, cell_w, N):
    """Scalar weighted least-squares update for every document score."""
    b = beta[cols]
    num = np.bincount(rows, weights=cell_w * b * (s - alpha[cols]), minlength=N)
    den = np.bincount(rows, weights=cell_w * b * b, minlength=N)
    threshold = 64.0 * _EPS * np.maximum(den.max(initial=0.0), 1.0)
    return np.divide(num, den, out=np.zeros(N, dtype=np.float64), where=den > threshold)


def _fit_stage1(rows, cols, y, s, w, N, T, orientation, max_sweeps, tol_tail,
                patience, verbose):
    cell_w = w[rows] * y
    active_weight = float(w[rows].sum())
    theta = _stage1_initialize_theta(rows, y, s, w, N, orientation)

    alpha0 = np.divide(
        np.bincount(cols, weights=cell_w * s, minlength=T),
        np.bincount(cols, weights=cell_w, minlength=T),
        out=np.zeros(T, dtype=np.float64),
        where=np.bincount(cols, weights=cell_w, minlength=T) > 0.0,
    )
    beta0 = np.zeros(T, dtype=np.float64)
    sse_prev = _stage1_sse(theta, alpha0, beta0, rows, cols, s, cell_w)

    history = []
    largest_improvement = 0.0
    stable = 0
    converged = False
    alpha, beta = alpha0, beta0

    for sweep in range(1, max_sweeps + 1):
        alpha, beta = _stage1_update_items(theta, rows, cols, s, cell_w, T)
        theta_raw = _stage1_update_theta(alpha, beta, rows, cols, s, cell_w, N)
        theta, alpha, beta = _reidentify_stage1(theta_raw, alpha, beta, w)

        sse_now = _stage1_sse(theta, alpha, beta, rows, cols, s, cell_w)
        improvement = max(0.0, sse_prev - sse_now)
        tail_fraction = _loss_tail_ratio(improvement, largest_improvement)
        largest_improvement = max(largest_improvement, improvement)
        sigma2 = max(sse_now / active_weight, _TINY)

        row = {
            "sweep": float(sweep),
            "sse": float(sse_now),
            "sigma2": float(sigma2),
            "improvement": float(improvement),
            "tail_fraction": float(tail_fraction),
        }
        history.append(row)
        if verbose:
            print(
                f"stage 1 | sweep={sweep:3d} sse={sse_now:.8g} "
                f"tail={tail_fraction:.2e} sigma2={sigma2:.6g}"
            )

        is_stable = sweep > 1 and tail_fraction <= tol_tail
        stable = stable + 1 if is_stable else 0
        if stable >= patience:
            converged = True
            break
        sse_prev = sse_now

    return theta, alpha, beta, float(sigma2), history, converged


# ----- Stage 2: profiled count items on the frozen linear stance scale
def _solve_batched_3x3(H, g):
    """Solve all 3x3 Newton systems with only scale-relative numerical jitter."""
    Hs = H.copy()
    diag_scale = np.maximum(np.trace(Hs, axis1=1, axis2=2) / 3.0, 1.0)
    Hs[:, np.arange(3), np.arange(3)] += (1e-12 * diag_scale)[:, None]
    try:
        return np.linalg.solve(Hs, g[..., None])[..., 0]
    except np.linalg.LinAlgError:
        out = np.zeros_like(g)
        for t in range(Hs.shape[0]):
            out[t] = np.linalg.lstsq(Hs[t], g[t], rcond=None)[0]
        return out


def _stage2_observed_statistics(rows, cols, y, w, theta, h, T):
    """Sparse observed sufficient statistics for z_i=(1, theta_i, h_i)."""
    wy = w[rows] * y
    obs0 = np.bincount(cols, weights=wy, minlength=T)
    obs1 = np.bincount(cols, weights=wy * theta[rows], minlength=T)
    obs2 = np.bincount(cols, weights=wy * h[rows], minlength=T)
    return obs0, obs1, obs2


def _stage2_objective(psi, rho, kappa, theta, h, weighted_totals, obs0, obs1,
                      obs2, chunk_size):
    """Profiled negative Poisson log-likelihood, up to constants."""
    lse_sum = 0.0
    N = theta.size
    for lo in range(0, N, chunk_size):
        hi = min(lo + chunk_size, N)
        r = psi[None, :] + theta[lo:hi, None] * rho[None, :] + h[lo:hi, None] * kappa[None, :]
        lse_sum += float(np.dot(weighted_totals[lo:hi], logsumexp(r, axis=1)))
    observed = float(np.dot(psi, obs0) + np.dot(rho, obs1) + np.dot(kappa, obs2))
    return lse_sum - observed


def _stage2_gradient_hessian(psi, rho, kappa, theta, h, weighted_totals, obs0,
                             obs1, obs2, chunk_size):
    """Conditional-on-profiled-a_i triplet score and Poisson IRLS curvature."""
    T = psi.size
    e0 = np.zeros(T, dtype=np.float64)
    e1 = np.zeros(T, dtype=np.float64)
    e2 = np.zeros(T, dtype=np.float64)
    e11 = np.zeros(T, dtype=np.float64)
    e12 = np.zeros(T, dtype=np.float64)
    e22 = np.zeros(T, dtype=np.float64)

    N = theta.size
    for lo in range(0, N, chunk_size):
        hi = min(lo + chunk_size, N)
        th = theta[lo:hi]
        hh = h[lo:hi]
        r = psi[None, :] + th[:, None] * rho[None, :] + hh[:, None] * kappa[None, :]
        p = np.exp(r - logsumexp(r, axis=1, keepdims=True))
        mu_w = weighted_totals[lo:hi, None] * p

        e0 += mu_w.sum(axis=0)
        e1 += (mu_w * th[:, None]).sum(axis=0)
        e2 += (mu_w * hh[:, None]).sum(axis=0)
        e11 += (mu_w * (th[:, None] * th[:, None])).sum(axis=0)
        e12 += (mu_w * (th[:, None] * hh[:, None])).sum(axis=0)
        e22 += (mu_w * (hh[:, None] * hh[:, None])).sum(axis=0)

    g = np.column_stack((e0 - obs0, e1 - obs1, e2 - obs2))
    H = np.empty((T, 3, 3), dtype=np.float64)
    H[:, 0, 0] = e0
    H[:, 0, 1] = H[:, 1, 0] = e1
    H[:, 0, 2] = H[:, 2, 0] = e2
    H[:, 1, 1] = e11
    H[:, 1, 2] = H[:, 2, 1] = e12
    H[:, 2, 2] = e22
    return g, H


def _profile_doc_intercepts(theta, h, yplus, psi, rho, kappa, chunk_size):
    """Exact a_i = log(Y_i+) - logsumexp_t r_it for fixed count items."""
    N = theta.size
    a = np.empty(N, dtype=np.float64)
    for lo in range(0, N, chunk_size):
        hi = min(lo + chunk_size, N)
        r = psi[None, :] + theta[lo:hi, None] * rho[None, :] + h[lo:hi, None] * kappa[None, :]
        a[lo:hi] = np.log(yplus[lo:hi]) - logsumexp(r, axis=1)
    return a


def _fit_stage2(Y, rows, cols, y, w, theta, beta, max_sweeps, max_backtracks,
                tol_tail, patience, chunk_size, verbose):
    N, T = Y.shape
    yplus = np.asarray(Y.sum(axis=1)).ravel().astype(np.float64)
    weighted_totals = w * yplus
    h = theta * theta - 1.0
    obs0, obs1, obs2 = _stage2_observed_statistics(rows, cols, y, w, theta, h, T)

    psi = np.log(np.maximum(obs0, _TINY))
    psi -= psi.mean()
    rho = np.zeros(T, dtype=np.float64)
    kappa = np.zeros(T, dtype=np.float64)

    fix_kappa = beta == 0.0
    L_prev = _stage2_objective(
        psi, rho, kappa, theta, h, weighted_totals, obs0, obs1, obs2, chunk_size
    )

    history = []
    largest_improvement = 0.0
    stable = 0
    converged = False

    for sweep in range(1, max_sweeps + 1):
        g, H = _stage2_gradient_hessian(
            psi, rho, kappa, theta, h, weighted_totals, obs0, obs1, obs2, chunk_size
        )
        if np.any(fix_kappa):
            g[fix_kappa, 2] = 0.0
            H[fix_kappa, 0, 2] = H[fix_kappa, 2, 0] = 0.0
            H[fix_kappa, 1, 2] = H[fix_kappa, 2, 1] = 0.0
            H[fix_kappa, 2, 2] = 1.0

        delta = _solve_batched_3x3(H, g)
        delta[fix_kappa, 2] = 0.0

        accepted = False
        step_used = 0.0
        L_now = L_prev
        for bt in range(max_backtracks):
            step = 0.5 ** bt
            psi_try = psi - step * delta[:, 0]
            psi_try -= psi_try.mean()
            rho_try = rho - step * delta[:, 1]
            kappa_try = kappa - step * delta[:, 2]
            kappa_try[fix_kappa] = 0.0

            L_try = _stage2_objective(
                psi_try, rho_try, kappa_try, theta, h, weighted_totals,
                obs0, obs1, obs2, chunk_size,
            )
            if np.isfinite(L_try) and L_try <= L_prev + 1e-12 * (1.0 + abs(L_prev)):
                psi, rho, kappa = psi_try, rho_try, kappa_try
                L_now = L_try
                accepted = True
                step_used = step
                break

        improvement = max(0.0, L_prev - L_now)
        tail_fraction = _loss_tail_ratio(improvement, largest_improvement)
        largest_improvement = max(largest_improvement, improvement)
        newton_step_max = float(np.max(np.abs(delta)))

        row = {
            "sweep": float(sweep),
            "nll": float(L_now),
            "improvement": float(improvement),
            "tail_fraction": float(tail_fraction),
            "newton_step_max": newton_step_max,
            "accepted_step": float(step_used),
            "step_accepted": float(accepted),
        }
        history.append(row)
        if verbose:
            print(
                f"stage 2 | sweep={sweep:3d} nll={L_now:.8g} "
                f"tail={tail_fraction:.2e} step={step_used:.3g}"
            )

        is_stable = sweep > 1 and tail_fraction <= tol_tail
        stable = stable + 1 if is_stable else 0
        if stable >= patience:
            converged = True
            break
        if not accepted:
            converged = bool(tail_fraction <= tol_tail)
            break
        L_prev = L_now

    return psi, rho, kappa, history, converged


# ----- Stage 3: fixed-item combined document scoring ------------------
def _stage3_nll_all(theta, Y, s, w, alpha, beta, sigma2, psi, rho, kappa,
                    rows, cols, y, yplus, chunk_size):
    """Fixed-item Stage-3 negative objective for all documents."""
    N = theta.size
    out = np.empty(N, dtype=np.float64)

    for lo in range(0, N, chunk_size):
        hi = min(lo + chunk_size, N)
        th = theta[lo:hi]
        hh = th * th - 1.0
        r = psi[None, :] + th[:, None] * rho[None, :] + hh[:, None] * kappa[None, :]
        lse = logsumexp(r, axis=1)

        aa, bb = Y.indptr[lo], Y.indptr[hi]
        lr = rows[aa:bb] - lo
        cc = cols[aa:bb]
        yy = y[aa:bb]
        wr = w[rows[aa:bb]]
        q_active = alpha[cc] + beta[cc] * th[lr]
        stance_sse = np.bincount(
            lr, weights=wr * yy * (s[aa:bb] - q_active) ** 2, minlength=hi - lo,
        )
        observed_r = np.bincount(
            lr, weights=wr * yy * r[lr, cc], minlength=hi - lo,
        )
        out[lo:hi] = 0.5 * stance_sse / sigma2 + w[lo:hi] * yplus[lo:hi] * lse - observed_r

    return out


def _stage3_full_all(theta, Y, s, w, alpha, beta, sigma2, psi, rho, kappa,
                     rows, cols, y, yplus, chunk_size):
    """Document NLL, gradient, exact Hessian, and positive Fisher/GN Hessian."""
    N = theta.size
    nll = np.empty(N, dtype=np.float64)
    grad = np.empty(N, dtype=np.float64)
    hess = np.empty(N, dtype=np.float64)
    hess_fisher = np.empty(N, dtype=np.float64)
    rpp_const = 2.0 * kappa

    for lo in range(0, N, chunk_size):
        hi = min(lo + chunk_size, N)
        th = theta[lo:hi]
        hh = th * th - 1.0
        r = psi[None, :] + th[:, None] * rho[None, :] + hh[:, None] * kappa[None, :]
        rp = rho[None, :] + 2.0 * th[:, None] * kappa[None, :]
        lse = logsumexp(r, axis=1)
        p = np.exp(r - lse[:, None])

        erp = (p * rp).sum(axis=1)
        erpp = (p * rpp_const[None, :]).sum(axis=1)
        varrp = np.maximum((p * rp * rp).sum(axis=1) - erp * erp, 0.0)

        aa, bb = Y.indptr[lo], Y.indptr[hi]
        lr = rows[aa:bb] - lo
        cc = cols[aa:bb]
        yy = y[aa:bb]
        wr = w[rows[aa:bb]]
        q_active = alpha[cc] + beta[cc] * th[lr]
        residual = q_active - s[aa:bb]
        b_active = beta[cc]

        stance_sse = np.bincount(lr, weights=wr * yy * residual * residual, minlength=hi - lo)
        stance_g = np.bincount(lr, weights=wr * yy * residual * b_active, minlength=hi - lo) / sigma2
        stance_h = np.bincount(lr, weights=wr * yy * b_active * b_active, minlength=hi - lo) / sigma2

        obs_r = np.bincount(lr, weights=wr * yy * r[lr, cc], minlength=hi - lo)
        obs_rp = np.bincount(lr, weights=wr * yy * rp[lr, cc], minlength=hi - lo)
        obs_rpp = np.bincount(lr, weights=wr * yy * rpp_const[cc], minlength=hi - lo)
        wyplus = w[lo:hi] * yplus[lo:hi]

        nll[lo:hi] = 0.5 * stance_sse / sigma2 + wyplus * lse - obs_r
        grad[lo:hi] = stance_g + wyplus * erp - obs_rp
        hess[lo:hi] = stance_h + wyplus * (erpp + varrp) - obs_rpp
        hess_fisher[lo:hi] = stance_h + wyplus * varrp

    return nll, grad, hess, hess_fisher


def _stage3_doc_quantities(i, theta, Y, s, w, alpha, beta, sigma2, psi, rho, kappa):
    """Scalar Stage-3 NLL / gradient / exact Hessian / Fisher-GN Hessian."""
    a, b = Y.indptr[i], Y.indptr[i + 1]
    cc = Y.indices[a:b]
    yy = Y.data[a:b]
    ss = s[a:b]
    wi = float(w[i])
    yplus = float(yy.sum())

    h = theta * theta - 1.0
    r = psi + rho * theta + kappa * h
    rp = rho + 2.0 * kappa * theta
    rpp = 2.0 * kappa
    lse = float(logsumexp(r))
    p = np.exp(r - lse)
    erp = float(np.dot(p, rp))
    erpp = float(np.dot(p, rpp))
    varrp = max(float(np.dot(p, rp * rp) - erp * erp), 0.0)

    q = alpha[cc] + beta[cc] * theta
    residual = q - ss
    stance_nll = 0.5 * wi * float(np.dot(yy, residual * residual)) / sigma2
    stance_g = wi * float(np.dot(yy, residual * beta[cc])) / sigma2
    stance_h = wi * float(np.dot(yy, beta[cc] * beta[cc])) / sigma2

    obs_r = wi * float(np.dot(yy, r[cc]))
    obs_rp = wi * float(np.dot(yy, rp[cc]))
    obs_rpp = wi * float(np.dot(yy, rpp[cc]))
    nll = stance_nll + wi * yplus * lse - obs_r
    grad = stance_g + wi * yplus * erp - obs_rp
    hess = stance_h + wi * yplus * (erpp + varrp) - obs_rpp
    hess_fisher = stance_h + wi * yplus * varrp
    return nll, grad, hess, hess_fisher


def _fallback_rescore_document(i, grid_lo, grid_hi, grid_size, max_backtracks,
                               tol_step, *args):
    """Deterministic grid initialization followed by safeguarded scalar Newton."""
    grid = np.linspace(grid_lo, grid_hi, grid_size)
    values = np.array([_stage3_doc_quantities(i, value, *args)[0] for value in grid])
    theta = float(grid[int(np.argmin(values))])

    for _ in range(30):
        nll, grad, hess, hess_fisher = _stage3_doc_quantities(i, theta, *args)
        denom = hess if np.isfinite(hess) and hess > 1e-14 else hess_fisher
        if not np.isfinite(denom) or denom <= 1e-14:
            break
        direction = -grad / denom
        if abs(direction) <= tol_step * (1.0 + abs(theta)):
            break

        accepted = False
        for bt in range(max_backtracks):
            candidate = theta + (0.5 ** bt) * direction
            cand_nll = _stage3_doc_quantities(i, candidate, *args)[0]
            if np.isfinite(cand_nll) and cand_nll <= nll + 1e-12 * (1.0 + abs(nll)):
                theta = candidate
                accepted = True
                break
        if not accepted:
            break
    return theta


def _fit_stage3(Y, s, w, theta_stance, alpha, beta, sigma2, psi, rho, kappa,
                rows, cols, y, chunk_size, max_iter, max_backtracks, tol_step,
                grid_size, verbose):
    """Parallel document rescoring with document-specific safeguarded Newton."""
    N = Y.shape[0]
    yplus = np.asarray(Y.sum(axis=1)).ravel().astype(np.float64)
    theta = theta_stance.copy()
    converged = np.zeros(N, dtype=bool)
    failed = np.zeros(N, dtype=bool)

    for iteration in range(1, max_iter + 1):
        nll, grad, hess, hess_fisher = _stage3_full_all(
            theta, Y, s, w, alpha, beta, sigma2, psi, rho, kappa,
            rows, cols, y, yplus, chunk_size,
        )
        denom = np.where(np.isfinite(hess) & (hess > 1e-14), hess, hess_fisher)
        direction = np.divide(
            -grad, denom, out=np.zeros_like(grad),
            where=np.isfinite(denom) & (denom > 1e-14),
        )
        small = np.abs(direction) <= tol_step * (1.0 + np.abs(theta))
        converged |= small
        todo = ~(converged | failed)
        if not np.any(todo):
            break

        accepted = np.zeros(N, dtype=bool)
        remaining = todo.copy()
        step = np.ones(N, dtype=np.float64)
        for _ in range(max_backtracks):
            if not np.any(remaining):
                break
            candidate = theta.copy()
            candidate[remaining] = theta[remaining] + step[remaining] * direction[remaining]
            cand_nll = _stage3_nll_all(
                candidate, Y, s, w, alpha, beta, sigma2, psi, rho, kappa,
                rows, cols, y, yplus, chunk_size,
            )
            ok = remaining & np.isfinite(cand_nll) & (cand_nll <= nll + 1e-12 * (1.0 + np.abs(nll)))
            if np.any(ok):
                theta[ok] = candidate[ok]
                accepted[ok] = True
                converged[ok] |= np.abs(step[ok] * direction[ok]) <= tol_step * (1.0 + np.abs(theta[ok]))
            remaining &= ~ok
            step[remaining] *= 0.5

        failed |= remaining
        if verbose:
            print(
                f"stage 3 | iter={iteration:2d} accepted={int(accepted.sum()):,} "
                f"converged={int(converged.sum()):,} fallback_pending={int(failed.sum()):,}"
            )
        if not np.any(accepted) and np.any(todo):
            break

    fallback = (~converged) | failed
    if np.any(fallback):
        grid_lo = float(theta_stance.min())
        grid_hi = float(theta_stance.max())
        if grid_hi - grid_lo <= 64.0 * _EPS:
            grid_lo, grid_hi = -3.0, 3.0
        common = (Y, s, w, alpha, beta, sigma2, psi, rho, kappa)
        for i in np.flatnonzero(fallback):
            theta[i] = _fallback_rescore_document(
                int(i), grid_lo, grid_hi, grid_size, max_backtracks, tol_step, *common
            )

    h_final = theta * theta - 1.0
    doc_intercept = _profile_doc_intercepts(theta, h_final, yplus, psi, rho, kappa, chunk_size)
    return theta, doc_intercept, int(fallback.sum())


def _fit_linear(Y, stance, *, orientation=1.0, chunk_size=32768,
                stage1_max_sweeps=100, stage2_max_sweeps=100, max_backtracks=20,
                tol_loss_tail=1e-4, patience=3, stage3_max_iter=30,
                stage3_tol_step=1e-6, stage3_grid_size=33, verbose=True):
    """Identical orchestration to fit_two_stage_stance_anchored, but kappa is
    pinned to 0 (LINEAR variant). Ported verbatim from
    ``two_stage_linear_fit.fit_linear``: kappa is pinned by passing a zero beta
    into Stage 2 (its only use of beta is fix_kappa)."""
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
    w = _prepare_sample_weight(None, N)
    if verbose:
        print(f"LINEAR two-step stance-anchored IRT | N={N:,} T={T:,} active cells={Y.nnz:,}")

    theta_stance, alpha, beta, sigma2, s1h, s1c = _fit_stage1(
        rows, cols, y, s, w, N, T, orientation, stage1_max_sweeps,
        tol_loss_tail, patience, verbose)

    # Pin kappa == 0 for ALL topics: pass a zero beta into Stage 2.
    beta_zero = np.zeros(T, dtype=np.float64)
    psi, rho, kappa, s2h, s2c = _fit_stage2(
        Y, rows, cols, y, w, theta_stance, beta_zero, stage2_max_sweeps,
        max_backtracks, tol_loss_tail, patience, chunk_size, verbose)
    assert np.allclose(kappa, 0.0), "kappa must be identically zero in the linear model"

    theta, doc_intercept, fb = _fit_stage3(
        Y, s, w, theta_stance, alpha, beta, sigma2, psi, rho, kappa, rows, cols, y,
        chunk_size, stage3_max_iter, max_backtracks, stage3_tol_step,
        stage3_grid_size, verbose)

    return TwoStageStanceAnchoredFit(
        theta_stance=theta_stance, alpha=alpha, beta=beta, sigma2=float(sigma2),
        psi=psi, rho=rho, kappa=kappa, theta=theta, doc_intercept=doc_intercept,
        stage1_history=s1h, stage2_history=s2h, stage1_converged=s1c,
        stage2_converged=s2c, stage3_fallback_documents=fb)


class TwoStageIRT(IdealPointEstimator):
    """Sparse three-stage stance-anchored IRT, LINEAR variant (kappa == 0).

        Stage 1: S_it = alpha_t + beta_t theta_i^S       (weighted linear ALS)
        Stage 2: log mu_it = a_i + psi_t + rho_t theta_i^S   (kappa_t pinned to 0)
        Stage 3: fixed-item combined document scoring; theta = Stage-3 theta.

    Ported from ``two_stage_linear_fit.fit_linear`` + the stage fitters in
    ``two_stage_stance_anchored_sparse``. The theta output is the Stage-3 combined
    score (NOT re-standardized, original sign).
    """

    name = "2-stage IRT"

    def __init__(self, **fit_linear_kwargs):
        fit_linear_kwargs.setdefault("verbose", False)
        self.fit_linear_kwargs = dict(fit_linear_kwargs)

    def fit(self, Y, S):
        # Y: CSR doc x topic count; S: CSR doc x topic stance-sum.
        Yc = Y if issparse(Y) else csr_matrix(np.asarray(Y, dtype=np.float64))
        Sc = S if issparse(S) else csr_matrix(np.asarray(S, dtype=np.float64))
        fit = _fit_linear(Yc, Sc, **self.fit_linear_kwargs)
        self.fit_ = fit
        self.theta_ = np.asarray(fit.theta, dtype=np.float64).ravel()
        return self


if __name__ == "__main__":
    print("07A ideal-point estimators:",
          [c.name for c in (EmbeddingPCA, StanceIRT, TopicIRT, TwoStageIRT)])
