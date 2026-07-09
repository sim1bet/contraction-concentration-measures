"""
Common utilities for the Automatica submission
"Contraction and concentration of measures with applications to theoretical
neuroscience" (autosam.tex).

This module gathers everything shared across the validation experiments:
  * the input-driven (IDP) Hopfield model of Definition 12 (Sec. 5),
  * a vectorized Euler-Maruyama ensemble integrator (supports both the
    spatially-inhomogeneous diffusion of Sec. 3 and the constant isotropic
    diffusion of Sec. 4),
  * an exact (grid-search) local/global contraction-rate estimator based on
    the symmetrized Jacobian, as used throughout Secs. 3-5,
  * light-weight statistical helpers (bootstrap CIs, subsampled empirical
    2-Wasserstein distance via optimal assignment) that avoid any extra
    dependency beyond numpy/scipy/pandas/matplotlib/sklearn,
  * a uniform plotting style and a "save figure + companion .csv" helper so
    that every figure can be reproduced in tikz/pgfplots.

Every experiment script under this folder imports from here and is
independently runnable; `run_all.py` regenerates all figures/CSVs used in
the paper with fixed, documented random seeds.
"""
from __future__ import annotations

import dataclasses
import json
import os
import zlib
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import root
from scipy.optimize import linear_sum_assignment

# --------------------------------------------------------------------------
# Paths & reproducibility
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "results", "figures")
CSV_DIR = os.path.join(ROOT, "results", "csv")
LOG_DIR = os.path.join(ROOT, "results", "logs")
for _d in (FIG_DIR, CSV_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

MASTER_SEED = 20260706  # fixed & documented (cf. manuscript, numerical methodology)


def rng_for(tag: str, master_seed: int = MASTER_SEED) -> np.random.Generator:
    """Deterministic, experiment-scoped RNG independent of Python's (randomized)
    string hashing, so results are reproducible across processes/runs."""
    tag_seed = zlib.crc32(tag.encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence([master_seed, tag_seed]))


def log_json(name: str, payload: dict) -> str:
    path = os.path.join(LOG_DIR, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    return path


# --------------------------------------------------------------------------
# Input-driven Hopfield model (Definition 12)
# --------------------------------------------------------------------------
_M_MATRIX = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)


@dataclasses.dataclass(frozen=True)
class HopfieldConfig:
    name: str
    beta: float
    u: np.ndarray
    R_max: float
    description: str = ""

    @property
    def W(self) -> np.ndarray:
        return _M_MATRIX @ np.diag(self.u) @ _M_MATRIX.T

    @property
    def M(self) -> np.ndarray:
        return _M_MATRIX


REGIMES = {
    "global": HopfieldConfig(
        name="global", beta=2.0, u=np.array([0.2, 0.25]), R_max=1.5,
        description="Globally contracting regime (Theorem 6, Fig. 2)",
    ),
    "multistable": HopfieldConfig(
        name="multistable", beta=2.0, u=np.array([1.0, 3.0]), R_max=1.2,
        description="B_r-contracting regime, 4 equilibria (Theorem 10, "
                     "Corollary and Conjecture 15, Fig. 3)",
    ),
}


def phi(x: np.ndarray, beta: float) -> np.ndarray:
    return np.tanh(beta * x)


def dphi(x: np.ndarray, beta: float) -> np.ndarray:
    t = np.tanh(beta * x)
    return beta * (1.0 - t * t)


def drift(x: np.ndarray, cfg: HopfieldConfig) -> np.ndarray:
    """f(x) = -x + W phi(x); x has shape (...,2)."""
    p = phi(x, cfg.beta)
    return -x + p @ cfg.W.T


def jacobian(x: np.ndarray, cfg: HopfieldConfig) -> np.ndarray:
    """J(x) = -I + W diag(phi'(x)); vectorized over leading dims of x (...,2),
    including the unbatched single-point case x.shape == (2,).
    Returns array of shape (...,2,2)."""
    d = dphi(x, cfg.beta)  # (...,2)
    Jb = np.einsum('ij,...j->...ij', cfg.W, d)
    return Jb - np.eye(2)


def energy(x: np.ndarray, cfg: HopfieldConfig) -> np.ndarray:
    """E(x) = -1/2 phi(x)^T W phi(x) + x^T phi(x) - sum_i (1/beta) log cosh(beta x_i)."""
    p = phi(x, cfg.beta)
    quad = -0.5 * np.einsum('...i,ij,...j->...', p, cfg.W, p)
    lin = np.einsum('...i,...i->...', x, p)
    integ = np.sum(np.log(np.cosh(cfg.beta * x)) / cfg.beta, axis=-1)
    return quad + lin - integ


@dataclasses.dataclass
class Equilibrium:
    x: np.ndarray
    eigs: np.ndarray
    stable: bool
    energy: float
    saliency: float = float("nan")


def saliency_of(x: np.ndarray, cfg: HopfieldConfig) -> float:
    """Saliency/input-strength proxy for a memory equilibrium.

    M[:, i] is an eigenvector of W_u = M diag(u) M^T with eigenvalue u_i
    (M is orthogonal), so u_i plays exactly the role of the saliency weight
    alpha_mu of Sec. 5.1 for the canonical pattern directions of this 2-D
    example: existence/stability/energy-depth are all governed by u_i,
    matching items (1)-(3) of Sec. 5.1. Returns 0 at the origin."""
    if np.linalg.norm(x) < 1e-6:
        return 0.0
    proj = cfg.M.T @ x
    mu_idx = int(np.argmax(np.abs(proj)))
    return float(cfg.u[mu_idx])


def find_equilibria(cfg: HopfieldConfig) -> list[Equilibrium]:
    """Root-finds all equilibria relevant to a regime and classifies them by
    local stability (eigenvalues of the Jacobian), energy, and saliency."""
    if cfg.name == "global":
        guesses = [np.zeros(2)]
    else:
        M = cfg.M
        guesses = [3 * M[:, 0], -3 * M[:, 0], 3 * M[:, 1], -3 * M[:, 1], np.zeros(2)]

    found: list[Equilibrium] = []
    for g in guesses:
        sol = root(lambda x: drift(x, cfg), g, tol=1e-13)
        if not sol.success:
            continue
        x = sol.x
        if any(np.linalg.norm(x - e.x) < 1e-4 for e in found):
            continue
        J = jacobian(x, cfg)
        eig = np.linalg.eigvals(J)
        stable = bool(np.all(eig.real < -1e-9))
        sal = saliency_of(x, cfg)
        found.append(Equilibrium(x=x, eigs=eig, stable=stable,
                                  energy=float(energy(x, cfg)), saliency=sal))
    found.sort(key=lambda e: e.energy)
    return found


# --------------------------------------------------------------------------
# Exact contraction-rate estimator (symmetrized Jacobian, grid search)
# --------------------------------------------------------------------------
def exact_contraction_rate(cfg: HopfieldConfig, center: np.ndarray, R: float,
                            n_r: int = 60, n_th: int = 180, r_min: float = 0.0) -> float:
    """c_R = -max_{x in B_R(center)} lambda_max( sym(J(x)) ), estimated on a
    polar grid; c-strong (B_r-)contractivity (Definitions 5/13) is equivalent
    to this quantity being positive, by the mean-value theorem for vector
    fields applied along segments in the (convex) ball."""
    rs = np.linspace(r_min, R, n_r)
    ths = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    RR, TT = np.meshgrid(rs, ths, indexing='ij')
    pts = center + np.stack([RR * np.cos(TT), RR * np.sin(TT)], axis=-1)
    J = jacobian(pts, cfg)
    Jsym = 0.5 * (J + np.swapaxes(J, -1, -2))
    eigmax = np.linalg.eigvalsh(Jsym)[..., -1]
    return float(-eigmax.max())


def global_contraction_rate(cfg: HopfieldConfig, box: float = 4.0,
                             n_r: int = 200, n_th: int = 400) -> float:
    """Same estimator evaluated over a large disk, used as a proxy for the
    global contraction rate c of Theorem 6 (the sup is attained near the
    origin since phi' is maximal there and decays monotonically with |x|)."""
    return exact_contraction_rate(cfg, np.zeros(2), box, n_r=n_r, n_th=n_th)


def max_valid_radius(cfg: HopfieldConfig, center: np.ndarray, R_cap: float,
                      n_grid: int = 60, n_r: int = 40, n_th: int = 120) -> float:
    """Largest radius R <= R_cap on which Assumption 13 (local B_R-contractivity
    around `center`) actually holds, i.e. the sup search of
    `exact_contraction_rate` stays positive. This is the numerical stand-in for
    "R is limited by the surrounding saddle regions" (remark after Assumption
    13): the search shrinks R from R_cap until c_R(R) > 0."""
    for R in np.linspace(R_cap, 1e-3, n_grid):
        if exact_contraction_rate(cfg, center, R, n_r=n_r, n_th=n_th) > 0:
            return float(R)
    return 0.0


# --------------------------------------------------------------------------
# Diffusion terms
# --------------------------------------------------------------------------
def diagonal_diffusion_inhomogeneous(x: np.ndarray, scale: float = 0.4) -> np.ndarray:
    """G(x) = scale * diag(sin x_1, cos x_2), used in the globally contracting
    example (Sec. 5.2). Exactly `scale`-Lipschitz in Frobenius norm since sin,
    cos are 1-Lipschitz, so L_G = scale exactly."""
    d1 = scale * np.sin(x[..., 0])
    d2 = scale * np.cos(x[..., 1])
    return np.stack([d1, d2], axis=-1)


L_G_INHOMOGENEOUS = 0.4  # exact Lipschitz constant of diagonal_diffusion_inhomogeneous


# --------------------------------------------------------------------------
# Vectorized Euler-Maruyama ensemble integrator
# --------------------------------------------------------------------------
def simulate_ensemble(cfg: HopfieldConfig, x0: np.ndarray, T: float, dt: float,
                       rng: np.random.Generator,
                       diffusion: str = "isotropic", omega: float = 0.4,
                       diag_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                       record_every: int = 0) -> np.ndarray:
    """Integrates N particles x0 (N,2) forward for time T with step dt.

    diffusion:
      "isotropic"  -> constant G = omega * I_2 (Theorem 10 / Sec. 4-5 setting)
      "diagonal"   -> multiplicative diagonal noise via diag_fn(x) -> (N,2)
                      entries (Sec. 3 spatially-inhomogeneous example)

    If record_every > 0, returns an array of shape (n_snapshots, N, 2) with a
    snapshot every `record_every` steps; otherwise returns only the final
    state (N, 2).
    """
    x = np.array(x0, dtype=float, copy=True)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("x0 must have shape (N, 2)")
    n_steps = int(round(T / dt))
    sqdt = np.sqrt(dt)
    recorded = []
    for k in range(n_steps):
        xi = rng.standard_normal(x.shape)
        if diffusion == "isotropic":
            noise = omega * sqdt * xi
        elif diffusion == "diagonal":
            noise = diag_fn(x) * sqdt * xi
        else:
            raise ValueError(f"unknown diffusion mode {diffusion!r}")
        x = x + dt * drift(x, cfg) + noise
        if record_every and (k + 1) % record_every == 0:
            recorded.append(x.copy())
    if record_every:
        return np.stack(recorded, axis=0)
    return x


def simulate_ensemble_coupled(cfg: HopfieldConfig, x0: np.ndarray, z0: np.ndarray,
                               T: float, dt: float, rng: np.random.Generator,
                               diffusion: str = "isotropic", omega: float = 0.4,
                               diag_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                               record_every: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Synchronous (parallel) coupling of two ensembles X_t, Z_t driven by the
    SAME Brownian increments, exactly as in the proof of Theorem 6 /
    Proposition 7. Returns (X snapshots, Z snapshots), each of shape
    (n_snapshots, N, 2)."""
    x = np.array(x0, dtype=float, copy=True)
    z = np.array(z0, dtype=float, copy=True)
    if x.shape != z.shape:
        raise ValueError("x0 and z0 must have identical shape")
    n_steps = int(round(T / dt))
    sqdt = np.sqrt(dt)
    xs, zs = [], []
    for k in range(n_steps):
        xi = rng.standard_normal(x.shape)  # SAME noise realization for X and Z
        if diffusion == "isotropic":
            nx = omega * sqdt * xi
            nz = omega * sqdt * xi
        elif diffusion == "diagonal":
            nx = diag_fn(x) * sqdt * xi
            nz = diag_fn(z) * sqdt * xi
        else:
            raise ValueError(f"unknown diffusion mode {diffusion!r}")
        x = x + dt * drift(x, cfg) + nx
        z = z + dt * drift(z, cfg) + nz
        if record_every and (k + 1) % record_every == 0:
            xs.append(x.copy())
            zs.append(z.copy())
    return np.stack(xs, axis=0), np.stack(zs, axis=0)


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def bootstrap_mean_ci(indicator: np.ndarray, n_boot: int = 2000,
                       rng: Optional[np.random.Generator] = None,
                       alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of a (Bernoulli-like) indicator."""
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(indicator, dtype=float)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def bootstrap_ratio_ci(a_indicator: np.ndarray, b_indicator: np.ndarray,
                        n_boot: int = 2000, rng: Optional[np.random.Generator] = None,
                        alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the ratio of two Bernoulli-like means,
    resampled jointly (paired resampling of the same particle cloud)."""
    if rng is None:
        rng = np.random.default_rng(0)
    a = np.asarray(a_indicator, dtype=float)
    b = np.asarray(b_indicator, dtype=float)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    ratios = a[idx].mean(axis=1) / np.clip(b[idx].mean(axis=1), 1e-12, None)
    point = a.mean() / max(b.mean(), 1e-12)
    lo, hi = np.percentile(ratios, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def empirical_w2_squared(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator,
                          n_sub: int = 250, n_reps: int = 8) -> tuple[float, float]:
    """Subsampled optimal-assignment estimator of W_2^2(empirical(X), empirical(Y)).
    Exact discrete optimal transport (Hungarian algorithm) on equal-size random
    subsamples, averaged over independent repetitions -- accurate for the
    moderate cloud sizes used here without requiring extra OT dependencies."""
    nX, nY = len(X), len(Y)
    m = min(n_sub, nX, nY)
    vals = []
    for _ in range(n_reps):
        ix = rng.choice(nX, size=m, replace=False)
        iy = rng.choice(nY, size=m, replace=False)
        A, B = X[ix], Y[iy]
        C = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        r, c = linear_sum_assignment(C)
        vals.append(C[r, c].mean())
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std())


# --------------------------------------------------------------------------
# Plotting style & figure/csv export
# --------------------------------------------------------------------------
PALETTE = {
    "empirical": "#1f6fb4",
    "theory": "#d1495b",
    "conservative": "#2a9d8f",
    "secondary": "#8e44ad",
    "aux": "#7f7f7f",
    "memA": "#264653",
    "memB": "#e76f51",
}


def set_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 13,
        "axes.titlesize": 12.5,
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.2,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 18,
        "figure.autolayout": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, name: str) -> str:
    path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    return path


def save_csv(name: str, columns: dict) -> str:
    import pandas as pd
    df = pd.DataFrame(columns)
    path = os.path.join(CSV_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    return path
