"""
Grid-based stationary Fokker-Planck-Kolmogorov (FPK) solver, used as an
independent cross-check of the particle/KDE estimate of mu* (manuscript
numerical methodology, item (iv)).

We discretize
    0 = -div(f mu) + (omega^2/2) Laplacian(mu)
on a rectangular grid with absorbing (Dirichlet, mu=0) far-field boundary,
using a first-order upwind finite-volume scheme for the advective term
(stable, produces an M-matrix) and standard central differences for the
diffusive term. The discretized generator A has a near-zero eigenvalue
whose eigenvector approximates the (truncated) stationary density; we find
it via shift-invert Arnoldi iteration (`scipy.sparse.linalg.eigs`, sigma=0).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs

import common as C


def solve_stationary_fpk(cfg: C.HopfieldConfig, omega: float, L: float, n: int):
    """Returns (xs, ys, mu_grid) with mu_grid shape (n, n), mu_grid[i, j]
    the estimated stationary density at (xs[i], ys[j]), normalized so that
    sum(mu_grid) * hx * hy == 1."""
    xs = np.linspace(-L, L, n)
    ys = np.linspace(-L, L, n)
    hx = xs[1] - xs[0]
    hy = ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.stack([X, Y], axis=-1)
    F = C.drift(pts, cfg)
    f1, f2 = F[..., 0], F[..., 1]

    def idx(i, j):
        return i * n + j

    diffC = omega ** 2 / 2.0
    rows, cols, vals = [], [], []

    # interior physics (first-order upwind advection + central diffusion)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            ii = idx(i, j)
            # diffusion
            rows += [ii, ii, ii, ii, ii]
            cols += [ii, idx(i + 1, j), idx(i - 1, j), idx(i, j + 1), idx(i, j - 1)]
            vals += [-2 * diffC / hx ** 2 - 2 * diffC / hy ** 2,
                      diffC / hx ** 2, diffC / hx ** 2, diffC / hy ** 2, diffC / hy ** 2]

            f1e = 0.5 * (f1[i, j] + f1[i + 1, j])
            f1w = 0.5 * (f1[i, j] + f1[i - 1, j])
            f2n = 0.5 * (f2[i, j] + f2[i, j + 1])
            f2s = 0.5 * (f2[i, j] + f2[i, j - 1])

            if f1e >= 0:
                rows.append(ii); cols.append(ii); vals.append(-f1e / hx)
            else:
                rows.append(ii); cols.append(idx(i + 1, j)); vals.append(-f1e / hx)
            if f1w >= 0:
                rows.append(ii); cols.append(idx(i - 1, j)); vals.append(f1w / hx)
            else:
                rows.append(ii); cols.append(ii); vals.append(f1w / hx)
            if f2n >= 0:
                rows.append(ii); cols.append(ii); vals.append(-f2n / hy)
            else:
                rows.append(ii); cols.append(idx(i, j + 1)); vals.append(-f2n / hy)
            if f2s >= 0:
                rows.append(ii); cols.append(idx(i, j - 1)); vals.append(f2s / hy)
            else:
                rows.append(ii); cols.append(ii); vals.append(f2s / hy)

    # boundary: decoupled trivial equations forcing (near-)zero density
    for i in range(n):
        for j in (0, n - 1):
            ii = idx(i, j)
            rows.append(ii); cols.append(ii); vals.append(-1.0)
    for j in range(n):
        for i in (0, n - 1):
            ii = idx(i, j)
            rows.append(ii); cols.append(ii); vals.append(-1.0)

    N = n * n
    A = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()

    eigval, eigvec = eigs(A, k=1, sigma=0.0, which='LM')
    mu = np.real(eigvec[:, 0]).reshape(n, n)
    mu = np.clip(mu, 0, None)
    mass = mu.sum() * hx * hy
    if mass <= 0:
        mu = -mu
        mass = mu.sum() * hx * hy
    mu = mu / mass
    return xs, ys, mu, complex(eigval[0])


def mass_in_ball(xs, ys, mu_grid, center, r):
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    hx, hy = xs[1] - xs[0], ys[1] - ys[0]
    mask = (X - center[0]) ** 2 + (Y - center[1]) ** 2 <= r ** 2
    return float(mu_grid[mask].sum() * hx * hy)
