"""
Proposition 7 (contraction for different diffusions), Sec. 3:
bounds W_2^2(mu*, nu*) between the stationary measures of two processes that
share the same (globally contracting) drift but differ in their diffusion
terms G, Q, via

    W_2^2(mu*, nu*) <= (1+eps^-1) chi^2 / (2c - (1+eps) L^2),   L^2=min(L_G^2,L_Q^2)

for every eps>0 with c > (1+eps)L^2/2, where chi^2 = sup||G-Q||_F^2.

We use the same globally contracting Hopfield drift as Sec. 5.2, with:
  G(x) = 0.4 diag(sin x1, cos x2)   (the spatially-inhomogeneous diffusion of
                                      Fig. 2)
  Q(x) = 0.4 diag(cos x1, sin x2)   (a second, comparably-scaled diffusion
                                      with the same exact Lipschitz constant
                                      L_Q = 0.4, giving the closed-form
                                      chi^2 = sup||G-Q||_F^2 = 0.64)
and estimate the stationary measures of both processes independently, then
compare the empirical (subsampled, optimal-assignment) W_2^2 distance to the
bound minimized over eps.

Outputs:
  results/figures/fig_proposition7_diffusion_gap.pdf
  results/csv/proposition7_bound_vs_eps.csv
  results/logs/proposition7_summary.json
"""
import numpy as np
import matplotlib.pyplot as plt

import common as C


def diffusion_G(x, scale=0.4):
    return np.stack([scale * np.sin(x[..., 0]), scale * np.cos(x[..., 1])], axis=-1)


def diffusion_Q(x, scale=0.4):
    return np.stack([scale * np.cos(x[..., 0]), scale * np.sin(x[..., 1])], axis=-1)


def main():
    C.set_style()
    cfg = C.REGIMES["global"]
    rng = C.rng_for("proposition7")

    eqs = C.find_equilibria(cfg)
    x_star = eqs[0].x
    c = C.global_contraction_rate(cfg, box=3.0)
    L_G = L_Q = 0.4
    L2 = min(L_G ** 2, L_Q ** 2)
    chi2 = 4 * (0.4 ** 2)  # sup||G-Q||_F^2 = scale^2 * [(sinx-cosx)^2_max + (cosy-siny)^2_max] = scale^2*(2+2)
    print(f"[prop7] c={c:.4f}, L_G=L_Q={L_G}, L^2={L2:.4f}, chi^2={chi2:.4f}")

    eps_grid = np.linspace(0.02, 5.0, 400)
    valid = c > (1 + eps_grid) * L2 / 2
    bound = np.full_like(eps_grid, np.nan)
    bound[valid] = (1 + 1 / eps_grid[valid]) * chi2 / (2 * c - (1 + eps_grid[valid]) * L2)
    best_idx = np.nanargmin(bound)
    eps_star, bound_star = eps_grid[best_idx], bound[best_idx]
    print(f"[prop7] best eps*={eps_star:.3f}, tightest bound = {bound_star:.4f}")

    dt = 1e-2
    N = 2500
    T_burn, T_rec, rec_every = 80.0, 60.0, 20
    x0 = x_star + 0.2 * rng.standard_normal((N, 2))

    xg_burn = C.simulate_ensemble(cfg, x0, T_burn, dt, rng, diffusion="diagonal", diag_fn=diffusion_G)
    traj_g = C.simulate_ensemble(cfg, xg_burn, T_rec, dt, rng, diffusion="diagonal",
                                  diag_fn=diffusion_G, record_every=rec_every)
    cloud_g = traj_g.reshape(-1, 2)

    xq_burn = C.simulate_ensemble(cfg, x0, T_burn, dt, rng, diffusion="diagonal", diag_fn=diffusion_Q)
    traj_q = C.simulate_ensemble(cfg, xq_burn, T_rec, dt, rng, diffusion="diagonal",
                                  diag_fn=diffusion_Q, record_every=rec_every)
    cloud_q = traj_q.reshape(-1, 2)

    print(f"[prop7] cloud_G: mean={cloud_g.mean(0)}, std={cloud_g.std(0)}")
    print(f"[prop7] cloud_Q: mean={cloud_q.mean(0)}, std={cloud_q.std(0)}")

    w2sq, w2std = C.empirical_w2_squared(cloud_g, cloud_q, rng, n_sub=300, n_reps=12)
    print(f"[prop7] empirical W2^2(mu*,nu*) = {w2sq:.4f} +- {w2std:.4f}  vs bound {bound_star:.4f}")
    satisfied = w2sq <= bound_star


    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    #ax = axes[0]
    ax.plot(eps_grid[valid], bound[valid], color=C.PALETTE["theory"])
    ax.axvline(eps_star, ymax=0.2, color=C.PALETTE["conservative"], ls=':')
    ax.axhline(w2sq, color=C.PALETTE["empirical"], ls='--',
               label=fr"empirical $W_2^2(\mu^\star,\nu^\star)={w2sq:.3f}$")
    ax.text(0.75,9, fr"minimizing $\varepsilon^\star={eps_star:.2f}$", color=C.PALETTE["conservative"], fontsize=16)
    ax.text(4,22, r"$\frac{(1+\varepsilon^{-1})\chi^{2}}{2c-(1+\varepsilon)L^{2}}$", color=C.PALETTE["theory"], fontsize=16)
    ax.fill_between(eps_grid[valid], w2sq - w2std, w2sq + w2std, color=C.PALETTE["empirical"], alpha=0.15)
    ax.set_xlabel(r"$\varepsilon$"); #ax.set_ylabel(r"$\tfrac{(1+\varepsilon^{-1})\chi^{2}}{2c-(1+\varepsilon)L^{2}}$")
    #ax.set_title(r"Tightness of the $\varepsilon$-bound")
    ax.legend(fontsize=13)
    ax.grid(False)

    # ax = axes[1]
    # ax.scatter(cloud_g[:4000, 0], cloud_g[:4000, 1], s=2, alpha=0.25,
    #            color=C.PALETTE["memA"])
    # ax.scatter(cloud_q[:4000, 0], cloud_q[:4000, 1], s=2, alpha=0.25,
    #            color=C.PALETTE["memB"])
    # ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    # ax.text(0.1,0.75, r"$G$-stationary cloud", color=C.PALETTE["memA"])
    # ax.text(-0.9,-0.25, r"$Q$-stationary cloud", color=C.PALETTE["memB"])
    # #ax.set_title("Stationary clouds under $G$ vs. $Q$")
    # #ax.legend(fontsize=9, markerscale=4)
    fig.tight_layout()
    C.save_fig(fig, "fig_proposition7_diffusion_gap")
    plt.close(fig)

    C.save_csv("proposition7_bound_vs_eps", {
        "eps": eps_grid, "bound": bound,
    })
    C.log_json("proposition7_summary", {
        "c": c, "L2": L2, "chi2": chi2, "eps_star": eps_star, "bound_star": bound_star,
        "empirical_w2_squared": w2sq, "empirical_w2_squared_std": w2std,
        "bound_satisfied": bool(satisfied),
    })


if __name__ == "__main__":
    main()
