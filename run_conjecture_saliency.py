"""
Conjecture 15 (saliency-ranked stationary concentration), Sec. 5: for stable retrievable memories x_mu, x_nu with saliency
alpha_mu > alpha_nu > alpha*, there should exist omega_0 > 0 such that for
every omega in (0, omega_0), mu*(B_r(x_mu)) >= mu*(B_r(x_nu)).

We test this with the concrete saliency proxy u_i (see common.saliency_of)
across a sweep of noise levels omega, using several independent ensemble
repeats per omega level (mean +- std / bootstrap CI, as in the manuscript's
numerical methodology). Both equilibria pairs (+-) are pooled by symmetry.

Outputs:
  results/figures/fig_saliency_conjecture.pdf
  results/csv/saliency_conjecture_mass.csv
  results/csv/saliency_conjecture_ratio.csv
  results/logs/saliency_conjecture_summary.json
"""
import numpy as np
import matplotlib.pyplot as plt

import common as C


def mass_in_ball(samples, center, r):
    return np.mean(np.linalg.norm(samples - center, axis=1) <= r)


def main():
    C.set_style()
    cfg = C.REGIMES["multistable"]

    eqs = [e for e in C.find_equilibria(cfg) if e.stable]
    by_u = {}
    for e in eqs:
        by_u.setdefault(e.saliency, []).append(e.x)
    u_hi, u_lo = sorted(by_u.keys(), reverse=True)[:2]
    centers_hi = by_u[u_hi]   # e.g. u_i = 3 memory, both signs
    centers_lo = by_u[u_lo]   # e.g. u_i = 1 memory, both signs
    print(f"[conjecture] higher-saliency memory u={u_hi}: {centers_hi}")
    print(f"[conjecture] lower-saliency memory  u={u_lo}: {centers_lo}")

    r_ball = 0.4
    omegas = np.linspace(0.05, 0.4, 50)
    n_repeats = 30
    N = 350
    dt = 1e-2
    T_burn, T_rec, rec_every = 60.0, 60.0, 10

    mean_hi, mean_lo = [], []
    ci_hi, ci_lo_ = [], []
    ratio_mean, ratio_lo, ratio_hi = [], [], []

    for om in omegas:
        hi_vals, lo_vals, ratios = [], [], []
        for rep in range(n_repeats):
            rng = C.rng_for(f"conjecture_om{om:.2f}_rep{rep}")
            x0 = np.concatenate([
                np.tile(c, (N // 2, 1)) + 0.1 * rng.standard_normal((N // 2, 2))
                for c in (centers_hi[0], centers_lo[0])
            ], axis=0)
            x_burn = C.simulate_ensemble(cfg, x0, T_burn, dt, rng,
                                          diffusion="isotropic", omega=om)
            traj = C.simulate_ensemble(cfg, x_burn, T_rec, dt, rng, diffusion="isotropic",
                                        omega=om, record_every=rec_every)
            samples = traj.reshape(-1, 2)
            m_hi = np.mean([mass_in_ball(samples, c, r_ball) for c in centers_hi])
            m_lo = np.mean([mass_in_ball(samples, c, r_ball) for c in centers_lo])
            hi_vals.append(m_hi); lo_vals.append(m_lo)
            ratios.append(m_hi / max(m_lo, 1e-8))
        hi_vals, lo_vals, ratios = map(np.array, (hi_vals, lo_vals, ratios))
        mean_hi.append(hi_vals.mean()); mean_lo.append(lo_vals.mean())
        ci_hi.append(hi_vals.std()); ci_lo_.append(lo_vals.std())
        ratio_mean.append(ratios.mean())
        lo_p, hi_p = np.percentile(ratios, [10, 90])
        ratio_lo.append(lo_p); ratio_hi.append(hi_p)
        print(f"[conjecture] omega={om:.2f}  mass(u={u_hi})={hi_vals.mean():.4f}+-{hi_vals.std():.4f}"
              f"   mass(u={u_lo})={lo_vals.mean():.4f}+-{lo_vals.std():.4f}"
              f"   ratio={ratios.mean():.3f}")

    mean_hi, mean_lo, ci_hi, ci_lo_ = map(np.array, (mean_hi, mean_lo, ci_hi, ci_lo_))
    ratio_mean, ratio_lo, ratio_hi = map(np.array, (ratio_mean, ratio_lo, ratio_hi))
    conjecture_holds = ratio_lo > 1.0  # lower 10th percentile still above 1

    C.set_style()

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.errorbar(omegas, mean_hi, yerr=ci_hi, fmt='o-', color=C.PALETTE["memB"],
                capsize=3, label=fr"higher saliency, $u_i={u_hi:.0f}$")
    ax.errorbar(omegas, mean_lo, yerr=ci_lo_, fmt='s-', color=C.PALETTE["memA"],
                capsize=3, label=fr"lower saliency, $u_i={u_lo:.0f}$")
    ax.set_xlabel(r"noise level $\omega$")
    ax.grid(False)
    ax.legend(fontsize=12)
    fig.tight_layout()
    C.save_fig(fig, "fig_saliency_mass")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.semilogy(omegas, ratio_mean, color=C.PALETTE["secondary"])
    ax.fill_between(omegas, ratio_lo, ratio_hi, color=C.PALETTE["secondary"], alpha=0.2,
                     label="10-90% across repeats")
    ax.axhline(1.0, color='k', ls='--', lw=1.2, label="equal mass")
    ax.set_xlabel(r"noise level $\omega$")
    ax.grid(False)
    ax.legend(fontsize=12)
    fig.tight_layout()
    C.save_fig(fig, "fig_saliency_conjecture")
    plt.close(fig)

    C.save_csv("saliency_conjecture_mass", {
        "omega": omegas, "mass_hi": mean_hi, "mass_hi_std": ci_hi,
        "mass_lo": mean_lo, "mass_lo_std": ci_lo_,
    })
    C.save_csv("saliency_conjecture_ratio", {
        "omega": omegas, "ratio_mean": ratio_mean, "ratio_p10": ratio_lo, "ratio_p90": ratio_hi,
    })
    C.log_json("saliency_conjecture_summary", {
        "u_hi": u_hi, "u_lo": u_lo, "r_ball": r_ball, "n_repeats": n_repeats, "N": N,
        "conjecture_holds_per_omega": conjecture_holds.tolist(),
        "omegas": omegas.tolist(),
    })


if __name__ == "__main__":
    main()
