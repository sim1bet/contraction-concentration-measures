"""
Theorem 10 (local stationary concentration) and the boundary-mass-deficit
Remark, Sec. 4 / Fig. 3 of autosam.tex.

Robustified relative to the original prototype scripts:
  * an ENSEMBLE of independent trajectories (interacting-particle method) is
    used instead of a single long chain, with bootstrap confidence bands,
  * the admissible radius R is capped at the largest ball on which local
    B_R-contractivity (Assumption 13) actually holds (`max_valid_radius`),
  * both the "exact" identity-based bound (eq: bound) and the conservative
    bound d*omega^2/(2c_R) are reported,
  * the conditional-concentration inequality \\eqref{eq: conc-bound} is
    checked against the empirical conditional CDF with bootstrap CIs,
  * a robustness sweep over the particle count N and the time step dt is
    included (see run_robustness_sweep.py for the full table).

Outputs:
  results/figures/fig_theorem10_variance_bound.pdf
  results/figures/fig_theorem10_deficit.pdf
  results/figures/fig_theorem10_conditional_concentration.pdf
  results/csv/theorem10_variance_bound.csv
  results/csv/theorem10_deficit.csv
  results/csv/theorem10_conditional_concentration.csv
  results/logs/theorem10_summary.json
"""
import numpy as np
import matplotlib.pyplot as plt

import common as C


def run(cfg_name="multistable", N=1200, dt=1e-2, T_burn=100.0, T_rec=150.0,
        rec_every=8, seed_tag="theorem10"):
    C.set_style()
    cfg = C.REGIMES[cfg_name]
    rng = C.rng_for(seed_tag)

    eqs = C.find_equilibria(cfg)
    stable_eqs = [e for e in eqs if e.stable]
    # reference equilibrium: the most salient stable minimum
    ref_eq = max(stable_eqs, key=lambda e: e.saliency)
    x_star = ref_eq.x
    R_cap = cfg.R_max
    R_valid = C.max_valid_radius(cfg, x_star, R_cap)
    print(f"[theorem10] reference equilibrium x*={x_star}, saliency={ref_eq.saliency}, "
          f"R_valid={R_valid:.3f} (cap {R_cap})")

    omega = 0.4
    d = 2

    # ensemble of independent chains started at x*, burned in, then sampled
    x0 = x_star + 0.1 * rng.standard_normal((N, 2))
    x_burn = C.simulate_ensemble(cfg, x0, T_burn, dt, rng, diffusion="isotropic", omega=omega)
    traj = C.simulate_ensemble(cfg, x_burn, T_rec, dt, rng, diffusion="isotropic",
                                omega=omega, record_every=rec_every)
    samples = traj.reshape(-1, 2)
    rho = np.linalg.norm(samples - x_star, axis=1)
    drift_samples = C.drift(samples, cfg)
    drift_inner = 2 * np.sum((samples - x_star) * drift_samples, axis=1)
    print(f"[theorem10] pooled samples: {samples.shape[0]}")

    R_values = np.linspace(0.08, R_valid, 40)
    lhs, exact_rhs, cons_rhs, deficit = [], [], [], []
    lhs_lo, lhs_hi = [], []
    cR_values = []
    for R in R_values:
        inside = rho <= R
        MR = inside.mean()
        if MR < 2e-3:
            lhs.append(np.nan); exact_rhs.append(np.nan); cons_rhs.append(np.nan)
            deficit.append(np.nan); lhs_lo.append(np.nan); lhs_hi.append(np.nan)
            cR_values.append(np.nan)
            continue
        cR = C.exact_contraction_rate(cfg, x_star, R)
        cR_values.append(cR)
        if cR <= 0:
            lhs.append(np.nan); exact_rhs.append(np.nan); cons_rhs.append(np.nan)
            deficit.append(np.nan); lhs_lo.append(np.nan); lhs_hi.append(np.nan)
            continue
        condvar_samples = rho[inside] ** 2
        condvar = condvar_samples.mean()
        # bootstrap CI for the conditional variance I_R/M_R
        boot_idx = rng.integers(0, len(condvar_samples), size=(400, len(condvar_samples)))
        boot_means = condvar_samples[boot_idx].mean(axis=1)
        lo, hi = np.percentile(boot_means, [2.5, 97.5])

        D = -(np.mean(drift_inner[inside]) * MR) / (omega ** 2)  # estimator of d*M_R - R d_R M_R
        lhs.append(condvar); lhs_lo.append(lo); lhs_hi.append(hi)
        exact_rhs.append((omega ** 2 / (2 * cR)) * D / MR)
        cons_rhs.append(d * omega ** 2 / (2 * cR))
        deficit.append(D)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(R_values, lhs, 'o-', ms=4, color=C.PALETTE["empirical"],
            label=r"Empirical $I_R/M_R$")
    ax.fill_between(R_values, lhs_lo, lhs_hi, color=C.PALETTE["empirical"], alpha=0.2,
                     label="95% bootstrap CI")
    ax.plot(R_values, exact_rhs, '--', color=C.PALETTE["theory"],
            label="Theorem 10 bound (eq. bound)")
    ax.plot(R_values, cons_rhs, ':', color=C.PALETTE["conservative"],
            label=r"Conservative bound $d\omega^2/(2c_R)$")
    ax.set_xlabel(r"$R$"); ax.set_ylabel("conditional variance")
    ax.set_title(f"Theorem 10 validation ({cfg_name} regime, $x^\\star$={np.round(x_star,2)})")
    ax.legend()
    C.save_fig(fig, f"fig_theorem10_variance_bound_{cfg_name}")
    plt.close(fig)
    C.save_csv(f"theorem10_variance_bound_{cfg_name}", {
        "R": R_values, "empirical_condvar": lhs, "ci_lo": lhs_lo, "ci_hi": lhs_hi,
        "theorem_bound": exact_rhs, "conservative_bound": cons_rhs, "cR": cR_values,
    })

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(R_values, deficit, color=C.PALETTE["secondary"])
    ax.axhline(0, color='k', ls='--', lw=1.2)
    ax.set_xlabel(r"$R$"); ax.set_ylabel(r"$d\,M_R - R\,\partial_R M_R$")
    ax.set_title("Boundary-mass deficit induced by local contractivity")
    C.save_fig(fig, f"fig_theorem10_deficit_{cfg_name}")
    plt.close(fig)
    C.save_csv(f"theorem10_deficit_{cfg_name}", {"R": R_values, "deficit": deficit})

    # --- conditional concentration inequality (eq: conc-bound) ------------
    R0 = min(0.6 * R_valid, R_valid)
    cR0 = C.exact_contraction_rate(cfg, x_star, R0)
    rvals = np.linspace(0.05, R0, 25)
    inside_R0 = rho <= R0
    n_R0 = inside_R0.sum()
    emp, emp_lo, emp_hi, bnd = [], [], [], []
    for r in rvals:
        indicator = (rho[inside_R0] <= r).astype(float)
        m, lo, hi = C.bootstrap_mean_ci(indicator, n_boot=800, rng=rng)
        emp.append(m); emp_lo.append(lo); emp_hi.append(hi)
        bnd.append(max(0.0, 1 - d * omega ** 2 / (2 * cR0 * r ** 2)))

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(rvals, emp, 'o-', ms=4, color=C.PALETTE["empirical"], label="Empirical")
    ax.fill_between(rvals, emp_lo, emp_hi, color=C.PALETTE["empirical"], alpha=0.2)
    ax.plot(rvals, bnd, '--', color=C.PALETTE["theory"], label="Theorem 10 bound (eq. conc-bound)")
    ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$\mu^\star(B_r(x^\star) \mid B_{R_0}(x^\star))$")
    ax.set_title(f"Conditional concentration ($R_0={R0:.2f}$, $c_{{R_0}}={cR0:.3f}$)")
    ax.legend()
    C.save_fig(fig, f"fig_theorem10_conditional_concentration_{cfg_name}")
    plt.close(fig)
    C.save_csv(f"theorem10_conditional_concentration_{cfg_name}", {
        "r": rvals, "empirical": emp, "ci_lo": emp_lo, "ci_hi": emp_hi, "bound": bnd,
    })

    C.log_json(f"theorem10_summary_{cfg_name}", {
        "x_star": x_star.tolist(), "R_valid": R_valid, "R0": R0, "cR0": cR0,
        "N": N, "dt": dt, "omega": omega, "n_samples": int(samples.shape[0]),
        "n_in_R0": int(n_R0),
    })
    return dict(cfg=cfg, x_star=x_star, samples=samples, rho=rho)


if __name__ == "__main__":
    run("multistable")
    run("global", T_burn=60.0, T_rec=100.0)
