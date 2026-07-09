"""
Theorem 6 (contraction of measures under spatially-inhomogeneous diffusion),
Sec. 3 / Fig. 2 of autosam.tex.

We validate:
  (a) the standing hypothesis c > L_G^2/2 for the globally contracting
      Hopfield example of Sec. 5.2,
  (b) the predicted exponential rate 2c - L_G^2 for W_2^2(mu_t, mu*), using
      the SAME synchronous-coupling construction as the proof: two ensembles
      driven by identical Brownian increments, one started away from
      equilibrium, the other initialized at (an empirical approximation of)
      the stationary law mu*, so that E[||X_t - Z_t||^2] is *exactly* the
      quantity bounded in the proof (coupling inequality \\eqref{eq: coupling})
      and stays a valid upper bound for W_2^2(mu_t, mu*) for every t,
  (c) a cross-check against a genuine (subsampled, optimal-assignment)
      empirical W_2 estimate at a handful of checkpoints.

Outputs:
  results/figures/fig_theorem6_w2_decay.pdf
  results/csv/theorem6_w2_decay.csv          (coupling-based curve + fit)
  results/csv/theorem6_w2_crosscheck.csv     (independent OT cross-check)
  results/logs/theorem6_summary.json
"""
import numpy as np
import matplotlib.pyplot as plt

import common as C


def main():
    C.set_style()
    cfg = C.REGIMES["global"]
    rng = C.rng_for("theorem6")

    eqs = C.find_equilibria(cfg)
    x_star = eqs[0].x
    assert np.linalg.norm(x_star) < 1e-6

    c = C.global_contraction_rate(cfg, box=3.0)
    L_G = C.L_G_INHOMOGENEOUS
    predicted_rate = 2 * c - L_G ** 2
    hypothesis_ok = c > L_G ** 2 / 2

    print(f"[theorem6] c = {c:.4f}, L_G = {L_G:.4f}, L_G^2/2 = {L_G**2/2:.4f}, "
          f"hypothesis c > L_G^2/2: {hypothesis_ok}")
    print(f"[theorem6] predicted rate 2c - L_G^2 = {predicted_rate:.4f}")

    dt = 1e-2
    diag_fn = C.diagonal_diffusion_inhomogeneous

    # --- (1) build a large reference stationary sample -------------------
    N_ref = 3000
    x0_ref = x_star + 0.2 * rng.standard_normal((N_ref, 2))
    T_burn, T_rec, rec_every = 80.0, 60.0, 20
    x_ref = C.simulate_ensemble(cfg, x0_ref, T_burn, dt, rng,
                                 diffusion="diagonal", diag_fn=diag_fn)
    ref_traj = C.simulate_ensemble(cfg, x_ref, T_rec, dt, rng,
                                    diffusion="diagonal", diag_fn=diag_fn,
                                    record_every=rec_every)
    ref_cloud = ref_traj.reshape(-1, 2)  # pooled stationary sample, ~N_ref*T_rec/(rec_every*dt) points
    print(f"[theorem6] reference stationary cloud: {ref_cloud.shape[0]} points, "
          f"mean={ref_cloud.mean(0)}, std={ref_cloud.std(0)}")

    # --- (2) synchronous coupling: X0 far from equilibrium, Z0 ~ mu* -----
    N_pairs = 800
    far_point = x_star + np.array([2.5, 2.2])
    x0 = np.tile(far_point, (N_pairs, 1)) + 0.05 * rng.standard_normal((N_pairs, 2))
    z0 = ref_cloud[rng.integers(0, len(ref_cloud), size=N_pairs)]

    T_track, rec_every_track = 12.0, 2
    Xs, Zs = C.simulate_ensemble_coupled(cfg, x0, z0, T_track, dt, rng,
                                          diffusion="diagonal", diag_fn=diag_fn,
                                          record_every=rec_every_track)
    times = np.arange(1, Xs.shape[0] + 1) * rec_every_track * dt
    h_t = np.mean(np.sum((Xs - Zs) ** 2, axis=-1), axis=1)  # E[||X_t-Z_t||^2]
    h0 = np.mean(np.sum((x0 - z0) ** 2, axis=-1))

    # log-linear fit on the decaying part (skip a short initial transient,
    # stop before the finite-N noise floor dominates)
    floor = np.median(h_t[-20:])
    mask = (times > 0.3) & (h_t > max(3 * floor, 1e-6))
    slope, intercept = np.polyfit(times[mask], np.log(h_t[mask]), 1)
    print(f"[theorem6] fitted empirical decay rate = {-slope:.4f} "
          f"(predicted 2c-L_G^2 = {predicted_rate:.4f}); floor~{floor:.2e}")

    # --- (3) independent cross-check via subsampled empirical W2 ---------
    checkpoints_idx = np.unique(np.linspace(0, Xs.shape[0] - 1, 6).astype(int))
    w2_check_t, w2_check_val, w2_check_std = [], [], []
    for k in checkpoints_idx:
        w2sq, w2std = C.empirical_w2_squared(Xs[k], ref_cloud, rng, n_sub=200, n_reps=6)
        w2_check_t.append(times[k])
        w2_check_val.append(w2sq)
        w2_check_std.append(w2std)
        print(f"[theorem6]   t={times[k]:.2f}  empirical W2^2 (OT) = {w2sq:.4f} +- {w2std:.4f}"
              f"   coupling h(t) = {h_t[k]:.4f}")

    # --- plotting ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.semilogy(times, h_t, 'o', ms=4, color=C.PALETTE["empirical"], alpha=0.75)
    ax.text(0.4,0.01,r"$\mathbb{E}\|X_t-Z_t\|^2$",color=C.PALETTE["empirical"], fontsize=16)
    fit_line = np.exp(intercept) * np.exp(slope * times)
    theory_line = h_t[0] * np.exp(-predicted_rate * (times - times[0]))
    ax.semilogy(times, theory_line, '--', color=C.PALETTE["theory"], lw=2.2, label=fr"$2c-L_G^2={predicted_rate:.2f}$")
    ax.semilogy(times, fit_line, '-', color=C.PALETTE["secondary"], lw=2.0,
                label=fr"Least-squares fit, rate $= {-slope:.3f}$")
    
    ax.text(6,0.1, r"$\mathrm{e}^{-t(2 c-L_G^2)}$", color=C.PALETTE["theory"], fontsize=16)
    ax.errorbar(w2_check_t, w2_check_val, yerr=w2_check_std, fmt='s', ms=6,
                color=C.PALETTE["conservative"], capsize=3,
                label=r"Empirical $W_2^2$ (suboptimal assignment)")
    ax.set_xlabel(r"time $t$")
    #ax.set_ylabel(r"$\log$-scale squared distance")
    #ax.set_title("Theorem 6: exponential contraction to the stationary measure")
    ax.legend(loc='lower left', fontsize=12)
    ax.grid(False)
    C.save_fig(fig, "fig_theorem6_w2_decay")
    plt.close(fig)

    C.save_csv("theorem6_w2_decay", {
        "t": times, "coupling_h_t": h_t,
        "fit_line": fit_line, "theory_line": theory_line,
    })
    C.save_csv("theorem6_w2_crosscheck", {
        "t": w2_check_t, "w2_squared": w2_check_val, "w2_std": w2_check_std,
    })
    C.log_json("theorem6_summary", {
        "c_global": c, "L_G": L_G, "hypothesis_c_gt_LG2_over_2": hypothesis_ok,
        "predicted_rate": predicted_rate, "fitted_rate": -slope,
        "N_pairs": N_pairs, "N_ref": N_ref, "dt": dt,
    })


if __name__ == "__main__":
    main()
