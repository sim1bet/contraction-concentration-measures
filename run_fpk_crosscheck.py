"""
Independent cross-check: grid-based stationary Fokker-Planck solution (see
fpk_grid.py) versus the particle/KDE stationary estimate, for both regimes,
under the same constant isotropic diffusion omega*I used in Sec. 4-5
(Theorem 10 / Conjecture 15 setting). Manuscript numerical methodology,
item (iv).

Outputs:
  results/figures/fig_fpk_crosscheck_global.pdf
  results/figures/fig_fpk_crosscheck_multistable.pdf
  results/csv/fpk_grid_density_global.csv        (flattened grid, for pgfplots)
  results/csv/fpk_grid_density_multistable.csv
  results/csv/fpk_crosscheck_mass_table.csv       (per-well mass: grid vs particles)
  results/logs/fpk_crosscheck_summary.json
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import common as C
import fpk_grid as FPK


def run_regime(cfg_name, L, n, omega, r_ball, N=1500, dt=1e-2,
               T_burn=100.0, T_rec=100.0, rec_every=10):
    C.set_style()
    cfg = C.REGIMES[cfg_name]
    rng = C.rng_for(f"fpk_{cfg_name}")
    eqs = [e for e in C.find_equilibria(cfg) if e.stable]

    print(f"[fpk:{cfg_name}] solving grid FPK on {n}x{n}, L={L}, omega={omega} ...")
    xs, ys, mu_grid, eigval = FPK.solve_stationary_fpk(cfg, omega, L, n)
    mu_grid[mu_grid<1e-6] = 1e-6  # avoid log(0) in contour plots
    mu_grid = mu_grid/np.sum(mu_grid)
    print(f"[fpk:{cfg_name}] near-zero eigenvalue used: {eigval:.3e}")

    x0 = np.concatenate([e.x + 0.15 * rng.standard_normal((N // len(eqs), 2)) for e in eqs], axis=0)
    x_burn = C.simulate_ensemble(cfg, x0, T_burn, dt, rng, diffusion="isotropic", omega=omega)
    traj = C.simulate_ensemble(cfg, x_burn, T_rec, dt, rng, diffusion="isotropic",
                                omega=omega, record_every=rec_every)
    samples = traj.reshape(-1, 2)

    rows = []
    for e in eqs:
        m_grid = FPK.mass_in_ball(xs, ys, mu_grid, e.x, r_ball)
        m_part = float(np.mean(np.linalg.norm(samples - e.x, axis=1) <= r_ball))
        rows.append(dict(regime=cfg_name, x1=e.x[0], x2=e.x[1], energy=e.energy,
                          mass_grid=m_grid, mass_particles=m_part,
                          rel_diff=abs(m_grid - m_part) / max(m_part, 1e-6)))
        print(f"[fpk:{cfg_name}]  well {np.round(e.x,2)}: grid mass={m_grid:.4f}  "
              f"particle mass={m_part:.4f}")

    #fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    #ax = axes[0]
    cf = ax.contourf(xs, ys, mu_grid.T, levels=70, cmap="viridis", norm=matplotlib.colors.LogNorm(), alpha=0.95)
    fig.colorbar(cf, ax=ax)
    #ax.set_title("Grid stationary FPK solution")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.grid(False)

    # ax = axes[1]
    # ax.hist2d(samples[:, 0], samples[:, 1], bins=80, range=[[-L, L], [-L, L]],
    #           cmap="viridis", density=True)
    #ax.contour(xs, ys, mu_grid.T, levels=30, colors='white', linewidths=0.8, alpha=0.85)
    #ax.set_title("Particle histogram + grid-FPK contours")
    # ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    # ax.grid(False)
    #fig.suptitle(f"Stationary measure cross-check ({cfg_name} regime)")
    fig.tight_layout()
    C.save_fig(fig, f"fig_fpk_crosscheck_{cfg_name}")
    plt.close(fig)

    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    C.save_csv(f"fpk_grid_density_{cfg_name}", {
        "x1": XX.ravel(), "x2": YY.ravel(), "mu": mu_grid.ravel(),
    })
    return rows


def main():
    rows = []
    rows += run_regime("global", L=3.0, n=121, omega=0.5, r_ball=0.6)
    rows += run_regime("multistable", L=4.5, n=141, omega=0.5, r_ball=0.6)

    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    C.save_csv("fpk_crosscheck_mass_table", {c: df[c].to_numpy() for c in df.columns})
    C.log_json("fpk_crosscheck_summary", {"rows": rows})


if __name__ == "__main__":
    main()
