"""
Publication-quality phase-portrait figures for both regimes (nicer,
consistently-styled replacements/companions for Fig. 2-3):
energy landscape + drift streamlines + equilibria, overlaid with the
stationary measure (KDE contours) obtained from an ensemble simulation.

Outputs:
  results/figures/fig_phase_portrait_global.pdf
  results/figures/fig_phase_portrait_multistable.pdf
  results/csv/energy_grid_global.csv         (for tikz/pgfplots contour plots)
  results/csv/energy_grid_multistable.csv
  results/csv/equilibria_global.csv
  results/csv/equilibria_multistable.csv
"""
import numpy as np
import matplotlib.pyplot as plt

import common as C

try:
    from sklearn.neighbors import KernelDensity
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


def run_regime(cfg_name, L, n_grid=140, omega=0.4, N=1500, dt=1e-2,
               T_burn=100.0, T_rec=100.0, rec_every=10):
    C.set_style()
    cfg = C.REGIMES[cfg_name]
    rng = C.rng_for(f"phase_portrait_{cfg_name}")
    eqs = C.find_equilibria(cfg)
    stable_eqs = [e for e in eqs if e.stable]

    xs = np.linspace(-L, L, n_grid)
    ys = np.linspace(-L, L, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    pts = np.stack([X, Y], axis=-1)
    E = C.energy(pts, cfg)
    F = C.drift(pts, cfg)

    x0 = np.concatenate([e.x + 0.15 * rng.standard_normal((N // len(stable_eqs), 2))
                          for e in stable_eqs], axis=0)
    x_burn = C.simulate_ensemble(cfg, x0, T_burn, dt, rng, diffusion="isotropic", omega=omega)
    traj = C.simulate_ensemble(cfg, x_burn, T_rec, dt, rng, diffusion="isotropic",
                                omega=omega, record_every=rec_every)
    samples = traj.reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    cf = ax.contourf(X, Y, E, levels=40, cmap="magma", alpha=0.95)
    fig.colorbar(cf, ax=ax)
    ax.streamplot(X, Y, F[..., 0], F[..., 1], color="white", density=0.9,
                  linewidth=0.7, arrowsize=0.8)

    if _HAVE_SKLEARN:
        sub = samples[rng.choice(len(samples), size=min(20000, len(samples)), replace=False)]
        kde = KernelDensity(bandwidth=0.15).fit(sub)
        dens = np.exp(kde.score_samples(pts.reshape(-1, 2))).reshape(X.shape)
        levels = np.linspace(dens.max() * 0.05, dens.max() * 0.95, 6)
        ax.contour(X, Y, dens, levels=levels, colors="cyan", linewidths=1.3)

    for e in eqs:
        marker = 'o' if e.stable else 'x'
        color = 'lime' if e.stable else 'red'
        ax.plot(e.x[0], e.x[1], marker, color=color, ms=9, mew=2,
                markeredgecolor='k' if e.stable else color)

    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.grid(False)
    #ax.set_title(f"Energy landscape, drift streamlines, and $\\mu^\\star$ ({cfg_name} regime)")
    C.save_fig(fig, f"fig_phase_portrait_{cfg_name}")
    plt.close(fig)

    C.save_csv(f"energy_grid_{cfg_name}", {"x1": X.ravel(), "x2": Y.ravel(), "E": E.ravel()})
    C.save_csv(f"equilibria_{cfg_name}", {
        "x1": [e.x[0] for e in eqs], "x2": [e.x[1] for e in eqs],
        "energy": [e.energy for e in eqs], "stable": [e.stable for e in eqs],
        "u_i": [e.saliency for e in eqs],
    })


def main():
    run_regime("global", L=2.5)
    run_regime("multistable", L=4.5)


if __name__ == "__main__":
    main()
