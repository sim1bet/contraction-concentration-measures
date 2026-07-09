Contraction and Concentration of Stationary Measures
===================================================

This repository contains the numerical experiments accompanying the paper

    Contraction, stationarity, and concentration in stochastic dynamical systems

by Simone Betteti and Francesco Bullo.

The code provides reproducible validations of contraction, convergence to
stationarity, stationary concentration, and stochastic memory retrieval in
input-driven Hopfield networks.

The repository includes:

- Validation of exponential convergence to stationary measures under
  globally contracting drift fields and spatially inhomogeneous diffusion.
- Sensitivity analysis of stationary measures under diffusion perturbations.
- Local concentration bounds for stationary measures around stable
  equilibria.
- Numerical investigation of saliency-ranked concentration in stochastic
  Hopfield networks.
- Independent cross-validation through a stationary Fokker-Planck solver.

All figures, numerical tables, and summary logs used in the manuscript can
be regenerated directly from the provided scripts.

---------------------------------------------------------------------------
Repository Structure
---------------------------------------------------------------------------

common.py
    Shared numerical infrastructure:
    - Input-Driven Hopfield model
    - Equilibrium computation
    - Exact contraction-rate estimation
    - Euler-Maruyama ensemble simulation
    - Synchronous coupling simulation
    - Wasserstein-distance estimation
    - Bootstrap confidence intervals
    - Figure and data export

run_theorem6_global_contraction.py
    Validation of convergence to stationarity under global contraction.
    Includes:
    - synchronous coupling experiment,
    - theoretical convergence envelope,
    - empirical convergence-rate estimation,
    - independent optimal-transport cross-check.

run_proposition7_diffusion_gap.py
    Validation of the stationary sensitivity bound for systems sharing the
    same drift and differing only in the diffusion field.

run_local_concentration.py
    Validation of local stationary concentration results.
    Includes:
    - exact concentration estimate,
    - conservative concentration estimate,
    - boundary-mass-deficit condition,
    - conditional concentration inequality.

run_conjecture_saliency.py
    Numerical investigation of saliency-ranked concentration in multistable
    Hopfield networks through large-scale noise sweeps.

run_phase_portraits.py
    Generation of phase-portrait figures:
    - energy landscapes,
    - drift streamlines,
    - stable and unstable equilibria,
    - stationary density overlays.

fpk_grid.py
run_fpk_crosscheck.py
    Independent stationary Fokker-Planck solver used to cross-check the
    particle-based stationary distributions.

---------------------------------------------------------------------------
Numerical Methods
---------------------------------------------------------------------------

Stochastic simulations
----------------------

All stochastic differential equations are integrated using the
Euler-Maruyama scheme.

Two diffusion settings are considered:

1. Spatially inhomogeneous diffusion

   G(x) = 0.4 diag(sin(x1), cos(x2))

   used in the globally contracting regime.

2. Isotropic diffusion

   G(x) = omega I

   used in local concentration and memory-retrieval experiments.

Contraction rates
-----------------

Local and global contraction rates are computed numerically from the
symmetrized Jacobian

   c_R = -sup lambda_max((J + J^T)/2)

evaluated on a dense polar grid.

Wasserstein distance
--------------------

Empirical Wasserstein distances are estimated through:

- random subsampling,
- exact optimal assignment (Hungarian algorithm),
- repeated evaluations for variance reduction.

For convergence experiments, the repository additionally employs the
synchronous coupling construction used in the proof of the main theorem.

Fokker-Planck validation
------------------------

Stationary densities are independently approximated by solving the
stationary Fokker-Planck equation

   0 = -div(f mu) + (omega^2/2) Delta mu

using:

- first-order upwind finite-volume discretization,
- central finite differences for diffusion,
- sparse eigenvalue computation of the discretized generator.

---------------------------------------------------------------------------
Reproducing the Experiments
---------------------------------------------------------------------------

Each experiment can be executed independently:

    python run_theorem6_global_contraction.py
    python run_proposition7_diffusion_gap.py
    python run_local_concentration.py
    python run_conjecture_saliency.py
    python run_fpk_crosscheck.py
    python run_phase_portraits.py

Generated outputs are automatically stored under

    results/
        figures/
        csv/
        logs/

---------------------------------------------------------------------------
Dependencies
---------------------------------------------------------------------------

Required Python packages:

    numpy
    scipy
    pandas
    matplotlib
    scikit-learn

Installation:

    pip install -r requirements.txt

---------------------------------------------------------------------------
Citation
---------------------------------------------------------------------------

If you use this code in academic work, please cite:

    Betteti, S., Bullo, F.,
    "Contraction, stationarity, and concentration in stochastic dynamical systems",
    arXiv, 2026.

---------------------------------------------------------------------------
License
---------------------------------------------------------------------------

MIT License.
