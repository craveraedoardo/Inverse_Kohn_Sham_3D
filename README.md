# Kohn-Sham-inversion-3D

**Inverse Kohn–Sham problem in three dimensions, without symmetries.**
Reconstruction of the Kohn–Sham potential and orbitals from a given nuclear density on a 3D Cartesian grid, with the constrained-variational (CV) method solved by the interior-point optimiser IPOPT.

This repository contains the code of the bachelor's thesis

> E. Cravera, *Solving the three-dimensional inverse Kohn–Sham problem without symmetries*,
> Bachelor's Degree in Physics, Università degli Studi di Milano, A.Y. 2025/2026.
> Supervisor: Prof. G. Colò.

Equation, section, figure and table numbers quoted in the code comments and in this file refer to that thesis.

---

## Contents

1. [The method in a nutshell](#the-method-in-a-nutshell)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Repository structure](#repository-structure)
5. [Reproducing the results of the thesis](#reproducing-the-results-of-the-thesis)
6. [Using the solver in your own script](#using-the-solver-in-your-own-script)
7. [Conventions you need to know](#conventions-you-need-to-know)
8. [Known limitations and ideas for future work](#known-limitations-and-ideas-for-future-work)

---

## The method in a nutshell

Given a target density $\rho_t$, the Levy–Lieb constrained search looks for the occupied orbitals of minimum kinetic energy that reproduce it. With the change of variable

$$\varphi_k(\mathbf r) = f_k(\mathbf r)\,\sqrt{\rho_t(\mathbf r)}$$

the unknowns are the nodal values $f_{k,r}$, the objective

$$\Phi[f] = \sum_k q_k \int \Big[\alpha\,\rho_t\,|\nabla f_k + f_k\,\nabla\ln\sqrt{\rho_t}|^2 + \beta\,|\nabla f_k|^2\Big]\,d^3r ,\qquad \alpha=\frac{\hbar^2}{2m}$$

is an **exact quadratic form** (constant Hessian), and the density becomes an **algebraic, pointwise constraint**

$$\sum_k q_k f_{k,r}^2 = 1 \quad\text{on every live node } r,$$

together with orthonormality of the orbitals. The first term is the kinetic energy $T_\alpha$, the second a Tikhonov regulariser $T_\beta$ of weight $\beta$.

The Kohn–Sham potential is **not** an unknown: it is read off the Lagrange multipliers $\lambda_r$ of the density constraints,

$$v_{KS}(\mathbf r_r) = \frac{\lambda_r}{\rho_{t,r}\,W_r}\quad(\text{up to a constant}),$$

where $W_r$ is the quadrature weight of node $r$ (thesis Eq. 4.8).

Two discretisations share the same code and differ only in nodes, weights and derivative matrix (thesis Ch. 3):

| | Uniform grid | Gauss–Hermite Lagrange mesh |
|---|---|---|
| Nodes | equispaced on $[-L,L]^3$ | Gauss–Hermite nodes, $x_i = s\,b_x(A)\sqrt2\,t_i$ |
| Derivatives | 4th-order finite differences (antisymmetrised) | dense Lagrange derivative matrix (Fornberg) |
| Convergence | algebraic, $O(h^4)$ | spectral, in a window of the mesh scale $s$ |
| Selected by | `use_lagrange_mesh=False` | `use_lagrange_mesh=True` |

The test systems are drops of $A = 2, 4, 8$ **neutrons** in a harmonic trap, in the non-interacting limit, so that the exact Kohn–Sham potential is the trap itself and every reconstruction can be checked node by node.

---

## Installation

Python ≥ 3.10 is recommended.

```bash
git clone <repository-url>
cd Kohn-Sham-inversion-3D
pip install -r requirements.txt
```

`cyipopt` needs the IPOPT library. The simplest route is conda:

```bash
conda install -c conda-forge cyipopt
```

With pip on Linux, install IPOPT and the BLAS/LAPACK development files first (Debian/Ubuntu: `sudo apt install coinor-libipopt-dev liblapack-dev libblas-dev pkg-config`), then `pip install cyipopt`. On a cluster, load the IPOPT module provided by the system. See the [IPOPT](https://coin-or.github.io/Ipopt/) and [cyipopt](https://cyipopt.readthedocs.io/) documentation.

The code was last run with numpy 2.5, scipy 1.18, matplotlib 3.11, numexpr 2.14, findiff 0.16, cyipopt 1.7 and IPOPT 3.11.

---

## Quick start

```bash
python test_IKS_3D_2part.py
```

Two neutrons on a $41^3$ Lagrange mesh at the optimal scale $s = 0.794$ (about 1.5 minutes on a laptop). Expected output, among other lines:

```
E_kin (IKS)   : 48.8126 MeV   (rif. 48.8126)
Δρ L2         : 2.408e-13
V RMSE (|x|<3b): 6.234e-04 MeV
```

and the figure `test_3D_data_2part/IKS_2particles_validation.pdf`.

Every script is run from the repository root; it writes its figures into a folder `test_3D_data_*/` (or `*_out/`) and the solver writes `ipopt.out` in the current directory. All these outputs are ignored by git.

> **Note.** The run-time messages and the labels of the figures are in Italian; comments and docstrings are in English.

---

## Repository structure

```
Kohn-Sham-inversion-3D/
├── src/                                   core library
│   ├── consts3D.py                        constants, hbar^2/2m, oscillator lengths and frequencies vs A
│   ├── deformed.py                        axially deformed oscillator at constant volume
│   ├── eigenfunctions3D.py                Cartesian HO eigenfunctions, evaluable on any grid
│   ├── orbitals3D.py                      occupied states (nx, ny, nz) and occupations (aufbau)
│   ├── lagrange_mesh_3D.py                Gauss-Hermite nodes/weights, Fornberg derivative matrix, 3D mesh
│   ├── optimizer3D_lm.py                  class iKS_problem: the whole discrete problem + IPOPT interface
│   ├── multipoles3D.py                    real solid harmonics, moments Q_lm, multipole constraints
│   ├── plot_helpers.py                    error metrics (density-weighted RMSE) and plotting helpers
│   └── utility3D.py                       save/load of the results (pickle)
│
├── test_IKS_3D_2part.py                   validation A = 2, Lagrange mesh
├── test_IKS_3D_2part_uniform.py           validation A = 2, uniform grid
├── test_IKS_3D_4part.py                   A = 4, Lagrange mesh (+ --beta-scan, --Nbeta-scan)
├── test_IKS_3D_4part_uniform.py           A = 4, uniform grid (+ scans)
├── test_IKS_3D_8part.py                   A = 8 closed shell, Lagrange mesh (+ scans)
├── test_IKS_3D_8part_uniform.py           A = 8 closed shell, uniform grid (+ scans)
├── test_IKS_3D_deformed.py                deformed targets, delta = 0.3, 0.6
├── test_IKS_3D_deformed_dual.py           deformed targets + scan of IPOPT's dual_inf_tol
├── test_IKS_3D_deformato_eccitato_gh.py   ground vs excited configuration, oriented start
├── test_IKS_3D_{2,4,8}part_multipoles.py  multipole constraints (redundancy and gauge shift)
├── compare_grid.py                        Lagrange mesh vs uniform grid, A = 2 and 4
├── efficiency_precision_8part.py          accuracy vs N and vs cost, A = 8
├── h_scan_gh_Z2.py                        scan of the mesh scale s, A = 2
└── reffree_estimator.py                   choice of s without the exact potential
```

The `src/` modules form two layers: the **foundations** (`consts3D`, `deformed`, `eigenfunctions3D`, `orbitals3D`) know the physics of the oscillator but nothing about the inversion; the **engine** (`lagrange_mesh_3D`, `optimizer3D_lm`, `multipoles3D`) builds and solves the discrete problem. The scripts in the root are the **applications**: each one sets up a system and produces results. `compare_grid.py`, `efficiency_precision_8part.py` and `h_scan_gh_Z2.py` import the functions of the validation scripts (`run_iks`, `build_target`, …), so those files are also small libraries.

---

## Reproducing the results of the thesis

| Thesis | Script | Command |
|---|---|---|
| §5.1, Tab. 5.1, Fig. 5.1 | `test_IKS_3D_2part.py` | `python test_IKS_3D_2part.py` |
| §5.1, Fig. 5.2 | `test_IKS_3D_2part_uniform.py` | `python test_IKS_3D_2part_uniform.py` |
| §5.1, Fig. 5.3 | `test_IKS_3D_8part.py` | `python test_IKS_3D_8part.py` |
| §5.1, Fig. 5.4 | `test_IKS_3D_8part_uniform.py` | `python test_IKS_3D_8part_uniform.py` ¹ |
| §5.2, Figs. 5.5, 5.6 | `compare_grid.py` | `python compare_grid.py` |
| §5.2, Fig. 5.7 | `efficiency_precision_8part.py` | `python efficiency_precision_8part.py --no-pareto` |
| §5.3, Fig. 5.8 | `test_IKS_3D_4part.py` | `python test_IKS_3D_4part.py --beta-scan` ¹ |
| §5.3, Figs. 5.9, 5.10 | `test_IKS_3D_deformed_dual.py` | `python test_IKS_3D_deformed_dual.py --dual-scan` |
| §5.4, Figs. 5.11, 5.12 | `test_IKS_3D_4part.py`, `test_IKS_3D_4part_uniform.py` | default run |
| §5.4, Tab. 5.4, Fig. 5.13 | `test_IKS_3D_deformed.py` | `python test_IKS_3D_deformed.py` |
| §5.5, Figs. 5.14, 5.15, Tab. 5.5 | `h_scan_gh_Z2.py` | `python h_scan_gh_Z2.py` |
| §5.6, Figs. 5.16, 5.17 | `reffree_estimator.py` | `python reffree_estimator.py` (after `h_scan_gh_Z2.py`) |
| §6.1, Tabs. 6.1, 6.2 | `test_IKS_3D_{2,4,8}part_multipoles.py` | `python test_IKS_3D_4part_multipoles.py [uniform] [keep\|remove]` |
| §6.2, Tabs. 6.3, 6.4 | `test_IKS_3D_deformato_eccitato_gh.py` | `python test_IKS_3D_deformato_eccitato_gh.py orient` (`scan` for the basin of attraction) |

¹ The parameters at the top of some scripts were changed after the corresponding figure was produced; check them before reproducing a figure exactly. Known cases: Fig. 5.4 used $n = 31$ (the default `N_GRID` in `test_IKS_3D_8part_uniform.py` is 21); Fig. 5.8 used $N = 24$ (the β-scan runs at `N_RUN`, currently 50).

Each validation script first evaluates the objective on the exact target as a function of the resolution, **without optimising** (this isolates the discretisation error at no cost), then runs the full inversion and reports $T_\alpha$, $T_\beta$, the density error, the orthonormality and the error on the potential.

**Cost.** The Lagrange-mesh runs at $N \approx 20$–$30$ nodes per axis take from one to a few minutes each; the scans in $N$, $\beta$ or $s$ repeat the full inversion many times and can take hours. Memory is dominated by the KKT factorisation in MUMPS: on the uniform grid $n = 41$ with two orbitals already needs a few GB.

---

## Using the solver in your own script

A minimal example: two neutrons, Lagrange mesh with 20 nodes per axis at the optimal scale.

```python
import sys, numpy as np
sys.path.insert(0, "src")

from consts3D import b_x, b_y, b_z, nuclearNu_x
from orbitals3D import uncoupledHObasis, getOrbitalSet
from lagrange_mesh_3D import LagrangeMesh3D
from eigenfunctions3D import HO3D
from optimizer3D_lm import iKS_problem

A, N, s = 2, 20, 0.794

# 1. target density, tabulated on the SAME nodes the solver will use
mesh = LagrangeMesh3D(N, N, N, s * b_x(A), s * b_y(A), s * b_z(A))
rho = np.zeros((N, N, N))
for o in getOrbitalSet(A, uncoupledHObasis()):
    phi = HO3D(o.nx, o.ny, o.nz, b_x(A), b_y(A), b_z(A), mesh.dV)(mesh.x, mesh.y, mesh.z)
    phi /= np.sqrt(mesh.integrate(phi**2))          # normalise with the mesh weights
    rho += o.occupation * phi**2

# 2. the problem (Z=A, N=0: only the particle count matters)
prob = iKS_problem(Z=A, N=0, lb=[-2]*3, ub=[2]*3, data=[mesh.x, mesh.y, mesh.z, rho],
                   basis=uncoupledHObasis(), reg_parameter=1e-3,
                   rel_tol=1e-8, constr_viol=1e-8, dual_inf_tol=1.0, max_iter=500,
                   use_lagrange_mesh=True, Nx=N, Ny=N, Nz=N, mesh_scale=s,
                   output_folder="my_run")
prob.add_option(b"mu_init", 0.01)
results, info = prob.solve()

# 3. potential from the multipliers of the density constraints (live nodes only)
lam = np.full(prob.n_points, np.nan)
lam[prob.live_nodes] = results["lagrange"][:prob.n_live]
V = lam.reshape(rho.shape) / (rho * mesh.w3D)
ic = np.unravel_index(np.argmax(rho), rho.shape)
V -= V[ic]                                           # gauge: V = 0 at the density maximum
print("T =", results["obj"], "MeV;  V(x) - V(0) along x:", V[:, ic[1], ic[2]][N//2:N//2+4])
```

For a density that is not built from oscillator orbitals, you will normally want a different starting point: `solve()` always calls `getStartingPoint()`, so override it on the instance, e.g. `prob.getStartingPoint = lambda: f0.ravel()` (see `_install_deformed_start` in `test_IKS_3D_deformed.py` and the oriented start in `test_IKS_3D_deformato_eccitato_gh.py`).

---

## Conventions you need to know

* **Units.** Lengths in fm, energies in MeV; $\hbar\omega = 41\,A^{-1/3}$ MeV (`consts3D.py`).
* **Particle number.** All systems are neutrons only. The scripts pass `Z=A, N=0`; with the default `n_type='p'` the code fills `Z` particles. Orbitals are filled in the order of `uncoupledHObasis` with occupation 2.
* **Variables.** `x = f.ravel()` with shape `(n_orbitals, Nx, Ny, Nz)` in C order; the orbitals are $\varphi_k = f_k\sqrt{\rho_t}$ (`results['u']`).
* **Constraints and multipliers.** Order: `[density on live nodes] + [orthonormality pairs] + [multipoles]`. The density multipliers `results['lagrange'][:n_live]` refer to `prob.live_nodes`, **not** to all nodes.
* **Dead nodes.** Nodes with $\rho_t < 10^{-10}\rho_{max}$ (Lagrange mesh) or $< 10^{-6}\rho_{max}$ (uniform grid) are excluded from every integral and their variables are frozen at the plateau value $1/\sqrt{\sum_k q_k}$ (thesis §3.4). Never set them to zero.
* **Potential.** $V = \lambda/(\rho W)$, defined up to a constant; the scripts fix the gauge at the node of maximum density and compare on the region $\rho \ge 10^{-2}\rho_{max}$ with the density-weighted RMSE (`plot_helpers.potential_error`, key `rmse_pesato`).
* **Kinetic energy.** The objective is $T_\alpha + T_\beta$: compare $T_\alpha$ (from `decompose_from_f` in the scripts) with the exact kinetic energy, not the objective.
* **Mesh scale.** `mesh_scale` moves the nodes only; the physical density and the starting orbitals always use $b_x(A)$. The optimum for $A=2$ is $s \simeq 0.79$ (thesis §5.5); the default `1.0` is the historical value.
* **Jacobian and Hessian.** IPOPT receives the sparsity structure once and then flat value vectors with no consistency check: `jacobianstructure`/`jacobian` and `_build_hessian_structure`/`_hessian_vectorized` must be modified together and in the same order.

---

## Known limitations and ideas for future work

**Physics / method (from the conclusions of the thesis)**
* All targets come from harmonic-oscillator potentials. The natural next test is a non-harmonic potential, e.g. a deformed Woods–Saxon well, where the oriented start and the roughness indicator are no longer exact.
* A sequence of deformed densities at increasing quadrupole deformation would give a quadrupole counterpart of the monopole path used by Liardi et al. to reconstruct the energy functional by line integration; then interacting functionals and ab initio densities of neutron drops.
* On the uniform grid the true minimum of the discrete functional at $A = 4$ is spurious (thesis §5.3 and the note in `iKS_problem.setSolverOptions`): on the grid the expanded kinetic term $T_\alpha$ no longer equals $\alpha\int|\nabla\varphi|^2$, because the discrete product rule fails, $D(fs) \ne s\,Df + f\,Ds$, and the optimiser exploits the gap. Writing the kinetic objective directly as $\alpha\int|\nabla(f\sqrt{\rho})|^2$ is a possible structural fix (it changes gradient, Hessian and extraction formula).
* The multipole moments are redundant as constraints when the density is imposed pointwise (thesis §6.1); they remain useful as diagnostics and to orient the starting point.

**Code**
* `iKS_problem` always builds an **isotropic** mesh from $b_x(A)$ even for deformed targets; `LagrangeMesh3D` already supports different `bx, by, bz`, so an anisotropic mesh is a small extension.
* `jacobian()` rebuilds the full sparsity pattern at every call, and the orthonormality rows run over all nodes, dead ones included: both are easy speed-ups for large grids.
* The Lagrange derivative matrix is dense, so memory and the number of non-zeros of the Hessian grow as $N^4$; this, not accuracy, limits the Lagrange mesh at large $N$.
* Hard-coded choices that could become parameters: `numexpr` threads (8, in `optimizer3D_lm.py`), the output root `test_3D_data/`, the log file `ipopt.out`, the mask thresholds.
* The validation scripts share a lot of code (target construction, potential extraction, kinetic decomposition, figures) that could be factored into `src/`.
* `perturb`/`seed` of `run_iks` in `test_IKS_3D_{4,8}part.py` are inert (see the docstring): to perturb the start, override `getStartingPoint`.
* There is no automatic test suite; the validation scripts against the analytic oscillator results play that role.
