"""
test_IKS_3D_8part.py
====================
IKS test for A = 8 (closed p shell, four orbitals) on the Gauss-Hermite
Lagrange mesh. Thesis: Sections 5.1-5.2 (Fig. 5.3, Fig. 5.7 via
``efficiency_precision_8part.py``).

Why A = 8 is the natural second-level validation case
-----------------------------------------------------
A = 8 fills the N = 0 shell plus the WHOLE N = 1 shell of the 3D oscillator:
    (000) occ = 2  +  (100), (010), (001) occ = 2 each  =  8 neutrons.
Filling all three p states makes the density EXACTLY SPHERICAL, unlike A = 4
where the N = 1 shell is open and the density is anisotropic. The Kohn-Sham
potential of this density is therefore rigorously the ISOTROPIC oscillator
m omega^2 r^2 / 2 — exactly as for A = 2, but with four orbitals and a
genuine optimisation.

Differences with respect to A = 2
---------------------------------
  * the density constraint sum_k occ_k f_k^2 = 1 does NOT fix the individual
    f_k (it only couples them): there is a genuine optimisation;
  * the ORTHONORMALITY constraints int rho f_i f_j w = delta_ij appear for the
    10 pairs i >= j;
  * (100), (010), (001) each have a node along x, y, z respectively, but
    their SUM in the density restores spherical symmetry.

What the objective measures
---------------------------
T_obj = T_alpha + T_beta, with T_alpha = sum_k occ_k int alpha |grad phi_k|^2
the TRUE Kohn-Sham kinetic energy and T_beta = sum_k occ_k int beta |grad f_k|^2
the Tikhonov regulariser. For A = 8, f = phi/sqrt(rho) is not constant, so
T_beta != 0: the physical comparison is made on T_alpha.

Energy reference
----------------
T = sum_k occ_k (n_k + 3/2)(hbar*omega/2) for (000)+(100)+(010)+(001):
    T = 2 (3/2)(hbar*omega/2) + 6 (5/2)(hbar*omega/2) = 9 hbar*omega ~= 184.5 MeV.

Potential reference
-------------------
The density is spherical and the true V_KS IS the isotropic oscillator, so
RMSE(V_IKS - V_HO) on the physical region is a rigorous validation metric.

Usage
-----
    python test_IKS_3D_8part.py                 # convergence scan + full run
    python test_IKS_3D_8part.py --beta-scan     # + scan in beta
    python test_IKS_3D_8part.py --Nbeta-scan    # + 2D scan in (N, beta)

Output: test_3D_data_8part/IKS_8particles_validation.pdf

This module is also imported by ``efficiency_precision_8part.py``.
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Typographic style of the figures, designed for the THESIS: font sizes are
# given in points as they will appear on the page when the figure is
# included at full text width (\includegraphics[width=\textwidth]).
# Matplotlib works in real inches, so every size is multiplied by the
# reduction factor K = figure width / text width.
# For a different page layout just change TEXTWIDTH_CM.
# --------------------------------------------------------------------------
TEXTWIDTH_CM = 15.0                        # text width of the thesis [cm]
FIG_W, FIG_H = 10.0, 14.4           # figure size (3x2 grid) [in]
_K = FIG_W / (TEXTWIDTH_CM / 2.54)         # reduction factor in print

PT_TESTO   = 9.0    # generic text                   [pt on the page]
PT_TICK    = 8.5    # tick labels
PT_LEGENDA = 8.5    # legends
PT_ASSI    = 10.0   # axis and colour-bar labels
PT_TITOLO  = 10.5   # panel titles
PT_SUPTIT  = 12.0   # figure title

plt.rcParams.update({
    "font.size":             PT_TESTO   * _K,
    "axes.labelsize":        PT_ASSI    * _K,
    "axes.titlesize":        PT_TITOLO  * _K,
    "figure.titlesize":      PT_SUPTIT  * _K,
    "xtick.labelsize":       PT_TICK    * _K,
    "ytick.labelsize":       PT_TICK    * _K,
    "legend.fontsize":       PT_LEGENDA * _K,
    "legend.title_fontsize": PT_LEGENDA * _K,
    "axes.labelpad":      3.0 * _K,
    "axes.titlepad":      4.0 * _K,
    "axes.linewidth":     0.6 * _K,
    "xtick.major.width":  0.6 * _K,
    "ytick.major.width":  0.6 * _K,
    "xtick.major.size":   3.0 * _K,
    "ytick.major.size":   3.0 * _K,
    "lines.linewidth":    1.0 * _K,
    "lines.markersize":   3.5 * _K,
    "grid.linewidth":     0.4 * _K,
})

# Make the modules in src/ importable regardless of the working directory.
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# Shared plotting helpers (physical-region clipping, 2D maps)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_helpers import (physical_radius, potential_ylim,
                          masked_plane, symmetric_vlim, potential_error)

from consts3D import (a as ALPHA, hbc, nuclear_omega_x,
                      nuclearNu_x, nuclearNu_y, nuclearNu_z,
                      b_x, b_y, b_z)
from orbitals3D import uncoupledHObasis, getOrbitalSet
from optimizer3D_lm import iKS_problem
from lagrange_mesh_3D import LagrangeMesh3D
from eigenfunctions3D import HO3D

OUTPUT_DIR = "test_3D_data_8part"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RHO_FLOOR_FRAC = 1e-10       # physical-node mask threshold (Eq. 3.13)
REG            = 1e-3          # beta: Tikhonov regularisation weight
N_RUN          = 21          # nodes per axis of the full run (N >~ 20 needed)


# ──────────────────────────────────────────────────────────────────────────────
# References and target density
# ──────────────────────────────────────────────────────────────────────────────
def kinetic_reference(A, orbitals):
    """Analytic kinetic energy sum_k occ_k (n_k + 3/2)(hbar*omega/2) of the 3D HO
    (virial theorem: T = E/2) [MeV]."""
    hw = hbc * nuclear_omega_x(A)
    E = 0.0
    for o in orbitals:
        n = o.nx + o.ny + o.nz
        E += o.occupation * 0.5 * (n + 1.5) * hw
    return E


def build_target(mesh, A, n_part):
    """Target density from the aufbau orbitals, normalised with the mesh weights.

    Returns (rho_target, list of phi_norm, orbitals). The orbitals are built
    on the physical length b_x(A).
    """
    x, y, z = mesh.x, mesh.y, mesh.z
    orbitals = list(getOrbitalSet(n_part, uncoupledHObasis()))
    phi_norm = []
    for o in orbitals:
        wf  = HO3D(o.nx, o.ny, o.nz, b_x(A), b_y(A), b_z(A), mesh.dV)
        p   = wf(x, y, z)
        p  /= np.sqrt(mesh.integrate(p**2))
        phi_norm.append(p)
    rho = sum(orbitals[k].occupation * phi_norm[k]**2
              for k in range(len(orbitals)))
    return rho, phi_norm, orbitals


def decompose_from_f(mesh, rho, f_list, orbitals, beta):
    """T_alpha (physical kinetic energy) and T_beta (regulariser) given f_k = phi_k/sqrt(rho).

    Both are restricted to the physical nodes (same mask as the optimiser):
        T_alpha = sum_k occ_k int alpha rho (|grad f|^2 + f^2 |g|^2 + 2 f grad f.g)
        T_beta  = sum_k occ_k int beta |grad f|^2
    with g = grad log sqrt(rho). Every comparison with the analytic kinetic
    energy is made on T_alpha (thesis Appendix B.2).
    """
    mask = (rho >= RHO_FLOOR_FRAC * rho.max())
    s    = np.sqrt(np.maximum(rho, 1e-14 * rho.max()))
    glog = mesh.grad(np.log(s))
    nglog = np.sum(glog**2, axis=-1)
    T_a = T_b = 0.0
    for k, o in enumerate(orbitals):
        f   = f_list[k]
        gf  = mesh.grad(f)
        gm2 = np.sum(gf**2, axis=-1)
        dot = np.sum(glog * gf, axis=-1)
        part_a = ALPHA * rho * (gm2 + f**2 * nglog + 2 * f * dot)
        part_b = beta * gm2
        T_a += o.occupation * mesh.integrate(part_a * mask)
        T_b += o.occupation * mesh.integrate(part_b * mask)
    return T_a, T_b


def decompose_objective(mesh, rho, phi_norm, orbitals, beta):
    """As decompose_from_f, but starting from the orbitals phi_k (f_k = phi_k/sqrt(rho))."""
    s = np.sqrt(np.maximum(rho, 1e-14 * rho.max()))
    f_list = [phi_norm[k] / s for k in range(len(orbitals))]
    return decompose_from_f(mesh, rho, f_list, orbitals, beta)


# ──────────────────────────────────────────────────────────────────────────────
# Convergence scan of the physical kinetic energy T_alpha
# ──────────────────────────────────────────────────────────────────────────────
def convergence_scan(A, n_part, N_list, beta):
    """T_alpha and T_beta on the EXACT target for each N (no optimisation).

    Isolates the discretisation error (thesis Table 5.1). Returns
    (E_ref, rows) with rows = [(N, T_alpha, T_beta, err_%, n_live, n_total), ...].
    """
    orbitals0 = list(getOrbitalSet(n_part, uncoupledHObasis()))
    E_ref = kinetic_reference(A, orbitals0)
    rows = []
    for N in N_list:
        mesh = LagrangeMesh3D(N, N, N, b_x(A), b_y(A), b_z(A))
        rho, phi_norm, orbitals = build_target(mesh, A, n_part)
        T_a, T_b = decompose_objective(mesh, rho, phi_norm, orbitals, beta)
        mask = (rho >= RHO_FLOOR_FRAC * rho.max())
        err = 100.0 * (T_a - E_ref) / E_ref
        rows.append((N, T_a, T_b, err, int(mask.sum()), rho.size))
    return E_ref, rows


# ──────────────────────────────────────────────────────────────────────────────
# Full IKS run
# ──────────────────────────────────────────────────────────────────────────────
def run_iks(A, n_part, N, beta, perturb=0.0, seed=0, mesh_scale=1.0):
    """Solve the IKS problem on an N^3 Lagrange mesh with scale s and analyse it.

    Returns a dict with status, objective, T_alpha/T_beta/T_direct, density
    and (worst) orthonormality errors, potential error metrics and the arrays
    used by the figures. It also prints the pointwise violation of the density
    constraint and the multipole content of the residual density error.

    NOTE: ``perturb``/``seed`` are INERT. ``iKS_problem.solve()`` always calls
    ``getStartingPoint()`` internally, so the perturbed vector built below is
    never used. To really start from a perturbed point, override
    ``prob.getStartingPoint`` on the instance.
    """
    # mesh_scale s: b_mesh = s*b_x(A). It scales ONLY the geometry of the mesh;
    # the physics (build_target, V_ana) stays on b_x(A). Same s as the mesh
    # built inside iKS_problem (passed below).
    mesh = LagrangeMesh3D(N, N, N, mesh_scale * b_x(A),
                          mesh_scale * b_y(A), mesh_scale * b_z(A))
    x, y, z = mesh.x, mesh.y, mesh.z
    rho_target, phi_norm, orbitals = build_target(mesh, A, n_part)
    n_orb = len(orbitals)

    prob = iKS_problem(
        Z=A, N=0, lb=[-2, -2, -2], ub=[2, 2, 2],
        data=[x, y, z, rho_target],
        basis=uncoupledHObasis(), reg_parameter=beta,
        output_folder="test_8_particles",
        exact_hess=True, max_iter=500,
        rel_tol=1e-4, constr_viol=1e-4, dual_inf_tol=1.0,
        use_lagrange_mesh=True, Nx=N, Ny=N, Nz=N, mesh_scale=mesh_scale,
    )
    prob.add_option(b"mu_init", 0.01)

    # Starting point, optionally perturbed (inert: see the docstring)
    st = prob.getStartingPoint()
    if perturb > 0:
        rng = np.random.default_rng(seed)
        live = np.tile(prob.active_mask.ravel() > 0, prob.n_orbitals)
        noise = rng.standard_normal(st.shape) * perturb
        st = st + noise * live
        prob._perturbed_start = st     # stored only; solve() does not read it

    t0 = time.time()
    results, info = prob.solve()
    elapsed = time.time() - t0

    obj      = float(results.get('obj', float('nan')))
    status   = results.get('status', 'N/A')
    phi      = results['u']
    rho_calc = np.sum([orbitals[k].occupation * phi[k]**2
                       for k in range(n_orb)], axis=0)
    norm_calc   = mesh.integrate(rho_calc)
    density_err = np.sqrt(mesh.integrate((rho_calc - rho_target)**2))

    # Orthogonality of the reconstructed orbitals: with 4 orbitals there are 6
    # pairs; the WORST overlap (max absolute value) is reported as a single metric.
    f_sol = results['x'].reshape(n_orb, mesh.Nx, mesh.Ny, mesh.Nz)
    orto_val = 0.0
    for i in range(n_orb):
        for j in range(i + 1, n_orb):
            oij = mesh.integrate(rho_target * f_sol[i] * f_sol[j])
            if abs(oij) > abs(orto_val):
                orto_val = oij

    # Decomposition of the objective at the solution (directly from the f_k)
    f_list = [f_sol[k] for k in range(n_orb)]
    T_a, T_b = decompose_from_f(mesh, rho_target, f_list, orbitals, beta)

    # ─── Diagnostic: does the reconstructed density have any room left? ──────
    # The constraint of iKS_problem is POINTWISE: sum_k occ_k f_k(r)^2 - 1 = 0
    # on every live node. If it is satisfied exactly, rho_KS == rho_target
    # pointwise and its multipole moments are already those of the target
    # (zero for L > 0, since at A = 8 the density is spherical). Imposing
    # Q_LM[rho_KS] = 0 would then be an IDENTICALLY satisfied constraint
    # (dependent Jacobian row, LICQ violated). This analysis motivated the
    # multipole study of thesis Section 6.1.
    #
    # But constr_viol = 1e-4: IPOPT may stop with a residual violation. The two
    # questions, in order:
    #   (1) how large is the POINTWISE violation (not the integrated L2 norm)?
    #   (2) does the residual error rho_KS - rho_target have NON-SPHERICAL
    #       components?
    # Only if (2) is not negligible would multipole constraints have anything
    # to act on.

    # (1) pointwise violation of the constraint, live nodes only
    sum_occ_f2 = np.sum([orbitals[k].occupation * f_sol[k]**2
                        for k in range(n_orb)], axis=0)
    viol = np.abs(sum_occ_f2.ravel() - 1.0)[prob.live_nodes]
    print(f"\n  ── Vincolo di densità (puntuale) ──")
    print(f"    max |Σocc·f²−1| : {viol.max():.3e}   (constr_viol = 1e-4)")
    print(f"    rms             : {np.sqrt((viol**2).mean()):.3e}")
    print(f"    Δρ L2 (integr.) : {density_err:.3e}")

    # (2) multipole structure of the residual density error
    #     Q_LM[d_rho] = int d_rho(r) r^L Y_LM(r^) w dV, divided by A to make it
    #     dimensionless per particle. Compared with Q_LM[rho_target], which
    #     MUST be ~0: otherwise the problem is in the mesh or in the weights,
    #     not in the solution.
    d_rho = rho_calc - rho_target
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')
    RR2 = XX**2 + YY**2 + ZZ**2

    # real solid harmonics r^L Y_LM: only L = 2 and L = 4 (odd L vanish by parity)
    R_LM = {
        "Q2+0": (3*ZZ**2 - RR2) / 2.0,
        "Q2+1": np.sqrt(3.0) * XX * ZZ,
        "Q2-1": np.sqrt(3.0) * YY * ZZ,
        "Q2+2": np.sqrt(3.0) / 2.0 * (XX**2 - YY**2),
        "Q2-2": np.sqrt(3.0) * XX * YY,
        "Q4+0": (35*ZZ**4 - 30*ZZ**2*RR2 + 3*RR2**2) / 8.0,
    }

    print(f"\n  ── Momenti multipolari (normalizzati su Z={n_part}) ──")
    print(f"    {'canale':>8}  {'ρ_target':>12}  {'δρ = ρ_KS−ρ_tgt':>18}")
    for tag, R in R_LM.items():
        q_tgt = mesh.integrate(rho_target * R) / n_part
        q_res = mesh.integrate(d_rho * R) / n_part
        print(f"    {tag:>8}  {q_tgt:>12.2e}  {q_res:>18.2e}")

    print("""
        Lettura:
        colonna ρ_target ~1e-12  → mesh e pesi corretti (sanity check)
        colonna δρ       ~1e-12  → la densità ricostruita è esatta: i vincoli
                                    multipolari sono VUOTI, non c'è niente da
                                    imporre
        colonna δρ       ≳1e-5   → l'errore residuo HA componenti non sferiche:
                                    i vincoli Q_LM[ρ_KS]=0 hanno su cosa mordere
    """)

    # ── Physical diagnostic: consistency of the two forms of the kinetic energy
    # For a PHYSICAL solution the two expressions of the kinetic energy agree:
    #   T_alpha (expanded form, via f_k and grad log sqrt(rho))
    #       ~= T_direct = alpha sum_k q_k int |grad phi_k|^2.
    # For a sub-grid ARTEFACT (orbitals oscillating at the Nyquist scale) the
    # discrete product rule grad phi = sqrt(rho) grad f + f grad sqrt(rho)
    # no longer balances and the two forms DIVERGE. The ratio T_direct/T_alpha
    # is a dimensionless indicator of such sub-grid structure:
    #   ~ 1 -> no sub-grid structure;  >> 1 (or << 1) -> artefact suspected.
    # (The von Weizsaecker bound T_vW remains valid in theory, but computing it
    #  directly on a mesh with density nodes is inaccurate; the ratio is not.)
    mask_phys = (rho_target >= RHO_FLOOR_FRAC * rho_target.max())
    T_direct = 0.0
    for k, o in enumerate(orbitals):
        phi_masked = np.where(mask_phys, phi[k], 0.0)
        gphi = mesh.grad(phi_masked)
        T_direct += o.occupation * ALPHA * mesh.integrate(
            np.sum(gphi**2, axis=-1) * mask_phys)
    ratio = T_direct / T_a if abs(T_a) > 1e-12 else float('inf')
    # NOTE: on the Gauss-Hermite mesh T_direct = alpha int |grad phi|^2 is less
    # reliable than the expanded form T_alpha, because the Lagrange derivative
    # of the reconstructed orbitals is sensitive to structure near the edge of
    # the support. The physicality verdict on GH is therefore based on the
    # controlled CONVERGENCE of T_alpha to the analytic value and on the
    # constraints being satisfied, not on the ratio, which is reported for
    # information only.
    physical = (density_err < 1e-6) and (abs(orto_val) < 1e-6)

    # ── Potential: V_KS = lambda/(rho w), multipliers on the live nodes (Eq. 4.8)
    lam_live = results['lagrange'][:prob.n_live]
    lam_full = np.full(mesh.n_points, np.nan)
    lam_full[prob.live_nodes] = lam_live
    lam = lam_full.reshape(mesh.Nx, mesh.Ny, mesh.Nz)
    w3D = mesh.w3D
    mask = (rho_target >= RHO_FLOOR_FRAC * rho_target.max())
    with np.errstate(divide='ignore', invalid='ignore'):
        V_iks = lam / (rho_target * w3D)
    V_iks = np.where(mask, V_iks, np.nan)
    # Gauge: V = 0 at the node of maximum density
    icx, icy, icz = np.unravel_index(np.nanargmax(rho_target), rho_target.shape)
    V_iks -= V_iks[icx, icy, icz]

    # Reference: the ISOTROPIC harmonic oscillator. (000) and the whole triplet
    # (100), (010), (001) are the two lowest levels of the trap (E = 1.5 and
    # 2.5 hbar*omega), so the target is the density obtained by filling the
    # oscillator with 8 nucleons. With the N = 1 shell COMPLETE the density is
    # spherical and the TRUE Kohn-Sham potential of this density IS the
    # isotropic oscillator. RMSE(V_IKS - V_HO) is a legitimate, exact metric.
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    V_ho = (nuclearNu_x(A) * X**2 + nuclearNu_y(A) * Y**2 + nuclearNu_z(A) * Z**2)
    V_ho -= V_ho[icx, icy, icz]

    # Potential error on the physical region rho >= 1e-2 rho_max (Section 4.4).
    pot_err = potential_error(V_iks, V_ho, rho_target, x, frac=1e-2)

    return dict(
        N=N, mesh=mesh, status=status, obj=obj, elapsed=elapsed,
        T_alpha=T_a, T_beta=T_b, T_direct=T_direct, ratio=ratio, physical=physical,
        norm_calc=norm_calc, density_err=density_err, pot_err=pot_err,
        orto=orto_val, V_iks=V_iks, V_ho=V_ho, rho_target=rho_target,
        rho_calc=rho_calc, phi=phi, orbitals=orbitals,
        center=(icx, icy, icz), grid=(x, y, z),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Scan in beta: effect of the regularisation on kinetic energy and potential
# ──────────────────────────────────────────────────────────────────────────────
def beta_scan(A, n_part, N, beta_list, E_ref):
    """Run the full IKS problem for several values of beta and collect T_alpha,
    T_beta, the density error and the RMSE of the potential (vs harmonic).
    Quantifies the regularisation / accuracy trade-off (thesis Fig. 5.8): a
    larger beta stabilises the optimisation but distorts the orbitals and
    hence the extracted potential.

    NOTE: unlike the scan in N, this requires a complete optimisation for each
    beta (the effect of beta is in the solution, not in the functional
    evaluated on the exact density).

    Returns rows (beta, T_alpha, T_beta, err_%, density_err, V_rmse_weighted,
    V_max, status).
    """
    rows = []
    for beta in beta_list:
        res = run_iks(A, n_part, N, beta)
        err = 100.0 * (res['T_alpha'] - E_ref) / E_ref
        pe = res['pot_err']
        rows.append((beta, res['T_alpha'], res['T_beta'], err,
                     res['density_err'], pe['rmse_pesato'], pe['max_abs'],
                     res['status']))
    return rows


def make_beta_figure(beta_rows, E_ref, fig_path):
    """Figure of the trade-off in beta: kinetic error and RMSE of the potential.

    Left panel: |error T_alpha|(beta) and RMSE V(beta) on log axes, to show how
    both depend on the regularisation. Right panel: T_alpha(beta) with the
    analytic value, to read the bias directly.
    """
    betas = [r[0] for r in beta_rows]
    Tas   = [r[1] for r in beta_rows]
    errs  = [abs(r[3]) for r in beta_rows]
    Vrmse = [r[5] for r in beta_rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("IKS 3D Z=8 (Lagrange mesh): compromesso in β",
                 fontweight="bold", fontsize=12)

    ax = axes[0]
    l1 = ax.loglog(betas, Vrmse, "o-", color="#dc2626", label="RMSE potenziale [MeV]")
    ax.set_xlabel("β (regolarizzazione)")
    ax.set_ylabel("RMSE V [MeV]", color="#dc2626")
    ax.tick_params(axis="y", labelcolor="#dc2626")
    ax.grid(alpha=0.4, which="both")
    ax2 = ax.twinx()
    l2 = ax2.loglog(betas, np.array(errs) + 1e-6, "s--", color="#2563eb",
                    label="|errore T$_α$| [%]")
    ax2.set_ylabel("|errore T$_α$| [%]", color="#2563eb")
    ax2.tick_params(axis="y", labelcolor="#2563eb")
    ax.set_title("Errore cinetico e potenziale vs β")
    lines = l1 + l2
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="best")

    ax = axes[1]
    ax.axhline(E_ref, color="#16a34a", lw=2, ls="--",
               label=f"T$_α$ analitico = {E_ref:.1f} MeV")
    ax.semilogx(betas, Tas, "o-", color="#2563eb", label="T$_α$(β)")
    ax.set_xlabel("β (regolarizzazione)"); ax.set_ylabel("T$_α$ [MeV]")
    ax.set_title("Cinetica ricostruita vs β")
    ax.legend(fontsize=8); ax.grid(alpha=0.4, which="both")

    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figura β-scan salvata: {fig_path}")


def N_beta_scan(A, n_part, N_list, beta_list, E_ref):
    """2D scan over (N, beta): for each pair collect the RMSE of the potential
    and the kinetic error. Separates the DISCRETISATION error (decreasing with
    N) from the REGULARISATION bias (decreasing as beta -> 0). One complete
    optimisation per cell: expensive (len(N) * len(beta) runs).

    Returns (Vrmse, Terr) as arrays of shape (len(N_list), len(beta_list)).
    """
    nN, nB = len(N_list), len(beta_list)
    Vrmse = np.full((nN, nB), np.nan)
    Terr  = np.full((nN, nB), np.nan)
    for i, N in enumerate(N_list):
        for j, beta in enumerate(beta_list):
            res = run_iks(A, n_part, N, beta)
            Vrmse[i, j] = res['pot_err']['rmse_pesato']
            Terr[i, j]  = abs(100.0 * (res['T_alpha'] - E_ref) / E_ref)
            print(f"    N={N:>3}, β={beta:>7.0e}:  RMSE V={Vrmse[i,j]:8.3f} MeV, "
                  f"|err T_α|={Terr[i,j]:6.3f}%")
    return Vrmse, Terr


def make_Nbeta_figure(N_list, beta_list, Vrmse, fig_path):
    """Heat map of the RMSE of the potential on the (N, beta) grid, with a
    logarithmic colour scale (the error spans orders of magnitude)."""
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    data = np.clip(Vrmse, 1e-3, None)
    im = ax.imshow(data, origin="lower", aspect="auto",
                   norm=LogNorm(vmin=np.nanmin(data), vmax=np.nanmax(data)),
                   cmap="viridis_r")
    ax.set_xticks(range(len(beta_list)))
    ax.set_xticklabels([f"{b:.0e}" for b in beta_list])
    ax.set_yticks(range(len(N_list)))
    ax.set_yticklabels([str(N) for N in N_list])
    ax.set_xlabel("β (regolarizzazione)"); ax.set_ylabel("N (nodi per asse)")
    ax.set_title("RMSE potenziale [MeV] su griglia (N, β)")
    # annotate the values
    for i in range(len(N_list)):
        for j in range(len(beta_list)):
            if np.isfinite(Vrmse[i, j]):
                ax.text(j, i, f"{Vrmse[i,j]:.2g}", ha="center", va="center",
                        fontsize=8, color="white")
    fig.colorbar(im, ax=ax, label="RMSE V [MeV] (scala log)")
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Heatmap (N,β) salvata: {fig_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────
def _legend_on_top(ax, frac=0.5, **kw):
    """Extend the y axis upwards (by ``frac`` times the data range) and place the
    legend in the free space, so that it does not cover the curves even when
    the figure is shrunk to the text width."""
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        ax.set_ylim(lo, hi * (hi / lo) ** frac)
    else:
        ax.set_ylim(lo, hi + frac * (hi - lo))
    ax.legend(loc="upper center", **kw)


def make_figure(E_ref, scan_rows, res, fig_path):
    """3x2 validation figure: T_alpha(N), orbitals, density and potential along x,
    2D maps of density and potential on the plane z = 0."""
    Ns  = [r[0] for r in scan_rows]
    Tas = [r[1] for r in scan_rows]

    x, y = res['grid'][0], res['grid'][1]
    icx, icy, icz = res['center']

    # Physical region: beyond the radius R (rho >= 1e-3 max) the potential
    # V = lambda/(rho w) diverges because rho -> 0; it is excluded from the plots.
    rho_x = res['rho_target'][:, icy, icz]
    R = physical_radius(rho_x, x, frac=1e-3)
    Rlim = 1.15 * R

    # Regular 3x2 grid (portrait, fits the text width of the thesis): all
    # panels square and of the same size. The colour bars of the 2D maps live
    # in horizontal "inset" axes below the panel, so they neither shrink the
    # map nor widen the gap between the two columns.
    fig, ax = plt.subplots(3, 2, figsize=(FIG_W, FIG_H), constrained_layout=True)
    for a in ax.flat:
        a.set_box_aspect(1)
    # minimal spacing between the panels: compact figure
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.0, hspace=0.02)
    # row 1: kinetic energy, orbitals | row 2: 1D density and potential | row 3: 2D maps
    fig.suptitle("A=8 (4 orbitali) — Lagrange mesh", fontweight="bold")

    # ── Row 1: kinetic energy, orbitals ──────────────────────────────────────
    a = ax[0, 0]
    a.axhline(E_ref, color="#dc2626", lw=2, ls="--",
              label=f"analitico: {E_ref:.1f} MeV")
    a.plot(Ns, Tas, "o-", color="#2563eb", label="T$_\\alpha$(N)")
    a.set_xlabel("N$_{\\rm GH}$ (nodi per asse)"); a.set_ylabel("T$_\\alpha$ [MeV]")
    a.set_title("Energia cinetica"); _legend_on_top(a); a.grid(alpha=0.4)

    a = ax[0, 1]
    colors = ["#0891b2", "#be185d", "#ea580c", "#7c3aed"]
    for k, o in enumerate(res['orbitals']):
        a.plot(x, res['phi'][k][:, icy, icz], lw=1.8, color=colors[k % len(colors)],
               label=f"φ$_{k}$")
    a.axhline(0, color="k", lw=0.6, ls=":")
    a.set_xlim(-Rlim, Rlim)
    a.set_xlabel("x [fm]"); a.set_ylabel("φ [fm⁻³ᐟ²]")
    a.set_title("Orbitali ricostruiti")
    _legend_on_top(a, frac=0.7, ncol=2); a.grid(alpha=0.4)

    # ── Row 2: density and potential along x ─────────────────────────────────
    a = ax[1, 0]
    a.plot(x, res['rho_target'][:, icy, icz], color="#dc2626", lw=2, label="ρ target")
    a.plot(x, res['rho_calc'][:, icy, icz], "--", color="#16a34a", lw=1.8, label="ρ IKS")
    a.set_xlim(-Rlim, Rlim)
    a.set_xlabel("x [fm]"); a.set_ylabel("ρ [fm⁻³]")
    a.set_title(f"Densità lungo x (norma={res['norm_calc']:.3f})")
    _legend_on_top(a, ncol=2); a.grid(alpha=0.4)

    a = ax[1, 1]
    V_iks_x = res['V_iks'][:, icy, icz]
    V_ho_x  = res['V_ho'][:, icy, icz]
    a.plot(x, V_ho_x, color="#dc2626", lw=2, label="½mω²r²")
    a.plot(x, V_iks_x, "--", color="#2563eb", lw=1.8, label="V$_{KS}$ IKS")
    a.set_xlim(-Rlim, Rlim)
    yl = potential_ylim(V_iks_x, x, R)
    if yl:
        a.set_ylim(*yl)
    a.set_xlabel("x [fm]"); a.set_ylabel("V [MeV]")
    a.set_title(f"Potenziale lungo x\n(RMSE={res['pot_err']['rmse_pesato']:.2f} MeV)")
    _legend_on_top(a); a.grid(alpha=0.4)

    # ── Row 3: 2D maps on the plane z = 0, cropped to the physical region ────
    ic = (icx, icy, icz)

    a = ax[2, 0]
    rho2d, extent = masked_plane(res['rho_target'], x, y, ic, Rlim)
    im = a.imshow(rho2d, origin="lower", extent=extent, aspect="equal", cmap="viridis")
    fig.colorbar(im, cax=a.inset_axes([0.0, -0.29, 1.0, 0.045]),
                 orientation="horizontal", label="ρ [fm⁻³]")
    a.set_xlabel("x [fm]"); a.set_ylabel("y [fm]")
    a.set_title("Densità ρ(x,y), piano z=0")

    a = ax[2, 1]
    V2d, extent = masked_plane(res['V_iks'], x, y, ic, R)   # mask at the physical radius R
    vlo, vhi = symmetric_vlim(V2d)
    im = a.imshow(V2d, origin="lower", extent=extent, aspect="equal",
                  cmap="RdBu_r", vmin=vlo, vmax=vhi)
    fig.colorbar(im, cax=a.inset_axes([0.0, -0.29, 1.0, 0.045]),
                 orientation="horizontal", label="V [MeV]")
    a.set_xlabel("x [fm]"); a.set_ylabel("y [fm]")
    a.set_title("Potenziale V$_{KS}$(x,y), piano z=0")

    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figura salvata: {fig_path}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   TEST IKS 3D — Z=8 (shell chiusa, 4 orbitali)     [Lagrange Mesh]   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    A = 8; n_part = 8
    orbitals0 = list(getOrbitalSet(n_part, uncoupledHObasis()))
    print(f"  Orbitali: " + ", ".join(
        f"({o.nx},{o.ny},{o.nz}) occ={o.occupation}" for o in orbitals0))

    # ── Convergence scan of the physical kinetic energy (no optimisation) ────
    N_list = [8, 12, 16, 20, 24]
    E_ref, scan_rows = convergence_scan(A, n_part, N_list, REG)
    print(f"\n  E_kin analitico = {E_ref:.4f} MeV  (= 9ℏω)\n")
    print("  Convergenza cinetica fisica T_α (densità target, maschera):")
    print(f"  {'N':>4}  {'T_α [MeV]':>12}  {'T_β [MeV]':>12}  {'err T_α':>9}  {'nodi vivi':>12}")
    for (N, Ta, Tb, errp, live, tot) in scan_rows:
        print(f"  {N:>4}  {Ta:>12.3f}  {Tb:>12.3f}  {errp:>8.2f}%  {live:>6}/{tot:<6}")

    # ── Full IKS run ─────────────────────────────────────────────────────────
    print(f"\n  Run IKS completo (Hessiana esatta) a N={N_RUN} …")
    res = run_iks(A, n_part, N_RUN, REG, mesh_scale=0.794)    # s = 0.794 (Section 5.5)
    print(f"\n  Status         : {res['status']}")
    print(f"  Obiettivo tot  : {res['obj']:.4f} MeV  (= T_α + T_β)")
    print(f"  T_α (cinetica) : {res['T_alpha']:.4f} MeV   (rif. {E_ref:.4f}, "
          f"Δ {100*(res['T_alpha']-E_ref)/E_ref:+.2f}%)")
    print(f"  T_β (regolar.) : {res['T_beta']:.4f} MeV")
    print(f"  T diretta      : {res['T_direct']:.4f} MeV   (α∫|∇φ|², solo indicativo su GH)")
    err_Ta = 100*(res['T_alpha']-E_ref)/E_ref
    if res['physical'] and abs(err_Ta) < 5.0:
        print(f"  ✓ soluzione FISICA: T_α converge all'analitico (Δ {err_Ta:+.2f}%),")
        print(f"    densità e ortogonalità soddisfatte a precisione di macchina")
    else:
        print(f"  ✗ ATTENZIONE: verificare la soluzione (Δ T_α = {err_Ta:+.2f}%)")
    print(f"  Δρ L2          : {res['density_err']:.3e}")
    print(f"  Norma ρ_calc   : {res['norm_calc']:.4f}  (atteso {n_part})")
    print(f"  Ortogonalità ⟨φ_0|φ_1⟩_ρ : {res['orto']:.3e}  (atteso 0)")
    pe = res['pot_err']
    print(f"  Errore potenziale (vs armonico, regione fisica):")
    print(f"    RMSE uniforme : {pe['rmse_uniforme']:.3f} MeV  ({pe['n_nodi']} nodi)")
    print(f"    RMSE pesato ρ : {pe['rmse_pesato']:.3f} MeV")
    print(f"    max |ΔV|      : {pe['max_abs']:.3f} MeV")

    make_figure(E_ref, scan_rows, res,
                os.path.join(OUTPUT_DIR, "IKS_8particles_validation.pdf"))

    # ── Optional scan in beta (run with the flag --beta-scan) ────────────────
    if "--beta-scan" in sys.argv:
        print(f"\n  Scan in β a N={N_RUN} (un'ottimizzazione completa per ogni β)…")
        beta_list = [1.0, 0.1, 0.01, 0.001, 1e-4]
        brows = beta_scan(A, n_part, N_RUN, beta_list, E_ref)
        print(f"\n  {'β':>8}  {'T_α [MeV]':>12}  {'err T_α':>9}  "
              f"{'RMSE V':>9}  {'max|ΔV|':>9}  {'Δρ L2':>10}")
        for (beta, Ta, Tb, errp, drho, vrmse, vmax, status) in brows:
            print(f"  {beta:>8.0e}  {Ta:>12.4f}  {errp:>8.2f}%  "
                  f"{vrmse:>9.2f}  {vmax:>9.2f}  {drho:>10.2e}")
        make_beta_figure(brows, E_ref,
                         os.path.join(OUTPUT_DIR, "IKS_8particles_beta_scan.pdf"))
        print("\n  Interpretazione: al diminuire di β la cinetica T_α si avvicina al")
        print("  valore esatto e l'RMSE del potenziale CALA (orbitali meno distorti,")
        print("  V più vicino all'armonico). Attenzione però: β troppo piccolo può")
        print("  riavvicinare al problema di non-unicità (vedi diagnostica von")
        print("  Weizsäcker). Il valore ottimale bilancia accuratezza di V_KS e")
        print("  stabilità dell'ottimizzazione.")

    # ── 2D scan (N, beta): run with the flag --Nbeta-scan ────────────────────
    if "--Nbeta-scan" in sys.argv:
        print(f"\n  Scan 2D (N, β): separa errore di discretizzazione (N) da bias (β).")
        print(f"  ATTENZIONE: un'ottimizzazione completa per ogni cella della griglia.\n")
        N_grid    = [16, 20, 24, 28]
        beta_grid = [0.1, 0.01, 0.001]
        Vrmse, Terr = N_beta_scan(A, n_part, N_grid, beta_grid, E_ref)
        make_Nbeta_figure(N_grid, beta_grid, Vrmse,
                          os.path.join(OUTPUT_DIR, "IKS_8particles_Nbeta_scan.pdf"))
        print("\n  Lettura: lungo le righe (N crescente) cala l'errore di")
        print("  discretizzazione; lungo le colonne (β decrescente) cala il bias.")
        print("  Il minimo dell'RMSE indica la combinazione (N, β) ottimale.")

    print("\n  Fatto.\n")


if __name__ == "__main__":
    main()