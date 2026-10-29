"""
test_IKS_3D_8part_uniform.py
============================
IKS test for A = 8 (closed p shell) on the UNIFORM GRID (findiff), to be
compared with the Gauss-Hermite version ``test_IKS_3D_8part.py``.
Thesis: Sections 5.1-5.2 (Fig. 5.4; Fig. 5.7 via efficiency_precision_8part.py).

Same optimiser (optimizer3D_lm.py), same physics; the ONLY difference is the
discretisation (equispaced grid instead of Gauss-Hermite nodes).

A = 8: CLOSED SHELL. Four orbitals (000), (100), (010), (001), occ = 2 each,
with orthonormality constraints between all pairs. The complete N = 1 shell
(px, py, pz) makes the density SPHERICAL, so the true V_KS is rigorously the
isotropic oscillator m omega^2 r^2 / 2 and the RMSE of the potential against
it is an exact metric.
Analytic kinetic energy: 9 hbar*omega ~= 184.5 MeV.

The objective splits into T_alpha (physical kinetic energy) + T_beta
(Tikhonov regulariser); the physical comparison is made on T_alpha.

NOTE: Fig. 5.4 of the thesis was produced with n = 31; the default N_GRID
below is 21 (check the values before reproducing that figure).

Usage
-----
    python test_IKS_3D_8part_uniform.py                  # convergence scan + full run
    python test_IKS_3D_8part_uniform.py --beta-scan      # physicality vs (large) beta
    python test_IKS_3D_8part_uniform.py --beta-scan-acc  # accuracy vs beta
    python test_IKS_3D_8part_uniform.py --Nbeta-scan     # 2D scan in (n, beta)

Output: test_3D_data_8part_uniform/IKS_8particles_uniform*.pdf

This module is also imported by ``efficiency_precision_8part.py``.
"""

import sys, os, time
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
from findiff import FinDiff

# Make the modules in src/ importable regardless of the working directory.
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_helpers import (physical_radius, potential_ylim,
                          masked_plane, symmetric_vlim, potential_error)

from consts3D import (a as ALPHA, hbc, nuclear_omega_x,
                      nuclearNu_x, nuclearNu_y, nuclearNu_z, b_x)
from orbitals3D import uncoupledHObasis, getOrbitalSet
from optimizer3D_lm import iKS_problem
from eigenfunctions3D import HO3D

OUTPUT_DIR = "test_3D_data_8part_uniform"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REG    = 0.005   # beta: Tikhonov regularisation weight.
# NOTE on non-uniqueness: many sets of orthonormal orbitals reproduce the same
# density. On a fine uniform grid the optimiser can slide towards a spurious
# solution in which T_alpha drops BELOW the von Weizsaecker bound (sub-grid
# structure in the f_k). A larger beta discourages that structure by
# penalising |grad f|. The value must be calibrated: too small does not
# prevent the collapse, too large distorts the kinetic energy. The test
# reports whether the solution violates the von Weizsaecker bound.
#
# Resolution and box (A = 8, b = b_x(8) ~ 1.42 fm):
# The uniform grid uses N_GRID^3 points; IPOPT's linear solver (MUMPS)
# factorises the KKT matrix and the memory grows quickly. With 4 orbitals
# (vs 2 for A = 4) the number of variables doubles at equal n, so moderate
# values of n are advisable:
#   n=31, L=8.0 -> h~0.53 fm, ~120k variables -> fine
#   n=41, L=9.0 -> h~0.45 fm, ~276k variables -> at the limit / possible out of memory
# The box must contain the three p orbitals (100), (010), (001): with
# b ~ 1.42 fm the density drops below 1e-6 max around |x| ~ 5.5-6 fm, so
# L = 8.0 leaves some margin.
N_GRID = 21   # points per axis (the thesis Fig. 5.4 used n = 31; see the module docstring)
L_BOX  = 8.5  # half-width of the box [fm]. CRITICAL: a box that is too narrow cuts the
               # p orbitals where phi is still significant, producing artificially
               # large gradients at the edges that the one-sided acc=4 stencils
               # amplify. L >= 8.0 guarantees rho/rho_max < 1e-6 inside the box.


def kinetic_reference(A, orbitals):
    """Analytic kinetic energy sum_k occ_k (n_k + 3/2)(hbar*omega/2) [MeV]."""
    hw = hbc * nuclear_omega_x(A)
    return sum(o.occupation * 0.5 * (o.nx+o.ny+o.nz+1.5) * hw for o in orbitals)


def build_target_uniform(xs, h, A, n_part):
    """Target density on the cubic grid xs^3; returns (rho, phi_norm, orbitals)."""
    orbitals = list(getOrbitalSet(n_part, uncoupledHObasis()))
    phi_norm = []
    for o in orbitals:
        p = HO3D(o.nx, o.ny, o.nz, b_x(A), b_x(A), b_x(A), h**3)(xs, xs, xs)
        p /= np.sqrt(np.sum(p**2) * h**3)
        phi_norm.append(p)
    rho = sum(orbitals[k].occupation * phi_norm[k]**2 for k in range(len(orbitals)))
    return rho, phi_norm, orbitals


def decompose_from_f_uniform(xs, h, rho, f_list, orbitals, beta):
    """T_alpha and T_beta from f_k on the uniform grid (findiff derivatives),
    restricted to rho >= 1e-10 rho_max. Uniform-grid analogue of
    ``decompose_from_f`` in the Lagrange-mesh scripts."""
    mask = (rho >= 1e-10 * rho.max())
    s = np.sqrt(np.maximum(rho, 1e-14 * rho.max()))
    dx = FinDiff(0, h, 1, acc=4); dy = FinDiff(1, h, 1, acc=4); dz = FinDiff(2, h, 1, acc=4)
    glog = np.stack([dx(np.log(s)), dy(np.log(s)), dz(np.log(s))], axis=-1)
    nglog = np.sum(glog**2, axis=-1)
    T_a = T_b = 0.0
    for k, o in enumerate(orbitals):
        f = f_list[k]
        gf = np.stack([dx(f), dy(f), dz(f)], axis=-1)
        gm2 = np.sum(gf**2, axis=-1); dot = np.sum(glog * gf, axis=-1)
        T_a += o.occupation * np.sum(ALPHA * rho * (gm2 + f**2*nglog + 2*f*dot) * mask) * h**3
        T_b += o.occupation * np.sum(beta * gm2 * mask) * h**3
    return T_a, T_b


def convergence_scan_uniform(A, n_part, grids, beta):
    """T_alpha, T_beta on the exact target for each (n, L) (no optimisation).
    Returns (E_ref, rows) with rows = [(n, L, h, T_alpha, T_beta, err_%), ...]."""
    orbitals0 = list(getOrbitalSet(n_part, uncoupledHObasis()))
    E_ref = kinetic_reference(A, orbitals0)
    rows = []
    for (n, L) in grids:
        xs = np.linspace(-L, L, n); h = xs[1]-xs[0]
        rho, phi_norm, orbitals = build_target_uniform(xs, h, A, n_part)
        s = np.sqrt(np.maximum(rho, 1e-14*rho.max()))
        f_list = [phi_norm[k]/s for k in range(len(orbitals))]
        T_a, T_b = decompose_from_f_uniform(xs, h, rho, f_list, orbitals, beta)
        rows.append((n, L, h, T_a, T_b, 100*(T_a-E_ref)/E_ref))
    return E_ref, rows


def run_iks_uniform(A, n_part, n, L, beta):
    """Solve the IKS problem on an n^3 grid on [-L, L]^3 and analyse the solution.

    Returns a dict with status, objective, T_alpha/T_beta/T_vW/T_direct, the
    physicality flag, density/orthonormality errors, potential error metrics
    and the arrays for the figures. NOTE: unlike the A = 4 script, dual_inf_tol
    is not overridden here, so the value passed to the constructor (1.0) applies.
    """
    xs = np.linspace(-L, L, n); h = xs[1]-xs[0]
    rho_target, phi_norm, orbitals = build_target_uniform(xs, h, A, n_part)
    n_orb = len(orbitals)

    prob = iKS_problem(
        Z=A, N=0, lb=[-L, -L, -L], ub=[L, L, L],
        data=[xs, xs, xs, rho_target], hx=h, hy=h, hz=h,
        basis=uncoupledHObasis(), reg_parameter=beta,
        output_folder="test_8_uniform",
        exact_hess=True, max_iter=500,
        rel_tol=1e-3, constr_viol=1e-3, dual_inf_tol=1.0,
        use_lagrange_mesh=False,
    )
    prob.add_option(b"mu_init", 0.01)
    # Safety net for MUMPS memory on large grids: limits the over-allocation
    # percentage (default ~1000). Lowering it avoids "out of memory" failures
    # at the price of possible re-factorisations.
    prob.add_option(b"mumps_mem_percent", 20)

    t0 = time.time()
    results, info = prob.solve()
    elapsed = time.time() - t0

    obj      = float(results.get('obj', float('nan')))
    status   = results.get('status', 'N/A')
    phi      = results['u']
    rho_calc = np.sum([orbitals[k].occupation * phi[k]**2 for k in range(n_orb)], axis=0)
    norm_calc   = np.sum(rho_calc) * h**3
    density_err = np.sqrt(np.sum((rho_calc - rho_target)**2) * h**3)

    f_sol = results['x'].reshape(n_orb, n, n, n)
    # Worst orthogonality over all pairs (6 pairs for 4 orbitals)
    orto_val = 0.0
    for i in range(n_orb):
        for j in range(i + 1, n_orb):
            oij = np.sum(rho_target * f_sol[i] * f_sol[j]) * h**3
            if abs(oij) > abs(orto_val):
                orto_val = oij
    T_a, T_b = decompose_from_f_uniform(xs, h, rho_target,
                                        [f_sol[k] for k in range(n_orb)], orbitals, beta)

    # ── Physical diagnostic: consistency of the two forms of the kinetic energy
    # Primary criterion: the ratio T_direct/T_alpha. For a PHYSICAL solution the
    # two expressions of the kinetic energy agree (ratio ~ 1); for a sub-grid
    # ARTEFACT (orbitals oscillating at the Nyquist scale) the discrete product
    # rule grad phi = sqrt(rho) grad f + f grad sqrt(rho) does not balance and
    # the two forms DIVERGE (ratio >> 1). As additional information the von
    # Weizsaecker bound T_vW = alpha int |grad sqrt(rho)|^2 is reported
    # (rigorous lower bound T_s >= T_vW); on the uniform grid it is accurate.
    mask_phys = (rho_target >= 1e-10 * rho_target.max())
    s_t  = np.sqrt(np.maximum(rho_target, 1e-300))
    dxv = FinDiff(0, h, 1, acc=4); dyv = FinDiff(1, h, 1, acc=4); dzv = FinDiff(2, h, 1, acc=4)
    T_vW = ALPHA * np.sum((dxv(s_t)**2 + dyv(s_t)**2 + dzv(s_t)**2) * mask_phys) * h**3
    T_direct = 0.0
    for k, o in enumerate(orbitals):
        # Set phi to zero outside the physical region before differentiating:
        # avoids spurious gradients at the live/dead boundary.
        phi_masked = np.where(mask_phys, phi[k], 0.0)
        gphi = np.stack([dxv(phi_masked), dyv(phi_masked), dzv(phi_masked)], axis=-1)
        T_direct += o.occupation * np.sum(ALPHA*np.sum(gphi**2, axis=-1)*mask_phys) * h**3
    ratio = T_direct / T_a if abs(T_a) > 1e-12 else float('inf')
    physical = (0.5 < ratio < 2.0) and (T_a >= T_vW - 1e-6)

    # Potential V = lambda/(rho w), w = dV constant (Eq. 4.8)
    lam_live = results['lagrange'][:prob.n_live]
    lam_full = np.full(prob.n_points, np.nan)
    lam_full[prob.live_nodes] = lam_live
    lam = lam_full.reshape(n, n, n)
    w3D = prob.w3D
    mask = (rho_target >= 1e-10 * rho_target.max())
    with np.errstate(divide='ignore', invalid='ignore'):
        V_iks = lam / (rho_target * w3D)
    V_iks = np.where(mask, V_iks, np.nan)
    ic = n // 2                       # central node (n odd): gauge V(0) = 0
    V_iks -= V_iks[ic, ic, ic]

    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    V_ho = nuclearNu_x(A)*X**2 + nuclearNu_y(A)*Y**2 + nuclearNu_z(A)*Z**2
    V_ho -= V_ho[ic, ic, ic]

    # Potential error vs harmonic trap (the exact V_KS of this density), physical region
    pot_err = potential_error(V_iks, V_ho, rho_target, xs, frac=1e-2)

    return dict(n=n, L=L, h=h, status=status, obj=obj, elapsed=elapsed,
                T_alpha=T_a, T_beta=T_b, T_vW=T_vW, T_direct=T_direct, ratio=ratio,
                physical=physical, norm_calc=norm_calc, density_err=density_err,
                pot_err=pot_err,
                orto=orto_val, V_iks=V_iks, V_ho=V_ho, rho_target=rho_target,
                rho_calc=rho_calc, phi=phi, orbitals=orbitals, xs=xs, ic=ic)


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
    """3x2 figure: T_alpha(n), orbitals, density and potential along x,
    2D maps of density and potential on the plane z = 0."""
    Ns  = [r[0] for r in scan_rows]
    Tas = [r[3] for r in scan_rows]

    x = y = res['xs']
    icx = icy = icz = res['ic']

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
    fig.suptitle("A=8 (4 orbitali) — griglia uniforme", fontweight="bold")

    # ── Row 1: kinetic energy, orbitals ──────────────────────────────────────
    a = ax[0, 0]
    a.axhline(E_ref, color="#dc2626", lw=2, ls="--",
              label=f"analitico: {E_ref:.1f} MeV")
    a.plot(Ns, Tas, "o-", color="#2563eb", label="T$_\\alpha$(n)")
    a.set_xlabel("n (punti per asse)"); a.set_ylabel("T$_\\alpha$ [MeV]")
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
# Accuracy scans (beta and n-beta) — for direct comparison with the Lagrange mesh
# ──────────────────────────────────────────────────────────────────────────────
def beta_scan_acc(A, n_part, n, L, beta_list, E_ref):
    """Accuracy-oriented scan in beta (decreasing beta): T_alpha, kinetic error
    and RMSE of the potential for each beta. Analogue of ``beta_scan`` on the
    Lagrange mesh, for a direct comparison of the two discretisations.

    WARNING: on a fine uniform grid the IPOPT solution at A = 8 can fall into
    the spurious sub-grid solution; the RMSE measured here is that of the
    solution found, physical or not. Read it together with the diagnostic ratio.
    Returns rows (beta, T_alpha, err_%, V_rmse, V_max, ratio).
    """
    rows = []
    for beta in beta_list:
        r = run_iks_uniform(A, n_part, n, L, beta)
        err = 100.0 * (r['T_alpha'] - E_ref) / E_ref
        pe = r['pot_err']
        rows.append((beta, r['T_alpha'], err, pe['rmse_pesato'],
                     pe['max_abs'], r['ratio']))
    return rows


def make_beta_figure_u(beta_rows, E_ref, fig_path):
    """Figure of the trade-off in beta (uniform grid), same layout as on the Lagrange mesh."""
    betas = [r[0] for r in beta_rows]
    Tas   = [r[1] for r in beta_rows]
    errs  = [abs(r[2]) for r in beta_rows]
    Vrmse = [r[3] for r in beta_rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("IKS 3D Z=8 (griglia uniforme): compromesso in β",
                 fontweight="bold", fontsize=12)

    ax = axes[0]
    l1 = ax.loglog(betas, Vrmse, "o-", color="#dc2626", label="RMSE potenziale [MeV]")
    ax.set_xlabel("β (regolarizzazione)")
    ax.set_ylabel("RMSE V [MeV]", color="#dc2626")
    ax.tick_params(axis="y", labelcolor="#dc2626")
    ax.grid(alpha=0.4, which="both")
    ax2 = ax.twinx()
    l2 = ax2.loglog(betas, np.array(errs) + 1e-6, "s--", color="#ea580c",
                    label="|errore T$_α$| [%]")
    ax2.set_ylabel("|errore T$_α$| [%]", color="#ea580c")
    ax2.tick_params(axis="y", labelcolor="#ea580c")
    ax.set_title("Errore cinetico e potenziale vs β")
    lines = l1 + l2
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="best")

    ax = axes[1]
    ax.axhline(E_ref, color="#16a34a", lw=2, ls="--",
               label=f"T$_α$ analitico = {E_ref:.1f} MeV")
    ax.semilogx(betas, Tas, "o-", color="#ea580c", label="T$_α$(β)")
    ax.set_xlabel("β (regolarizzazione)"); ax.set_ylabel("T$_α$ [MeV]")
    ax.set_title("Cinetica ricostruita vs β")
    ax.legend(fontsize=8); ax.grid(alpha=0.4, which="both")

    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figura β-scan (uniforme) salvata: {fig_path}")


def n_beta_scan_u(A, n_part, n_list, L_list, beta_list, E_ref):
    """2D scan (n, beta) on the uniform grid: RMSE of the potential for each cell.
    n_list and L_list must have the same length (a consistent box for each n)."""
    nN, nB = len(n_list), len(beta_list)
    Vrmse = np.full((nN, nB), np.nan)
    Terr  = np.full((nN, nB), np.nan)
    for i, (n, L) in enumerate(zip(n_list, L_list)):
        for j, beta in enumerate(beta_list):
            r = run_iks_uniform(A, n_part, n, L, beta)
            Vrmse[i, j] = r['pot_err']['rmse_pesato']
            Terr[i, j]  = abs(100.0 * (r['T_alpha'] - E_ref) / E_ref)
            print(f"    n={n:>3}, β={beta:>7.0e}:  RMSE V={Vrmse[i,j]:8.3f} MeV, "
                  f"|err T_α|={Terr[i,j]:6.3f}%")
    return Vrmse, Terr


def make_nbeta_figure_u(n_list, beta_list, Vrmse, fig_path):
    """Heat map of the RMSE of the potential on the (n, beta) grid, uniform grid."""
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    data = np.clip(Vrmse, 1e-3, None)
    im = ax.imshow(data, origin="lower", aspect="auto",
                   norm=LogNorm(vmin=np.nanmin(data), vmax=np.nanmax(data)),
                   cmap="viridis_r")
    ax.set_xticks(range(len(beta_list)))
    ax.set_xticklabels([f"{b:.0e}" for b in beta_list])
    ax.set_yticks(range(len(n_list)))
    ax.set_yticklabels([str(n) for n in n_list])
    ax.set_xlabel("β (regolarizzazione)"); ax.set_ylabel("n (punti per asse)")
    ax.set_title("RMSE potenziale [MeV] su griglia (n, β) — uniforme")
    for i in range(len(n_list)):
        for j in range(len(beta_list)):
            if np.isfinite(Vrmse[i, j]):
                ax.text(j, i, f"{Vrmse[i,j]:.2g}", ha="center", va="center",
                        fontsize=8, color="white")
    fig.colorbar(im, ax=ax, label="RMSE V [MeV] (scala log)")
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Heatmap (n,β) uniforme salvata: {fig_path}")


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║     TEST IKS 3D — Z=8 : GRIGLIA UNIFORME (FinDiff)                  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    A = 8; n_part = 8
    orbitals0 = list(getOrbitalSet(n_part, uncoupledHObasis()))
    print("  Orbitali: " + ", ".join(
        f"({o.nx},{o.ny},{o.nz}) occ={o.occupation}" for o in orbitals0))

    E_ref = kinetic_reference(A, orbitals0)

    # ── Scan in beta (run with: ... --beta-scan) ─────────────────────────────
    # One complete IKS run per beta; the diagnostics establish whether the
    # solution is physical (T_alpha >= T_vW, ratio ~ 1) or a sub-grid artefact.
    if "--beta-scan" in sys.argv:
        print(f"\n  SCAN in β a n={N_GRID}, L={L_BOX}  (E_kin rif = {E_ref:.2f} MeV)")
        print("  Un run di ottimizzazione completo per ogni β.\n")
        beta_list = [0.1, 0.3, 1.0, 3.0, 10.0]
        print(f"  {'β':>6}  {'T_α':>9}  {'err':>8}  {'T_dir':>9}  "
              f"{'ratio':>7}  {'fisica?':>9}")
        for beta in beta_list:
            r = run_iks_uniform(A, n_part, N_GRID, L_BOX, beta)
            verdict = "SI" if r['physical'] else "NO (art.)"
            print(f"  {beta:>6.1f}  {r['T_alpha']:>9.2f}  "
                  f"{100*(r['T_alpha']-E_ref)/E_ref:>+7.1f}%  {r['T_direct']:>9.2f}  "
                  f"{r['ratio']:>7.2f}  {verdict:>9}")
        print("\n  Verdetto: il β buono è il più piccolo con 'fisica?=SI' e T_α vicino")
        print(f"  a {E_ref:.0f} MeV. Se nessun β dà una soluzione fisica, la")
        print("  regolarizzazione da sola non basta e serve ancorare il punto di partenza.")
        print("\n  Fatto.\n")
        return

    # ── Accuracy-oriented scan in beta (comparison with the Lagrange mesh) ───
    if "--beta-scan-acc" in sys.argv:
        print(f"\n  SCAN in β (accuratezza) a n={N_GRID}, L={L_BOX}")
        print("  Un run completo per ogni β; RMSE del potenziale vs armonico.\n")
        beta_list = [1.0, 0.1, 0.01, 0.001, 1e-4]
        brows = beta_scan_acc(A, n_part, N_GRID, L_BOX, beta_list, E_ref)
        print(f"  {'β':>8}  {'T_α [MeV]':>12}  {'err T_α':>9}  "
              f"{'RMSE V':>9}  {'max|ΔV|':>9}  {'ratio':>7}")
        for (beta, Ta, errp, vrmse, vmax, ratio) in brows:
            print(f"  {beta:>8.0e}  {Ta:>12.4f}  {errp:>+8.2f}%  "
                  f"{vrmse:>9.2f}  {vmax:>9.2f}  {ratio:>7.2f}")
        make_beta_figure_u(brows, E_ref,
                           os.path.join(OUTPUT_DIR, "IKS_8particles_uniform_beta_scan.pdf"))
        print("\n  NB: su griglia uniforme fine la soluzione può essere spuria")
        print("  (ratio≫1): l'RMSE riflette la soluzione trovata. Confrontare con")
        print("  la Lagrange mesh, dove la soluzione resta fisica.")
        print("\n  Fatto.\n")
        return

    # ── 2D scan (n, beta) on the uniform grid ────────────────────────────────
    if "--Nbeta-scan" in sys.argv:
        print(f"\n  SCAN 2D (n, β) su griglia uniforme (un run completo per cella).\n")
        n_grid    = [21, 31, 41]
        L_grid    = [7.0, 8.0, 9.0]
        beta_grid = [0.1, 0.01, 0.001]
        Vrmse, Terr = n_beta_scan_u(A, n_part, n_grid, L_grid, beta_grid, E_ref)
        make_nbeta_figure_u(n_grid, beta_grid, Vrmse,
                            os.path.join(OUTPUT_DIR, "IKS_8particles_uniform_Nbeta_scan.pdf"))
        print("\n  Confrontare la heatmap con quella della Lagrange mesh.")
        print("\n  Fatto.\n")
        return

    # Default mode: convergence scan on (points per axis, box half-width) + full run
    grids = [(21, 7.0), (31, 8.0), (41, 9.0), (51, 9.5), (61, 10.0)]
    E_ref, scan_rows = convergence_scan_uniform(A, n_part, grids, REG)
    print(f"\n  E_kin analitico = {E_ref:.4f} MeV  (= 9ℏω)\n")
    print("  Convergenza cinetica T_α (griglia uniforme, densità target):")
    print(f"  {'n':>4}  {'L[fm]':>6}  {'h[fm]':>6}  {'T_α [MeV]':>12}  {'T_β [MeV]':>12}  {'err':>8}")
    for (n, L, h, Ta, Tb, errp) in scan_rows:
        print(f"  {n:>4}  {L:>6.1f}  {h:>6.3f}  {Ta:>12.4f}  {Tb:>12.4f}  {errp:>7.2f}%")

    print(f"\n  Run IKS completo (uniforme, n={N_GRID}, L={L_BOX}, h={(2*L_BOX/(N_GRID-1)):.3f} fm) …")
    res = run_iks_uniform(A, n_part, N_GRID, L_BOX, REG)
    print(f"\n  Status         : {res['status']}")
    print(f"  Obiettivo tot  : {res['obj']:.4f} MeV  (= T_α + T_β)")
    print(f"  T_α (cinetica) : {res['T_alpha']:.4f} MeV   (rif. {E_ref:.4f}, "
          f"Δ {100*(res['T_alpha']-E_ref)/E_ref:+.2f}%)")
    print(f"  T_β (regolar.) : {res['T_beta']:.4f} MeV")
    print(f"  T_vW (bound)   : {res['T_vW']:.4f} MeV   (lower bound rigoroso)")
    print(f"  T diretta      : {res['T_direct']:.4f} MeV   (α∫|∇φ|², controllo)")
    print(f"  Consistenza T_direct/T_α : {res['ratio']:.3f}  (≈1 → fisica)")
    if res['physical']:
        print(f"  ✓ soluzione FISICA (forme della cinetica concordi, T_α ≥ T_vW)")
    else:
        print(f"  ✗ ATTENZIONE: soluzione NON fisica")
        print(f"    T_direct/T_α = {res['ratio']:.2f} (lontano da 1) e/o T_α < T_vW")
        print(f"    → artefatto sub-griglia della non-unicità. Vedi teoria_IKS.tex.")
    print(f"  Δρ L2          : {res['density_err']:.3e}")
    print(f"  Norma ρ_calc   : {res['norm_calc']:.4f}  (atteso {n_part})")
    print(f"  Ortogonalità ⟨φ_0|φ_1⟩_ρ : {res['orto']:.3e}  (atteso 0)")
    pe = res['pot_err']
    print(f"  Errore potenziale (vs armonico, regione fisica):")
    print(f"    RMSE uniforme : {pe['rmse_uniforme']:.3f} MeV  ({pe['n_nodi']} nodi)")
    print(f"    RMSE pesato ρ : {pe['rmse_pesato']:.3f} MeV")
    print(f"    max |ΔV|      : {pe['max_abs']:.3f} MeV")

    make_figure(E_ref, scan_rows, res,
                os.path.join(OUTPUT_DIR, "IKS_8particles_uniform.pdf"))
    print("\n  Confronto con Gauss-Hermite: vedi test_IKS_3D_4part.py")
    print("\n  Fatto.\n")


if __name__ == "__main__":
    main()