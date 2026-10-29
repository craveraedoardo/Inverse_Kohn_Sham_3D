"""
reffree_estimator.py
====================
Estimate of the behaviour of the error on the potential, and location of its
minimum in the mesh scale s, WITHOUT knowing the true potential.
Thesis: Section 5.6 (Figs. 5.16 and 5.17).

MOTIVATION
  For the oscillator test systems the exact potential is the harmonic one,
  so the true RMSE can be computed. In a realistic case (e.g. a Skyrme
  density of 16O, as in Liardi et al.) V_true does not exist: without a
  reference-free indicator s cannot be chosen. This module builds such
  indicators and — on the cases where the truth IS known — CALIBRATES them
  against the true RMSE. If the minimum of the indicator falls where the true
  one does, the indicator can be transferred to the unknown case.

INPUT
  The potential profiles V(x) along the x axis saved by h_scan_gh_Z2.py
  (h_scan_out_Z2/V_profiles.npz) and, for the calibration, the true errors in
  h_scan_out_Z2/H1_rmse_vs_s.csv.

THE INDICATORS

  (I1) NEIGHBOUR DIFFERENCES ("disagreement", thesis Fig. 5.16)
       D_i = ||V(s_i+1) - V(s_i)||_rho
       Writing V(s) = V_true + e(s), D_i = ||e_i+1 - e_i||, so by the triangle
       inequality max(E_i, E_i+1) >= D_i/2: a rigorous LOWER BOUND on the true
       error. Where D is large, the error is certainly large.
       LIMIT: the converse does not hold. A small D does NOT imply a small E
       (two points may err in the same way). In particular in the spurious
       arm, where the reconstruction collapses to machine noise, both
       neighbours are ~0 and D drops -> false minimum. I1 must therefore
       ALWAYS be combined with the physicality filter (coverage + live nodes).

  (I2) DISTANCE FROM AN EXTRAPOLATED PSEUDO-TRUTH  ||V(s) - V*||_rho
       The s grid is logarithmic => the ratio r = s_i+1/s_i is constant, the
       ideal setting to estimate the local order of convergence from three
       consecutive points,
           p = ln(D_1/D_2) / ln(r),   D_1 = ||V1-V2||, D_2 = ||V2-V3||,
       and to build the Richardson extrapolation
           V* = V3 + (V3 - V2)/(r^p - 1).
       KNOWN BIAS: it is artificially small near the points used to build V*.
       In practice p turns out to be scattered, because s is a POSITIONING
       parameter, not a refinement one (e(s) has an interior minimum and no
       limit to extrapolate to): I2 is reported for completeness only.

  (I3) OFF-NODE DENSITY RESIDUAL (idea, NOT implemented here)
       Evaluating the reconstructed orbitals between the nodes and comparing
       with the analytic density would expose the representation error that
       the pointwise constraints hide. It requires the reconstructed orbitals
       (results['u']), which are not saved in V_profiles.npz.

  (I4) ROUGHNESS OF THE POTENTIAL  R = ||d2V/dx2||_rho (thesis Fig. 5.17)
       Captures the noise amplification typical of ill-posed problems. It
       does not see a smooth systematic bias, so it is used TOGETHER with I1:
       the combined estimate is the geometric mean of the I1 and I4 minima.

INTERPOLATION (a necessary step)
  At different s the nodes sit at different physical positions: the V(s)
  cannot be subtracted element by element. All profiles are brought onto a
  common physical grid |x| < 3b with barycentric Lagrange interpolation on the
  GH nodes, stable even at large N (Berrut & Trefethen, SIAM Rev. 46 (2004) 501).

Sub-grid refinement: the minima are located with a parabola in
(log s, log y) through the three points around the discrete minimum.

Usage
-----
    python reffree_estimator.py                      # uses V_profiles.npz + CSV
    python reffree_estimator.py --profiles P.npz --truth H1_rmse_vs_s.csv
    python reffree_estimator.py --demo               # self-contained synthetic test

Output (h_scan_out_Z2/): reffree_N<N>.pdf, reffree_overlay_I1.pdf,
reffree_overlay_I4.pdf.
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the modules in src/ importable (not strictly needed by this script).
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "src"), HERE):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


# ──────────────────────────────────────────────────────────────────────────────
# Barycentric Lagrange interpolation on the GH nodes
# ──────────────────────────────────────────────────────────────────────────────
def barycentric_weights(x):
    """Barycentric weights w_j = 1/prod_{k!=j}(x_j - x_k), computed in log space to
    avoid overflow at large N (same strategy as derivative_matrix)."""
    n = len(x)
    diff = x[:, None] - x[None, :]
    np.fill_diagonal(diff, 1.0)
    sign = np.prod(np.sign(diff), axis=1)
    logabs = np.sum(np.log(np.abs(diff)), axis=1)
    logabs -= logabs.min()          # normalise: the weights matter only up to a scale
    return sign / np.exp(logabs)


def barycentric_interp(x_nodes, f_nodes, x_new):
    """Second-form barycentric interpolation. NaN values (masked nodes) are
    excluded from the support; points coinciding with nodes are handled exactly."""
    x_nodes = np.asarray(x_nodes, float)
    f_nodes = np.asarray(f_nodes, float)
    ok = np.isfinite(f_nodes)
    if ok.sum() < 2:
        return np.full_like(np.asarray(x_new, float), np.nan)
    xn, fn = x_nodes[ok], f_nodes[ok]
    w = barycentric_weights(xn)
    x_new = np.asarray(x_new, float)
    num = np.zeros_like(x_new)
    den = np.zeros_like(x_new)
    exact = np.full(x_new.shape, -1, dtype=int)
    for j, (xj, fj, wj) in enumerate(zip(xn, fn, w)):
        d = x_new - xj
        hit = np.abs(d) < 1e-14
        exact[hit] = j
        with np.errstate(divide="ignore", invalid="ignore"):
            t = wj / d
        t[hit] = 0.0
        num += t * fj
        den += t
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    m = exact >= 0
    if m.any():
        out[m] = fn[exact[m]]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Density-weighted norms
# ──────────────────────────────────────────────────────────────────────────────
def rho_weight_1d(xc, b, n_part=2):
    """1D weight rho(x) along the axis (proxy for the weighted norm): the HO
    Gaussian exp(-x^2/b^2) normalised to its maximum (exact for A = 2).
    ``n_part`` is unused."""
    w = np.exp(-(xc / b) ** 2)
    return w / w.max()


def wnorm(dV, w):
    """Weighted L2 norm (RMS), robust to NaN."""
    ok = np.isfinite(dV) & np.isfinite(w)
    if ok.sum() == 0:
        return np.nan
    return float(np.sqrt(np.sum(w[ok] * dV[ok] ** 2) / np.sum(w[ok])))


# ──────────────────────────────────────────────────────────────────────────────
# Indicators
# ──────────────────────────────────────────────────────────────────────────────
def align_profiles(profiles, xc):
    """Interpolate all profiles onto the common grid xc. Returns (s, V) sorted by
    increasing s."""
    ss, VV = [], []
    for p in sorted(profiles, key=lambda q: q["s"]):
        V = barycentric_interp(p["x"], p["Vx"], xc)
        ss.append(p["s"]); VV.append(V)
    return np.array(ss), np.array(VV)


def I1_neighbor(ss, VV, w):
    """Neighbour differences (I1), assigned to the GEOMETRIC MIDPOINT of the interval.

    CORRECTED BIAS (found on the real A = 2 data): D_i = ||V(s_i+1) - V(s_i)||
    is a property of the INTERVAL (s_i, s_i+1), not of its left end. Assigning
    it to s_i put the minimum systematically one grid step below the true one
    (0.631 instead of 0.794, with r = 1.2589: 0.631 * 1.2589 = 0.794 exactly).
    On a logarithmic grid the natural representative of the interval is the
    geometric mean sqrt(s_i * s_i+1).

    Returns (s_mid, D, bound) of length len(ss) - 1: the interval grid is
    shifted by half a step with respect to the points.
    """
    n = len(ss)
    s_mid = np.sqrt(ss[:-1] * ss[1:])          # geometric mean (midpoint in log s)
    D = np.array([wnorm(VV[i + 1] - VV[i], w) for i in range(n - 1)])
    return s_mid, D, D / 2.0


def parabolic_min_log(sx, y, mask=None):
    """Sub-grid minimum via a parabola in (log s, log y) through the 3 points
    around the discrete minimum. With r = 1.2589 (26% steps) the gain is real:
    the true minimum typically falls between two grid points.

    Returns (s_min_subgrid, y_min_interp, s_min_discrete). If the parabola is
    degenerate or the minimum is at the edge, it falls back to the discrete one.
    """
    sx = np.asarray(sx, float); y = np.asarray(y, float).copy()
    if mask is not None:
        y = np.where(mask, y, np.nan)
    ok = np.isfinite(y) & np.isfinite(sx) & (y > 0)
    if ok.sum() < 3:
        if ok.sum() == 0:
            return np.nan, np.nan, np.nan
        i = int(np.nanargmin(np.where(ok, y, np.inf)))
        return sx[i], y[i], sx[i]

    idx = np.where(ok)[0]
    i = idx[int(np.argmin(y[idx]))]
    s_disc = sx[i]
    # immediately adjacent valid neighbours
    left  = [j for j in idx if j < i]
    right = [j for j in idx if j > i]
    if not left or not right:
        return s_disc, y[i], s_disc       # minimum at the edge: no parabola
    a, c = left[-1], right[0]
    X = np.log(sx[[a, i, c]]); Y = np.log(y[[a, i, c]])
    # parabola Y = A X^2 + B X + C ; minimum at X* = -B/(2A), requires A > 0
    try:
        A, B, _ = np.polyfit(X, Y, 2)
    except Exception:
        return s_disc, y[i], s_disc
    if not np.isfinite(A) or A <= 0:
        return s_disc, y[i], s_disc
    Xstar = -B / (2 * A)
    # accept only if it falls INSIDE the bracket [X_a, X_c]
    if not (X[0] <= Xstar <= X[2]):
        return s_disc, y[i], s_disc
    s_star = float(np.exp(Xstar))
    y_star = float(np.exp(np.polyval([A, B, np.polyfit(X, Y, 2)[2]], Xstar)))
    return s_star, y_star, s_disc


def local_order(ss, VV, w):
    """Local order of convergence p from consecutive triples on the logarithmic
    grid (constant r). Returns (p for each centred triple, r)."""
    r = ss[1] / ss[0]
    p = np.full(len(ss), np.nan)
    for i in range(1, len(ss) - 1):
        D1 = wnorm(VV[i] - VV[i - 1], w)
        D2 = wnorm(VV[i + 1] - VV[i], w)
        if np.isfinite(D1) and np.isfinite(D2) and D2 > 0 and D1 > 0:
            with np.errstate(invalid="ignore"):
                p[i] = np.log(D1 / D2) / np.log(r)
    return p, r


def I2_extrapolated(ss, VV, w, use_idx=None):
    """I2: Richardson pseudo-truth V* and distance from it.
    use_idx: centre i of the triple (i-1, i, i+1) to use; default = the one with
    the most stable p. Returns (E, p, use_idx)."""
    p_loc, r = local_order(ss, VV, w)
    if use_idx is None:
        cand = [i for i in range(1, len(ss) - 1)
                if np.isfinite(p_loc[i]) and 0.2 < p_loc[i] < 12]
        if not cand:
            return np.full(len(ss), np.nan), np.nan, None
        # most stable triple: local p closest to its neighbours
        def stab(i):
            nb = [p_loc[j] for j in (i - 1, i + 1)
                  if 0 <= j < len(p_loc) and np.isfinite(p_loc[j])]
            return np.mean([abs(p_loc[i] - q) for q in nb]) if nb else 1e9
        use_idx = min(cand, key=stab)
    p = p_loc[use_idx]
    if not np.isfinite(p) or p <= 0:
        return np.full(len(ss), np.nan), np.nan, use_idx
    # Richardson on the last two points of the triple
    V2, V3 = VV[use_idx], VV[use_idx + 1]
    Vstar = V3 + (V3 - V2) / (r ** p - 1.0)
    E = np.array([wnorm(V - Vstar, w) for V in VV])
    return E, p, use_idx


def I4_roughness(ss, VV, xc, w):
    """I4, raw roughness: weighted ||d2V/dx2||, by finite differences on the
    common grid (uniform by construction)."""
    h = xc[1] - xc[0]
    R = np.full(len(ss), np.nan)
    for i, V in enumerate(VV):
        d2 = np.full_like(V, np.nan)
        d2[1:-1] = (V[2:] - 2 * V[1:-1] + V[:-2]) / h ** 2
        R[i] = wnorm(d2, w)
    return R


def I4_excess(R, mask=None):
    """EXCESS roughness above the physical floor.

    WARNING — DEGENERATE INDICATOR, DIAGNOSTIC USE ONLY.
    It QUANTIFIES the floor; it is not meant to locate the minimum.

    The floor is real: R(s) never drops below ~25.53 MeV/fm^2 for any s or N,
    and that value is exactly 2*nuclearNu_x(A=2) = 25.535, i.e. the EXACT
    second derivative of the true harmonic potential (= 2 nu_x). R measures
    the TOTAL curvature, R^2 ~ ||V''_true||^2 + ||V''_err||^2, so it cannot go
    below the physical curvature (thesis Fig. 5.17).

    BUT subtracting the floor does NOT give a better indicator, because
    R_floor is estimated as min(R) over the valid points, so that
        R_exc = sqrt(max(R^2 - R_floor^2, 0))
    is EXACTLY ZERO at the argmin of R. The excess is therefore not
    independent of R — it is a monotonic transformation of it that
    degenerates at its own minimum — and the best point is then discarded by
    the y > 0 filter of the parabolic refinement, which falls back on a
    neighbour. Effect measured on the real A = 2 data (max deviation from the
    true minimum):
        raw R      9.1%   (std 0.039)   <-- BEST
        excess R  25.9%   (std 0.166)
    The primary indicator is therefore the RAW R, and the combination with I1
    uses the raw R. The excess is reported only to document the floor.

    Returns (exc, R_floor).
    """
    R = np.asarray(R, float)
    Rm = np.where(mask, R, np.nan) if mask is not None else R
    if not np.any(np.isfinite(Rm)):
        return np.full_like(R, np.nan), np.nan
    R_floor = float(np.nanmin(Rm))
    exc = np.sqrt(np.maximum(R ** 2 - R_floor ** 2, 0.0))
    return exc, R_floor


# ──────────────────────────────────────────────────────────────────────────────
# Calibration against the known truth
# ──────────────────────────────────────────────────────────────────────────────
def argmin_s(ss, y, mask=None):
    """Discrete argmin of y over s (masked points excluded). Used by report()."""
    y = np.asarray(y, float).copy()
    if mask is not None:
        y[~mask] = np.nan
    if not np.any(np.isfinite(y)):
        return np.nan, np.nan
    i = int(np.nanargmin(y))
    return ss[i], y[i]


def report(ss, ind, truth=None, mask=None):
    """Compare the discrete minimum of each indicator with the true one
    (helper kept for interactive use; analyze() prints its own tables)."""
    print("\n  " + "-" * 66)
    print(f"  {'indicatore':<26} {'s del minimo':>14} {'valore':>14}")
    print("  " + "-" * 66)
    rows = []
    for name, y in ind.items():
        s_m, v_m = argmin_s(ss, y, mask)
        rows.append((name, s_m, v_m))
        print(f"  {name:<26} {s_m:>14.3f} {v_m:>14.3e}")
    if truth is not None:
        s_t, v_t = argmin_s(ss, truth, mask)
        print("  " + "-" * 66)
        print(f"  {'RMSE VERO (riferimento)':<26} {s_t:>14.3f} {v_t:>14.3e}")
        print("  " + "-" * 66)
        print("\n  Calibrazione (scarto del minimo rispetto al vero):")
        for name, s_m, _ in rows:
            if np.isfinite(s_m) and np.isfinite(s_t):
                fac = s_m / s_t
                ok = "OK" if 0.7 <= fac <= 1.4 else "  scostato"
                print(f"    {name:<26} s_ind/s_vero = {fac:5.2f}   {ok}")
    return rows


def figure(ss, ind, truth, mask, path, title, s_mid=None, s_marks=None,
           xlim=None, ylim=None):
    """Figure for a single N: indicators vs s with the true error (left), and
    indicator-vs-true-error correlation (right).

    COMPARABILITY ACROSS N: the points excluded by the physicality mask are not
    removed (NaN) but drawn faintly, so every figure shows the COMPLETE
    behaviour over the scanned s range and, with common xlim/ylim, the panels
    of the different N can be overlaid. The minima are still computed ONLY on
    the physical points: this changes the rendering, not the estimate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    fig.suptitle(title, fontweight="bold")
    ax = axes[0]
    styles = {"I1 vicini": ("o-", "#2563eb"),
              "I1 bound D/2": ("^--", "#60a5fa"),
              "I2 da V* estrap.": ("s-", "#16a34a"),
              "I4 rugosita'": ("d-", "#ea580c"),
              "I4 eccesso*": ("d:", "#fbbf24")}
    on_mid = {"I1 vicini", "I1 bound D/2"}
    mask_mid = None if mask is None else (mask[:-1] & mask[1:])

    for name, y in ind.items():
        st, c = styles.get(name, ("o-", "#888"))
        xg = s_mid if (name in on_mid and s_mid is not None) else ss
        mg = mask_mid if (name in on_mid and s_mid is not None) else mask
        yy = np.abs(np.asarray(y, float)) + 1e-16
        if len(xg) != len(yy):
            continue
        if mg is None or len(mg) != len(yy):
            ax.loglog(xg, yy, st, color=c, ms=5, label=name)
            continue
        # faint line = complete behaviour (non-physical points included)
        ax.loglog(xg, yy, st[1:] if len(st) > 1 else "-", color=c, lw=0.9,
                  alpha=0.28)
        ax.loglog(xg[~mg], yy[~mg], st[0], color=c, ms=4, mfc="none",
                  alpha=0.45, ls="none")
        # full line = physical points, the ones that determine the estimate
        ax.loglog(xg[mg], yy[mg], st, color=c, ms=5, label=name)

    if truth is not None:
        tt = np.asarray(truth, float) + 1e-16
        ax.loglog(ss, tt, "-", color="k", lw=0.9, alpha=0.28)
        if mask is not None:
            ax.loglog(ss[~mask], tt[~mask], "*", color="k", ms=6, mfc="none",
                      alpha=0.45, ls="none")
            ax.loglog(ss[mask], tt[mask], "k*-", ms=9, lw=2, label="RMSE VERO")
        else:
            ax.loglog(ss, tt, "k*-", ms=9, lw=2, label="RMSE VERO")
        s_t, _, _ = parabolic_min_log(ss, truth, mask)
        ax.axvline(s_t, color="k", ls=":", lw=1.6)
    if s_marks:
        cm = {"I1": "#2563eb", "I4": "#ea580c", "comb": "#7c3aed"}
        for k, v in s_marks.items():
            if np.isfinite(v):
                ax.axvline(v, color=cm.get(k, "#888"), ls="--", lw=1.1,
                           alpha=0.8)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("s"); ax.set_ylabel("indicatore / errore [MeV]")
    ax.set_title("Indicatori reference-free vs errore vero\n"
                 "(tenue = punti non fisici; tratteggi = minimi stimati)",
                 fontsize=9)
    ax.grid(alpha=0.4, which="both"); ax.legend(fontsize=7)

    ax = axes[1]
    if truth is not None:
        tt = np.asarray(truth, float)
        for name, y in ind.items():
            if name in on_mid:
                continue
            st, c = styles.get(name, ("o-", "#888"))
            yy = np.asarray(y, float)
            if len(yy) != len(tt):
                continue
            ok = np.isfinite(yy) & np.isfinite(tt)
            if mask is not None:
                ok &= mask
            if ok.sum() >= 2:
                ax.loglog(tt[ok] + 1e-16, np.abs(yy[ok]) + 1e-16, st[0],
                          color=c, ms=6, ls="none", label=name)
        fin = tt[np.isfinite(tt)]
        if fin.size:
            lim = [fin.min() + 1e-16, fin.max() + 1e-16]
            ax.plot(lim, lim, "k--", lw=1, label="identita'")
        ax.set_xlabel("RMSE vero [MeV]"); ax.set_ylabel("indicatore")
        ax.set_title("Correlazione indicatore-errore\n"
                     "(piu' vicino alla diagonale = meglio calibrato)",
                     fontsize=9)
        ax.grid(alpha=0.4, which="both"); ax.legend(fontsize=7)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {path}")


def figure_overlay(results, path, key="I1", title=None):
    """Overlay of all N in a single panel, on common axes (thesis Figs. 5.16-5.17).

    key: which indicator to overlay ("I1" or "I4"). The right panel shows the
    position of the estimated minimum vs the true one, as a function of N.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    fig.suptitle(title or f"Z=2: confronto fra N — indicatore {key}",
                 fontweight="bold")
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(results)))
    ax = axes[0]
    for (res, c) in zip(results, cmap):
        N = res["N"]
        if key == "I1":
            xg, yy, mg = res["s_mid"], res["D"], res["mask_mid"]
        else:
            xg, yy, mg = res["ss"], res["R"], res["mask"]
        yy = np.abs(np.asarray(yy, float)) + 1e-16
        ax.loglog(xg, yy, "-", color=c, lw=0.9, alpha=0.3)
        if mg is not None and len(mg) == len(yy):
            ax.loglog(xg[mg], yy[mg], "o-", color=c, ms=5, label=f"N={N}")
        else:
            ax.loglog(xg, yy, "o-", color=c, ms=5, label=f"N={N}")
        sm = res["s_" + key]
        if np.isfinite(sm):
            ax.axvline(sm, color=c, ls="--", lw=1.0, alpha=0.7)
    st = [r["s_true"] for r in results if np.isfinite(r.get("s_true", np.nan))]
    if st:
        ax.axvline(float(np.mean(st)), color="k", ls=":", lw=1.8,
                   label="vero (medio)")
    ax.set_xlabel("s"); ax.set_ylabel(f"indicatore {key}")
    ax.set_title("Tutti gli N su assi comuni\n(tenue = punti non fisici)",
                 fontsize=9)
    ax.grid(alpha=0.4, which="both"); ax.legend(fontsize=7)

    ax = axes[1]
    Ns = [r["N"] for r in results]
    for k, c, lab in (("I1", "#2563eb", "I1 vicini"),
                      ("I4", "#ea580c", "I4 rugosita'"),
                      ("comb", "#7c3aed", "combinato")):
        v = [r.get("s_" + k, np.nan) for r in results]
        ax.plot(Ns, v, "o-", color=c, ms=6, label=lab)
    v = [r.get("s_true", np.nan) for r in results]
    ax.plot(Ns, v, "k*--", ms=11, lw=1.6, label="vero")
    ax.set_xlabel("N"); ax.set_ylabel("s del minimo")
    ax.set_title("Stima vs verita', per N", fontsize=9)
    ax.grid(alpha=0.4); ax.legend(fontsize=7)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {path}")

def load_profiles(npz_path):
    """Read the profiles saved by h_scan_gh_Z2.py: list of dict(N, s, x, Vx)."""
    d = np.load(npz_path, allow_pickle=True)
    idx = sorted({int(k.split("_")[0][3:]) for k in d.files})
    out = []
    for i in idx:
        out.append(dict(N=int(d[f"run{i}_N"]), s=float(d[f"run{i}_s"]),
                        x=np.asarray(d[f"run{i}_x"], float),
                        Vx=np.asarray(d[f"run{i}_Vx"], float)))
    return out


def load_truth_csv(path):
    """Read H1_rmse_vs_s.csv -> dict (N, round(s, 3)) -> (rmse_w, iters)."""
    import csv
    t = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            N = int(float(row["N"])); s = float(row["s"])
            t[(N, round(s, 3))] = (float(row["rmse_w"]),
                                   float(row.get("iters", "nan")))
    return t


def compute_for_N(sub, N, truth_map, b, n_sample):
    """Compute indicators and minima for a single N (no plotting).

    The physicality mask uses the true-error file: finite RMSE, non-zero
    iterations and RMSE above 1e-10 (machine-noise points are spurious, see
    h_scan_gh_Z2._is_physical). The combined estimate is sqrt(s_I1 * s_I4).
    """
    xc = np.linspace(-3 * b, 3 * b, n_sample)
    w = rho_weight_1d(xc, b)
    ss, VV = align_profiles(sub, xc)

    s_mid, D, bound = I1_neighbor(ss, VV, w)
    E2, p_used, idx = I2_extrapolated(ss, VV, w)
    R = I4_roughness(ss, VV, xc, w)
    p_loc, r = local_order(ss, VV, w)

    truth = mask = None
    if truth_map:
        truth = np.array([truth_map.get((N, round(s, 3)), (np.nan,))[0]
                          for s in ss])
        iters = np.array([truth_map.get((N, round(s, 3)),
                                        (np.nan, np.nan))[1] for s in ss])
        mask = np.isfinite(truth) & (iters != 0) & (truth > 1e-10)
    mask_mid = None if mask is None else (mask[:-1] & mask[1:])
    R_exc, R_floor = I4_excess(R, mask)

    ind = {"I1 vicini": D, "I1 bound D/2": bound,
           "I2 da V* estrap.": E2,
           "I4 rugosita'": R, "I4 eccesso*": R_exc}

    s_I1, _, d_I1 = parabolic_min_log(s_mid, D, mask_mid)
    s_I4, _, d_I4 = parabolic_min_log(ss, R, mask)
    s_ex, _, d_ex = parabolic_min_log(ss, R_exc, mask)
    s_I2, _, d_I2 = parabolic_min_log(ss, E2, mask)
    s_cb = (float(np.sqrt(s_I1 * s_I4))
            if np.isfinite(s_I1) and np.isfinite(s_I4) else np.nan)
    s_tr = (parabolic_min_log(ss, truth, mask)[0]
            if truth is not None else np.nan)
    d_tr = (parabolic_min_log(ss, truth, mask)[2]
            if truth is not None else np.nan)

    return dict(N=N, ss=ss, s_mid=s_mid, D=D, R=R, R_exc=R_exc, E2=E2,
                ind=ind, truth=truth, mask=mask, mask_mid=mask_mid,
                R_floor=R_floor, p_loc=p_loc, r=r,
                s_I1=s_I1, s_I4=s_I4, s_ex=s_ex, s_I2=s_I2, s_comb=s_cb,
                s_true=s_tr,
                disc=dict(I1=d_I1, I4=d_I4, ex=d_ex, I2=d_I2, true=d_tr))


def common_limits(results, pad_x=1.05, pad_y=6.0):
    """Axis limits common to all N, so that the panels can be overlaid."""
    xs, ys = [], []
    for res in results:
        xs += [np.nanmin(res["ss"]), np.nanmax(res["ss"])]
        for y in res["ind"].values():
            y = np.abs(np.asarray(y, float))
            y = y[np.isfinite(y) & (y > 0)]
            if y.size:
                ys += [y.min(), y.max()]
        if res["truth"] is not None:
            t = res["truth"][np.isfinite(res["truth"]) & (res["truth"] > 0)]
            if t.size:
                ys += [t.min(), t.max()]
    if not xs or not ys:
        return None, None
    return ((min(xs) / pad_x, max(xs) * pad_x),
            (min(ys) / pad_y, max(ys) * pad_y))


def analyze(profiles, truth_map=None, b=1.1289, n_sample=241, out_dir="."):
    """Two passes: (1) compute everything, (2) plot with axes COMMON to all N.

    b is the oscillator length b_x(A) in fm (1.1289 for A = 2), which sets the
    common grid |x| < 3b and the density weight.
    """
    Ns = sorted({p["N"] for p in profiles})
    results = []
    for N in Ns:
        sub = [p for p in profiles if p["N"] == N]
        if len(sub) < 3:
            print(f"\n=== N={N}: solo {len(sub)} profili, servono >=3. Salto.")
            continue
        results.append(compute_for_N(sub, N, truth_map, b, n_sample))

    if not results:
        print("Nessun N con abbastanza profili."); return

    xlim, ylim = common_limits(results)
    print(f"\nAssi comuni a tutti gli N:  s in [{xlim[0]:.3f}, {xlim[1]:.3f}]"
          f"   y in [{ylim[0]:.1e}, {ylim[1]:.1e}]")

    for res in results:
        N = res["N"]; ss = res["ss"]
        print(f"\n=== N={N} — {len(ss)} punti in s ===")
        print(f"  rapporto griglia r = {res['r']:.4f}")
        pv = res["p_loc"][np.isfinite(res["p_loc"])]
        if pv.size:
            print(f"  ordine locale p: mediana {np.median(pv):.2f}, "
                  f"spread [{pv.min():.2f}, {pv.max():.2f}]")
            if pv.std() > 1.0:
                print("  ATTENZIONE: p disperso ⇒ Richardson NON applicabile "
                      "(s e' parametro di POSIZIONAMENTO, non di "
                      "raffinamento: e(s) ha minimo interno, nessun limite "
                      "verso cui estrapolare).")
        if np.isfinite(res["R_floor"]):
            print(f"  pavimento di rugosita' R_floor = {res['R_floor']:.3f} "
                  f"MeV/fm^2 (stimato dai dati)")

        d = res["disc"]
        print("\n  " + "-" * 70)
        print(f"  {'indicatore':<20} {'s discreto':>11} {'s sub-griglia':>14}")
        print("  " + "-" * 70)
        for lab, sd, sv in (("I1 vicini", d["I1"], res["s_I1"]),
                            ("I4 rugosita'", d["I4"], res["s_I4"]),
                            ("I4 eccesso*", d["ex"], res["s_ex"]),
                            ("I2 da V*", d["I2"], res["s_I2"])):
            print(f"  {lab:<20} {sd:>11.3f} {sv:>14.3f}")
        print("   (* degenere per costruzione: zero nell'argmin di R — solo "
              "diagnostico, escluso dalla combinazione)")
        print("  " + "-" * 70)
        print(f"  {'COMBINATO I1+I4':<20} {'':>11} {res['s_comb']:>14.3f}")
        if res["truth"] is not None:
            print("  " + "-" * 70)
            print(f"  {'RMSE VERO':<20} {d['true']:>11.3f} "
                  f"{res['s_true']:>14.3f}")
            print("  " + "-" * 70)
            print("\n  Calibrazione (s_indicatore / s_vero, sub-griglia):")
            for lab, sv in (("I1 vicini", res["s_I1"]),
                            ("I4 rugosita'", res["s_I4"]),
                            ("I4 eccesso*", res["s_ex"]),
                            ("I2 da V*", res["s_I2"]),
                            ("COMBINATO I1+I4", res["s_comb"])):
                if np.isfinite(sv) and np.isfinite(res["s_true"]):
                    fac = sv / res["s_true"]
                    tag = "OK" if 0.85 <= fac <= 1.18 else "  scostato"
                    print(f"    {lab:<20} {fac:5.2f}   {tag}")

        figure(ss, res["ind"], res["truth"], res["mask"],
               os.path.join(out_dir, f"reffree_N{N}.pdf"),
               f"Z=2, N={N}: stima dell'errore senza potenziale vero",
               s_mid=res["s_mid"],
               s_marks=dict(I1=res["s_I1"], I4=res["s_I4"],
                            comb=res["s_comb"]),
               xlim=xlim, ylim=ylim)

    figure_overlay(results, os.path.join(out_dir, "reffree_overlay_I1.pdf"),
                   key="I1")
    figure_overlay(results, os.path.join(out_dir, "reffree_overlay_I4.pdf"),
                   key="I4")

    if any(np.isfinite(r.get("s_true", np.nan)) for r in results):
        print("\n  " + "=" * 62)
        print("  RIEPILOGO FRA N (s sub-griglia)")
        print(f"  {'N':>4} {'I1':>8} {'I4':>8} {'comb':>8} {'vero':>8} "
              f"{'comb/vero':>10}")
        for r in results:
            fac = (r["s_comb"] / r["s_true"]
                   if np.isfinite(r["s_comb"]) and np.isfinite(r["s_true"])
                   else np.nan)
            print(f"  {r['N']:>4} {r['s_I1']:>8.3f} {r['s_I4']:>8.3f} "
                  f"{r['s_comb']:>8.3f} {r['s_true']:>8.3f} {fac:>10.3f}")
        facs = np.array([r["s_comb"] / r["s_true"] for r in results
                         if np.isfinite(r["s_comb"])
                         and np.isfinite(r["s_true"])])
        if facs.size:
            print(f"  scarto max {np.abs(facs - 1).max() * 100:.1f}%, "
                  f"dev.std {facs.std():.3f}")
        print("  " + "=" * 62)

def demo():
    """Synthetic test: V(s) with a known error growing as |log(s/0.79)| around
    s* = 0.79. It validates the mechanics of the module without real data."""
    print("\n[DEMO] profili sintetici, errore ~|log(s/0.79)| di ordine noto")
    b = 1.1289
    xc0 = np.linspace(-3 * b, 3 * b, 61)
    ss = np.logspace(np.log10(0.4), np.log10(1.6), 9)
    Vtrue = 0.5 * 41.0 * xc0 ** 2 / b ** 2      # reference harmonic potential
    profiles = []
    for s in ss:
        amp = 10 ** (2.0 * abs(np.log10(s / 0.79)) - 3.0)
        err = amp * np.cos(2.4 * xc0 / b)
        profiles.append(dict(N=20, s=float(s), x=xc0, Vx=Vtrue + err))
    truth = {(20, round(float(s), 3)):
             (float(np.sqrt(np.mean((10 ** (2.0 * abs(np.log10(s / 0.79)) - 3.0)
                                     * np.cos(2.4 * xc0 / b)) ** 2))), 1.0)
             for s in ss}
    analyze(profiles, truth, b=b, out_dir=".")


def main():
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="h_scan_out_Z2/V_profiles.npz")
    ap.add_argument("--truth", default="h_scan_out_Z2/H1_rmse_vs_s.csv")
    ap.add_argument("--b", type=float, default=1.1289,
                    help="b_x(A) in fm (Z=2: 1.1289)")
    ap.add_argument("--out", default="h_scan_out_Z2")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo(); return

    if not os.path.exists(args.profiles):
        print(f"Profili non trovati: {args.profiles}\n"
              "Rilancia h_scan_gh_Z2.py (ora salva V_profiles.npz), "
              "oppure usa --demo per il test sintetico.")
        return
    profiles = load_profiles(args.profiles)
    truth_map = load_truth_csv(args.truth) if os.path.exists(args.truth) else None
    if truth_map is None:
        print("(nessun CSV di verita': calibrazione saltata, "
              "gli indicatori restano comunque calcolabili)")
    os.makedirs(args.out, exist_ok=True)
    analyze(profiles, truth_map, b=args.b, out_dir=args.out)


if __name__ == "__main__":
    main()