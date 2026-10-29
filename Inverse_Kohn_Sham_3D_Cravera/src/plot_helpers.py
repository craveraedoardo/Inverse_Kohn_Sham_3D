"""
plot_helpers.py
===============
Error metrics and plotting conventions shared by all the test scripts.

  * physical_radius : radius within which the density is significant;
  * potential_ylim  : robust y limits for 1D potential cuts (inner region);
  * masked_plane    : 2D slice of a field on the central plane, with a radial
                      mask that hides the tails where V diverges;
  * symmetric_vlim  : symmetric, outlier-robust colour limits for maps of V;
  * potential_error : the error metrics of thesis Section 4.4 (Eq. 4.13).

The IKS potential V = lambda / (rho * w) is reliable only where the density
is significant; further out rho -> 0 and V is dominated by noise. These
helpers restrict plots and statistics to the physical region.
"""
import numpy as np


def physical_radius(rho_line, x, frac=1e-3):
    """Radius within which rho_line >= frac * max(rho_line) along the axis x.

    The smaller of the two sides is returned, so that the window is
    symmetric and fully inside the significant region.
    """
    m = rho_line >= frac * rho_line.max()
    if not np.any(m):
        return float(np.max(np.abs(x)))
    return float(min(abs(x[m].min()), abs(x[m].max())))


def potential_ylim(V_line, x, R, core_frac=0.7, pad=0.3):
    """y limits for a 1D potential cut, based on the inner region |x| <= core_frac*R
    where V is reliable (the diverging edge nodes are ignored).

    Returns (ymin, ymax), or None if no finite value lies in the inner region.
    """
    sel = np.abs(x) <= core_frac * R
    core = V_line[sel][np.isfinite(V_line[sel])]
    if core.size == 0:
        return None
    lo, hi = float(core.min()), float(core.max())
    span = max(hi - lo, 1.0)
    return (lo - pad * span, hi + pad * span)


def masked_plane(arr3d, x, y, ic, R, axis_pair=(0, 1)):
    """Return (field2d, extent) on the central x-y plane (z = z_c),
    cropped to |x|, |y| <= R and set to NaN outside the physical radius R.

    arr3d: array (Nx, Ny, Nz);  ic = (icx, icy, icz) index of the centre.
    The returned field is transposed to (y, x) for ``imshow``.
    NOTE: ``axis_pair`` is accepted for interface compatibility but not used:
    the plane is always x-y.
    """
    icx, icy, icz = ic
    sx = np.abs(x) <= R
    sy = np.abs(y) <= R
    xx, yy = x[sx], y[sy]
    plane = arr3d[np.ix_(sx, sy)][:, :, icz].T        # (y, x)
    XX, YY = np.meshgrid(xx, yy)
    radial = np.sqrt(XX**2 + YY**2) <= R
    plane = np.where(radial, plane, np.nan)
    extent = [xx.min(), xx.max(), yy.min(), yy.max()]
    return plane, extent


def symmetric_vlim(field2d, k=2.0, pct=95):
    """Symmetric, robust colour limits around the median of a field
    (typically the potential), insensitive to edge outliers:
    median +/- k * (pct-th percentile of |field - median|)."""
    vals = field2d[np.isfinite(field2d)]
    if vals.size == 0:
        return (0.0, 1.0)
    med = float(np.median(vals))
    a = float(np.percentile(np.abs(vals - med), pct))
    return (med - k * a, med + k * a)


def potential_error(V_iks, V_ref, rho, x, R=None, frac=1e-2):
    """Error of the IKS potential with respect to a reference V_ref, computed
    robustly on the physical region (thesis Section 4.4).

    The potential is defined up to a constant: V_iks and V_ref are assumed to
    be anchored to the same gauge already (e.g. both set to zero at the node
    of maximum density). The comparison is restricted to the nodes where rho
    is significant, rho >= frac * rho_max (frac = 1e-2 in the thesis), because
    V = lambda / (rho w) diverges in the tails.

    Parameters
    ----------
    V_iks, V_ref, rho : 3D arrays with the same shape.
    x                 : coordinates along one axis (currently unused).
    R                 : physical radius (currently unused: the selection is
                        done by the density threshold only).
    frac              : density threshold for the mask (fraction of the maximum).

    Returns
    -------
    dict with:
      rmse_uniforme : RMSE over all nodes with rho >= frac*max (uniform weight);
      rmse_pesato   : density-weighted RMSE, RMSE_rho(V) of Eq. 4.13 — the
                      principal metric of the thesis, since it emphasises the
                      region where the potential is best determined;
      max_abs       : maximum absolute error in the physical region;
      n_nodi        : number of nodes used in the comparison.
    """
    import numpy as np
    mask = np.isfinite(V_iks) & np.isfinite(V_ref) & (rho >= frac * rho.max())
    if R is not None:
        # Placeholder: a further radial restriction would require the full grid.
        pass
    if not np.any(mask):
        return dict(rmse_uniforme=np.nan, rmse_pesato=np.nan,
                    max_abs=np.nan, n_nodi=0)
    dV = (V_iks - V_ref)[mask]
    w  = rho[mask]
    rmse_u = float(np.sqrt(np.mean(dV**2)))
    rmse_w = float(np.sqrt(np.sum(w * dV**2) / np.sum(w)))
    return dict(
        rmse_uniforme=rmse_u,
        rmse_pesato=rmse_w,
        max_abs=float(np.max(np.abs(dV))),
        n_nodi=int(mask.sum()),
    )
