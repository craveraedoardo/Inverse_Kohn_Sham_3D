"""
lagrange_mesh_3D.py
===================
Gauss-Hermite Lagrange mesh for the 3D IKS solver (Baye & Heenen 1986;
Baye 2015). Thesis: Sections 3.1 (Scheme B) and 3.2.

Nodes and weights
-----------------
With {t_i, w_i^GH} the standard Gauss-Hermite nodes and weights (weight
function e^{-t^2}), the physical mesh along one axis is

    x_i = b * sqrt(2) * t_i,
    W_i = w_i^GH * exp(t_i^2) * b * sqrt(2),                     (thesis Eq. 3.6)

so that an ordinary integral is approximated by

    int f(x) dx  ~=  sum_i W_i f(x_i).

The factor exp(t_i^2) removes the Gaussian weight implicit in the Gauss
rule, which allows the same integration code to be used on both the uniform
grid and the Lagrange mesh. The price is a very large dynamic range of W_i,
which matters for the tail treatment and for the potential extraction.

The length b is the *mesh* scale, b_mesh = s * b_x(A) (Eq. 3.7): the factor s
changes only the geometry of the mesh, never the physical density.

Derivative matrix
-----------------
Derivatives are obtained by differentiating the Lagrange interpolant through
the nodes:

    f'(x_i) ~= sum_j D_ij f(x_j),

with D computed by Fornberg's stable formula (Eq. 3.8). The matrix is DENSE
(every nodal value contributes to every derivative), which gives spectral
accuracy but also makes spurious values in the tails non-local (Section 3.4).
Unlike the finite-difference case it is NOT antisymmetric, so the transposed
operators d_d*_T are provided explicitly for the discrete integration by
parts in the gradient of the objective.

3D mesh
-------
The 3D mesh is the tensor product of three 1D meshes; derivatives act along
one axis at a time and the 3D weights are W_ijk = W_i W_j W_k (Eq. 3.2).

References
----------
    D. Baye, P.-H. Heenen, J. Phys. A 19 (1986) 2041
    D. Baye, Phys. Rep. 565 (2015) 1-107  [Sections 2-4]
    B. Fornberg, A Practical Guide to Pseudospectral Methods (1996)
"""

import numpy as np
from numpy.polynomial.hermite import hermgauss


# ─────────────────────────────────────────────────────────────────────────────
# Nodes and weights
# ─────────────────────────────────────────────────────────────────────────────

def gauss_hermite_nodes_weights(N, scale=1.0):
    """
    Physical Gauss-Hermite nodes and weights for int f(x) dx (no Gaussian factor).

    numpy's ``hermgauss`` uses the weight e^{-t^2}; with x = b*sqrt(2)*t:

        int f(x) dx = int f(b sqrt2 t) e^{-t^2} e^{t^2} b sqrt2 dt
                    ~= sum_i [w_i^GH exp(t_i^2) b sqrt2] f(x_i)

    Parameters
    ----------
    N     : int    Number of nodes.
    scale : float  Length b of the mesh [fm] (b_mesh = s * b_x(A)).

    Returns
    -------
    x : (N,)  physical nodes  x_i = b*sqrt(2) * t_i
    W : (N,)  physical weights W_i = w_i^GH * exp(t_i^2) * b*sqrt(2)
    """
    t, w_gh = hermgauss(N)
    x = t * scale * np.sqrt(2.0)
    W = w_gh * np.exp(t**2) * scale * np.sqrt(2.0)
    return x, W


# ─────────────────────────────────────────────────────────────────────────────
# 1D derivative matrix (Fornberg's stable formula)
# ─────────────────────────────────────────────────────────────────────────────

def derivative_matrix(x):
    """
    Derivative matrix D (N x N) of the Lagrange interpolant on the nodes x:

        (D f)_i = f'(x_i) = sum_j D_ij f(x_j)

    Stable formula (Fornberg; Baye 2015, Eq. 4.2; thesis Eq. 3.8):

        D_ij = C_i / [C_j (x_i - x_j)]     for i != j
        D_ii = - sum_{j != i} D_ij          (so that D applied to a constant is 0)

    with C_k = prod_{m != k} (x_k - x_m).

    The products C_k overflow in double precision for large N, so log|C_k|
    and sign(C_k) are accumulated separately.
    """
    N = len(x)
    xi = x[:, None]   # (N, 1)
    xj = x[None, :]   # (1, N)
    diff = xi - xj    # diff_ij = x_i - x_j, zero on the diagonal

    # Off-diagonal mask
    mask_off = ~np.eye(N, dtype=bool)

    # C_k in log space: replace the zero diagonal by 1 to avoid log(0)
    diff_od = np.where(np.eye(N, dtype=bool), 1.0, diff)
    sign_C  = np.prod(np.sign(diff_od), axis=1)             # (N,)
    log_C   = np.sum(np.log(np.abs(diff_od)), axis=1)       # (N,)

    # D_ij = C_i / [C_j (x_i - x_j)]
    # log|D_ij|  = log|C_i| - log|C_j| - log|x_i - x_j|
    # sign(D_ij) = sign(C_i) * sign(C_j) * sign(x_i - x_j)
    diff_od2 = np.where(np.eye(N, dtype=bool), 1.0, diff)
    log_D = (log_C[:, None] - log_C[None, :]
             - np.log(np.abs(diff_od2)))
    sign_D = (sign_C[:, None] * sign_C[None, :]
              * np.sign(diff_od2))

    D = np.where(mask_off, sign_D * np.exp(log_D), 0.0)
    # Diagonal from the consistency condition: row sums vanish (D 1 = 0).
    np.fill_diagonal(D, -np.sum(D, axis=1))
    return D


# ─────────────────────────────────────────────────────────────────────────────
# Main class: LagrangeMesh3D
# ─────────────────────────────────────────────────────────────────────────────

class LagrangeMesh3D:
    """
    Separable 3D Lagrange mesh (tensor product of three Gauss-Hermite meshes).

    Builds the nodes, the integration weights and the three 1D derivative
    matrices, and exposes the methods used by ``iKS_problem``
    (``grad``, ``d_dx``/``d_dy``/``d_dz``, their transposes, ``integrate``).

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of nodes per axis. The thesis uses N ~ 20-36 per axis; the
        accuracy depends at least as much on the mesh scale as on N
        (Sections 3.3 and 5.5).
    bx, by, bz : float
        Mesh lengths [fm], i.e. s * b_x(A) with s the dimensionless mesh scale.

    Attributes
    ----------
    x, y, z    : 1D node arrays [fm]
    wx, wy, wz : 1D physical weights
    Dx, Dy, Dz : dense 1D derivative matrices
    grid       : tuple (x, y, z)
    w3D        : (Nx, Ny, Nz) array of 3D weights W_ijk = W_i W_j W_k
    dV         : mean of w3D. It is NOT a volume element (the weights vary by
                 orders of magnitude); it exists only for interfaces that
                 expect a scalar and is always re-normalised away by the callers.
    """

    def __init__(self, Nx, Ny, Nz, bx, by, bz):
        self.Nx = Nx; self.Ny = Ny; self.Nz = Nz
        self.bx = bx; self.by = by; self.bz = bz

        # Physical nodes and weights
        self.x, self.wx = gauss_hermite_nodes_weights(Nx, bx)
        self.y, self.wy = gauss_hermite_nodes_weights(Ny, by)
        self.z, self.wz = gauss_hermite_nodes_weights(Nz, bz)

        # 1D derivative matrices
        self.Dx = derivative_matrix(self.x)   # (Nx, Nx)
        self.Dy = derivative_matrix(self.y)   # (Ny, Ny)
        self.Dz = derivative_matrix(self.z)   # (Nz, Nz)

        # 3D grid and weights
        self.grid = (self.x, self.y, self.z)
        self.w3D  = (self.wx[:, None, None]
                     * self.wy[None, :, None]
                     * self.wz[None, None, :])   # (Nx, Ny, Nz)

        # Scalar "dV" kept only for compatibility with code expecting a scalar
        self.dV = float(np.mean(self.w3D))

        self.n_points_x = Nx
        self.n_points_y = Ny
        self.n_points_z = Nz
        self.n_points   = Nx * Ny * Nz

        # Extent of the mesh (first and last node along each axis)
        self.lb = [float(self.x[0]),  float(self.y[0]),  float(self.z[0])]
        self.ub = [float(self.x[-1]), float(self.y[-1]), float(self.z[-1])]

        print(f"[LagrangeMesh3D] {Nx}×{Ny}×{Nz} = {self.n_points} nodi")
        print(f"  x ∈ [{self.x[0]:.3f}, {self.x[-1]:.3f}] fm")
        print(f"  y ∈ [{self.y[0]:.3f}, {self.y[-1]:.3f}] fm")
        print(f"  z ∈ [{self.z[0]:.3f}, {self.z[-1]:.3f}] fm")

    # ── Derivatives ──────────────────────────────────────────────────────────
    # Each derivative is a contraction of one 1D matrix with one axis of the
    # (Nx, Ny, Nz) array, i.e. (D_x f)_ijk = sum_l Dx_il f_ljk (thesis Eq. 3.1).

    def d_dx(self, f):
        """df/dx via Dx. Input and output have shape (Nx, Ny, Nz)."""
        return np.einsum('il,ljk->ijk', self.Dx, f)

    def d_dy(self, f):
        """df/dy via Dy."""
        return np.einsum('jl,ilk->ijk', self.Dy, f)

    def d_dz(self, f):
        """df/dz via Dz."""
        return np.einsum('kl,ijl->ijk', self.Dz, f)

    def grad(self, f):
        """3D gradient. Returns an array of shape (Nx, Ny, Nz, 3)."""
        return np.stack((self.d_dx(f), self.d_dy(f), self.d_dz(f)), axis=-1)

    # ── Transposed derivatives (for the gradient of the objective) ───────────
    # The discrete integration by parts with non-constant weights needs D^T:
    # d/df_j sum_i W_i G_i (D f)_i = [D^T (W G)]_j (thesis Eq. 3.4). Since the
    # Gauss-Hermite matrix is NOT antisymmetric, D^T != -D and the transpose
    # has to be applied explicitly.

    def d_dx_T(self, f):
        """Apply Dx^T along the x axis."""
        return np.einsum('li,ljk->ijk', self.Dx, f)

    def d_dy_T(self, f):
        """Apply Dy^T along the y axis."""
        return np.einsum('lj,ilk->ijk', self.Dy, f)

    def d_dz_T(self, f):
        """Apply Dz^T along the z axis."""
        return np.einsum('lk,ijl->ijk', self.Dz, f)

    # ── Integrals ────────────────────────────────────────────────────────────

    def integrate(self, f):
        """Quadrature of a 3D array: int f dV ~= sum_ijk W_ijk f_ijk."""
        return float(np.sum(self.w3D * f))

    # ── Self-check ───────────────────────────────────────────────────────────

    def check(self):
        """
        Print two accuracy checks of the mesh:
          - the norm of psi_000 (should be 1 to machine precision);
          - the error of D on the Gaussian exp(-x^2/b^2) (should be ~0).

        Returns (norm2, err), with norm2 the norm of the un-normalised psi_000.
        """
        from numpy.polynomial.hermite import Hermite
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

        # Norm of psi_000
        H0 = Hermite.basis(0)
        psi = (H0(X/self.bx)*np.exp(-(X/self.bx)**2/2)
              *H0(Y/self.by)*np.exp(-(Y/self.by)**2/2)
              *H0(Z/self.bz)*np.exp(-(Z/self.bz)**2/2))
        norm2 = self.integrate(psi**2)
        psi_n = psi / np.sqrt(norm2)
        print(f"  check ∫|ψ₀₀₀|²dV = {self.integrate(psi_n**2):.12f}  (atteso 1.0)")

        # Derivative error: d/dx [e^{-x^2/b^2}] = -2x/b^2 * e^{-x^2/b^2}
        fx    = np.exp(-X**2 / self.bx**2)
        exact = -2*X/self.bx**2 * fx
        num   = self.d_dx(fx)
        err   = np.max(np.abs(num - exact))
        print(f"  check max|∂f/∂x - exact| = {err:.3e}  (su gaussiana, atteso ~0)")
        return norm2, err
