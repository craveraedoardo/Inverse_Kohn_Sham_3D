"""
optimizer3D_lm.py
=================
Core of the code: the class ``iKS_problem``, which formulates the inverse
Kohn-Sham (IKS) problem with the constrained-variational (CV) method and
solves it with IPOPT (through cyipopt).

The problem (thesis Sections 2.6, 3 and 4)
------------------------------------------
Given a target density rho_t on a 3D Cartesian grid, find the occupied
orbitals of minimum kinetic energy that reproduce it. With the change of
variable

    phi_k = f_k * sqrt(rho_t)                                     (Eq. 2.35)

the unknowns are the nodal values f_{k,r} of every orbital k on every node r,
and the discrete problem is

    min_f  Phi(f) = sum_k q_k sum_r W_r^eff [ (alpha rho + beta) |grad f_k|^2
                                            + alpha rho f_k^2 |g|^2
                                            + 2 alpha rho f_k (grad f_k . g) ]
    s.t.   sum_k q_k f_{k,r}^2 - 1 = 0              on every live node r
           sum_r W_r rho_r f_{i,r} f_{j,r} = delta_ij   for every pair i <= j
           [optional] multipole constraints (Section 6.1)
           f bounded in [-1, 1]; dead nodes frozen at the plateau value.

Here alpha = hbar^2/2m, beta is the regularisation weight, g = grad log
sqrt(rho_t), q_k the occupations and W^eff = W * chi the quadrature weights
multiplied by the active mask chi. The terms in alpha reproduce the kinetic
energy T_alpha = alpha sum_k q_k int |grad phi_k|^2; the term in beta is the
regulariser T_beta = beta sum_k q_k int |grad f_k|^2.

The Kohn-Sham potential is NOT an unknown: it is read off the Lagrange
multipliers lambda_r of the density constraints,

    v_KS(r) = lambda_r / (rho_t,r * W_r)        up to a constant   (Eq. 4.8)

which is done by the test scripts after ``solve()``.

Two discretisations, one code
-----------------------------
The flag ``use_lagrange_mesh`` selects the scheme (Section 3.1):

  * False: uniform grid on [lb, ub]^3 with spacing h and 4th-order finite
    differences (``findiff``); the derivative matrix is antisymmetrised.
  * True:  Gauss-Hermite Lagrange mesh (``lagrange_mesh_3D.LagrangeMesh3D``)
    with N nodes per axis and mesh scale s (b_mesh = s * b_x(A)).

The derivative operators ``_d_dx`` ... and their transposes dispatch to the
active scheme, so that everything else (objective, gradient, constraints,
Jacobian, Hessian, starting point, solver options) is shared.

Numerical ingredients worth knowing before modifying the code
-------------------------------------------------------------
1. Density floor (Eq. 3.12): rho_t <- max(rho_t, 1e-14 rho_max) avoids
   sqrt(0) and log(0).
2. Active mask (Eq. 3.13): nodes with rho_t < eps * rho_max are "dead" and
   are excluded from every integral (eps = 1e-10 on the Lagrange mesh,
   1e-6 on the uniform grid). The mask must enter INSIDE the transposed
   operator in the gradient, D^T(W^eff G), for the gradient to be exact.
3. Plateau (Eq. 3.14): dead-node variables are frozen (lb = ub) at
   f = 1/sqrt(sum_k q_k), the natural continuation of the solution;
   freezing them at 0 would create a jump that the dense derivative matrix
   turns into a spurious kinetic energy.
4. Density constraints exist only on live nodes, so the first ``n_live``
   multipliers in ``mult_g`` refer to ``live_nodes`` (not to all nodes).
   Constraint order: [density (live nodes)] + [orthonormality pairs] +
   [multipoles].
5. The objective is an exact quadratic form in f, so its Hessian is
   constant (Section 4.2): it is assembled once as a sparse matrix M.
6. ``solve()`` ALWAYS calls ``self.getStartingPoint()``. To start from a
   different point, replace that method on the instance (the deformed and
   excited-configuration tests do this).

Output
------
``solve()`` writes, in ``test_3D_data/<output_folder>/``: ``data`` (pickled
dictionary with the full result), ``f.dat`` (f_k on the nodes) and
``phi.dat`` (phi_k on the nodes). IPOPT writes its log to ``ipopt.out`` in
the current working directory.
"""

import numpy as np
import numexpr as ne
ne.set_num_threads(8)   # threads used by numexpr in objective and gradient
import os

from cyipopt import Problem as ipopt_Problem
from findiff import FinDiff

from consts3D import a, b_x, b_y, b_z
from eigenfunctions3D import HO3D
from orbitals3D import uncoupledHObasis, getOrbitalSet
from utility3D import saveData
from lagrange_mesh_3D import LagrangeMesh3D
from multipoles3D import (parse_lm_list, build_constraint_coefficients)


class iKS_problem(ipopt_Problem):
    """Inverse Kohn-Sham problem in 3D Cartesian coordinates, solved with IPOPT.

    The class is itself a ``cyipopt.Problem``: it provides the callbacks
    ``objective``, ``gradient``, ``constraints``, ``jacobianstructure``,
    ``jacobian``, ``hessianstructure`` and ``hessian`` that IPOPT calls during
    the optimisation.

    Parameters
    ----------
    Z, N : int
        Numbers of protons and neutrons. The number of particles used to fill
        the orbitals is Z if ``n_type == 'p'`` and A = Z + N if ``n_type == 'n'``.
        All the thesis scripts describe a pure-neutron drop by passing Z = A,
        N = 0 with the default n_type = 'p': only the particle count matters.
    lb, ub : list of 3 floats
        Box [lb, ub] of the uniform grid [fm]. Ignored on the Lagrange mesh
        (the scripts pass dummy values).
    data : list
        [x, y, z, rho]. If rho is a 3D array already tabulated on the nodes of
        the computational grid (the recommended usage, and the only one used
        in the thesis), it is used as is. Otherwise a cubic interpolator is
        built from it.
    rho : callable, optional
        Target density as a function of (X, Y, Z); takes precedence over data.
    exp : bool
        Unused (kept for backward compatibility).
    hx, hy, hz : float
        Grid spacing of the uniform grid [fm]; hy, hz default to hx.
    n_type : {'p', 'n'}
        See Z, N above.
    basis : OrbitalSet, optional
        Ordered list of states to be filled (default: ``uncoupledHObasis()``).
    reg_parameter : float
        Regularisation weight beta [MeV fm^2].
    max_iter, rel_tol, constr_viol, dual_inf_tol : IPOPT options
        max_iter, tol, constr_viol_tol and dual_inf_tol (Section 4.3).
    output_folder : str
        Sub-folder of ``test_3D_data/`` where the results are written.
    exact_hess : bool
        Use the exact (constant) Hessian of the objective; otherwise L-BFGS.
        The exact Hessian is used throughout the thesis.
    com_correction, debug :
        Unused (kept for backward compatibility).
    use_lagrange_mesh : bool
        True for the Gauss-Hermite Lagrange mesh, False for the uniform grid.
    Nx, Ny, Nz : int
        Nodes per axis of the Lagrange mesh.
    mesh_scale : float
        Dimensionless scale s of the Lagrange mesh, b_mesh = s * b_x(A)
        (Section 3.3). s = 1 is the historical choice; s ~ 0.79 is the
        optimum found for A = 2 (Section 5.5).
    multipole_constraints : list of (l, m), optional
        Optional multipole constraints Q_lm[rho_KS] = Q_lm[rho_t] (Section 6.1).
    multipole_scale : float, optional
        Length b used to normalise those constraints (default b_x(A)).
    """

    def __init__(self, Z, N, lb, ub, data, rho=None, exp=True,
                 hx=0.1, hy=None, hz=None, n_type="p",
                 basis=None, reg_parameter=0.1,
                 max_iter=2000, rel_tol=1e-3, constr_viol=1e-3, dual_inf_tol=1e-3,
                 output_folder="Output", exact_hess=True, com_correction=False, debug='n',
                 use_lagrange_mesh=False, Nx=12, Ny=12, Nz=12, mesh_scale=1.0,
                 multipole_constraints=None, multipole_scale=None):

        self.Z = Z
        self.N = N
        self.A = Z + N
        self.n_type = n_type if (n_type == 'p' or n_type == 'n') else "p"
        self.n_particles = Z if self.n_type == 'p' else self.A
        self.beta = reg_parameter
        self.alpha = a

        self.basis = basis if basis is not None else uncoupledHObasis()
        # IMPORTANT: OrbitalSet is an iterator that is exhausted after its
        # first use. It is materialised as a list here; otherwise the call to
        # getStartingPoint() inside solve() would find it already consumed and
        # return an all-zero starting point (IPOPT would then start from x = 0
        # and fail).
        self.orbital_set = list(getOrbitalSet(self.n_particles, self.basis))
        self.n_orbitals = len(self.orbital_set)
        print("Numero di orbitali usati nel problema: ", self.n_orbitals)
        self.pairs = self.getPairs()

        self.data = data
        # The target density can be given as:
        #   (a) a callable ``rho``;
        #   (b) an array already tabulated on the nodes of the grid (preferred:
        #       on the Lagrange mesh it avoids interpolating on a non-regular
        #       grid);
        #   (c) data on a different grid -> a cubic interpolator is built.
        if rho is not None:
            self.rho = rho
        elif (data is not None and len(data) >= 4
              and isinstance(data[3], np.ndarray)
              and data[3].ndim == 3):
            # Density already tabulated on the nodes (Lagrange mesh OR uniform
            # grid): use it DIRECTLY. Re-interpolating would introduce an error
            # (~1e-6) which, where rho ~ 1e-15, makes sqrt(rho) and
            # f = phi/sqrt(rho) completely spurious.
            self._rho_tab = np.asarray(data[3])
            self.rho = None
        else:
            # Data on a grid different from the computational one -> interpolate.
            self.rho = self.getRhoFromData(data[0], data[1], data[2], data[3])
        self.exponential = exp

        self.use_lagrange_mesh = use_lagrange_mesh
        self.lm_Nx = Nx
        self.lm_Ny = Ny
        self.lm_Nz = Nz
        # Dimensionless factor s: b_mesh = s * b_x(A). s = 1 reproduces EXACTLY
        # the historical behaviour (mesh scale = physical oscillator length).
        # NOTE: s scales ONLY the geometry of the mesh (nodes, weights, D
        # matrices); the physical orbitals in getStartingPoint stay on b_x(A).
        self.mesh_scale = float(mesh_scale)

        # ── Multipole constraints (optional, Section 6.1) ───────────────────
        # List of (l, m) pairs, e.g. [(2,0)] or [(2,0),(2,2)]. The actual
        # coefficients are built in set_grid(), once grid, weights, tabulated
        # density and live nodes are available.
        # WARNING: with a POINTWISE density constraint these constraints are
        # exact linear combinations of the existing ones (see multipoles3D.py).
        # They are here to verify this, not because they add information.
        self.multipole_constraints = parse_lm_list(multipole_constraints)
        self.multipole_scale = (float(multipole_scale)
                                if multipole_scale is not None
                                else float(b_x(self.A)))

        self.set_grid(lb, ub, hx, hy, hz)

        self.exact_hess = exact_hess
        self.dual_inf_tol = dual_inf_tol
        self.debug = True if debug == 'y' else False

        # Exact Hessian: the objective is a quadratic form in f, so its Hessian
        # is CONSTANT and is precomputed as a sparse per-orbital matrix 2*occ_k*M.
        if self.exact_hess:
            self._build_objective_hessian()
            self._build_hessian_structure()

        self.output_folder = "test_3D_data/" + output_folder if len(output_folder) > 0 else ""
        if len(self.output_folder) > 0 and not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.f_file       = f"{self.output_folder}/f.dat"
        self.phi_file     = f"{self.output_folder}/phi.dat"
        # The following three names are defined but not written by this class.
        self.pot_file     = f"{self.output_folder}/potential.dat"
        self.epsilon_file = f"{self.output_folder}/epsilon.dat"
        self.eigen_file   = f"{self.output_folder}/eigenvalues.dat"

        self.results  = dict()
        self.datafile = f"{self.output_folder}/data"

        # Registers the problem with IPOPT (bounds, constraint bounds, options).
        self.setIpopt(max_iter, rel_tol, constr_viol, dual_inf_tol, exact_hess)
        self.n_runs = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Grid setup
    # ─────────────────────────────────────────────────────────────────────────

    def set_grid(self, lb, ub, hx, hy, hz):
        """Build the grid and every quantity that depends only on rho_t.

        The order matters, since each step uses the previous ones:
        grid -> tabulated density + floor -> g = grad log sqrt(rho) ->
        active mask -> live nodes -> multipole coefficients -> constraint count.
        """
        self.lb = lb
        self.ub = ub
        self.hx = hx
        self.hy = hy if hy is not None else self.hx
        self.hz = hz if hz is not None else self.hx

        # The only point where the two schemes diverge.
        if self.use_lagrange_mesh:
            self._set_grid_lagrange()
        else:
            self._set_grid_uniform()

        X, Y, Z = np.meshgrid(*self.grid, indexing='ij')

        # Target density on the nodes, then the floor of Eq. (3.12) to avoid
        # sqrt(0) / log(0).
        if self.rho is not None:
            self.tab_rho = self.rho((X, Y, Z))
        else:
            # Density already tabulated on the nodes
            self.tab_rho = self._rho_tab.reshape(
                self.n_points_x, self.n_points_y, self.n_points_z)
        rho_floor = 1e-14 * self.tab_rho.max()
        self.tab_rho = np.maximum(self.tab_rho, rho_floor)

        if self.use_lagrange_mesh:
            # ── g = grad log sqrt(rho), computed NUMERICALLY ─────────────────
            # It is obtained with the derivative matrix of the mesh. An earlier
            # analytic expression, -x/(2b^2), was WRONG on two counts:
            #   (a) wrong factor of 2 — for rho ~ exp(-x^2/b^2) it is -x/b^2;
            #   (b) valid only for the Gaussian ground state: with excited
            #       orbitals (A >= 4) the density is not Gaussian and the
            #       Hermite polynomials contribute to grad log sqrt(rho).
            # The numerical evaluation is exact for any density; the floor
            # applied above prevents log(rho) from blowing up at the outermost
            # nodes, and the active mask excludes them anyway from objective
            # and gradient.
            s = np.sqrt(self.tab_rho)
            self.grad_log_s = self.grad(np.log(s))

            # ── Active mask: physical vs dead nodes (Eq. 3.13) ───────────────
            # At the outer nodes rho << rho_max, so f_k = phi/sqrt(rho) >> 1.
            # The matrix D is global (dense): those spurious values would
            # contaminate the derivatives at ALL nodes. They are therefore
            # excluded from objective and gradient.
            self.active_mask = (
                self.tab_rho >= 1e-10 * self.tab_rho.max()
            ).astype(float)
            n_dead = int((self.active_mask == 0).sum())
            print(f"  active_mask: {self.n_points - n_dead}/{self.n_points} nodi fisici "
                  f"({n_dead} nodi morti esclusi)")
            print("  [VERSIONE CORRETTA v10: grad_log_s numerico + Hessiana esatta]")
        else:
            # Uniform grid: g computed numerically as well.
            s = np.sqrt(self.tab_rho)
            self.grad_log_s = self.grad(np.log(s))
            # Mask threshold on the uniform grid: here the transition rho -> 0
            # is more gradual and the "borderline" nodes
            # (1e-10 < rho/rho_max < 1e-6) give values of f = phi/sqrt(rho)
            # that are large but finite, which the derivative operator still
            # propagates as spurious gradients onto the neighbouring physical
            # nodes. Raising the threshold to 1e-6 excludes those nodes and
            # restricts the effective domain to the region where rho is truly
            # significant. The plateau value on the dead nodes guarantees that
            # f is continuous at the boundary (no jump). See Section 3.4.
            self.active_mask = (
                self.tab_rho >= 1e-6 * self.tab_rho.max()
            ).astype(float)
            n_dead = int((self.active_mask == 0).sum())
            print(f"  active_mask: {self.n_points - n_dead}/{self.n_points} nodi fisici "
                  f"({n_dead} nodi a bassa densita esclusi, soglia 1e-6*rho_max)")

        # |g|^2, needed by objective, gradient and Hessian.
        self.norm_grad_log_s = np.einsum(
            'ijkm,ijkm->ijk', self.grad_log_s, self.grad_log_s)

        self.n_variables = self.n_orbitals * self.n_points

        # Indices of the nodes where the density constraint is ACTIVE (physical
        # nodes). On the dead nodes the constraint (sum occ f^2 - 1)*mask = 0
        # would be identically zero (mask = 0): a redundant constraint that
        # IPOPT would flag as "linearly dependent". It is removed upstream: the
        # list of density constraints only contains the live nodes.
        self.live_nodes = np.where(self.active_mask.ravel() > 0)[0]
        self.n_live     = len(self.live_nodes)

        # ── Coefficients of the multipole constraints ────────────────────────
        # C_j(f) = sum_{r in live} a_jr * (sum_k occ_k f_kr^2 - 1)
        # with a_jr = w_r rho_r R_lm(r) / scale^l. Defined on the live nodes
        # ONLY: on the dead nodes f is fixed by the bounds and the density
        # constraint does not exist, so including them would introduce a
        # spurious constant offset.
        self.mp_coefs, self.mp_scales = build_constraint_coefficients(
            self.multipole_constraints, self.grid, self.w3D, self.tab_rho,
            self.live_nodes, length_scale=self.multipole_scale)
        self.n_multipole = len(self.multipole_constraints)
        if self.n_multipole > 0:
            names = ", ".join(f"(l={l},m={m})"
                              for (l, m) in self.multipole_constraints)
            print(f"  vincoli multipolari attivi: {self.n_multipole} → {names} "
                  f"(scala di lunghezza {self.multipole_scale:.4f} fm)")

        # Constraint order: [density (live)] + [pairs] + [multipoles]
        self.n_constr = self.n_live + len(self.pairs) + self.n_multipole

    def _set_grid_uniform(self):
        """Scheme A: uniform grid on [lb, ub]^3, 4th-order finite differences."""
        self.dV = self.hx * self.hy * self.hz

        self.n_points_x = round((self.ub[0] - self.lb[0]) / self.hx) + 1
        self.n_points_y = round((self.ub[1] - self.lb[1]) / self.hy) + 1
        self.n_points_z = round((self.ub[2] - self.lb[2]) / self.hz) + 1
        self.n_points   = self.n_points_x * self.n_points_y * self.n_points_z

        x = np.linspace(self.lb[0], self.ub[0], self.n_points_x)
        y = np.linspace(self.lb[1], self.ub[1], self.n_points_y)
        z = np.linspace(self.lb[2], self.ub[2], self.n_points_z)
        self.grid = (x, y, z)

        # findiff operators, used for the forward derivatives (Eq. 3.3).
        self.d_dx_op = FinDiff(0, self.hx, 1, acc=4)
        self.d_dy_op = FinDiff(1, self.hy, 1, acc=4)
        self.d_dz_op = FinDiff(2, self.hz, 1, acc=4)

        # Explicit 1D derivative matrices (to apply the transpose in the
        # gradient and to build the Hessian). The raw findiff operator is NOT
        # antisymmetric -> D^T != -D, hence the explicit (antisymmetrised) matrix.
        self.Dx_mat = self._build_findiff_matrix(self.n_points_x, self.hx)
        self.Dy_mat = self._build_findiff_matrix(self.n_points_y, self.hy)
        self.Dz_mat = self._build_findiff_matrix(self.n_points_z, self.hz)

        self.mesh = None
        self.w3D  = np.full(
            (self.n_points_x, self.n_points_y, self.n_points_z), self.dV)
        print(f"[Griglia uniforme] "
              f"{self.n_points_x}×{self.n_points_y}×{self.n_points_z} = {self.n_points} punti")

    @staticmethod
    def _build_findiff_matrix(n, h):
        """Build the explicit matrix of the 1D derivative operator (thesis Eq. 3.5).

        findiff with acc=4 uses asymmetric one-sided stencils at the edges, so
        the raw matrix D_raw does not satisfy D_raw^T = -D_raw. This breaks the
        discrete integration by parts in the gradient of the objective,

            d/df_j sum_i w G_i (D f)_i = sum_i w G_i D_ij = [D^T (w G)]_j,

        which requires D^T to be the adjoint of D with respect to the scalar
        product with uniform weights w = dV = const, i.e. D^T = -D
        (antisymmetric derivative).

        The antisymmetric projection D = (D_raw - D_raw^T)/2 restores this
        property: on the interior nodes (where findiff is already centred) the
        correction is numerically zero; at the edges it replaces the one-sided
        stencils with their antisymmetric average, which is locally less
        accurate but leaves the global accuracy intact, because the edge nodes
        lie in the dead-node region (active_mask = 0) and do not enter the
        effective integral.

        NOTE: the forward derivatives ``_d_dx`` etc. still use the raw findiff
        operator; only the transposes and the Hessian use this matrix.
        """
        op = FinDiff(0, h, 1, acc=4)
        M = np.zeros((n, n))
        e = np.zeros(n)
        # Column i of the matrix = operator applied to the i-th unit vector.
        for i in range(n):
            e[:] = 0.0
            e[i] = 1.0
            M[:, i] = op(e)
        # Antisymmetrise: guarantees D^T = -D for the discrete integration by
        # parts. Interior (centred) rows change only at machine-epsilon level;
        # at the edges the one-sided stencil that would make D^T != -D is removed.
        return (M - M.T) / 2.0

    def _set_grid_lagrange(self):
        """Scheme B: Gauss-Hermite Lagrange mesh with b_mesh = s * b(A)."""
        s  = getattr(self, "mesh_scale", 1.0)
        bx = s * b_x(self.A)
        by = s * b_y(self.A)
        bz = s * b_z(self.A)
        if s != 1.0:
            print(f"[Lagrange mesh] mesh_scale s={s:.4f} → b_mesh = {bx:.4f} fm "
                  f"(fisico b_x(A) = {b_x(self.A):.4f} fm)")

        self.mesh = LagrangeMesh3D(self.lm_Nx, self.lm_Ny, self.lm_Nz, bx, by, bz)

        self.n_points_x = self.mesh.Nx
        self.n_points_y = self.mesh.Ny
        self.n_points_z = self.mesh.Nz
        self.n_points   = self.mesh.n_points

        self.grid = self.mesh.grid
        self.dV   = self.mesh.dV
        self.w3D  = self.mesh.w3D
        print(f"[Lagrange mesh] "
              f"{self.n_points_x}×{self.n_points_y}×{self.n_points_z} = {self.n_points} punti")

    # ─────────────────────────────────────────────────────────────────────────
    # Differential operators (dispatch to the active scheme)
    # ─────────────────────────────────────────────────────────────────────────

    def _d_dx(self, f):
        if self.use_lagrange_mesh:
            return self.mesh.d_dx(f)
        return self.d_dx_op(f)

    def _d_dy(self, f):
        if self.use_lagrange_mesh:
            return self.mesh.d_dy(f)
        return self.d_dy_op(f)

    def _d_dz(self, f):
        if self.use_lagrange_mesh:
            return self.mesh.d_dz(f)
        return self.d_dz_op(f)

    def grad(self, f):
        """Gradient of a scalar field (Nx, Ny, Nz) -> (Nx, Ny, Nz, 3)."""
        return np.stack([self._d_dx(f), self._d_dy(f), self._d_dz(f)], axis=-1)

    def derive(self, x):
        """Gradients of all orbitals: (n_orb, Nx, Ny, Nz) -> (n_orb, Nx, Ny, Nz, 3)."""
        return np.stack([self.grad(x[k]) for k in range(self.n_orbitals)], axis=0)

    # Transposed derivatives, needed by the gradient of the objective.
    # The derivative matrix is not antisymmetric on the Lagrange mesh, so D^T
    # must be applied explicitly; on the uniform grid the antisymmetrised
    # matrix of _build_findiff_matrix is used.
    def _d_dx_T(self, f):
        if self.use_lagrange_mesh:
            return self.mesh.d_dx_T(f)
        return np.einsum('li,ljk->ijk', self.Dx_mat, f)

    def _d_dy_T(self, f):
        if self.use_lagrange_mesh:
            return self.mesh.d_dy_T(f)
        return np.einsum('lj,ilk->ijk', self.Dy_mat, f)

    def _d_dz_T(self, f):
        if self.use_lagrange_mesh:
            return self.mesh.d_dz_T(f)
        return np.einsum('lk,ijl->ijk', self.Dz_mat, f)

    # ─────────────────────────────────────────────────────────────────────────
    # Objective (thesis Appendix A.2)
    # ─────────────────────────────────────────────────────────────────────────

    def integral_k(self, xk, grad_1_x_k, k):
        """Contribution of orbital k to the objective.

        occ_k * sum_r W_r^eff [ (alpha rho + beta)|grad f|^2
                                + alpha rho (f^2 |g|^2 + 2 f grad f . g) ],
        i.e. alpha*q_k*int|grad(f sqrt(rho))|^2 (expanded) + beta*q_k*int|grad f|^2.
        """
        grad_mod2 = np.sum(grad_1_x_k**2, axis=-1)
        dot_grad  = np.sum(self.grad_log_s * grad_1_x_k, axis=-1)

        arr = ne.evaluate(
            "occ*( (alpha*rho+beta)*grad_mod2 +"
            " alpha*rho*( xk**2*norm_grad_log_s + 2*xk*dot_grad) )",
            {
                "occ":             self.orbital_set[k].occupation,
                "alpha":           self.alpha,
                "beta":            self.beta,
                "rho":             self.tab_rho,
                "grad_mod2":       grad_mod2,
                "norm_grad_log_s": self.norm_grad_log_s,
                "dot_grad":        dot_grad,
                "xk":              xk,
            })

        # The active mask removes the dead nodes from the integral.
        return np.sum(arr * self.w3D * self.active_mask)

    def objective(self, x):
        """IPOPT callback: value of the objective Phi(f) [MeV]."""
        x = np.reshape(
            x, (self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z))
        grad_1x = self.derive(x)
        return np.sum([self.integral_k(x[k], grad_1x[k], k)
                       for k in range(self.n_orbitals)])

    def gradient(self, x):
        """IPOPT callback: exact gradient of the objective (thesis Eq. 4.1)."""
        x = np.reshape(
            x, (self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z))
        grad    = np.zeros_like(x)
        grad1_x = self.derive(x)

        # Effective weight: excludes the dead nodes
        w_eff = self.w3D * self.active_mask

        for k in range(self.n_orbitals):
            xk  = x[k]
            deg = self.orbital_set[k].occupation

            # Vector integrand (derivative flux, Eq. 4.2):
            #   G = alpha f rho g + (alpha rho + beta) grad f
            derivand = (
                self.alpha * xk[..., None] * self.tab_rho[..., None] * self.grad_log_s
                + (self.alpha * self.tab_rho[..., None] + self.beta) * grad1_x[k]
            )

            # Divergence term from the discrete integration by parts:
            #   d/df_j sum_i w_i G_i.(grad f)_i = sum_i w_i G_i D_ij = [D^T (w G)]_j
            # The weight w_eff goes INSIDE the transposed operator (non-uniform
            # mesh); applying it afterwards would not give the exact gradient.
            div = (self._d_dx_T(derivand[..., 0] * w_eff)
                   + self._d_dy_T(derivand[..., 1] * w_eff)
                   + self._d_dz_T(derivand[..., 2] * w_eff))

            # Local term: derivative of the algebraic part, weighted by w_eff
            scalar_dlog_s_df = np.sum(self.grad_log_s * grad1_x[k], axis=-1)

            grad[k] = ne.evaluate(
                "2*deg*( alpha*rho*( xk*norm_grad_log_s + scalar_dlog_s_df)*w_eff"
                " + div )",
                {
                    "deg":              deg,
                    "alpha":            self.alpha,
                    "rho":              self.tab_rho,
                    "xk":               xk,
                    "norm_grad_log_s":  self.norm_grad_log_s,
                    "scalar_dlog_s_df": scalar_dlog_s_df,
                    "w_eff":            w_eff,
                    "div":              div,
                })
        return grad.ravel()

    # ─────────────────────────────────────────────────────────────────────────
    # Constraints (thesis Appendix A.4)
    # ─────────────────────────────────────────────────────────────────────────

    def constraints(self, x):
        """IPOPT callback: constraint vector [density (live), orthonormality, multipoles]."""
        x = np.reshape(
            x, (self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z))

        # NORMALISED density constraint: (sum_k occ_k f_k^2) - 1 = 0, on the
        # physical nodes only (dead nodes would give a redundant 0 = 0 and are
        # excluded upstream). It is deliberately NOT rescaled by w or rho
        # (thesis Remark 4.1).
        sum_occ_f2 = np.sum([self.orbital_set[n].occupation * x[n]**2
                             for n in range(self.n_orbitals)], axis=0)
        dens = (sum_occ_f2.ravel() - 1.0)[self.live_nodes]

        # Orthonormality: int rho f_i f_j w dV = delta_ij, for each pair i >= j
        orto = np.zeros(len(self.pairs))
        for n, (i, j) in enumerate(self.pairs):
            val = np.sum(self.tab_rho * x[i] * x[j] * self.w3D)
            orto[n] = val - 1 if (i == j) else val

        # Multipole constraint: C_j = sum_r a_jr * dens_r.
        # Written DIRECTLY as a contraction with ``dens`` to make the linear
        # dependence on the density constraints explicit (see multipoles3D.py).
        if self.n_multipole > 0:
            mp = self.mp_coefs @ dens
            return np.concatenate((dens, orto, mp))

        return np.concatenate((dens, orto))

    # ─────────────────────────────────────────────────────────────────────────
    # Jacobian
    # ─────────────────────────────────────────────────────────────────────────
    # IPOPT receives the sparsity pattern once (jacobianstructure) and then, at
    # each iteration, a flat vector of values (jacobian) with NO check that the
    # two match: both are built with identical loops, in the same order
    # (density blocks orbital by orbital, then orthonormality, then multipoles).
    # Any change to one MUST be mirrored in the other.

    def jacobianstructure(self):
        """IPOPT callback: (rows, cols) of the non-zero Jacobian entries."""
        rows = []
        cols = []
        # Density: one row per LIVE node, one column per orbital on that node.
        # Row c (0..n_live-1) is the constraint on node live_nodes[c].
        for k in range(self.n_orbitals):
            base = k * self.n_points
            for c, r in enumerate(self.live_nodes):
                rows.append(c)
                cols.append(base + r)
        # Orthonormality: rows starting at n_live; each row is dense over the
        # blocks of the orbitals involved (all nodes).
        for k, (i, j) in enumerate(self.pairs):
            counter = self.n_live + k
            i_index = i * self.n_points
            j_index = j * self.n_points
            for r in range(self.n_points):
                if i == j:
                    rows.append(counter)
                    cols.append(i_index + r)
                else:
                    rows.append(counter); cols.append(i_index + r)
                    rows.append(counter); cols.append(j_index + r)
        # Multipoles: rows starting at n_live + n_pairs. Each row touches ALL
        # orbitals on ALL live nodes -> dense row (n_orbitals*n_live entries),
        # unlike the density constraints, which are sparse.
        for j in range(self.n_multipole):
            counter = self.n_live + len(self.pairs) + j
            for k in range(self.n_orbitals):
                base = k * self.n_points
                for r in self.live_nodes:
                    rows.append(counter)
                    cols.append(base + r)
        return np.array(rows, dtype=int), np.array(cols, dtype=int)

    def jacobian(self, x):
        """IPOPT callback: values of the Jacobian, in the order of jacobianstructure()."""
        x4 = np.reshape(
            x, (self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z))
        x = np.reshape(x4, (self.n_orbitals, self.n_points))

        rows, cols = self.jacobianstructure()   # only its length is needed here
        data = np.zeros(len(rows), dtype=float)
        ptr  = 0

        # Density: d/df_kr (sum occ_k f_kr^2 - 1) = 2 occ_k f_kr, live nodes only.
        for k in range(self.n_orbitals):
            deg = self.orbital_set[k].occupation
            for r in self.live_nodes:
                data[ptr] = 2 * deg * x[k, r]
                ptr += 1

        # Orthonormality: d/df_i,r [ sum_r rho_r f_i,r f_j,r w_r ] = rho_r f_j,r w_r
        # (factor 2 on the diagonal pairs i == j).
        rho_flat = self.tab_rho.ravel()
        w_flat   = self.w3D.ravel()
        for kk, (i, j) in enumerate(self.pairs):
            for r in range(self.n_points):
                if i == j:
                    data[ptr] = 2 * rho_flat[r] * x[j, r] * w_flat[r]
                    ptr += 1
                else:
                    data[ptr] = rho_flat[r] * x[j, r] * w_flat[r]; ptr += 1
                    data[ptr] = rho_flat[r] * x[i, r] * w_flat[r]; ptr += 1

        # Multipoles: dC_j/df_kr = a_jr * 2 occ_k f_kr (live nodes only).
        # This is exactly a_jr times row r of the density Jacobian: the linear
        # dependence is visible row by row.
        for j in range(self.n_multipole):
            a_j = self.mp_coefs[j]
            for k in range(self.n_orbitals):
                deg = self.orbital_set[k].occupation
                block = 2.0 * deg * x[k, self.live_nodes] * a_j
                data[ptr:ptr + self.n_live] = block
                ptr += self.n_live
        return data

    # ─────────────────────────────────────────────────────────────────────────
    # Exact Hessian of the Lagrangian (thesis Section 4.2, Appendix A.5-A.6)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_objective_hessian(self):
        """Precompute M, the constant per-orbital Hessian of the objective / (2 occ_k).

        For each orbital the objective is the quadratic form
            f^T M f = sum_r w_eff_r [ (alpha rho + beta)|grad f|^2
                                      + alpha rho f^2 |grad log s|^2
                                      + 2 alpha rho f (grad f . grad log s) ]
        whose Hessian (with respect to f) is 2M, with
            M = sum_a D_a^T diag((alpha rho + beta) w_eff) D_a          (kinetic)
              + diag(alpha rho |grad log s|^2 w_eff)                     (local)
              + sum_a [ diag(c_a) D_a + D_a^T diag(c_a) ],
                c_a = alpha rho (grad log s)_a w_eff                     (mixed)
        (thesis Eq. A.12). The 3D operators D_a are Kronecker products of the
        1D matrices with identities.
        """
        from scipy import sparse
        nx, ny, nz = self.n_points_x, self.n_points_y, self.n_points_z
        Ix, Iy, Iz = sparse.identity(nx), sparse.identity(ny), sparse.identity(nz)

        if self.use_lagrange_mesh:
            dx1, dy1, dz1 = self.mesh.Dx, self.mesh.Dy, self.mesh.Dz
        else:
            dx1, dy1, dz1 = self.Dx_mat, self.Dy_mat, self.Dz_mat

        # 3D derivative operators on the flattened (C-order) node index.
        Dx = sparse.kron(sparse.csr_matrix(dx1), sparse.kron(Iy, Iz))
        Dy = sparse.kron(Ix, sparse.kron(sparse.csr_matrix(dy1), Iz))
        Dz = sparse.kron(Ix, sparse.kron(Iy, sparse.csr_matrix(dz1)))
        Da = [Dx.tocsr(), Dy.tocsr(), Dz.tocsr()]

        alpha, beta = self.alpha, self.beta
        rho_f = self.tab_rho.ravel()
        w_eff = (self.w3D * self.active_mask).ravel()
        glog  = [self.grad_log_s[..., a].ravel() for a in range(3)]
        nglog = self.norm_grad_log_s.ravel()

        W1 = (alpha * rho_f + beta) * w_eff
        M  = sum(Da[a].T @ sparse.diags(W1) @ Da[a] for a in range(3))   # kinetic
        M  = M + sparse.diags(alpha * rho_f * nglog * w_eff)               # local
        for a in range(3):                                                 # mixed
            c = alpha * rho_f * glog[a] * w_eff
            Dc = sparse.diags(c)
            M  = M + Dc @ Da[a] + Da[a].T @ Dc

        self._M_obj = (M + M.T) * 0.5          # symmetrise (numerical robustness)
        self._M_obj = self._M_obj.tocsr()

    def _build_hessian_structure(self):
        """Sparse structure (lower triangle) of the Hessian of the Lagrangian.

        The order of the entries MUST match _hessian_vectorized:
        for each orbital k, the tril entries of M_obj (same COO order);
        then, for each pair i != j, the nodes r of the cross block.
        """
        N  = self.n_points
        no = self.n_orbitals
        rows, cols = [], []

        Mcoo = self._M_obj.tocoo()
        tril = Mcoo.row >= Mcoo.col
        Mrow, Mcol = Mcoo.row[tril], Mcoo.col[tril]
        self._M_tril_row = Mrow            # cached for filling the values
        self._M_tril_col = Mcol
        self._M_tril_dat = Mcoo.data[tril]
        self._M_tril_diag = (Mrow == Mcol)

        # Diagonal blocks (one per orbital), same pattern as M.
        for k in range(no):
            off = k * N
            rows.append(off + Mrow); cols.append(off + Mcol)

        # Off-diagonal blocks from orthonormality (i != j): diagonal in the node.
        for (i, j) in self.pairs:
            if i == j:
                continue
            lo, hi = max(i, j), min(i, j)
            r = np.arange(N)
            rows.append(lo * N + r); cols.append(hi * N + r)

        self._hess_rows = np.concatenate(rows).astype(int)
        self._hess_cols = np.concatenate(cols).astype(int)

    def hessianstructure(self):
        """IPOPT callback: (rows, cols) of the lower triangle of the Hessian."""
        return self._hess_rows, self._hess_cols

    def hessian(self, x, lagrange, obj_factor):
        """IPOPT callback: Hessian of the Lagrangian, obj_factor*Hess(Phi) + sum lambda_c Hess(c).

        The objective is quadratic -> Hess(Phi) is constant (blocks 2 occ_k M_obj).
        Density constraints: diagonal 2 occ_k lambda. Orthonormality: i-j blocks.
        The order of the data matches hessianstructure(). ``x`` is not needed.
        """
        return self._hessian_vectorized(obj_factor, lagrange)

    def _hessian_vectorized(self, obj_factor, lagrange):
        N  = self.n_points
        no = self.n_orbitals
        rho_f = self.tab_rho.ravel()
        w_f   = self.w3D.ravel()

        # The density multipliers exist only on the live nodes: they are mapped
        # back onto the full grid (zero on the dead nodes) to index the diagonal.
        n_pairs = len(self.pairs)
        lam_dens_full = np.zeros(N)
        lam_dens_full[self.live_nodes] = lagrange[:self.n_live]
        lam_orto = lagrange[self.n_live:self.n_live + n_pairs]
        lam_mp   = lagrange[self.n_live + n_pairs:]

        # The multipole constraint has the SAME Hessian structure as the density
        # constraint (diagonal, 2 occ_k * coefficient): its effect is a simple
        # shift of the multiplier field,
        #     lam~_r = lam_r + sum_j mu_j a_jr
        # which is exactly the unfixed gauge discussed in multipoles3D.py.
        if self.n_multipole > 0:
            lam_dens_full[self.live_nodes] += lam_mp @ self.mp_coefs

        Mrow      = self._M_tril_row
        Mdat      = self._M_tril_dat
        diag_mask = self._M_tril_diag

        data = []
        for k in range(no):
            occ_k = self.orbital_set[k].occupation
            block = obj_factor * 2.0 * occ_k * Mdat.copy()
            # density: diagonal 2 occ_k lambda_r (lambda already zero on dead nodes)
            add = 2.0 * occ_k * lam_dens_full
            # orthonormality i == i on orbital k: 2 lambda rho_r w_r
            for p, (i, j) in enumerate(self.pairs):
                if i == j and i == k:
                    add = add + 2.0 * lam_orto[p] * rho_f * w_f
            block[diag_mask] += add[Mrow[diag_mask]]
            data.append(block)

        # orthonormality i != j: cross block, lambda rho_r w_r per node r
        for p, (i, j) in enumerate(self.pairs):
            if i == j:
                continue
            data.append(lam_orto[p] * rho_f * w_f)

        return np.concatenate(data)

    # ─────────────────────────────────────────────────────────────────────────
    # Solve
    # ─────────────────────────────────────────────────────────────────────────

    def solve(self):
        """Run IPOPT from ``self.getStartingPoint()`` and store/write the results.

        Returns
        -------
        results : dict with keys
            'status'   IPOPT status message
            'x'        solution vector f (flattened, n_orb * n_points)
            'u'        orbitals phi_k = f_k sqrt(rho), shape (n_orb, Nx, Ny, Nz)
            'obj'      final objective value T_alpha + T_beta [MeV]
            'lagrange' constraint multipliers (mult_g); the first n_live are
                       the density multipliers on ``self.live_nodes``
            'summary', 'grid', 'start'
        info : the raw dictionary returned by cyipopt.
        """
        st = self.getStartingPoint()
        x, info = super().solve(st)
        print(info['status_msg'])
        self.results = dict()
        keys    = ['status', 'x', 'u', 'obj', 'lagrange', 'summary', 'grid', 'start']
        entries = [info['status_msg'], x, self.getU(x), info['obj_val'],
                   info['mult_g'], str(self), self.grid, st]
        for k, ee in zip(keys, entries):
            self.results[k] = ee
        saveData(self.datafile, self.results)

        # Plain-text dump of f_k and phi_k: "x y z value", one block per orbital.
        with open(self.f_file, 'w') as fx, open(self.phi_file, 'w') as fu:
            u  = self.getU(x)
            x4 = np.reshape(
                x, (self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z))
            X, Y, Z = np.meshgrid(*self.grid, indexing='ij')
            Xf, Yf, Zf = X.ravel(), Y.ravel(), Z.ravel()
            for k in range(self.n_orbitals):
                for ff in (fx, fu):
                    ff.write(f"# {self.orbital_set[k].getName()}\n")
                x_flat = x4[k].ravel()
                u_flat = u[k].ravel()
                for xx, yy, zz, psi, uu in zip(Xf, Yf, Zf, x_flat, u_flat):
                    fx.write(f"{xx:.6e}\t{yy:.6e}\t{zz:.6e}\t{psi:.18E}\n")
                    fu.write(f"{xx:.6e}\t{yy:.6e}\t{zz:.6e}\t{uu:.18E}\n")

        self.kinetic = float(self.results['obj'])
        return self.results, info

    # ─────────────────────────────────────────────────────────────────────────
    # IPOPT options (thesis Section 4.3)
    # ─────────────────────────────────────────────────────────────────────────

    def setIpopt(self, max_iter, rel_tol, constr_viol, dual_inf_tol, exact_hess=True):
        """Set variable bounds (with frozen dead nodes), constraint bounds and options."""
        self.max_iter     = max_iter
        self.rel_tol      = rel_tol
        self.constr_viol  = constr_viol
        self.exact_hess   = exact_hess
        self.dual_inf_tol = dual_inf_tol
        # Bounds [-1, 1] on the live nodes: deliberately looser than what the
        # density constraint implies (|f_k| <= 1/sqrt(q_k)), so that the barrier
        # terms stay away from the solution and do not perturb the multipliers.
        ub_x = np.ones(self.n_variables)
        lb_x = -1. * ub_x

        # On the dead nodes (active_mask = 0) the variables appear neither in
        # the objective (weight w_eff = 0) nor in the constraints. They must be
        # fixed to remove them from the KKT system, BUT fixing them at 0 creates
        # a JUMP f: plateau -> 0 at the live/dead boundary, which the dense
        # derivative matrix propagates as a spurious |grad f|^2 (for A = 2:
        # +13 MeV on the objective). They are fixed instead at the PLATEAU
        # value f = 1/sqrt(sum occ_k), which continues the physical solution
        # without jumps (Eq. 3.14). This holds for both grids.
        occ_tot   = sum(o.occupation for o in self.orbital_set)
        plateau   = 1.0 / np.sqrt(occ_tot)
        mask_flat = self.active_mask.ravel()
        dead      = np.where(mask_flat == 0)[0]
        self._plateau_value = plateau          # also used in getStartingPoint
        for k in range(self.n_orbitals):
            base = k * self.n_points
            lb_x[base + dead] = plateau
            ub_x[base + dead] = plateau

        # All constraints are equalities: cl = cu = 0.
        super().__init__(n=self.n_variables, m=self.n_constr,
                         lb=lb_x, ub=ub_x,
                         cl=np.zeros(self.n_constr), cu=np.zeros(self.n_constr))
        self.setSolverOptions(exact_hess)

    def setSolverOptions(self, exact_hess):
        """Baseline IPOPT configuration (thesis Section 4.3).

        Further options (e.g. ``mu_init``, ``acceptable_iter``) are added by
        the test scripts with ``prob.add_option(...)`` after construction.
        """
        self.add_option(b'mu_strategy', b'adaptive')
        self.add_option(b'max_iter', self.max_iter)
        self.add_option('tol', self.rel_tol)
        self.add_option(b'constr_viol_tol', self.constr_viol)
        # HISTORY NOTE on dual_inf_tol: it used to be stored in
        # self.dual_inf_tol but was NEVER passed to IPOPT, which therefore used
        # its own default (= 1). With that value the termination test is
        # satisfied very early: on the uniform grid at A = 4 the solver stopped
        # at iteration 3 with dual infeasibility 7.17e-3, at a NON-stationary
        # point. That point is physical (T_direct/T_alpha = 0.997, RMSE 2.0 MeV
        # on the potential), whereas the true minimum of the discrete functional
        # is spurious (T_alpha = 22.0 MeV, below the von Weizsaecker bound,
        # RMSE 52.7 MeV): early termination acted as an accidental regulariser.
        # Now that the option is passed the behaviour changes: with tight
        # tolerances the uniform-grid runs converge to the spurious minimum.
        # This is not a regression but the correct behaviour of the solver on a
        # functional whose minimiser is not the physical solution; if early
        # termination is wanted it must be made explicit (a declared max_iter,
        # or a loose and documented dual_inf_tol) rather than depend on an
        # option that is silently dropped. See thesis Section 5.3, Figs. 5.9-5.10.
        self.add_option(b'dual_inf_tol', self.dual_inf_tol)
        self.add_option(b"output_file", b"ipopt.out")
        # The Hessian of the OBJECTIVE is constant, but that of the LAGRANGIAN
        # depends on the multipliers, hence "no".
        self.add_option(b"hessian_constant", b"no")
        if exact_hess:
            self.add_option(b"hessian_approximation", b"exact")
        else:
            self.add_option(b'hessian_approximation', b'limited-memory')

        if self.use_lagrange_mesh:
            # Enable detection of linearly dependent constraint rows and make
            # MUMPS more robust to near-singularity, to avoid factorisation
            # failures ("Error=-3").
            self.add_option(b"dependency_detector", b"mumps")
            self.add_option(b"mumps_pivtol", 1e-6)
            self.add_option(b"mumps_pivtolmax", 0.1)
            # No relaxation of the bounds: the dead nodes are frozen by
            # lb = ub, and any relaxation would turn them back into free
            # variables (flat directions of the KKT system).
            self.add_option(b"bound_relax_factor", 0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Starting point (thesis Eq. 4.7)
    # ─────────────────────────────────────────────────────────────────────────

    def getStartingPoint(self):
        """
        Build f_k = phi_norm_k / sqrt(rho_target) from the spherical HO orbitals.

        If rho_target is built from the same phi_norm_k (as in the tests), then
        sum_k occ_k phi_norm_k^2 = rho_target pointwise, hence
        sum_k occ_k f_k^2 = 1 on every node: the density constraint is
        satisfied exactly.

        The orbitals are normalised with the mesh weights w3D, not with a
        constant dV: on the Lagrange mesh the weights vary by orders of
        magnitude and dV would give a starting point with the wrong norm.

        The dead nodes (rho < 1e-10 * rho_max) are set to the plateau value:
        otherwise f_k ~ phi_HO(r_far)/sqrt(rho_floor) ~ 1e-8/sqrt(1e-14) ~ 100,
        and the global matrix D would give |grad f|^2 ~ 1e6, T_beta ~ 1e7 MeV.

        NOTE: ``solve()`` always calls this method. To start from a different
        point, override it on the instance, e.g.
        ``prob.getStartingPoint = lambda: my_start.ravel()``.
        """
        st = np.zeros(
            (self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z))
        x, y, z  = self.grid
        rho_safe  = np.maximum(self.tab_rho, 1e-14 * self.tab_rho.max())
        dead_mask = (self.tab_rho < 1e-10 * self.tab_rho.max())

        for j, oo in enumerate(self.orbital_set):
            wf  = HO3D(oo.nx, oo.ny, oo.nz,
                       b_x(self.A), b_y(self.A), b_z(self.A), self.dV)
            phi = wf(x, y, z)

            # Normalise with the mesh weights (w3D = dV constant on the uniform grid)
            norm2    = np.sum(self.w3D * phi**2)
            phi_norm = phi / np.sqrt(norm2)

            f = phi_norm / np.sqrt(rho_safe)

            # On the dead nodes use the PLATEAU value (not 0): avoids the jump
            # f: plateau -> 0 at the boundary, which D would propagate as a
            # spurious gradient. Consistent with the bounds fixed in setIpopt.
            plateau = getattr(self, "_plateau_value", None)
            if plateau is None:
                occ_tot = sum(o.occupation for o in self.orbital_set)
                plateau = 1.0 / np.sqrt(occ_tot)
            f[dead_mask] = plateau

            st[j] = f

        return st.ravel()

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def getRhoFromData(self, x, y, z, rho):
        """Cubic interpolator of a density given on the grid (x, y, z)."""
        return get_rho_from_data(x, y, z, rho)

    def getPairs(self):
        """List of orbital pairs [i, j] with j <= i for the orthonormality constraints.

        With a single orbital the list is empty: its normalisation already
        follows from the density constraint.
        """
        ll = []
        if self.n_orbitals == 1:
            return []
        for i in range(self.n_orbitals):
            for j in range(i + 1):
                ll.append([i, j])
        return ll

    def getU(self, x):
        """Orbitals phi_k = f_k sqrt(rho) from the solution vector, shape (n_orb, Nx, Ny, Nz)."""
        x = np.reshape(
            x, (self.n_orbitals, self.n_points_x, self.n_points_y, self.n_points_z))
        u = np.zeros_like(x)
        for k in range(self.n_orbitals):
            u[k] = x[k] * np.sqrt(self.tab_rho)
        return u

    def getXformU(self, u):
        """Inverse of getU: flattened f_k = phi_k / sqrt(rho) from the orbitals."""
        x = np.zeros_like(u)
        for k in range(self.n_orbitals):
            x[k] = u[k] / np.sqrt(self.tab_rho)
        return x.flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_rho_from_data(x, y, z, rho):
    """Return a callable p -> rho(p) interpolating a density tabulated on (x, y, z)."""
    from scipy.interpolate import RegularGridInterpolator
    points = (x, y, z)
    assert rho.shape == (len(x), len(y), len(z))
    interp_rho = RegularGridInterpolator(
        points, rho, method='cubic', bounds_error=True)
    return lambda p: interp_rho(p)
