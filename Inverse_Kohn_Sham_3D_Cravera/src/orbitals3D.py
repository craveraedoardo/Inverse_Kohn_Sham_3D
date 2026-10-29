"""
orbitals3D.py
=============
Bookkeeping of the occupied single-particle states.

An ``Orbital`` is a Cartesian harmonic-oscillator state (nx, ny, nz) with
occupation q = 2 (spin degeneracy; all test systems contain neutrons only).
``uncoupledHObasis`` lists the states in order of increasing major shell
N = nx + ny + nz, and ``getOrbitalSet`` fills them in that order (aufbau)
until the requested number of particles is reached:

    A = 2  ->  (000)
    A = 4  ->  (000), (100)
    A = 8  ->  (000), (100), (010), (001)    (closed p shell)

Excited or deformed configurations are built explicitly in the scripts that
need them (e.g. ``test_IKS_3D_deformato_eccitato_gh.py``) by passing a custom
``OrbitalSet``-like basis.
"""


class Orbital(object):
    """A single Cartesian HO state (nx, ny, nz) with occupation 2."""

    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.N = self.nx + self.ny + self.nz           # major shell number
        self.name = self.getName()
        # Degeneracy of the whole major shell N (spin included); informative only.
        self.degeneracy = 2 * (self.N + 1) * (self.N + 2) / 2
        self.occupation = 2                             # spin up + spin down

    def getName(self):
        """Short label of the state, written as a header line in the output files.

        NOTE: the label only encodes the shell N (and a rough tag), not the
        full (nx, ny, nz); it is a legacy of the original code and is used for
        output headers only.
        """
        ss = ""
        if self.N == 0: ss = "(0, 0, 0)"
        if self.N == 1: ss = "nx, ny, nz=1"
        if self.N == 2: ss = "3s"
        if self.N == 3: ss = "4s"
        name = str(self.N) + ss
        return name

    def __str__(self):
        return self.getName()


class OrbitalSet(object):
    """An ordered collection of ``Orbital`` objects.

    WARNING: the class is its own iterator (``__iter__`` returns ``self``), so
    a ``for`` loop consumes it and a second loop yields nothing until
    ``reset()`` is called. Code that needs to iterate more than once should
    materialise it first with ``list(...)`` (this is what ``iKS_problem`` and
    the test scripts do).
    """

    def __init__(self, orb_list):
        self.orb_list = orb_list
        self.index = 0
        self.n_particles = self.count_particles()

    def __iter__(self):
        return self

    def __next__(self):
        try: result = self.orb_list[self.index]
        except IndexError: raise StopIteration
        self.index += 1
        return result

    def __getitem__(self, k):
        return self.orb_list[k]

    def __len__(self):
        return len(self.orb_list)

    def count_particles(self):
        """Total number of particles, i.e. the sum of the occupations."""
        s = 0
        for oo in self.orb_list:
            s += oo.occupation
        return s

    def reset(self):
        """Rewind the internal iterator and recount the particles."""
        self.index = 0
        self.n_particles = self.count_particles()


class uncoupledHObasis(OrbitalSet):
    """Cartesian HO states ordered by major shell (N = 0, 1, 2 and part of N = 3)."""

    def __init__(self):
        orbs = [Orbital(0, 0, 0),
                # N = 1
                Orbital(1, 0, 0), Orbital(0, 1, 0), Orbital(0, 0, 1),
                # N = 2
                Orbital(2, 0, 0), Orbital(0, 2, 0), Orbital(0, 0, 2),
                Orbital(1, 1, 0), Orbital(1, 0, 1), Orbital(0, 1, 1),
                # N = 3 (partial)
                Orbital(3, 0, 0), Orbital(0, 3, 0), Orbital(0, 0, 3),
                Orbital(2, 1, 0), Orbital(2, 0, 1), Orbital(1, 2, 0)]
        super().__init__(orbs)


def getOrbitalSet(n_particles, basis=uncoupledHObasis()):
    """Fill the states of ``basis`` in order until ``n_particles`` are placed.

    Returns an ``OrbitalSet``. If the basis runs out of states (or the
    particle number does not fill the last orbital exactly) a message is
    printed and the partial set is returned.

    NOTE: the default ``basis`` is created once at import time and, being an
    iterator, is consumed by the first call. Always pass a fresh
    ``uncoupledHObasis()`` explicitly, as all the scripts in this repository do.
    """
    ll = []
    nn = n_particles
    for oo in basis:
        nn -= oo.occupation
        if nn >= 0:
            ll.append(oo)
            if nn == 0: break
        else:
            print("Most orbitals are required")
            break
    return OrbitalSet(ll)
