"""
utility3D.py
============
Input/output helpers.

``iKS_problem.solve`` stores a dictionary with the full result of a run
(IPOPT status, solution vector, orbitals, multipliers, grid, starting point)
in ``test_3D_data/<output_folder>/data`` through ``saveData``; it can be read
back with ``loadData`` for post-processing.
"""
import pickle


def saveData(filename, data):
    """Save an (almost) arbitrary Python object to disk with pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def loadData(filename):
    """Load and return an object saved with ``saveData``."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
