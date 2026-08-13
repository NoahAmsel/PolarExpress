"""
PolarExpress: optimal matrix polar decomposition via polynomial compositions.

Reference: *The Polar Express: Optimal Matrix Sign Methods and Their Application
to the Muon Algorithm* (arXiv:2505.16932).

Quick start
-----------
>>> from polar_express import PolarExpress
>>> U = PolarExpress(G, steps=8)   # approximate polar factor of G (requires torch)

Coefficient utilities
---------------------
>>> from polar_express import optimal_composition, save_coefficients
>>> coeffs = optimal_composition(l=1e-3, num_iters=8, degree=5)
>>> save_coefficients(coeffs, "my_coeffs", degree=5)

Standalone (no-torch) version
------------------------------
>>> from polar_express.basic import PolarExpress as PolarExpressBasic
"""

from polar_express._core import (
    # Polynomial solvers
    optimal_cubic,
    optimal_quintic,
    # Composition & I/O
    optimal_composition,
    save_coefficients,
    load_coefficients,
    # Default coefficient lists
    coeffs_list,
    # Matrix functions (require torch)
    PolarExpress,
)

__all__ = [
    "optimal_cubic",
    "optimal_quintic",
    "optimal_composition",
    "save_coefficients",
    "load_coefficients",
    "coeffs_list",
    "PolarExpress",
]

__version__ = "0.1.0"
