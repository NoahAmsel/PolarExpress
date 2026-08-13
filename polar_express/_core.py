"""
Core PolarExpress implementation: polynomial solvers and matrix iteration.

Reference: https://arxiv.org/abs/2505.16932
"""

import json
import textwrap
from itertools import repeat
from math import inf, sqrt

import numpy as np

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Optimal polynomial solvers (Remez / simplified Remez algorithm)
# ---------------------------------------------------------------------------

def optimal_cubic(l, u):
    """Optimal odd cubic  p(x) = ax + bx^3  that best approximates 1 on [l, u].

    Returns:
        (a, b): coefficients.
    """
    alpha = sqrt(3 / (u**2 + l*u + l**2))
    beta  = 4 / (2 + l*u*(l + u)*(alpha**3))
    return (3/2)*alpha*beta, (-1/2)*(alpha**3)*beta


def optimal_quintic(l, u):
    """Optimal odd quintic  p(x) = ax + bx^3 + cx^5  that best approximates 1 on [l, u].

    Uses the simplified Remez algorithm (4-point equioscillation).

    Returns:
        (a, b, c): coefficients.
    """
    assert 0 <= l <= u
    if 1 - 5e-6 <= l / u:
        return (15/8)/u, (-10/8)/(u**3), (3/8)/(u**5)
    q = (3*l + u) / 4
    r = (l + 3*u) / 4
    E, old_E = inf, None
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array([
            [l, l**3, l**5,  1],
            [q, q**3, q**5, -1],
            [r, r**3, r**5,  1],
            [u, u**3, u**5, -1],
        ])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        q, r = np.sqrt((-3*b + np.array([-1, 1]) * sqrt(9*b**2 - 20*a*c)) / (10*c))
    return float(a), float(b), float(c)


# Map degree -> solver
_POLY_SOLVERS = {3: optimal_cubic, 5: optimal_quintic}


# ---------------------------------------------------------------------------
# Coefficient composition
# ---------------------------------------------------------------------------

def _eval_poly(coeffs, x):
    """Evaluate p(x) = c0·x + c1·x^3 + c2·x^5 + ... given coeffs = (c0, c1, ...)."""
    return sum(c * x**(2*i + 1) for i, c in enumerate(coeffs))


def optimal_composition(
    l,
    num_iters,
    degree=5,
    safety_factor_eps=0,
    cushion=0,
    save_to=None,
):
    """Compute a composition of optimal odd polynomials that maps [l, 1] towards 1.

    Args:
        l:                 Initial lower bound on singular values (after normalisation).
        num_iters:         Number of polynomial iterations.
        degree:            Polynomial degree: 3 or 5.
        safety_factor_eps: Extra contraction factor sf = 1 + eps applied to each
                           polynomial (except the last) for numerical stability.
        cushion:           Clamp the effective lower bound to cushion·u at each step,
                           then re-centre the polynomial to straddle 1 symmetrically.
        save_to:           If given, save the coefficients to ``save_to`` (without
                           extension). Creates ``<save_to>.json`` and ``<save_to>.py``.

    Returns:
        List of coefficient tuples, one per iteration.
    """
    assert degree in _POLY_SOLVERS, f"degree must be one of {list(_POLY_SOLVERS)}"
    poly_fn = _POLY_SOLVERS[degree]
    u  = 1.0
    sf = 1 + safety_factor_eps
    coefficients = []

    for it in range(num_iters):
        use_cushion = cushion > 0
        effective_l = max(l, cushion * u) if use_cushion else l
        coeffs = list(poly_fn(effective_l, u))

        if use_cushion and cushion * u > l:
            # Re-centre: polynomial was optimised for [cushion·u, u]; rescale so
            # that the midpoint of p([l, u]) equals 1.
            pl = _eval_poly(coeffs, l)
            pu = _eval_poly(coeffs, u)
            rescaler = 2 / (pl + pu)
            coeffs = [c * rescaler for c in coeffs]

        if it < num_iters - 1 and sf != 1.0:
            # Scale x -> x/sf, i.e. divide coefficient k by sf^(2k+1)
            coeffs = [c / sf**(2*i + 1) for i, c in enumerate(coeffs)]

        coefficients.append(tuple(coeffs))
        l = _eval_poly(coeffs, l)
        u = 2 - l
        l = min(l, u)  # guard: safety factor can over-contract when nearly converged

    if save_to is not None:
        save_coefficients(
            coefficients,
            save_to,
            degree=degree,
            params=dict(l=l, num_iters=num_iters, degree=degree,
                        safety_factor_eps=safety_factor_eps, cushion=cushion),
        )

    return coefficients


# ---------------------------------------------------------------------------
# Coefficient I/O
# ---------------------------------------------------------------------------

def save_coefficients(coefficients, path, degree=5, params=None):
    """Save polynomial coefficients to JSON and a copy-pasteable Python file.

    Creates two files:
        <path>.json   – machine-readable; load with load_coefficients()
        <path>.py     – paste-ready Python list literal

    Args:
        coefficients: List of coefficient tuples from optimal_composition().
        path:         File path without extension.
        degree:       Polynomial degree (for metadata only).
        params:       Dict of parameters to embed in the JSON header.
    """
    data = {
        "degree": degree,
        "params": params or {},
        "coefficients": [list(c) for c in coefficients],
    }
    json_path = f"{path}.json"
    with open(json_path, "w") as fh:
        json.dump(data, fh, indent=2)

    lines = [f"coeffs_list = [  # degree={degree}"]
    for tup in coefficients:
        lines.append("    (" + ", ".join(f"{v!r}" for v in tup) + "),")
    lines.append("]")
    py_src = textwrap.dedent(f"""\
        # PolarExpress coefficients — generated by optimal_composition()
        # degree={degree}, params={params!r}
        # Paste this block into your script or load from '{path}.json'.
        #
        # Usage:
        #   from polar_express import PolarExpress
        #   # replace the module-level coeffs_list with this one, or
        #   coeffs = load_coefficients('{path}.json')
        """) + "\n".join(lines) + "\n"
    py_path = f"{path}.py"
    with open(py_path, "w") as fh:
        fh.write(py_src)

    print(f"Saved coefficients → {json_path}  and  {py_path}")


def load_coefficients(path):
    """Load coefficients saved by save_coefficients() or optimal_composition(save_to=...).

    Args:
        path: Path to the ``.json`` file.

    Returns:
        List of coefficient tuples.
    """
    with open(path) as fh:
        data = json.load(fh)
    return [tuple(c) for c in data["coefficients"]]


# ---------------------------------------------------------------------------
# Default coefficients (degree 5, l=1e-3)
# ---------------------------------------------------------------------------

coeffs_list = optimal_composition(l=1e-3, num_iters=10, safety_factor_eps=1e-2, cushion=0.02)


# ---------------------------------------------------------------------------
# PolarExpress functions (require torch)
# ---------------------------------------------------------------------------

def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PolarExpress requires PyTorch. Install it with:\n"
            "    pip install torch\n"
            "or see https://pytorch.org/get-started for GPU builds."
        )


if _TORCH_AVAILABLE:
    @_torch.compile
    def PolarExpress(G: _torch.Tensor, steps: int) -> _torch.Tensor:
        """Compute the polar factor of G using a composition of quintic (degree-5) polynomials.

        Args:
            G:     Input matrix (or batch of matrices), shape (..., m, n).
            steps: Number of polynomial iterations. 5–8 suffices for most uses.

        Returns:
            Approximate polar factor U with orthonormal columns/rows, same shape as G.
        """
        assert G.ndim >= 2
        X = G.bfloat16()
        if G.size(-2) > G.size(-1):
            X = X.mT
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
        hs = coeffs_list[:steps] + list(repeat(coeffs_list[-1], steps - len(coeffs_list)))
        for a, b, c in hs:
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X  # X <- aX + bX^3 + cX^5
        if G.size(-2) > G.size(-1):
            X = X.mT
        return X

else:  # pragma: no cover
    def PolarExpress(G, steps):  # type: ignore[misc]
        """Stub — raises ImportError when torch is not installed."""
        _require_torch()
