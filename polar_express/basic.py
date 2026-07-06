"""
Minimal, self-contained PolarExpress with hardcoded quintic coefficients.

Drop this file into any project and call PolarExpress(G, steps).
No dependency on the rest of this repo.
"""

from itertools import repeat
import torch

# Coefficients computed by optimal_composition(l=1e-3, num_iters=8, safety_factor_eps=1e-2, cushion=0.02)
# Each tuple is (a, b, c) for the polynomial  X <- aX + bX^3 + cX^5
_COEFFS = [
    (8.28721201814563,    -23.595886519098837, 17.300387312530933),
    (4.107059111542203,   -2.9478499167379106,  0.5448431082926601),
    (3.9486908534822946,  -2.908902115962949,   0.5518191394370137),
    (3.3184196573706015,  -2.488488024314874,   0.51004894012372),
    (2.300652019954817,   -1.6689039845747493,  0.4188073119525673),
    (1.891301407787398,   -1.2679958271945868,  0.37680408948524835),
    (1.8750014808534479,  -1.2500016453999487,  0.3750001645474248),
    (1.875,               -1.25,                0.375),  # all subsequent iters use this
]

# Apply safety factor 1.01 for numerical stability (excludes the final steady-state polynomial)
_COEFFS = [
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for (a, b, c) in _COEFFS[:-1]
] + [_COEFFS[-1]]


@torch.compile
def PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Compute the polar factor of G using a composition of quintic polynomials.

    Args:
        G:     Input matrix (or batch of matrices), shape (..., m, n).
        steps: Number of polynomial iterations. 5-8 is sufficient for most uses.

    Returns:
        Approximate polar factor U, same shape as G, with orthonormal columns/rows.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT  # work on the smaller gram matrix to reduce FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    hs = _COEFFS[:steps] + list(repeat(_COEFFS[-1], steps - len(_COEFFS)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX^3 + cX^5
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
