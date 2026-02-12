from itertools import repeat
from math import inf, sqrt

import numpy as np
import torch


def optimal_cubic(l, u):
    """Find an approximation to the constant function x -> 1
    of the form p(x) = a*x + b*x^3 that minimizes the maximum approximation error over the interval [l, u].

    Returns:
        a, b: coefficients of the optimal odd cubic approximant.
    """
    alpha = sqrt(3/(u**2 + l*u + l**2))
    beta = 4/(2 + l*u*(l + u)*(alpha**3))
    return (3/2)*alpha*beta, (-1/2)*(alpha**3)*beta


def optimal_quintic(l, u):
    """Use the simplified Remez algorithm to find an approximation to the constant function x -> 1
    of the form p(x) = a*x + b*x^3 + c*x^5 that minimizes the maximum approximation error over the interval [l, u].

    Returns:
        a, b, c: coefficients of the optimal odd quintic approximant.
    """
    assert 0 <= l <= u
    if 1 - 5e-6 <= l / u:
        # Above this threshold, the equioscillating polynomials 
        # is numerically equal to...
        return (15/8)/u, (-10/8)/(u**3), (3/8)/(u**5)
    # This initialization becomes exact as l -> u
    q = (3*l + u) / 4
    r = (l + 3*u) / 4
    E, old_E = inf, None
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array([
            [l, l**3, l**5, 1],
            [q, q**3, q**5, -1],
            [r, r**3, r**5, 1],
            [u, u**3, u**5, -1],
        ])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        q, r = np.sqrt((-3*b + np.array([-1, 1]) * 
                        sqrt(9*b**2 - 20*a*c)) / (10*c))
    return float(a), float(b), float(c)


def optimal_composition(l, num_iters, safety_factor_eps=0, cushion=0):
    u = 1
    assert 0 <= l <= u
    safety_factor = 1 + safety_factor_eps
    coefficients = []
    for iter in range(num_iters):
        a, b, c = optimal_quintic(max(l, cushion*u), u)
        if cushion*u > l:
            # Due to cushioning, this may be centered around 1 with 
            # respect to 0.024*u, u. Recenter it around 1 with respect 
            # to l, u, meaning find c so that 1 - c*p(l) = c*p(u) - 1:
            pl = a*l + b*l**3 + c*l**5
            pu = a*u + b*u**3 + c*u**5
            rescaler = 2/(pl + pu)
            a *= rescaler; b *= rescaler; c *= rescaler
        # Optionally incorporate safety factor here:
        if iter < num_iters - 1:  # don't apply to last polynomial
            a /= safety_factor; b /= safety_factor**3; c /= safety_factor**5
        coefficients.append((a, b, c))
        l = a*l + b*l**3 + c*l**5
        u = 2 - l
    return coefficients


coeffs_list = optimal_composition(l=1e-3, num_iters=10, safety_factor_eps=1e-2, cushion=0.02)
# print("Polar Express Coefficient Series:", *coeffs_list, sep="\n")


@torch.compile
def PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    hs = coeffs_list[:steps] + list( 
        repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX^3 + cX^5
    if G.size(-2) > G.size(-1): X = X.mT
    return X

