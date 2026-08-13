"""
Shared plotting/runner code for the convergence examples.

Used by both `convergence_test.py` (fixed d=5 comparison) and
`generate_coefficients.py` (custom coefficient sets), so that both scripts
render matching figures from the same underlying kernels.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "apply_plot_style",
    "NS_COEFFS",
    "DEFAULT_COLOURS",
    "DEFAULT_STYLES",
    "pad_coeffs",
    "pe_step",
    "orthogonality_error",
    "run_one",
    "run_trials",
    "plot_convergence",
]

# Newton-Schulz's fixed cubic:  p(x) = 1.5x - 0.5x^3
NS_COEFFS = (1.5, -0.5)

# Method colours/styles shared across all figures; extend for custom methods
# with a colour not already used here (see generate_coefficients.py).
DEFAULT_COLOURS = {
    "Newton-Schulz":      "#d62728",   # red
    "PolarExpress (d=5)": "#1f77b4",   # blue
}
DEFAULT_STYLES = {
    "Newton-Schulz":      "-",
    "PolarExpress (d=5)": "-",
}


def apply_plot_style():
    """Matplotlib style matching the paper: thick lines, uncluttered, readable legend."""
    mpl.rcParams.update({
        "font.family":         "serif",
        "mathtext.fontset":    "cm",
        "font.size":           12,
        "axes.titlesize":      13,
        "axes.labelsize":      12,
        "legend.fontsize":     11,
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "lines.linewidth":     2.5,
        "axes.linewidth":      1.2,
        "xtick.major.width":   1.2,
        "ytick.major.width":   1.2,
        "xtick.minor.width":   0.8,
        "ytick.minor.width":   0.8,
        "axes.grid":           True,
        "grid.linestyle":      "--",
        "grid.linewidth":      0.6,
        "grid.alpha":          0.35,
        "legend.framealpha":   0.95,
        "legend.edgecolor":    "0.7",
        "figure.dpi":          150,
        "savefig.dpi":         300,
        "savefig.bbox":        "tight",
    })


# ---------------------------------------------------------------------------
# Iteration kernel (pure numpy, float64)
# ---------------------------------------------------------------------------

def pad_coeffs(coeffs, n):
    """Pad/truncate a coefficient sequence to exactly n entries, repeating the last."""
    coeffs = list(coeffs)
    return coeffs[:n] + [coeffs[-1]] * max(0, n - len(coeffs))


def pe_step(X, coeffs):
    """One polynomial-iteration step:  X <- (c0*I + c1*A + c2*A² + ...) X,  A = X Xᵀ.

    Covers Newton-Schulz (coeffs = NS_COEFFS) and any odd-degree PolarExpress
    polynomial (coeffs = (a, b), (a, b, c), ...) with a single implementation.
    """
    A = X @ X.T
    B = np.zeros_like(A)
    A_pow = np.eye(A.shape[0])
    for c in coeffs:
        B += c * A_pow
        A_pow = A_pow @ A
    return B @ X


def orthogonality_error(X):
    """||X^T X - I||_F  — zero iff X has orthonormal columns."""
    I = np.eye(X.shape[1])
    return np.linalg.norm(X.T @ X - I, "fro")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_one(G, n_iters, coeffs_map):
    """Run every method in coeffs_map on the same starting matrix G.

    Args:
        G:          Starting matrix.
        n_iters:    Number of iterations.
        coeffs_map: dict[method_name -> per-iteration coeff sequence, length n_iters].

    Returns:
        dict[method_name -> np.ndarray of orthogonality errors, length n_iters+1]
    """
    sigma_max = np.linalg.svd(G, compute_uv=False)[0]
    X0 = G / sigma_max

    errors = {name: [orthogonality_error(X0)] for name in coeffs_map}
    Xs = {name: X0.copy() for name in coeffs_map}

    for k in range(n_iters):
        for name, coeffs in coeffs_map.items():
            Xs[name] = pe_step(Xs[name], coeffs[k])
            errors[name].append(orthogonality_error(Xs[name]))

    return {name: np.array(v) for name, v in errors.items()}


def run_trials(coeffs_map, n_iters, l_min, size, trials, seed):
    """Run `trials` random matrices (log-uniform singular values in [l_min, 1])
    through every method in coeffs_map.

    Returns:
        List of dicts, one per trial, each as returned by run_one().
    """
    rng = np.random.default_rng(seed)
    errors_list = []
    for _ in range(trials):
        U, _ = np.linalg.qr(rng.standard_normal((size, size)))
        V, _ = np.linalg.qr(rng.standard_normal((size, size)))
        svs  = np.exp(np.linspace(np.log(l_min), 0.0, size))
        G    = U @ np.diag(svs) @ V.T
        errors_list.append(run_one(G, n_iters, coeffs_map))
    return errors_list


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence(ax, errors_list, methods, n_iters, l_min, matrix_size, n_trials):
    """Plot median convergence curves (shaded band = min/max over trials).

    Args:
        methods: list of (name, colour, style) tuples; name must be a key in
                  every dict in errors_list.
    """
    iters = np.arange(n_iters + 1)
    for name, colour, style in methods:
        mat = np.stack([e[name] for e in errors_list])  # (n_trials, n_iters+1)
        median = np.median(mat, axis=0)
        lo     = mat.min(axis=0)
        hi     = mat.max(axis=0)
        ax.semilogy(iters, median, color=colour, linestyle=style,
                    linewidth=2.5, label=name, zorder=3)
        if n_trials > 1:
            ax.fill_between(iters, lo, hi, color=colour, alpha=0.12, zorder=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|X_k^\top X_k - I\|_F$")
    cond = int(round(1 / l_min))
    ax.set_title(
        rf"$\kappa \approx {cond}$,  $n = {matrix_size}$"
        + (f",  {n_trials} trials" if n_trials > 1 else ""),
        pad=8,
    )
    ax.set_xlim(0, n_iters)
    ax.legend(loc="upper right", frameon=True)
