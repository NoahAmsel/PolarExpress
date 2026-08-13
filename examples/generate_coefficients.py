"""
Generate a custom set of PolarExpress polynomial coefficients.

Edit the parameters below, then run:
    python examples/generate_coefficients.py

This will:
    1. Print the generated per-iteration polynomial coefficients.
    2. Save them to figures/<OUT_NAME>.json and figures/<OUT_NAME>.py
       (via polar_express.save_coefficients).
    3. Plot orthogonality-error convergence against Newton-Schulz using these
       coefficients, reusing the same runner/plotting code as convergence_test.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from polar_express import optimal_composition, save_coefficients

from _convergence_lib import (
    apply_plot_style,
    NS_COEFFS,
    DEFAULT_COLOURS,
    pad_coeffs,
    run_trials,
    plot_convergence,
)

# ---------------------------------------------------------------------------
# Parameters — edit these to generate a new coefficient set
# ---------------------------------------------------------------------------
L                 = 1e-4   # Initial lower bound on singular values (after normalisation)
NUM_ITERS         = 12     # Number of polynomial iterations
DEGREE            = 5      # Polynomial degree: 3 or 5
SAFETY_FACTOR_EPS = 1e-2   # Extra contraction margin for numerical stability
CUSHION           = 0.02   # Lower-bound clamp/re-centring factor

OUT_NAME = "my_coeffs"     # Coefficients saved to figures/<OUT_NAME>.json and .py

# Convergence-plot parameters
MATRIX_SIZE = 64           # Test matrix dimension n×n
TRIALS      = 5            # Number of random trials
SEED        = 42
OUT_DIR     = "figures"    # Output directory for coefficients and the plot


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Generate ──────────────────────────────────────────────────────────
    coeffs = optimal_composition(
        L, NUM_ITERS, degree=DEGREE,
        safety_factor_eps=SAFETY_FACTOR_EPS, cushion=CUSHION,
    )

    print(f"Generated {len(coeffs)} polynomials (degree={DEGREE}, l={L:.1e}):")
    for t, c in enumerate(coeffs, start=1):
        print(f"  p{t}: {c}")

    # ── 2. Save ──────────────────────────────────────────────────────────────
    save_coefficients(
        coeffs, out_dir / OUT_NAME, degree=DEGREE,
        params=dict(l=L, num_iters=NUM_ITERS, degree=DEGREE,
                    safety_factor_eps=SAFETY_FACTOR_EPS, cushion=CUSHION),
    )

    # ── 3. Plot convergence against Newton-Schulz ───────────────────────────
    apply_plot_style()

    method_name = f"PolarExpress (d={DEGREE}, l={L:.0e})"
    coeffs_map = {
        "Newton-Schulz": pad_coeffs([NS_COEFFS], NUM_ITERS),
        method_name:     pad_coeffs(coeffs, NUM_ITERS),
    }
    methods = [
        ("Newton-Schulz", DEFAULT_COLOURS["Newton-Schulz"], "-"),
        (method_name,     "#2ca02c", "--"),   # green, dashed — distinct from d=5 defaults
    ]

    errors_list = run_trials(coeffs_map, NUM_ITERS, L, MATRIX_SIZE, TRIALS, SEED)

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_convergence(ax, errors_list, methods, NUM_ITERS, L, MATRIX_SIZE, TRIALS)
    fig.tight_layout()

    fname = out_dir / f"{OUT_NAME}_convergence.pdf"
    fig.savefig(fname)
    print(f"\nConvergence plot → {fname}")
    plt.close(fig)


if __name__ == "__main__":
    main()
