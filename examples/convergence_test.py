"""
Convergence comparison: Newton-Schulz vs PolarExpress degree 5.

Run:
    python convergence_test.py

Saves figures to figures/convergence_d<cond>.pdf (one per condition number tested).
Requires: numpy, matplotlib, scipy (all available via `module load python`).
torch is NOT required — everything runs in float64 numpy.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polar_express import optimal_composition

from _convergence_lib import (
    apply_plot_style,
    NS_COEFFS,
    DEFAULT_COLOURS,
    DEFAULT_STYLES,
    pad_coeffs,
    run_trials,
    plot_convergence,
)

apply_plot_style()

_METHODS = [
    ("Newton-Schulz",      DEFAULT_COLOURS["Newton-Schulz"],      DEFAULT_STYLES["Newton-Schulz"]),
    ("PolarExpress (d=5)", DEFAULT_COLOURS["PolarExpress (d=5)"], DEFAULT_STYLES["PolarExpress (d=5)"]),
]


def _coeffs_map(l_min, n_iters):
    coeffs5 = optimal_composition(l_min, n_iters, degree=5,
                                  safety_factor_eps=1e-2, cushion=0.02)
    return {
        "Newton-Schulz":      pad_coeffs([NS_COEFFS], n_iters),
        "PolarExpress (d=5)": pad_coeffs(coeffs5, n_iters),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iters",     type=int,   default=12,
                        help="Number of iterations to run (default: 12)")
    parser.add_argument("--size",        type=int,   default=64,
                        help="Matrix dimension n×n (default: 64)")
    parser.add_argument("--trials",      type=int,   default=5,
                        help="Number of random trials per condition (default: 5)")
    parser.add_argument("--conditions",  type=float, nargs="+",
                        default=[1e2, 1e3, 1e4],
                        help="Condition numbers to test (default: 1e2 1e3 1e4)")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--out-dir",     default="figures",
                        help="Directory for output figures (default: figures)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Matrix convergence ───────────────────────────────────────────────────
    for cond in args.conditions:
        l_min = 1.0 / cond
        print(f"\nCondition number κ ≈ {cond:.0f}  (l_min = {l_min:.2e})")

        errors_list = run_trials(_coeffs_map(l_min, args.n_iters), args.n_iters,
                                  l_min, args.size, args.trials, args.seed)

        fig, ax = plt.subplots(figsize=(6, 4))
        plot_convergence(ax, errors_list, _METHODS, args.n_iters, l_min,
                          args.size, args.trials)
        fig.tight_layout()

        fname = out_dir / f"convergence_cond{int(cond)}.pdf"
        fig.savefig(fname)
        print(f"  → {fname}")
        plt.close(fig)

    # Also produce a single combined figure (subplots, one per condition)
    n_conds = len(args.conditions)
    fig, axes = plt.subplots(1, n_conds, figsize=(5 * n_conds, 4), sharey=True)
    if n_conds == 1:
        axes = [axes]

    for ax, cond in zip(axes, args.conditions):
        l_min = 1.0 / cond
        errors_list = run_trials(_coeffs_map(l_min, args.n_iters), args.n_iters,
                                  l_min, args.size, args.trials, args.seed)
        plot_convergence(ax, errors_list, _METHODS, args.n_iters, l_min,
                          args.size, args.trials)
        if ax is not axes[0]:
            ax.set_ylabel("")
            ax.legend_.remove()

    # Single shared legend at the top, in its own reserved margin above the titles
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove()
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.99), frameon=True,
               fontsize=11, handlelength=2.5)
    combined_fname = out_dir / "convergence_combined.pdf"
    fig.savefig(combined_fname, bbox_inches="tight")
    print(f"\nCombined figure → {combined_fname}")
    plt.close(fig)


if __name__ == "__main__":
    main()
