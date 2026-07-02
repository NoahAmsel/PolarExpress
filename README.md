# PolarExpress

This package implements the PolarExpress method from the paper
[**The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm**](https://arxiv.org/abs/2505.16932).

## Installation

```bash
pip install .
```

This installs the core package (numpy only). To also install PyTorch for the
`PolarExpress` GPU function:

```bash
# CPU-only torch (small, good for testing):
pip install ".[torch]"

# For GPU builds follow https://pytorch.org/get-started, then:
pip install .
```

To run the convergence benchmarks in `examples/`:

```bash
pip install ".[examples]"
python examples/convergence_test.py
```

## Quick start

```python
from polar_express import PolarExpress

U = PolarExpress(G, steps=8)   # approximate polar factor, same shape as G
```

`PolarExpress` works on any `torch.Tensor` of shape `(..., m, n)` and returns
a tensor with orthonormal columns (if m ≥ n) or rows (if m < n).

## Coefficient utilities

Coefficients are computed once at import time. To generate and save custom
coefficients (e.g. different condition number or degree):

```python
from polar_express import optimal_composition, save_coefficients, load_coefficients

coeffs = optimal_composition(l=1e-4, num_iters=12, degree=5,
                             safety_factor_eps=1e-2, cushion=0.02)
save_coefficients(coeffs, "my_coeffs", degree=5)

# Later:
coeffs = load_coefficients("my_coeffs.json")
```

`save_coefficients` writes both a `.json` file (for `load_coefficients`) and a
`.py` file with a paste-ready Python list, so it is easy to ship coefficients
alongside a script.

To generate a custom coefficient set from the command line without writing any
code, edit the parameters at the top of `examples/generate_coefficients.py`
(`L`, `NUM_ITERS`, `DEGREE`, `SAFETY_FACTOR_EPS`, `CUSHION`) and run it:

```bash
pip install ".[examples]"
python examples/generate_coefficients.py
```

This prints the generated polynomials, saves them to `figures/<OUT_NAME>.json`
and `figures/<OUT_NAME>.py`, and plots their orthogonality-error convergence
against Newton-Schulz to `figures/<OUT_NAME>_convergence.pdf` — using the same
runner/plotting code as `examples/convergence_test.py`.

## Standalone (no-install) version

If you just want to drop a single file into your project:

```bash
cp polar_express/basic.py your_project/polar_express_basic.py
```

`basic.py` contains a self-contained implementation with hardcoded coefficients
and no dependencies beyond `torch`.

## Running the examples

```bash
# Convergence comparison: Newton-Schulz vs PolarExpress degree 5
# Figures are saved to figures/
python examples/convergence_test.py

# Generate a custom coefficient set (edit parameters at the top of the file first)
# and plot its convergence, saved to figures/
python examples/generate_coefficients.py

# Interactive polynomial composition visualizer (open in a browser)
open examples/Visualize.html   # macOS
xdg-open examples/Visualize.html  # Linux
```

## Reproducing the paper experiments

See the `polar` branch of the [GPT-opt repo](https://github.com/modichirag/GPT-opt/tree/polar).
