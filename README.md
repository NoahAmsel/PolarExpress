# PolarExpress

This repo implements the PolarExpress method from the paper
[**The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm**](https://arxiv.org/abs/2505.16932).

To use it, simply copy the file `polar_express.py` into your project. Requires `numpy` and `torch`.

You may also be interested in [`gram-newton-schulz`](https://github.com/Dao-AILab/gram-newton-schulz).

## Quick start
`PolarExpress` approximates the **polar factor** of a matrix. If $G = U \Sigma V^T$ is the (reduced) singular value decomposition of $G$, then its polar factor is $\text{polar}(G) = UV^T$.

```python
import torch
from polar_express import PolarExpress

# Generate a random 100 x 50 matrix
G = torch.randn(100, 50)

# Approximate the polar factor, same shape as G
P = PolarExpress(G, steps=8)

# Verify that the columns are approximately orthonormal
error = torch.linalg.norm(P.T @ P - torch.eye(G.shape[1])).item()
print(error)
```

`PolarExpress` works on any `torch.Tensor` of shape `(..., m, n)` and returns
a tensor with approximately orthonormal columns (if m ≥ n) or rows (if m < n).

## Customizing PolarExpress
The coefficients of the polynomials used by PolarExpress are generated once at import time and stored in the variable `coeffs_list`:
```python
coeffs_list = optimal_composition(l=1e-3, num_iters=10, degree=5, safety_factor_eps=1e-2, cushion=0.02)
```
We designed them to work well out-of-the-box for training LLMs with Muon, but you can easily customize the coefficients to better suit your problem by editing that^ line in `polar_express.py`.
You can control:
- `l`: This controls the range of singular values we care about: PolarExpress optimizes convergence for the interval $[\ell u, u]$, where $u$ is the Frobenius norm of the input. The algorithm will converge for *any* $\ell \in [0, 1]$, but setting it correctly ensures that convergence is as fast as possible. If $\ell$ is set too high, convergence will be slow for the first few iterations; if it is set too low, it will be slow for the middle iterations.
- `num_iters`: Ensure this is higher than number of `steps` you will use when calling `PolarExpress`.
- `degree`: We support degree 3 and 5. We find that degree 5 converges slightly faster in terms of FLOPs.
- `safety_factor_eps`: Limits the impact of floating point error. PolarExpress will converge even if singular values are perturbed by `safety_factor_eps`. Necessary for stability of degree-5 method in `bfloat16`. (In degree 3, it may not be necessary.) The safety factor is removed at the final iteration to preserve high accuracy.
- `cushion`: Another form of numerical hedge, inspired by the stability analysis of [Nakatsukasa and Higham [2012]](https://doi.org/10.1137/110857544). Increases accuracy of the final output, though it may not be essential.

## Polynomial visualizer
The notebook `plotsforpolar.ipynb` provides a visualization of the polynomials used by `PolarExpress`. To open the notebook:
```bash
jupyter notebook plotsforpolar.ipynb
```

## Reproducing the paper experiments

For experiments with Muon for training LLMs, see the `polar` branch of the [GPT-opt repo](https://github.com/modichirag/GPT-opt/tree/polar).

## Kernel-optimized implementation
The [`gram-newton-schulz`](https://github.com/Dao-AILab/gram-newton-schulz) package is a heavy-duty implementation of Newton Schulz / PolarExpress.
Features include:
- Custom GPU kernels
- [Gram Newton-Schulz](https://tridao.me/blog/2026/gram-newton-schulz/) reformulation for greater speed
- pip-installable package
