# PolarExpress
This repo implements the PolarExpress method from our paper, [The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm](https://arxiv.org/abs/2505.16932).
For now, simply copy `polar_express.py` in your repo, and then use the `PolarExpress` function.
Coefficients are generated upfront. You can adjust the safety factors `safety_factor_eps` and `cushion` as you like, but for now the degree is fixed to 5.
Note that if `safety_factor_eps > 0` the method may not converge all the way to full precision, though for deep learning applications this is not important.

If you wish to reproduce our experiments, see the `polar` branch of the [GPT-opt repo](https://github.com/modichirag/GPT-opt/tree/polar ).