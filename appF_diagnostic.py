from itertools import repeat

from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import torch


def rescale_PE_below_1(coeffs):
    scaled_coeffs = []
    max_val = 1
    for a, b, c in coeffs:
        a *= max_val
        b *= max_val**3
        c *= max_val**5
        max_val = a + b + c
        scaled_coeffs.append((a / max_val, b / max_val, c / max_val))
    return scaled_coeffs


class PolarExpressDiagnostic:

    def __init__(
        self,
        coeffs_name: str,
        steps: int,
        ambient_dtype,
        restarts: list[int] = [],
        force_symmetry: bool = True,  # Our CUDA kernels actually do enforce symmetry.
        xxt_dtype=None,
        xxt_posthoc_dtype=None,
        post_restart_ambient_dtype=None,
        qT_posthoc_dtype=None,
        qx_dtype=None,
        do_diagnostics=True,
    ):
        self.coeffs = dict(
            ns3=[(1.5, -0.5)],
            ns5=[((15/8) / 1.02, (-10/8) / (1.02**3), (3/8) / (1.02**5))],
            polar5=self.PE_coeffs_list,
            rescaled_polar5=self.rescaled_polar5,
        )[coeffs_name]
        self.coeffs = self.coeffs[:steps] + list( 
            repeat(self.coeffs[-1], steps - len(self.coeffs)))

        self.restarts = sorted(restarts)
        self.force_symmetry = force_symmetry
        self.do_diagnostics = do_diagnostics

        self.ambient_dtype = ambient_dtype
        self.mm_dtype = ambient_dtype
        self.xxt_dtype = self.mm_dtype if xxt_dtype is None else xxt_dtype
        self.xxt_posthoc_dtype = self.ambient_dtype if xxt_posthoc_dtype is None else xxt_posthoc_dtype
        self.post_restart_ambient_dtype = self.ambient_dtype if post_restart_ambient_dtype is None else post_restart_ambient_dtype
        self.qx_dtype = self.mm_dtype if qx_dtype is None else qx_dtype
        self.qT_posthoc_dtype = self.ambient_dtype if qT_posthoc_dtype is None else qT_posthoc_dtype

    def __call__(self, G: torch.Tensor) -> torch.Tensor:
        assert G.ndim >= 2
        assert G.dtype == torch.float64, "Input must be float64 for diagnostic purposes"
        assert G.size(-2) <= G.size(-1), f"Input must be short and fat, but size is {G.size()}"
        assert torch.linalg.matrix_norm(G, ord=2) < .999
        return self.appF(G)

    def appF(self, X: torch.Tensor) -> torch.Tensor:
        true_ambient_dtype = self.ambient_dtype  # stashing this to restore it later in case we use post_restart_ambient_dtype
        diagnostics = []
        if self.do_diagnostics:
            Xorig = X.clone()
            starting_left_svs, _, starting_right_svs = torch.linalg.svd(Xorig, full_matrices=False)
            starting_right_svs = starting_right_svs.mT  # svd returns V^T, not V
        X = X.to(self.ambient_dtype)
        for iter, coeff in enumerate(self.coeffs):
            if (iter == 0) or (iter in self.restarts):
                if iter > 0:
                    X = self.mm(Q.to(dtype=self.qT_posthoc_dtype).to(dtype=self.ambient_dtype), X, dtype=self.qx_dtype)  # apply the current Q to X before restarting
                    self.ambient_dtype = self.post_restart_ambient_dtype
                Q = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
                R = self.sym_mm(X, X.mT, dtype=self.xxt_dtype).to(dtype=self.xxt_posthoc_dtype).to(dtype=self.ambient_dtype)  # R = X @ X.mT
                # TODO: record eigenvectors of XXT at iter 0 to use for diagonalizing inside diagnostics
            Z = self.sym_polynomial(R, coeff)  # Z = aI + bR + cR^2
            if self.do_diagnostics:
                X_if_we_stopped_here = self.mm(Q.to(dtype=self.qT_posthoc_dtype).to(dtype=self.ambient_dtype), X, dtype=self.qx_dtype)
                diagnostics.append(
                    {
                        f"{name}_{k}": v
                        for name, M in (("R", R), ("Z", Z), ("Q", Q),)
                        for k, v in self.diagnostics(M, starting_left_svs, symmetric=True).items()
                    } | {
                        f"X_{k}": v
                        for k, v in self.diagnostics(X_if_we_stopped_here, starting_left_svs, starting_right_svs).items()
                    } | self.polar_accuracy_metrics(Xorig, X_if_we_stopped_here)
                )
            Q = self.sym_mm(Q, Z)  # Q = Q Z
            R = self.sym_mm(Z.T, self.sym_mm(R, Z))  # R = Z.T @ R @ Z
        Q = Q.to(dtype=self.qT_posthoc_dtype).to(dtype=self.ambient_dtype)
        X = self.mm(Q, X, dtype=self.qx_dtype)
        self.ambient_dtype = true_ambient_dtype
        if self.do_diagnostics:
            diagnostics.append(
                {
                    f"{name}_{k}": v
                    for name, M in (("R", R), ("Z", Z), ("Q", Q),)  # R doesn't matter and Z hasn't changed but whatever
                    for k, v in self.diagnostics(M, starting_left_svs, symmetric=True).items()
                } | {
                    f"X_{k}": v
                    for k, v in self.diagnostics(X, starting_left_svs, starting_right_svs).items()
                } | self.polar_accuracy_metrics(Xorig, X)
            )
        return X, pd.DataFrame(diagnostics)

    def track_eigvals(self, x_eigvals, r_shift):
        rs = []
        qs = []
        for iter, coeff in enumerate(self.coeffs):
            if (iter == 0) or (iter in self.restarts):
                if iter == 0:
                    r = x_eigvals**2
                else:
                    x_eigvals = q * x_eigvals
                    r = x_eigvals**2 - r_shift
                q = torch.ones_like(x_eigvals)
            z = coeff[-1] * torch.ones_like(r)
            for c in reversed(coeff[:-1]):
                z = c + r * z
            rs.append(r.clone().clone().cpu().numpy()); qs.append(q.clone().cpu().numpy())
            q *= z
            r *= z**2
        rs.append(r.clone().cpu().numpy()); qs.append(q.clone().cpu().numpy())
        return dict(R=rs, Q=qs)

    def mm(self, A, B, symmetrize=False, dtype=None):
        if dtype is None: dtype = self.mm_dtype
        A = A.to(dtype=dtype)
        B = B.to(dtype=dtype)
        out = (A @ B).to(dtype=self.ambient_dtype)
        return ((out + out.mT) / 2) if symmetrize else out

    def sym_mm(self, A, B, dtype=None):
        return self.mm(A, B, symmetrize=self.force_symmetry, dtype=dtype)

    def sym_polynomial(self, M, coeff):
        I = torch.eye(M.shape[-2], device=M.device, dtype=M.dtype)
        out = coeff[-1] * I
        for c in reversed(coeff[:-1]):
            out = c * I + self.sym_mm(M, out.mT)
        return out

    @staticmethod
    def diagnostics(M, starting_left_singular_vecs, starting_right_singular_vecs=None, symmetric=False):
        assert (symmetric and starting_right_singular_vecs is None) or (not symmetric and starting_right_singular_vecs is not None), "Must specify either symmetric or right singular vecs, but not both"
        M = M.to(dtype=torch.float64)  # ensure diagnostics are in high precision
        if not M.isfinite().all():
            hopefully_diagonalized = torch.full_like(M, float('nan'))
            singvals = torch.full((min(M.shape[-2:]),), float('nan'), device=M.device, dtype=M.dtype)
        elif symmetric:
            hopefully_diagonalized = starting_left_singular_vecs.mT @ M @ starting_left_singular_vecs
            # assert (M == M.mT).all(), "Matrix must be symmetric for diagnostics"
            M = (M + M.mT) / 2
            singvals = torch.flip(torch.linalg.eigvalsh(M), dims=(-1,))  # flip because other functions return in decreasing order
        else:
            hopefully_diagonalized = starting_left_singular_vecs.mT @ M @ starting_right_singular_vecs
            singvals = torch.linalg.svdvals(M)
        singvals_from_starting_vecs = torch.diag(hopefully_diagonalized)
        diagonalizability_residual = hopefully_diagonalized.clone().fill_diagonal_(0)
        diagonalizability_error = torch.linalg.matrix_norm(diagonalizability_residual, ord='fro') / torch.linalg.matrix_norm(hopefully_diagonalized, ord='fro')
        return dict(
            min_singval_from_starting_vecs=singvals_from_starting_vecs.min().item(),
            max_singval_from_starting_vecs=singvals_from_starting_vecs.max().item(),
            min_singval=singvals.min().item(),
            max_singval=singvals.max().item(),
            diagonalizability=diagonalizability_error.item(),
            largest_entry=torch.linalg.vector_norm(M, ord=float('inf')).item(),
            singvals_from_starting_vecs=singvals_from_starting_vecs.cpu().numpy(),
            singvals=singvals.cpu().numpy(),
        )

    @staticmethod
    def polar_accuracy_metrics(A: torch.Tensor, estimated_polar_A: torch.Tensor) -> dict[str, float]:
        metrics = dict(
            orth_error=float('inf'),
            residual_error=float('inf'),
            psd_error=float('inf'),
            dual_obj=float('inf'),
            bound_violation=float('inf'),
        )
        if not estimated_polar_A.isfinite().all():
            return metrics

        estimated_polar_A = estimated_polar_A.to(A.dtype)
        H = estimated_polar_A.mT @ A
        H = (H + H.mT) / 2
        Heigs = torch.linalg.eigvalsh(H)
        nuc = torch.linalg.matrix_norm(A, ord='nuc')
        estimate_spectral_norm = torch.linalg.matrix_norm(estimated_polar_A, ord=2)
        I = torch.eye(estimated_polar_A.shape[0], device=estimated_polar_A.device, dtype=estimated_polar_A.dtype)

        metrics["orth_error"] = ((estimated_polar_A @ estimated_polar_A.mT - I).norm() / I.norm()).item()
        metrics["residual_error"] = ((estimated_polar_A @ H - A).norm() / A.norm()).item()
        metrics["psd_error"] = ((Heigs[Heigs < 0]).norm() / (Heigs[Heigs > 0]).norm()).item()
        metrics["dual_obj"] = ((nuc - torch.inner(A.flatten(), estimated_polar_A.flatten()))/nuc).item()
        metrics["bound_violation"] = max((estimate_spectral_norm - 1).item(), 0)
        return metrics

    PE_coeffs_list = [
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
        (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
        (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
        (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
    ]
    rescaled_polar5 = rescale_PE_below_1(PE_coeffs_list)
    # safety factor for numerical stability (but exclude last polynomial)
    PE_coeffs_list = [
        (a / 1.02, b / 1.02**3, c / 1.02**5) for (a, b, c) in PE_coeffs_list[:-1]
    ] + [PE_coeffs_list[-1]]
    rescaled_polar5 = [
        (a / 1.02, b / 1.02**3, c / 1.02**5) for (a, b, c) in rescaled_polar5[:-1]
    ] + [rescaled_polar5[-1]]


def spectrum2matrix(spectrum, aspect_ratio):
    n = int(len(spectrum) * aspect_ratio)
    m = len(spectrum)
    U, _, Vh = torch.linalg.svd(torch.randn(m, n, device=spectrum.device, dtype=spectrum.dtype), full_matrices=False)
    return U @ torch.diag(spectrum) @ Vh


def spectrum_evolution_plot(df, yscale='linear', frames=None, yscale_kw={}):
    init_spectrum = df.loc[0, 'X_singvals_from_starting_vecs']
    if frames is None: frames = df.index.tolist()

    # since Z and Q are decreasing functions of the corresponding singular value of X_0, flip them for plotting purposes
    df['Z_singvals'] = df['Z_singvals'].apply(np.sort)
    df['Q_singvals'] = df['Q_singvals'].apply(np.sort)

    colname_suffix = "singvals_from_starting_vecs"
    # colname_suffix = "singvals"  # ONLY use this when the underlying polynomials are monotonic, like newton schulz, and there is no blowup that causes non-monotonicity. Otherwise the eigenvalues won't match those of X.
    title2col = {
        'R eigenvalues': f'R_{colname_suffix}',
        # 'Z eigenvalues': f'Z_{colname_suffix}',
        'Q eigenvalues': f'Q_{colname_suffix}',
        'X singular values': f'X_{colname_suffix}',
        # 'X max singular value': f'X_max_singval',
    }

    fig, axes = plt.subplots(1, len(title2col), figsize=(15, 4))

    def update(frame):
        for ax, (title, col) in zip(axes, title2col.items()):
            if pd.api.types.is_numeric_dtype(df[col]):
                ax.plot(df.loc[:frame, col], marker='o')
                ax.set_title(title)
                ax.set_xlabel('Step (t)')
                ax.set_yscale(yscale, **yscale_kw)
            else:
                vals = df.loc[frame, col]
                ax.plot(init_spectrum, vals, label=f'Step {frame}')
                ax.set_title(f'{title} (Steps 0 – {frame})')
                ax.set_xlabel('X_0 singular values')
                ax.set_yscale(yscale, **yscale_kw)
                ax.legend(loc='upper right', fontsize='small')
                current_lower, current_upper = ax.get_ylim()
                ax.set_ylim(
                    min(current_lower, float(vals.min())/1.1, 0),
                    max(current_upper, float(vals.max())*1.1, 1)
                )

    ani = FuncAnimation(fig, update, frames=frames, init_func=lambda: None, interval=500, repeat=False)
    plt.close(fig)
    return ani


if __name__ == "__main__":
    n = 512
    # aspect_ratio = 4
    # spectrum = torch.cat((
    #     torch.logspace(0, -2, steps=n//2, dtype=torch.float64),
    #     torch.zeros(n - n//2, dtype=torch.float64),
    # ))
    # G = spectrum2matrix(spectrum, aspect_ratio)

    G = torch.diag(torch.logspace(-.1, -8, steps=n, dtype=torch.float64))


    PE = PolarExpressDiagnostic(coeffs_name='ns3', steps=5, restarts=[], sym_mm_name='avg')
    _, diagnostics = PE(G)
    df = pd.DataFrame(diagnostics)
    print(df.head())


# NOTE TO SELF:
# I tried to show that loss of precision could also be due to eigenvalue drift, but I didn't succeed.
# The only way I could get significant eigenvalue drift was when there were large negative eigenvalues.
if False:
    # ### Eigenvector Drift
    # Even in the absence of spurious negative eigenvalues, the algorithm may still be unstable due to eigenvector drift.
    # So far, we have analyzed Gram Newton Schulz solely in terms of its effect on the eigenvalues of the matrices.
    # This is because, in exact arithmetic, the eigenvectors of any $R_t, Q_t$ or $Z_t$ are all identical — they are the left singular vectors of the input $G$.
    # Let $G = U \\Sigma V^\top$ be the singular value decomposition, and then $Q_T = U \Lambda_T U^\top$, where $\lim_{T \to \infty} \Lambda_T = \\Sigma^{-1}$.
    # Therefore, when we multiply $Q_T G$ in the final step, we expect $U^\top U$ to cancel, leaving
    # $$Q_T \cdot G = U\Lambda_T U^\top \cdot U \\Sigma V^\top = U \Lambda_T \\Sigma V^\top \to U \\Sigma^{-1} \\Sigma V^\top = U V^\top =: \mathrm{polar}(G)$$
    # as desired.
    # However, in floating point arithmetic, the eigenvectors of $Q_T$ will not match $U$, since each matrix operation performed by the algorithm causes the eigenvectors to drift slightly.
    # To demonstrate this drift, we can measure how far $Q_T$ is from the nearest matrix of the form $UDU^\top$ for some diagonal matrix $D$.

    _, well_conditioned_diagnostics = PolarExpressDiagnostic(
        coeffs_name="ns5",
        steps=30,
        ambient_dtype=torch.float16,
        xxt_dtype=torch.float64,
    )(spectrum2matrix(torch.logspace(-0.01, -3, steps=n, dtype=torch.float64, device=DEVICE), aspect_ratio))

    fig, ax = plt.subplots(figsize=(6, 4))
    for col, title in zip(['Q_diagonalizability'], ['Q_t']):
        ax.plot(well_conditioned_diagnostics[col], marker='o')
        ax.set_xlabel('Step (t)')
    ax.set_title("Relative distance from\nQ_t to nearest UDU^T");

    HTML(spectrum_evolution_plot(well_conditioned_diagnostics, frames=list(range(0, 30, 5))).to_jshtml())