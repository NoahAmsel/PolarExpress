from itertools import repeat

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import pandas as pd
import torch


class PolarExpressDiagnostic:

    def __init__(
        self,
        coeffs_name: str,
        steps: int,
        restarts: list[int] = [],
        force_symmetry: bool = True,
        ambient_dtype=torch.float64,
        xxt_dtype=None,
        xxt_posthoc_dtype=None,
        qx_dtype=None,
        do_diagnostics=True,
    ):
        self.coeffs = dict(
            ns3=[(1.5, -0.5)],
            ns5=[(15/8, -10/8, 3/8)],
            polar5=self.PE_coeffs_list,
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
        self.qx_dtype = self.mm_dtype if qx_dtype is None else qx_dtype

    def __call__(self, G: torch.Tensor) -> torch.Tensor:
        assert G.ndim >= 2
        assert G.dtype == torch.float64, "Input must be float64 for diagnostic purposes"
        assert G.size(-2) <= G.size(-1), f"Input must be short and fat, but size is {G.size()}"
        assert torch.linalg.matrix_norm(G, ord=2) < .999
        return self.appF(G)

    def appF(self, X: torch.Tensor) -> torch.Tensor:
        diagnostics = []
        if self.do_diagnostics:
            Xorig = X.clone()
            starting_left_svs, _, starting_right_svs = torch.linalg.svd(Xorig, full_matrices=False)
            starting_right_svs = starting_right_svs.mT  # svd returns V^T, not V
        X = X.to(self.ambient_dtype)
        for iter, coeff in enumerate(self.coeffs):
            if (iter == 0) or (iter in self.restarts):
                if (iter in self.restarts) and (iter > 0):
                    X = self.mm(Q, X, dtype=self.qx_dtype)  # apply the current Q to X before restarting
                Q = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
                R = self.sym_mm(X, X.mT, dtype=self.xxt_dtype).to(dtype=self.xxt_posthoc_dtype).to(dtype=self.ambient_dtype)  # R = X @ X.mT
                # TODO: record eigenvectors of XXT at iter 0 to use for diagonalizing inside diagnostics
            Z = self.polynomial(R, coeff)  # Z = aI + bR + cR^2
            if self.do_diagnostics:
                X_if_we_stopped_here = self.mm(Q, X, dtype=self.qx_dtype)
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
            Q = self.mm(Q, Z)  # Q = Q Z
            R = self.sym_mm(Z.T, self.sym_mm(R, Z))  # R = Z.T @ R @ Z
        X = self.mm(Q, X, dtype=self.qx_dtype)
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

    def track_eigvals(self, eigvals):
        rs = []
        qs = []
        for iter, coeff in enumerate(self.coeffs):
            if (iter == 0) or (iter in self.restarts):
                q = torch.ones_like(eigvals)
                r = eigvals.clone()
            z = coeff[-1] * torch.ones_like(r)
            for c in reversed(coeff[:-1]):
                z = c + r * z
            rs.append(r.clone()); qs.append(q.clone())
            q *= z
            r *= z**2
        rs.append(r.clone()); qs.append(q.clone())
        return rs, qs

    def mm(self, A, B, symmetrize=False, dtype=None):
        if dtype is None: dtype = self.mm_dtype
        A = A.to(dtype=dtype)
        B = B.to(dtype=dtype)
        out = (A @ B).to(dtype=self.ambient_dtype)
        return ((out + out.mT) / 2) if symmetrize else out

    def sym_mm(self, A, B, dtype=None):
        return self.mm(A, B, symmetrize=self.force_symmetry, dtype=dtype)

    def polynomial(self, M, coeff):
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
            singvals = torch.full((min(M.shape[-2:]),), float('nan'), device=M.device, dtype=M.dtype)
            singvals_from_starting_vecs = torch.full((min(M.shape[-2:]),), float('nan'), device=M.device, dtype=M.dtype)
        elif symmetric:
            # assert (M == M.mT).all(), "Matrix must be symmetric for diagnostics"
            M = (M + M.mT) / 2
            singvals = torch.flip(torch.linalg.eigvalsh(M), dims=(-1,))  # flip because other functions return in decreasing order
            singvals_from_starting_vecs = torch.diag(starting_left_singular_vecs.mT @ M @ starting_left_singular_vecs)
        else:
            singvals = torch.linalg.svdvals(M)
            singvals_from_starting_vecs = torch.diag(starting_left_singular_vecs.mT @ M @ starting_right_singular_vecs)
        diag = torch.zeros_like(M)
        diag[range(len(singvals_from_starting_vecs)), range(len(singvals_from_starting_vecs))] = singvals_from_starting_vecs
        diagonalizability_error = torch.linalg.matrix_norm(M - diag, ord='fro') / torch.linalg.matrix_norm(M, ord='fro')
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
    # safety factor for numerical stability (but exclude last polynomial)
    PE_coeffs_list = [
        (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in PE_coeffs_list[:-1]
    ] + [PE_coeffs_list[-1]]


def spectrum2matrix(spectrum, aspect_ratio):
    n = int(len(spectrum) * aspect_ratio)
    m = len(spectrum)
    U, _, Vh = torch.linalg.svd(torch.randn(m, n, device=spectrum.device, dtype=spectrum.dtype), full_matrices=False)
    return U @ torch.diag(spectrum) @ Vh


def spectrum_evolution_plot(df, yscale='linear', **yscale_kw):
    init_spectrum = df.loc[0, 'X_singvals_from_starting_vecs']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    columns = ['R_singvals_from_starting_vecs', 'Q_singvals_from_starting_vecs', 'X_singvals_from_starting_vecs']
    # columns = ['R_singvals', 'Q_singvals', 'X_singvals']
    titles = ['R eigenvalues', 'Q eigenvalues', 'X singular values']

    def update(frame):
        for ax, col, title in zip(axes, columns, titles):
            vals = df[col].iloc[frame]
            ax.plot(init_spectrum, vals, label=f'Step {frame}')
            ax.set_title(f'{title} (Steps 0 – {frame})')
            ax.set_xlabel('X_0 singular values')
            ax.set_yscale(yscale, **yscale_kw)
            ax.legend(loc='upper right', fontsize='small')
            current_lower, current_upper = ax.get_ylim()
            ax.set_ylim(min(current_lower, float(vals.min())/1.1, 0),
                        max(current_upper, float(vals.max())*1.1, 1))

    ani = FuncAnimation(fig, update, frames=len(df), init_func=lambda: None, interval=500, repeat=False)
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
