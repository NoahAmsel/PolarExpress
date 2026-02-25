from itertools import repeat

import pandas as pd
import torch


class PolarExpressDiagnostic:

    def __init__(
        self,
        coeffs_name: str,
        steps: int,
        restarts: list[int],
        sym_mm_name: str,
        ambient_dtype=torch.float64,
        xxt_dtype=None,
        xxt_posthoc_dtype=None,
        qx_dtype=None,
        do_diagnostics=True,
    ):
        self.coeffs = dict(
            ns3=[(1.5, -0.5, 0)],
            ns5=[(15/8, -10/8, 3/8)],
            polar5=self.PE_coeffs_list,
        )[coeffs_name]
        self.coeffs = self.coeffs[:steps] + list( 
            repeat(self.coeffs[-1], steps - len(self.coeffs)))

        self.restarts = sorted(restarts)
        self.sym_mm_name = sym_mm_name
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
            starting_eigvecs, _, starting_right_svs = torch.linalg.svd(Xorig, full_matrices=False)
            starting_right_svs = starting_right_svs.mT  # svd returns V^T, not V
        X = X.to(self.ambient_dtype)
        for iter, (a, b, c) in enumerate(self.coeffs):
            if (iter == 0) or (iter in self.restarts):
                if (iter in self.restarts) and (iter > 0):
                    X = self.sym_mm(Q, X, dtype=self.qx_dtype)  # apply the current Q to X before restarting
                Q = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
                R = self.sym_mm(X, X.mT, dtype=self.xxt_dtype).to(dtype=self.xxt_posthoc_dtype).to(dtype=self.ambient_dtype)  # R = X @ X.mT
                # TODO: record eigenvectors of XXT at iter 0 to use for diagonalizing inside diagnostics
            Z = self.quadratic(R, a, b, c)  # Z = aI + bR + cR^2
            if self.do_diagnostics:
                X_if_we_stopped_here = self.applyQ(Q, X)
                diagnostics.append(
                    {
                        f"{name}_{k}": v
                        for name, M in (("R", R), ("Z", Z), ("Q", Q),)
                        for k, v in self.diagnostics(M, starting_eigvecs).items()
                    } | {
                        f"X_{k}": v
                        for k, v in self.X_diagnostics(X_if_we_stopped_here, starting_eigvecs, starting_right_svs).items()
                    } | self.polar_accuracy_metrics(Xorig, X_if_we_stopped_here)
                )
            Q = self.sym_mm(Q, Z)  # Q = Q Z
            R = self.sym_mm(Z.T, self.sym_mm(R, Z))
        X = self.sym_mm(Q, X, dtype=self.qx_dtype)
        if self.do_diagnostics:
            diagnostics.append(
                {
                    f"{name}_{k}": v
                    for name, M in (("R", R), ("Z", Z), ("Q", Q),)  # R doesn't matter and Z hasn't changed but whatever
                    for k, v in self.diagnostics(M, starting_eigvecs).items()
                } | {
                    f"X_{k}": v
                    for k, v in self.X_diagnostics(X, starting_eigvecs, starting_right_svs).items()
                } | self.polar_accuracy_metrics(Xorig, X)
            )
        return X, diagnostics

    def sym_mm(self, A, B, dtype=None):
        if dtype is None: dtype = self.mm_dtype
        A = A.to(dtype=dtype)
        B = B.to(dtype=dtype)
        if self.sym_mm_name == "basic":
            return (A @ B).to(dtype=self.ambient_dtype)
        elif self.sym_mm_name == "avg":
            out = (A @ B).to(dtype=self.ambient_dtype)
            return (out + out.mT) / 2
        else:
            raise ValueError(f"Unknown sym_mm_name: {self.sym_mm_name}")

    # TODO: support arbitrary polynomial
    def quadratic(self, M, a, b, c):
        I = torch.eye(M.shape[-2], device=M.device, dtype=M.dtype)
        return a * I + b * M + c * self.sym_mm(M, M.mT)

    @staticmethod
    def diagnostics(M, starting_eigvecs):
        M = M.to(dtype=torch.float64)  # ensure diagnostics are in high precision
        # assert (M == M.mT).all(), "Matrix must be symmetric for diagnostics"
        M = (M + M.mT) / 2
        if not M.isfinite().all():
            eigvals = torch.full((min(M.shape[-2:]),), float('nan'), device=M.device, dtype=M.dtype)
            eigvals_from_starting_vecs = torch.full((min(M.shape[-2:]),), float('nan'), device=M.device, dtype=M.dtype)
        else:
            eigvals = torch.flip(torch.linalg.eigvalsh(M), dims=(-1,))  # flip because other functions return in decreasing order
            Lambda = starting_eigvecs.mT @ M @ starting_eigvecs
            eigvals_from_starting_vecs = torch.diag(Lambda)
            diagonalizability_error = torch.linalg.matrix_norm(M - Lambda, ord='fro') / torch.linalg.matrix_norm(M, ord='fro')
        return dict(
            min_eigval_from_starting_vecs=eigvals_from_starting_vecs.min().item(),
            max_eigval_from_starting_vecs=eigvals_from_starting_vecs.max().item(),
            min_eigval=eigvals.min().item(),
            max_eigval=eigvals.max().item(),
            diagonalizability=diagonalizability_error.item(),
            largest_entry=torch.linalg.vector_norm(M, ord=float('inf')).item(),
            eigvals=eigvals.cpu().numpy(),
            diagonalizability=diagonalizability_error.item(),
        )

    # TODO: fold this into previous method
    @staticmethod
    def X_diagnostics(M, starting_left_singular_vecs, starting_right_singular_vecs):
        M = M.to(dtype=torch.float64)  # ensure diagnostics are in high precision
        if not M.isfinite().all():
            singvals = torch.full((min(M.shape[-2:]),), float('nan'), device=M.device, dtype=M.dtype)
            singvals_from_starting_svs = torch.full((min(M.shape[-2:]),), float('nan'), device=M.device, dtype=M.dtype)
        else:
            singvals = torch.linalg.svdvals(M)
            singvals_from_starting_svs = torch.diag(starting_left_singular_vecs.mT @ M @ starting_right_singular_vecs)
        return dict(
            eigvals=singvals.cpu().numpy(),
            min_eigval=singvals.min().item(),
            max_eigval=singvals.max().item(),
            eigvals_from_starting_vecs=singvals_from_starting_svs.cpu().numpy(),
            min_eigval_from_starting_vecs=singvals_from_starting_svs.min().item(),
            max_eigval_from_starting_vecs=singvals_from_starting_svs.max().item(),
            largest_entry=torch.linalg.vector_norm(M, ord=float('inf')).item(),
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
