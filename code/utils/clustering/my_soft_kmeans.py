#
from typing import Any, Optional, Tuple, Union
from warnings import warn

import torch
from torch import LongTensor, Tensor

from .distances import BaseDistance, CosineSimilarity
from .kmeans import KMeans

__all__ = ["SoftKMeans"]


class SoftKMeans(KMeans):
    """
    Implements differentiable soft k-means clustering.
    Method adapted from https://github.com/bwilder0/clusternet
    to support batches.

    Paper:
        Wilder et al., "End to End Learning and Optimization on Graphs" (NeurIPS'2019)

    Args:
        init_method: Method to initialize cluster centers: ['rnd', 'topk']
                        (default: 'rnd')
        num_init: Number of different initial starting configurations,
                    i.e. different sets of initial centers.
                    If >1 selects the best configuration before
                    propagating through fixpoint (default: 1).
        max_iter: Maximum number of iterations (default: 100).
        distance: batched distance evaluator (default: CosineSimilarity).
        p_norm: norm for lp distance (default: 1).
        normalize: id of method to use to normalize input. (default: 'unit').
        tol: Relative tolerance with regards to Frobenius norm of the difference
                    in the cluster centers of two consecutive iterations to
                    declare convergence. (default: 1e-4)
        n_clusters: Default number of clusters to use if not provided in call
                (optional, default: 8).
        verbose: Verbosity flag to print additional info (default: True).
        seed: Seed to fix random state for randomized center inits
                (default: True).
        temp: temperature for soft cluster assignments (default: 5.0).
        **kwargs: additional key word arguments for the distance function.

    """

    def __init__(
        self,
        init_method: str = "rnd",
        num_init: int = 1,
        max_iter: int = 100,
        distance: BaseDistance = CosineSimilarity,
        p_norm: int = 1,
        normalize: str = "unit",
        tol: float = 1e-5,
        n_clusters: Optional[int] = 8,
        verbose: bool = True,
        seed: Optional[int] = 123,
        temp: float = 5.0,
        **kwargs,
    ):
        super(SoftKMeans, self).__init__(
            init_method=init_method,
            num_init=num_init,
            max_iter=max_iter,
            distance=distance,
            p_norm=p_norm,
            tol=tol,
            normalize=normalize,
            n_clusters=n_clusters,
            verbose=verbose,
            seed=seed,
            **kwargs,
        )
        self.temp = temp
        if self.temp <= 0.0:
            raise ValueError(f"temp should be > 0, but got {self.temp}.")
        if not self.distance.is_inverted:
            raise ValueError(
                "soft k-means requires inverted " "distance measure (i.e. similarity)."


            )


    @torch.no_grad()
    def _cluster(
        self, x: Tensor, centers: Tensor, k: LongTensor, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Union[Tensor, Any]]:
        """
        Run Lloyd's k-means algorithm.

        Args:
            x: (BS, N, D)
            centers: (BS, num_init, k_max, D)
            k: (BS, )

        """
        if not isinstance(self.distance, LpDistance):
            warn("standard k-means should use a non-inverted distance measure.")
        bs, n, d = x.size()
        # mask centers for which  k < k_max with inf to get correct assignment
        k_max = torch.max(k).cpu().item()
        k_max_range = torch.arange(k_max, device=x.device)[None, :].expand(bs, -1)
        k_mask = k_max_range >= k[:, None]
        k_mask = k_mask[:, None, :].expand(bs, self.num_init, -1)

        for i in range(self.max_iter):
            centers[k_mask] = float("inf")
            old_centers = centers.clone()
            # get cluster assignments
            c_assign = self._assign(x, centers)
            # update cluster centers
            centers = group_by_label_mean(x, c_assign, k_max_range)
            if self.tol is not None:
                # calculate center shift
                shift = self._calculate_shift(centers, old_centers, p=self.p_norm)
                if (shift < self.tol).all():
                    if self.verbose:
                        print(
                            f"Full batch converged at iteration "
                            f"{i+1}/{self.max_iter} "
                            f"with center shifts = "
                            f"{shift.view(-1, self.num_init).mean(-1)}."
                        )
                    self.n_iter = {i + 1}
                    break

        # select best rnd restart according to inertia
        if self.n_iter is None:
            self.n_iter = {self.max_iter}
        centers[k_mask] = float("inf")
        c_assign = self._assign(x, centers)
        inertia = self._calculate_inertia(x, centers, c_assign)
        best_init = torch.argmin(inertia, dim=-1)
        b_idx = torch.arange(bs, device=x.device)

        return (
            c_assign[b_idx, best_init],
            centers[b_idx, best_init],
            inertia[b_idx, best_init],
            None,
        )



    @torch.no_grad()
    def _pairwise_distance(
        self,
        x: torch.Tensor,              # (bs, n, d)
        centers: torch.Tensor,        # (bs, num_init, k, d)
        *,
        precisions: torch.Tensor | None = None,   # (bs, num_init, k, d) or broadcastable
        variances: torch.Tensor | None = None,    # same shapes; used if precisions is None
        return_sqrt: bool = True,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Diagonal-Mahalanobis distances using batched matmuls.

        Returns: (bs, num_init, n, k)
        If both `precisions` and `variances` are None -> falls back to plain L2.
        """

        bs, n, d = x.shape
        _, num_init, k, d_ = centers.shape
        assert d == d_, "x and centers must have same feature dim"

        # Reshape to merge (bs, num_init) for one batched GEMM per init
        B = bs * num_init
        X = x[:, None, :, :].expand(bs, num_init, n, d).reshape(B, n, d).contiguous()  # (B, n, d)
        C = centers.reshape(B, k, d).contiguous()                                       # (B, k, d)

        # Decide precision τ = 1/σ^2 per component dimension
        if precisions is None:
            if variances is None:
                # Euclidean fallback: τ = 1 for all dims
                tau = torch.ones((B, k, d), dtype=X.dtype, device=X.device)
            else:
                var = variances
                # Broadcast var to (bs, num_init, k, d) then to (B, k, d)
                while var.dim() < 4:
                    var = var.unsqueeze(0)
                var = var.expand(bs, num_init, k, d).reshape(B, k, d).contiguous()
                tau = (var.clamp_min(eps)).reciprocal_()
        else:
            tau = precisions
            while tau.dim() < 4:
                tau = tau.unsqueeze(0)
            tau = tau.expand(bs, num_init, k, d).reshape(B, k, d).contiguous()

        # Precompute per-component pieces
        # (i) x^2 ⋅ τ_k  -> (B, n, k)
        X2_tau = (X.square()) @ tau.transpose(1, 2)

        # (ii) x ⋅ (μ_k ⊙ τ_k) -> (B, n, k)
        mu_tau = C * tau
        X_mu_tau = X @ mu_tau.transpose(1, 2)

        # (iii) (μ_k^2) ⋅ τ_k -> (B, 1, k), broadcast over n
        mu2_tau_sum = (C.square() * tau).sum(dim=2).unsqueeze(1)  # (B, 1, k)

        # Quadratic form for all pairs
        dist2 = X2_tau - 2.0 * X_mu_tau + mu2_tau_sum            # (B, n, k)
        dist2.clamp_(min=0.0)

        if return_sqrt:
            dist = dist2.sqrt_()
        else:
            dist = dist2

        return dist.view(bs, num_init, n, k)
    