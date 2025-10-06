#
from typing import Any, Optional, Tuple, Union
from warnings import warn

import torch
import torch.nn as nn
from torch import LongTensor, Tensor

from .distances import (
    BaseDistance,
    CosineSimilarity,
    DotProductSimilarity,
    LpDistance,
)
from .utils import ClusterResult, group_by_label_mean

# import numpy as np
# from sklearn.cluster._kmeans import _kmeans_plusplus, row_norms

__all__ = ["KMeans"]


#

class KMeans(nn.Module):
    """
    Implements k-means clustering in terms of
    pytorch tensor operations which can be run on GPU.
    Supports batches of instances for use in
    batched training (e.g. for neural networks).

    Partly based on ideas from:
        - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        - https://github.com/overshiki/kmeans_pytorch


    Args:
            init_method: Method to initialize cluster centers ['rnd', 'k-means++']
                            (default: 'rnd')
            num_init: Number of different initial starting configurations,
                        i.e. different sets of initial centers (default: 8).
            max_iter: Maximum number of iterations (default: 100).
            distance: batched distance evaluator (default: LpDistance).
            p_norm: norm for lp distance (default: 2).
            tol: Relative tolerance with regards to Frobenius norm of the difference
                        in the cluster centers of two consecutive iterations to
                        declare convergence. (default: 1e-4)
            normalize: String id of method to use to normalize input.
                        one of ['mean', 'minmax', 'unit'].
                        None to disable normalization. (default: None).
            n_clusters: Default number of clusters to use if not provided in call
                    (optional, default: 8).
            verbose: Verbosity flag to print additional info (default: True).
            seed: Seed to fix random state for randomized center inits
                    (default: True).
            **kwargs: additional key word arguments for the distance function.
    """

    INIT_METHODS = ["random", "k-means++", "kmeans"]
    NORM_METHODS = ["mean", "minmax", "unit"]

    def __init__(
        self,
        init_method: str = "random",
        num_init: int = 8,
        max_iter: int = 300,
        tol: float = 1e-8,
        normalize: Optional[Union[str, bool]] = None,
        n_clusters: Optional[int] = 8,
        verbose: bool = True,
        seed: Optional[int] = 123,
        is_for_init=False,
        **kwargs,
    ):
        super(KMeans, self).__init__()
        self.init_method = init_method.lower()
        self.num_init = num_init
        self.max_iter = max_iter
        self.tol = tol
        self.normalize = normalize
        self.n_clusters = n_clusters
        self.verbose = verbose
        self.seed = seed

        self._check_params()
        self.is_for_init = is_for_init

        self.eps = None
        self._k_max = None
        self._result = None
        self.n_iter = None  # number of iterations run in last fit


    @property
    def is_fitted(self) -> bool:
        """True if model was already fitted."""
        return self._result is not None

    @property
    def num_clusters(self) -> Union[int, Tensor, Any]:
        """
        Number of clusters in fitted model.
        Returns a tensor with possibly different
        numbers of clusters per instance for whole batch.
        """
        if not self.is_fitted:
            return None
        return self._result.k

    def _check_params(self):
        if self.init_method not in self.INIT_METHODS:
            raise ValueError(
                f"unknown <init_method>: {self.init_method}. "
                f"Please choose one of {self.INIT_METHODS}"
            )
        if self.num_init <= 0:
            raise ValueError(f"num_init should be > 0, but got {self.num_init}.")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, but got {self.max_iter}.")
      
        if self.tol < 0 or self.tol > 1:
            raise ValueError(f"tol should be > 0 and < 1, but got {self.tol}.")
        if isinstance(self.normalize, bool):
            if self.normalize:
                self.normalize = "mean"
            else:
                self.normalize = None
        if self.normalize is not None and self.normalize not in self.NORM_METHODS:
            raise ValueError(
                f"unknown <normalize> method: {self.normalize}. "
                f"Please choose one of {self.NORM_METHODS}"
            )
        if self.n_clusters is not None and self.n_clusters < 2:
            raise ValueError(f"n_clusters should be > 1, but got {self.n_clusters}.")

    def _check_x(self, x) -> Tensor:
        """Check and (re-)format input samples x."""
        if not isinstance(x, Tensor):
            raise TypeError(f"x has to be a torch.Tensor but got {type(x)}.")
        shp = x.shape
        if len(shp) < 2:
            raise ValueError(
                f"input <x> should be at least of shape (N, D) "
                f"with number of points N and number of dimensions D but got {shp}."
            )
        elif len(shp) > 2:
            x = x.squeeze()
            x = self._check_x(x)
        self.eps = torch.finfo(x.dtype).eps
        return x

    def _check_k(
        self, k,  device: torch.device = torch.device("cpu")
    ) -> LongTensor:
        """Check and (re-)format number of clusters k."""
         
        if not isinstance(k, Tensor):
            if k is None:  # use specified default number of clusters
                if self.n_clusters is None:
                    raise ValueError(
                        "Did not provide number of clusters k on call and "
                        "did not specify default 'n_clusters' at initialization."
                    )
                k = self.n_clusters
            if isinstance(k, int):  # convert to tensor
                k = torch.tensor(k, dtype=torch.long)
            else:
                raise TypeError(
                    f"k has to be int, torch.Tensor or None " f"but got {type(k)}."
                )
        if len(k.shape) > 1:
            k = k.squeeze()
            assert len(k.shape) == 1
        # if k.shape[0] == 1:
        #     k = k.repeat(bs)
        if (k >= self.n).any():
            raise ValueError(
                f"Specified 'k' must be smaller than "
                f"number of samples n={n}, but got: {k}."
            )
        if (k <= 1).any():
            raise ValueError("Clustering for k=1 is ambiguous.")
        self._k_max = int(k.max())
        return k.to(dtype=torch.long, device=device)

    def _check_centers(
        self,
        centers,
        dims: Tuple,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        if not isinstance(centers, Tensor):
            raise TypeError(
                f"centers has to be a torch.Tensor " f"but got {type(centers)}."
            )
        bs, n, d = dims
        if len(centers.shape) == 3:
            if (
                centers.size(0) != bs
                or centers.size(1) != self._k_max
                or centers.size(2) != d
            ):
                raise ValueError(
                    f"centers needs to be of shape "
                    f"({bs}, {self._k_max}, {d}),"
                    f"but got {tuple(centers.shape)}."
                )
            if self.num_init > 1:
                warn(
                    f"Specified num_init={self.num_init} > 1 but provided "
                    f"only 1 center configuration per instance. "
                    f"Using same center configuration for all {self.num_init} runs."
                )
                # expand to num_init size
                centers = centers[:, None, :, :].expand(
                    centers.size(0), self.num_init, centers.size(1), centers.size(2)
                )
            else:
                centers = centers.unsqueeze(1)
        elif len(centers.shape) == 4:
            if (
                centers.size(0) != bs
                or centers.size(1) != self.num_init
                or centers.size(2) != self._k_max
                or centers.size(3) != d
            ):
                raise ValueError(
                    f"centers needs to be of shape "
                    f"({bs}, {self.num_init}, {self._k_max}, {d}),"
                    f"but got {tuple(centers.shape)}."
                )
        else:
            raise ValueError(
                f"centers have unsupported shape of "
                f"{tuple(centers.shape)} "
                f"instead of "
                f"({bs}, {self.num_init}, {self._k_max}, {d})."
            )
        return centers.contiguous().to(dtype=dtype, device=device)

    def forward(
        self,
        x: Tensor,
        k: Optional[Union[LongTensor, Tensor, int]] = None,
        centers: Optional[Tensor] = None,
        **kwargs,
    ) -> ClusterResult:
        """torch.nn like forward pass.

        Args:
            x: input features/coordinates (BS, N, D)
            k: optional batch of (possibly different)
                numbers of clusters per instance (BS, )
            centers: optional batch of initial centers to use (BS, K, D)
            **kwargs: additional kwargs for initialization or cluster procedure

        Returns:
            ClusterResult tuple

        """
        x = self._check_x(x)
        self.n, self.d = x.shape
        x_ = x
        self.bs = k.shape[0]
        k = self._check_k(k,  device=x.device)

        # normalize input
        if self.normalize is not None:
            x = self._normalize(x, self.normalize, self.eps)
        # init centers
        if centers is None:
            centers = self._center_init(x, k, **kwargs)
        centers = self._check_centers(
            centers, dims=(self.bs, self.n, self.d), dtype=x.dtype, device=x.device
        )

        if not self.is_for_init:
            labels, new_centers, inertia = self._cluster(
                x, centers, k, **kwargs
            )
            return ClusterResult(
                labels=labels.cpu(),  # type: ignore
                centers=new_centers.cpu(),
                inertia=inertia.cpu(),
                k=k,
            )
        else:
            centers = self._cluster(
                x, centers, k, **kwargs
            )
            return centers

    def fit(
        self,
        x: Tensor,
        k: Optional[Union[LongTensor, Tensor, int]] = None,
        centers: Optional[Tensor] = None,
        **kwargs,
    ) -> nn.Module:
        """Compute cluster centers and predict cluster index for each sample.

        Args:
            x: input features/coordinates (N, D)
            k: optional batch of (possibly different)
                numbers of clusters per instance (BS, )
            centers: optional batch of initial centers to use (BS, K, D)
            **kwargs: additional kwargs for initialization or cluster procedure

        Returns:
            KMeans model
        """
        self._result = self(x, k=k, centers=centers, **kwargs)

    
    @torch.no_grad()
    def predict(self, x: Tensor, **kwargs) -> LongTensor:
        """Hard assignments (argmax over responsibilities)."""
        assert self.is_fitted, "Call fit() first."
        x = self._check_x(x)  # (bs, n, d)
        if self.normalize is not None:
            x = self._normalize(x, self.normalize, self.eps)

        # Fitted params (one set per batch)
        centers    = self._result.centers        # (bs, k_max, d)
        k_vec    = self._result.k            # (bs,)

        n, d = x.shape
        bs, k_max, d2 = centers.shape
        assert  d == d2

        # Build invalid-component mask from k (don’t rely on any saved mask)
        k_range = torch.arange(k_max, device=x.device).expand(bs, -1)    # (bs, k_max)
        invalid = (k_range >= k_vec.unsqueeze(1)).unsqueeze(1)  

        labels = self._e_step(
            x, 
            centers.unsqueeze(1), 
            invalid
            )
        return labels.squeeze(1)  # type: ignore
        
        # def predict(self, x: Tensor, **kwargs) -> LongTensor:
        #     """Predict the closest cluster each sample in X belongs to.

        #     Args:
        #         x: input features/coordinates (BS, N, D)
        #         **kwargs: additional kwargs for assignment procedure

        #     Returns:
        #         batch tensor of cluster labels for each sample (BS, N)

        #     """
        #     assert self.is_fitted
        #     x = self._check_x(x)
        #     return self._assign(
        #         x, centers=self._result.centers[:, None, :, :], **kwargs
        #     ).squeeze(1)
    @torch.no_grad()
    def fit_predict(
        self,
        x: Tensor,
        k: Optional[Union[LongTensor, Tensor, int]] = None,
        centers: Optional[Tensor] = None,
        **kwargs,
    ) -> LongTensor:
        """Compute cluster centers and predict cluster index for each sample.

        Args:
            x: input features/coordinates (BS, N, D)
            k: optional batch of (possibly different)
                numbers of clusters per instance (BS, )
            centers: optional batch of initial centers to use (BS, K, D)
            **kwargs: additional kwargs for initialization or cluster procedure

        Returns:
            batch tensor of cluster labels for each sample (BS, N)

        """
        self._result = self(x, k=k, centers=centers, **kwargs)
        return self._result.labels

    @torch.no_grad()
    def _center_init(self, x: Tensor, k: LongTensor, **kwargs) -> Tensor:
        """Wrapper to apply different methods for
        initialization of initial centers (centroids)."""
        if self.init_method == "random":
            return self._init_rnd(x, k)
        elif self.init_method == "k-means++":
            return self._init_plus(x, k)
        else:
            raise ValueError(f"unknown initialization method: {self.init_method}.")

    @staticmethod
    def _normalize(x: Tensor, normalize: str, eps: float = 1e-8):
        """Normalize input samples x according to specified method:

        - mean: subtract sample mean
        - minmax: min-max normalization subtracting sample min and divide by sample max
        - unit: normalize x to lie on D-dimensional unit sphere

        """
        if normalize == "mean":
            x -= x.mean(dim=1)[:, None, :]
        elif normalize == "minmax":
            x -= x.min(-1, keepdims=True).values  # type: ignore
            x /= x.max(-1, keepdims=True).values  # type: ignore
        elif normalize == "unit":
            # normalize x to unit sphere
            z_msk = x == 0
            x = x.clone()
            x[z_msk] = eps
            x = torch.diag_embed(1.0 / (torch.norm(x, p=2, dim=-1))) @ x
        else:
            raise ValueError(f"unknown normalization type {normalize}.")
        return x

    def _init_rnd(self, x: torch.Tensor, k: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: (N, D)  -- shared dataset (no batch dim)
            k: (BS,)   -- clusters per batch item (we’ll use k_max)

        Returns:
            centers: (BS, num_init, k_max, D)  -- same init for each batch item
        """
        N, D = x.shape
        bs = int(k.shape[0])
        k_max = int(k.max())

        if self.seed is not None:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(self.seed)
        else:
            gen = None

        # (num_init, k_max) distinct row indices per init
        # probs is uniform over N rows
        probs = torch.full((self.num_init, N), 1.0 / N, device=x.device, dtype=x.dtype)

        idx   = torch.multinomial(probs, num_samples=k_max, replacement=False, generator=gen)  # (num_init, k_max), Long

        # Select those rows from x
        base_centers = x.index_select(0, idx.reshape(-1)).view(self.num_init, k_max, D).contiguous()
        # Make per-batch copies (don’t use expand if you’ll modify centers)
        centers = base_centers.unsqueeze(0).repeat(bs, 1, 1, 1).contiguous()  # (bs, num_init, k_max, D)

        return centers

    def _init_skl_plus(self, x: Tensor, k: LongTensor) -> Tensor:
        """Choose initial centers via kmeans++ method.
        https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/cluster/_kmeans.py#L50

        Args:
            x: (BS, N, D)
            k: (BS, )

        Returns:
            centers: (BS, num_init, k, D)

        """
        raise NotImplementedError
        # would require sklearn as additional dependency

        # bs, n, d = x.size()
        # k_max = torch.max(k).cpu().item()
        # rs = np.random.RandomState(self.seed if self.seed is not None else 1)
        # device = x.device
        # x = x.cpu().numpy()
        # k = k.cpu().numpy()
        # centers = []
        # for smp, nc in zip(x, k):
        #     center_inits = []
        #     x_squared_norms = row_norms(smp, squared=True)
        #     for i in range(self.num_init):
        #         c = np.zeros((k_max, d))
        #         c_init, _ = _kmeans_plusplus(
        #             smp, nc, random_state=rs, x_squared_norms=x_squared_norms
        #         )
        #         c[:nc] = c_init
        #         center_inits.append(c)
        #     centers.append(torch.from_numpy(np.stack(center_inits)))
        #
        # return torch.stack(centers).to(device)

    # def _init_plus(self, x: Tensor, k: LongTensor) -> Tensor:
    #     """Choose initial centers via k-means++ method

    #     Args:
    #         x: (BS, N, D)
    #         k: (BS, )

    #     Returns:
    #         centers: (BS, num_init, k, D)

    #     """
    #     n, d = x.size()
    #     bs, = k.shape
    #     k_max = torch.max(k).cpu().item()

    #     if self.seed is not None:
    #         # make random init reproducible independent of current iteration,
    #         # which otherwise would step and change the torch generator state
    #         gen = torch.Generator(device=x.device)
    #         gen.manual_seed(self.seed)
    #     else:
    #         gen = None

    #     bsm = bs * self.num_init
    #     bsm_idx = torch.arange(bsm, device=x.device)
    #     centers = torch.empty((bsm, k_max, d), dtype=x.dtype, device=x.device)

    #     # select first center randomly
    #     assert n > self.num_init, (
    #         f"Number of samples must be larger than <num_init> "
    #         f"but got {n} <= {self.num_init}"
    #     )
    #     idx = torch.multinomial(
    #         torch.empty((bs, n), device=x.device, dtype=x.dtype).fill_(1 / n),
    #         num_samples=self.num_init,
    #         replacement=False,
    #         generator=gen,
    #     )
    #     centers[:, 0] = x.gather(index=idx[:, :, None].expand(-1, -1, d), dim=1).view(
    #         -1, d
    #     )
    #     msk = torch.zeros((bsm, n, k_max), dtype=torch.bool, device=x.device)
    #     msk[bsm_idx, idx.view(-1), 0] = True

    #     # select the remaining k-1 centers
    #     for nc in range(1, k_max):
    #         dist = self._pairwise_distance(
    #             x, centers[:, :nc].view(bs, self.num_init, -1, d)
    #         ).view(bsm, n, nc)
    #         pot = dist**2
    #         pot[msk[:, :, :nc]] = 0
    #         pot = pot.min(dim=-1).values
    #         idx = torch.multinomial(pot, 1, generator=gen).view(bs, self.num_init)
    #         centers[:, nc] = x.gather(
    #             index=idx[:, :, None].expand(-1, -1, d), dim=1
    #         ).view(-1, d)
    #         msk[bsm_idx, idx.view(-1), nc] = True

    #     return centers.view(bs, self.num_init, k_max, d)
    @torch.no_grad()
    def _init_plus(self, x: torch.Tensor, k: torch.LongTensor) -> torch.Tensor:
        """
        x: (n, d) shared dataset (no batch dim)
        k: (bs,) number of clusters per batch
        return: centers (bs, num_init, k_max, d) with the same init for all bs
        """
        n, d = x.shape
        bs,  = k.shape
        m     = self.num_init
        k_max = int(k.max())

        # RNG
        gen = None
        if self.seed is not None:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(self.seed)

        # We build a single base init (m, k, d) and repeat for bs
        centers = torch.empty((m, k_max, d), dtype=x.dtype, device=x.device)

        # Keep just a 2D "picked" mask and the current min squared distances
        picked = torch.zeros((m, n), dtype=torch.bool, device=x.device)

        # Precompute x^2 once
        x2 = x.square().sum(dim=1)                    # (n,)

        # ---- 1) First center: uniform over rows per init ----
        assert n > m, f"n={n} must be > num_init={m}"
        probs0 = torch.full((m, n), 1.0 / n, dtype=torch.float32, device=x.device)
        idx = torch.multinomial(probs0, num_samples=1, replacement=False, generator=gen).squeeze(1)  # (m,)

        centers[:, 0, :] = x.index_select(0, idx)    # (m, d)
        picked[torch.arange(m, device=x.device), idx] = True

        # Initialize best min-squared-distance to chosen set: distance to first center
        c = centers[:, 0, :]                          # (m, d)
        # d2_to_c[r, i] = ||x_i - c_r||^2 = ||x_i||^2 + ||c_r||^2 - 2 x_i·c_r
        c2 = c.square().sum(dim=1)                    # (m,)
        xTc = torch.einsum('nd,md->mn', x, c)         # (m, n)
        best_d2 = (x2[None, :] + c2[:, None] - 2.0 * xTc).clamp_min_(0)  # (m, n)
        best_d2.masked_fill_(picked, 0)               # don't resample picked points

        eps = torch.finfo(x.dtype).eps

        # ---- 2) Remaining centers: incremental D^2 sampling ----
        for nc in range(1, k_max):
            # Sample next index using current best_d2
            pot = best_d2.clamp_min(eps)              # (m, n)
            next_idx = torch.multinomial(pot, num_samples=1, generator=gen).squeeze(1)  # (m,)

            # Add new center
            centers[:, nc, :] = x.index_select(0, next_idx)
            picked[torch.arange(m, device=x.device), next_idx] = True

            # Update best_d2 with distance to the newly picked center only (O(m*n))
            c = centers[:, nc, :]                     # (m, d)
            c2 = c.square().sum(dim=1)                # (m,)
            xTc = torch.einsum('nd,md->mn', x, c)     # (m, n)
            new_d2 = (x2[None, :] + c2[:, None] - 2.0 * xTc).clamp_min_(0)
            best_d2 = torch.minimum(best_d2, new_d2)
            best_d2.masked_fill_(picked, 0)

        # Repeat to all batch items
        return centers.unsqueeze(0).repeat(bs, 1, 1, 1).contiguous()   # (bs, m, k, d)
    
    @torch.no_grad()
    def _m_step(self, x, labels, k_max):

        n,d = x.shape
        bs, m, n_ = labels.shape
        K = int(k_max)
        M = torch.nn.functional.one_hot(labels, num_classes=K).to(x.dtype)  # (bs, m, N, K)
        M = M.permute(0, 1, 3, 2)                                            # (bs, m, K, N)

        nk = M.sum(dim=-1) + 1e-12                                           # (bs, m, K)
        means = torch.einsum('bmkN,Nd->bmkd', M, x)                          # (bs, m, K, D)
        means = means / nk[..., None]                                         # broadcast divide

        return means

    @torch.no_grad()
    def _cluster(
        self, x: Tensor, centers: Tensor, k: LongTensor, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Union[Tensor, Any]]:
        """
        Run Lloyd's k-means algorithm.

        Args:
            x: (N, D)
            centers: (BS, num_init, k_max, D)
            k: (BS, )

        """
     
        n, d = x.size()
        bs, = k.shape
        # mask centers for which  k < k_max with inf to get correct assignment
        k_max = torch.max(k).cpu().item()
        k_max_range = torch.arange(k_max, device=x.device)[None, :].expand(bs, -1)
        k_mask = k_max_range >= k[:, None]
        k_mask = k_mask[:, None, :].expand(bs, self.num_init, -1)
        # print("k shape", k_mask.size())
                # Build invalid-component mask from k (don’t rely on any saved mask)
        k_max = int(k.max())
        k_max_range = torch.arange(k_max, device=x.device)[None, :].expand(bs, -1)
        invalid = (k_max_range >= k[:, None])[:, None, :].expand(bs, self.num_init, k_max)  # (bs,m,k)
  

        for i in range(self.max_iter):
            # print("centers size:", centers.size())
            # centers[k_mask] = float("inf")
            # print("centers size:", centers.size())
            old_centers = centers.clone()
            # get cluster assignments
            c_assign = self._e_step(x, centers, invalid=invalid)
            # update cluster centers
            # print("labels shape:", c_assign.size())
            centers = self._m_step(x, c_assign, k_max)
            if self.tol is not None:
                # calculate center shift
                shift = self._calculate_shift(centers, old_centers, p=2)
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
        # centers[k_mask] = float("inf")
        c_assign = self._e_step(x, centers, invalid=invalid)

        if self.is_for_init:
            return centers 
        else:

            inertia = self._calculate_inertia(x, centers, c_assign)
            best_init = torch.argmin(inertia, dim=-1)
            b_idx = torch.arange(bs, device=x.device)

            return (
                c_assign[b_idx, best_init],
                centers[b_idx, best_init],
                inertia[b_idx, best_init],
            )

    def storage_bytes(t): 
        GIB = 1024 ** 3
        return t.untyped_storage().nbytes() / GIB
    
    # def _pairwise_distance(
    #     self,
    #     x: torch.Tensor,              # (bs, n, d)
    #     centers: torch.Tensor,        # (bs, num_init, k, d)
    #     *,
    #     squared: bool = False,        # keep squared distances to avoid sqrt cost
    #     chunk_n: int | None = None,   # e.g., 8192 to bound (bs*m*chunk*k)
    # ) -> torch.Tensor:
    #     """
    #     Returns L2 distances of shape (bs, num_init, n, k) without copying x.
    #     Uses 4D batched matmul: (bs, m, n, d) @ (bs, m, d, k) -> (bs, m, n, k).
    #     No expand+reshape(...).contiguous() on x; only a small contiguous() on the
    #     transposed centers for fast GEMM.
    #     """
    #     bs, n, d = x.shape
    #     bs2, m, k, d2 = centers.shape
    #     assert bs == bs2 and d == d2

    #     # Broadcast x along num_init as a VIEW (no data copy).
    #     X = x[:, None, :, :].expand(bs, m, n, d)              # (bs, m, n, d) view

    #     # Prepare centers^T as contiguous for matmul (this is much smaller than x replicated m times).
    #     Kt = centers.transpose(-1, -2).contiguous()           # (bs, m, d, k)

    #     # Precompute squared norms (kept small; no (n,k,d) tensor is formed).
    #     C2 = (centers * centers).sum(dim=-1).unsqueeze(-2)    # (bs, m, 1, k)

    #     if chunk_n is None or chunk_n >= n:
    #         # One shot
    #         X2 = (X * X).sum(dim=-1, keepdim=True)            # (bs, m, n, 1)
    #         XC = X @ Kt                                       # (bs, m, n, k)
    #         dist2 = X2 + C2 - 2.0 * XC                        # (bs, m, n, k)
    #     else:
    #         # Chunk along samples to bound peak memory to O(bs*m*chunk_n*k)
    #         out = X.new_empty(bs, m, n, k)
    #         for s in range(0, n, chunk_n):
    #             e = min(s + chunk_n, n)
    #             Xs = X[:, :, s:e, :]                          # (bs, m, b, d) view
    #             X2s = (Xs * Xs).sum(dim=-1, keepdim=True)     # (bs, m, b, 1)
    #             XCs = Xs @ Kt                                 # (bs, m, b, k)
    #             out[:, :, s:e, :] = X2s + C2 - 2.0 * XCs
    #         dist2 = out

    #     dist2.clamp_(min=0.0)
    #     if squared:
    #         return dist2
    #     else:
    #         return dist2.sqrt_()

    def _pairwise_distance(self, x: Tensor, centers: Tensor, **kwargs):
        def storage_bytes(t): 
            GIB = 1024 ** 3
            return t.untyped_storage().nbytes() / GIB
        """
        x:       (bs, n, d)
        centers: (bs, num_init, k, d)
        retourne: (bs, num_init, n, k) distances L2
        """
        n, d = x.shape
        bs, num_init, k, d2 = centers.shape

        # Réplique x le long de num_init sans copie "réelle"
        # x_rep = x[:, None, :, :].expand(bs, num_init, n, d)  # (bs, num_init, n, d)
       # print(x_rep.shape, x_rep.is_contiguous(), x_rep.stride(), storage_bytes(x_rep))  # small

        # Aplatis les dims (bs, num_init) pour un matmul batched propre
        # X_flat = x_rep.reshape(-1, n, d)      # (B, n, d), B = bs*num_init
        # print(X_flat.is_contiguous(), storage_bytes(X_flat))                             
        # X = X_flat.contiguous()                     # (B, n, d)
        # print(X.is_contiguous(), storage_bytes(X))
        # C = centers.reshape(-1, k, d).contiguous()     # (B, k, d)

        # Normes au carré
        X2 = x.square().sum(dim=1, keepdim=True)           # (n, 1)
        C2 = centers.square().sum(dim=3)         # (bs, m , k)

        # Produits scalaires
        # (B, n, d) @ (B, d, k) -> (B, n, k)
        # XC = X @ C.transpose(1, 2)
        XC = torch.einsum('nd,bmkd->bmnk', x, centers) # (bs, m , n, k)


        # Distances au carré (numériquement sûr)
        dist2 = X2[None, None, :, :] + C2[:, :, None, :] - 2.0 * XC # (bs, m, n, k)
        dist2.clamp_(min=0.0)
        dist = dist2.sqrt_()                           # retire sqrt si tu veux l'inertie
        return dist
        # return dist.view(bs, num_init, n, k)

    # def _pairwise_distance(self, x: Tensor, centers: Tensor, **kwargs):
    #     """Calculate pairwise distances between samples in x and all centers."""
    #     # expand tensors to calculate pairwise distance over (d) dimensions
    #     # of each point (n) to each center (k_max)
    #     # for each random restart (num_init) in each batch instance (bs)
        
    #     bs, n, d = x.size()
    #     bs, num_init, k_max, d = centers.size()
    #     x = x[:, None, :, None, :].expand(bs, num_init, n, k_max, d).reshape(-1, d)
    #     centers = (
    #         centers[:, :, None, :, :].expand(bs, num_init, n, k_max, d).reshape(-1, d)
    #     )
    #     return self.distance.pairwise_distance(x, centers, **kwargs).view(
    #         bs, num_init, n, k_max
    #     )

    def _e_step(self, x: Tensor, centers: Tensor, invalid, **kwargs) -> LongTensor:
        """Infer cluster assignment for each sample in x."""
        # dist: (bs, num_init, n, k_max)
        dist = self._pairwise_distance(x, centers)
        # get cluster assignments (center with minimal distance)
      
        dist = dist.masked_fill(invalid[:, :, None, :], float('inf'))
        return torch.argmin(dist, dim=-1)  # type: ignore

    @staticmethod
    @torch.jit.script
    def _calculate_shift(centers: Tensor, old_centers: Tensor, p: int = 2) -> Tensor:
        """Calculate center shift w.r.t. centers from last iteration."""
        # calculate euclidean distance while replacing inf with 0 in sum
        d = torch.norm((centers - old_centers), p=p, dim=-1)
        d[d == float("inf")] = 0
        # sum(d, dim=-1)**2 -> use mean to be independent of number of points
        return torch.mean(d, dim=-1)

    @staticmethod
    @torch.jit.script
    def _calculate_inertia(x: Tensor, centers: Tensor, labels: Tensor) -> Tensor:
        """Compute sum of squared distances of samples
        to their closest cluster center."""
        n, d = x.size()
        bs, m, k, d = centers.shape
        assert m == labels.size(1)
        # select assigned center by label and calculate squared distance
        assigned_centers = centers.gather(
            index=labels[:, :, :, None].expand(
                labels.size(0), labels.size(1), labels.size(2), d
            ),
            dim=2,
        )
        # squared distance to closest center
        d = (
            torch.norm(
                (x[None, None, :, :].expand(bs, m, n, d) - assigned_centers), p=2, dim=-1
            )
            ** 2
        )
        d[d == float("inf")] = 0
        return torch.sum(d, dim=-1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"init: '{self.init_method}', "
            f"num_init: {self.num_init}, "
            f"max_iter: {self.max_iter}, "
            # f"distance: {self.distance}, "
            f"tolerance: {self.tol}, "
            f"normalize: {self.normalize}"
            f")"
        )