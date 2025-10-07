#
from typing import Any, Optional, Tuple, Union
from warnings import warn

import torch
from torch import LongTensor, Tensor

from .distances import BaseDistance, CosineSimilarity
from .kmeans import KMeans
from .utils import SoftClusterResult

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
        init_method: str = "random",
        num_init: int = 1,
        max_iter: int = 100,
        normalize: str = "unit",
        tol: float = 1e-5,
        n_clusters: Optional[int] = 8,
        verbose: bool = True,
        seed: Optional[int] = 123,
        temp: float = 5.0,
        reg_covar = 1e-06,
        **kwargs,
    ):
        super(SoftKMeans, self).__init__(
            init_method=init_method,
            num_init=num_init,
            max_iter=max_iter,
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
      
        self.reg_covar = reg_covar
        
   
    

    # def _estimate_log_gaussian_prob_diag(
    #     self,
    #     x: Tensor,                 # (bs, n, d)
    #     means: Tensor,             # (bs, m, k, d)
    #     cov_diag: Tensor,          # (bs, m, k, d)  (variances)
    #     eps: float = 1e-12,
    # ) -> Tensor:
    #     """
    #     Returns log N(x; mean_k, diag(cov_k)) for all (bs, m, n, k).
    #     No big (n,k,d) intermediates are materialized.
    #     means : (bs, num_init, k, d)
    #     cov_diag : (bs, num_init, k, d)
    #     x : (n, d)

    #     """
    #     bs, n, d = x.shape
    #     bs2, m, k, d2 = means.shape
    #     assert (bs, d) == (bs2, d2)
    #     B = bs * m

    #     # Merge (bs, m) to a single batch B for two GEMMs
    #     # print(" xhsape:", x.size())
    #     # X = x[:, None, :, :].expand(bs, m, n, d).reshape(B, n, d).contiguous()      # (B,n,d)
    #     # MU = means.reshape(B, k, d).contiguous()    
    #     X = x[:, None, :, :].expand(bs, m, n, d).reshape(B, n, d).contiguous()      # (B,n,d)
    #     MU = means.reshape(B, k, d).contiguous()                                     # (B,k,d)
    #     VAR = cov_diag.clamp_min(eps).reshape(B, k, d).contiguous()                  # (B,k,d)
    #     # print("VAR shape:", VAR.size())
    #     # print("X shape:", X.size())

    #     # Precisions τ = 1/σ^2 and precision-weighted means
    #     TAU = VAR.reciprocal()                              # (B,k,d)
    #     # print("tau shape:", TAU.size())
    #     MU_TAU = MU * TAU                                   # (B,k,d)
    #     # print("MU_TAU shape:", MU_TAU.size())
    #     MU2_TAU_SUM = (MU.square() * TAU).sum(dim=2)        # (B,k)
    #     # print("MU2_TAU_SUM shape:", MU2_TAU_SUM.size())

    #     # Quadratic form: (x^2)·τ  - 2 x·(μ⊙τ)  +  (μ^2)·τ
    #     # print("x squra shape:", X.square().size())
    #     # print("tau transpose shape:", TAU.transpose(1, 2).size())
    #     X2_TAU   = X.square() @ TAU.transpose(1, 2)         # (B,n,k)
    #     # print("X2_TAU shape:", X2_TAU.size())
    #     X_MU_TAU = X @ MU_TAU.transpose(1, 2)               # (B,n,k)
    #     # print("X_MU_TAU shape:", X_MU_TAU.size())
    #     quad = X2_TAU - 2.0 * X_MU_TAU + MU2_TAU_SUM.unsqueeze(1)   # (B,n,k)
    #     # print("quad shape:", quad.size())
    #     # quad.clamp_(min=0.0)                                # numeric safety

    #     # Log-det term: sum_d log σ^2  (per component)
    #     LOGDET = VAR.log().sum(dim=2)                       # (B,k)

    #     # Final log-pdf per component
    #     log_prob = -0.5 * (d * torch.log(torch.tensor(2.0 * torch.pi, dtype=X.dtype, device=X.device))
    #                     + LOGDET.unsqueeze(1) + quad)    # (B,n,k)
    #     log_prob = log_prob.view(bs, m, n, k)
    #     # print("log_prob shape:", log_prob.size())
    #     return log_prob
    
    def _estimate_log_gaussian_prob_diag(
        self,
        x: Tensor,                 # (bs, n, d)
        means: Tensor,             # (bs, m, k, d)
        cov_diag: Tensor,          # (bs, m, k, d)  (variances)
        eps: float = 1e-12,
    ) -> Tensor:
        """
        Returns log N(x; mean_k, diag(cov_k)) for all (bs, m, n, k).
        No big (n,k,d) intermediates are materialized.
        means : (bs, num_init, k, d)
        cov_diag : (bs, num_init, k, d)
        x : (n, d)

        """
        n, d = x.shape
        bs, m, k, d2 = means.shape
        assert d == d2



        # MU = means.reshape(B, k, d).contiguous()                                     # (B,k,d)
        # VAR = cov_diag.clamp_min(eps).reshape(B, k, d).contiguous()                  # (B,k,d)
        # print("VAR shape:", VAR.size())
        # print("X shape:", X.size())
        MU = means
        # Precisions τ = 1/σ^2 and precision-weighted means
        TAU = cov_diag.reciprocal()                              # (bs, m, k, d)
        # print("tau shape:", TAU.size())
        MU_TAU = MU * TAU                                   # (bs, m, k, d)
        # print("MU_TAU shape:", MU_TAU.size())
        MU2_TAU_SUM = (MU.square() * TAU).sum(dim=3)        # (bs, m, k)
        # print("MU2_TAU_SUM shape:", MU2_TAU_SUM.size())

        # Quadratic form: (x^2)·τ  - 2 x·(μ⊙τ)  +  (μ^2)·τ
        # print("x squra shape:", X.square().size())
        # print("tau transpose shape:", TAU.transpose(1, 2).size())
        X2_TAU   = torch.einsum("nd,bmkd->bmnk", x.square(), TAU)         # (b, m, n, k)
        # print("X2_TAU shape:", X2_TAU.size())
        X_MU_TAU = torch.einsum("nd,bmkd->bmnk", x, MU_TAU)               # (b, m, n, k)
        # print("X_MU_TAU shape:", X_MU_TAU.size())
        quad = X2_TAU - 2.0 * X_MU_TAU + MU2_TAU_SUM.unsqueeze(2)   # (bs, m, n, k)
        # print("quad shape:", quad.size())
        # quad.clamp_(min=0.0)                                # numeric safety

        # Log-det term: sum_d log σ^2  (per component)
        LOGDET = cov_diag.log().sum(dim=3)                       # (bs, m, k)

        # Final log-pdf per component
        log_prob = -0.5 * (d * torch.log(torch.tensor(2.0 * torch.pi, dtype=x.dtype, device=x.device))
                        + LOGDET.unsqueeze(2) + quad)    # (bs, m, n, k)
        # log_prob = log_prob.view(bs, m, n, k)
        # print("log_prob shape:", log_prob.size())
        return log_prob
    
    def _estimate_log_weights(self, weights):
        return torch.log(weights)
    
    def _estimate_weighted_log_prob(self, x, weights, means, cov_diag):
        return self._estimate_log_gaussian_prob_diag(x, means, cov_diag) + self._estimate_log_weights(weights)[:, :, None, :]

    def _estimate_log_prob_resp(self, x, weights, means, cov_diag, invalid):

        weighted_log_prob = self._estimate_weighted_log_prob(x, weights, means, cov_diag)
        weighted_log_prob = weighted_log_prob.masked_fill(invalid[:, :, None, :], float('-inf'))
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=-1)

        log_resp = weighted_log_prob - log_prob_norm.unsqueeze(-1)
        return log_prob_norm, log_resp

    def _e_step(self, x, weights, means, cov_diag, invalid):

        log_prob_norm, log_resp = self._estimate_log_prob_resp(x, weights, means, cov_diag, invalid)

        return torch.mean(log_prob_norm, dim=-1), log_resp 


    def _m_step(self, X, log_resp, invalid):

        weights, means, covariances = self._estimate_gaussian_parameters(
            X,
            resp=log_resp.exp(),
            reg_covar=self.reg_covar,
        )
        weights = weights.masked_fill(invalid, 0)
        weights /= weights.sum(dim=2, keepdim=True)

        return weights, means, covariances

    # def _estimate_gaussian_parameters(self, x, resp, reg_covar, covariance_type="diag"):
    #     """
        
    #     Estimate the Gaussian distribution parameters.
    #     Parameters
    #     ----------
    #     x : array-like of shape (B, n, d)

    #     resp : array-like of shape (B, num_init, n, k)


    #     """

    #     bs, n, d = x.size()
    #     bs2, num_init, n2, k = resp.size()
    #     assert bs == bs2 and n == n2

    #     # print("shape resp before:", resp.size())
    #     B = bs * num_init
    #     x_rep = x[:, None, :, :].expand(bs, num_init, n, d).reshape(B, n, d).contiguous()  # (bs, num_init, n, d)
    #     resp = resp.reshape(B, n, k).contiguous()   # (B, n, k)
    #     # print("shape resp before:", resp.size())
    #     nk = resp.sum(dim=1) + 10 * torch.finfo(resp.dtype).eps

    #     # print("resp.transpose(1, 2) shape:", resp.transpose(1, 2).size())
    #     # print("x rep shape:", x_rep.size())
    #     means = (resp.transpose(1, 2) @  x_rep )  
    #     # print("shape means before:", means.size())
    #     # print("shape nk before:", nk.size())
    #     means = means * nk[:,:, None].reciprocal()  # (B, k, d)
    #     covariances = {
    #     # "full": _estimate_gaussian_covariances_full,
    #     # "tied": _estimate_gaussian_covariances_tied,
    #     "diag": self._estimate_gaussian_covariances_diag,
    #     # "spherical": _estimate_gaussian_covariances_spherical,
    #     }[covariance_type](resp, x_rep, nk, means, reg_covar)
    #     return nk.view(bs, num_init, k), means.view(bs, num_init, k, d), covariances.view(bs, num_init, k, d)

    def _estimate_gaussian_parameters(self, x, resp, reg_covar, covariance_type="diag"):
        """
        
        Estimate the Gaussian distribution parameters.
        Parameters
        ----------
        x : array-like of shape (bs, m, n, d)

        resp : array-like of shape (b, m, n, k)


        """

        n, d = x.size()
        bs, num_init, n2, k = resp.size()
        assert n == n2

        nk = resp.sum(dim=2) + 10 * torch.finfo(resp.dtype).eps # (bs, m, k)
        means = torch.einsum('bmnk, nd -> bmkd', resp, x)  # (b, m, k, d)
        means = means * nk[:,:, :, None].reciprocal()  # (B, k, d)
        covariances = {
        # "full": _estimate_gaussian_covariances_full,
        # "tied": _estimate_gaussian_covariances_tied,
        "diag": self._estimate_gaussian_covariances_diag,
        # "spherical": _estimate_gaussian_covariances_spherical,
        }[covariance_type](resp, x, nk, means, reg_covar)
        return nk, means, covariances


    def _estimate_gaussian_covariances_diag(self, resp, x, nk, means, reg_covar):
        """Estimate the diagonal covariance vectors.

        Parameters
        ----------
        responsibilities : array-like of shape (B, n_samples, n_components)

        X : array-like of shape (n, d)

        nk : array-like of shape (bs, m, k)

        resp : (bs, m, n, k)

        means : array-like of shape (bs, m, k, d)

        reg_covar : float

        Returns
        -------
        covariances : array, shape (n_components, d)
            The covariance vector of the current components.
        """
        avg_X2 = torch.einsum('bmnk, nd -> bmkd', resp, x.square()) / nk[:, :, :, None]
        # avg_X2 = resp.transpose(1, 2) @  torch.square(x_rep) / nk[:, :, None]
        avg_means2 = torch.square(means)
        return avg_X2 - avg_means2 + reg_covar

    # def _estimate_gaussian_covariances_diag(self, resp, x_rep, nk, means, reg_covar):
    #     """Estimate the diagonal covariance vectors.

    #     Parameters
    #     ----------
    #     responsibilities : array-like of shape (B, n_samples, n_components)

    #     X : array-like of shape (B, n_samples, n_features)

    #     nk : array-like of shape (B, n_components,)

    #     means : array-like of shape (B, n_components, n_features)

    #     reg_covar : float

    #     Returns
    #     -------
    #     covariances : array, shape (n_components, n_features)
    #         The covariance vector of the current components.
    #     """
    #     avg_X2 = resp.transpose(1, 2) @  torch.square(x_rep) / nk[:, :, None]
    #     avg_means2 = torch.square(means)
    #     return avg_X2 - avg_means2 + reg_covar


    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm
    
    @torch.no_grad()
    def _cluster(
        self, x: Tensor, weights, means: Tensor, cov_diag,  k: LongTensor, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor, Union[Tensor, Any]]:
        """
        Run Lloyd's k-means algorithm.

        Args:
            x: (BS, N, D)
            weights: (BS, num_init, k_max)
            means: (BS, num_init, k_max, D)
            cov_diag: (BS, num_init, k_max, D)
            k: (BS, )

        """
        n, d = x.size()
        bs, = k.shape
        k_max = int(k.max())
        k_max_range = torch.arange(k_max, device=x.device)[None, :].expand(bs, -1)
        self.invalid = (k_max_range >= k[:, None])[:, None, :].expand(bs, self.num_init, k_max)  # (bs,m,k)

        # Track per-init lower bound
        lower_bound = torch.full((bs, self.num_init), -float('inf'), device=x.device, dtype=x.dtype)
        self.n_iter = None

        # print("intialization shape means:", means.size())
        # print('intialization shape cov_diag:', cov_diag.size())
        # print("intialization shape weights:", weights.size())

        for i in range(self.max_iter):
            # means[k_mask] = float("inf")
            # weights[k_mask] = 0
            # cov_diag[k_mask] = float("inf")
            prev_lower_bound = lower_bound
            # get cluster assignments
            log_prob_norm, log_resp = self._e_step(x, weights, means, cov_diag, self.invalid) # (bs, num_init), (bs, num_init, n, k_max)
            # print("shape log_resp:", log_resp.size())
            # print("shape log_prob_norm:", log_prob_norm.size())
            # update cluster centers
            weights, means, cov_diag = self._m_step(x, log_resp, invalid=self.invalid)


            lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)
           

            change = lower_bound - prev_lower_bound
            if (torch.abs(change) < self.tol).all():
                converged = True
                self.n_iter = {i + 1}
                break
        # select best rnd restart according to inertia
        
        if self.n_iter is None:
            self.n_iter = {self.max_iter}
        # means[k_mask] = float("inf")
        # print("lower_bound", lower_bound.shape)
        log_prob_norm, log_resp = self._e_step(x, weights, means, cov_diag, self.invalid)
        # print("shape log_prob_norm:", log_prob_norm.size())
        # inertia = self._calculate_inertia(x, means, c_assign)
        best_init = torch.argmax(lower_bound, dim=-1)
        b_idx = torch.arange(bs, device=x.device)

        return (
            log_resp[b_idx, best_init],
            weights[b_idx, best_init],
            means[b_idx, best_init],
            cov_diag[b_idx, best_init],
            log_prob_norm[b_idx, best_init]
        )
    

    def _cov_diag_init(self, x, k, **kwargs):
        """Choose k random nodes as initial centers.

        Args:
            x: (BS, N, D)
            k: (BS, )

        Returns:
            cov_diag: (BS, num_init, k_max, D)

        """

        n, d = x.size()
        bs, = k.shape
        k_max = torch.max(k).cpu().item()

        # if self.seed is not None:
        #     # make random init reproducible independent of current iteration,
        #     # which otherwise would step and change the torch generator state
        #     gen = torch.Generator(device=x.device)
        #     gen.manual_seed(self.seed)
        # else:
        #     gen = None

        cov_diag = torch.ones((bs, self.num_init, k_max, d), device=x.device, dtype=x.dtype)
        # cov_diag = cov_diag.expand(bs, self.num_init, k_max, d).contiguous()
        return cov_diag

    def _weights_init(self, k: torch.LongTensor, device=None, dtype=None, **kwargs) -> torch.Tensor:
        """
        k: (bs,) number of clusters per batch item
        returns: weights (bs, num_init, k_max) with uniform weights over valid components
        """
        if device is None: device = k.device
        if dtype  is None: dtype  = torch.float32

        bs = k.shape[0]
        k_max = int(k.max())

        k_range = torch.arange(k_max, device=device).unsqueeze(0).expand(bs, -1)  # (bs, k_max)
        mask = (k_range < k.unsqueeze(1))                                         # True where component is valid

        # Row-wise divide by k: valid entries become 1/k[b], padded stay 0
        w = mask.to(dtype) / k.to(dtype).unsqueeze(1)                              # (bs, k_max)

        # replicate across num_init (same weights for each restart)
        w = w.unsqueeze(1).expand(bs, self.num_init, k_max).contiguous()           # (bs, num_init, k_max)
        return w

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
    #     # print("shape means:", self._result.means.size())
    #     # print("shape weights:", self._result.weights.size())
    #     # print("shape cov_diag:", self._result.cov_diags.size())
    #     # print('shape invalid:', self.invalid.size())
    #     _, log_resp = self._e_step(
    #         x, 
    #         self._result.weights.unsqueeze(1), 
    #         self._result.means.unsqueeze(1), 
    #         self._result.cov_diags.unsqueeze(1), 
    #         self.invalid
    #         )
    #     return log_resp.squeeze(1).argmax(dim=-1)  # type: ignore

    @torch.no_grad()
    def predict(self, x: Tensor, **kwargs) -> LongTensor:
        """Hard assignments (argmax over responsibilities)."""
        assert self.is_fitted, "Call fit() first."
        x = self._check_x(x)  # (bs, n, d)
        if self.normalize is not None:
            x = self._normalize(x, self.normalize, self.eps)

        # Fitted params (one set per batch)
        means    = self._result.means        # (bs, k_max, d)
        cov_diags = self._result.cov_diags    # (bs, k_max, d)
        weights  = self._result.weights      # (bs, k_max)
        k_vec    = self._result.k            # (bs,)

        n, d = x.shape
        bs, k_max, d2 = means.shape
        assert  d == d2

        # Build invalid-component mask from k (don’t rely on any saved mask)
        k_range = torch.arange(k_max, device=x.device).expand(bs, -1)    # (bs, k_max)
        invalid = (k_range >= k_vec.unsqueeze(1)).unsqueeze(1)  

        _, log_resp = self._e_step(
            x, 
            weights.unsqueeze(1), 
            means.unsqueeze(1), 
            cov_diags.unsqueeze(1), 
            invalid
            )
        return log_resp.squeeze(1).argmax(dim=-1)  # type: ignore

    
    def fit_predict(
        self,
        x: Tensor,
        k: Optional[Union[LongTensor, Tensor, int]] = None,
        centers: Optional[Tensor] = None,
        **kwargs,
    ) -> LongTensor:
        """Compute cluster centers and predict cluster index for each sample.

        Args:
            x: input features/coordinates (N, D)
            k: (bs,)
            centers: optional batch of initial centers to use (BS, K, D)
            **kwargs: additional kwargs for initialization or cluster procedure

        Returns:
            batch tensor of cluster labels for each sample (BS, N)

        """
        self._result = self(x, k=k, centers=centers, **kwargs)
        return self._result.log_resp.argmax(dim=-1)  # type: ignore

    @torch.no_grad()
    def _init_kmeans(self, x: Tensor, k: LongTensor, **kwargs) -> Tensor:
        """Wrapper to apply different methods for
        initialization of initial centers (centroids)."""
        km = KMeans(
            init_method="k-means++",
            num_init=self.num_init,
            max_iter=self.max_iter,
            normalize=self.normalize,
            tol=self.tol,
            verbose=False,
            seed=self.seed,
        )
        km.fit(x, k=k, **kwargs)
        means = km._result.centers.unsqueeze(1).expand(-1, self.num_init, -1, -1).contiguous()
     
        return means

    
    @torch.no_grad()
    def _center_init(self, x: Tensor, k: LongTensor, **kwargs) -> Tensor:
        """Wrapper to apply different methods for
        initialization of initial centers (centroids)."""
        if self.init_method == "random":
            return self._init_rnd(x, k)
        elif self.init_method == "k-means++":
            return self._init_plus(x, k)
        elif self.init_method == "kmeans":
            return self._init_kmeans(x, k, **kwargs)
        else:
            raise ValueError(f"unknown initialization method: {self.init_method}.")
    def forward(
        self,
        x: Tensor,
        weights: Optional[Tensor] = None,
        means: Optional[Tensor] = None,
        cov_diag: Optional[Tensor] = None,
        k: Optional[Union[LongTensor, Tensor, int]] = None,
        **kwargs,
    ) -> SoftClusterResult:
        """torch.nn like forward pass.

        Args:
            x: input features/coordinates (N, D)
            k: (bs,)
            centers: optional batch of initial centers to use (BS, K, D)
            **kwargs: additional kwargs for initialization or cluster procedure

        Returns:
            ClusterResult tuple

        """
        # print("x shape:", x.size())
        x = self._check_x(x)
        self.n, self.d = x.shape
        x_ = x
        self.bs = k.shape[0]
        k = self._check_k(k,  device=x.device)
      
        # normalize input
        if self.normalize is not None:
            x = self._normalize(x, self.normalize, self.eps)
        # init centers
        if means is None:
            means = self._center_init(x, k, **kwargs) # (bs, num_init, k_max, d)
        if cov_diag is None:
            cov_diag = self._cov_diag_init(x, k, **kwargs) # (bs, num_init, k_max, d)
        if weights is None:
            weights = self._weights_init(k, device=x.device, dtype=x.dtype) # (bs, num_init, k_max)
        means = self._check_centers(
            means, dims=(self.bs, self.n, self.d), dtype=x.dtype, device=x.device
        )

        log_resp, new_weights, new_means, new_cov_diags, lower_bound = self._cluster(
            x, weights, means, cov_diag, k, **kwargs
        )
        return SoftClusterResult(
            log_resp=log_resp,  # type: ignore
            means=new_means,
            weights=new_weights,
            cov_diags=new_cov_diags,
            lower_bound=lower_bound,
            k=k,

        )


    # @torch.no_grad()
    # def _pairwise_distance(
    #     self,
    #     x: torch.Tensor,              # (bs, n, d)
    #     centers: torch.Tensor,        # (bs, num_init, k, d)
    #     *,
    #     precisions: torch.Tensor | None = None,   # (bs, num_init, k, d) or broadcastable
    #     variances: torch.Tensor | None = None,    # same shapes; used if precisions is None
    #     return_sqrt: bool = True,
    #     eps: float = 1e-12,
    # ) -> torch.Tensor:
    #     """
    #     Diagonal-Mahalanobis distances using batched matmuls.

    #     Returns: (bs, num_init, n, k)
    #     If both `precisions` and `variances` are None -> falls back to plain L2.
    #     """

    #     bs, n, d = x.shape
    #     _, num_init, k, d_ = centers.shape
    #     assert d == d_, "x and centers must have same feature dim"

    #     # Reshape to merge (bs, num_init) for one batched GEMM per init
    #     B = bs * num_init
    #     X = x[:, None, :, :].expand(bs, num_init, n, d).reshape(B, n, d).contiguous()  # (B, n, d)
    #     C = centers.reshape(B, k, d).contiguous()                                       # (B, k, d)

    #     # Decide precision τ = 1/σ^2 per component dimension
    #     if precisions is None:
    #         if variances is None:
    #             # Euclidean fallback: τ = 1 for all dims
    #             tau = torch.ones((B, k, d), dtype=X.dtype, device=X.device)
    #         else:
    #             var = variances
    #             # Broadcast var to (bs, num_init, k, d) then to (B, k, d)
    #             while var.dim() < 4:
    #                 var = var.unsqueeze(0)
    #             var = var.expand(bs, num_init, k, d).reshape(B, k, d).contiguous()
    #             tau = (var.clamp_min(eps)).reciprocal_()
    #     else:
    #         tau = precisions
    #         while tau.dim() < 4:
    #             tau = tau.unsqueeze(0)
    #         tau = tau.expand(bs, num_init, k, d).reshape(B, k, d).contiguous()

    #     # Precompute per-component pieces
    #     # (i) x^2 ⋅ τ_k  -> (B, n, k)
    #     X2_tau = (X.square()) @ tau.transpose(1, 2)

    #     # (ii) x ⋅ (μ_k ⊙ τ_k) -> (B, n, k)
    #     mu_tau = C * tau
    #     X_mu_tau = X @ mu_tau.transpose(1, 2)

    #     # (iii) (μ_k^2) ⋅ τ_k -> (B, 1, k), broadcast over n
    #     mu2_tau_sum = (C.square() * tau).sum(dim=2).unsqueeze(1)  # (B, 1, k)

    #     # Quadratic form for all pairs
    #     dist2 = X2_tau - 2.0 * X_mu_tau + mu2_tau_sum            # (B, n, k)
    #     dist2.clamp_(min=0.0)

    #     if return_sqrt:
    #         dist = dist2.sqrt_()
    #     else:
    #         dist = dist2

    #     return dist.view(bs, num_init, n, k)
    