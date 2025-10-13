"""
Validation Utilities for Theoretical Guarantees

This module provides tools to validate that the assumptions of Theorem 3.1
are satisfied and to check empirical coverage.

Key checks:
-----------
1. Independence: r and Dcal must be independent
2. Finite resolution: |Z| < ∞
3. Empirical coverage: Verify P{η ∈ Ĉn} ≥ 1-α on test data
4. Empty clusters: Detect and warn about Nz = 0 cases
"""

import warnings
import torch
import numpy as np
from typing import Optional, Dict, Any


class TheoreticalGuaranteesValidator:
    """
    Validates assumptions required for Theorem 3.1 coverage guarantees.

    Theorem 3.1 states that for any z ∈ Z:
        P{ηf,P,r(z) ∈ Ĉn(z; Dn, f)} ≥ 1 - α

    This holds IF AND ONLY IF:
        1. r is fixed independently of Dn (r ⊥ Dn)
        2. |Z| < ∞ (finite number of clusters)
        3. Samples in Dn are i.i.d. from P

    This class helps verify these conditions.
    """

    @staticmethod
    def check_independence(
        resolution_data_hash: int,
        calibration_data_hash: int
    ) -> None:
        """
        Verify that resolution function r was learned independently of calibration data.

        This is the MOST CRITICAL assumption for Theorem 3.1.

        Args:
            resolution_data_hash: Hash of logits used to learn r (e.g., GMM)
            calibration_data_hash: Hash of logits used to compute Ĉn

        Raises:
            ValueError: If hashes match (independence violated)

        Example:
        --------
        >>> # In learn_resolution()
        >>> hash_res = hash(logits_res.cpu().numpy().tobytes())
        >>>
        >>> # In fit()
        >>> hash_cal = hash(logits_cal.cpu().numpy().tobytes())
        >>> TheoreticalGuaranteesValidator.check_independence(hash_res, hash_cal)
        """
        if resolution_data_hash == calibration_data_hash:
            raise ValueError(
                "❌ CRITICAL VIOLATION OF THEOREM 3.1: INDEPENDENCE\n"
                "\n"
                "The same data was used for:\n"
                "  1. Learning resolution function r (e.g., GMM clustering)\n"
                "  2. Computing confidence intervals Ĉn\n"
                "\n"
                "This violates the independence assumption r ⊥ Dcal.\n"
                "\n"
                "Consequence: Coverage guarantee P{ηf,P,r(z) ∈ Ĉn(z)} ≥ 1-α "
                "DOES NOT HOLD!\n"
                "\n"
                "Solution: Use sample splitting (see code.utils.sample_splitting.SampleSplitter)\n"
                "  - Split data into D_resolution and D_calibration\n"
                "  - Learn r on D_resolution\n"
                "  - Compute Ĉn on D_calibration\n"
            )

    @staticmethod
    def check_finite_resolution(n_clusters: int) -> None:
        """
        Verify that |Z| is finite and reasonable.

        Args:
            n_clusters: Number of clusters |Z|

        Raises:
            ValueError: If n_clusters is not a positive integer

        Warns:
            If n_clusters is very large (risk of many empty clusters)
        """
        if not isinstance(n_clusters, (int, np.integer)) or n_clusters <= 0:
            raise ValueError(
                f"Number of clusters |Z| must be a positive integer, "
                f"got {n_clusters} (type: {type(n_clusters)})"
            )

        if n_clusters > 10000:
            warnings.warn(
                f"⚠️  Very large number of clusters: |Z| = {n_clusters}\n"
                f"\n"
                f"With limited calibration data, many clusters may be empty (Nz=0).\n"
                f"This leads to uninformative intervals Ĉn(z) = [0, 1].\n"
                f"\n"
                f"Recommendation: Reduce n_clusters or increase calibration size.",
                UserWarning
            )

    @staticmethod
    def check_sample_size(
        n_calibration: int,
        n_clusters: int,
        alpha: float = 0.05
    ) -> None:
        """
        Check if calibration sample size is adequate for the number of clusters.

        Rule of thumb: Need at least ~20/α samples per cluster on average
        for reasonable interval widths.

        Args:
            n_calibration: Number of calibration samples
            n_clusters: Number of clusters
            alpha: Confidence level parameter

        Warns:
            If samples per cluster is too low
        """
        avg_samples_per_cluster = n_calibration / n_clusters
        min_recommended = 20 / alpha  # Heuristic: 20/α ≈ 400 for α=0.05

        if avg_samples_per_cluster < min_recommended:
            warnings.warn(
                f"⚠️  Low samples per cluster\n"
                f"\n"
                f"Average samples per cluster: {avg_samples_per_cluster:.1f}\n"
                f"Recommended minimum: {min_recommended:.0f} (for α={alpha})\n"
                f"\n"
                f"With few samples per cluster, confidence intervals will be wide.\n"
                f"\n"
                f"Options:\n"
                f"  1. Reduce n_clusters (e.g., from {n_clusters} to {int(n_calibration/min_recommended)})\n"
                f"  2. Increase calibration set size\n"
                f"  3. Accept wider intervals",
                UserWarning
            )

    @staticmethod
    def check_empty_clusters(
        cluster_counts: torch.Tensor,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Check for empty clusters (Nz = 0) and compute statistics.

        When Nz = 0, Theorem 3.1 sets Ĉn(z) = [0, 1] (uninformative).

        Args:
            cluster_counts: Tensor of shape (n_clusters,) with counts per cluster
            threshold: Warn if fraction of empty clusters exceeds this

        Returns:
            Dictionary with:
                - 'n_empty': Number of empty clusters
                - 'n_total': Total number of clusters
                - 'fraction_empty': Fraction of empty clusters
                - 'empty_indices': List of empty cluster indices
                - 'has_warning': True if fraction exceeds threshold

        Example:
        --------
        >>> counts = detector.cluster_counts  # (n_clusters,)
        >>> diag = TheoreticalGuaranteesValidator.check_empty_clusters(counts)
        >>> if diag['has_warning']:
        ...     print(f"{diag['fraction_empty']:.1%} clusters are empty!")
        """
        empty_mask = (cluster_counts == 0)
        n_empty = empty_mask.sum().item()
        n_total = cluster_counts.numel()
        fraction_empty = n_empty / n_total if n_total > 0 else 0.0

        empty_indices = torch.where(empty_mask)[0].tolist()

        has_warning = (fraction_empty > threshold)

        if has_warning:
            warnings.warn(
                f"⚠️  High fraction of empty clusters: {fraction_empty:.1%}\n"
                f"\n"
                f"Empty clusters: {n_empty} / {n_total}\n"
                f"\n"
                f"For empty clusters (Nz=0), the confidence interval is set to [0,1],\n"
                f"which is uninformative (covers all possible error probabilities).\n"
                f"\n"
                f"Predictions falling in empty clusters will receive upper bound = 1.0.\n"
                f"\n"
                f"Recommendation:\n"
                f"  - Reduce n_clusters (current: {n_total})\n"
                f"  - Increase calibration set size\n"
                f"  - Use different clustering method or hyperparameters",
                UserWarning
            )

        return {
            'n_empty': n_empty,
            'n_total': n_total,
            'fraction_empty': fraction_empty,
            'empty_indices': empty_indices,
            'has_warning': has_warning
        }

    @staticmethod
    def compute_empirical_coverage(
        detector,
        test_loader: torch.utils.data.DataLoader,
        alpha: float = 0.05,
        device: Optional[torch.device] = None
    ) -> Dict[str, float]:
        """
        Compute empirical coverage on test set.

        Checks if the true error indicator E ∈ {0,1} falls within the
        confidence interval Ĉn(z) for each test sample.

        NOTE: This is a sanity check, not a formal proof. Empirical coverage
        may be < 1-α due to:
          1. Finite sample effects
          2. Violations of assumptions (e.g., r not independent of Dcal)
          3. Model being incorrect

        Args:
            detector: MegaPartitionDetector instance (already fitted)
            test_loader: DataLoader for test data
            alpha: Confidence level (default: 0.05 for 95% coverage)
            device: Device for computation

        Returns:
            Dictionary with:
                - 'empirical_coverage': Fraction of test samples where E ∈ Ĉn
                - 'expected_coverage': 1 - α (theoretical minimum)
                - 'n_covered': Number of samples covered
                - 'n_total': Total number of test samples
                - 'coverage_deficit': max(0, expected - empirical)
                - 'has_violation': True if deficit > tolerance

        Warns:
            If empirical coverage is significantly below 1-α

        Example:
        --------
        >>> detector.fit(...)
        >>> cov = TheoreticalGuaranteesValidator.compute_empirical_coverage(
        ...     detector, test_loader, alpha=0.05
        ... )
        >>> print(f"Coverage: {cov['empirical_coverage']:.3f} "
        ...       f"(expected ≥ {cov['expected_coverage']:.3f})")
        """
        if device is None:
            device = detector.device

        detector.model.to(device)
        detector.model.eval()

        n_covered = 0
        n_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Get model predictions
                logits = detector.model(inputs)
                preds = torch.argmax(logits, dim=1)

                # True error indicators
                errors = (preds != labels).float()  # 1 if wrong, 0 if correct

                # Get cluster assignments
                clusters = detector.predict_clusters(logits=logits)

                # Get confidence intervals for these clusters
                # intervals shape: (batch_size, 2) where [:, 0] is lower, [:, 1] is upper
                if clusters.dim() == 1:
                    # (batch_size,)
                    lower = detector.cluster_intervals[..., 0].gather(0, clusters)
                    upper = detector.cluster_intervals[..., 1].gather(0, clusters)
                else:
                    # (bs, batch_size) - take first batch
                    lower = detector.cluster_intervals[0, :, 0].gather(0, clusters[0])
                    upper = detector.cluster_intervals[0, :, 1].gather(0, clusters[0])

                # Check if error is within interval
                in_interval = (errors >= lower) & (errors <= upper)

                n_covered += in_interval.sum().item()
                n_total += errors.numel()

        empirical_coverage = n_covered / n_total if n_total > 0 else 0.0
        expected_coverage = 1 - alpha
        coverage_deficit = max(0, expected_coverage - empirical_coverage)

        # Tolerance: Allow 2% deviation (due to finite sample effects)
        tolerance = 0.02
        has_violation = (coverage_deficit > tolerance)

        result = {
            'empirical_coverage': empirical_coverage,
            'expected_coverage': expected_coverage,
            'n_covered': n_covered,
            'n_total': n_total,
            'coverage_deficit': coverage_deficit,
            'has_violation': has_violation
        }

        if has_violation:
            warnings.warn(
                f"⚠️  Empirical coverage below expected level\n"
                f"\n"
                f"Empirical coverage: {empirical_coverage:.3f}\n"
                f"Expected (1-α): {expected_coverage:.3f}\n"
                f"Deficit: {coverage_deficit:.3f}\n"
                f"\n"
                f"Possible causes:\n"
                f"  1. Finite sample effects (test set too small)\n"
                f"  2. Independence assumption violated (r not independent of Dcal)\n"
                f"  3. Data distribution shift between calibration and test\n"
                f"  4. Implementation bug\n"
                f"\n"
                f"This suggests the theoretical guarantees may not hold in practice.",
                UserWarning
            )

        return result


def validate_detector_assumptions(
    detector,
    resolution_data_hash: Optional[int] = None,
    calibration_data_hash: Optional[int] = None,
    n_calibration: int = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Run all validation checks on a fitted detector.

    Args:
        detector: MegaPartitionDetector instance
        resolution_data_hash: Hash of data used to learn r (if available)
        calibration_data_hash: Hash of data used for Ĉn (if available)
        n_calibration: Number of calibration samples
        alpha: Confidence level

    Returns:
        Dictionary with validation results from all checks
    """
    validator = TheoreticalGuaranteesValidator()
    results = {}

    # Check 1: Independence
    if resolution_data_hash is not None and calibration_data_hash is not None:
        try:
            validator.check_independence(resolution_data_hash, calibration_data_hash)
            results['independence'] = {'status': 'OK', 'violated': False}
        except ValueError as e:
            results['independence'] = {'status': 'VIOLATED', 'violated': True, 'error': str(e)}

    # Check 2: Finite resolution
    n_clusters = detector.list_n_cluster[0].item() if hasattr(detector, 'list_n_cluster') else None
    if n_clusters is not None:
        try:
            validator.check_finite_resolution(n_clusters)
            results['finite_resolution'] = {'status': 'OK', 'n_clusters': n_clusters}
        except (ValueError, Warning) as e:
            results['finite_resolution'] = {'status': 'WARNING', 'n_clusters': n_clusters, 'message': str(e)}

    # Check 3: Sample size
    if n_calibration is not None and n_clusters is not None:
        validator.check_sample_size(n_calibration, n_clusters, alpha)
        results['sample_size'] = {
            'n_calibration': n_calibration,
            'n_clusters': n_clusters,
            'avg_per_cluster': n_calibration / n_clusters
        }

    # Check 4: Empty clusters
    if hasattr(detector, 'cluster_counts'):
        empty_diag = validator.check_empty_clusters(detector.cluster_counts)
        results['empty_clusters'] = empty_diag

    return results
