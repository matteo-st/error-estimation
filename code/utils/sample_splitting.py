"""
Sample Splitting Utility for Theoretical Guarantees

This module implements sample splitting to ensure independence between:
- D_resolution: data used to learn the resolution function r
- D_calibration: data used to construct confidence intervals Ĉn

This independence is REQUIRED by Theorem 3.1 for valid coverage guarantees.

References:
    Paper Section 4: "Because Theorem 3.1 requires r to be fixed independently
    of the calibration data used for interval construction, one should enforce
    independence by sample splitting"
"""

import random
import warnings
import torch
from typing import Dict, Optional, Any


class SampleSplitter:
    """
    Manages sample splitting to guarantee independence between resolution
    learning and confidence interval construction.

    Key Principles:
    ---------------
    1. Theorem 3.1 requires r ⊥ Dcal (independence)
    2. If dataset is large enough, split into D_res and D_cal
    3. If dataset is too small, warn user that guarantees may not hold

    Example:
    --------
    >>> splitter = SampleSplitter(min_samples_for_splitting=5000)
    >>> result = splitter.split(dataset)
    >>> if result['can_split']:
    ...     # Use D_res for learning GMM, D_cal for intervals
    ...     learn_resolution(result['D_res'])
    ...     compute_intervals(result['D_cal'])
    ... else:
    ...     # Small dataset: guarantees may not hold
    ...     print(result['warning'])
    """

    def __init__(
        self,
        min_samples_for_splitting: int = 5000,
        resolution_ratio: float = 0.05,
        seed: int = 42
    ):
        """
        Initialize the sample splitter.

        Args:
            min_samples_for_splitting: Minimum number of samples required
                to perform sample splitting. Below this threshold, all data
                is used for both resolution and calibration (with a warning).

            resolution_ratio: Fraction of data used for learning resolution r.
                Typical value: 0.05 (5% for resolution, 95% for calibration).
                Must be in (0, 1).

            seed: Random seed for reproducibility.

        Raises:
            ValueError: If resolution_ratio not in (0, 1)
        """
        if not (0 < resolution_ratio < 1):
            raise ValueError(
                f"resolution_ratio must be in (0, 1), got {resolution_ratio}"
            )

        self.min_samples = min_samples_for_splitting
        self.resolution_ratio = resolution_ratio
        self.seed = seed

    def split(
        self,
        dataset: torch.utils.data.Dataset,
        enforce_splitting: bool = False
    ) -> Dict[str, Any]:
        """
        Split dataset into D_resolution and D_calibration.

        Logic:
        ------
        1. If len(dataset) >= min_samples: Perform sample splitting
           - D_res gets resolution_ratio fraction (e.g., 5%)
           - D_cal gets remaining (e.g., 95%)
           - Theoretical guarantees hold ✓

        2. If len(dataset) < min_samples and enforce_splitting=False:
           - Return entire dataset for both D_res and D_cal
           - Emit warning that guarantees may not hold
           - This is the CIFAR-10/100 scenario from the paper

        3. If len(dataset) < min_samples and enforce_splitting=True:
           - Raise ValueError (strict mode)

        Args:
            dataset: PyTorch Dataset to split
            enforce_splitting: If True, raise error instead of warning when
                sample size is insufficient.

        Returns:
            Dictionary with keys:
                - 'can_split' (bool): True if splitting was performed
                - 'D_res' (Dataset or None): Data for learning resolution r
                - 'D_cal' (Dataset): Data for computing confidence intervals
                - 'warning' (str or None): Warning message if applicable
                - 'info' (str or None): Informational message if splitting succeeded

        Raises:
            ValueError: If enforce_splitting=True and insufficient samples
        """
        n = len(dataset)

        # Check if we can do sample splitting
        can_split = (n >= self.min_samples)

        if not can_split:
            # Insufficient data for splitting
            if enforce_splitting:
                raise ValueError(
                    f"Sample splitting required but only {n} samples available "
                    f"(minimum: {self.min_samples}). Either:\n"
                    f"  1. Use more data (recommended)\n"
                    f"  2. Set enforce_splitting=False to proceed with warning\n"
                    f"  3. Use a fixed resolution (method='uniform') instead of data-dependent GMM"
                )

            # Proceed with warning
            warning_msg = (
                f"⚠️  THEORETICAL GUARANTEES MAY NOT HOLD\n"
                f"\n"
                f"Dataset size ({n} samples) is below the minimum for proper "
                f"sample splitting ({self.min_samples} samples).\n"
                f"\n"
                f"Using ALL DATA for both:\n"
                f"  - Learning resolution function r (GMM clustering)\n"
                f"  - Computing confidence intervals Ĉn\n"
                f"\n"
                f"This violates the independence assumption r ⊥ Dcal required by "
                f"Theorem 3.1, which may invalidate coverage guarantees.\n"
                f"\n"
                f"Recommendations:\n"
                f"  1. Use a larger calibration set (recommended)\n"
                f"  2. Accept that coverage may be < 1-α in practice\n"
                f"  3. Use fixed resolution (method='uniform', n_clusters=100)\n"
            )

            return {
                'can_split': False,
                'D_res': None,
                'D_cal': dataset,
                'warning': warning_msg,
                'info': None
            }

        # Perform sample splitting
        indices = list(range(n))
        random.Random(self.seed).shuffle(indices)

        n_res = int(n * self.resolution_ratio)
        res_idx = indices[:n_res]
        cal_idx = indices[n_res:]

        D_res = torch.utils.data.Subset(dataset, res_idx)
        D_cal = torch.utils.data.Subset(dataset, cal_idx)

        info_msg = (
            f"✓ Sample splitting successful:\n"
            f"  - D_resolution: {n_res:,} samples ({100*self.resolution_ratio:.1f}%)\n"
            f"  - D_calibration: {len(cal_idx):,} samples ({100*(1-self.resolution_ratio):.1f}%)\n"
            f"  - Theoretical guarantees (Theorem 3.1) should hold"
        )

        return {
            'can_split': True,
            'D_res': D_res,
            'D_cal': D_cal,
            'warning': None,
            'info': info_msg
        }


def check_independence_violation(
    resolution_data_hash: int,
    calibration_data_hash: int
) -> None:
    """
    Check if the same data was used for both resolution learning and calibration.

    This is a critical violation of Theorem 3.1's independence assumption.

    Args:
        resolution_data_hash: Hash of data used to learn resolution r
        calibration_data_hash: Hash of data used to compute Ĉn

    Raises:
        ValueError: If hashes match (same data used twice)

    Example:
    --------
    >>> # In HyperparameterSearch.learn_resolution()
    >>> self._resolution_data_hash = hash(logits_res.cpu().numpy().tobytes())
    >>>
    >>> # In HyperparameterSearch.fit()
    >>> cal_hash = hash(logits_cal.cpu().numpy().tobytes())
    >>> check_independence_violation(self._resolution_data_hash, cal_hash)
    """
    if resolution_data_hash == calibration_data_hash:
        raise ValueError(
            "❌ CRITICAL VIOLATION OF THEOREM 3.1\n"
            "\n"
            "The same data was used for:\n"
            "  1. Learning resolution function r (GMM clustering)\n"
            "  2. Computing confidence intervals Ĉn\n"
            "\n"
            "This violates the independence assumption r ⊥ Dcal.\n"
            "Coverage guarantees P{ηf,P,r(z) ∈ Ĉn(z)} ≥ 1-α DO NOT HOLD!\n"
            "\n"
            "Fix: Use SampleSplitter to create independent D_res and D_cal sets."
        )


def emit_splitting_warning(splitting_info: Dict[str, Any]) -> None:
    """
    Emit appropriate warnings based on splitting result.

    Args:
        splitting_info: Result dictionary from SampleSplitter.split()
    """
    if splitting_info.get('warning'):
        warnings.warn(
            splitting_info['warning'],
            UserWarning,
            stacklevel=2
        )
    elif splitting_info.get('info'):
        print(splitting_info['info'])


# Convenience function for common use case
def split_for_partition_detector(
    dataset: torch.utils.data.Dataset,
    min_samples: int = 5000,
    resolution_ratio: float = 0.05,
    seed: int = 42,
    enforce: bool = False
) -> Dict[str, Any]:
    """
    Convenience wrapper for splitting data for MegaPartitionDetector.

    Args:
        dataset: Dataset to split
        min_samples: Minimum samples for splitting (default: 5000)
        resolution_ratio: Fraction for resolution (default: 0.05 = 5%)
        seed: Random seed (default: 42)
        enforce: Raise error if insufficient data (default: False)

    Returns:
        Splitting result dictionary

    Example:
    --------
    >>> result = split_for_partition_detector(train_dataset)
    >>> if result['can_split']:
    ...     # Learn GMM on D_res, compute intervals on D_cal
    ...     pass
    ... else:
    ...     # Proceed with caution (guarantees may not hold)
    ...     pass
    """
    splitter = SampleSplitter(
        min_samples_for_splitting=min_samples,
        resolution_ratio=resolution_ratio,
        seed=seed
    )
    result = splitter.split(dataset, enforce_splitting=enforce)
    emit_splitting_warning(result)
    return result
