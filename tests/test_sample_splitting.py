"""
Tests for Sample Splitting Module

These tests verify that sample splitting correctly enforces the independence
assumption r ⊥ Dcal required by Theorem 3.1.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset

# Add parent directory to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from code.utils.sample_splitting import (
    SampleSplitter,
    check_independence_violation,
    split_for_partition_detector
)


class TestSampleSplitter:
    """Test suite for SampleSplitter class"""

    def test_large_dataset_splitting(self):
        """Test that large datasets are properly split"""
        # Create a dataset with 10k samples (like ImageNet after 50/50 split)
        data = torch.randn(10000, 100)
        labels = torch.randint(0, 10, (10000,))
        dataset = TensorDataset(data, labels)

        splitter = SampleSplitter(min_samples_for_splitting=5000, resolution_ratio=0.05)
        result = splitter.split(dataset, enforce_splitting=False)

        # Should be able to split
        assert result['can_split'] is True
        assert result['D_res'] is not None
        assert result['D_cal'] is not None
        assert result['warning'] is None
        assert result['info'] is not None

        # Check sizes
        expected_res_size = int(10000 * 0.05)  # 500
        expected_cal_size = 10000 - expected_res_size  # 9500

        assert len(result['D_res']) == expected_res_size
        assert len(result['D_cal']) == expected_cal_size

    def test_small_dataset_warning(self):
        """Test that small datasets trigger a warning"""
        # Create a dataset with 2k samples (like CIFAR-10 after 50/50 split)
        data = torch.randn(2000, 100)
        labels = torch.randint(0, 10, (2000,))
        dataset = TensorDataset(data, labels)

        splitter = SampleSplitter(min_samples_for_splitting=5000)

        # Should NOT be able to split
        result = splitter.split(dataset, enforce_splitting=False)

        assert result['can_split'] is False
        assert result['D_res'] is None
        assert result['D_cal'] is dataset  # Returns entire dataset
        assert result['warning'] is not None
        assert 'THEORETICAL GUARANTEES MAY NOT HOLD' in result['warning']

    def test_enforce_splitting_raises_error(self):
        """Test that enforce_splitting=True raises error for small datasets"""
        data = torch.randn(2000, 100)
        labels = torch.randint(0, 10, (2000,))
        dataset = TensorDataset(data, labels)

        splitter = SampleSplitter(min_samples_for_splitting=5000)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Sample splitting required"):
            splitter.split(dataset, enforce_splitting=True)

    def test_resolution_ratio_validation(self):
        """Test that resolution_ratio must be in (0, 1)"""
        with pytest.raises(ValueError):
            SampleSplitter(resolution_ratio=0.0)  # Must be > 0

        with pytest.raises(ValueError):
            SampleSplitter(resolution_ratio=1.0)  # Must be < 1

        with pytest.raises(ValueError):
            SampleSplitter(resolution_ratio=1.5)  # Must be < 1

    def test_reproducibility(self):
        """Test that same seed gives same split"""
        data = torch.randn(10000, 100)
        labels = torch.randint(0, 10, (10000,))
        dataset = TensorDataset(data, labels)

        splitter1 = SampleSplitter(seed=42)
        result1 = splitter1.split(dataset)

        splitter2 = SampleSplitter(seed=42)
        result2 = splitter2.split(dataset)

        # Should get same indices
        indices1 = result1['D_res'].indices
        indices2 = result2['D_res'].indices

        assert indices1 == indices2

    def test_disjoint_splits(self):
        """Test that D_res and D_cal are disjoint"""
        data = torch.randn(10000, 100)
        labels = torch.randint(0, 10, (10000,))
        dataset = TensorDataset(data, labels)

        splitter = SampleSplitter()
        result = splitter.split(dataset)

        res_indices = set(result['D_res'].indices)
        cal_indices = set(result['D_cal'].indices)

        # Should have no overlap
        assert len(res_indices & cal_indices) == 0

        # Should cover all indices
        assert len(res_indices | cal_indices) == len(dataset)


class TestIndependenceCheck:
    """Test suite for independence violation detection"""

    def test_same_data_raises_error(self):
        """Test that using same data for r and Dcal raises error"""
        data = torch.randn(1000, 10)
        hash1 = hash(data.numpy().tobytes())
        hash2 = hash(data.numpy().tobytes())  # Same data

        with pytest.raises(ValueError, match="CRITICAL VIOLATION"):
            check_independence_violation(hash1, hash2)

    def test_different_data_passes(self):
        """Test that different data passes check"""
        data1 = torch.randn(1000, 10)
        data2 = torch.randn(1000, 10)  # Different data

        hash1 = hash(data1.numpy().tobytes())
        hash2 = hash(data2.numpy().tobytes())

        # Should not raise
        check_independence_violation(hash1, hash2)


class TestConvenienceFunction:
    """Test suite for split_for_partition_detector convenience function"""

    def test_imagenet_like_dataset(self):
        """Test with ImageNet-like dataset (50k samples after split)"""
        data = torch.randn(25000, 100)  # 50k after train/test split
        labels = torch.randint(0, 1000, (25000,))
        dataset = TensorDataset(data, labels)

        result = split_for_partition_detector(
            dataset,
            min_samples=5000,
            resolution_ratio=0.05
        )

        # Should split successfully
        assert result['can_split'] is True
        assert len(result['D_res']) == int(25000 * 0.05)  # 1250
        assert len(result['D_cal']) == 25000 - int(25000 * 0.05)  # 23750

    def test_cifar10_like_dataset(self):
        """Test with CIFAR-10-like dataset (5k samples after split)"""
        data = torch.randn(5000, 100)
        labels = torch.randint(0, 10, (5000,))
        dataset = TensorDataset(data, labels)

        # Capture warnings
        with pytest.warns(UserWarning, match="THEORETICAL GUARANTEES"):
            result = split_for_partition_detector(
                dataset,
                min_samples=5000,
                resolution_ratio=0.05,
                enforce=False
            )

        # Should not split (exactly at threshold)
        assert result['can_split'] is False
        assert result['D_res'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
