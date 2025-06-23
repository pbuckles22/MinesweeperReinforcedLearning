"""
Unit tests for device detection and hyperparameter optimization in train_agent.py

This module tests the device detection, performance testing, and hyperparameter
optimization functions that are critical for cross-platform training.
"""

import pytest
import torch
import sys
from unittest.mock import patch, MagicMock
from src.core.train_agent import (
    detect_optimal_device,
    get_optimal_hyperparameters,
    test_device_performance
)


class TestDeviceDetection:
    """Test device detection functionality."""
    
    def test_detect_optimal_device_mps_available(self):
        """Test MPS device detection when available."""
        with patch('torch.backends.mps.is_available', return_value=True), \
             patch('torch.backends.mps.is_built', return_value=True):
            
            device_info = detect_optimal_device()
            
            assert device_info['device'] == 'mps'
            assert 'Apple M1 GPU' in device_info['description']
            assert '2-4x faster' in device_info['performance_notes']
    
    def test_detect_optimal_device_cuda_available(self):
        """Test CUDA device detection when available."""
        with patch('torch.backends.mps.is_available', return_value=False), \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_name', return_value='RTX 4090'):
            
            device_info = detect_optimal_device()
            
            assert device_info['device'] == 'cuda'
            assert 'NVIDIA GPU' in device_info['description']
            assert 'RTX 4090' in device_info['description']
            assert 'Fastest option' in device_info['performance_notes']
    
    def test_detect_optimal_device_cpu_fallback(self):
        """Test CPU fallback when no GPU is available."""
        with patch('torch.backends.mps.is_available', return_value=False), \
             patch('torch.cuda.is_available', return_value=False):
            
            device_info = detect_optimal_device()
            
            assert device_info['device'] == 'cpu'
            assert device_info['description'] == 'CPU (fallback)'
            assert 'Slowest option' in device_info['performance_notes']
    
    def test_detect_optimal_device_mps_not_built(self):
        """Test that MPS is not used when not built."""
        with patch('torch.backends.mps.is_available', return_value=True), \
             patch('torch.backends.mps.is_built', return_value=False), \
             patch('torch.cuda.is_available', return_value=False):
            
            device_info = detect_optimal_device()
            
            assert device_info['device'] == 'cpu'
            assert device_info['description'] == 'CPU (fallback)'


class TestHyperparameterOptimization:
    """Test hyperparameter optimization for different devices."""
    
    def test_get_optimal_hyperparameters_mps(self):
        """Test MPS hyperparameter optimization."""
        device_info = {
            'device': 'mps',
            'description': 'Apple M1 GPU (MPS)',
            'performance_notes': '2-4x faster than CPU'
        }
        
        params = get_optimal_hyperparameters(device_info)
        
        assert params['batch_size'] == 128
        assert params['n_steps'] == 2048
        assert params['n_epochs'] == 12
        assert params['learning_rate'] == 3e-4
    
    def test_get_optimal_hyperparameters_cuda(self):
        """Test CUDA hyperparameter optimization."""
        device_info = {
            'device': 'cuda',
            'description': 'NVIDIA GPU (RTX 4090)',
            'performance_notes': 'Fastest option for NVIDIA GPUs'
        }
        
        params = get_optimal_hyperparameters(device_info)
        
        assert params['batch_size'] == 256
        assert params['n_steps'] == 2048
        assert params['n_epochs'] == 10
        assert params['learning_rate'] == 3e-4
    
    def test_get_optimal_hyperparameters_cpu(self):
        """Test CPU hyperparameter optimization."""
        device_info = {
            'device': 'cpu',
            'description': 'CPU (fallback)',
            'performance_notes': 'Slowest option, suitable for testing only'
        }
        
        params = get_optimal_hyperparameters(device_info)
        
        assert params['batch_size'] == 32
        assert params['n_steps'] == 1024
        assert params['n_epochs'] == 8
        assert params['learning_rate'] == 3e-4
    
    def test_get_optimal_hyperparameters_base_values(self):
        """Test that base hyperparameters are consistent across devices."""
        device_info = {
            'device': 'cpu',
            'description': 'CPU (fallback)',
            'performance_notes': 'Slowest option'
        }
        
        params = get_optimal_hyperparameters(device_info)
        
        # Test base values that should be consistent
        assert params['gamma'] == 0.99
        assert params['gae_lambda'] == 0.95
        assert params['clip_range'] == 0.2
        assert params['ent_coef'] == 0.01
        assert params['vf_coef'] == 0.5
        assert params['max_grad_norm'] == 0.5


class TestDevicePerformance:
    """Test device performance benchmarking."""
    
    @patch('torch.device')
    @patch('torch.randn')
    @patch('torch.mm')
    @patch('time.time')
    def test_device_performance_benchmark_success(self, mock_time, mock_mm, mock_randn, mock_device):
        """Test successful device performance testing."""
        # Mock time.time to return increasing values
        mock_time.side_effect = [0.0, 1.0]  # start_time, end_time
        
        # Mock tensor operations
        mock_tensor = MagicMock()
        mock_randn.return_value = mock_tensor
        mock_mm.return_value = mock_tensor
        
        device_info = {
            'device': 'mps',
            'description': 'Apple M1 GPU (MPS)',
            'performance_notes': '2-4x faster than CPU'
        }
        
        # Should not raise any exceptions
        result = test_device_performance(device_info)
        
        assert result == 0.1  # (1.0 - 0.0) / 10
        assert mock_device.called
        assert mock_randn.called
        assert mock_mm.called
    
    @patch('torch.device')
    @patch('torch.randn')
    def test_device_performance_benchmark_exception_handling(self, mock_randn, mock_device):
        """Test that device performance testing handles exceptions gracefully."""
        # Mock torch.randn to raise an exception
        mock_randn.side_effect = RuntimeError("CUDA out of memory")
        
        device_info = {
            'device': 'cuda',
            'description': 'NVIDIA GPU (RTX 4090)',
            'performance_notes': 'Fastest option'
        }
        
        # Should handle the exception gracefully
        with pytest.raises(RuntimeError):
            test_device_performance(device_info)
    
    @patch('torch.device')
    @patch('torch.randn')
    @patch('torch.mm')
    @patch('time.time')
    def test_device_performance_benchmark_mps_performance(self, mock_time, mock_mm, mock_randn, mock_device):
        """Test MPS performance benchmarking with different performance levels."""
        mock_tensor = MagicMock()
        mock_randn.return_value = mock_tensor
        mock_mm.return_value = mock_tensor
        
        # Test excellent performance
        mock_time.side_effect = [0.0, 0.5]  # 0.05s per operation
        
        device_info = {
            'device': 'mps',
            'description': 'Apple M1 GPU (MPS)',
            'performance_notes': '2-4x faster than CPU'
        }
        
        result = test_device_performance(device_info)
        assert result == 0.05
    
    @patch('torch.device')
    @patch('torch.randn')
    @patch('torch.mm')
    @patch('time.time')
    def test_device_performance_benchmark_cuda_performance(self, mock_time, mock_mm, mock_randn, mock_device):
        """Test CUDA performance benchmarking."""
        mock_tensor = MagicMock()
        mock_randn.return_value = mock_tensor
        mock_mm.return_value = mock_tensor
        
        # Test good performance
        mock_time.side_effect = [0.0, 0.8]  # 0.08s per operation
        
        device_info = {
            'device': 'cuda',
            'description': 'NVIDIA GPU (RTX 4090)',
            'performance_notes': 'Fastest option'
        }
        
        result = test_device_performance(device_info)
        assert result == 0.08


class TestDeviceDetectionIntegration:
    """Integration tests for device detection workflow."""
    
    def test_device_detection_workflow_mps(self):
        """Test complete device detection workflow for MPS."""
        with patch('torch.backends.mps.is_available', return_value=True), \
             patch('torch.backends.mps.is_built', return_value=True):
            
            # Detect device
            device_info = detect_optimal_device()
            assert device_info['device'] == 'mps'
            
            # Get hyperparameters
            params = get_optimal_hyperparameters(device_info)
            assert params['batch_size'] == 128
            assert params['n_epochs'] == 12
    
    def test_device_detection_workflow_cuda(self):
        """Test complete device detection workflow for CUDA."""
        with patch('torch.backends.mps.is_available', return_value=False), \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_name', return_value='RTX 4090'):
            
            # Detect device
            device_info = detect_optimal_device()
            assert device_info['device'] == 'cuda'
            
            # Get hyperparameters
            params = get_optimal_hyperparameters(device_info)
            assert params['batch_size'] == 256
            assert params['n_epochs'] == 10
    
    def test_device_detection_workflow_cpu(self):
        """Test complete device detection workflow for CPU."""
        with patch('torch.backends.mps.is_available', return_value=False), \
             patch('torch.cuda.is_available', return_value=False):
            
            # Detect device
            device_info = detect_optimal_device()
            assert device_info['device'] == 'cpu'
            
            # Get hyperparameters
            params = get_optimal_hyperparameters(device_info)
            assert params['batch_size'] == 32
            assert params['n_epochs'] == 8


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_detect_optimal_device_with_none_values(self):
        """Test device detection with None values from torch."""
        with patch('torch.backends.mps.is_available', return_value=None), \
             patch('torch.cuda.is_available', return_value=None):
            
            device_info = detect_optimal_device()
            assert device_info['device'] == 'cpu'
    
    def test_get_optimal_hyperparameters_invalid_device(self):
        """Test hyperparameter optimization with invalid device."""
        device_info = {
            'device': 'invalid_device',
            'description': 'Invalid Device',
            'performance_notes': 'Unknown performance'
        }
        
        # Should fall back to CPU parameters
        params = get_optimal_hyperparameters(device_info)
        assert params['batch_size'] == 32
        assert params['n_steps'] == 1024
    
    @patch('torch.device')
    def test_device_performance_benchmark_invalid_device(self, mock_device):
        """Test performance testing with invalid device."""
        mock_device.side_effect = RuntimeError("Invalid device")
        
        device_info = {
            'device': 'invalid_device',
            'description': 'Invalid Device',
            'performance_notes': 'Unknown performance'
        }
        
        with pytest.raises(RuntimeError):
            test_device_performance(device_info) 