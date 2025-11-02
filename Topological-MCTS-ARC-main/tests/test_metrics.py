import numpy as np
import pytest
from topomcts.game import ARCGame
from topomcts.metrics import success, quality

def test_success_metric():
    """Test success metric on trivial cases."""
    # Perfect match
    init = np.array([[0, 1, 2], [3, 4, 5]])
    target = np.array([[0, 1, 2], [3, 4, 5]])
    game = ARCGame(init, target)
    assert success(game) == 1.0
    
    # Complete mismatch
    init = np.array([[0, 1, 2], [3, 4, 5]])
    target = np.array([[5, 4, 3], [2, 1, 0]])
    game = ARCGame(init, target)
    assert success(game) == 0.0

def test_quality_metric():
    """Test quality metric on trivial cases."""
    # Perfect match
    init = np.array([[0, 1, 2], [3, 4, 5]])
    target = np.array([[0, 1, 2], [3, 4, 5]])
    game = ARCGame(init, target)
    assert quality(game) == 1.0
    
    # Half match
    init = np.array([[0, 1, 2], [3, 4, 5]])
    target = np.array([[0, 1, 2], [0, 0, 0]])
    game = ARCGame(init, target)
    expected_quality = 3.0 / 6.0  # 3 out of 6 cells match
    assert abs(quality(game) - expected_quality) < 1e-6
    
    # No match
    init = np.array([[0, 1, 2], [3, 4, 5]])
    target = np.array([[5, 4, 3], [2, 1, 0]])
    game = ARCGame(init, target)
    assert quality(game) == 0.0

def test_quality_with_missing_cells():
    """Test quality metric with missing cells."""
    # Grid with missing cells
    init = np.array([[0, -1, 2], [3, 4, -1]])
    target = np.array([[0, 1, 2], [3, 4, 5]])
    game = ARCGame(init, target)
    
    # Quality should only consider non-missing cells
    # Cells (0,0), (0,2), (1,0), (1,1) are filled and match
    # So 4 out of 4 filled cells match = 1.0
    assert quality(game) == 1.0

def test_quality_edge_cases():
    """Test quality metric on edge cases."""
    # Single cell
    init = np.array([[0]])
    target = np.array([[0]])
    game = ARCGame(init, target)
    assert quality(game) == 1.0
    
    # Single cell mismatch
    init = np.array([[0]])
    target = np.array([[1]])
    game = ARCGame(init, target)
    assert quality(game) == 0.0
    
    # Empty grid
    init = np.array([[]])
    target = np.array([[]])
    game = ARCGame(init, target)
    assert quality(game) == 1.0  # Empty grids are considered perfect matches

def test_success_vs_quality_consistency():
    """Test that success and quality are consistent."""
    # When success is 1.0, quality should also be 1.0
    init = np.array([[0, 1, 2], [3, 4, 5]])
    target = np.array([[0, 1, 2], [3, 4, 5]])
    game = ARCGame(init, target)
    assert success(game) == 1.0
    assert quality(game) == 1.0
    
    # When success is 0.0, quality should be < 1.0
    init = np.array([[0, 1, 2], [3, 4, 5]])
    target = np.array([[5, 4, 3], [2, 1, 0]])
    game = ARCGame(init, target)
    assert success(game) == 0.0
    assert quality(game) < 1.0

def test_metrics_return_types():
    """Test that metrics return correct types."""
    init = np.array([[0, 1, 2], [3, 4, 5]])
    target = np.array([[0, 1, 2], [3, 4, 5]])
    game = ARCGame(init, target)
    
    success_val = success(game)
    quality_val = quality(game)
    
    assert isinstance(success_val, float)
    assert isinstance(quality_val, float)
    assert not np.isnan(success_val)
    assert not np.isnan(quality_val)
    assert not np.isinf(success_val)
    assert not np.isinf(quality_val)

def test_metrics_range():
    """Test that metrics are in expected ranges."""
    init = np.array([[0, 1, 2], [3, 4, 5]])
    target = np.array([[0, 1, 2], [3, 4, 5]])
    game = ARCGame(init, target)
    
    success_val = success(game)
    quality_val = quality(game)
    
    assert 0.0 <= success_val <= 1.0
    assert 0.0 <= quality_val <= 1.0

def test_quality_with_different_sizes():
    """Test quality metric with different grid sizes."""
    # 1x1 grid
    init = np.array([[0]])
    target = np.array([[0]])
    game = ARCGame(init, target)
    assert quality(game) == 1.0
    
    # 2x2 grid
    init = np.array([[0, 1], [2, 3]])
    target = np.array([[0, 1], [2, 3]])
    game = ARCGame(init, target)
    assert quality(game) == 1.0
    
    # 3x3 grid
    init = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    target = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    game = ARCGame(init, target)
    assert quality(game) == 1.0
