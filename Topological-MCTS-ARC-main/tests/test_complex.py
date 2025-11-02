
import numpy as np
import pytest
from topomcts.complex import SimplicialComplex

def test_fiedler_vector_runs():
    grid = np.array([[0,0,0],[0,-1,1],[1,1,1]])
    sc = SimplicialComplex.from_grid(grid)
    f = sc.fiedler_vector()
    assert len(f) == grid.size

def test_laplacian_shape():
    """Test that Laplacian has correct shape."""
    grid = np.array([[0,0,0],[0,-1,1],[1,1,1]])
    sc = SimplicialComplex.from_grid(grid)
    L = sc.laplacian()
    assert L.shape == (grid.size, grid.size)

def test_fiedler_vector_exists():
    """Test that Fiedler vector exists and has correct properties."""
    grid = np.array([[0,0,0],[0,-1,1],[1,1,1]])
    sc = SimplicialComplex.from_grid(grid)
    f = sc.fiedler_vector()
    assert len(f) == grid.size
    assert not np.all(np.isnan(f))
    assert not np.all(np.isinf(f))

def test_centrality_score():
    """Test centrality score computation."""
    grid = np.array([[0,0,0],[0,-1,1],[1,1,1]])
    sc = SimplicialComplex.from_grid(grid)
    centrality = sc.centrality_score()
    assert isinstance(centrality, float)
    assert centrality >= 0

def test_diffusion_score():
    """Test diffusion score computation."""
    grid = np.array([[0,0,0],[0,-1,1],[1,1,1]])
    sc = SimplicialComplex.from_grid(grid)
    diffusion = sc.diffusion_score()
    assert isinstance(diffusion, float)
    assert diffusion >= 0

def test_symmetry_score():
    """Test symmetry score computation."""
    grid = np.array([[0,0,0],[0,-1,1],[1,1,1]])
    sc = SimplicialComplex.from_grid(grid)
    symmetry = sc.symmetry_score()
    assert isinstance(symmetry, float)
    assert 0 <= symmetry <= 1

def test_topological_features():
    """Test combined topological features."""
    grid = np.array([[0,0,0],[0,-1,1],[1,1,1]])
    sc = SimplicialComplex.from_grid(grid)
    features = sc.topological_features()
    assert len(features) == 3
    assert all(isinstance(f, float) for f in features)

def test_variance_asymmetric_toy():
    """Test that variance > 0 on asymmetric toy example."""
    # Create asymmetric grid
    grid = np.array([[0,0,0],[1,1,1],[2,2,2]])
    sc = SimplicialComplex.from_grid(grid)
    centrality = sc.centrality_score()
    assert centrality > 0, "Centrality should be positive for asymmetric grid"

def test_empty_grid():
    """Test behavior on empty grid."""
    grid = np.array([[]])
    sc = SimplicialComplex.from_grid(grid)
    features = sc.topological_features()
    assert all(f == 0.0 for f in features)

def test_single_cell():
    """Test behavior on single cell grid."""
    grid = np.array([[0]])
    sc = SimplicialComplex.from_grid(grid)
    features = sc.topological_features()
    assert all(isinstance(f, float) for f in features)
