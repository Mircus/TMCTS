
import numpy as np
import networkx as nx
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SimplicialComplex:
    """
    Enhanced graph-based complex with centrality, diffusion, and symmetry features.
    Built from grid adjacency of missing/filled cells.
    """
    G: nx.Graph

    @staticmethod
    def from_grid(grid: np.ndarray) -> "SimplicialComplex":
        """Build adjacency graph from grid with 4-neighborhood connectivity."""
        n, m = grid.shape
        G = nx.Graph()
        
        # Add nodes with values
        for i in range(n):
            for j in range(m):
                G.add_node((i, j), value=int(grid[i, j]))
        
        # 4-neighborhood connectivity
        for i in range(n):
            for j in range(m):
                for di, dj in [(1, 0), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < m:
                        G.add_edge((i, j), (ni, nj))
        
        return SimplicialComplex(G)

    def laplacian(self) -> np.ndarray:
        """Compute normalized Laplacian matrix."""
        A = nx.to_numpy_array(self.G, nodelist=list(self.G.nodes()))
        L = csgraph.laplacian(A, normed=True)
        return L

    def fiedler_vector(self) -> np.ndarray:
        """Compute Fiedler vector (second smallest eigenvector)."""
        L = self.laplacian()
        n = L.shape[0]
        
        # Handle empty graph
        if n == 0:
            return np.array([])
        
        # Use sparse solver for larger graphs
        if n > 25:
            try:
                w, v = eigsh(L, k=min(2, n-1), which='SM', sigma=0)
                if len(w) > 1:
                    return v[:, 1]
                return v[:, 0]
            except:
                # Fallback to dense
                w, v = np.linalg.eigh(L)
                if len(w) > 1:
                    return v[:, 1]
                return v[:, 0]
        else:
            w, v = np.linalg.eigh(L)
            if len(w) > 1:
                return v[:, 1]
            return v[:, 0]

    def centrality_score(self) -> float:
        """Centrality proxy: variance of Fiedler vector."""
        f = self.fiedler_vector()
        if len(f) == 0:
            return 0.0
        var_val = np.var(f)
        return 0.0 if np.isnan(var_val) else float(var_val)

    def diffusion_score(self) -> float:
        """Diffusion proxy: based on graph connectivity and spectral properties."""
        if self.G.number_of_nodes() == 0:
            return 0.0
        
        # Compute algebraic connectivity (second smallest eigenvalue)
        L = self.laplacian()
        n = L.shape[0]
        
        if n <= 1:
            return 0.0
        
        try:
            if n > 25:
                w, _ = eigsh(L, k=min(2, n-1), which='SM', sigma=0)
            else:
                w, _ = np.linalg.eigh(L)
            
            # Algebraic connectivity is second smallest eigenvalue
            algebraic_connectivity = w[1] if len(w) > 1 else w[0]
            return float(algebraic_connectivity)
        except:
            return 0.0

    def symmetry_score(self) -> float:
        """Symmetry proxy: based on graph automorphism properties."""
        if self.G.number_of_nodes() == 0:
            return 0.0
        
        # Simple symmetry measure: degree distribution uniformity
        degrees = [d for _, d in self.G.degree()]
        if not degrees:
            return 0.0
        
        # Compute coefficient of variation of degrees
        mean_deg = np.mean(degrees)
        if mean_deg == 0:
            return 0.0
        
        std_deg = np.std(degrees)
        cv = std_deg / mean_deg
        
        # Convert to symmetry score (lower CV = higher symmetry)
        return float(1.0 / (1.0 + cv))

    def topological_features(self) -> Tuple[float, float, float]:
        """Return all three topological features: centrality, diffusion, symmetry."""
        return (
            self.centrality_score(),
            self.diffusion_score(),
            self.symmetry_score()
        )
