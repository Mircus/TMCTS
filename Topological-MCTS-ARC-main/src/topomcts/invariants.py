
import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh

@dataclass
class GameInvariants:
    beta0: int  # number of connected components
    beta1: int  # cyclomatic number approximation
    symmetry: float  # simple symmetry score proxy
    spectral_radius: float
    branching_factor: float  # average degree
    clustering_coeff: float  # local clustering coefficient

def compute_invariants(G: nx.Graph) -> GameInvariants:
    """Compute comprehensive game invariants for transfer learning."""
    # β0 via connected components
    beta0 = nx.number_connected_components(G)
    
    # β1 ~ cycles = m - n + c (cyclomatic number)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    beta1 = m - n + beta0
    
    # symmetry proxy: ratio of distinct degrees
    degs = [d for _, d in G.degree()]
    symmetry = 1.0 / (1 + len(set(degs))) if degs else 0.0
    
    # spectral radius of adjacency
    A = nx.to_numpy_array(G)
    if A.size > 0:
        if n > 25:
            try:
                w, _ = eigsh(A, k=min(1, n-1), which='LM')
                spectral_radius = float(max(abs(w)))
            except:
                w = np.linalg.eigvals(A)
                spectral_radius = float(max(abs(w)))
        else:
            w = np.linalg.eigvals(A)
            spectral_radius = float(max(abs(w)))
    else:
        spectral_radius = 0.0
    
    # branching factor (average degree)
    branching_factor = float(np.mean(degs)) if degs else 0.0
    
    # clustering coefficient
    clustering_coeff = float(nx.average_clustering(G)) if n > 0 else 0.0
    
    return GameInvariants(
        beta0=beta0, 
        beta1=int(beta1), 
        symmetry=float(symmetry), 
        spectral_radius=float(spectral_radius),
        branching_factor=float(branching_factor),
        clustering_coeff=float(clustering_coeff)
    )

def signature_dict(inv: GameInvariants) -> Dict[str, Any]:
    """Convert invariants to dictionary for serialization."""
    return {
        'beta0': inv.beta0, 
        'beta1': inv.beta1, 
        'symmetry': inv.symmetry, 
        'spectral_radius': inv.spectral_radius,
        'branching_factor': inv.branching_factor,
        'clustering_coeff': inv.clustering_coeff
    }

def compute_topological_distance(inv1: GameInvariants, inv2: GameInvariants) -> float:
    """Compute normalized L2 distance between invariant signatures."""
    # Normalize features to [0,1] range for fair comparison
    features1 = np.array([
        inv1.beta0 / 10.0,  # normalize beta0
        inv1.beta1 / 20.0,  # normalize beta1
        inv1.symmetry,      # already in [0,1]
        inv1.spectral_radius / 5.0,  # normalize spectral radius
        inv1.branching_factor / 8.0,  # normalize branching factor
        inv1.clustering_coeff  # already in [0,1]
    ])
    
    features2 = np.array([
        inv2.beta0 / 10.0,
        inv2.beta1 / 20.0,
        inv2.symmetry,
        inv2.spectral_radius / 5.0,
        inv2.branching_factor / 8.0,
        inv2.clustering_coeff
    ])
    
    # Compute L2 distance
    distance = float(np.linalg.norm(features1 - features2))
    return distance

def compute_pairwise_distances(invariants_list: List[GameInvariants]) -> np.ndarray:
    """Compute pairwise distances between all invariant signatures."""
    n = len(invariants_list)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = compute_topological_distance(invariants_list[i], invariants_list[j])
            distances[i, j] = dist
            distances[j, i] = dist  # symmetric
    
    return distances

def find_most_similar_task(target_inv: GameInvariants, 
                          candidate_invs: List[GameInvariants], 
                          exclude_indices: List[int] = None) -> Tuple[int, float]:
    """Find the most similar task to the target based on topological distance."""
    if exclude_indices is None:
        exclude_indices = []
    
    best_idx = -1
    best_distance = float('inf')
    
    for i, inv in enumerate(candidate_invs):
        if i in exclude_indices:
            continue
        
        distance = compute_topological_distance(target_inv, inv)
        if distance < best_distance:
            best_distance = distance
            best_idx = i
    
    return best_idx, best_distance
