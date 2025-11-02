"""
Constraint graph construction for solution space topology analysis.

Builds a graph where nodes are (cell, color) assignments and edges represent
compatible assignments under the detected pattern rule. The topology of this
constraint graph encodes solution space structure and difficulty.
"""

from typing import Dict, List, Tuple, Set
import numpy as np
import networkx as nx
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

from .game import ARCGame
from .pattern_detector import PatternDetector


class ConstraintGraph:
    """Graph representation of valid color assignments (solution space topology)."""

    def __init__(self, state: ARCGame, rule: str):
        """
        Initialize constraint graph for a game state.

        Args:
            state: Game state with partial grid
            rule: Pattern rule (detected or provided)
                 e.g., "rotational_symmetry_180", "color_frequency"
        """
        self.state = state
        self.rule = rule
        self.detector = PatternDetector()
        self.graph = nx.Graph()
        self._build_constraint_graph()

    def _build_constraint_graph(self):
        """Build graph of compatible (cell, color) assignments."""
        missing_cells = self._get_missing_cells()
        alphabet = self.state.alphabet

        # Add nodes: (cell, color) tuples
        for cell in missing_cells:
            for color in alphabet:
                self.graph.add_node((cell, color))

        # Add edges: compatible assignments
        if self.rule.startswith("rotational_symmetry"):
            self._add_symmetry_edges(missing_cells, self.rule)
        elif self.rule.startswith("reflective_symmetry"):
            self._add_symmetry_edges(missing_cells, self.rule)
        elif self.rule == "color_frequency":
            self._add_frequency_edges(missing_cells)
        elif self.rule == "arithmetic_progression":
            self._add_progression_edges(missing_cells)
        else:  # spatial_pattern
            self._add_spatial_edges(missing_cells)

    def _get_missing_cells(self) -> List[Tuple[int, int]]:
        """Get coordinates of missing cells."""
        grid = self.state.initial
        missing = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == -1:
                    missing.append((r, c))
        return missing

    def _add_symmetry_edges(self, missing_cells: List, rule: str):
        """Add edges for symmetric assignments."""
        # Get symmetric pairs from detector
        pairs = self.detector.get_symmetry_pairs(self.state, rule)

        # For each symmetric pair, both cells should have same color
        for cell1, cell2 in pairs:
            if cell1 in missing_cells and cell2 in missing_cells:
                for color in self.state.alphabet:
                    # (cell1, color) and (cell2, color) are compatible
                    # (they satisfy symmetry if both cells get same color)
                    if self.graph.has_node((cell1, color)) and self.graph.has_node((cell2, color)):
                        self.graph.add_edge((cell1, color), (cell2, color))

        # Also add edges within same cell for all valid colors
        # (different colors for same cell are all individually valid)
        for cell in missing_cells:
            colors_for_cell = [color for color in self.state.alphabet
                              if (cell, color) in self.graph.nodes()]
            # Connect all colors for this cell to each other
            for i, c1 in enumerate(colors_for_cell):
                for c2 in colors_for_cell[i+1:]:
                    self.graph.add_edge((cell, c1), (cell, c2))

    def _add_frequency_edges(self, missing_cells: List):
        """Add edges for frequency-based assignments."""
        # Count current colors in filled cells
        filled_grid = self.state.initial.copy()
        filled_colors = filled_grid[filled_grid != -1]

        # Target frequency: roughly uniform
        total_filled = len(filled_colors)
        total_cells = self.state.initial.size
        cells_to_fill = len(missing_cells)

        # Estimate how many times each color should appear
        target_freq = (total_filled + cells_to_fill) / len(self.state.alphabet)

        # Build frequency compatibility
        # Two assignments are compatible if they don't exceed target frequency
        for cell1 in missing_cells:
            for cell2 in missing_cells:
                if cell1 < cell2:  # Avoid duplicate edges
                    for color in self.state.alphabet:
                        if self.graph.has_node((cell1, color)) and self.graph.has_node((cell2, color)):
                            # These are compatible (both can be same color without exceeding frequency)
                            self.graph.add_edge((cell1, color), (cell2, color))

    def _add_progression_edges(self, missing_cells: List):
        """Add edges for progression-based assignments."""
        # For arithmetic progression, consecutive cells should follow pattern
        # This is complex without knowing the direction, so we use a simple heuristic:
        # cells should have colors that maintain some ordering

        grid = self.state.initial
        n, m = grid.shape

        # Try to find progression direction and step
        progression_step = self._detect_progression_step()

        for cell1 in missing_cells:
            for cell2 in missing_cells:
                if cell1 < cell2:
                    # Check if assigning colors maintains progression
                    for col1 in self.state.alphabet:
                        for col2 in self.state.alphabet:
                            # If progression detected, later cells should have incrementing colors
                            if progression_step is not None:
                                if col2 >= col1:  # Non-decreasing is safe
                                    if self.graph.has_node((cell1, col1)) and self.graph.has_node((cell2, col2)):
                                        self.graph.add_edge((cell1, col1), (cell2, col2))

    def _add_spatial_edges(self, missing_cells: List):
        """Add edges for spatial pattern consistency."""
        # For unknown patterns, use locality: nearby cells can have any colors
        # but we prefer cells with similar neighbors to have similar colors

        grid = self.state.initial
        n, m = grid.shape

        for cell in missing_cells:
            r, c = cell
            # Get neighbors
            neighbors = []
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < m and grid[nr, nc] != -1:
                    neighbors.append(grid[nr, nc])

            if not neighbors:
                # Isolated cell - can be any color
                for color in self.state.alphabet:
                    for other_cell in missing_cells:
                        if cell != other_cell:
                            for other_color in self.state.alphabet:
                                if self.graph.has_node((cell, color)) and self.graph.has_node((other_cell, other_color)):
                                    self.graph.add_edge((cell, color), (other_cell, other_color))

    def _detect_progression_step(self) -> float:
        """Detect arithmetic progression step size."""
        grid = self.state.initial
        filled_colors = grid[grid != -1]

        if len(filled_colors) < 2:
            return None

        diffs = np.diff(np.sort(filled_colors))
        if len(diffs) > 0 and np.allclose(diffs, diffs[0]):
            return float(diffs[0])

        return None

    def topological_features(self) -> Dict:
        """
        Compute topological features of constraint graph.

        Returns:
            Dict with keys:
            - 'num_nodes': number of (cell, color) assignments
            - 'num_edges': number of compatible assignment pairs
            - 'algebraic_connectivity': second smallest Laplacian eigenvalue
            - 'cell_importance': dict mapping cells to centrality scores
            - 'color_importance': dict mapping colors to importance
            - 'constraint_density': ratio of edges to possible edges
        """
        n = self.graph.number_of_nodes()

        if n == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "algebraic_connectivity": 0.0,
                "cell_importance": {},
                "color_importance": {},
                "constraint_density": 0.0,
            }

        num_edges = self.graph.number_of_edges()
        max_edges = n * (n - 1) / 2

        # Compute Laplacian spectral properties
        try:
            L = nx.laplacian_matrix(self.graph).toarray()
            eigenvalues = np.linalg.eigvalsh(L)
            algebraic_connectivity = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        except:
            # Handle disconnected or degenerate graphs
            algebraic_connectivity = 0.0

        # Compute cell and color importance
        cell_importance = self._compute_cell_centrality()
        color_importance = self._compute_color_centrality()

        return {
            "num_nodes": n,
            "num_edges": num_edges,
            "algebraic_connectivity": float(algebraic_connectivity),
            "cell_importance": cell_importance,
            "color_importance": color_importance,
            "constraint_density": float(num_edges / max(1, max_edges)),
        }

    def _compute_cell_centrality(self) -> Dict[Tuple, float]:
        """
        Compute how important each cell is in the solution space.

        High importance = few valid colors (bottleneck)
        Low importance = many valid colors (flexible)
        """
        missing_cells = self._get_missing_cells()
        centrality = {}

        for cell in missing_cells:
            # Count how many valid colors for this cell
            valid_colors = 0
            for color in self.state.alphabet:
                if (cell, color) in self.graph.nodes():
                    valid_colors += 1

            # Fewer options = higher importance
            if valid_colors > 0:
                centrality[cell] = 1.0 / valid_colors
            else:
                centrality[cell] = 1.0  # Impossible cell

        # Normalize to [0, 1]
        if centrality:
            max_val = max(centrality.values())
            if max_val > 0:
                centrality = {k: v / max_val for k, v in centrality.items()}

        return centrality

    def _compute_color_centrality(self) -> Dict[int, float]:
        """
        Compute how important each color is in the solution space.

        High importance = many cells can use it
        Low importance = few cells can use it
        """
        missing_cells = self._get_missing_cells()
        color_importance = {}

        for color in self.state.alphabet:
            # Count how many cells can use this color
            cells_using_color = 0
            for cell in missing_cells:
                if (cell, color) in self.graph.nodes():
                    cells_using_color += 1

            # More cells = higher importance (more used)
            color_importance[color] = cells_using_color / max(1, len(missing_cells))

        return color_importance

    def fiedler_vector(self) -> np.ndarray:
        """
        Compute Fiedler vector (second eigenvector of Laplacian).

        Higher values indicate more central positions in solution space.
        """
        # Handle empty graphs
        if self.graph.number_of_nodes() == 0:
            return np.array([])

        L = nx.laplacian_matrix(self.graph).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        if len(eigenvalues) > 1:
            return eigenvectors[:, 1]
        elif len(eigenvalues) > 0:
            return eigenvectors[:, 0]
        else:
            return np.array([])

    def is_solvable(self) -> bool:
        """Check if solution space is non-empty."""
        return self.graph.number_of_nodes() > 0

    def solution_space_size(self) -> int:
        """Estimate number of valid solutions."""
        # Simple approximation: number of connected components in constraint graph
        if self.graph.number_of_nodes() == 0:
            return 0
        return nx.number_connected_components(self.graph)

    def get_node_label(self, node_index: int) -> Tuple:
        """Get (cell, color) label for a node index."""
        return list(self.graph.nodes())[node_index]

    def get_nodes_for_cell(self, cell: Tuple) -> List[int]:
        """Get all (cell, color) nodes for a specific cell."""
        missing_cells = self._get_missing_cells()
        if cell not in missing_cells:
            return []

        nodes = []
        for i, node in enumerate(self.graph.nodes()):
            if node[0] == cell:
                nodes.append(i)
        return nodes

    def get_valid_colors_for_cell(self, cell: Tuple) -> List[int]:
        """Get valid colors for a specific cell."""
        missing_cells = self._get_missing_cells()
        if cell not in missing_cells:
            return []

        valid_colors = []
        for color in self.state.alphabet:
            if (cell, color) in self.graph.nodes():
                valid_colors.append(color)
        return valid_colors

    def degree_distribution(self) -> Dict[int, int]:
        """Get distribution of node degrees."""
        degrees = dict(self.graph.degree())
        distribution = {}
        for degree in degrees.values():
            distribution[degree] = distribution.get(degree, 0) + 1
        return distribution
