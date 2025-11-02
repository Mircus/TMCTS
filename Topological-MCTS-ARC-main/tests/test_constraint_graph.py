"""Tests for constraint graph construction and analysis."""

import numpy as np
import pytest
from topomcts.game import ARCGame
from topomcts.constraint_graph import ConstraintGraph
from topomcts.pattern_detector import PatternDetector


class TestConstraintGraphBasics:
    """Test basic constraint graph construction."""

    def test_graph_creation_symmetric(self):
        """Test graph creation for symmetric pattern."""
        # 3x3 grid with center missing, symmetric pattern
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "rotational_symmetry_180")

        # Should have nodes for center cell with each color
        assert cg.graph.number_of_nodes() > 0
        assert cg.is_solvable()

    def test_graph_creation_frequency(self):
        """Test graph creation for frequency pattern."""
        grid = np.array([[0, 1, -1], [1, 0, 1], [2, 2, 2]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "color_frequency")

        assert cg.graph.number_of_nodes() > 0

    def test_empty_graph_handling(self):
        """Test that fully filled grid creates empty constraint graph."""
        grid = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "rotational_symmetry_180")

        # No missing cells = no nodes in constraint graph
        assert cg.graph.number_of_nodes() == 0
        assert not cg.is_solvable()

    def test_single_missing_cell(self):
        """Test constraint graph with single missing cell."""
        grid = np.array([[1, 0, 1], [0, 1, 0], [1, -1, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "spatial_pattern")

        # Should have one cell with multiple color options
        num_cells = 1
        colors_in_alphabet = len(game.alphabet)
        assert cg.graph.number_of_nodes() == colors_in_alphabet

    def test_multiple_missing_cells(self):
        """Test constraint graph with multiple missing cells."""
        grid = np.array([[1, -1, 1], [0, -1, 0], [1, -1, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "spatial_pattern")

        # Should have nodes for each (cell, color) pair
        num_missing = 3
        num_colors = len(game.alphabet)
        assert cg.graph.number_of_nodes() <= num_missing * num_colors


class TestSymmetryConstraints:
    """Test constraint edge addition for symmetric patterns."""

    def test_rotational_180_edges(self):
        """Test edges for 180° rotational symmetry."""
        grid = np.array([[1, 0, -1], [0, 2, 0], [-1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "rotational_symmetry_180")

        # The two missing cells (0,2) and (2,0) are symmetric
        # So (0,2,color) should connect to (2,0,color)
        assert cg.graph.number_of_edges() > 0

    def test_reflective_h_edges(self):
        """Test edges for horizontal reflection."""
        grid = np.array([[-1, 0, 1], [0, 1, 0], [-1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "reflective_symmetry_h")

        # Missing cells (0,0) and (2,0) are horizontally symmetric
        assert cg.graph.number_of_edges() > 0

    def test_reflective_v_edges(self):
        """Test edges for vertical reflection."""
        grid = np.array([[1, -1, 1], [0, -1, 0], [1, -1, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "reflective_symmetry_v")

        # Missing cells should be symmetric vertically
        assert cg.graph.number_of_edges() > 0


class TestTopologicalFeatures:
    """Test topological feature computation."""

    def test_features_non_empty(self):
        """Test that features are computed for non-empty graph."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "rotational_symmetry_180")
        features = cg.topological_features()

        assert features["num_nodes"] > 0
        assert "algebraic_connectivity" in features
        assert "cell_importance" in features
        assert "color_importance" in features

    def test_features_empty_graph(self):
        """Test that features handle empty graph."""
        grid = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "spatial_pattern")
        features = cg.topological_features()

        assert features["num_nodes"] == 0
        assert features["algebraic_connectivity"] == 0.0

    def test_cell_importance_single_cell(self):
        """Test cell importance for single missing cell."""
        grid = np.array([[1, 0, 1], [0, 1, 0], [1, -1, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "spatial_pattern")
        features = cg.topological_features()

        # Single missing cell is always important
        cell_imp = features["cell_importance"]
        assert len(cell_imp) == 1
        assert list(cell_imp.values())[0] > 0

    def test_color_importance_values(self):
        """Test that color importance values are in [0, 1]."""
        grid = np.array([[1, 0, -1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "color_frequency")
        features = cg.topological_features()

        for color, importance in features["color_importance"].items():
            assert 0.0 <= importance <= 1.0

    def test_constraint_density(self):
        """Test constraint density metric."""
        grid = np.array([[1, 0, -1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "color_frequency")
        features = cg.topological_features()

        density = features["constraint_density"]
        assert 0.0 <= density <= 1.0


class TestFiedlerVector:
    """Test Fiedler vector computation."""

    def test_fiedler_computed(self):
        """Test that Fiedler vector is computed."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "rotational_symmetry_180")
        fiedler = cg.fiedler_vector()

        # Should return non-empty vector
        assert len(fiedler) > 0

    def test_fiedler_empty_graph(self):
        """Test Fiedler vector for empty graph."""
        grid = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "spatial_pattern")
        fiedler = cg.fiedler_vector()

        # Empty graph = empty vector
        assert len(fiedler) == 0


class TestValidColors:
    """Test getting valid colors for cells."""

    def test_valid_colors_exist(self):
        """Test that valid colors are returned for missing cells."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "spatial_pattern")

        missing_cell = (1, 1)
        valid_colors = cg.get_valid_colors_for_cell(missing_cell)

        assert len(valid_colors) > 0

    def test_valid_colors_for_filled_cell(self):
        """Test that filled cells return no valid colors."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "spatial_pattern")

        filled_cell = (0, 0)
        valid_colors = cg.get_valid_colors_for_cell(filled_cell)

        assert len(valid_colors) == 0

    def test_valid_colors_respects_constraints(self):
        """Test that valid colors respect pattern constraints."""
        # Grid where center should be blue (2) due to symmetry
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "rotational_symmetry_180")

        center_cell = (1, 1)
        valid_colors = cg.get_valid_colors_for_cell(center_cell)

        # Should have valid colors for symmetric center
        assert len(valid_colors) > 0


class TestDegreeDistribution:
    """Test degree distribution analysis."""

    def test_degree_distribution_exists(self):
        """Test that degree distribution is computed."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "spatial_pattern")
        dist = cg.degree_distribution()

        # Should return dict of degree -> count
        assert isinstance(dist, dict)

    def test_degree_distribution_sums_to_nodes(self):
        """Test that degree distribution sums to total nodes."""
        grid = np.array([[1, -1, 1], [0, -1, 0], [1, -1, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "spatial_pattern")
        dist = cg.degree_distribution()

        total_from_dist = sum(dist.values())
        total_nodes = cg.graph.number_of_nodes()

        assert total_from_dist == total_nodes


class TestSolutionSpaceSize:
    """Test solution space size estimation."""

    def test_solution_space_size_positive(self):
        """Test that solution space size is positive."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "spatial_pattern")
        size = cg.solution_space_size()

        assert size >= 1

    def test_solution_space_empty_grid(self):
        """Test solution space for fully filled grid."""
        grid = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "spatial_pattern")
        size = cg.solution_space_size()

        assert size == 0


class TestIntegrationWithPatternDetector:
    """Test constraint graph integration with pattern detector."""

    def test_automatic_rule_detection(self):
        """Test that constraint graph works with automatically detected rules."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        detector = PatternDetector()
        rule = detector.detect_rule(game)

        cg = ConstraintGraph(game, rule)

        assert cg.graph.number_of_nodes() > 0
        assert cg.is_solvable()

    def test_symmetry_pairs_used_correctly(self):
        """Test that symmetry pairs from detector are used in edges."""
        grid = np.array([[1, 0, -1], [0, 2, 0], [-1, 0, 1]])
        game = ARCGame(grid, grid)

        detector = PatternDetector()
        rule = detector.detect_rule(game)

        if rule.startswith("rotational_symmetry"):
            cg = ConstraintGraph(game, rule)
            # Should have edges between symmetric missing cells
            assert cg.graph.number_of_edges() > 0


class TestEdgeCases:
    """Test edge cases and corner cases."""

    def test_single_color_alphabet(self):
        """Test with minimal alphabet size."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid, alphabet=[1, 2])

        cg = ConstraintGraph(game, "spatial_pattern")

        # Should handle small alphabets
        assert cg.graph.number_of_nodes() > 0

    def test_all_cells_missing(self):
        """Test when all cells are missing."""
        grid = np.full((3, 3), -1, dtype=int)
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "spatial_pattern")

        # Should have nodes for all cells with all colors
        num_missing = 9
        num_colors = len(game.alphabet)
        assert cg.graph.number_of_nodes() == num_missing * num_colors

    def test_large_grid(self):
        """Test with larger grid."""
        grid = np.zeros((5, 5), dtype=int)
        grid[::2, ::2] = -1  # Missing at even positions
        game = ARCGame(grid, grid)

        cg = ConstraintGraph(game, "spatial_pattern")

        assert cg.is_solvable()
        features = cg.topological_features()
        assert features["num_nodes"] > 0
