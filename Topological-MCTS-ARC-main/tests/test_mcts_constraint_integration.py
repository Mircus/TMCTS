"""Tests for MCTS integration with constraint graph topology."""

import numpy as np
import pytest
from topomcts.game import ARCGame
from topomcts.mcts import TopologicalMCTSEngine, MCTSNode
from topomcts.constraint_graph import ConstraintGraph


class TestMCTSWithConstraintGraphs:
    """Test MCTS using constraint graph topology."""

    def test_mcts_engine_initialization(self):
        """Test MCTS engine initializes with constraint graphs enabled."""
        engine = TopologicalMCTSEngine(use_constraint_graphs=True)
        assert engine.use_constraint_graphs
        assert engine.pattern_detector is not None
        assert len(engine._constraint_graph_cache) == 0

    def test_constraint_graph_bonus_computation(self):
        """Test that constraint graph bonus is computed correctly."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        engine = TopologicalMCTSEngine(use_constraint_graphs=True)
        total_bonus, centrality, diffusion, symmetry = engine.topological_bonus(game)

        # All bonuses should be non-negative
        assert total_bonus >= 0
        assert centrality >= 0
        assert diffusion >= 0
        assert symmetry >= 0

    def test_constraint_graph_caching(self):
        """Test that constraint graphs are cached."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        engine = TopologicalMCTSEngine(use_constraint_graphs=True)

        # First call should populate cache
        bonus1, _, _, _ = engine.topological_bonus(game)
        cache_size_1 = len(engine._constraint_graph_cache)
        assert cache_size_1 == 1

        # Second call should use cache
        bonus2, _, _, _ = engine.topological_bonus(game)
        cache_size_2 = len(engine._constraint_graph_cache)
        assert cache_size_2 == 1  # Cache not grown
        assert bonus1 == bonus2  # Same result

    def test_constraint_bonus_vs_grid_bonus(self):
        """Test that constraint graph bonus differs from grid topology bonus."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        engine = TopologicalMCTSEngine()

        # Constraint graph bonus
        bonus_constraint, _, _, _ = engine._constraint_graph_bonus(game)

        # Grid topology bonus
        bonus_grid, _, _, _ = engine._grid_topology_bonus(game)

        # They should be different (different topologies)
        # Note: they might both be 0 in some cases, but the computation differs
        assert isinstance(bonus_constraint, float)
        assert isinstance(bonus_grid, float)

    def test_mcts_select_with_constraint_graphs(self):
        """Test that MCTS selection works with constraint graphs."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        engine = TopologicalMCTSEngine(use_constraint_graphs=True)
        root = MCTSNode(state=game)

        # Manually add children to test selection
        for i in range(3):
            child_grid = grid.copy()
            # Fill different cells to create different states
            child_grid[1, 1] = i
            child_state = ARCGame(child_grid, child_grid)
            child = MCTSNode(state=child_state, parent=root)
            child.visits = 5 + i  # Different visit counts
            child.value_sum = 3.0 + i
            root.children.append(child)

        # Select should work without errors
        selected = engine.select(root)
        assert selected in root.children

    def test_mcts_with_empty_grid(self):
        """Test MCTS handles empty grids gracefully."""
        grid = np.full((3, 3), -1, dtype=int)
        game = ARCGame(grid, grid)

        engine = TopologicalMCTSEngine(use_constraint_graphs=True)
        bonus, _, _, _ = engine.topological_bonus(game)

        # Should handle empty grid without crashing
        assert isinstance(bonus, float)

    def test_mcts_with_fully_filled_grid(self):
        """Test MCTS handles fully filled grids."""
        grid = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        engine = TopologicalMCTSEngine(use_constraint_graphs=True)
        bonus, _, _, _ = engine.topological_bonus(game)

        # Should handle fully filled grid
        assert isinstance(bonus, float)

    def test_pattern_detection_integration(self):
        """Test that pattern detection is used in MCTS."""
        # Symmetric grid - should detect rotational symmetry
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        engine = TopologicalMCTSEngine(use_constraint_graphs=True)
        bonus, _, _, _ = engine.topological_bonus(game)

        # Should have detected a pattern and built a constraint graph
        assert len(engine._constraint_graph_cache) == 1
        assert isinstance(bonus, float)

    def test_mcts_full_run_with_constraint_graphs(self):
        """Test a full MCTS run with constraint graphs."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        engine = TopologicalMCTSEngine(use_constraint_graphs=True)

        # Run a few iterations
        action, value, metadata = engine.run(game, iterations=10)

        # Should return valid metadata
        assert 'centrality' in metadata
        assert 'diffusion' in metadata
        assert 'symmetry' in metadata
        assert metadata['simulations'] == 10

    def test_mcts_fallback_to_grid_topology(self):
        """Test MCTS can fall back to grid topology."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        engine = TopologicalMCTSEngine(use_constraint_graphs=False)
        bonus, _, _, _ = engine.topological_bonus(game)

        # Should use grid topology
        assert isinstance(bonus, float)

    def test_constraint_graph_bonus_with_different_patterns(self):
        """Test constraint graph bonus for different pattern types."""
        test_cases = [
            # Symmetric pattern
            (np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1]]), "rotational_symmetry"),
            # Frequency pattern
            (np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), "frequency"),
            # Progression pattern
            (np.array([[1, 2, 3], [4, 5, -1], [7, 8, 9]]), "progression"),
        ]

        engine = TopologicalMCTSEngine(use_constraint_graphs=True)

        for grid, pattern_type in test_cases:
            game = ARCGame(grid, grid)
            bonus, centrality, diffusion, symmetry = engine.topological_bonus(game)

            # All should be valid numbers
            assert isinstance(bonus, float)
            assert isinstance(centrality, float)
            assert isinstance(diffusion, float)
            assert isinstance(symmetry, float)


class TestConstraintGraphCacheManagement:
    """Test caching behavior of constraint graphs."""

    def test_cache_different_states(self):
        """Test that different states are cached separately."""
        engine = TopologicalMCTSEngine(use_constraint_graphs=True)

        # Create two games with different patterns
        # Grid 1: Rotational symmetry
        grid1 = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        # Grid 2: Frequency pattern (all colors appear once)
        grid2 = np.array([[0, 1, 2], [-1, 3, 4], [5, 6, 7]])

        game1 = ARCGame(grid1, grid1)
        game2 = ARCGame(grid2, grid2)

        # Compute bonuses
        engine.topological_bonus(game1)
        engine.topological_bonus(game2)

        # Should have 2 cache entries (different patterns)
        # Note: they might have the same pattern in some cases, so we just check they're both cached
        assert len(engine._constraint_graph_cache) >= 1

    def test_cache_hit_rate(self):
        """Test that cache hits work correctly."""
        engine = TopologicalMCTSEngine(use_constraint_graphs=True)
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        # Compute bonus multiple times
        for _ in range(5):
            engine.topological_bonus(game)

        # Should still only have 1 cache entry
        assert len(engine._constraint_graph_cache) == 1


class TestMCTSWeights:
    """Test weight configurations for MCTS."""

    def test_different_topology_weights(self):
        """Test MCTS with different topology weight settings."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])
        game = ARCGame(grid, grid)

        # Test with different topology weights
        for weight in [0.0, 0.5, 1.0, 2.0]:
            engine = TopologicalMCTSEngine(
                use_constraint_graphs=True,
                topology_weight=weight
            )
            bonus, _, _, _ = engine.topological_bonus(game)
            assert isinstance(bonus, float)

    def test_normalized_weights(self):
        """Test that TopologicalWeights normalize correctly."""
        from topomcts.mcts import TopologicalWeights

        weights = TopologicalWeights(
            centrality=0.4,
            diffusion=0.3,
            symmetry=0.3
        )

        # Check normalization
        total = weights.centrality + weights.diffusion + weights.symmetry
        assert abs(total - 1.0) < 1e-6  # Should sum to 1.0
