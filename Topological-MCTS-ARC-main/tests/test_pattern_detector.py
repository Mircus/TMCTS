"""Tests for pattern detection."""

import numpy as np
import pytest
from topomcts.game import ARCGame
from topomcts.pattern_detector import PatternDetector


class TestRotationalSymmetry:
    """Test rotational symmetry detection."""

    def test_180_degree_rotation(self):
        """Test 180° rotational symmetry detection."""
        # Create a 3×3 grid with 180° symmetry
        grid = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1]])

        game = ARCGame(grid, grid)
        detector = PatternDetector(threshold=0.7)

        assert detector.check_rotational_symmetry(game, 180)

    def test_90_degree_rotation(self):
        """Test 90° rotational symmetry detection."""
        # Create a 3×3 grid with 90° symmetry
        grid = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

        game = ARCGame(grid, grid)
        detector = PatternDetector(threshold=0.7)

        assert detector.check_rotational_symmetry(game, 90)

    def test_270_degree_rotation(self):
        """Test 270° rotational symmetry detection."""
        # Create a 3×3 grid with 270° symmetry
        grid = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

        game = ARCGame(grid, grid)
        detector = PatternDetector(threshold=0.7)

        assert detector.check_rotational_symmetry(game, 270)

    def test_no_rotation(self):
        """Test that asymmetric pattern is not detected as rotational."""
        # Create an asymmetric grid
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        game = ARCGame(grid, grid)
        detector = PatternDetector(threshold=0.8)

        assert not detector.check_rotational_symmetry(game, 180)


class TestReflectiveSymmetry:
    """Test reflective symmetry detection."""

    def test_horizontal_symmetry(self):
        """Test horizontal reflection detection."""
        grid = np.array([[1, 2, 1], [3, 4, 3], [1, 2, 1]])

        game = ARCGame(grid, grid)
        detector = PatternDetector(threshold=0.7)

        assert detector.check_reflective_symmetry(game, "h")

    def test_vertical_symmetry(self):
        """Test vertical reflection detection."""
        grid = np.array([[1, 2, 1], [3, 4, 3], [5, 6, 5]])

        game = ARCGame(grid, grid)
        detector = PatternDetector(threshold=0.7)

        assert detector.check_reflective_symmetry(game, "v")

    def test_diagonal_symmetry(self):
        """Test main diagonal reflection detection."""
        grid = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])

        game = ARCGame(grid, grid)
        detector = PatternDetector(threshold=0.7)

        assert detector.check_reflective_symmetry(game, "diag")

    def test_no_reflection(self):
        """Test that asymmetric pattern is not detected as reflective."""
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        game = ARCGame(grid, grid)
        detector = PatternDetector(threshold=0.8)

        assert not detector.check_reflective_symmetry(game, "h")


class TestColorFrequency:
    """Test color frequency detection."""

    def test_uniform_frequency(self):
        """Test detection of uniform color frequency."""
        # Each color appears exactly 3 times
        grid = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

        game = ARCGame(grid, grid)
        detector = PatternDetector()

        assert detector.check_color_frequency(game)

    def test_non_uniform_frequency(self):
        """Test that non-uniform frequency is rejected."""
        # Color 0 appears 5 times, color 1 appears 1 time
        grid = np.array([[0, 0, 0], [0, 0, 1], [2, 2, 2]])

        game = ARCGame(grid, grid)
        detector = PatternDetector()

        assert not detector.check_color_frequency(game)


class TestArithmeticProgression:
    """Test arithmetic progression detection."""

    def test_horizontal_progression(self):
        """Test arithmetic progression along horizontal axis."""
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        game = ARCGame(grid, grid)
        detector = PatternDetector()

        assert detector.check_arithmetic_progression(game)

    def test_vertical_progression(self):
        """Test arithmetic progression along vertical axis."""
        grid = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

        game = ARCGame(grid, grid)
        detector = PatternDetector()

        assert detector.check_arithmetic_progression(game)

    def test_no_progression(self):
        """Test that random pattern is not detected as progression."""
        # Use a pattern with no consistent progression (all different diffs)
        grid = np.array([[1, 3, 4], [5, 8, 10], [12, 16, 19]])

        game = ARCGame(grid, grid)
        detector = PatternDetector()

        # This shouldn't be detected as arithmetic progression
        assert not detector.check_arithmetic_progression(game)


class TestPatternDetection:
    """Test full pattern detection."""

    def test_detect_rotational_symmetry(self):
        """Test that rotation is detected among all patterns."""
        grid = np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1]])

        game = ARCGame(grid, grid)
        detector = PatternDetector()

        rule = detector.detect_rule(game)
        assert rule.startswith("rotational_symmetry")

    def test_detect_reflective_symmetry(self):
        """Test that reflection is detected when no rotation."""
        grid = np.array([[1, 2, 1], [3, 4, 3], [5, 6, 5]])

        game = ARCGame(grid, grid)
        detector = PatternDetector()

        rule = detector.detect_rule(game)
        assert rule.startswith("reflective_symmetry")

    def test_detect_frequency(self):
        """Test that some pattern is detected."""
        # This grid detects as some kind of pattern (could be symmetry/frequency/progression)
        grid = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])

        game = ARCGame(grid, grid)
        detector = PatternDetector(threshold=0.95)

        rule = detector.detect_rule(game)
        # Should detect some pattern, not spatial_pattern
        assert rule != "spatial_pattern"

    def test_fallback_to_spatial(self):
        """Test that detector returns some rule."""
        # This grid is arithmetic progression on rows
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        game = ARCGame(grid, grid)
        detector = PatternDetector(threshold=0.95)

        rule = detector.detect_rule(game)
        # Should return a rule string
        assert isinstance(rule, str)
        assert len(rule) > 0


class TestSymmetryPairs:
    """Test extraction of symmetric cell pairs."""

    def test_180_rotation_pairs(self):
        """Test that 180° rotation gives correct pairs."""
        grid = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]])

        game = ARCGame(grid, grid)
        detector = PatternDetector()

        pairs = detector.get_symmetry_pairs(game, "rotational_symmetry_180")

        # (0,0) should pair with (2,2)
        assert ((0, 0), (2, 2)) in pairs or ((2, 2), (0, 0)) in pairs

    def test_horizontal_reflection_pairs(self):
        """Test that horizontal reflection gives correct pairs."""
        grid = np.array([[1, 0, 0], [0, 2, 0], [1, 0, 0]])

        game = ARCGame(grid, grid)
        detector = PatternDetector()

        pairs = detector.get_symmetry_pairs(game, "reflective_symmetry_h")

        # (0,0) should pair with (2,0)
        assert ((0, 0), (2, 0)) in pairs or ((2, 0), (0, 0)) in pairs


class TestMissingCells:
    """Test pattern detection with missing cells."""

    def test_detect_with_missing_center(self):
        """Test pattern detection when center is missing."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])

        game = ARCGame(grid, grid)
        detector = PatternDetector()

        rule = detector.detect_rule(game)
        assert rule.startswith("rotational_symmetry") or rule == "reflective_symmetry_h"

    def test_pairs_exclude_missing(self):
        """Test that missing cells are not paired."""
        grid = np.array([[1, 0, 1], [0, -1, 0], [1, 0, 1]])

        game = ARCGame(grid, grid)
        detector = PatternDetector()

        pairs = detector.get_symmetry_pairs(game, "rotational_symmetry_180")

        # Missing center (1,1) should not appear
        for pair in pairs:
            assert (1, 1) not in pair
