"""
Pattern detection for ARC-style grid tasks.

Identifies pattern rules (symmetry, frequency, progression) from partially filled grids.
This enables constraint graph construction for solution space topology analysis.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from .game import ARCGame


class PatternDetector:
    """Detect pattern rules from game states."""

    def __init__(self, threshold: float = 0.8):
        """
        Initialize pattern detector.

        Args:
            threshold: Confidence threshold for pattern detection (0-1).
                      Higher = more conservative.
        """
        self.threshold = threshold

    def detect_rule(self, state: ARCGame) -> str:
        """
        Detect the pattern rule governing the task.

        Returns one of:
        - "rotational_symmetry_90"
        - "rotational_symmetry_180"
        - "rotational_symmetry_270"
        - "reflective_symmetry_h" (horizontal)
        - "reflective_symmetry_v" (vertical)
        - "reflective_symmetry_diag" (main diagonal)
        - "reflective_symmetry_antidiag" (anti-diagonal)
        - "color_frequency" (uniform color distribution)
        - "arithmetic_progression" (color progression)
        - "spatial_pattern" (unknown/fallback)
        """
        # Try symmetries first (most specific)
        for angle in [90, 180, 270]:
            if self.check_rotational_symmetry(state, angle):
                return f"rotational_symmetry_{angle}"

        for axis in ["h", "v", "diag", "antidiag"]:
            if self.check_reflective_symmetry(state, axis):
                return f"reflective_symmetry_{axis}"

        # Then frequency/progression
        if self.check_color_frequency(state):
            return "color_frequency"

        if self.check_arithmetic_progression(state):
            return "arithmetic_progression"

        # Fallback
        return "spatial_pattern"

    def check_rotational_symmetry(self, state: ARCGame, angle: int) -> bool:
        """
        Check if filled cells have rotational symmetry.

        Args:
            state: Game state
            angle: Rotation angle (90, 180, 270)

        Returns:
            True if pattern has this rotational symmetry
        """
        grid = state.initial
        n, m = grid.shape

        if angle == 90:
            # After 90° rotation: (r, c) -> (c, n-1-r)
            matches = 0
            total = 0
            for r in range(n):
                for c in range(m):
                    if grid[r, c] != -1:  # If filled
                        new_r, new_c = c, n - 1 - r
                        if 0 <= new_r < n and 0 <= new_c < m:
                            if grid[new_r, new_c] != -1:
                                if grid[r, c] == grid[new_r, new_c]:
                                    matches += 1
                                total += 1
            return matches / max(1, total) >= self.threshold

        elif angle == 180:
            # After 180° rotation: (r, c) -> (n-1-r, m-1-c)
            matches = 0
            total = 0
            for r in range(n):
                for c in range(m):
                    if grid[r, c] != -1:  # If filled
                        new_r, new_c = n - 1 - r, m - 1 - c
                        if 0 <= new_r < n and 0 <= new_c < m:
                            if grid[new_r, new_c] != -1:
                                if grid[r, c] == grid[new_r, new_c]:
                                    matches += 1
                                total += 1
            return matches / max(1, total) >= self.threshold

        elif angle == 270:
            # After 270° rotation: (r, c) -> (m-1-c, r)
            matches = 0
            total = 0
            for r in range(n):
                for c in range(m):
                    if grid[r, c] != -1:  # If filled
                        new_r, new_c = m - 1 - c, r
                        if 0 <= new_r < n and 0 <= new_c < m:
                            if grid[new_r, new_c] != -1:
                                if grid[r, c] == grid[new_r, new_c]:
                                    matches += 1
                                total += 1
            return matches / max(1, total) >= self.threshold

        return False

    def check_reflective_symmetry(self, state: ARCGame, axis: str) -> bool:
        """
        Check if filled cells have reflective symmetry.

        Args:
            state: Game state
            axis: Axis of symmetry ("h"=horizontal, "v"=vertical, "diag", "antidiag")

        Returns:
            True if pattern has this reflective symmetry
        """
        grid = state.initial
        n, m = grid.shape

        matches = 0
        total = 0

        if axis == "h":
            # Horizontal mirror: (r, c) -> (n-1-r, c)
            for r in range(n):
                for c in range(m):
                    if grid[r, c] != -1:
                        new_r = n - 1 - r
                        if grid[new_r, c] != -1:
                            if grid[r, c] == grid[new_r, c]:
                                matches += 1
                            total += 1

        elif axis == "v":
            # Vertical mirror: (r, c) -> (r, m-1-c)
            for r in range(n):
                for c in range(m):
                    if grid[r, c] != -1:
                        new_c = m - 1 - c
                        if grid[r, new_c] != -1:
                            if grid[r, c] == grid[r, new_c]:
                                matches += 1
                            total += 1

        elif axis == "diag":
            # Main diagonal: (r, c) -> (c, r)
            for r in range(n):
                for c in range(m):
                    if grid[r, c] != -1:
                        if c < n and r < m:  # Ensure in bounds
                            if grid[c, r] != -1:
                                if grid[r, c] == grid[c, r]:
                                    matches += 1
                                total += 1

        elif axis == "antidiag":
            # Anti-diagonal: (r, c) -> (m-1-c, n-1-r)
            for r in range(n):
                for c in range(m):
                    if grid[r, c] != -1:
                        new_r, new_c = m - 1 - c, n - 1 - r
                        if 0 <= new_r < n and 0 <= new_c < m:
                            if grid[new_r, new_c] != -1:
                                if grid[r, c] == grid[new_r, new_c]:
                                    matches += 1
                                total += 1

        return matches / max(1, total) >= self.threshold if total > 0 else False

    def check_color_frequency(self, state: ARCGame) -> bool:
        """
        Check if colors appear with approximately uniform frequency.

        Returns:
            True if color distribution is uniform
        """
        grid = state.initial
        filled_cells = grid[grid != -1]

        if len(filled_cells) == 0:
            return False

        unique_colors, counts = np.unique(filled_cells, return_counts=True)

        if len(unique_colors) < 2:
            return False

        # Check coefficient of variation
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        cv = std_count / mean_count if mean_count > 0 else 0

        # Uniform if CV < 0.3 (colors appear with similar frequency)
        return cv < 0.3

    def check_arithmetic_progression(self, state: ARCGame) -> bool:
        """
        Check if colors follow arithmetic progression along any axis.

        Returns:
            True if arithmetic progression detected
        """
        grid = state.initial
        n, m = grid.shape

        # Check horizontal progressions
        for r in range(n):
            row = grid[r, grid[r, :] != -1]
            if len(row) >= 3:
                diffs = np.diff(row)
                if np.allclose(diffs, diffs[0]):
                    return True

        # Check vertical progressions
        for c in range(m):
            col = grid[grid[:, c] != -1, c]
            if len(col) >= 3:
                diffs = np.diff(col)
                if np.allclose(diffs, diffs[0]):
                    return True

        return False

    def get_symmetry_pairs(
        self, state: ARCGame, rule: str
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Get pairs of symmetric cells for a detected symmetry rule.

        Args:
            state: Game state
            rule: Pattern rule (e.g., "rotational_symmetry_180")

        Returns:
            List of (cell1, cell2) pairs that are symmetric
        """
        grid = state.initial
        n, m = grid.shape
        pairs = []

        if rule == "rotational_symmetry_90":
            for r in range(n):
                for c in range(m):
                    if grid[r, c] != -1:
                        new_r, new_c = c, n - 1 - r
                        if (new_r, new_c) not in [p[1] for p in pairs]:
                            pairs.append(((r, c), (new_r, new_c)))

        elif rule == "rotational_symmetry_180":
            for r in range(n):
                for c in range(m):
                    if grid[r, c] != -1:
                        new_r, new_c = n - 1 - r, m - 1 - c
                        if (new_r, new_c) not in [p[1] for p in pairs]:
                            pairs.append(((r, c), (new_r, new_c)))

        elif rule == "rotational_symmetry_270":
            for r in range(n):
                for c in range(m):
                    if grid[r, c] != -1:
                        new_r, new_c = m - 1 - c, r
                        if (new_r, new_c) not in [p[1] for p in pairs]:
                            pairs.append(((r, c), (new_r, new_c)))

        elif rule == "reflective_symmetry_h":
            for r in range(n):
                for c in range(m):
                    if grid[r, c] != -1:
                        new_r = n - 1 - r
                        if (new_r, c) not in [p[1] for p in pairs]:
                            pairs.append(((r, c), (new_r, c)))

        elif rule == "reflective_symmetry_v":
            for r in range(n):
                for c in range(m):
                    if grid[r, c] != -1:
                        new_c = m - 1 - c
                        if (r, new_c) not in [p[1] for p in pairs]:
                            pairs.append(((r, c), (r, new_c)))

        return pairs
