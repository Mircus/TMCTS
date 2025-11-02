# Contributing to Topological MCTS

Thank you for your interest in contributing! This project is focused on validating the solution space topology principle for MCTS-guided puzzle solving.

## Getting Started

1. **Fork the repository** and clone your fork locally
2. **Set up the environment**: `pip install -e .`
3. **Run tests**: `pytest tests/ -v` (should see 63/63 passing)
4. **Create a new branch** for your feature or bugfix

## Development Guidelines

- **Code style**: Follow PEP 8. Use `black` for formatting.
- **Tests**: All new features must include tests. Aim for >90% coverage.
- **Documentation**: Update docstrings and README when adding features.
- **Commits**: Use clear, descriptive commit messages.

## Areas for Contribution

### High Priority
- **Higher-level action spaces** for real ARC tasks (e.g., "Rotate object", "Fill region")
- **Improved pattern detection** for complex rules (currently rule-based, consider learning-based)
- **Dynamic constraint graphs** for sequential/multi-step reasoning

### Medium Priority
- Application to other CSP domains (Sudoku, Nonogram, SAT, scheduling)
- Performance optimizations for large grids
- Extended test coverage for edge cases

### Lower Priority
- Visualization tools for constraint graphs
- Benchmark comparisons with other MCTS variants
- Curriculum learning strategies based on λ₂

## Testing

Before submitting a PR:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/topomcts

# Run specific experiment
python experiments/test_pattern_detection_accuracy.py
python experiments/compare_mcts_topologies.py
cd experiments/ARC-1 && python run_real_arc_experiments.py
```

## Paper References

Our method is based on:
- Spectral graph theory (algebraic connectivity, Fiedler vector)
- Monte Carlo Tree Search (UCB1, PUCT)
- Constraint Satisfaction Problems (CSP hardness, phase transitions)
- Abstraction and Reasoning Corpus (ARC) benchmark

See `paper_camera_ready.tex` for formal definitions and proofs.

## Questions?

- Read the paper (`paper_camera_ready.tex`) for theoretical background
- Check existing tests for usage examples
- Look at `src/topomcts/` for implementation details

Thank you for contributing!
