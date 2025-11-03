# Solution Space Topology for MCTS: Constraint Graphs in ARC-style Tasks

A clean, production-ready implementation of topological Monte Carlo Tree Search using solution space topology (not grid topology) to guide search in puzzle-solving tasks.



## Quick Summary

**Problem:** Grid topology (cell connectivity) is **constant across all tasks** and provides no guidance.

**Solution:** Measure **solution space topology** via compatibility graphs that encode which color assignments are compatible under detected pattern rules.

**Results:**
- Synthetic tasks: Topological features are **2.01× more discriminative** than grid topology (3.40× for symmetric patterns)
- Real ARC-1 tasks: **2.04× average rollout efficiency**, up to **6.25× best-case** speedup on 20 hand-selected tasks

## Repository Structure

```
src/topomcts/
├── pattern_detector.py            # Pattern detection (5 types: rotational, reflective, frequency, etc.)
├── constraint_graph.py            # Build & analyze solution space topology
├── mcts.py                        # MCTS with constraint graph guidance
├── game.py                        # ARC-style game mechanics
├── arc_transformation_game.py     # Real ARC task formulation (NEW)
└── ...                            # Supporting modules

tests/
├── test_pattern_detector.py                    # 21 tests
├── test_constraint_graph.py                    # 27 tests
└── test_mcts_constraint_integration.py         # 15 tests

experiments/
├── test_pattern_detection_accuracy.py          # Validates 100% detection (synthetic)
├── compare_mcts_topologies.py                  # Validates 2.01× improvement (synthetic)
└── ARC-1/                                      # Real ARC-1 task evaluation (NEW)
    ├── run_real_arc_experiments.py             # Runs baseline + topological MCTS on 20 tasks
    ├── select_arc_tasks.py                     # Task selection script
    ├── selected_task_ids.json                  # 20 curated task IDs
    ├── selected_tasks.json                     # Full task data
    └── results/
        ├── real_arc_results.json               # Raw results (efficiency, quality, etc.)
        └── real_arc_results.csv                # CSV format

figs/
├── fig1_grid_vs_gc.pdf        # Grid vs. compatibility graph schematic
├── fig2_lambda2_hist.pdf      # Algebraic connectivity distribution
└── fig3_ablation.pdf          # Ablation study results


```

## Installation & Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run all unit tests (should pass 63/63)
pytest tests/ -v
```

## How to Replicate Paper Results

### 1. Synthetic Task Experiments (Section IV of Paper)

**Pattern Detection Accuracy (100%):**
```bash
python experiments/test_pattern_detection_accuracy.py
# Validates pattern detection on 48 synthetic tasks with exact match verification
```

**Topology Comparison (2.01× improvement):**
```bash
python experiments/compare_mcts_topologies.py
# Compares topological MCTS vs. vanilla MCTS across synthetic puzzle variants
# Output: Efficiency ratios, statistical significance, ablation analysis
```

### 2. Real ARC-1 Task Experiments (Section V.6 of Paper)

**Run evaluation on 20 hand-selected real ARC tasks:**
```bash
cd experiments/ARC-1
python run_real_arc_experiments.py
# Output: results/real_arc_results.json and results/real_arc_results.csv
# Reports: rollout efficiency (2.04× avg), quality metrics, per-task breakdown
```

**View experiment setup and task selection:**
```bash
# Task selection criteria & methodology:
python experiments/ARC-1/select_arc_tasks.py

# Test infrastructure on a single task:
python experiments/ARC-1/test_single_task.py
```

**Data Used:**
- `experiments/ARC-1/selected_task_ids.json` - 20 curated ARC-1 task IDs
- `experiments/ARC-1/selected_tasks.json` - Full task data (inputs, outputs, metadata)
- `experiments/ARC-1/results/` - Pre-computed results from paper evaluation

## What's Implemented

### Core Algorithm (Paper Sections III-IV)

✅ **Pattern Detection** (`pattern_detector.py`)
- 5 pattern types: rotational symmetry (90°/180°/270°), reflective symmetry (h/v/diag), color frequency, arithmetic progression, spatial pattern fallback
- 100% detection accuracy on synthetic tasks
- Deterministic threshold-based approach

✅ **Constraint Graphs** (`constraint_graph.py`)
- Nodes: (cell, color) pairs for missing cells
- Edges: Compatible assignments under pattern rules
- Features: algebraic connectivity λ₂, cell rigidity (entropy-based), color distribution
- Incremental updates with warm-started eigenvalue computation

✅ **MCTS Integration** (`mcts.py`)
- Selection formula: `UCB1 + β · sibling_normalized_topological_bonus`
- Sibling normalization: prevents single strong signal from dominating all children
- Constraint graph caching by (grid_bytes, pattern_rule)
- Backward compatible with vanilla MCTS

### Validation (Paper Sections IV-V)

**Synthetic Tasks (Section IV):**
✅ **Pattern Detection Accuracy**: 100% on 48 synthetic tasks
✅ **Topological Feature Discrimination**: 2.01× vs. grid topology
✅ **Ablation Study**: Shows λ₂ alone recovers 67% of improvement
✅ **Feature Analysis**: Grid Laplacian invariant, compatibility graph varies with task

**Real ARC-1 Tasks (Section V.6, NEW):**
✅ **Real-world Validation**: 20 hand-selected ARC-1 tasks with complexity metrics
✅ **Efficiency Gain**: 2.04× average rollout efficiency (baseline vs. topological MCTS)
✅ **Best-Case Performance**: 6.25× speedup on optimal tasks (spatial/reflective patterns)
✅ **Scaling Analysis**: 4.11× gain on large search spaces (>20 cells)

## Test Suite

**Total:** 63 tests, all passing ✅

| Module | Tests | Status |
|--------|-------|--------|
| Pattern Detection | 21 | ✅ |
| Constraint Graphs | 27 | ✅ |
| MCTS Integration | 15 | ✅ |

**Coverage:** Edge cases (empty grids, fully filled, disconnected graphs), integration pipeline, caching behavior.

## Key Results

**Synthetic Tasks:**
| Metric | Value |
|--------|-------|
| Pattern detection accuracy | 100% (48/48) |
| Constraint graph bonus | 2.01× vs. grid |
| Symmetric patterns | 3.40× improvement |
| MCTS success rate (full method) | 54±7% |
| Runtime overhead | 1.22× |

**Real ARC-1 Tasks:**
| Metric | Value |
|--------|-------|
| Average efficiency gain | 2.04× |
| Best-case efficiency gain | 6.25× |
| Pattern detection accuracy | 75% (15/20 specific) |
| Solution quality (both methods) | 69% |
| Tasks with search space >20 cells | 4.11× efficiency gain |

## Design Principles

- **Grid Topology is Invariant** (Lemma 1): The Laplacian of an m×n grid is constant across all tasks. Cannot discriminate difficulty.
- **Solution Space Varies** (Core Insight): Compatibility graphs encode which assignments are compatible—varies with pattern constraints and task difficulty.
- **Sibling Normalization** (Selection Formula): Features normalized within decision points prevent single signals from dominating globally.
- **Deterministic** (Reproducibility): All core algorithms are deterministic (only MCTS rollouts have randomness).

## Paper Details

**File:** `paper_camera_ready.tex` (670 lines)

**Contents:**
- 1 Lemma with proof (Grid-Laplacian invariance)
- 4 Formal definitions (Compatibility graph, Laplacian, Rigidity, ARCTransformationGame)
- 3 Algorithms (Pattern Detection, Incremental CG Update, SelectChildWithTopoUCB)
- 3 Figures (Grid vs Gc, Lambda2 histogram, Ablation results)
- 6 Tables (Notation, Detection, Ablation, Feature analysis, ARC-1 Results, ARC-1 Scaling)
- 5 Sections (Intro, Methods, Experiments on Synthetic, Experiments on Real ARC-1, Reproducibility)
- 8 References (Spectral graph theory, MCTS, CSP, ARC)

**Status:** Camera-ready with real ARC-1 validation (Section 5.6)
**Ready for:** arXiv, top-tier conferences (ICML, NeurIPS, ALENEX)

## Dependencies

```
Python ≥3.9
numpy ≥1.23
scipy ≥1.10
networkx ≥3.0
matplotlib ≥3.7
```

## Citation

If you use this code, please cite:

```bibtex
@article{topologicalMCTS,
  title={The Wrong Topology: Why Grid Structure Fails, and How Solution Space Topology Guides Search},
  author={...},
  journal={arXiv preprint arXiv:...},
  year={2025}
}
```

## License

MIT

## Reproducibility

All experiments are fully reproducible:

```bash
# Synthetic Task Experiments
# Pattern detection (deterministic, no randomness)
python experiments/test_pattern_detection_accuracy.py
# Output: 100% accuracy, 0 errors

# Topology comparison (deterministic, except MCTS rollouts)
python experiments/compare_mcts_topologies.py
# Output: 2.01× improvement with 95% CI across 4 seeds

# Real ARC-1 Task Experiments (NEW)
cd experiments/ARC-1
python run_real_arc_experiments.py
# Output: 20 tasks evaluated, results saved to results/real_arc_results.json
# 2.04× average efficiency gain, 6.25× best-case speedup
```

Hardware: Intel i7 CPU, 32 GB RAM, single-threaded wall-clock.
Runtime: Real ARC-1 experiments (~30 min for all 20 tasks with full MCTS evaluation)

## Key Files for Understanding

1. **Read first:** `paper_camera_ready.tex` (formal paper with all proofs/theorems)
2. **Core algorithm:** `src/topomcts/pattern_detector.py` + `src/topomcts/constraint_graph.py` + `src/topomcts/mcts.py`
3. **Validation:** `experiments/test_pattern_detection_accuracy.py` + `experiments/compare_mcts_topologies.py`
4. **Tests:** `tests/test_*.py` (verify all components work correctly)

## Next Steps (Future Work)

- Higher-level action abstractions for real ARC tasks (e.g., "Rotate object", "Fill region")
- Learning-based pattern detection for complex patterns (currently rule-based)
- Dynamic compatibility graphs for sequential/multi-step ARC tasks
- Application to other CSP domains (Sudoku, Nonogram, SAT, scheduling)
- Curriculum learning based on task topology (leveraging λ₂ analysis)

---

**Status:** Camera-ready with real ARC-1 validation, ready for submission to arXiv and top venues.
**Test Suite:** 63/63 passing ✅
**Code Quality:** 750 lines production, 831 lines tests (1:1 ratio)
**Real ARC Validation:** 20 tasks evaluated, 2.04× average efficiency gain
**Last Updated:** November 2, 2025
