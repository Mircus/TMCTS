# ARC-1 Real-World Evaluation Experiments

## Overview

This directory contains experiments evaluating **Topological MCTS** on real ARC-1 (Abstraction and Reasoning Corpus) tasks from the official François Chollet benchmark.

### Key Difference: Real vs. Synthetic Tasks

- **Synthetic Tasks** (current implementation): Grids with marked missing cells (-1), which must be filled
- **Real ARC-1 Tasks**: Input grids are complete; the task is to **learn the transformation** from training examples and **apply it** to test inputs

## Data Preparation

### Selected Tasks

We selected **20 middle-difficulty tasks** from the 400 ARC-1 training tasks using complexity metrics:

- **Grid size**: Moderate area (not too small, not too large)
- **Color diversity**: Multiple unique colors in train/test
- **Transformation difficulty**: Moderate number of cells affected

### Task IDs

See `selected_task_ids.json` for the list of 20 selected tasks.

## Current State

### What Works

✅ **Data loading and task selection**
- Successfully cloned official ARC-AGI repository
- Analyzed all 400 training tasks
- Selected 20 middle-difficulty tasks

✅ **Pattern detection on real tasks**
- Pattern detector runs on training examples
- Detects rules: rotational/reflective symmetry, color frequency, etc.

✅ **MCTS engines**
- StandardMCTSEngine (baseline)
- TopologicalMCTSEngine (with topological guidance)
- Both engines execute successfully

### Current Limitation

❌ **Task formulation mismatch**

The current ARCGame implementation expects:
- **Input**: Partial grid with missing cells marked as -1
- **Output**: Filled grid with all cells completed

But real ARC-1 tasks provide:
- **Input**: Complete grid (no missing cells)
- **Output**: Transformed grid (different shape/content)

The transformation is **learned from training examples**, not filled cell-by-cell.

## Next Steps

### Option 1: Adapt Current Implementation (Recommended for Paper)

Modify the game to treat each cell as a "decision" and learn the transformation pattern:

1. **Pattern Learning Phase**: Analyze training examples to extract transformation rule
2. **Test Application Phase**: Use MCTS to iteratively apply the learned pattern to test input
3. **Quality Metric**: Compare generated output with ground truth

Implementation points:
- Extend `PatternDetector` to learn more complex transformation rules
- Create `ARCTransformationGame` wrapper that treats pattern application as a search problem
- Use topological features of the transformation graph

### Option 2: Create Synthetic Subset

Generate synthetic versions of real tasks with marked missing cells:

1. Take real task training examples
2. Randomly occlude cells from outputs
3. Task becomes: fill occlusions while preserving learned pattern

This leverages existing infrastructure but doesn't directly evaluate on real ARC tasks.

### Option 3: Skip Real ARC for Now

Focus on:
- Expanding the paper with better synthetic task evaluation
- Adding more pattern types and complexity metrics
- Implementing the Gemini reviewer's suggestions (acknowledgments, bibliography, metrics clarification)

## Files in This Directory

```
ARC-1/
├── README.md (this file)
├── select_arc_tasks.py         # Select middle-difficulty tasks (already run)
├── selected_task_ids.json      # List of 20 selected task IDs
├── selected_tasks.json         # Full task data for selected tasks
├── test_single_task.py         # Test infrastructure on one task
├── run_arc1_experiments.py     # Full experiment runner (framework ready)
└── results/                     # Output directory (will be created after running experiments)
    └── arc1_results.csv        # Results in CSV format
    └── arc1_results.json       # Results in JSON format
```

## Recommended Action

**For the paper submission**: Implement Option 1 to properly evaluate on real ARC-1 tasks:

1. Design `ARCTransformationGame` that treats rule application as search
2. Extend `PatternDetector` to identify when an output was successfully generated
3. Run experiments on the 20 selected tasks
4. Report:
   - Pattern detection accuracy on real tasks
   - Success rate on real test cases
   - Rollout efficiency (topological vs. baseline)
   - Comparison with synthetic task performance

This directly addresses Gemini's suggestion: **"Integrate a small, secondary experiment on a hand-selected set of real ARC tasks"**

## Running the Code

### Step 1: Task Selection (Already Done)

```bash
python experiments/ARC-1/select_arc_tasks.py
```

Outputs: `selected_task_ids.json`, `selected_tasks.json`

### Step 2: Quick Test

```bash
python experiments/ARC-1/test_single_task.py
```

Verifies infrastructure works on first selected task.

### Step 3: Full Experiment (After Implementing Option 1)

```bash
python experiments/ARC-1/run_arc1_experiments.py
```

Will output results to `results/arc1_results.{csv,json}`

## Resources

- **Official ARC-AGI Repository**: https://github.com/fchollet/ARC-AGI
- **Paper Review**: `review-gemini.txt`
- **Main Implementation**: `src/topomcts/`
- **Synthetic Experiments**: `experiments/test_pattern_detection_accuracy.py`, `experiments/compare_mcts_topologies.py`

## Questions to Resolve

1. **Pattern Learning**: How do we learn and represent transformation rules from examples?
   - Current: Detect one rule type per grid
   - Needed: Learn complex, multi-step transformations

2. **Search Space**: What is the search space for real ARC tasks?
   - Current: Fill missing cells with colors from alphabet
   - Needed: Generate output grid cells from learned transformation

3. **Evaluation**: How do we measure success?
   - Current: Exact match with target grid
   - Needed: Pixel-wise accuracy, structural similarity, or task-specific metrics

4. **Topological Guidance**: How does solution space topology apply?
   - Current: Constraint graphs on color assignments
   - Needed: Graphs on transformation parameters or output space

## Contact

For questions about this evaluation framework, see `review-gemini.txt` for the original suggestions from the paper review.
