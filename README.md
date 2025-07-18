# Batched Tensor Networks Contraction 

This module provides efficient batching of tensor contractions across multiple contraction trees by identifying common patterns and executing them in parallel using einsum operations.

## Installation

Install the package directly via pip:

```bash
pip install git+https://github.com/kinianlo/batch-tn.git
```

Or clone and install locally:

```bash
git clone https://github.com/kinianlo/batch-tn.git
cd batch-tn
pip install -e .
```

## Quick Start

```python
from batch_tn import batch_einsum
import numpy as np

eqs = [
    'ij,jk->ik', # matrix-matrix multiplication
    'i,ikj,j->k' # vector-matrix-vector multiplication
]

shapes_list = [
    [(3, 4), (4, 5)], # shapes for the first equation
    [(3,), (3, 5, 4), (4,)] # shapes for the second equation
]

arrays_list = [[np.random.rand(*shape) for shape in shapes] for shapes in shapes_list]

batch_einsum(eqs, arrays_list)
```

## API Reference

### `batch_tn.batch_einsum(eqs, arrays)`

Main function for batched contraction.

**Parameters:**
- `eqs`: List of einsum equations
- `arrays`: List of arrays for each equation

## How It Works

1. **Pattern Recognition**: Identifies identical contraction patterns across contraction trees
2. **Batching**: Groups contractions with the same einsum equation and tensor shapes
3. **Result Distribution**: Distributes results back to individual trees
4. **Progressive Contraction**: Continues until all trees are fully contracted

## Performance Benefits

- **Reduced Operations**: N individual contractions → 1 batched contraction
- **GPU Ready**: Batching approach naturally extends to GPU computation