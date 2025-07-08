# Batched Tensor Networks Contraction 

This module provides efficient batching of tensor contractions across multiple contraction trees by identifying common patterns and executing them in parallel using einsum operations.

## Quick Start

```python
import numpy as np
import cotengra as ctg
from batch_contract import batch_contract


trees = [
    ctg.einsum_tree('ij,jk->ik', *[(4, 4), (4, 4)]),
    ctg.einsum_tree('ij,jk->ik', *[(4, 4), (4, 4)]),
    ctg.einsum_tree('ab,bc->ac', *[(4, 4), (4, 4)])
]

arrays = [
    [np.random.rand(4, 4), np.random.rand(4, 4)],
    [np.random.rand(4, 4), np.random.rand(4, 4)],
    [np.random.rand(4, 4), np.random.rand(4, 4)]
]

batch_contract(trees, arrays=arrays)
```

## API Reference

### `batch_contract(trees, arrays)`

Main function for batched contraction.

**Parameters:**
- `trees`: List of contraction trees
- `arrays`: List of arrays for each tree

## How It Works

1. **Pattern Recognition**: Identifies identical contraction patterns across trees
2. **Batching**: Groups contractions with the same einsum equation and tensor shapes
3. **Parallel Execution**: Uses `np.einsum` with `...` notation to process batches
4. **Result Distribution**: Distributes results back to individual trees
5. **Progressive Contraction**: Continues until all trees are fully contracted

## Performance Benefits

- **Reduced Operations**: N individual contractions → 1 batched contraction
- **GPU Ready**: Batching approach naturally extends to GPU computation

## Examples

See `demo.ipynb` for a demonstration of how to use the `batch_contract` function.
