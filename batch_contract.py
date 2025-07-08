"""
Batched tensor contraction for multiple trees.

This module provides efficient batching of tensor contractions across multiple
contraction trees by identifying common patterns and executing them in parallel.
"""

from collections import defaultdict

import numpy as np
from cotengra import ContractionTree

Array = np.ndarray
ArrayShape = tuple[int, ...]
Signature = tuple[str, ArrayShape, ArrayShape]
TreeNode = frozenset[int]

def get_node_shape(tree: ContractionTree, node: TreeNode) -> ArrayShape:
    """
    Get the shape of a node in the contraction tree.

    Parameters
    ----------
    tree : ContractionTree
        The contraction tree containing the node.
    node : TreeNode
        The node for which to get the shape.

    Returns
    -------
    ArrayShape
        The shape of the node as a tuple of integers.
    """
    return tuple(tree.size_dict[i] for i in tree.get_inds(node))

def extract_signature(tree: ContractionTree, parent: TreeNode) -> Signature:
    """
    Extract the signature for a given node in the contraction tree.

    Parameters
    ----------
    tree : ContractionTree
        The contraction tree containing the node.
    parent : TreeNode
        The parent node for which to extract the signature.

    Returns
    -------
    Signature
        A tuple representing the signature of the node.
    """
    eq = tree.get_einsum_eq(parent)
    left, right = tree.children[parent]
    left_shape = get_node_shape(tree, left)
    right_shape = get_node_shape(tree, right)

    return (eq, left_shape, right_shape)

def get_batched_einsum_equation(original_eq: str) -> str:
    """
    Convert an einsum equation to a batched version by prepending a batch symbol.
    This function modifies the equation to allow for batched operations.
    
    Parameters
    ----------
    original_eq : str
        The original einsum equation string, e.g., "abc,cd->abd".

    Returns
    -------
    str
        The modified einsum equation string with a batch symbol prepended,
        e.g., "...abc,...cd->...abd".
    """
    BATCH_SYMBOL = '...'
    left_side, right_side = original_eq.split('->')
    terms = left_side.split(',')
    
    batched_terms = [f'{BATCH_SYMBOL}{term}' for term in terms]
    batched_output = f'{BATCH_SYMBOL}{right_side}'

    return f"{','.join(batched_terms)}->{batched_output}"

def batch_contract(trees: list[ContractionTree], arrays: list[list[Array]]) -> list[Array]:
    """
    Perform batched tensor contraction across multiple trees.

    Parameters
    ----------
    trees : List
        List of contraction trees to process.
    arrays : List[List[np.ndarray]]
        List of numpy arrays corresponding to each tree.

    Returns
    -------
    List[np.ndarray]
        List of results for each tree.
    """
    ready: dict[Signature, tuple[int, TreeNode, TreeNode, TreeNode]] = defaultdict(list)
    tree_active_nodes: list[dict[TreeNode, Array]] = [dict() for _ in trees]
    tree_parent: list[dict[TreeNode, TreeNode]] = [
        {node: parent for parent, (left, right) in tree.children.items()
         for node in (left, right)}
        for tree in trees
    ]

    for tree_idx, tree in enumerate(trees):
        # Populate the leaves with the initial arrays
        array = arrays[tree_idx]
        leaves = [frozenset([i]) for i in range(tree.N)]
        tree_active_nodes[tree_idx] = {n: a for n, a in zip(leaves, array)}

        for parent, (left, right) in tree.children.items():
            if left in tree_active_nodes[tree_idx] and right in tree_active_nodes[tree_idx]:
                signature = extract_signature(tree, parent)
                ready[signature].append((tree_idx, parent, left, right))

    # Contract until no more contractions are possible
    while any(ready.values()):
        signature = max(ready, key=lambda s: len(ready[s]))
        to_contract = ready[signature]
        ready.pop(signature)

        # Carry out the batched contraction
        eq, left_shape, right_shape = signature
        batched_eq = get_batched_einsum_equation(eq)

        print(f"Contracting with signature: {signature} using equation: {batched_eq} on a batch of {len(to_contract)} contractions")

        left_arrays = [tree_active_nodes[tree_idx][left] for tree_idx, _, left, _ in to_contract]
        right_arrays = [tree_active_nodes[tree_idx][right] for tree_idx, _, _, right in to_contract]

        left_array_stack = np.stack(left_arrays, axis=0)
        right_array_stack = np.stack(right_arrays, axis=0)

        result_stack = np.einsum(batched_eq, left_array_stack, right_array_stack)

        # Update the active nodes with the results
        for result, (tree_idx, parent, left, right) in zip(result_stack, to_contract):
            tree_active_nodes[tree_idx].pop(left)
            tree_active_nodes[tree_idx].pop(right)    
            tree_active_nodes[tree_idx][parent] = result

            # If their grandparent has both children active, add it to ready
            grandparent = tree_parent[tree_idx].get(parent)
            if grandparent is not None:
                left_child, right_child = trees[tree_idx].children[grandparent]
                if left_child in tree_active_nodes[tree_idx] and right_child in tree_active_nodes[tree_idx]:
                    signature = extract_signature(trees[tree_idx], grandparent)
                    ready[signature].append((tree_idx, grandparent, left_child, right_child))
    
    assert all(len(nodes) == 1 for nodes in tree_active_nodes), "Not all trees reduced to a single node"

    return [next(iter(nodes.values())) for nodes in tree_active_nodes]
