from typing import Optional
from collections import defaultdict
from cotengra import einsum_tree
from cotengra.contract import extract_contractions
from cotengra.contract import einsum as ctg_einsum
from autoray import do
from functools import lru_cache

def get_result_shape(eq, shapes):
    inputs, output = eq.split('->')
    terms = inputs.split(',')
    if len(terms) != len(shapes):
        raise ValueError(f"Number of terms {len(terms)} does not match number of shapes {len(shapes)}")

    size_dict = {}
    for term, shape in zip(terms, shapes):
        if len(shape) != len(term):
            raise ValueError(f"Shape {shape} does not match term {term}")
        for c, size in zip(term, shape):
            if c not in size_dict:
                size_dict[c] = size
            elif size_dict[c] != size:
                raise ValueError(f"Size mismatch for character {c}: {size_dict[c]} vs {size}")
    return [size_dict[c] for c in output]

def get_batched_einsum_equation(original_eq: str, batch_symbol: Optional[str] = None) -> str:
    """
    Convert an einsum equation to a batched version by prepending a batch symbol.
    This function modifies the equation to allow for batched operations.
    
    Parameters
    ----------
    original_eq : str
        The original einsum equation string, e.g., "abc,cd->abd".
    batch_symbol : optional str
        The symbol to use for the batch dimension.
        If not given, it will be automatically generated as the next unicode character
        after the highest character in the original equation.

    Returns
    -------
    str
        The modified einsum equation string with a batch symbol prepended,
        e.g., "eabc,ecd->eabd".
        where 'e' is the next character after the highest character in the original equation.
    """
    if batch_symbol is None:
        batch_symbol = chr(max(ord(c) for c in original_eq) + 1)

    left_side, right_side = original_eq.split('->')
    terms = left_side.split(',')

    batched_terms = [f'{batch_symbol}{term}' for term in terms]
    batched_output = f'{batch_symbol}{right_side}'

    return f"{','.join(batched_terms)}->{batched_output}"

def get_batched_contractions(contractions_batch, shapes_batch):
    """
    Generate a list of batched contractions given a collection of contractions from `extract_contractions`.

    Arguments:
    - contractions_batch: A list of lists, where each inner list contains tuples of the form
      (parent, left, right, eq) representing the contractions.
    """

    node2shape_batch = [{frozenset([n]): arr for n, arr in enumerate(shapes)} for shapes in shapes_batch]
    
    node2children_batch = []
    node2parent_batch = []
    node2eq = []
    
    for contractions in contractions_batch:
        children_map = {}
        parent_map = {}
        eq_map = {}
        
        for parent, left, right, eq in contractions:
            if left is not None and right is not None:
                children_map[parent] = (left, right)
                parent_map[left] = parent
                parent_map[right] = parent
                eq_map[parent] = eq

        node2children_batch.append(children_map)
        node2parent_batch.append(parent_map)
        node2eq.append(eq_map)

    contractible = defaultdict(list)
    for i, (contractions, node2shape) in enumerate(zip(contractions_batch, node2shape_batch)):
        for parent, left, right, eq in contractions:
            if left in node2shape and right in node2shape:
                shape_left = node2shape[left]
                shape_right = node2shape[right]
                key = (eq, tuple(shape_left), tuple(shape_right))
                target = (i, parent, left, right)
                contractible[key].append(target)

    batched_contractions = []
    while contractible:
        key = max(contractible, key=lambda k: len(contractible[k]))
        targets = contractible.pop(key)
        eq, shape_left, shape_right = key

        batched_contractions.append((eq, targets))
        result_shape = get_result_shape(eq, [shape_left, shape_right])

        for i, parent, left, right in targets:
            node2shape_batch[i].pop(left)
            node2shape_batch[i].pop(right)
            node2shape_batch[i][parent] = result_shape

            # Check if this parent can now be contracted with its sibling
            grandparent = node2parent_batch[i].get(parent, None)
            if grandparent is not None:
                grand_left, grand_right = node2children_batch[i][grandparent]
                if grand_left in node2shape_batch[i] and grand_right in node2shape_batch[i]:
                    # Get shapes in the correct order according to original contraction
                    shape_left = node2shape_batch[i][grand_left]
                    shape_right = node2shape_batch[i][grand_right]
                    eq = node2eq[i][grandparent]
                    key = (eq, tuple(shape_left), tuple(shape_right))
                    target = (i, grandparent, grand_left, grand_right)
                    contractible[key].append(target)

    return batched_contractions

def _batched_einsum(eq, targets, node2array_batch):
    """
    Perform batched binary contraction for a given equation.
    """
    batched_eq = get_batched_einsum_equation(eq)

    arrays_left, arrays_right = [], []
    for network_idx, parent, left, right in targets:
        arrays_left.append(node2array_batch[network_idx].pop(left))
        arrays_right.append(node2array_batch[network_idx].pop(right))

    stack_left = do("stack", arrays_left)
    stack_right = do("stack", arrays_right)

    result = ctg_einsum(batched_eq, stack_left, stack_right)
    for i, (network_idx, parent, left, right) in enumerate(targets):
        node2array_batch[network_idx][parent] = result[i]

def batch_contract(contractions_batch, arrays_batch):
    """
    Perform batched contractions for a batch of contractions and their corresponding arrays.
    
    Parameters
    ----------
    contractions_batch : list of lists
        Each inner list contains tuples of the form (parent, left, right, eq).
    arrays_batch : list of list of arrays
        Each inner list contains arrays corresponding to the contractions.

    Returns
    -------
    list of dicts
        Each dict contains the updated arrays after performing the batched contractions.
    """
    shapes_batch = [[array.shape for array in arrays] for arrays in arrays_batch]
    batched_contractions = get_batched_contractions(contractions_batch, shapes_batch)

    node2array_batch = [{frozenset([n]): arr for n, arr in enumerate(arrays)} for arrays in arrays_batch]

    for eq, targets in batched_contractions:
        _batched_einsum(eq, targets, node2array_batch)

    return [next(iter(node2array.values())) for node2array in node2array_batch]

@lru_cache(maxsize=None)
def get_contractions(eq, *shapes):
    tree = einsum_tree(eq, *shapes)
    return [(c[0], c[1], c[2], c[4]) for c in extract_contractions(tree, prefer_einsum=True)]

def batch_einsum(eq_batch, arrays_batch):
    """
    Perform batched einsum for a list of equations and their corresponding arrays.
    
    Parameters
    ----------
    eqs : list of str
        List of einsum equations, e.g., ["ij,jk->ik", "ab,bc->ac"].
    arrays : list of list of arrays
        Each inner list contains arrays corresponding to the equations.

    Returns
    -------
    list of dicts
        Each dict contains the updated arrays after performing the batched einsum.
    """
    shapes_batch = [[array.shape for array in arrays] for arrays in arrays_batch]
    contractions_batch = [get_contractions(eq, *shapes) for eq, shapes in zip(eq_batch, shapes_batch)]

    return batch_contract(contractions_batch, arrays_batch)
