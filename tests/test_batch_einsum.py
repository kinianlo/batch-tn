from batch_tn import batch_einsum
import numpy as np

def test_batch_einsum():
    eq = 'ij,jk->ik'
    size_dict = {'i': 2, 'j': 3, 'k': 4}

    shapes = [tuple(size_dict[t] for t in term) for term in eq.split('->')[0].split(',')]
    arrays = [np.random.rand(*shape) for shape in shapes]

    results = batch_einsum([eq]*10, [arrays]*10)
    # check that the results are all the same 
    assert all(np.allclose(results[0], res) for res in results)

    # check that the result matches with the numpy einsum
    expected = np.einsum(eq, *arrays)
    assert np.allclose(results[0], expected)
