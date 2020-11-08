from itertools import product

import tensorflow as tf
import numpy as np
import pytest

from random_matrix_sampler import GaussianOrthogonalRandomMatrix
from performer import Performer


@pytest.mark.parametrize('rows, columns', product([1, 10, 20], [1, 10, 20]))
def test_gaussian_orthogonal_random_matrix_has_correct_shape(rows, columns):
    sampler = GaussianOrthogonalRandomMatrix(rows, columns, scaling=0)
    out = sampler.get_2d_array()
    assert out.shape == (rows, columns)


@pytest.mark.parametrize('shape, scaling', product([2, 4, 100], [0, 1]))
def test_gaussian_orthogonal_random_matrix_off_diags_are_zeros(shape, scaling):
    rows, columns, scaling = shape, shape, scaling
    sampler = GaussianOrthogonalRandomMatrix(rows, columns, scaling)
    out = sampler.get_2d_array()
    out = out @ out.T
    out = out - np.diag(np.diag(out))
    assert np.allclose(out, np.zeros(out.shape))


def test_gaussian_orthogonal_random_matrix_raises_on_invalid_scaling_factor():
    with pytest.raises(AssertionError) as e:
        GaussianOrthogonalRandomMatrix(10, 10, scaling=0.1)
    assert "Scaling must be one of {0, 1}" in str(e)


def test_performer_compute_attention_gets_correct_output_shape():
    layer = Performer(attention_method='quadratic', num_heads=3, key_dim=2)
    query = tf.random.uniform(shape=[1, 18, 16], dtype='float32')
    value = tf.random.uniform(shape=[1, 4, 16], dtype='float32')
    output_tensor, weights = layer(query, value, return_attention_scores=True)
    assert all(np.array(output_tensor.shape) == [1, 18, 16])
    assert all(np.array(weights.shape) == [1, 3, 18, 4])
