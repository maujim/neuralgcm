# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Numerical parity tests for the MLX JAXPR interpreter."""

import contextlib
import jax
import jax.numpy as jnp
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')

from neuralgcm import _mlx_jaxpr


@contextlib.contextmanager
def _threefry_partitionable(value):
    previous = jax.config.jax_threefry_partitionable
    jax.config.update('jax_threefry_partitionable', value)
    try:
        yield
    finally:
        jax.config.update('jax_threefry_partitionable', previous)


def _host(value):
    return np.asarray(value)


def _assert_equal(actual, expected):
    dtype = getattr(expected, 'dtype', None)
    if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
        expected = jax.random.key_data(expected)
    np.testing.assert_array_equal(_host(actual), np.asarray(expected))


def _assert_close(actual, expected, *, atol=1e-5, rtol=1e-5):
    np.testing.assert_allclose(_host(actual),
                               np.asarray(expected),
                               atol=atol,
                               rtol=rtol)


@pytest.mark.parametrize('partitionable', [False, True])
def test_threefry_key_operations_match_jax(partitionable):
    with _threefry_partitionable(partitionable):
        key = jax.random.PRNGKey(123)
        for count in (1, 2, 3):
            actual = _mlx_jaxpr.MLXFunction(
                lambda k: jax.random.split(k, count))(key)
            _assert_equal(actual, jax.random.split(key, count))
        for datum in (17, 0xffffffff):
            actual = _mlx_jaxpr.MLXFunction(
                lambda k, d: jax.random.fold_in(k, d))(key, jnp.uint32(datum))
            _assert_equal(actual, jax.random.fold_in(key, jnp.uint32(datum)))
        actual = _mlx_jaxpr.MLXFunction(
            lambda k: jax.random.bits(k, (8,), dtype=jnp.uint32))(key)
        _assert_equal(actual, jax.random.bits(key, (8,), dtype=jnp.uint32))


@pytest.mark.parametrize('partitionable', [False, True])
@pytest.mark.parametrize('shape', [(), (0,), (1,), (3,), (4,), (2, 3)])
def test_random_bits_shapes_match_jax(partitionable, shape):
    with _threefry_partitionable(partitionable):
        key = jax.random.PRNGKey(123)
        actual = _mlx_jaxpr.MLXFunction(
            lambda k: jax.random.bits(k, shape, dtype=jnp.uint32))(key)
        _assert_equal(actual, jax.random.bits(key, shape, dtype=jnp.uint32))


@pytest.mark.parametrize('partitionable', [False, True])
def test_typed_and_legacy_keys_match_jax(partitionable):
    with _threefry_partitionable(partitionable):
        keys = (jax.random.PRNGKey(71), jax.random.key(71))
        for key in keys:
            actual = _mlx_jaxpr.MLXFunction(lambda k: (
                jax.random.split(k, 3), jax.random.fold_in(k, 0xffffffff),
                jax.random.bits(k, (2, 3), dtype=jnp.uint32)))(key)
            expected = (jax.random.split(key, 3),
                        jax.random.fold_in(key, 0xffffffff),
                        jax.random.bits(key, (2, 3), dtype=jnp.uint32))
            for got, want in zip(actual, expected):
                _assert_equal(got, want)


@pytest.mark.parametrize('partitionable', [False, True])
def test_normal_randomness_matches_jax(partitionable):
    with _threefry_partitionable(partitionable):
        key = jax.random.PRNGKey(51)
        actual = _mlx_jaxpr.MLXFunction(lambda k: jax.random.normal(k, (64,)))(
            key)
        expected = jax.random.normal(key, (64,))
        _assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_gather_clip_and_fill_boundaries_match_jax():
    operand = jnp.array([10., 20., 30.])
    indices = jnp.array([-1, 0, 2, 3])
    clipped = _mlx_jaxpr.MLXFunction(lambda x, i: jnp.take(x, i, mode='clip'))(
        operand, indices)
    filled = _mlx_jaxpr.MLXFunction(
        lambda x, i: jnp.take(x, i, mode='fill', fill_value=-7.))(operand,
                                                                  indices)
    _assert_equal(clipped, jnp.take(operand, indices, mode='clip'))
    _assert_equal(filled, jnp.take(operand,
                                   indices,
                                   mode='fill',
                                   fill_value=-7.))


def test_scatter_drops_out_of_bounds_updates():
    indices = jnp.array([[-1], [1], [4]])
    updates = jnp.array([100., 7., 200.])
    actual = _mlx_jaxpr.MLXFunction(lambda x, i, u: x.at[i[:, 0]].set(u))(
        jnp.zeros((3,)), indices, updates)
    expected = jnp.zeros((3,)).at[indices[:, 0]].set(updates, mode='drop')
    _assert_equal(actual, expected)


def test_scatter_sliced_windows_match_jax():
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(0,),
        inserted_window_dims=(),
        scatter_dims_to_operand_dims=(0,),
    )
    operand = jnp.arange(5., dtype=jnp.float32)
    indices = jnp.array([[1], [3]], dtype=jnp.int32)
    updates = jnp.array([[10., 30.], [11., 31.]], dtype=jnp.float32)
    actual = _mlx_jaxpr.MLXFunction(
        lambda x, i, u: jax.lax.scatter(x, i, u, dimension_numbers))(operand,
                                                                     indices,
                                                                     updates)
    expected = jax.lax.scatter(operand, indices, updates, dimension_numbers)
    _assert_equal(actual, expected)


@pytest.mark.parametrize('mode', ['clip', 'drop'])
def test_scatter_sliced_windows_out_of_bounds_match_jax(mode):
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(0,),
        inserted_window_dims=(),
        scatter_dims_to_operand_dims=(0,),
    )
    operand = jnp.zeros((5,), dtype=jnp.float32)
    indices = jnp.array([[-1], [1], [5]], dtype=jnp.int32)
    updates = jnp.array([[10., 20., 50.], [20., 21., 51.]], dtype=jnp.float32)
    actual = _mlx_jaxpr.MLXFunction(
        lambda x, i, u: jax.lax.scatter(x, i, u, dimension_numbers, mode=mode))(
            operand, indices, updates)
    expected = jax.lax.scatter(operand,
                               indices,
                               updates,
                               dimension_numbers,
                               mode=mode)
    _assert_equal(actual, expected)


def test_scatter_multidimensional_starts_and_overlapping_windows_match_jax():
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(0,),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1),
    )
    operand = jnp.zeros((2, 3, 4), dtype=jnp.float32)
    indices = jnp.array([[0, 1], [0, 1], [1, 0]], dtype=jnp.int32)
    # The first two indexed starts target the same two-element window. Keeping
    # those updates equal makes the set result unambiguous across backends.
    updates = jnp.array([[5., 5., 8.], [6., 6., 9.]], dtype=jnp.float32)
    actual = _mlx_jaxpr.MLXFunction(
        lambda x, i, u: jax.lax.scatter(x, i, u, dimension_numbers))(operand,
                                                                     indices,
                                                                     updates)
    expected = jax.lax.scatter(operand, indices, updates, dimension_numbers)
    _assert_equal(actual, expected)


def test_gather_sliced_windows_match_jax():
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
    )
    operand = jnp.arange(20., dtype=jnp.float32).reshape((5, 4))
    indices = jnp.array([[0], [2], [4]], dtype=jnp.int32)
    actual = _mlx_jaxpr.MLXFunction(lambda x, i: jax.lax.gather(
        x, i, dimension_numbers, slice_sizes=(1, 2), mode='clip'))(operand,
                                                                   indices)
    expected = jax.lax.gather(operand,
                              indices,
                              dimension_numbers,
                              slice_sizes=(1, 2),
                              mode='clip')
    _assert_equal(actual, expected)


def test_integer_division_truncation_and_wrap_match_jax():
    lhs = jnp.array([-7, 7, -8, 8], dtype=jnp.int32)
    rhs = jnp.array([3, -3, 3, -3], dtype=jnp.int32)
    actual = _mlx_jaxpr.MLXFunction(
        lambda x, y: (jax.lax.div(x, y), jax.lax.rem(x, y)))(lhs, rhs)
    expected = (jax.lax.div(lhs, rhs), jax.lax.rem(lhs, rhs))
    for got, want in zip(actual, expected):
        _assert_equal(got, want)

    high = jnp.array([np.iinfo(np.int32).max], dtype=jnp.int32)
    wrapped = _mlx_jaxpr.MLXFunction(lambda x: x + jnp.int32(1))(high)
    _assert_equal(wrapped, high + jnp.int32(1))


def test_mlx_function_output_can_feed_next_call():
    transition = _mlx_jaxpr.MLXFunction(lambda x: x * 2 + 1)
    initial = jnp.array([1., -2., 4.], dtype=jnp.float32)
    first = transition(initial)
    second = transition(first)
    _assert_equal(second, (initial * 2 + 1) * 2 + 1)


def test_scan_carry_outputs_reverse_and_zero_length():

    def scan(xs, initial, reverse):
        return jax.lax.scan(
            lambda carry, x: (carry + x, carry * 2 + x),
            initial,
            xs,
            reverse=reverse,
        )

    xs = jnp.arange(1, 6, dtype=jnp.float32)
    for reverse in (False, True):
        actual = _mlx_jaxpr.MLXFunction(lambda x: scan(x, 3., reverse))(xs)
        expected = scan(xs, 3., reverse)
        for got, want in zip(actual, expected):
            _assert_equal(got, want)
    empty = jnp.empty((0,), dtype=jnp.float32)
    actual_empty = _mlx_jaxpr.MLXFunction(
        lambda x: jax.lax.scan(lambda c, y: (c + y, y), 4., x))(empty)
    expected_empty = jax.lax.scan(lambda c, y: (c + y, y), 4., empty)
    for got, want in zip(actual_empty, expected_empty):
        _assert_equal(got, want)


def test_cached_callable_uses_new_values_not_traced_constants():
    compiled = _mlx_jaxpr.MLXFunction(lambda x, bias: x * 3 + bias)
    _assert_equal(compiled(jnp.array([1., 2.]), jnp.array([4., 5.])), [7., 11.])
    _assert_equal(compiled(jnp.array([8., -2.]), jnp.array([1., 9.])),
                  [25., 3.])


def test_dot_general_batched_and_contracting_dimensions():
    lhs = jnp.arange(2 * 3 * 4., dtype=jnp.float32).reshape(2, 3, 4)
    rhs = jnp.arange(2 * 4 * 5., dtype=jnp.float32).reshape(2, 4, 5)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    actual = _mlx_jaxpr.MLXFunction(
        lambda x, y: jax.lax.dot_general(x, y, dimension_numbers))(lhs, rhs)
    expected = jax.lax.dot_general(lhs, rhs, dimension_numbers)
    _assert_close(actual, expected)
