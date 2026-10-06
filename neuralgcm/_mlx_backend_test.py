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
"""Numerical regressions for model-shaped MLX interpreter computations."""

import jax
from jax._src.lax import lax as lax_internal
import jax.numpy as jnp
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')

from neuralgcm import _mlx_jaxpr


@pytest.fixture(autouse=True)
def _isolated_jax_config():
    previous_x64 = jax.config.jax_enable_x64
    previous_partitionable = jax.config.jax_threefry_partitionable
    previous_prng_impl = jax.config.jax_default_prng_impl
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_threefry_partitionable', True)
    jax.config.update('jax_default_prng_impl', 'threefry2x32')
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous_x64)
        jax.config.update('jax_threefry_partitionable', previous_partitionable)
        jax.config.update('jax_default_prng_impl', previous_prng_impl)


def _assert_tree_equal(actual, expected):
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    assert actual_tree == expected_tree
    for got, want in zip(actual_leaves, expected_leaves):
        assert isinstance(got, mx.array)
        if jax.dtypes.issubdtype(want.dtype, jax.dtypes.prng_key):
            want = jax.random.key_data(want)
        got_host, want_host = np.asarray(got), np.asarray(want)
        assert got_host.dtype == want_host.dtype
        np.testing.assert_array_equal(got_host, want_host)
        assert got_host.shape == want_host.shape


def test_host_float64_inputs_use_jax_float32_canonicalization():
    # Rounding must precede arithmetic: these doubles straddle float32 ULPs.
    values = np.array([16777217., -16777217., 1. + 2.**-25, 2.**-140],
                      dtype=np.float64)

    def computation(x):
        return {
            'input': x,
            'residual': x - jnp.float32(16777216.),
            'scaled': x * jnp.float32(0.5)
        }

    actual = _mlx_jaxpr.MLXFunction(computation)(values)
    expected = computation(jnp.asarray(values))
    _assert_tree_equal(actual, expected)


def test_device_resident_pytree_recurrence_matches_jax():
    weights = jnp.array(
        [[0.5, -0.25, 0.125], [0.25, 0.5, -0.125], [-0.125, 0.25, 0.5]],
        dtype=jnp.float32)

    def transition(state, forcing):
        updated = jnp.tanh(state['field'] @ weights + forcing)
        return {
            'field': updated,
            'integral': state['integral'] + updated,
            'step': state['step'] + jnp.int32(1)
        }

    compiled = _mlx_jaxpr.MLXFunction(transition)
    initial = {
        'field': jnp.array([[0.25, -0.5, 0.75], [-0.25, 0.5, -0.75]]),
        'integral': jnp.zeros((2, 3), dtype=jnp.float32),
        'step': jnp.int32(0)
    }
    actual = initial
    expected = initial
    for step in range(8):
        forcing = jnp.full((2, 3), (step - 3) / 32., dtype=jnp.float32)
        actual = compiled(actual, mx.array(np.asarray(forcing)))
        expected = transition(expected, forcing)
        for leaf in jax.tree_util.tree_leaves(actual):
            assert isinstance(leaf, mx.array)
        for got, want in zip(jax.tree_util.tree_leaves(actual),
                             jax.tree_util.tree_leaves(expected)):
            np.testing.assert_allclose(np.asarray(got),
                                       np.asarray(want),
                                       atol=1e-6,
                                       rtol=1e-6)
        np.testing.assert_array_equal(np.asarray(actual['step']),
                                      np.asarray(expected['step']))


@pytest.mark.parametrize('partitionable', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('shape', [(3, 2), (3, 0), (0, 2)])
def test_nested_scans_captured_constants_and_typed_keys(partitionable, reverse,
                                                        shape):
    jax.config.update('jax_threefry_partitionable', partitionable)
    coefficients = jnp.array([0.5, -0.25, 0.125], dtype=jnp.float32)

    def nested(key, initial, inputs, bias):

        def outer(carry, row):

            def inner(state, value):
                inner_key, field = state
                next_key, sample_key = jax.random.split(inner_key)
                noise = jax.random.bits(sample_key, (3,), dtype=jnp.uint32)
                increment = (noise & jnp.uint32(7)).astype(jnp.float32)
                field = field + value * coefficients + bias + increment
                return (next_key, field), {'field': field, 'key': sample_key}

            updated, history = jax.lax.scan(inner, carry, row, reverse=reverse)
            return updated, history

        return jax.lax.scan(outer, (key, initial), inputs, reverse=reverse)

    key = jax.random.key(123)
    initial = jnp.array([1., -2., 3.], dtype=jnp.float32)
    inputs = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    bias = jnp.array([0.25, 0.5, -0.25], dtype=jnp.float32)
    actual = _mlx_jaxpr.MLXFunction(nested)(key, initial, inputs, bias)
    expected = nested(key, initial, inputs, bias)
    _assert_tree_equal(actual, expected)


@pytest.mark.parametrize('partitionable', [False, True])
@pytest.mark.parametrize('typed', [False, True])
@pytest.mark.parametrize('key_prefix', [(10,), (2, 5)])
def test_batched_random_split_and_fold_in(partitionable, typed, key_prefix):
    jax.config.update('jax_threefry_partitionable', partitionable)
    root = jax.random.key(123) if typed else jax.random.PRNGKey(123)
    keys = jax.random.split(root, 10)
    # Include high-bit data to catch signed conversion or broadcasting mistakes.
    data = jnp.arange(10, dtype=jnp.uint32) * jnp.uint32(0x80000001)

    def per_key(key, datum):
        return {
            'split_two': jax.random.split(key, 2),
            'split_three': jax.random.split(key, 3),
            'folded': jax.random.fold_in(key, datum),
            'folded_scalar': jax.random.fold_in(key, jnp.uint32(0xFFFFFFFF)),
        }

    def computation(key_batch, fold_data):
        physical_suffix = () if typed else (2,)
        key_batch = key_batch.reshape(key_prefix + physical_suffix)
        fold_data = fold_data.reshape(key_prefix)
        mapped = jax.vmap(per_key)
        if len(key_prefix) == 2:
            mapped = jax.vmap(mapped)
        return mapped(key_batch, fold_data)

    expected = computation(keys, data)
    actual = _mlx_jaxpr.MLXFunction(computation)(keys, data)
    _assert_tree_equal(actual, expected)
    for name, suffix in (('split_two', (2, 2)), ('split_three', (3, 2)),
                         ('folded', (2,)), ('folded_scalar', (2,))):
        assert actual[name].shape == key_prefix + suffix
        assert np.asarray(actual[name]).dtype == np.dtype(np.uint32)


@pytest.mark.parametrize('partitionable', [False, True])
@pytest.mark.parametrize('typed', [False, True])
@pytest.mark.parametrize('key_prefix', [(10,), (2, 5)])
@pytest.mark.parametrize('shape', [(5,), (6,), (3, 5), (128, 65)])
@pytest.mark.parametrize('dtype', [jnp.uint8, jnp.uint16, jnp.uint32])
def test_batched_random_bits(partitionable, typed, key_prefix, shape, dtype):
    jax.config.update('jax_threefry_partitionable', partitionable)
    root = jax.random.key(456) if typed else jax.random.PRNGKey(456)
    keys = jax.random.split(root, 10)

    def computation(key_batch):
        physical_suffix = () if typed else (2,)
        key_batch = key_batch.reshape(key_prefix + physical_suffix)
        mapped = jax.vmap(lambda key: jax.random.bits(key, shape, dtype=dtype))
        if len(key_prefix) == 2:
            mapped = jax.vmap(mapped)
        return mapped(key_batch)

    expected = computation(keys)
    actual = _mlx_jaxpr.MLXFunction(computation)(keys)
    assert actual.shape == key_prefix + shape
    _assert_tree_equal(actual, expected)


@pytest.mark.parametrize('partitionable', [False, True])
def test_vmapped_typed_key_broadcast_and_input_reshape(partitionable):
    jax.config.update('jax_threefry_partitionable', partitionable)
    keys = jax.random.split(jax.random.key(789), 10)
    data = jnp.arange(6, dtype=jnp.uint32)

    def per_key(key, fold_data):
        # Reshape and broadcast live typed-key inputs, not captured constants.
        expanded = jnp.broadcast_to(key.reshape((1, 1)), (2, 3))
        flattened = expanded.reshape((6,))
        folded = jax.vmap(jax.random.fold_in)(flattened, fold_data)
        return {
            'expanded':
                expanded,
            'flattened':
                flattened,
            'folded':
                folded,
            'bits':
                jax.vmap(lambda k: jax.random.bits(k, (5,), dtype=jnp.uint32))
                (folded),
        }

    def computation(key_batch, fold_data):
        key_batch = key_batch.reshape((2, 5))
        return jax.vmap(jax.vmap(per_key, in_axes=(0, None)),
                        in_axes=(0, None))(key_batch, fold_data)

    expected = computation(keys, data)
    actual = _mlx_jaxpr.MLXFunction(computation)(keys, data)
    _assert_tree_equal(actual, expected)
    assert actual['expanded'].shape == (2, 5, 2, 3, 2)
    assert actual['flattened'].shape == (2, 5, 6, 2)
    assert actual['folded'].shape == (2, 5, 6, 2)
    assert actual['bits'].shape == (2, 5, 6, 5)


@pytest.mark.parametrize('stride,padding,dilation,groups', [
    (1, ((0, 0),), 1, 1),
    (2, ((1, 3),), 1, 1),
    (1, ((2, 1),), 2, 1),
    (2, ((3, 0),), 2, 2),
])
def test_conv1d_ncw_wio_channels_padding_and_dilation(stride, padding, dilation,
                                                      groups):
    # Distinct N/C/W extents and non-square channels expose axis confusion.
    lhs = (jnp.arange(2 * 4 * 11, dtype=jnp.float32).reshape(2, 4, 11) % 17 -
           8) / 8
    rhs = (jnp.arange(3 * (4 // groups) * 6, dtype=jnp.float32).reshape(
        3, 4 // groups, 6) % 13 - 6) / 8

    def convolution(x, kernel):
        return jax.lax.conv_general_dilated(x,
                                            kernel,
                                            window_strides=(stride,),
                                            padding=padding,
                                            rhs_dilation=(dilation,),
                                            dimension_numbers=('NCW', 'WIO',
                                                               'NCW'),
                                            feature_group_count=groups)

    actual = _mlx_jaxpr.MLXFunction(convolution)(lhs, rhs)
    expected = convolution(lhs, rhs)
    # Dyadic inputs keep these small convolution sums exactly representable.
    _assert_tree_equal(actual, expected)


def test_nextafter_float32_special_values_match_jax_bits():
    smallest = np.array([1], dtype=np.uint32).view(np.float32)[0]
    lhs = np.array([
        0., -0., 0., -0., 0., -0., smallest, -smallest, np.inf, -np.inf, np.inf,
        -np.inf, 1., -1., np.nan, 2.
    ],
                   dtype=np.float32)
    rhs = np.array([
        1., 1., -1., -1., -0., 0., 0., -0., 0., 0., np.inf, -np.inf, np.inf,
        -np.inf, 0., np.nan
    ],
                   dtype=np.float32)
    actual = _mlx_jaxpr.MLXFunction(jnp.nextafter)(lhs, rhs)
    expected = np.asarray(jnp.nextafter(jnp.asarray(lhs), jnp.asarray(rhs)))
    actual_host = np.asarray(actual)
    np.testing.assert_array_equal(np.isnan(actual_host), np.isnan(expected))
    finite_or_inf = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual_host.view(np.uint32)[finite_or_inf],
        expected.view(np.uint32)[finite_or_inf])


@pytest.mark.parametrize('inclusive', [False, True])
def test_total_order_float32_signed_zero_infinities_and_nan_payloads(inclusive):
    # Bind the JAX total-order primitive: ordinary lax.lt/le use partial order.
    primitive = lax_internal.le_to_p if inclusive else lax_internal.lt_to_p
    bits = np.array([
        0xffc00002, 0xffc00001, 0xff800000, 0xbf800000, 0x80000001, 0x80000000,
        0, 1, 0x3f800000, 0x7f800000, 0x7fc00001, 0x7fc00002
    ],
                    dtype=np.uint32)
    values = bits.view(np.float32)
    lhs = jnp.asarray(
        np.broadcast_to(values[:, None], (len(values), len(values))))
    rhs = jnp.asarray(
        np.broadcast_to(values[None, :], (len(values), len(values))))
    actual = _mlx_jaxpr.MLXFunction(lambda x, y: primitive.bind(x, y))(lhs, rhs)
    _assert_tree_equal(actual, primitive.bind(lhs, rhs))


@pytest.mark.parametrize('dtype', [np.int32, np.uint32])
def test_integer_div_rem_precision_extremes(dtype):
    limits = np.iinfo(dtype)
    if dtype == np.int32:
        lhs = [
            limits.min, limits.min, limits.min + 1, limits.max, limits.max,
            16777217, -16777217, -1073741825, 1073741825, 0
        ]
        rhs = [-1, 3, -7, 3, -7, 3, 3, -17, -17, 5]
    else:
        lhs = [
            limits.max, limits.max - 1, 2147483649, 2147483647, 16777217,
            33554431, limits.max, 1, 0
        ]
        rhs = [3, 7, 17, 65537, 3, 5, 2147483649, limits.max, 7]
    lhs, rhs = jnp.asarray(np.array(lhs, dtype=dtype)), jnp.asarray(
        np.array(rhs, dtype=dtype))

    def divide_and_remainder(x, y):
        return jax.lax.div(x, y), jax.lax.rem(x, y)

    actual = _mlx_jaxpr.MLXFunction(divide_and_remainder)(lhs, rhs)
    _assert_tree_equal(actual, divide_and_remainder(lhs, rhs))


@pytest.mark.parametrize('shape,new_sizes,dimensions', [
    ((3, 5), (15,), (1, 0)),
    ((3, 5), (5, 3), (1, 0)),
    ((2, 3, 5), (5, 6), (2, 0, 1)),
])
def test_reshape_permuted_dimensions_match_jax(shape, new_sizes, dimensions):
    operand = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)

    def permuted_reshape(x):
        return jax.lax.reshape(x, new_sizes, dimensions=dimensions)

    actual = _mlx_jaxpr.MLXFunction(permuted_reshape)(operand)
    _assert_tree_equal(actual, permuted_reshape(operand))


def test_select_n_integer_three_way_selector_matches_jax():
    selector = jnp.array([[0, 1, 2], [2, 0, 1]], dtype=jnp.int32)
    first = jnp.array([[11, 12, 13], [14, 15, 16]], dtype=jnp.int32)
    second = jnp.array([[-21, -22, -23], [-24, -25, -26]], dtype=jnp.int32)
    third = jnp.array([[31, 32, 33], [34, 35, 36]], dtype=jnp.int32)

    def select(which, a, b, c):
        return jax.lax.select_n(which, a, b, c)

    actual = _mlx_jaxpr.MLXFunction(select)(selector, first, second, third)
    _assert_tree_equal(actual, select(selector, first, second, third))


def _assert_float32_close_with_special_values(actual, expected):
    assert isinstance(actual, mx.array)
    got, want = np.asarray(actual), np.asarray(expected)
    assert got.dtype == want.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_array_equal(np.isposinf(got), np.isposinf(want))
    np.testing.assert_array_equal(np.isneginf(got), np.isneginf(want))
    np.testing.assert_allclose(got, want, atol=0, rtol=1e-6, equal_nan=True)
    # allclose cannot distinguish the IEEE sign of zero.
    zero = (want == 0)
    np.testing.assert_array_equal(np.signbit(got[zero]), np.signbit(want[zero]))


def test_float32_division_and_fmod_nonintegral_and_ieee_edges():
    lhs = jnp.array([
        7.5, -7.5, 7.5, -7.5, 1e-30, -1e-30, 1e30, -1e30, 0., -0., 1., -1., 0.,
        -0., np.inf, -np.inf, 3.5, np.nan
    ],
                    dtype=jnp.float32)
    rhs = jnp.array([
        2., 2., -2., -2., 3e-5, 3e-5, 3e5, 3e5, -2., 2., 0., -0., 0., -0., 2.,
        -2., np.inf, 2.
    ],
                    dtype=jnp.float32)

    def divide_and_fmod(x, y):
        return jax.lax.div(x, y), jnp.fmod(x, y)

    actual = _mlx_jaxpr.MLXFunction(divide_and_fmod)(lhs, rhs)
    expected = divide_and_fmod(lhs, rhs)
    for got, want in zip(actual, expected):
        _assert_float32_close_with_special_values(got, want)


def test_float32_pressure_field_divided_by_scalar_gravity():
    pressure = (jnp.arange(128 * 64, dtype=jnp.float32).reshape(1, 128, 64) *
                jnp.float32(3.125) + jnp.float32(1000.))
    gravity = jnp.float32(9.80665)
    actual = _mlx_jaxpr.MLXFunction(jax.lax.div)(pressure, gravity)
    expected = jax.lax.div(pressure, gravity)
    _assert_float32_close_with_special_values(actual, expected)


def test_complex64_division_preserves_real_and_imaginary_components():
    lhs = jnp.array(
        [3. + 4.j, -7.5 + 2.25j, 0.125 - 0.5j, 1e-12 + 2e-12j, 1e12 - 3e12j],
        dtype=jnp.complex64)
    rhs = jnp.array([1. - 2.j, -2. + 3.j, 0.5 + 0.25j, 2. + 1.j, -3. + 2.j],
                    dtype=jnp.complex64)
    actual = _mlx_jaxpr.MLXFunction(jax.lax.div)(lhs, rhs)
    expected = jax.lax.div(lhs, rhs)
    assert isinstance(actual, mx.array)
    got, want = np.asarray(actual), np.asarray(expected)
    assert got.dtype == want.dtype == np.dtype(np.complex64)
    np.testing.assert_allclose(got, want, atol=0, rtol=1e-6)
