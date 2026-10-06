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
"""Numerical integration tests using the bundled stochastic TL63 checkpoint."""

import contextlib
import dataclasses
import threading

import pytest

mx = pytest.importorskip('mlx.core')

import jax
from neuralgcm import demo
from neuralgcm import mlx
from neuralgcm.legacy import api
import numpy as np

pytestmark = pytest.mark.integration

# The CPU reference and MLX interpreter must use the same random algorithm,
# independent of collection-time configuration changes in legacy/api_test.py.
_EXECUTION_LOCK = threading.RLock()
# Bundled TL63 seed-0 float32 parity measured through two advances:
# state maxima (steps 0/1/2): 2.341e-6 / 7.831e-5 / 6.895e-5;
# weather maxima: 3.843e-5 / 2.066e-5 / 3.631e-5. Unroll agrees
# with the corresponding weather trajectory. Allow at least 2.5x the measured
# stage maxima for backend variation, rather than an uncalibrated global bound.
_MAX_NORMALIZED_ERROR = {
    'state': 2e-4,
    'weather': 1e-4,
    'unroll': 1e-4,
}
# Complementary per-field RMS criterion: an initial float32 backend margin,
# not a measured RMS envelope. Keep the recorded values for future calibration.
_MAX_NORMALIZED_RMS_ERROR = 2e-4


@contextlib.contextmanager
def _execution_context():
    with _EXECUTION_LOCK:
        partitionable = jax.config.jax_threefry_partitionable
        x64 = jax.config.jax_enable_x64
        try:
            jax.config.update('jax_threefry_partitionable', False)
            jax.config.update('jax_enable_x64', False)
            with jax.default_device(jax.devices('cpu')[0]):
                yield
        finally:
            jax.config.update('jax_enable_x64', x64)
            jax.config.update('jax_threefry_partitionable', partitionable)


def _ready(tree):
    """Finish each backend operation before starting work on the other backend."""
    mlx_leaves = []
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, mx.array):
            mlx_leaves.append(leaf)
        elif isinstance(leaf, jax.Array):
            leaf.block_until_ready()
    if mlx_leaves:
        mx.eval(*mlx_leaves)
    return tree


def _snapshot(tree):
    _ready(tree)

    def copy_leaf(value):
        array = np.array(value, copy=True)
        assert np.isfinite(array).all(), 'snapshot contains non-finite values'
        return array

    return jax.tree.map(copy_leaf, tree)


def _assert_tree_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for (path, actual_leaf), expected_leaf in zip(
            jax.tree_util.tree_flatten_with_path(actual)[0],
            jax.tree.leaves(expected),
            strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            err_msg=jax.tree_util.keystr(path),
        )


def _assert_parity(actual, expected, *, stage, record_property):
    """Check all numerical leaves, including exact RNG keys and step counters."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected), stage
    _ready(actual)
    errors = []
    rms_errors = []
    max_error_bound = _MAX_NORMALIZED_ERROR[stage.split('_', 1)[0]]
    for (path, actual_leaf), expected_leaf in zip(
            jax.tree_util.tree_flatten_with_path(actual)[0],
            jax.tree.leaves(expected),
            strict=True,
    ):
        name = f'{stage}{jax.tree_util.keystr(path)}'
        assert isinstance(actual_leaf, mx.array), name
        actual_array = np.asarray(actual_leaf)
        expected_array = np.asarray(expected_leaf)
        assert actual_array.shape == expected_array.shape, name
        assert actual_array.dtype == expected_array.dtype, name
        assert np.isfinite(actual_array).all(), name
        assert np.isfinite(expected_array).all(), name
        if np.issubdtype(expected_array.dtype, np.inexact):
            assert expected_array.dtype in (np.dtype('float32'),
                                            np.dtype('complex64'))
            absolute_difference = np.abs(actual_array - expected_array)
            reference_magnitude = np.abs(expected_array)
            scale = max(float(np.max(reference_magnitude, initial=0)), 1e-10)
            error = float(np.max(absolute_difference, initial=0)) / scale
            reference_rms = float(
                np.sqrt(
                    np.mean(np.square(reference_magnitude, dtype=np.float64))))
            difference_rms = float(
                np.sqrt(
                    np.mean(np.square(absolute_difference, dtype=np.float64))))
            rms_scale = max(reference_rms, 1e-10)
            rms_error = difference_rms / rms_scale
            errors.append(error)
            rms_errors.append(rms_error)
            record_property(f'{name}_normalized_max_error', error)
            record_property(f'{name}_normalized_rms_error', rms_error)
            assert error < max_error_bound, (
                f'{name}: normalized max error {error:.8g} '
                f'>= {max_error_bound}; scale={scale:.8g}')
            assert rms_error < _MAX_NORMALIZED_RMS_ERROR, (
                f'{name}: normalized RMS error {rms_error:.8g} '
                f'>= {_MAX_NORMALIZED_RMS_ERROR}; RMS scale={rms_scale:.8g}')
        else:
            np.testing.assert_array_equal(actual_array,
                                          expected_array,
                                          err_msg=name)
    assert errors, f'{stage}: no floating-point leaves compared'
    record_property(f'{stage}_normalized_max_error', max(errors))
    record_property(f'{stage}_normalized_rms_error', max(rms_errors))


@dataclasses.dataclass
class _BundledModel:
    model: mlx.PressureLevelModel
    inputs: dict
    forcings: dict
    temporal_forcings: dict
    initial_datetime: np.datetime64
    states: list
    weather: list
    reference_states: list
    reference_weather: list


@pytest.fixture(scope='module')
def bundled_model():
    """Load once, execute the reference independently, then execute MLX."""
    with _execution_context():
        reference = api.PressureLevelModel.from_checkpoint(
            demo.load_checkpoint_tl63_stochastic())
        dataset = demo.load_data(reference.data_coords)
        inputs, forcings = reference.data_from_xarray(dataset.isel(time=0))
        temporal_forcings = reference.forcings_from_xarray(dataset)
        initial_datetime = dataset.time.values[0]

        # Never feed an MLX-produced state into the JAX oracle, or vice versa.
        # Synchronization keeps the heavyweight CPU and MLX graphs serial.
        reference_states = []
        reference_weather = []
        state = _ready(
            reference.encode(inputs, forcings, rng_key=jax.random.PRNGKey(0)))
        for step in range(3):
            reference_states.append(_snapshot(state))
            reference_weather.append(
                _snapshot(reference.decode(state, forcings)))
            if step < 2:
                state = _ready(reference.advance(state, forcings))
        del state

        model = mlx.PressureLevelModel(reference)
        states = []
        weather = []
        state = _ready(
            model.encode(inputs, forcings, rng_key=jax.random.PRNGKey(0)))
        for step in range(3):
            states.append(state)
            weather.append(_ready(model.decode(state, forcings)))
            if step < 2:
                state = _ready(model.advance(state, forcings))

    return _BundledModel(
        model,
        inputs,
        forcings,
        temporal_forcings,
        initial_datetime,
        states,
        weather,
        reference_states,
        reference_weather,
    )


@pytest.fixture(autouse=True)
def isolated_execution(bundled_model):
    # Restore configuration after every test, not only after this entire module.
    with _execution_context():
        yield


@pytest.mark.parametrize('step', [0, 1, 2])
def test_bundled_encode_advance_decode_matches_jax(bundled_model, step,
                                                   record_property):
    case = bundled_model
    _assert_parity(
        case.states[step],
        case.reference_states[step],
        stage=f'state_{step}',
        record_property=record_property,
    )
    _assert_parity(
        case.weather[step],
        case.reference_weather[step],
        stage=f'weather_{step}',
        record_property=record_property,
    )
    expected_variables = set(case.model.input_variables) | {'sim_time'}
    assert set(case.weather[step]) == expected_variables
    for name in case.model.input_variables:
        assert case.weather[step][
            name].shape == case.model.data_coords.nodal_shape
    assert np.asarray(case.states[step].randomness.prng_key).dtype == np.uint32
    assert np.asarray(case.states[step].randomness.prng_key).shape == (2,)
    np.testing.assert_array_equal(
        np.asarray(case.states[step].randomness.prng_step),
        case.reference_states[step].randomness.prng_step,
    )


def _floating_fields_differ(first, second):
    assert jax.tree.structure(first) == jax.tree.structure(second)
    pairs = [(np.asarray(a), np.asarray(b))
             for a, b in zip(
                 jax.tree.leaves(first), jax.tree.leaves(second), strict=True)
             if np.issubdtype(np.asarray(a).dtype, np.inexact)]
    assert pairs, 'stochastic state must contain floating-point fields'
    return any(not np.array_equal(a, b) for a, b in pairs)


def test_seed_zero_one_zero_reproduces_stochastic_state(bundled_model):
    case = bundled_model
    encoded = []
    advanced = []
    for seed in (0, 1, 0):
        state = _ready(
            case.model.encode(case.inputs,
                              case.forcings,
                              rng_key=jax.random.PRNGKey(seed)))
        encoded.append(_snapshot(state))
        advanced.append(_snapshot(case.model.advance(state, case.forcings)))

    _assert_tree_equal(encoded[0], encoded[2])
    _assert_tree_equal(advanced[0], advanced[2])
    _assert_tree_equal(encoded[0], case.states[0])
    _assert_tree_equal(advanced[0], case.states[1])
    # Comparing keys alone could pass even if the random field were ignored.
    assert _floating_fields_differ(encoded[0].randomness.core,
                                   encoded[1].randomness.core)
    assert _floating_fields_differ(advanced[0].randomness.core,
                                   advanced[1].randomness.core)
    assert _floating_fields_differ(
        advanced[0].randomness.nodal_value,
        advanced[1].randomness.nodal_value,
    )


def test_decode_preserves_persistent_state_and_continuation(bundled_model):
    case = bundled_model
    state = case.states[0]
    before = _snapshot(state)
    first = _ready(case.model.decode(state, case.forcings))
    _assert_tree_equal(state, before)
    second = _ready(case.model.decode(state, case.forcings))
    _assert_tree_equal(state, before)
    _assert_tree_equal(first, second)
    _assert_tree_equal(first, case.weather[0])
    advanced = _ready(case.model.advance(state, case.forcings))
    _assert_tree_equal(state, before)
    _assert_tree_equal(advanced, case.states[1])


@pytest.mark.parametrize('start_with_input', [False, True])
def test_segmented_two_step_continuation_and_output_timing(
        bundled_model, start_with_input, record_property):
    case = bundled_model
    before = _snapshot(case.states[0])
    final, uninterrupted = _ready(
        case.model.unroll(
            case.states[0],
            case.temporal_forcings,
            steps=2,
            timedelta=case.model.timestep,
            start_with_input=start_with_input,
        ))
    middle, first = _ready(
        case.model.unroll(
            case.states[0],
            case.temporal_forcings,
            steps=1,
            timedelta=case.model.timestep,
            start_with_input=start_with_input,
        ))
    resumed, second = _ready(
        case.model.unroll(
            middle,
            case.temporal_forcings,
            steps=1,
            timedelta=case.model.timestep,
            start_with_input=start_with_input,
        ))
    segmented = _ready(
        jax.tree.map(lambda a, b: mx.concatenate([a, b], axis=0), first,
                     second))
    _assert_tree_equal(case.states[0], before)
    _assert_tree_equal(middle, case.states[1])
    _assert_tree_equal(final, case.states[2])
    _assert_tree_equal(resumed, final)
    _assert_tree_equal(segmented, uninterrupted)

    first_step = 0 if start_with_input else 1
    output_steps = np.arange(first_step, first_step + 2)
    expected_weather = jax.tree.map(
        lambda *values: np.stack(values),
        *[case.reference_weather[step] for step in output_steps],
    )
    _assert_parity(
        uninterrupted,
        expected_weather,
        stage=f'unroll_start_with_input_{start_with_input}',
        record_property=record_property,
    )
    state_times = np.array(
        [np.asarray(case.states[step].state.sim_time) for step in output_steps])
    np.testing.assert_array_equal(np.asarray(uninterrupted['sim_time']),
                                  state_times)
    assert np.all(np.diff(np.asarray(uninterrupted['sim_time'])) > 0)

    times = case.initial_datetime + output_steps * case.model.timestep
    expected_sim_times = case.model.datetime64_to_sim_time(times)
    # Simulation time is a large negative float32 for the 1959 demo input;
    # account for at most one rounding unit per internal transition.
    time_atol = 2 * abs(float(np.spacing(np.float32(expected_sim_times[0]))))
    np.testing.assert_allclose(
        np.asarray(uninterrupted['sim_time']),
        expected_sim_times,
        rtol=0,
        atol=time_atol,
    )
    forecast = case.model.data_to_xarray(uninterrupted, times=times)
    np.testing.assert_array_equal(forecast.time.values, times)
    for name in case.model.input_variables:
        assert forecast[name].dims == ('time', 'level', 'longitude', 'latitude')
        assert forecast[name].shape == (2,) + case.model.data_coords.nodal_shape
        np.testing.assert_array_equal(forecast[name].values,
                                      np.asarray(uninterrupted[name]))
