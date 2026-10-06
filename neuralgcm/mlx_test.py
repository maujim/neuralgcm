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
"""Consumer-visible tests for MLX forecast unrolling."""

import pytest

mx = pytest.importorskip('mlx.core')

from datetime import timedelta
import json
import logging

import jax
import numpy as np

from neuralgcm import mlx
from neuralgcm import inference_logging


class _State:
    """Small pytree state with simulation time and an accumulated value."""

    def __init__(self, time, value):
        self.sim_time = time
        self.value = value


jax.tree_util.register_pytree_node(
    _State,
    lambda state: ((state.sim_time, state.value), None),
    lambda _, children: _State(*children),
)


class _DeterministicReference:
    """Metadata for the parameter-free deterministic numerical test model."""

    timestep = timedelta(hours=1)
    gin_config = 'state.value += forcings["drive"]; state.sim_time += 1'
    params = {}
    data_coords = None


class _DeterministicModel(mlx.PressureLevelModel):
    timestep = timedelta(hours=1)

    def __init__(self, reference_model=None):
        self._reference_model = (_DeterministicReference() if reference_model
                                 is None else reference_model)
        self._log_metadata = inference_logging.describe_model(
            self._reference_model, 'mlx')

    def advance(self, state, forcings):
        # Simulation times are numeric model-time units; timestep is metadata used
        # by unroll to determine the number of internal transitions.
        return _State(state.sim_time + 1., state.value + forcings['drive'])

    def decode(self, state, forcings):
        return {'value': state.value, 'drive': forcings['drive']}


def _array(value):
    return np.asarray(value)


def _fixture():
    model = _DeterministicModel(None)
    state = _State(mx.array(0.), mx.array(0.))
    forcings = {
        'sim_time': mx.array([0., 1., 3.]),
        'drive': mx.array([2., 5., 11.]),
    }
    return model, state, forcings


@pytest.fixture(autouse=True)
def _enable_inference_logging(caplog):
    # Exercise the public logged unroll in every temporal case, restoring the
    # logger level after each test so collection/execution order cannot leak it.
    caplog.set_level(logging.INFO, logger='neuralgcm.inference')


def test_unroll_selects_nearest_forcings_and_earlier_tie():
    model, state, forcings = _fixture()
    state = _State(mx.array(0.5), state.value)
    final_state, outputs = model.unroll(state, forcings, steps=2)
    np.testing.assert_array_equal(_array(outputs['drive']), [5., 11.])
    np.testing.assert_array_equal(_array(outputs['value']), [2., 7.])
    np.testing.assert_array_equal(_array(final_state.value), 7.)
    np.testing.assert_array_equal(_array(final_state.sim_time), 2.5)


def test_unroll_midpoint_rounds_fractional_forcing_index_to_even():
    model, state, forcings = _fixture()
    state = _State(mx.array(2.), state.value)
    final_state, outputs = model.unroll(state, forcings, steps=1)
    # Time 2 lies halfway between forcing times 1 and 3. Its fractional forcing
    # index is 1.5, which rounds to even index 2.
    np.testing.assert_array_equal(_array(outputs['drive']), [11.])
    np.testing.assert_array_equal(_array(final_state.value), 11.)


def test_unroll_timedelta_must_be_integer_multiple_of_timestep():
    model, state, forcings = _fixture()
    final_state, outputs = model.unroll(state,
                                        forcings,
                                        steps=2,
                                        timedelta=timedelta(hours=2))
    np.testing.assert_array_equal(_array(final_state.value), 29.)
    np.testing.assert_array_equal(_array(outputs['value']), [7., 29.])
    with pytest.raises(ValueError):
        model.unroll(state, forcings, steps=1, timedelta=timedelta(minutes=90))


def test_unroll_start_with_input_aligns_outputs_to_input_times():
    model, state, forcings = _fixture()
    final_state, outputs = model.unroll(state,
                                        forcings,
                                        steps=3,
                                        start_with_input=True)
    np.testing.assert_array_equal(_array(outputs['drive']), [2., 5., 11.])
    np.testing.assert_array_equal(_array(outputs['value']), [0., 2., 7.])
    np.testing.assert_array_equal(_array(final_state.value), 18.)


def test_unroll_zero_steps_preserves_state_and_empty_output_shapes():
    model, state, forcings = _fixture()
    final_state, outputs = model.unroll(state, forcings, steps=0)
    assert final_state is state
    assert outputs['value'].shape == (0,)
    assert outputs['drive'].shape == (0,)


def test_unroll_custom_postprocess_and_invalid_steps():
    model, state, forcings = _fixture()
    final_state, outputs = model.unroll(
        state,
        forcings,
        steps=2,
        post_process_fn=lambda current, forcing: {
            'combined': current.value + forcing['drive'] * 10,
        },
    )
    np.testing.assert_array_equal(_array(outputs['combined']), [52., 117.])
    np.testing.assert_array_equal(_array(final_state.value), 7.)
    with pytest.raises(ValueError):
        model.unroll(state, forcings, steps=-1)


def test_unroll_populates_metadata_when_logging_enabled_after_construction(
        caplog):
    caplog.set_level(logging.WARNING, logger='neuralgcm.inference')
    model, state, forcings = _fixture()
    assert model._log_metadata == {}
    caplog.set_level(logging.INFO, logger='neuralgcm.inference')

    final_state, outputs = model.unroll(state, forcings, steps=2)

    np.testing.assert_array_equal(_array(final_state.value), 7.)
    np.testing.assert_array_equal(_array(outputs['value']), [2., 7.])
    records = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == 'neuralgcm.inference'
    ]
    unroll_record = next(
        record for record in records if record.get('operation') == 'unroll')
    assert unroll_record['status'] == 'ok'
    assert unroll_record['model']['model_class'] == '_DeterministicReference'
    assert unroll_record['model']['backend'] == 'mlx'
    assert unroll_record['model']['parameter_count'] == 0
