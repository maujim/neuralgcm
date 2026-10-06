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
"""User-facing inference diagnostics from real small numerical operations."""

import hashlib
import json
import logging
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from neuralgcm import inference_logging
import numpy as np
import pytest

LOGGER = 'neuralgcm.inference'


@pytest.fixture(autouse=True)
def isolated_inference_logger():
    logger = logging.getLogger(LOGGER)
    level = logger.level
    handlers = list(logger.handlers)
    propagate = logger.propagate
    try:
        logger.setLevel(logging.WARNING)
        logger.handlers.clear()
        logger.propagate = True
        yield
    finally:
        new_handlers = [
            handler for handler in logger.handlers if handler not in handlers
        ]
        for handler in new_handlers:
            logger.removeHandler(handler)
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate
        for handler in new_handlers:
            handler.close()


def _events(caplog):
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == LOGGER
    ]


@pytest.fixture
def model():
    return SimpleNamespace(
        params={
            'encoder': {
                'weights': np.arange(6, dtype=np.float32).reshape(2, 3)
            },
            'decoder': (np.array([1, 2], dtype=np.float64),),
        },
        gin_config='Model.width = 3\nModel.activation = "tanh"\n',
        data_coords=SimpleNamespace(
            nodal_shape=(2, 8, 4),
            horizontal=SimpleNamespace(nodal_shape=(8, 4),
                                       longitude_nodes=8,
                                       latitude_nodes=4),
            vertical=SimpleNamespace(layers=2),
        ),
        model_coords=SimpleNamespace(
            nodal_shape=(3, 8, 4),
            horizontal=SimpleNamespace(nodal_shape=(8, 4),
                                       longitude_nodes=8,
                                       latitude_nodes=4),
            vertical=SimpleNamespace(layers=3),
        ),
        timestep=np.timedelta64(30, 'm'),
        input_variables=['temperature', 'geopotential'],
        forcing_variables=['sea_surface_temperature'],
    )


def test_log_event_is_structured_and_retains_explicit_run_metadata(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    inference_logging.log_event(
        'run_started',
        backend='jax',
        steps=2,
        seed=42,
        initial_data='historical ERA5; not live weather',
    )
    [event] = _events(caplog)
    assert event['event'] == 'run_started'
    assert event['schema_version'] == 1
    assert event['backend'] == 'jax'
    assert event['steps'] == 2
    assert event['seed'] == 42
    assert event['initial_data'] == 'historical ERA5; not live weather'


def test_info_disabled_does_not_serialize_fields(caplog):

    class Unserializable:

        def __str__(self):
            raise AssertionError(
                'suppressed logging must not stringify metadata')

    caplog.set_level(logging.WARNING, logger=LOGGER)
    inference_logging.log_event('run_started', metadata=Unserializable())
    assert not _events(caplog)


def test_model_metadata_is_value_free_and_identifies_configuration(
        caplog, model):
    caplog.set_level(logging.INFO, logger=LOGGER)
    metadata = inference_logging.describe_model(
        model, backend='jax', checkpoint='/models/example.pkl')
    inference_logging.log_event('model_loaded', **metadata)
    [event] = _events(caplog)
    assert event['parameter_count'] == 8
    assert event['parameter_bytes'] == 40
    assert event['parameter_array_count'] == 2
    assert event['precision'] == ['float32', 'float64']
    assert event['checkpoint'] == '/models/example.pkl'
    assert event['configuration_sha256'] == hashlib.sha256(
        model.gin_config.encode()).hexdigest()
    assert event['configuration_summary'] == model.gin_config
    assert event['timestep'] == '30 minutes'
    assert event['grid']['vertical_layers'] == 2
    assert event['grid']['longitude_nodes'] == 8
    assert event['grid']['latitude_nodes'] == 4
    assert event['grid']['horizontal_shape'] == [8, 4]
    assert event['backend'] == 'jax'
    assert event['device']
    assert event['versions']['jax'] == jax.__version__
    assert event['versions']['numpy'] == np.__version__
    assert 'params' not in event
    assert 'values' not in event


def test_numerical_logged_call_reports_shapes_types_bytes_and_elapsed(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    inputs = {
        'left': np.arange(6, dtype=np.float32).reshape(2, 3),
        'right': np.array([2, 3, 4], dtype=np.float32),
    }

    def transform(fields, *, scale):
        return {'forecast': np.square(fields['left'] + fields['right']) * scale}

    result = inference_logging.logged_call(
        'transform',
        transform,
        inputs,
        backend='jax',
        scale=2,
        model_metadata={'checkpoint': '/models/example.pkl'},
    )
    np.testing.assert_array_equal(result['forecast'],
                                  [[8, 32, 72], [50, 98, 162]])
    [event] = _events(caplog)
    assert event['event'] == 'operation'
    assert event['schema_version'] == 1
    assert event['operation'] == 'transform'
    assert event['status'] == 'ok'
    assert event['backend'] == 'jax'
    assert np.isfinite(event['elapsed_seconds'])
    assert event['elapsed_seconds'] > 0
    assert event['inputs'] == {
        'array_count': 2,
        'arrays': [
            {
                'shape': [2, 3],
                'dtype': 'float32',
                'bytes': 24
            },
            {
                'shape': [3],
                'dtype': 'float32',
                'bytes': 12
            },
        ],
        'bytes': 36,
    }
    assert event['outputs'] == {
        'array_count': 1,
        'arrays': [{
            'shape': [2, 3],
            'dtype': 'float32',
            'bytes': 24
        }],
        'bytes': 24,
    }
    assert event['model']['checkpoint'] == '/models/example.pkl'
    assert 'seed' not in event
    assert 'values' not in event['outputs']['arrays'][0]


def test_failed_numerical_operation_logs_and_preserves_exception(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    inputs = np.array([1., 0.], dtype=np.float32)
    original_errors = []

    def divide_by_zero(values):
        try:
            with np.errstate(divide='raise', invalid='raise'):
                return np.divide(1., values)
        except FloatingPointError as error:
            original_errors.append(error)
            raise

    with pytest.raises(FloatingPointError, match='divide by zero') as caught:
        inference_logging.logged_call('decode',
                                      divide_by_zero,
                                      inputs,
                                      backend='jax')
    [event] = _events(caplog)
    assert event['status'] == 'error'
    assert event['operation'] == 'decode'
    assert event['error'] == f'FloatingPointError: {caught.value}'
    assert event['elapsed_seconds'] > 0
    assert event['inputs']['bytes'] == 8
    assert 'outputs' not in event
    assert caught.value.__traceback__ is not None
    assert caught.value is original_errors[0]


def test_jax_success_is_logged_after_actual_result_is_ready(
        caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger=LOGGER)
    synchronize = jax.block_until_ready
    synchronized = []

    def observe_synchronization(value):
        assert not _events(caplog)
        ready = synchronize(value)
        synchronized.append(ready)
        return ready

    monkeypatch.setattr(jax, 'block_until_ready', observe_synchronization)
    inputs = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    result = inference_logging.logged_call(
        'advance',
        lambda values: {'state': jnp.sin(values) + 2},
        inputs,
        backend='jax',
    )
    assert synchronized
    assert isinstance(result['state'], jax.Array)
    np.testing.assert_allclose(
        result['state'],
        np.sin(np.arange(6, dtype=np.float32).reshape(2, 3)) + 2,
        rtol=1e-6,
    )
    [event] = _events(caplog)
    assert event['status'] == 'ok'
    assert event['elapsed_seconds'] > 0
    assert event['outputs']['bytes'] == 24


def test_suppressed_info_does_not_inspect_operation_metadata(caplog):

    class UninspectedArray:

        def __init__(self, values):
            self.values = values

        def __array__(self, dtype=None, copy=None):
            if copy:
                return np.array(self.values, dtype=dtype, copy=True)
            return np.asarray(self.values, dtype=dtype)

        @property
        def shape(self):
            raise AssertionError('disabled INFO must not inspect array shape')

        @property
        def dtype(self):
            raise AssertionError('disabled INFO must not inspect array dtype')

        @property
        def nbytes(self):
            raise AssertionError('disabled INFO must not inspect array bytes')

    def square(values):
        return UninspectedArray(np.square(values))

    caplog.set_level(logging.WARNING, logger=LOGGER)
    result = inference_logging.logged_call(
        'transform',
        square,
        UninspectedArray(np.array([2, 3], dtype=np.int32)),
        backend='jax',
    )
    assert isinstance(result, UninspectedArray)
    np.testing.assert_array_equal(result.values, [4, 9])
    assert not _events(caplog)


def test_optional_mlx_operation_uses_real_materialized_array(caplog):
    mx = pytest.importorskip('mlx.core')
    caplog.set_level(logging.INFO, logger=LOGGER)
    inputs = mx.array([[1., 2.], [3., 4.]], dtype=mx.float32)

    def transform(values):
        result = mx.square(values) + 3
        # MLX operation wrappers own synchronization; logged_call must not repeat it.
        mx.eval(result)
        return result

    result = inference_logging.logged_call('decode',
                                           transform,
                                           inputs,
                                           backend='mlx')
    np.testing.assert_array_equal(np.asarray(result), [[4, 7], [12, 19]])
    [event] = _events(caplog)
    assert event['backend'] == 'mlx'
    assert event['status'] == 'ok'
    assert event['elapsed_seconds'] > 0
    assert event['inputs']['bytes'] == 16
    assert event['outputs']['bytes'] == 16
    assert event['outputs']['arrays'][0]['shape'] == [2, 2]
    assert event['outputs']['arrays'][0]['dtype'] == str(mx.float32)


def test_numpy_metadata_summary_does_not_require_np_asarray(
        caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger=LOGGER)
    inputs = np.arange(4, dtype=np.float32)

    def unexpected_conversion(*args, **kwargs):
        raise AssertionError('NumPy metadata summary must not call np.asarray')

    with monkeypatch.context() as patch:
        patch.setattr(np, 'asarray', unexpected_conversion)
        result = inference_logging.logged_call('advance',
                                               np.square,
                                               inputs,
                                               backend='jax')
    np.testing.assert_array_equal(result, [0, 1, 4, 9])
    [event] = _events(caplog)
    assert event['inputs']['bytes'] == 16
    assert event['outputs']['bytes'] == 16


def test_suppressed_info_does_not_inspect_model(caplog):

    class UninspectedModel:

        @property
        def params(self):
            raise AssertionError(
                'disabled INFO must not traverse model parameters')

        @property
        def gin_config(self):
            raise AssertionError('disabled INFO must not inspect configuration')

        @property
        def data_coords(self):
            raise AssertionError('disabled INFO must not inspect coordinates')

    caplog.set_level(logging.WARNING, logger=LOGGER)
    assert inference_logging.describe_model(
        UninspectedModel(), backend='jax',
        checkpoint='/models/example.pkl') == {}
    assert not _events(caplog)
