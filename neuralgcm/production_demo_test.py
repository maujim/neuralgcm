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
"""Real-file tests for forecast serialization and runner file boundaries."""

import copy
import io
import json
import logging
import pickle

import jax
from neuralgcm import demo
from neuralgcm import production_demo
import numpy as np
import pandas as pd
import pytest
import xarray


@pytest.fixture(autouse=True)
def isolated_jax_config():
    # legacy/api_test.py changes the process-wide PRNG setting during collection.
    with jax.threefry_partitionable(True), jax.default_device(
            jax.devices('cpu')[0]):
        yield


@pytest.fixture
def isolated_inference_logger():
    logger = logging.getLogger('neuralgcm.inference')
    level = logger.level
    handlers = list(logger.handlers)
    propagate = logger.propagate
    try:
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


@pytest.fixture
def netcdf_available():
    engines = xarray.backends.list_engines()
    if not {'netcdf4', 'h5netcdf', 'scipy'}.intersection(engines):
        pytest.skip('NetCDF roundtrips require netCDF4, h5netcdf, or scipy')


@pytest.fixture
def forecast_dataset():
    temperature = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)
    temperature[0, 0, 0, 0] = np.nan
    temperature[1, 1, 1, 2] = np.inf
    return xarray.Dataset(
        data_vars={
            'temperature': (
                ('time', 'level', 'latitude', 'longitude'),
                temperature,
                {
                    'units': 'K',
                    'long_name': 'Air temperature'
                },
            ),
            'sea_surface_temperature': (
                ('time', 'latitude', 'longitude'),
                np.linspace(270., 300., 12, dtype=np.float64).reshape(2, 2, 3),
                {
                    'units': 'K'
                },
            ),
            'quality': (
                ('time', 'latitude', 'longitude'),
                np.arange(12, dtype=np.int16).reshape(2, 2, 3),
            ),
            'member': ((), np.int32(3)),
        },
        coords={
            'time':
                np.array(
                    ['1959-01-02T01:00:00', '1959-01-02T02:00:00'],
                    dtype='datetime64[ns]',
                ),
            'level':
                np.array([500, 850], dtype=np.int32),
            'latitude':
                np.array([-45., 45.], dtype=np.float64),
            'longitude':
                np.array([0., 120., 240.], dtype=np.float64),
        },
        attrs={
            'neuralgcm_checkpoint':
                '/checkpoints/stochastic.pkl',
            'neuralgcm_backend':
                'mlx',
            'neuralgcm_initial_data': ('bundled historical ERA5 demonstration '
                                       '(1959-01-02; not live weather)'),
            'neuralgcm_seed':
                42,
            'neuralgcm_forecast_steps':
                2,
            'neuralgcm_data_caveat':
                ('Historical demonstration/initial condition; '
                 'not a live weather forecast.'),
        },
    )


def test_write_forecast_netcdf_preserves_data_and_provenance(
        tmp_path, netcdf_available, forecast_dataset):
    path = tmp_path / 'forecast.nc'
    before = forecast_dataset.copy(deep=True)

    production_demo.write_forecast_netcdf(forecast_dataset, str(path))

    with xarray.open_dataset(path) as opened:
        restored = opened.load()
    xarray.testing.assert_identical(restored, before)
    assert set(restored.data_vars) == set(before.data_vars)
    assert set(restored.coords) == set(before.coords)
    for name in before.variables:
        assert restored[name].dtype == before[name].dtype
        np.testing.assert_array_equal(restored[name].values,
                                      before[name].values)
    xarray.testing.assert_identical(forecast_dataset, before)


def test_write_forecast_netcdf_sanitizes_all_attribute_scopes_without_mutation(
        tmp_path, netcdf_available, forecast_dataset):
    metadata = {
        'unused': None,
        'enabled': True,
        'disabled': np.bool_(False),
        'details': {
            'levels': [500, 850],
            'nested': {
                'historical': True,
                'unavailable': None
            },
            'seed': np.int64(42),
        },
        'flags': np.array([True, False], dtype=np.bool_),
        'mixed': [None, {
            'source': 'ERA5'
        }],
    }
    # Coordinates are NetCDF variables too; their attrs need the same treatment.
    forecast_dataset.attrs.update(copy.deepcopy(metadata))
    for name in forecast_dataset.variables:
        forecast_dataset[name].attrs.update(copy.deepcopy(metadata))
    before = forecast_dataset.copy(deep=True)
    expected = forecast_dataset.copy(deep=True)
    for attrs in [expected.attrs
                 ] + [v.attrs for v in expected.variables.values()]:
        del attrs['unused']
        attrs['enabled'] = 1
        attrs['disabled'] = 0
        attrs['details'] = json.dumps({
            'levels': [500, 850],
            'nested': {
                'historical': True,
                'unavailable': None
            },
            'seed': 42,
        })
        attrs['flags'] = np.array([1, 0], dtype=np.int8)
        attrs['mixed'] = json.dumps([None, {'source': 'ERA5'}])

    path = tmp_path / 'sanitized-forecast.nc'
    production_demo.write_forecast_netcdf(forecast_dataset, str(path))

    with xarray.open_dataset(path) as opened:
        restored = opened.load()
    xarray.testing.assert_identical(restored, expected)
    for attrs in [restored.attrs
                 ] + [v.attrs for v in restored.variables.values()]:
        assert 'unused' not in attrs
        assert attrs['enabled'] == 1
        assert attrs['disabled'] == 0
        assert json.loads(attrs['details']) == {
            'levels': [500, 850],
            'nested': {
                'historical': True,
                'unavailable': None
            },
            'seed': 42,
        }
        assert json.loads(attrs['mixed']) == [None, {'source': 'ERA5'}]
        np.testing.assert_array_equal(attrs['flags'], [1, 0])
    xarray.testing.assert_identical(forecast_dataset, before)
    for attrs in [forecast_dataset.attrs
                 ] + [v.attrs for v in forecast_dataset.variables.values()]:
        assert attrs['unused'] is None
        assert attrs['enabled'] is True
        assert isinstance(attrs['details'], dict)
        assert attrs['flags'].dtype == np.dtype(np.bool_)


@pytest.mark.parametrize('backend', ['jax', 'mlx'])
def test_runner_rejects_missing_selected_checkpoint_before_creating_output(
        tmp_path, backend, isolated_inference_logger):
    checkpoint_path = tmp_path / 'missing.pkl'
    output_path = tmp_path / 'forecast.nc'
    with pytest.raises(FileNotFoundError) as caught:
        production_demo.main([
            '--checkpoint',
            str(checkpoint_path),
            '--backend',
            backend,
            '--output',
            str(output_path),
        ])
    assert caught.value.filename == str(checkpoint_path)
    assert not output_path.exists()


@pytest.mark.parametrize('backend', ['jax', 'mlx'])
def test_runner_rejects_corrupt_selected_checkpoint_before_creating_output(
        tmp_path, backend, isolated_inference_logger):
    checkpoint_path = tmp_path / 'corrupt.pkl'
    checkpoint_path.write_bytes(b'not a pickle checkpoint')
    output_path = tmp_path / 'forecast.nc'
    with pytest.raises(pickle.UnpicklingError):
        production_demo.main([
            '--checkpoint',
            str(checkpoint_path),
            '--backend',
            backend,
            '--output',
            str(output_path),
        ])
    assert not output_path.exists()


@pytest.mark.integration
def test_runner_writes_real_bundled_mlx_forecast(tmp_path, netcdf_available,
                                                 isolated_inference_logger):
    pytest.importorskip('mlx.core')
    checkpoint = demo.load_checkpoint_tl63_stochastic()
    checkpoint_path = tmp_path / 'bundled-trusted.pkl'
    with checkpoint_path.open('wb') as checkpoint_file:
        pickle.dump(checkpoint, checkpoint_file)
    output_path = tmp_path / 'forecast.nc'
    seed = 17
    provenance = ('bundled historical ERA5 demonstration '
                  '(1959-01-02; not live weather)')
    # main disables propagation: capture the actual named logger, not the root.
    # The isolation fixture restores its handlers, level, and propagation flag.
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger('neuralgcm.inference')
    logger.addHandler(handler)

    production_demo.main([
        '--checkpoint',
        str(checkpoint_path),
        '--backend',
        'mlx',
        '--steps',
        '1',
        '--seed',
        str(seed),
        '--output',
        str(output_path),
    ])

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert all(event['schema_version'] == 1 for event in events)
    starts = [event for event in events if event['event'] == 'inference_start']
    assert len(starts) == 1
    start = starts[0]
    assert start['seed'] == seed
    assert start['requested_steps'] == 1
    assert start['initial_data'] == provenance
    metadata = start['model']
    assert metadata['model_class'] == 'PressureLevelModel'
    assert metadata['checkpoint'] == str(checkpoint_path)
    assert metadata['backend'] == 'mlx'
    assert metadata['seed'] == seed
    assert metadata['requested_steps'] == 1
    assert metadata['initial_data'] == provenance
    assert len(metadata['configuration_sha256']) == 64
    int(metadata['configuration_sha256'], 16)
    assert metadata['parameter_count'] == sum(
        np.size(value) for value in jax.tree.leaves(checkpoint['params']))
    assert metadata['parameter_count'] > 0

    operations = [event for event in events if event['event'] == 'operation']
    assert [event['operation'] for event in operations
           ] == ['encode', 'advance', 'decode', 'unroll']
    for event in operations:
        assert event['backend'] == 'mlx'
        assert event['status'] == 'ok'
        assert event['model'] == metadata
        assert np.isfinite(event['elapsed_seconds'])
        assert event['elapsed_seconds'] > 0
        assert event['outputs']['array_count'] > 0
        assert event['outputs']['bytes'] > 0
        if event['operation'] != 'unroll':
            # MLXFunction reports execution only after mx.eval has materialized it.
            profile = event['profile']
            assert profile['trace'] is True
            assert profile['cachehit'] is False
            assert profile['cache_size'] == 1
            assert profile['primitive_counts']
            assert profile['output_bytes'] == event['outputs']['bytes']
            for timing in ('trace_seconds', 'execution_seconds'):
                assert np.isfinite(profile[timing])
                assert profile[timing] > 0
            assert event['elapsed_seconds'] >= profile['execution_seconds']

    with xarray.open_dataset(output_path) as opened:
        forecast = opened.load()
    assert forecast.attrs['neuralgcm_checkpoint'] == str(checkpoint_path)
    assert forecast.attrs['neuralgcm_backend'] == 'mlx'
    assert forecast.attrs['neuralgcm_seed'] == seed
    assert forecast.attrs['neuralgcm_forecast_steps'] == 1
    assert forecast.attrs['neuralgcm_initial_data'] == provenance
    assert 'not a live weather forecast' in forecast.attrs[
        'neuralgcm_data_caveat']
    expected_time = (np.datetime64('1959-01-02T00:00:00', 'ns') +
                     pd.Timedelta(metadata['timestep']).to_timedelta64())
    np.testing.assert_array_equal(forecast.time.values, [expected_time])
    auxiliary = xarray.Dataset.from_dict(checkpoint['aux_ds_dict'])
    for name in ('level', 'latitude', 'longitude'):
        assert forecast[name].dims == (name,)
        np.testing.assert_allclose(forecast[name].values,
                                   auxiliary[name].values)
    assert {
        'geopotential',
        'specific_humidity',
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
        'specific_cloud_ice_water_content',
        'specific_cloud_liquid_water_content',
    } <= set(forecast.data_vars)
    for field in forecast.data_vars.values():
        assert field.sizes['time'] == 1
        assert np.isfinite(field.values).all(), field.name
