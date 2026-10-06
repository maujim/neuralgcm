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
"""Consumer checks for archived initial weather without model inference."""

import importlib.resources
import json
import queue
import threading

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm
from neuralgcm.local_app_test import _error
from neuralgcm.local_app_test import _post
from neuralgcm.local_app_test import _request
from neuralgcm.local_app_test import _serving
import numpy as np
import pytest
import xarray

_INITIAL_PATH = '/api/initial'
_FIELDS = ('temperature', 'wind_speed', 'specific_humidity')
_INITIAL_TIME = '1959-01-02T00:00:00Z'


@pytest.fixture(scope='module')
def http_server():
    with _serving() as server:
        yield server


def _json_response(response, status=200):
    actual_status, headers, raw = response
    assert actual_status == status, raw
    assert headers['Content-Type'] == 'application/json; charset=utf-8'
    assert headers['Cache-Control'] == 'no-store'
    assert int(headers['Content-Length']) == len(raw)

    def reject_nonfinite(token):
        raise ValueError(f'Non-finite HTTP JSON: {token}')

    return json.loads(raw, parse_constant=reject_nonfinite)


@pytest.fixture(scope='module')
def archived_initial(http_server):
    return _json_response(_request(http_server, 'GET', _INITIAL_PATH))


@pytest.fixture(scope='module')
def raw_era5_850():
    """Read and regrid the archive independently of all server data helpers."""
    resource = importlib.resources.files(neuralgcm).joinpath(
        'data/era5_tl31_19590102T00.nc')
    with resource.open('rb') as handle:
        dataset = xarray.load_dataset(handle)
    np.testing.assert_array_equal(
        np.asarray(dataset.time.values, dtype='datetime64[s]').reshape(-1),
        np.array(['1959-01-02T00:00:00'], dtype='datetime64[s]'),
    )
    if 'time' in dataset.dims:
        dataset = dataset.isel(time=0)
    # Exact labeled selection checks the advertised pressure surface, not the
    # endpoint's nearest-level selection. No checkpoint or model is loaded.
    pressure_slice = dataset[[
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
        'specific_humidity',
    ]].sel(level=850)
    regridder = horizontal_interpolation.ConservativeRegridder(
        spherical_harmonic.Grid.TL31(radius=1.0),
        spherical_harmonic.Grid.TL63(radius=1.0),
    )
    regridded = xarray_utils.regrid_horizontal(pressure_slice, regridder)
    regridded = regridded.assign_coords(
        longitude=(regridded.longitude + 180.0) % 360.0 - 180.0)
    regridded = regridded.sortby(['latitude', 'longitude'])
    assert regridded.sizes['latitude'] == 64
    assert regridded.sizes['longitude'] == 128
    # Integer stride 2 is required by the existing 48 x 96 display cap.
    return regridded.isel(latitude=slice(None, None, 2),
                          longitude=slice(None, None, 2))


def test_initial_is_archived_reanalysis_not_a_future_or_a_job(
        archived_initial, http_server):
    assert set(archived_initial) == {'grid', 'frame', 'provenance'}
    frame = archived_initial['frame']
    assert set(frame) == {'index', 'valid_time', 'lead_hours', 'kind', 'fields'}
    assert frame['index'] == frame['lead_hours'] == 0
    assert frame['valid_time'] == _INITIAL_TIME
    assert frame['kind'] == 'initial_reanalysis'
    assert set(frame['fields']) == set(_FIELDS)
    provenance = archived_initial['provenance']
    assert provenance['initial_condition'] == 'ERA5 reanalysis'
    assert provenance['initial_time'] == frame['valid_time']
    assert provenance['level_hpa'] == 850
    assert 'model_id' not in provenance
    assert 'seed' not in provenance
    config = _json_response(_request(http_server, 'GET', '/api/config'))
    assert {
        field['id']: field['unit'] for field in config['fields']
    } == {
        'temperature': '°C',
        'wind_speed': 'm/s',
        'specific_humidity': 'g/kg',
    }


def test_initial_grid_matches_independent_archive_coordinates(
        archived_initial, raw_era5_850):
    grid = archived_initial['grid']
    assert grid['level_hpa'] == 850
    assert grid['layout'] == 'latitude-major'
    assert grid['display_stride'] == {'latitude': 2, 'longitude': 2}
    np.testing.assert_array_equal(grid['latitudes'],
                                  raw_era5_850.latitude.values)
    np.testing.assert_array_equal(grid['longitudes'],
                                  raw_era5_850.longitude.values)


@pytest.mark.parametrize('field', _FIELDS)
def test_initial_values_match_independent_850_hpa_regrid_and_units(
        archived_initial, raw_era5_850, field):
    temperature = raw_era5_850.temperature.astype(np.float64)
    u = raw_era5_850.u_component_of_wind.astype(np.float64)
    v = raw_era5_850.v_component_of_wind.astype(np.float64)
    humidity = raw_era5_850.specific_humidity.astype(np.float64)
    expected = {
        'temperature': temperature - 273.15,
        'wind_speed': np.sqrt(u * u + v * v),
        'specific_humidity': humidity * 1000.0,
    }[field].transpose('latitude', 'longitude')
    actual = np.asarray(archived_initial['frame']['fields'][field])
    assert actual.shape == (2048,)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual.reshape(32, 64),
        expected.values,
        rtol=1e-6,
        atol=1e-6,
        err_msg=f'archived ERA5 850 hPa {field}, latitude-major display units',
    )


@pytest.mark.parametrize('query', [
    '',
    '?model_id=deterministic_2_8_deg&duration_hours=24&seed=42',
    '?level_hpa=1000&lead_hours=1&time=2026-10-06',
    '?checkpoint=/private/checkpoint.pkl&model_id=unknown',
])
def test_repeated_initial_reads_and_query_cannot_change_archive(
        http_server, query):
    before = _request(http_server, 'GET', _INITIAL_PATH)
    requested = _request(http_server, 'GET', _INITIAL_PATH + query)
    after = _request(http_server, 'GET', _INITIAL_PATH)
    for response in (before, requested, after):
        _json_response(response)
    assert before[2] == requested[2] == after[2]


@pytest.mark.parametrize('host', [
    'example.com:{port}',
    '127.0.0.1:1',
    '127.0.0.1:{port}@example.com',
    '',
])
def test_initial_rejects_non_loopback_host(http_server, host):
    _error(
        _request(http_server,
                 'GET',
                 _INITIAL_PATH,
                 headers={'Host': host.format(port=http_server.server_port)}),
        400, 'invalid_host')


def test_initial_reads_leave_all_real_http_job_slots_available(monkeypatch):
    """Hold genuine dequeued work so admission is independent of model speed.

    The Event instruments only the stdlib Queue.get boundary, as in the queue
    consumer tests. HTTP admission, job records and clearing are real. Every
    admitted job is cancelled before execution; no inference/model double runs.
    """
    original_get = queue.Queue.get
    captured = threading.Event()
    release = threading.Event()
    work_queue = None

    def gated_get(instance, block=True, timeout=None):
        job_id = original_get(instance, block=block, timeout=timeout)
        if block and instance is work_queue and job_id is not None:
            captured.set()
            release.wait(30)
        return job_id

    monkeypatch.setattr(queue.Queue, 'get', gated_get)
    with _serving() as server:
        work_queue = server.work
        try:
            initial = _json_response(_request(server, 'GET', _INITIAL_PATH))
            assert _json_response(_request(server, 'GET',
                                           _INITIAL_PATH)) == initial
            admitted = []
            for seed in range(8):
                request = {
                    'model_id': 'toy_tl63',
                    'duration_hours': 1,
                    'seed': seed
                }
                job = _json_response(_post(server, request), 202)
                assert job['request'] == request
                assert job['status'] == 'queued'
                admitted.append(job)
            assert captured.wait(
                5), 'Inference worker did not dequeue real work'
            _error(
                _post(server, {
                    'model_id': 'toy_tl63',
                    'duration_hours': 1,
                    'seed': 8
                }), 503, 'capacity_reached')
            # Read-only archive access must also work when every slot is full.
            assert _json_response(_request(server, 'GET',
                                           _INITIAL_PATH)) == initial
            cleared = _json_response(_post(server, {}, path='/api/jobs/clear'))
            assert cleared == {'cleared_count': 8}
            for job in admitted:
                snapshot = _json_response(
                    _request(server, 'GET', f'/api/jobs/{job["id"]}'))
                assert snapshot['status'] == snapshot['phase'] == 'cancelled'
                assert snapshot['available_frames'] == 0
                assert snapshot['progress']['completed_steps'] == 0
        finally:
            try:
                _json_response(_post(server, {}, path='/api/jobs/clear'))
            finally:
                release.set()
