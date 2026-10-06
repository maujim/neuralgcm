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
"""Serial real-MLX toy forecasts consumed through the loopback HTTP API."""

import contextlib
import dataclasses
import http.client
import importlib.resources
import json
import threading
import time

import pytest

pytest.importorskip('mlx.core')

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm

from neuralgcm import local_app
import numpy as np
import xarray

pytestmark = pytest.mark.integration

_FIELDS = ('temperature', 'wind_speed', 'specific_humidity')
_FRAME_TIMES = ('1959-01-02T00:00:00Z', '1959-01-02T01:00:00Z')
_JOB_TIMEOUT_SECONDS = 900


@dataclasses.dataclass(frozen=True)
class _ObservedJob:
    accepted: dict
    snapshots: tuple[dict, ...]
    pending_prediction: tuple[int, dict]
    initial_while_running: dict | None
    frames: tuple[dict, dict]

    @property
    def completed(self):
        return self.snapshots[-1]


@dataclasses.dataclass(frozen=True)
class _JobResults:
    config: dict
    jobs: tuple[_ObservedJob, ...]
    consumer_errors: dict[str, tuple[int, dict]]


def _reject_nonfinite(token):
    raise ValueError(f'HTTP JSON contains a non-finite number: {token}')


def _request(port, method, path, body=None):
    headers = {}
    if body is not None:
        body = json.dumps(body, allow_nan=False).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    with contextlib.closing(
            http.client.HTTPConnection('127.0.0.1', port,
                                       timeout=30)) as connection:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        assert response.getheader('Content-Type').startswith('application/json')
        payload = json.loads(response.read(), parse_constant=_reject_nonfinite)
        assert isinstance(payload, dict)
        return response.status, payload


@contextlib.contextmanager
def _running_server():
    server = local_app.create_server(host='127.0.0.1', port=0)
    serving = threading.Thread(
        target=server.serve_forever,
        kwargs={'poll_interval': 0.05},
        name='local-app-integration-http',
        daemon=True,
    )
    started = False
    try:
        serving.start()
        started = True
        yield server.server_port
    finally:
        try:
            if started:
                server.shutdown()
        finally:
            # server_close drains accepted jobs, joins the inference worker, and
            # releases its model cache even when an assertion or HTTP call fails.
            server.server_close()
            if started:
                serving.join(timeout=10)
                assert not serving.is_alive(
                ), 'HTTP serving thread did not stop'


def _observe_job(port, seed):
    request = {'model_id': 'toy_tl63', 'duration_hours': 1, 'seed': seed}
    status, accepted = _request(port, 'POST', '/api/jobs', request)
    assert status == 202, accepted
    assert accepted['status'] in ('queued', 'running'), accepted
    assert accepted['request'] == request
    job_path = f'/api/jobs/{accepted["id"]}'
    # Probe the actual endpoint, not the status response's promise. A warmed
    # model may finish between requests, so the cold first job supplies the
    # strict unfinished-frame assertion below.
    pending_prediction = _request(port, 'GET', f'{job_path}/frames/1')
    snapshots = [accepted]
    initial_while_running = None
    deadline = time.monotonic() + _JOB_TIMEOUT_SECONDS
    while True:
        status, snapshot = _request(port, 'GET', job_path)
        assert status == 200, snapshot
        snapshots.append(snapshot)
        if snapshot['status'] == 'failed':
            pytest.fail(f'real MLX seed={seed} job failed: {snapshot["error"]}')
        if snapshot['status'] == 'completed':
            break
        if initial_while_running is None and snapshot['available_frames'] == 1:
            status, initial_while_running = _request(port, 'GET',
                                                     f'{job_path}/frames/0')
            assert status == 200, initial_while_running
        assert time.monotonic() < deadline, (
            f'real MLX seed={seed} job exceeded {_JOB_TIMEOUT_SECONDS}s; '
            f'last status: {snapshot}')
        time.sleep(0.1)
    frames = []
    for index in (0, 1):
        status, frame = _request(port, 'GET', f'{job_path}/frames/{index}')
        assert status == 200, frame
        frames.append(frame)
    return _ObservedJob(accepted, tuple(snapshots), pending_prediction,
                        initial_while_running, tuple(frames))


@pytest.fixture(scope='module')
def real_job_results():
    """Amortize one model build across sequential seed 0, 1, and 0 jobs.

    No process environment, global random state, or JAX configuration is changed:
    these jobs exercise the same worker defaults as the browser application.
    """
    with _running_server() as port:
        status, config = _request(port, 'GET', '/api/config')
        assert status == 200, config
        toy = next(
            model for model in config['models'] if model['id'] == 'toy_tl63')
        assert toy['available'], toy['unavailable_reason']
        assert toy['kind'] == 'toy' and toy['stochastic'] is True
        jobs = tuple(_observe_job(port, seed) for seed in (0, 1, 0))
        completed_path = f'/api/jobs/{jobs[0].completed["id"]}'
        paths = {
            'past_horizon': f'{completed_path}/frames/2',
            'negative_index': f'{completed_path}/frames/-1',
            'fractional_index': f'{completed_path}/frames/1.5',
            'unknown_job': '/api/jobs/not-issued-by-server',
            'unknown_job_frame': '/api/jobs/not-issued-by-server/frames/0',
        }
        consumer_errors = {
            name: _request(port, 'GET', path) for name, path in paths.items()
        }
    return _JobResults(config, jobs, consumer_errors)


@pytest.fixture(scope='module')
def initial_era5_850():
    """Independent raw-resource oracle, with no second forecast-model build."""
    resource = importlib.resources.files(neuralgcm).joinpath(
        'data/era5_tl31_19590102T00.nc')
    with resource.open('rb') as handle:
        raw = xarray.load_dataset(handle)
    np.testing.assert_array_equal(
        np.asarray(raw.time.values, dtype='datetime64[s]').reshape(-1),
        np.array(['1959-01-02T00:00:00'], dtype='datetime64[s]'),
    )
    if 'time' in raw.dims:
        raw = raw.isel(time=0)
    # Exact labeled pressure selection is independent of the server's nearest
    # pressure-level helper. Regrid only these four horizontal slices.
    raw = raw[[
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
        'specific_humidity',
    ]].sel(level=850)
    regridder = horizontal_interpolation.ConservativeRegridder(
        spherical_harmonic.Grid.TL31(radius=1.0),
        spherical_harmonic.Grid.TL63(radius=1.0),
    )
    regridded = xarray_utils.regrid_horizontal(raw, regridder)
    regridded = regridded.assign_coords(
        longitude=(regridded.longitude + 180.0) % 360.0 - 180.0)
    regridded = regridded.sortby(['latitude', 'longitude'])
    # TL63 has 128 × 64 real nodes. The declared display cap requires stride 2
    # in both axes, producing 64 × 32 sampled nodes, not an invented 96 × 48.
    assert regridded.sizes['longitude'] == 128
    assert regridded.sizes['latitude'] == 64
    return regridded.isel(latitude=slice(None, None, 2),
                          longitude=slice(None, None, 2))


def test_initial_frame_matches_raw_era5_850_hpa_and_units(
        real_job_results, initial_era5_850):
    sampled = initial_era5_850
    temperature = sampled.temperature.astype(np.float64)
    u = sampled.u_component_of_wind.astype(np.float64)
    v = sampled.v_component_of_wind.astype(np.float64)
    humidity = sampled.specific_humidity.astype(np.float64)
    expected = {
        'temperature': temperature - 273.15,
        'wind_speed': np.sqrt(u * u + v * v),
        'specific_humidity': humidity * 1000.0,
    }
    for observed in real_job_results.jobs:
        grid = observed.completed['grid']
        np.testing.assert_array_equal(grid['latitudes'],
                                      sampled.latitude.values)
        np.testing.assert_array_equal(grid['longitudes'],
                                      sampled.longitude.values)
        assert grid['display_stride'] == {'latitude': 2, 'longitude': 2}
        for field in _FIELDS:
            expected_values = expected[field].transpose('latitude', 'longitude')
            assert expected_values.shape == (32, 64)
            initial_values = np.asarray(observed.frames[0]['fields'][field])
            prediction_values = np.asarray(observed.frames[1]['fields'][field])
            assert initial_values.shape == prediction_values.shape == (2048,)
            # Covers pressure selection, all unit conversions and latitude-major
            # orientation. Float32 regridding and sqrt/hypot rounding may differ
            # at the last numerical digits; neither permits a unit-scale error.
            np.testing.assert_allclose(
                initial_values.reshape(32, 64),
                expected_values.values,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f'raw ERA5 850 hPa {field}',
            )


@pytest.mark.parametrize('job_index', (0, 1, 2))
def test_actual_job_progress_and_frame_availability(real_job_results,
                                                    job_index):
    observed = real_job_results.jobs[job_index]
    snapshots = observed.snapshots
    assert any(snapshot['status'] == 'running' for snapshot in snapshots)
    assert observed.completed['status'] == 'completed'
    previous_frames = 0
    previous_steps = 0
    for snapshot in snapshots:
        assert snapshot['id'] == observed.accepted['id']
        assert snapshot['request'] == observed.accepted['request']
        assert snapshot['error'] is None
        progress = snapshot['progress']
        assert type(progress['completed_steps']) is int
        assert progress['total_steps'] == 1
        assert progress['completed_steps'] in (0, 1)
        assert np.isfinite(progress['fraction'])
        assert previous_steps <= progress['completed_steps']
        previous_steps = progress['completed_steps']
        available = snapshot['available_frames']
        assert type(available) is int and previous_frames <= available <= 2
        previous_frames = available
        assert snapshot['frame_times'] == list(_FRAME_TIMES[:available])
        assert len(snapshot['frame_times']) == available
        if snapshot['status'] == 'queued':
            assert snapshot['phase'] == 'queued'
            assert available == 0
        elif snapshot['status'] == 'running':
            assert snapshot['phase'] in ('loading', 'encoding', 'forecasting')
            assert available == (0 if snapshot['phase'] == 'loading' else 1)
        else:
            assert snapshot['status'] == snapshot['phase'] == 'completed'
            assert available == 2
        if snapshot['status'] == 'completed':
            assert progress == {
                'completed_steps': 1,
                'total_steps': 1,
                'fraction': 1.0
            }
        else:
            # Raw initial ERA5 is available independently of a completed step;
            # loading/encoding must never advertise progress or future frames.
            assert progress == {
                'completed_steps': 0,
                'total_steps': 1,
                'fraction': 0.0
            }
        assert (snapshot['grid'] is None) == (available == 0)
    assert observed.initial_while_running == observed.frames[0]
    pending_status, pending_payload = observed.pending_prediction
    if pending_status == 200:
        assert pending_payload == observed.frames[1]
    else:
        assert pending_status == 404
        assert pending_payload['error']['code'] == 'frame_unavailable'


def test_unfinished_prediction_is_not_fabricated(real_job_results):
    first = real_job_results.jobs[0]
    status, payload = first.pending_prediction
    assert status == 404, 'cold toy job returned a future frame before inference'
    assert payload['error']['code'] == 'frame_unavailable'
    assert payload['error']['message'].strip()
    assert first.initial_while_running is not None
    assert first.initial_while_running['kind'] == 'initial_reanalysis'
    assert first.initial_while_running['lead_hours'] == 0


@pytest.mark.parametrize('job_index', (0, 1, 2))
def test_completed_frames_grid_units_and_provenance(real_job_results,
                                                    job_index):
    result = real_job_results
    observed = result.jobs[job_index]
    completed = observed.completed
    assert result.config['level_hpa'] == 850
    assert {
        field['id']: field['unit'] for field in result.config['fields']
    } == {
        'temperature': '°C',
        'wind_speed': 'm/s',
        'specific_humidity': 'g/kg'
    }
    grid = completed['grid']
    assert grid['level_hpa'] == 850
    assert grid['layout'] == 'latitude-major'
    latitudes = np.asarray(grid['latitudes'])
    longitudes = np.asarray(grid['longitudes'])
    assert latitudes.ndim == longitudes.ndim == 1
    assert 1 < latitudes.size <= 48
    assert 1 < longitudes.size <= 96
    assert np.isfinite(latitudes).all() and np.isfinite(longitudes).all()
    assert np.all(np.diff(latitudes) > 0) and np.all(np.diff(longitudes) > 0)
    assert np.all((-90 <= latitudes) & (latitudes <= 90))
    assert np.all((-180 <= longitudes) & (longitudes < 180))
    assert set(grid['display_stride']) == {'latitude', 'longitude'}
    assert all(
        type(stride) is int and stride >= 1
        for stride in grid['display_stride'].values())
    provenance = completed['provenance']
    assert provenance['engine'] == 'MLX'
    assert provenance['initial_condition'] == 'ERA5 reanalysis'
    assert provenance['initial_time'] == _FRAME_TIMES[0]
    assert provenance['forcing'] == (
        'Historical sea-surface temperature and sea-ice held constant')
    assert provenance['level_hpa'] == 850
    assert provenance['model_id'] == 'toy_tl63'
    assert provenance['model_kind'] == 'toy'
    assert provenance['checkpoint'] == 'bundled:tl63_stochastic_mini.pkl'
    assert provenance['seed'] == (0, 1, 0)[job_index]
    assert provenance['display_grid'].strip()
    warnings = ' '.join(provenance['warnings']).lower()
    assert 'toy' in warnings and 'not live weather' in warnings
    assert '850 hpa' in warnings and 'not surface weather' in warnings
    assert completed['frame_times'] == list(_FRAME_TIMES)
    for index, frame in enumerate(observed.frames):
        assert frame['index'] == frame['lead_hours'] == index
        assert frame['valid_time'] == _FRAME_TIMES[index]
        assert frame['kind'] == ('initial_reanalysis'
                                 if index == 0 else 'prediction')
        assert set(frame['fields']) == set(_FIELDS)
        for field in _FIELDS:
            values = np.asarray(frame['fields'][field])
            assert values.shape == (latitudes.size * longitudes.size,), field
            assert np.issubdtype(values.dtype, np.number), field
            assert np.isfinite(values).all(), field
        assert np.all(np.asarray(frame['fields']['wind_speed']) >= 0)
    assert any(not np.array_equal(observed.frames[0]['fields'][field],
                                  observed.frames[1]['fields'][field])
               for field in _FIELDS), 'forecast replayed the initial reanalysis'


def test_seed_zero_one_zero_reproduces_actual_forecast(real_job_results):
    first, different, repeated = real_job_results.jobs
    assert len({job.completed['id'] for job in real_job_results.jobs}) == 3
    assert first.completed['grid'] == different.completed['grid']
    assert first.completed['grid'] == repeated.completed['grid']
    for field in _FIELDS:
        # Seeds cannot change the source ERA5 frame, and intervening seed 1 must
        # not contaminate the same model's subsequent seed-0 inference.
        np.testing.assert_array_equal(first.frames[0]['fields'][field],
                                      different.frames[0]['fields'][field],
                                      err_msg=f'initial ERA5 {field}')
        np.testing.assert_array_equal(first.frames[0]['fields'][field],
                                      repeated.frames[0]['fields'][field],
                                      err_msg=f'repeated initial ERA5 {field}')
        np.testing.assert_array_equal(
            first.frames[1]['fields'][field],
            repeated.frames[1]['fields'][field],
            err_msg=f'repeated seed-0 prediction {field}')


def test_changed_seed_changes_actual_prediction(real_job_results,
                                                record_property):
    first, different, _ = real_job_results.jobs
    differences = []
    for field in _FIELDS:
        seed_zero = np.asarray(first.frames[1]['fields'][field])
        seed_one = np.asarray(different.frames[1]['fields'][field])
        maximum_difference = float(np.max(np.abs(seed_zero - seed_one)))
        record_property(f'{field}_seed_0_1_max_absolute_difference',
                        maximum_difference)
        differences.append(maximum_difference)
    # This compares returned weather, not merely seed metadata or hidden RNG
    # keys: a stochastic control must have a genuine numerical forecast effect.
    assert any(difference > 0 for difference in differences), (
        'bundled stochastic toy has no seed-0/seed-1 prediction difference; '
        'do not replace this with a metadata-only assertion')


@pytest.mark.parametrize('case, code', (
    ('past_horizon', 'frame_unavailable'),
    ('negative_index', 'not_found'),
    ('fractional_index', 'not_found'),
    ('unknown_job', 'job_not_found'),
    ('unknown_job_frame', 'job_not_found'),
))
def test_frame_consumer_errors_for_real_jobs(real_job_results, case, code):
    status, payload = real_job_results.consumer_errors[case]
    assert status == 404
    assert set(payload) == {'error'}
    assert payload['error']['code'] == code
    assert isinstance(payload['error']['message'], str)
    assert payload['error']['message'].strip()
