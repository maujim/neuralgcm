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
"""Serial HTTP queue cancellation with real MLX forecasts, not model doubles."""

import json
import threading
import time

from neuralgcm.local_app_test import _error
from neuralgcm.local_app_test import _post
from neuralgcm.local_app_test import _request
from neuralgcm.local_app_test import _serving
import pytest

pytestmark = pytest.mark.integration

_CLEAR_PATH = '/api/jobs/clear'
_JOB_TIMEOUT_SECONDS = 1800


@pytest.fixture(scope='module')
def http_server():
    with _serving() as server:
        yield server


def _json_response(response, status):
    actual_status, headers, raw = response
    assert actual_status == status, raw
    assert headers['Content-Type'] == 'application/json; charset=utf-8'
    assert headers['Cache-Control'] == 'no-store'
    assert int(headers['Content-Length']) == len(raw)

    def reject_nonfinite(token):
        raise ValueError(f'Non-finite HTTP JSON: {token}')

    return json.loads(raw, parse_constant=reject_nonfinite)


def _clear(server):
    result = _json_response(_post(server, {}, path=_CLEAR_PATH), 200)
    assert set(result) == {'cleared_count'}
    assert type(result['cleared_count']) is int
    assert result['cleared_count'] >= 0
    return result['cleared_count']


@pytest.mark.parametrize('hostname', ['127.0.0.1', 'localhost'])
def test_empty_clear_and_repeated_clear_are_successful(http_server, hostname):
    host = f'{hostname}:{http_server.server_port}'
    headers = {'Host': host, 'Origin': f'http://{host}'}
    for _ in range(2):
        response = _post(http_server, {}, path=_CLEAR_PATH, headers=headers)
        assert _json_response(response, 200) == {'cleared_count': 0}


@pytest.mark.parametrize('host', [
    'example.com:{port}',
    '127.0.0.1',
    '127.0.0.1:1',
    '127.0.0.1:{port}@example.com',
    'localhost:{port}/escape',
    '127.0.0.1:{port}?query',
    '127.0.0.1:{port}#fragment',
    '[::1]:{port}',
    '127.0.0.1:invalid',
    '',
])
def test_clear_rejects_invalid_host(http_server, host):
    _error(
        _post(http_server, {},
              path=_CLEAR_PATH,
              headers={'Host': host.format(port=http_server.server_port)}), 400,
        'invalid_host')


@pytest.mark.parametrize('origin', [
    'https://127.0.0.1:{port}',
    'http://example.com:{port}',
    'http://127.0.0.1:1',
    'http://localhost:{port}',
    'http://127.0.0.1:{port}/',
    'http://127.0.0.1:{port}?x',
    'http://127.0.0.1:{port}#x',
    'http://user@127.0.0.1:{port}',
    'null',
])
def test_clear_rejects_cross_origin_mutation(http_server, origin):
    _error(
        _post(http_server, {},
              path=_CLEAR_PATH,
              headers={'Origin': origin.format(port=http_server.server_port)}),
        403, 'cross_origin')


@pytest.mark.parametrize('body', [
    b'',
    b'{',
    b'\xff',
    b'{"x":NaN}',
    b'{"x":Infinity}',
    b'{"x":-Infinity}',
    b'{"x":0,"x":1}',
    [],
    None,
    True,
    1,
    'clear',
    {
        'clear': True
    },
    {
        'model_id': 'toy_tl63',
        'duration_hours': 1,
        'seed': 0
    },
])
def test_clear_requires_exact_empty_object(http_server, body):
    _error(_post(http_server, body, path=_CLEAR_PATH), 400, 'invalid_request')


def test_clear_requires_json_media_type(http_server):
    _error(
        _request(http_server, 'POST', _CLEAR_PATH, b'{}',
                 {'Content-Type': 'text/plain'}), 415, 'unsupported_media_type')


@pytest.mark.parametrize('headers,status,code', [
    ({
        'Content-Length': '-1'
    }, 400, 'invalid_request'),
    ({
        'Content-Length': 'not-a-number'
    }, 400, 'invalid_request'),
    ({
        'Content-Length': '16385'
    }, 413, 'request_too_large'),
    ({
        'Content-Length': '2',
        'Transfer-Encoding': 'chunked'
    }, 400, 'invalid_request'),
])
def test_clear_rejects_invalid_body_framing(http_server, headers, status, code):
    response = _post(http_server, b'{}', path=_CLEAR_PATH, headers=headers)
    _error(response, status, code)
    assert response[1]['Connection'] == 'close'


def test_clear_body_size_boundary(http_server):
    body = b'{}' + b' ' * (16384 - 2)
    assert _json_response(_post(http_server, body, path=_CLEAR_PATH), 200) == {
        'cleared_count': 0
    }
    response = _post(http_server, body + b' ', path=_CLEAR_PATH)
    _error(response, 413, 'request_too_large')
    assert response[1]['Connection'] == 'close'


def _snapshot(server, job):
    return _json_response(_request(server, 'GET', f'/api/jobs/{job["id"]}'),
                          200)


def _frame(server, job, index):
    path = f'/api/jobs/{job["id"]}/frames/{index}'
    return _json_response(_request(server, 'GET', path), 200)


def _submit(server, duration, seed):
    body = {'model_id': 'toy_tl63', 'duration_hours': duration, 'seed': seed}
    return _json_response(_post(server, body), 202)


def _wait_for(server, job, predicate):
    deadline = time.monotonic() + _JOB_TIMEOUT_SECONDS
    poll_interval = threading.Event()
    while True:
        observed = _snapshot(server, job)
        assert observed['status'] != 'failed', observed
        if predicate(observed):
            return observed
        assert time.monotonic() < deadline, observed
        assert observed['status'] not in ('completed', 'cancelled'), observed
        # This is a bounded HTTP-state poll, not a timing assumption about MLX.
        poll_interval.wait(0.1)


def _assert_cancelled(server, job):
    observed = _snapshot(server, job)
    assert observed['status'] == observed['phase'] == 'cancelled'
    assert observed['request'] == job['request']
    assert observed['progress'] == {
        'completed_steps': 0,
        'total_steps': job['request']['duration_hours'],
        'fraction': 0.0,
    }
    assert observed['available_frames'] == 0
    assert observed['frame_times'] == []
    assert observed['grid'] is None
    assert observed['error'] is None
    _error(_request(server, 'GET', f'/api/jobs/{job["id"]}/frames/0'), 404,
           'frame_unavailable')
    return observed


@pytest.fixture(scope='module')
def real_queue_results(http_server):
    """One real 24-hour active run plus one real one-hour survivor, serially.

    The only instrumentation is an Event at the actual stdlib Queue.get
    boundary. It holds one dequeued ID *before* the worker's queued-to-running
    transition, letting HTTP clear win that race deterministically. Inference,
    model loading, forecast frames, job records, and HTTP handlers are untouched.
    """
    pytest.importorskip('mlx.core')
    server = http_server
    original_get = server.work.get
    captured = threading.Event()
    release = threading.Event()
    target = None

    def gated_get(*args, **kwargs):
        job_id = original_get(*args, **kwargs)
        if job_id == target and target is not None:
            captured.set()
            release.wait(_JOB_TIMEOUT_SECONDS)
        return job_id

    server.work.get = gated_get
    try:
        active = _submit(server, 24, 0)
        running = _wait_for(
            server, active, lambda job: job['status'] == 'running' and job[
                'available_frames'] >= 1)
        initial_frame = _frame(server, active, 0)
        queued = []
        # Discover the consumer's actual admission bound rather than asserting
        # a private limit or a helper's count. Every accepted job is real work.
        for seed in range(1, 65):
            response = _post(server, {
                'model_id': 'toy_tl63',
                'duration_hours': 1,
                'seed': seed
            })
            if response[0] == 503:
                _error(response, 503, 'capacity_reached')
                break
            queued.append(_json_response(response, 202))
        else:
            pytest.fail('No bounded HTTP admission limit within 64 queued jobs')
        assert queued
        assert _snapshot(server, active)['status'] == 'running'
        for job in queued:
            assert _snapshot(server, job)['status'] == 'queued'
        cleared_count = _clear(server)
        cancelled = tuple(_assert_cancelled(server, job) for job in queued)
        assert cleared_count == len(queued)
        assert _clear(server) == 0

        # The worker is still occupied, so acceptance proves slots are reclaimed
        # by clear itself, not by waiting for it to consume cancelled entries.
        replacement = _submit(server, 1, 100)
        target = replacement['id']
        assert _snapshot(server, replacement)['status'] == 'queued'
        reused_while_running = _snapshot(server, active)
        assert reused_while_running['status'] == 'running'
        assert _frame(server, active, 0) == initial_frame

        completed = _wait_for(server, active,
                              lambda job: job['status'] == 'completed')
        assert captured.wait(30), 'Worker did not dequeue the replacement job'
        assert _snapshot(server, replacement)['status'] == 'queued'
        final_frame = _frame(server, active, 24)
        assert _clear(server) == 1
        captured_cancelled = _assert_cancelled(server, replacement)
        assert _clear(server) == _clear(server) == 0
        retained_completed = _snapshot(server, active)
        retained_initial = _frame(server, active, 0)
        retained_final = _frame(server, active, 24)

        release.set()
        # Completion of later real work proves the worker passed the captured
        # cancelled entry. Its retained GET/frame responses must still show
        # that it never executed, not merely that clear initially changed it.
        survivor = _submit(server, 1, 101)
        survivor_completed = _wait_for(server, survivor,
                                       lambda job: job['status'] == 'completed')
        survivor_frame = _frame(server, survivor, 1)
        after_survivor = _assert_cancelled(server, replacement)
        return {
            'running': running,
            'reused_while_running': reused_while_running,
            'cancelled': cancelled,
            'cleared_count': cleared_count,
            'completed': completed,
            'retained_completed': retained_completed,
            'initial_frame': initial_frame,
            'retained_initial': retained_initial,
            'final_frame': final_frame,
            'retained_final': retained_final,
            'captured_cancelled': captured_cancelled,
            'after_survivor': after_survivor,
            'survivor_completed': survivor_completed,
            'survivor_frame': survivor_frame,
        }
    finally:
        try:
            # Failure cleanup cancels only genuinely queued work; a running
            # forecast still finishes normally before _serving joins its worker.
            _clear(server)
        finally:
            release.set()
            server.work.get = original_get


def test_clear_reports_actual_queued_count_and_reclaims_capacity(
        real_queue_results):
    observed = real_queue_results
    assert observed['cleared_count'] == len(observed['cancelled']) > 0
    assert observed['reused_while_running']['status'] == 'running'
    assert observed['reused_while_running']['progress']['completed_steps'] >= (
        observed['running']['progress']['completed_steps'])


def test_active_real_forecast_completes_without_losing_frames(
        real_queue_results):
    observed = real_queue_results
    completed = observed['completed']
    assert completed['status'] == completed['phase'] == 'completed'
    assert completed['progress'] == {
        'completed_steps': 24,
        'total_steps': 24,
        'fraction': 1.0
    }
    assert completed['available_frames'] == 25
    assert completed['error'] is None
    assert observed['initial_frame']['kind'] == 'initial_reanalysis'
    assert observed['final_frame']['kind'] == 'prediction'
    assert observed['final_frame']['lead_hours'] == 24
    assert observed['final_frame']['valid_time'] == '1959-01-03T00:00:00Z'
    assert observed['final_frame']['fields'] != observed['initial_frame'][
        'fields']


def test_clear_retains_completed_status_and_actual_forecast_frames(
        real_queue_results):
    observed = real_queue_results
    assert observed['retained_completed'] == observed['completed']
    assert observed['retained_initial'] == observed['initial_frame']
    assert observed['retained_final'] == observed['final_frame']


def test_dequeued_cancelled_job_never_executes_and_worker_continues(
        real_queue_results):
    observed = real_queue_results
    assert observed['after_survivor'] == observed['captured_cancelled']
    survivor = observed['survivor_completed']
    assert survivor['status'] == survivor['phase'] == 'completed'
    assert survivor['progress'] == {
        'completed_steps': 1,
        'total_steps': 1,
        'fraction': 1.0
    }
    assert survivor['available_frames'] == 2
    assert survivor['error'] is None
    assert observed['survivor_frame']['kind'] == 'prediction'
    assert observed['survivor_frame']['lead_hours'] == 1
