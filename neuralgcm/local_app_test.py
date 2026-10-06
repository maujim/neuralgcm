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
"""Real loopback HTTP consumer tests; no forecast is submitted or mocked."""

from contextlib import contextmanager
from html.parser import HTMLParser
import http.client
import json
from pathlib import PurePosixPath
import threading
from urllib.parse import urljoin, urlsplit

from neuralgcm import local_app
import pytest


@contextmanager
def _serving(web_root=None):
    server = local_app.create_server(port=0, web_root=web_root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.fixture(scope='module')
def http_server():
    with _serving() as server:
        yield server


def _request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection('127.0.0.1',
                                            server.server_port,
                                            timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def _post(server, body, path='/api/jobs', headers=None):
    request_headers = {'Content-Type': 'application/json'}
    request_headers.update(headers or {})
    if not isinstance(body, bytes):
        body = json.dumps(body).encode()
    return _request(server, 'POST', path, body, request_headers)


def _error(response, status, code):
    actual_status, headers, raw = response
    assert actual_status == status
    assert headers['Content-Type'] == 'application/json; charset=utf-8'
    assert int(headers['Content-Length']) == len(raw)
    assert headers['Cache-Control'] == 'no-store'
    value = json.loads(raw)
    assert set(value) == {'error'}
    assert set(value['error']) == {'code', 'message'}
    assert value['error']['code'] == code
    assert isinstance(value['error']['message'], str)
    assert value['error']['message']
    return value['error']['message']


def test_config_is_stable_and_describes_historical_catalog(http_server):
    first = _request(http_server, 'GET', '/api/config')
    second = _request(http_server, 'GET', '/api/config')
    assert first[0] == second[0] == 200
    assert first[2] == second[2]
    status, headers, raw = first
    assert status == 200
    assert headers['Content-Type'] == 'application/json; charset=utf-8'
    assert headers['Cache-Control'] == 'no-store'
    config = json.loads(raw)
    assert config['schema_version'] == 1
    assert config['defaults'] == {
        'model_id': 'toy_tl63',
        'duration_hours': 1,
        'seed': 0
    }
    assert config['duration_hours'] == [1, 3, 6, 12, 24]
    assert config['output_interval_hours'] == 1
    assert config['level_hpa'] == 850
    assert config['limits'] == {'max_scenarios': 2}
    assert {
        field['id']: field['unit'] for field in config['fields']
    } == {
        'temperature': '°C',
        'wind_speed': 'm/s',
        'specific_humidity': 'g/kg'
    }
    models = {model['id']: model for model in config['models']}
    expected_paths = {
        'deterministic_0_7_deg': 'v1/deterministic_0_7_deg.pkl',
        'deterministic_1_4_deg': 'v1/deterministic_1_4_deg.pkl',
        'deterministic_2_8_deg': 'v1/deterministic_2_8_deg.pkl',
        'stochastic_1_4_deg': 'v1/stochastic_1_4_deg.pkl',
        'stochastic_precip_2_8_deg': 'v1_precip/stochastic_precip_2_8_deg.pkl',
        'stochastic_evap_2_8_deg': 'v1_precip/stochastic_evap_2_8_deg.pkl',
        'toy_tl63': 'bundled:tl63_stochastic_mini.pkl',
    }
    assert set(models) == set(expected_paths)
    assert len(config['models']) == len(models)
    for model_id, model in models.items():
        assert model['checkpoint'] == expected_paths[model_id]
        assert model['label']
        assert model['kind'] == ('toy'
                                 if model_id == 'toy_tl63' else 'production')
        assert model['stochastic'] == model_id.startswith(
            ('stochastic_', 'toy_'))
        assert type(model['available']) is bool
        if model['available']:
            assert model['unavailable_reason'] is None
        else:
            assert isinstance(model['unavailable_reason'], str)
            assert model['unavailable_reason'].strip()
    assert models['toy_tl63']['available']
    provenance = config['provenance']
    assert provenance['engine'] == 'MLX'
    assert provenance['initial_condition'] == 'ERA5 reanalysis'
    assert provenance['initial_time'] == '1959-01-02T00:00:00Z'
    assert provenance['level_hpa'] == 850


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
@pytest.mark.parametrize('method', ['GET', 'POST'])
def test_invalid_host_is_rejected(http_server, host, method):
    host = host.format(port=http_server.server_port)
    path = '/api/config' if method == 'GET' else '/api/jobs'
    _error(_request(http_server, method, path, b'{}', {'Host': host}), 400,
           'invalid_host')


@pytest.mark.parametrize('origin', [
    'https://127.0.0.1:{port}',
    'http://example.com:{port}',
    'http://127.0.0.1:1',
    'http://localhost:{port}',
    'http://127.0.0.1:{port}/',
    'http://127.0.0.1:{port}?x',
    'http://127.0.0.1:{port}#x',
    'null',
    'http://user@127.0.0.1:{port}',
])
def test_cross_origin_mutation_is_rejected(http_server, origin):
    _error(
        _post(http_server, {},
              headers={'Origin': origin.format(port=http_server.server_port)}),
        403, 'cross_origin')


@pytest.mark.parametrize('hostname', ['127.0.0.1', 'localhost'])
def test_matching_origin_reaches_schema_validation(http_server, hostname):
    host = f'{hostname}:{http_server.server_port}'
    _error(
        _post(http_server, {},
              headers={
                  'Host': host,
                  'Origin': f'http://{host}'
              }), 400, 'invalid_request')


@pytest.mark.parametrize('body', [
    b'',
    b'{',
    b'{"model_id":',
    b'\xff',
    b'{"model_id":"toy_tl63","duration_hours":1,"seed":NaN}',
    b'{"model_id":"toy_tl63","duration_hours":1,"seed":Infinity}',
    b'{"model_id":"toy_tl63","duration_hours":1,"seed":-Infinity}',
    b'{"model_id":"toy_tl63","duration_hours":1,"seed":0,"seed":1}',
    [],
    None,
    True,
    1,
    'toy_tl63',
    {},
    {
        'model_id': 'toy_tl63',
        'duration_hours': 1
    },
    {
        'model_id': 'toy_tl63',
        'duration_hours': 1,
        'seed': 0,
        'checkpoint': '/private/model.pkl'
    },
])
def test_malformed_json_and_schema_are_rejected(http_server, body):
    _error(_post(http_server, body), 400, 'invalid_request')


@pytest.mark.parametrize('field,value', [
    ('model_id', None),
    ('model_id', True),
    ('model_id', 1),
    ('model_id', []),
    ('model_id', {}),
    ('duration_hours', True),
    ('duration_hours', False),
    ('duration_hours', 1.0),
    ('duration_hours', '1'),
    ('duration_hours', None),
    ('duration_hours', []),
    ('duration_hours', 0),
    ('duration_hours', 2),
    ('duration_hours', -1),
    ('duration_hours', 25),
    ('seed', True),
    ('seed', False),
    ('seed', 0.0),
    ('seed', '0'),
    ('seed', None),
    ('seed', []),
    ('seed', {}),
    ('seed', -1),
    ('seed', 4294967296),
    ('seed', 10**100),
])
def test_field_types_and_numeric_bounds_are_rejected(http_server, field, value):
    body = {'model_id': 'toy_tl63', 'duration_hours': 1, 'seed': 0}
    body[field] = value
    _error(_post(http_server, body), 400, 'invalid_request')


@pytest.mark.parametrize('model_id', [
    'missing_model',
    '/private/checkpoint.pkl',
    '../models/v1/model.pkl',
    'https://example.com/model.pkl',
    'TOY_TL63',
    '',
])
def test_model_lookup_is_an_allowlist(http_server, model_id):
    _error(
        _post(http_server, {
            'model_id': model_id,
            'duration_hours': 1,
            'seed': 0
        }), 400, 'invalid_request')


def test_disabled_models_are_rejected_as_advertised(http_server):
    config = json.loads(_request(http_server, 'GET', '/api/config')[2])
    for model in config['models']:
        if not model['available']:
            response = _post(http_server, {
                'model_id': model['id'],
                'duration_hours': 1,
                'seed': 0
            })
            assert _error(response, 409,
                          'model_unavailable') == model['unavailable_reason']


def test_query_cannot_supply_or_override_job_body(http_server):
    query = '?model_id=toy_tl63&duration_hours=1&seed=0&checkpoint=/private/a.pkl'
    _error(_post(http_server, {}, path='/api/jobs' + query), 400,
           'invalid_request')
    _error(
        _post(http_server, {
            'model_id': 'unknown',
            'duration_hours': 1,
            'seed': 0
        },
              path='/api/jobs' + query), 400, 'invalid_request')


@pytest.mark.parametrize('path,code', [
    ('/api/jobs/no-such-job', 'job_not_found'),
    ('/api/jobs/no-such-job/frames/0', 'job_not_found'),
    ('/api/jobs/no-such-job/frames/-1', 'job_not_found'),
    ('/api/jobs/no-such-job/frames/999999999', 'job_not_found'),
    ('/api/jobs/', 'not_found'),
    ('/api/jobs/no-such-job/frames', 'not_found'),
    ('/api/jobs/no-such-job/frames/0/extra', 'not_found'),
    ('/api/unknown', 'not_found'),
])
def test_unknown_jobs_frames_and_routes_return_json_errors(
        http_server, path, code):
    _error(_request(http_server, 'GET', path), 404, code)


def test_post_unknown_endpoint_and_media_type(http_server):
    _error(_post(http_server, {}, path='/api/unknown'), 404, 'not_found')
    _error(
        _request(http_server, 'POST', '/api/jobs', b'{}',
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
def test_invalid_body_framing_closes_connection(http_server, headers, status,
                                                code):
    response = _post(http_server, b'{}', headers=headers)
    _error(response, status, code)
    assert response[1]['Connection'] == 'close'


def test_request_body_size_boundary(http_server):
    # Invalid schema prevents inference, including at the accepted size limit.
    body = b'{}' + b' ' * (16384 - 2)
    _error(_post(http_server, body), 400, 'invalid_request')
    response = _post(http_server, body + b' ')
    _error(response, 413, 'request_too_large')
    assert response[1]['Connection'] == 'close'


def test_missing_content_length_is_a_json_error(http_server):
    connection = http.client.HTTPConnection('127.0.0.1',
                                            http_server.server_port,
                                            timeout=5)
    try:
        connection.putrequest('POST', '/api/jobs')
        connection.putheader('Content-Type', 'application/json')
        connection.endheaders()
        response = connection.getresponse()
        _error((response.status, dict(response.getheaders()), response.read()),
               400, 'invalid_request')
    finally:
        connection.close()


def test_static_traversal_symlinks_and_errors_do_not_expose_files(tmp_path):
    web_root = tmp_path / 'web'
    web_root.mkdir()
    private = tmp_path / 'private-secret.txt'
    private.write_text('PRIVATE SENTINEL: not browser content')
    (web_root / 'escape.txt').symlink_to(private)
    with _serving(web_root) as server:
        for path in [
                '/../private-secret.txt',
                '/%2e%2e/private-secret.txt',
                '/%2e%2e%2fprivate-secret.txt',
                '/escape.txt',
                '/missing.txt',
                '/%00',
                '/bad\\path',
        ]:
            response = _request(server, 'GET', path)
            _error(response, 404, 'not_found')
            assert str(tmp_path).encode() not in response[2]
            assert private.read_bytes() not in response[2]
            assert b'Traceback' not in response[2]


class _ResourceLinks(HTMLParser):

    def __init__(self):
        super().__init__()
        self.resources = []
        self.base_urls = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'base' and attrs.get('href'):
            self.base_urls.append(attrs['href'])
        if tag in ('script', 'img', 'iframe', 'source', 'audio', 'video'):
            if attrs.get('src'):
                self.resources.append(attrs['src'])
        if tag == 'link' and attrs.get('href'):
            self.resources.append(attrs['href'])


def test_served_browser_entry_resources_are_local_and_have_correct_mime(
        http_server):
    status, headers, raw = _request(http_server, 'GET', '/')
    assert status == 200
    assert headers['Content-Type'] == 'text/html; charset=utf-8'
    assert headers['X-Content-Type-Options'] == 'nosniff'
    assert int(headers['Content-Length']) == len(raw)
    document = _ResourceLinks()
    document.feed(raw.decode('utf-8'))
    assert document.resources
    origin = f'http://127.0.0.1:{http_server.server_port}/'
    for resource in document.base_urls + document.resources:
        url = urlsplit(urljoin(origin, resource))
        assert url.scheme == 'http'
        assert url.netloc == urlsplit(origin).netloc
    for resource in document.resources:
        url = urlsplit(urljoin(origin, resource))
        status, headers, body = _request(http_server, 'GET', url.path)
        assert status == 200
        assert body
        assert int(headers['Content-Length']) == len(body)
        assert headers['X-Content-Type-Options'] == 'nosniff'
        suffix = PurePosixPath(url.path).suffix
        if suffix == '.css':
            assert headers['Content-Type'] == 'text/css; charset=utf-8'
        elif suffix == '.js':
            assert headers['Content-Type'].split(';')[0] in (
                'text/javascript', 'application/javascript')
        assert headers['Cache-Control'] == 'no-cache'
