"""Offline loopback web application for genuine local NeuralGCM MLX forecasts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import mimetypes
from pathlib import Path
import pickle
import queue
import threading
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import jax
import numpy as np

_DURATION_HOURS = (1, 3, 6, 12, 24)
_MAX_JOBS = 8
_MAX_BODY_BYTES = 16 * 1024
_MODEL_CATALOG = (
    ('deterministic_0_7_deg', 'Deterministic 0.7°', False,
     'v1/deterministic_0_7_deg.pkl'),
    ('deterministic_1_4_deg', 'Deterministic 1.4°', False,
     'v1/deterministic_1_4_deg.pkl'),
    ('deterministic_2_8_deg', 'Deterministic 2.8°', False,
     'v1/deterministic_2_8_deg.pkl'),
    ('stochastic_1_4_deg', 'Stochastic 1.4°', True,
     'v1/stochastic_1_4_deg.pkl'),
    ('stochastic_precip_2_8_deg', 'Stochastic precipitation 2.8°', True,
     'v1_precip/stochastic_precip_2_8_deg.pkl'),
    ('stochastic_evap_2_8_deg', 'Stochastic evaporation 2.8°', True,
     'v1_precip/stochastic_evap_2_8_deg.pkl'),
    ('toy_tl63', 'Bundled TL63 stochastic toy', True,
     'bundled:tl63_stochastic_mini.pkl'),
)


def _iso(value: Any) -> str:
    stamp = np.datetime64(value, 's').astype(
        dt.datetime).replace(tzinfo=dt.timezone.utc)
    return stamp.isoformat().replace('+00:00', 'Z')


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('non-finite number in response')
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        f'cannot encode response value of type {type(value).__name__}')


def _catalog() -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parent.parent / 'models'
    result = []
    for model_id, label, stochastic, checkpoint in _MODEL_CATALOG:
        if model_id == 'toy_tl63':
            available = True
            reason = None
        else:
            available = (root / checkpoint).is_file()
            reason = None if available else 'Checkpoint file is not installed locally.'
        result.append({
            'id': model_id,
            'label': label,
            'kind': 'toy' if model_id == 'toy_tl63' else 'production',
            'stochastic': stochastic,
            'available': available,
            'unavailable_reason': reason,
            'checkpoint': checkpoint,
        })
    return result


class _ForecastServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], web_root: Path):
        super().__init__(address, _Handler)
        self.web_root = web_root.resolve()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.job_lock = threading.RLock()
        self.work: queue.Queue[str | None] = queue.Queue()
        self.models: dict[str, Any] = {}
        self._initial_snapshot: dict[str, Any] | None = None
        self._initial_lock = threading.Lock()
        self._closing = False
        self._worker_thread = threading.Thread(target=_worker,
                                               args=(self,),
                                               name='neuralgcm-inference',
                                               daemon=True)
        self._worker_thread.start()

    def server_close(self) -> None:
        # Place the sentinel after all accepted jobs, then join to ensure inference
        # and device-backed model references are released before returning.
        with self.job_lock:
            if not self._closing:
                self._closing = True
                self.work.put(None)
        if self._worker_thread is not threading.current_thread():
            self._worker_thread.join()
        self.models.clear()
        super().server_close()

    def _clear_queued_jobs(self) -> int | None:
        """Cancel all queued jobs and remove their work items atomically."""
        with self.job_lock:
            if self._closing:
                return None
            queued = [
                job for job in self.jobs.values() if job['status'] == 'queued'
            ]
            for job in queued:
                job['status'], job['phase'] = 'cancelled', 'cancelled'

            # A worker may already have dequeued an item but not yet acquired
            # job_lock. Marking jobs first ensures it observes cancellation.
            retained: list[str | None] = []
            while True:
                try:
                    job_id = self.work.get_nowait()
                except queue.Empty:
                    break
                try:
                    if job_id is None:
                        retained.append(job_id)
                    else:
                        job = self.jobs.get(job_id)
                        if job is None or job['status'] != 'cancelled':
                            retained.append(job_id)
                finally:
                    self.work.task_done()
            for job_id in retained:
                self.work.put_nowait(job_id)
            return len(queued)


def create_server(
    host: str = '127.0.0.1',
    port: int = 8765,
    web_root: Path | None = None,
) -> _ForecastServer:
    """Bind and return the ready HTTP server; port 0 requests an ephemeral port."""
    if host != '127.0.0.1':
        raise ValueError(
            'host must be 127.0.0.1; the application only binds to IPv4 loopback'
        )
    if not 0 <= port < 65536:
        raise ValueError('port must be between 0 and 65535')
    if web_root is None:
        web_root = Path(__file__).resolve().parent.parent / 'web'
    return _ForecastServer((host, port), web_root)


def _provenance(model_id: str | None = None,
                seed: int | None = None) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        'engine':
            'MLX',
        'initial_condition':
            'ERA5 reanalysis',
        'initial_time':
            '1959-01-02T00:00:00Z',
        'forcing':
            'Historical sea-surface temperature and sea-ice held constant',
        'level_hpa':
            850,
        'warnings': [
            'Historical reanalysis experiment, not live weather or a public forecast.',
            'Values are on the 850 hPa pressure surface (about 1.5 km altitude), not surface weather.',
            'Ocean forcing remains fixed to the historical sample; no future observations are implied.',
        ],
    }
    if model_id is not None:
        m = next(m for m in _catalog() if m['id'] == model_id)
        provenance.update({
            'model_id':
                model_id,
            'model_kind':
                m['kind'],
            'checkpoint':
                m['checkpoint'],
            'seed':
                seed,
            'display_grid':
                'Exact model coordinates, sorted and normalized; integer-strided to at most 48 latitudes × 96 longitudes.',
        })
        if m['kind'] == 'toy':
            provenance['warnings'].append(
                'The bundled mini model is an explicitly labeled toy.')
        if not m['stochastic']:
            provenance['warnings'].append(
                'This deterministic model has no stochastic effect from the submitted seed.'
            )
    return provenance


def _config() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'models': _catalog(),
        'defaults': {
            'model_id': 'toy_tl63',
            'duration_hours': 1,
            'seed': 0
        },
        'duration_hours': list(_DURATION_HOURS),
        'output_interval_hours': 1,
        'level_hpa': 850,
        'fields': [
            {
                'id': 'temperature',
                'label': 'Temperature',
                'unit': '°C'
            },
            {
                'id': 'wind_speed',
                'label': 'Wind speed',
                'unit': 'm/s'
            },
            {
                'id': 'specific_humidity',
                'label': 'Specific humidity',
                'unit': 'g/kg'
            },
        ],
        'limits': {
            'max_scenarios': 2
        },
        'provenance': _provenance(),
    }


def _progress(job: dict[str, Any]) -> dict[str, Any]:
    done, total = job['completed_steps'], job['request']['duration_hours']
    return {
        'completed_steps':
            done,
        'total_steps':
            total,
        'fraction':
            1.0 if job['status'] == 'completed' else
            (done / total if job['status'] == 'running' else 0.0)
    }


def _job_response(job: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': job['id'],
        'status': job['status'],
        'phase': job['phase'],
        'progress': _progress(job),
        'request': job['request'],
        'available_frames': len(job['frames']),
        'frame_times': [frame['valid_time'] for frame in job['frames']],
        'grid': job['grid'],
        'provenance': job['provenance'],
        'error': job['error'],
    }


def _coordinates(ds: Any) -> tuple[np.ndarray, np.ndarray, int]:
    levels = np.asarray(ds['level'].values, dtype=np.float64)
    if levels.ndim != 1 or not np.isfinite(levels).all():
        raise ValueError('model output has invalid pressure coordinates')
    hpa_levels = levels / 100.0 if np.nanmax(np.abs(levels)) > 2000 else levels
    level_index = int(np.argmin(np.abs(hpa_levels - 850.0)))
    if abs(float(hpa_levels[level_index]) - 850.0) > 1.0:
        raise ValueError(
            'model output does not contain the required 850 hPa level')
    lat = np.asarray(ds['latitude'].values, dtype=np.float64)
    lon = ((np.asarray(ds['longitude'].values, dtype=np.float64) + 180.0) %
           360.0) - 180.0
    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)
    lat, lon = lat[lat_order], lon[lon_order]
    stride_lat = max(1, math.ceil(len(lat) / 48))
    stride_lon = max(1, math.ceil(len(lon) / 96))
    return lat[::stride_lat], lon[::stride_lon], level_index


def _grid(ds: Any) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    latitudes, longitudes, _ = _coordinates(ds)
    full_lat = np.asarray(ds.latitude.values, dtype=np.float64)
    full_lon = ((np.asarray(ds.longitude.values, dtype=np.float64) + 180.0) %
                360.0) - 180.0
    lat_indices = np.argsort(full_lat)[::max(1, math.ceil(len(full_lat) / 48))]
    lon_indices = np.argsort(full_lon)[::max(1, math.ceil(len(full_lon) / 96))]
    return ({
        'latitudes': latitudes.tolist(),
        'longitudes': longitudes.tolist(),
        'level_hpa': 850,
        'layout': 'latitude-major',
        'display_stride': {
            'latitude': max(1, math.ceil(len(full_lat) / 48)),
            'longitude': max(1, math.ceil(len(full_lon) / 96))
        }
    }, lat_indices, lon_indices)


def _frame_fields(ds: Any, lat_indices: np.ndarray,
                  lon_indices: np.ndarray) -> dict[str, list[float]]:
    _, _, level = _coordinates(ds)

    def read_field(name: str) -> np.ndarray:
        variable = ds[name]
        if 'time' in variable.dims:
            variable = variable.isel(time=0)
        variable = variable.isel(level=level).transpose('latitude', 'longitude')
        values = np.asarray(variable.values,
                            dtype=np.float64)[np.ix_(lat_indices, lon_indices)]
        if not np.isfinite(values).all():
            raise ValueError(f'non-finite values in forecast field {name}')
        return values

    temperature = read_field('temperature') - 273.15
    u = read_field('u_component_of_wind')
    v = read_field('v_component_of_wind')
    humidity = read_field('specific_humidity') * 1000.0
    return {
        'temperature': temperature.ravel(order='C').tolist(),
        'wind_speed': np.hypot(u, v).ravel(order='C').tolist(),
        'specific_humidity': humidity.ravel(order='C').tolist(),
    }


def _model_for(server: _ForecastServer, model_id: str) -> Any:
    cached = server.models.get(model_id)
    if cached is not None:
        return cached
    server.models.clear()
    from neuralgcm import demo
    from neuralgcm.mlx import PressureLevelModel
    if model_id == 'toy_tl63':
        checkpoint = demo.load_checkpoint_tl63_stochastic()
    else:
        entry = next(entry for entry in _MODEL_CATALOG if entry[0] == model_id)
        checkpoint = Path(
            __file__).resolve().parent.parent / 'models' / entry[3]
        with checkpoint.open('rb') as handle:
            checkpoint = pickle.load(handle)
    model = PressureLevelModel.from_checkpoint(checkpoint)
    server.models[model_id] = model
    return model


def _execute(server: _ForecastServer, job_id: str) -> None:
    job = server.jobs[job_id]
    model_id = job['request']['model_id']
    seed = job['request']['seed']
    duration = job['request']['duration_hours']
    try:
        import mlx.core as mx
        from neuralgcm import demo, inference_logging

        with server.job_lock:
            job['phase'] = 'loading'
        model = _model_for(server, model_id)
        dataset = demo.load_data(model.data_coords)
        epoch = np.datetime64(dataset.time.values[0], 'h')
        raw_time = dataset.isel(time=0)
        raw_time = raw_time[list(model.input_variables)]
        initial_grid, lat_ids, lon_ids = _grid(raw_time)
        initial_fields = _frame_fields(raw_time, lat_ids, lon_ids)
        initial_valid = _iso(epoch)
        with server.job_lock:
            job['grid'] = initial_grid
            job['frames'].append({
                'index': 0,
                'valid_time': initial_valid,
                'lead_hours': 0,
                'kind': 'initial_reanalysis',
                'fields': initial_fields
            })
            job['phase'] = 'encoding'
        inputs, forcings = model.data_from_xarray(dataset.isel(time=0))
        model._log_metadata['checkpoint'] = (
            'bundled:tl63_stochastic_mini.pkl' if model_id == 'toy_tl63' else
            next(m[3] for m in _MODEL_CATALOG if m[0] == model_id))
        model._log_metadata.update({
            'requested_steps':
                duration,
            'seed':
                seed,
            'initial_data':
                'bundled historical ERA5 demonstration (1959-01-02; not live weather)'
        })
        inference_logging.log_event('inference_start',
                                    model=model._operation_metadata(),
                                    requested_steps=duration,
                                    seed=seed)
        state = model.encode(inputs, forcings, rng_key=jax.random.PRNGKey(seed))
        mx.eval(*jax.tree.leaves(state))
        forcing_data = dataset[list(model.forcing_variables)]
        hourly_times = epoch + np.arange(duration + 1).astype('timedelta64[h]')
        temporal_forcings = forcing_data.reindex(time=hourly_times,
                                                 method='nearest')
        temporal_forcings = temporal_forcings.assign_coords(time=hourly_times)
        forcings_by_time = model.forcings_from_xarray(temporal_forcings)
        with server.job_lock:
            job['phase'] = 'forecasting'
        for lead in range(1, duration + 1):
            state, output = model.unroll(state,
                                         forcings_by_time,
                                         steps=1,
                                         timedelta=np.timedelta64(1, 'h'),
                                         start_with_input=False)
            mx.eval(*jax.tree.leaves(output))
            valid_time = epoch + np.timedelta64(lead, 'h')
            times = np.asarray([valid_time], dtype='datetime64[ns]')
            output_ds = model.data_to_xarray(output, times=times)
            fields = _frame_fields(output_ds, lat_ids, lon_ids)
            frame = {
                'index': lead,
                'valid_time': _iso(valid_time),
                'lead_hours': lead,
                'kind': 'prediction',
                'fields': fields
            }
            with server.job_lock:
                job['frames'].append(frame)
                job['completed_steps'] = lead
                if lead == duration:
                    job['status'], job['phase'] = 'completed', 'completed'
    except Exception as error:
        logging.getLogger(__name__).exception('Forecast job %s failed', job_id)
        with server.job_lock:
            job['status'], job['phase'] = 'failed', 'failed'
            job['error'] = {
                'code': 'inference_failed',
                'message': str(error)[:1000] or type(error).__name__
            }


def _worker(server: _ForecastServer) -> None:
    while True:
        job_id = server.work.get()
        try:
            if job_id is None:
                return
            with server.job_lock:
                job = server.jobs.get(job_id)
                if job is None or job['status'] != 'queued':
                    continue
                job['status'], job['phase'] = 'running', 'loading'
            _execute(server, job_id)
        finally:
            server.work.task_done()


def _error(handler: BaseHTTPRequestHandler, status: int, code: str,
           message: str) -> None:
    handler._send_json(status, {'error': {'code': code, 'message': message}})


class _Handler(BaseHTTPRequestHandler):
    server: _ForecastServer
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.getLogger(__name__).info('%s - %s', self.address_string(),
                                         fmt % args)

    def _send_json(self, status: int, value: Any) -> None:
        data = json.dumps(_json_safe(value),
                          allow_nan=False,
                          separators=(',', ':')).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        if self.close_connection:
            self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(data)

    def _valid_host(self) -> bool:
        host = self.headers.get('Host', '')
        try:
            parsed = urllib.parse.urlsplit('http://' + host)
            return (not parsed.path and not parsed.query and
                    not parsed.fragment and not parsed.username and
                    not parsed.password and
                    parsed.hostname in ('127.0.0.1', 'localhost') and
                    parsed.port == self.server.server_port)
        except ValueError:
            return False

    def _valid_origin(self) -> bool:
        origin = self.headers.get('Origin')
        if origin is None:
            return True
        try:
            host = self.headers.get('Host', '')
            parsed = urllib.parse.urlsplit(origin)
            return (parsed.scheme == 'http' and parsed.netloc == host and
                    not parsed.path and not parsed.query and
                    not parsed.fragment and not parsed.username and
                    not parsed.password and
                    parsed.hostname in ('127.0.0.1', 'localhost') and
                    parsed.port == self.server.server_port)
        except ValueError:
            return False

    def _body_json(self) -> Any:
        length_text = self.headers.get('Content-Length')
        if length_text is None or not length_text.isascii(
        ) or not length_text.isdecimal():
            raise ValueError('Content-Length is required')
        length = int(length_text)
        if length > _MAX_BODY_BYTES:
            raise OverflowError('Request body is too large')
        if self.headers.get('Transfer-Encoding'):
            raise ValueError('Transfer-Encoding is not supported')
        raw = self.rfile.read(length)

        def pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError(f'duplicate JSON field: {key}')
                obj[key] = value
            return obj

        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda token:
            (_ for _ in ()).throw(ValueError(f'invalid number: {token}')))

    def do_GET(self) -> None:
        if not self._valid_host():
            _error(self, 400, 'invalid_host',
                   'Request Host must identify this loopback server.')
            return
        path = urllib.parse.urlsplit(self.path).path
        if path == '/api/config':
            self._send_json(200, _config())
        elif path == '/api/initial':
            try:
                payload = self._get_initial_snapshot()
            except Exception:
                logging.getLogger(__name__).exception(
                    'Unable to load the archived initial conditions')
                _error(self, 500, 'initial_unavailable',
                       'Historical initial conditions are unavailable.')
                return
            self._send_json(200, payload)
        elif path.startswith('/api/jobs/'):
            self._get_job(path)
        else:
            self._static(path)

    def _get_initial_snapshot(self) -> dict[str, Any]:
        cached = self.server._initial_snapshot
        if cached is not None:
            return cached
        with self.server._initial_lock:
            cached = self.server._initial_snapshot
            if cached is None:
                from neuralgcm import demo
                from neuralgcm.legacy import gin_utils, model_builder
                import xarray

                checkpoint = demo.load_checkpoint_tl63_stochastic()
                aux_dataset = xarray.Dataset.from_dict(
                    checkpoint['aux_ds_dict'])
                model_config = (checkpoint['model_config_str'].replace(
                    'GridWithWavenumbers.radius = None',
                    'GridWithWavenumbers.radius = 1.0',
                ) + '\n\n' + '\n'.join([
                    'GridTL63.radius = 1.0',
                    'GridTL127.radius = 1.0',
                    'GridTL255.radius = 1.0',
                ]))
                with gin_utils.specific_config(model_config):
                    coords = model_builder.coordinate_system_from_dataset(
                        aux_dataset)
                dataset = demo.load_data(coords)
                raw_time = dataset.isel(time=0)
                grid, lat_indices, lon_indices = _grid(raw_time)
                frame = {
                    'index': 0,
                    'valid_time': _iso(dataset.time.values[0]),
                    'lead_hours': 0,
                    'kind': 'initial_reanalysis',
                    'fields': _frame_fields(raw_time, lat_indices, lon_indices)
                }
                cached = {
                    'grid': grid,
                    'frame': frame,
                    'provenance': _provenance(),
                }
                self.server._initial_snapshot = cached
        return cached

    def do_POST(self) -> None:
        # Rejected requests may leave unread bodies. Closing avoids interpreting
        # those bytes as a subsequent request on this HTTP/1.1 connection.
        self.close_connection = True
        if not self._valid_host():
            _error(self, 400, 'invalid_host',
                   'Request Host must identify this loopback server.')
            return
        if not self._valid_origin():
            _error(self, 403, 'cross_origin',
                   'Cross-origin requests are not allowed.')
            return
        path = urllib.parse.urlsplit(self.path).path
        if path not in ('/api/jobs', '/api/jobs/clear'):
            _error(self, 404, 'not_found', 'No such API endpoint.')
            return
        if self.headers.get_content_type() != 'application/json':
            _error(self, 415, 'unsupported_media_type',
                   'Content-Type must be application/json.')
            return
        try:
            body = self._body_json()
            if path == '/api/jobs/clear':
                if not isinstance(body, dict) or body:
                    raise ValueError(
                        'Request body must be exactly an empty JSON object.')
            elif not isinstance(body, dict) or set(body) != {
                    'model_id', 'duration_hours', 'seed'
            }:
                raise ValueError(
                    'Request must contain exactly model_id, duration_hours, and seed.'
                )
            if path == '/api/jobs/clear':
                cleared_count = self.server._clear_queued_jobs()
                if cleared_count is None:
                    _error(self, 503, 'server_closing',
                           'The local forecast server is shutting down.')
                    return
                self._send_json(200, {'cleared_count': cleared_count})
                return

            model_id, duration, seed = body['model_id'], body[
                'duration_hours'], body['seed']
            if not isinstance(model_id, str):
                raise ValueError('model_id must be a string.')
            catalog_entry = next((m for m in _catalog() if m['id'] == model_id),
                                 None)
            if catalog_entry is None:
                raise ValueError('Unknown model_id.')
            if not catalog_entry['available']:
                _error(self, 409, 'model_unavailable',
                       catalog_entry['unavailable_reason'])
                return
            if type(duration) is not int or duration not in _DURATION_HOURS:
                raise ValueError(
                    'duration_hours must be one of 1, 3, 6, 12, or 24.')
            if type(seed) is not int or not 0 <= seed <= 4294967295:
                raise ValueError(
                    'seed must be an integer from 0 through 4294967295.')
        except OverflowError as error:
            _error(self, 413, 'request_too_large', str(error))
            return
        except (ValueError, json.JSONDecodeError) as error:
            _error(self, 400, 'invalid_request', str(error))
            return
        with self.server.job_lock:
            if self.server._closing:
                _error(self, 503, 'server_closing',
                       'The local forecast server is shutting down.')
                return
            finished = sorted(
                (j for j in self.server.jobs.values()
                 if j['status'] in ('completed', 'failed', 'cancelled')),
                key=lambda j: j['created'])
            while len(self.server.jobs) >= _MAX_JOBS and finished:
                del self.server.jobs[finished.pop(0)['id']]
            if len(self.server.jobs) >= _MAX_JOBS:
                _error(
                    self, 503, 'capacity_reached',
                    'All job slots are occupied; wait for a forecast to finish before submitting another.'
                )
                return
            job_id = uuid.uuid4().hex
            job = {
                'id': job_id,
                'status': 'queued',
                'phase': 'queued',
                'request': {
                    'model_id': model_id,
                    'duration_hours': duration,
                    'seed': seed
                },
                'completed_steps': 0,
                'frames': [],
                'grid': None,
                'provenance': _provenance(model_id, seed),
                'error': None,
                'created': dt.datetime.now(dt.timezone.utc).timestamp(),
            }
            self.server.jobs[job_id] = job
            self.server.work.put(job_id)
        self._send_json(202, _job_response(job))

    def _get_job(self, path: str) -> None:
        parts = path.split('/')
        if len(parts) not in (4, 6) or parts[2] != 'jobs' or not parts[3]:
            _error(self, 404, 'not_found', 'No such job endpoint.')
            return
        with self.server.job_lock:
            job = self.server.jobs.get(parts[3])
            if job is None:
                _error(self, 404, 'job_not_found',
                       'Job not found or expired from bounded retention.')
                return
            if len(parts) == 4:
                payload = _job_response(job)
            elif parts[4] == 'frames' and parts[5].isascii(
            ) and parts[5].isdecimal():
                index = int(parts[5])
                if index >= len(job['frames']):
                    _error(self, 404, 'frame_unavailable',
                           'This forecast frame is not available yet.')
                    return
                payload = job['frames'][index]
            else:
                _error(self, 404, 'not_found', 'No such job endpoint.')
                return
        self._send_json(200, payload)

    def _static(self, path: str) -> None:
        if path == '/':
            path = '/index.html'
        if path.startswith('/api/') or '\\' in path or '\x00' in path:
            _error(self, 404, 'not_found', 'No such endpoint.')
            return
        try:
            target = (self.server.web_root /
                      urllib.parse.unquote(path.lstrip('/'))).resolve()
            if not target.is_relative_to(
                    self.server.web_root) or not target.is_file():
                raise FileNotFoundError
            data = target.read_bytes()
        except (OSError, ValueError):
            _error(self, 404, 'not_found', 'Static asset not found.')
            return
        content_type = mimetypes.guess_type(
            target.name)[0] or 'application/octet-stream'
        self.send_response(200)
        self.send_header(
            'Content-Type',
            content_type + ('; charset=utf-8' if content_type.startswith(
                ('text/', 'application/javascript')) else ''))
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(data)


def _configure_logging() -> None:
    logger = logging.getLogger('neuralgcm.inference')
    logger.setLevel(logging.INFO)
    if not any(
            isinstance(handler, logging.StreamHandler)
            for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    logger.propagate = False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args(argv)
    if args.host != '127.0.0.1':
        parser.error(
            '--host must be 127.0.0.1; the application only binds to IPv4 loopback'
        )
    if not 0 <= args.port < 65536:
        parser.error('--port must be between 0 and 65535')
    _configure_logging()
    try:
        server = create_server(args.host, args.port)
    except (OSError, ValueError) as error:
        parser.exit(
            1,
            f'Cannot start local NeuralGCM server on {args.host}:{args.port}: {error}\n'
        )
    bound_host, bound_port = server.server_address[:2]
    address = f'http://{bound_host}:{bound_port}/'
    print(f'NeuralGCM local app ready at {address}', flush=True)
    if args.open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
