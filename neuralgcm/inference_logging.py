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
"""Structured, in-process logging helpers for inference operations.

The library logger is silent unless the application configures it. Records are
JSON strings sent to ``neuralgcm.inference``; this module never configures a
handler or writes persistent telemetry.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import time
from typing import Any, TypedDict

import numpy as np
import jax


class _ArrayInfo(TypedDict):
    shape: tuple[int, ...]
    dtype: str
    bytes: int


class _ArraySummary(TypedDict):
    array_count: int
    arrays: list[_ArrayInfo]
    bytes: int


_LOGGER = logging.getLogger('neuralgcm.inference')
_SCHEMA_VERSION = 1


def log_event(event: str, **fields: Any) -> None:
    """Emit one versioned JSON event to the standard inference logger."""
    if not _LOGGER.isEnabledFor(logging.INFO):
        return
    record = {'event': event, 'schema_version': _SCHEMA_VERSION, **fields}
    _LOGGER.info(json.dumps(record, sort_keys=True, default=str))


def _array_summary(tree: Any) -> _ArraySummary:
    leaves = jax.tree.leaves(tree)
    arrays: list[_ArrayInfo] = []
    total_bytes = 0
    for leaf in leaves:
        shape = getattr(leaf, 'shape', None)
        dtype = getattr(leaf, 'dtype', None)
        if shape is None or dtype is None:
            continue
        shape = tuple(int(size) for size in shape)
        try:
            itemsize = int(dtype.itemsize)
        except (AttributeError, TypeError, ValueError):
            try:
                dtype_name = str(dtype)
                try:
                    itemsize = int(np.dtype(dtype_name).itemsize)
                except TypeError:
                    itemsize = int(
                        np.dtype(dtype_name.rsplit('.', 1)[-1]).itemsize)
            except (TypeError, ValueError):
                itemsize = 0
        nbytes = itemsize
        for size in shape:
            nbytes *= size
        arrays.append({'shape': shape, 'dtype': str(dtype), 'bytes': nbytes})
        total_bytes += nbytes
    return {'array_count': len(arrays), 'arrays': arrays, 'bytes': total_bytes}


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _coord_summary(coords: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    horizontal = getattr(coords, 'horizontal', None)
    if horizontal is not None:
        nodal_shape = getattr(horizontal, 'nodal_shape', None)
        if nodal_shape is not None:
            summary['horizontal_shape'] = tuple(
                int(size) for size in nodal_shape)
        for name in ('latitude_nodes', 'longitude_nodes'):
            value = getattr(horizontal, name, None)
            if value is not None:
                summary[name] = int(value)
        for name in ('latitude_spacing', 'longitude_wavenumbers'):
            value = getattr(horizontal, name, None)
            if value is not None:
                summary[name] = int(value) if isinstance(value,
                                                         int) else str(value)
    vertical = getattr(coords, 'vertical', None)
    if vertical is not None:
        for name in ('layers', 'levels'):
            value = getattr(vertical, name, None)
            if value is not None:
                summary['vertical_' + name] = int(value) if isinstance(
                    value, int) else str(value)
        centers = getattr(vertical, 'centers', None)
        if centers is not None:
            summary['vertical_levels'] = int(len(centers))
    return summary


def describe_model(model: Any,
                   backend: str,
                   checkpoint: str | None = None) -> dict[str, Any]:
    """Build stable, value-free model metadata for an inference log event."""
    if not _LOGGER.isEnabledFor(logging.INFO):
        return {}
    reference_model = getattr(model, '_reference_model', model)
    params = getattr(reference_model, 'params', None)
    parameter_summary: _ArraySummary
    if params is not None:
        parameter_summary = _array_summary(params)
    else:
        parameter_summary = {'array_count': 0, 'arrays': [], 'bytes': 0}
    config = getattr(reference_model, 'gin_config', None)
    config_text = (config if isinstance(config, str) else json.dumps(
        config, sort_keys=True, default=str))
    metadata: dict[str, Any] = {
        'model_class':
            type(reference_model).__name__,
        'backend':
            backend,
        'checkpoint':
            checkpoint,
        'configuration_sha256':
            hashlib.sha256(config_text.encode()).hexdigest(),
        'configuration_summary':
            config_text[:512],
        'parameter_count':
            sum(
                math.prod(array['shape'])
                for array in parameter_summary['arrays']),
        'parameter_bytes':
            parameter_summary['bytes'],
        'parameter_array_count':
            parameter_summary['array_count'],
        'grid':
            _coord_summary(getattr(reference_model, 'data_coords', None)),
        'model_grid':
            _coord_summary(getattr(reference_model, 'model_coords', None)),
        'timestep':
            str(getattr(reference_model, 'timestep', 'unknown')),
        'device':
            _device_name(backend),
        'versions': {
            'neuralgcm': _version('neuralgcm'),
            'jax': _version('jax'),
            'mlx': _version('mlx') if backend == 'mlx' else None,
            'numpy': _version('numpy'),
        },
        'precision':
            sorted({array['dtype'] for array in parameter_summary['arrays']}),
    }
    return metadata


def _device_name(backend: str) -> str:
    if backend == 'mlx':
        try:
            import mlx.core as mx
            return str(mx.default_device())
        except ImportError:
            return 'unknown'
    try:
        return ','.join(str(device) for device in jax.devices())
    except (RuntimeError, ValueError):
        return 'unknown'


def _synchronize(value: Any, backend: str) -> None:
    # MLXFunction materializes every call itself. Re-evaluating its result here
    # would duplicate potentially expensive graph evaluation.
    if backend != 'mlx':
        jax.block_until_ready(value)


def _mlx_memory() -> dict[str, int]:
    try:
        import mlx.core as mx
    except ImportError:
        return {}
    result = {}
    for field, name in (('active_bytes', 'get_active_memory'),
                        ('peak_bytes', 'get_peak_memory')):
        getter = getattr(mx, name, None)
        if getter is not None:
            try:
                result[field] = int(getter())
            except (RuntimeError, TypeError, ValueError):
                pass
    return result


def logged_call(
    operation: str,
    function: Any,
    *args: Any,
    backend: str,
    model_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Run and time an inference operation, logging success or failure."""
    if not _LOGGER.isEnabledFor(logging.INFO):
        return function(*args, **kwargs)
    inputs = _array_summary((args, kwargs))
    started = time.perf_counter()
    try:
        result = function(*args, **kwargs)
        _synchronize(result, backend)
    except BaseException as error:
        fields = {
            'operation': operation,
            'backend': backend,
            'elapsed_seconds': time.perf_counter() - started,
            'status': 'error',
            'error': f'{type(error).__name__}: {error}',
        }
        if inputs is not None:
            fields['inputs'] = inputs
        if model_metadata is not None:
            fields['model'] = model_metadata
        if backend == 'mlx':
            fields.update(_mlx_memory())
        log_event('operation', **fields)
        raise
    fields = {
        'operation': operation,
        'backend': backend,
        'elapsed_seconds': time.perf_counter() - started,
        'status': 'ok',
    }
    if inputs is not None:
        fields['inputs'] = inputs
        fields['outputs'] = _array_summary(result)
    profile = getattr(function, 'last_profile', None)
    if isinstance(profile, dict):
        fields['profile'] = profile
    if model_metadata is not None:
        fields['model'] = model_metadata
    if backend == 'mlx':
        fields.update(_mlx_memory())
    log_event('operation', **fields)
    return result
