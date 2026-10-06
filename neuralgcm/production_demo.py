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
"""Run a local NeuralGCM checkpoint on historical or supplied initial data.

The bundled input is a historical ERA5 demonstration, not a live weather feed.
"""

import argparse
import json
import logging
import pickle
import time
from typing import Any, Protocol, cast

import jax

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import numpy as np
import xarray

import neuralgcm
from neuralgcm import demo
from neuralgcm import inference_logging


class _InferenceModel(Protocol):
    """Subset of the pressure-level model API used by this runner."""

    input_variables: list[str]
    forcing_variables: list[str]
    data_coords: Any
    timestep: np.timedelta64

    def data_from_xarray(self, dataset: xarray.Dataset) -> tuple[Any, Any]:
        ...

    def forcings_from_xarray(self, dataset: xarray.Dataset) -> Any:
        ...

    def encode(self,
               inputs: Any,
               forcings: Any,
               rng_key: Any | None = None) -> Any:
        ...

    def unroll(
        self,
        state: Any,
        forcings: Any,
        *,
        steps: int,
        timedelta: Any | None = None,
        start_with_input: bool = False,
        post_process_fn: Any | None = None,
    ) -> tuple[Any, Any]:
        ...

    def data_to_xarray(self, data: Any, times: Any) -> xarray.Dataset:
        ...

    def _operation_metadata(self) -> dict[str, Any]:
        ...


def _netcdf_attribute(value: Any) -> Any | None:
    """Return a NetCDF-safe attribute value, or None to omit it."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, dict):
        value = json.dumps(value,
                           default=lambda item: np.asarray(item).tolist())
    elif isinstance(value, np.ndarray):
        if value.dtype.kind == 'b':
            value = value.astype(np.int8)
        elif value.dtype.kind not in 'iufSU':
            value = json.dumps(value.tolist(), default=str)
    elif isinstance(value, (list, tuple)):
        if any(item is None or isinstance(item, (dict, list, tuple))
               for item in value):
            value = json.dumps(value,
                               default=lambda item: np.asarray(item).tolist())
        else:
            value = [_netcdf_attribute(item) for item in value]
    elif not isinstance(value, (str, bytes, int, float, np.ndarray)):
        value = str(value)
    return value


def write_forecast_netcdf(dataset: xarray.Dataset, path: str) -> None:
    """Write a forecast while removing or encoding non-NetCDF xarray metadata."""
    clean = dataset.copy(deep=False)
    clean.attrs = {
        name: converted
        for name, value in dataset.attrs.items()
        if (converted := _netcdf_attribute(value)) is not None
    }
    for variable in clean.variables.values():
        variable.attrs = {
            name: converted
            for name, value in variable.attrs.items()
            if (converted := _netcdf_attribute(value)) is not None
        }
    clean.to_netcdf(path)


def _load_initial_data(path: str | None,
                       model: _InferenceModel) -> tuple[xarray.Dataset, str]:
    if path is None:
        return demo.load_data(
            model.data_coords), ('bundled historical ERA5 demonstration '
                                 '(1959-01-02; not live weather)')

    with xarray.open_dataset(path) as source:
        dataset = source.load()
    if 'time' not in dataset.dims:
        dataset = dataset.expand_dims(time=[np.datetime64('2000-01-01')])
    wanted = list(dict.fromkeys(model.input_variables +
                                model.forcing_variables))
    missing = sorted(set(wanted) - set(dataset.data_vars))
    if missing:
        raise ValueError(
            f'initial-condition file is missing model variables: {missing}')
    dataset = dataset[wanted]

    source_grid = spherical_harmonic.Grid(
        latitude_nodes=dataset.sizes['latitude'],
        longitude_nodes=dataset.sizes['longitude'],
        latitude_spacing=xarray_utils.infer_latitude_spacing(dataset.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(dataset.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        source_grid, model.data_coords.horizontal, skipna=True)
    dataset = xarray_utils.regrid(dataset, regridder)
    dataset = xarray_utils.fill_nan_with_nearest(dataset)
    return dataset, f'initial-condition NetCDF: {path}'


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='path to a downloaded NeuralGCM checkpoint pickle',
    )
    parser.add_argument(
        '--backend',
        choices=('jax', 'mlx'),
        default='jax',
        help=
        'inference backend (MLX requires the optional Apple Silicon install)',
    )
    parser.add_argument(
        '--initial-condition',
        help=
        'input NetCDF; defaults to the bundled historical ERA5 demonstration',
    )
    parser.add_argument('--steps',
                        type=int,
                        default=1,
                        help='number of internal model timesteps (default: 1)')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='random seed for stochastic models (default: 42)')
    parser.add_argument('--output',
                        default='forecast.nc',
                        help='output NetCDF path (default: forecast.nc)')
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error('--steps must be at least 1')
    return args


def main(argv: list[str] | None = None) -> None:
    logger = logging.getLogger('neuralgcm.inference')
    logger.setLevel(logging.INFO)
    if not any(
            isinstance(handler, logging.StreamHandler)
            for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    logger.propagate = False
    args = _parse_args(argv)
    started = time.perf_counter()
    with open(args.checkpoint, 'rb') as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    model: _InferenceModel
    if args.backend == 'mlx':
        try:
            from neuralgcm.mlx import PressureLevelModel
        except ImportError as error:
            raise ImportError(
                "MLX backend is optional; install it with `pip install -e '.[mlx]'` "
                'on Apple Silicon.') from error
        model = cast(_InferenceModel,
                     PressureLevelModel.from_checkpoint(checkpoint))
    else:
        model = cast(_InferenceModel,
                     neuralgcm.PressureLevelModel.from_checkpoint(checkpoint))

    if args.backend == 'mlx':
        model_metadata = model._operation_metadata()
        model_metadata['checkpoint'] = args.checkpoint
    else:
        model_metadata = inference_logging.describe_model(
            model, args.backend, args.checkpoint)
    model_metadata.update({
        'requested_steps': args.steps,
        'seed': args.seed,
    })
    dataset, provenance = _load_initial_data(args.initial_condition, model)
    model_metadata['initial_data'] = provenance
    inference_logging.log_event(
        'inference_start',
        model=model_metadata,
        initial_data=provenance,
        requested_steps=args.steps,
        seed=args.seed,
    )
    inputs, forcings = model.data_from_xarray(dataset.isel(time=0))
    rng_key = jax.random.PRNGKey(args.seed)
    if args.backend == 'jax':
        state = inference_logging.logged_call(
            'encode',
            model.encode,
            inputs,
            forcings,
            rng_key=rng_key,
            backend='jax',
            model_metadata=model_metadata,
        )
    else:
        state = model.encode(inputs, forcings, rng_key=rng_key)
    temporal_forcings = model.forcings_from_xarray(
        dataset.isel(time=slice(0, 1)))
    if args.backend == 'jax':
        state, forecast = inference_logging.logged_call(
            'unroll',
            model.unroll,
            state,
            temporal_forcings,
            steps=args.steps,
            start_with_input=False,
            backend='jax',
            model_metadata=model_metadata,
        )
    else:
        state, forecast = model.unroll(state,
                                       temporal_forcings,
                                       steps=args.steps,
                                       start_with_input=False)

    times = dataset.time.values[0] + np.arange(1,
                                               args.steps + 1) * model.timestep
    forecast_dataset = model.data_to_xarray(forecast, times=times)
    field_stats = {}
    for name, values in forecast_dataset.data_vars.items():
        array = np.asarray(values.values)
        finite = np.isfinite(array)
        field_stats[name] = {
            'finite': bool(finite.all()),
            'minimum': float(np.min(array[finite])) if finite.any() else None,
            'maximum': float(np.max(array[finite])) if finite.any() else None,
        }
    forecast_dataset.attrs.update({
        'neuralgcm_checkpoint':
            args.checkpoint,
        'neuralgcm_backend':
            args.backend,
        'neuralgcm_initial_data':
            provenance,
        'neuralgcm_seed':
            args.seed,
        'neuralgcm_forecast_steps':
            args.steps,
        'neuralgcm_data_caveat':
            'Historical demonstration/initial condition; not a live weather forecast.',
    })
    write_forecast_netcdf(forecast_dataset, args.output)

    elapsed = time.perf_counter() - started
    print(f'Backend: {args.backend}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Initial data: {provenance}')
    print(f'Forecast: {args.steps} model steps; seed={args.seed}')
    print('Finite field statistics (minimum, maximum):')
    for name, stats in field_stats.items():
        print(
            f'  {name}: finite={stats["finite"]}, min={stats["minimum"]}, max={stats["maximum"]}'
        )
    print(f'Output: {args.output}')
    print(f'Elapsed: {elapsed:.2f}s')


if __name__ == '__main__':
    main()
