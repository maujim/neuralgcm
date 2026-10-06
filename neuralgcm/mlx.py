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
"""NeuralGCM inference on Apple Silicon using MLX arrays.

Checkpoint construction and graph tracing use the reference JAX model. All
numerical encode, advance, decode, and forecast operations run through the MLX
JAXPR interpreter.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from typing import Any

import jax
import mlx.core as mx
from neuralgcm import _mlx_jaxpr
from neuralgcm import demo
from neuralgcm import inference_logging
from neuralgcm.legacy import api as legacy_api
import numpy as np


class PressureLevelModel:
    """MLX-backed inference wrapper around a reference PressureLevelModel."""

    def __init__(self, reference_model: legacy_api.PressureLevelModel):
        self._reference_model = reference_model
        # Wrap the model's existing public operations so checkpoint-specific
        # parameters and constants are captured during JAXPR tracing.
        self._encode = _mlx_jaxpr.MLXFunction(reference_model.encode)
        self._advance = _mlx_jaxpr.MLXFunction(reference_model.advance)
        self._decode = _mlx_jaxpr.MLXFunction(reference_model.decode)
        self._log_metadata = inference_logging.describe_model(
            reference_model, 'mlx')

    def _operation_metadata(self) -> dict[str, Any]:
        if not self._log_metadata:
            self._log_metadata = inference_logging.describe_model(
                self._reference_model, 'mlx')
        return self._log_metadata

    def __getattr__(self, name: str) -> Any:
        # Preserve the reference API for metadata, checkpoint parameters, unit/time
        # conversion, and xarray helpers without duplicating their implementations.
        return getattr(self._reference_model, name)

    def data_to_xarray(self,
                       data: dict[str, Any],
                       times: np.ndarray | None = None) -> Any:
        """Converts model data to xarray, transferring MLX arrays to the host."""
        host_data = jax.tree.map(
            lambda value: np.asarray(value)
            if isinstance(value, mx.array) else value,
            data,
        )
        return self._reference_model.data_to_xarray(host_data, times=times)

    def encode(
        self,
        inputs: dict[str, Any],
        forcings: dict[str, Any],
        rng_key: Any | None = None,
    ) -> Any:
        """Encode input conditions into a model state of MLX arrays."""
        return inference_logging.logged_call(
            'encode',
            self._encode,
            inputs,
            forcings,
            rng_key,
            backend='mlx',
            model_metadata=self._operation_metadata(),
        )

    def advance(self, state: Any, forcings: dict[str, Any]) -> Any:
        """Advance model state one internal timestep using MLX."""
        return inference_logging.logged_call(
            'advance',
            self._advance,
            state,
            forcings,
            backend='mlx',
            model_metadata=self._operation_metadata(),
        )

    def decode(self, state: Any, forcings: dict[str, Any]) -> dict[str, Any]:
        """Decode model state to pressure-level predictions using MLX."""
        return inference_logging.logged_call(
            'decode',
            self._decode,
            state,
            forcings,
            backend='mlx',
            model_metadata=self._operation_metadata(),
        )

    @staticmethod
    def _sim_time(state: Any) -> Any:
        return legacy_api._sim_time_from_state(state)

    @staticmethod
    def _select_nearest_forcings(forcings: dict[str, Any],
                                 sim_time: Any) -> dict[str, Any]:
        times = forcings['sim_time']
        if times.size == 1:
            approx_index = mx.array(0, dtype=mx.float32)
        else:
            upper = mx.searchsorted(times, sim_time,
                                    side='left').astype(mx.int32)
            lower = mx.clip(upper - 1, 0, times.size - 2)
            lower_time = times[lower]
            upper_time = times[lower + 1]
            fraction = mx.clip(
                (sim_time - lower_time) / (upper_time - lower_time), 0, 1)
            approx_index = lower + fraction
        index = mx.round(approx_index).astype(mx.int32)
        return jax.tree.map(lambda value: value[index, ...], forcings)

    def unroll(
        self,
        state: Any,
        forcings: dict[str, Any],
        *,
        steps: int,
        timedelta: legacy_api.TimedeltaLike | None = None,
        start_with_input: bool = False,
        post_process_fn: Callable[[Any, dict[str, Any]], Any] | None = None,
    ) -> tuple[Any, Any]:
        """Unroll inference over temporal forcings and log synchronized timings."""
        return inference_logging.logged_call(
            'unroll',
            self._unroll_impl,
            state,
            forcings,
            steps=steps,
            timedelta=timedelta,
            start_with_input=start_with_input,
            post_process_fn=post_process_fn,
            backend='mlx',
            model_metadata=self._operation_metadata(),
        )

    def _unroll_impl(
        self,
        state: Any,
        forcings: dict[str, Any],
        *,
        steps: int,
        timedelta: legacy_api.TimedeltaLike | None = None,
        start_with_input: bool = False,
        post_process_fn: Callable[[Any, dict[str, Any]], Any] | None = None,
    ) -> tuple[Any, Any]:
        """Unroll MLX inference over a temporal forcing sequence.

    The forcing sample nearest to each state's simulation time is used for both
    advancement and output decoding. ``timedelta`` is an integer multiple of
    the reference model's internal timestep.
    """
        if steps < 0:
            raise ValueError(f'{steps=} must be non-negative')
        # Convert temporal inputs once, rather than transferring NumPy forcings at
        # every forecast step. MLX inputs remain on device through mx.array.
        forcing_arrays = jax.tree.map(mx.array, forcings)
        if timedelta is None:
            inner_steps = 1
        else:
            inner_steps = legacy_api._calculate_sub_steps(
                self.timestep, timedelta)

        def get_forcings(current_state):
            return self._select_nearest_forcings(forcing_arrays,
                                                 self._sim_time(current_state))

        def process(current_state):
            current_forcings = get_forcings(current_state)
            if post_process_fn is None:
                return self.decode(current_state, current_forcings)
            return post_process_fn(current_state, current_forcings)

        outputs = []
        if steps == 0:
            # Discover the output structure and shapes without advancing the state.
            empty_template = process(state)
            empty_outputs = jax.tree.map(
                lambda value: mx.zeros((0,) + value.shape, dtype=value.dtype),
                empty_template,
            )
            mx.eval(*jax.tree.leaves(empty_outputs))
            return state, empty_outputs

        for _ in range(steps):
            if start_with_input:
                outputs.append(process(state))
            for _ in range(inner_steps):
                state = self.advance(state, get_forcings(state))
            if not start_with_input:
                outputs.append(process(state))

        forecast = jax.tree.map(lambda *values: mx.stack(list(values)),
                                *outputs)
        # The primitive calls synchronize individually; materialize only the new
        # stacked trajectory arrays before returning the operation result.
        mx.eval(*jax.tree.leaves(forecast))
        return state, forecast

    @classmethod
    def from_checkpoint(cls, checkpoint: Any) -> PressureLevelModel:
        """Creates an MLX model from a standard NeuralGCM checkpoint."""
        return cls(legacy_api.PressureLevelModel.from_checkpoint(checkpoint))


def _demo_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',
                        type=int,
                        default=3,
                        help='number of internal model timesteps to forecast')
    parser.add_argument('--output', help='optional output NetCDF path')
    logger = logging.getLogger('neuralgcm.inference')
    logger.setLevel(logging.INFO)
    if not any(
            isinstance(handler, logging.StreamHandler)
            for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    logger.propagate = False
    args = parser.parse_args(argv)
    if args.steps < 0:
        parser.error('--steps must be non-negative')

    model = PressureLevelModel.from_checkpoint(
        demo.load_checkpoint_tl63_stochastic())
    dataset = demo.load_data(model.data_coords)
    inputs, forcings = model.data_from_xarray(dataset.isel(time=0))
    temporal_forcings = model.forcings_from_xarray(dataset)
    model._log_metadata = model._operation_metadata()
    model._log_metadata.update({
        'requested_steps':
            args.steps,
        'seed':
            0,
        'initial_data':
            'bundled historical ERA5 demonstration (1959-01-02; not live weather)',
    })
    inference_logging.log_event(
        'inference_start',
        model=model._log_metadata,
        initial_data=model._log_metadata['initial_data'],
        requested_steps=args.steps,
        seed=0,
    )
    state = model.encode(inputs, forcings, rng_key=jax.random.PRNGKey(0))
    state, forecast = model.unroll(state,
                                   temporal_forcings,
                                   steps=args.steps,
                                   start_with_input=False)

    # Inspect the synchronized forecast before writing output.
    leaves = jax.tree.leaves(forecast)
    if not leaves:
        raise RuntimeError('forecast contains no output arrays')
    if args.steps:
        finite = all(
            bool(mx.all(mx.isfinite(value)).item()) for value in leaves)
        if not finite:
            raise FloatingPointError('MLX forecast contains non-finite values')
        mx.eval(*[mx.min(value) for value in leaves],
                *[mx.max(value) for value in leaves])
        minimum = min(float(np.real(mx.min(value).item())) for value in leaves)
        maximum = max(float(np.real(mx.max(value).item())) for value in leaves)
        statistics = f'min={minimum}, max={maximum}'
    else:
        finite = True
        statistics = 'min=n/a, max=n/a (empty forecast)'
    print(f'MLX device: {mx.default_device()}')
    print(f'Forecast: steps={args.steps}, finite={finite}, {statistics}')

    times = dataset.time.values[0] + np.arange(1,
                                               args.steps + 1) * model.timestep
    forecast_dataset = model.data_to_xarray(forecast, times=times)
    if args.output:
        from neuralgcm.production_demo import write_forecast_netcdf

        write_forecast_netcdf(forecast_dataset, args.output)
        print(f'Wrote {args.output}')


if __name__ == '__main__':
    _demo_main()
