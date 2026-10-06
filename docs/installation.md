# Installation

For best performance, we recommend running NeuralGCM models on a computer with
an attached GPU or TPU. Otherwise, performance will be very slow.

You can install NeuralGCM from source using pip, which should automatically
install its dependencies, including [JAX](https://github.com/google/jax) and
[Dinosaur](https://github.com/neuralgcm/dinosaur):
```
pip install neuralgcm
```

## Apple Silicon MLX

Install the optional MLX backend on Apple Silicon:
```
pip install -e '.[mlx]'
```
JAX remains required for checkpoint setup and tracing; forecasts execute with
MLX. Use `PressureLevelModel` from `neuralgcm.mlx`:
```
from neuralgcm.mlx import PressureLevelModel
```
Run a forecast from a downloaded checkpoint with the production demo. The
default initial data is a bundled historical ERA5 demonstration (not live
weather); pass an initial-condition NetCDF to use your own data. The demo
regrids the input to the checkpoint's grid, writes decoded model fields to
NetCDF, and reports timing and finite-value statistics:
```
python -m neuralgcm.production_demo \
    --checkpoint /path/to/model.pkl \
    --backend jax \
    --steps 1 \
    --seed 42 \
    --output forecast.nc
```
Use `--backend mlx` for MLX inference on Apple Silicon after installing the
optional dependency above. The same demo accepts all six checkpoints listed
in [Pre-trained model checkpoints](./checkpoints.md), including the
precipitation/evaporation checkpoints, and supports `--initial-condition
/path/to/input.nc`.

The command emits structured JSON inference records to the console through the
`neuralgcm.inference` logger. Records include checkpoint/model configuration
identity, parameter and grid summaries, backend/device and package versions,
requested run settings, historical initial-data provenance, input/output array
shapes and dtypes, and synchronized operation timings. They are in-memory
logging only; no telemetry files or weather-array contents are written. Python
applications can configure the same standard logger to capture these records.

## Local browser explorer

With the project’s prepared `.venv`, start the local browser interface from the
project root:
```
.venv/bin/python -m neuralgcm.local_app
```
Open the printed URL (normally `http://127.0.0.1:8765`) in a browser. On a
prepared Apple Silicon Mac, the double-click launcher described below starts
this local server process for you. Inspect London's nearest-grid temperature
under **Starting weather**, then choose **Predict next hour** to request the
one-hour model prediction. Toggle between **Starting weather** and **1 hour
later**; the interface shows the predicted temperature delta at that same
London grid cell.

After the first successful run, explore another city, wind, or a three-hour
run. A/B scenarios, model and seed selection, and **Clear queued jobs** are
under **Advanced**. Clearing cancels queued jobs only; a forecast already
running continues, and completed results remain available.

The interface displays 850 hPa atmospheric fields on globes and hourly city
charts from the nearest model-grid cell. Checkpoint choices are discovered from
downloaded files in the project’s `models/` directory. Files stay local and are
not served at browser paths.

The bundled starting conditions are historical ERA5 reanalysis from 2 January
1959, an estimate rather than a direct measurement. The future values are
model predictions, not subsequently observed weather; this is not live weather
or an official forecast. 850 hPa is approximately 1.5 km above sea level, not
surface/street-level weather. The bundled mini model is a toy, distinct from
published production checkpoints. MLX is an execution engine for the same
model, not another model, and no training occurs during a run. Scenario seed
controls stochastic models; deterministic models retain the seed for
provenance without a stochastic effect.

## Launch the local explorer on macOS

The double-clickable app is `launch/NeuralGCM Explorer.app`. Keep it inside
the project’s `launch` folder: it locates the prepared project beside itself.
On a prepared Apple Silicon Mac, double-click it to start the local server
process and open the browser; after setup, this is the only entry point needed,
with no Terminal steps. It binds only to `127.0.0.1`; if the default port 8765
is in use, it selects the next free port in the range 8765–8775. It reuses an
existing NeuralGCM explorer server in that range.

This launcher is not a self-contained Python distribution. The project must
already have its compatible prepared `.venv` and bundled demo data, and any
desired downloaded checkpoints in `models/`. It does not install Python or
dependencies, or download checkpoints. Missing prerequisites are reported in
a macOS dialog.
