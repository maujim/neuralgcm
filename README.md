![](https://github.com/neuralgcm/neuralgcm/raw/main/docs/_static/neuralgcm-logo-light.png)

# Neural General Circulation Models for Weather and Climate

NeuralGCM is a Python library for building hybrid ML/physics atmospheric models
for weather and climate simulation.

- **[Paper](https://arxiv.org/abs/2311.07222)**
- **[Documentation](https://neuralgcm.readthedocs.io/)**
- **License**:
    - Code: [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
    - Trained model weights: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).

To stay up to date on NeuralGCM, **[subscribe to our mailing list](https://groups.google.com/g/neuralgcm-announce)**!

## Apple Silicon MLX

Install the optional backend with `pip install -e '.[mlx]'`. JAX remains
required for checkpoint setup and tracing; forecasts execute with MLX. Import
`PressureLevelModel` from `neuralgcm.mlx`:
```python
from neuralgcm.mlx import PressureLevelModel
```
Run the bundled executable to write a one-step forecast to `forecast.nc`:
```sh
python -m neuralgcm.mlx --steps 1 --output forecast.nc
```

## Local browser explorer

On a prepared Apple Silicon Mac, start the offline browser interface at
`http://127.0.0.1:8765` by double-clicking
`launch/NeuralGCM Explorer.app` (see [installation](docs/installation.md)).
Alternatively, from the project root with its prepared `.venv`, run:
```sh
.venv/bin/python -m neuralgcm.local_app
```
The server prints the local URL; open it in a browser. It binds to loopback
only. The explorer compares two scenarios (model, duration, and seed), shows
850 hPa atmospheric fields on globes, and plots hourly city series at the
nearest model-grid cell.

The bundled experiment starts from historical ERA5 data for 2 January 1959;
it is not live weather, surface weather, or an official forecast. The bundled
mini model is a toy and is identified as such; other choices are published
production checkpoints. Place downloaded checkpoints in `models/` for the
explorer to discover them. Checkpoint files remain local and are never opened
through browser paths.
