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

On a prepared Apple Silicon Mac, double-click
`launch/NeuralGCM Explorer.app` (see [installation](docs/installation.md)).
The app opens the offline browser interface; after setup, this prepared
launcher is the only entry point needed, with no Terminal steps. Launching it
starts a local server process bound to loopback. The project's verification
services are not left running in the background after testing. For development,
from the project root with its prepared
`.venv`, you can instead run:
```sh
.venv/bin/python -m neuralgcm.local_app
```

See London's nearest-grid temperature under **Starting weather**. Nothing is
predicted until you choose **Predict next hour**, which compares that
historical starting frame with a real one-hour model prediction.
Toggle between **Starting weather** and **1 hour later**; the interface shows
the predicted temperature delta at that same London grid cell.
After the first successful run, explore another city, wind, or a three-hour
run; A/B scenarios, model, seed, and clearing queued jobs are under **Advanced**.

This is an educational historical experiment, not live weather or an official
forecast. The starting value is a reanalysis estimate, not a direct measurement;
future values are model predictions, not subsequently observed weather. Values
are for the 850 hPa pressure level (approximately 1.5 km above sea level), not
street-level weather. The bundled mini model is a toy; other choices are
published production checkpoints. MLX is the Apple Silicon execution engine
for the same model, not a separate model, and no model training happens during
a run. Place downloaded checkpoints in `models/` for discovery. Checkpoint
files remain local and are never opened through browser paths.
