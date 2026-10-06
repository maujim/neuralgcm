# Local NeuralGCM explorer — mini PRD

## Goal and boundaries

Let friends and family double-click a prepared macOS app and explore local atmospheric predictions in a browser, without a cloud account, terminal steps after setup, or npm. The default is a guided lesson: inspect London's historical starting weather, choose **Predict next hour**, then toggle between the starting frame and the future prediction. Once the first run succeeds, invite exploration of another city, wind, or a three-hour run. Keep A/B comparisons, model and seed selection, and queued-job clearing under collapsed **Advanced** controls. The requested views are an interactive globe and city weather evolution; Attachment #1 is unavailable, so no additional features are assumed.

This is an educational historical experiment, **not live weather or a public forecast service**. Initial conditions are the bundled ERA5 reanalysis sample from 1959-01-02; reanalysis is an estimate, not a direct measurement. Future values are model predictions, not subsequently observed weather. Ocean forcing is held at that historical sample. The bundled mini model is explicitly a toy. Production checkpoints remain research models and do not turn the historical sample into today's weather. MLX is the Apple Silicon execution engine for the selected model, not another model; no model training occurs during a run.

## Architecture

The Python standard-library HTTP server (`neuralgcm/local_app.py`) binds to loopback and serves plain ES modules/CSS from `web/`; no frontend build or external CDN. One background inference worker serializes jobs to avoid competing GPU allocations. The browser polls status and lazily requests completed frames. It never downloads checkpoints or sends data off-device. `launch/NeuralGCM Explorer.app` starts the prepared local Python environment and opens the browser; missing prerequisites are shown in a GUI dialog, not a terminal. This launcher relies on the prepared project environment and bundled demo data, rather than being a self-contained Python distribution. A normal app launch starts a local server process; after project verification, no test server/browser is left running in the background.

Use the verified `neuralgcm.mlx.PressureLevelModel` only for predictions: `from_checkpoint`, `demo.load_data(model.data_coords)`, `data_from_xarray`, `encode` with the requested JAX seed, and real `unroll`/`decode`, materializing MLX results. JAX constructs/traces the reference graph; numerical inference executes on MLX. MLX is an execution backend for the selected model, not a different or separately trained model; the app does not train models during use. No synthetic frames, substitute models, JAX inference fallback, or interpolated future predictions. Display the raw initial reanalysis separately from decoded future predictions. Support every downloaded production checkpoint compatible with this verified pressure-level API; incompatible/unverified models remain visible with a specific disabled reason rather than masquerading as supported. Main supplies the verified support list.

Keep at most the last constructed model cached; release it before loading a different checkpoint. On the 24 GiB host, the 0.7° model alone can retain approximately 9–16 GiB of traced constants. Enable the existing `neuralgcm.inference` INFO structured console logger for model configuration, timings, cache profile and synchronized memory; do not create persistent logs by default.

Build persistence forcings from **only** `dataset[model.forcing_variables]`, reindexed hourly from initialization through the requested horizon using the initial historical values, then call `model.forcings_from_xarray`. Future forcing timestamps describe the declared persistence assumption, not fresh observations. Do not duplicate all pressure-level weather arrays or relax the model's forcing-time tolerance.

Checkpoint IDs map only to these trusted local files under `models/`: `deterministic_0_7_deg` → `v1/deterministic_0_7_deg.pkl`, `deterministic_1_4_deg` → `v1/deterministic_1_4_deg.pkl`, `deterministic_2_8_deg` → `v1/deterministic_2_8_deg.pkl`, `stochastic_1_4_deg` → `v1/stochastic_1_4_deg.pkl`, `stochastic_precip_2_8_deg` → `v1_precip/stochastic_precip_2_8_deg.pkl`, and `stochastic_evap_2_8_deg` → `v1_precip/stochastic_evap_2_8_deg.pkl`. `toy_tl63` uses `demo.load_checkpoint_tl63_stochastic()`. Never accept browser-supplied filesystem paths or arbitrary pickles. Downloads are a separate, pre-existing setup operation.

## Weather semantics

All displayed fields are **850 hPa lower-atmosphere values**, approximately 1.5 km above sea level, not surface/2 m or street-level weather; this pressure surface can lie below mountainous terrain. Label this persistently, including city charts. `temperature` is Celsius (`K − 273.15`); `wind_speed` is horizontal speed in m/s (`sqrt(u² + v²)`); `specific_humidity` is g/kg (`kg/kg × 1000`), **not relative humidity %**. Display nearest model-grid values for cities, not station measurements or downscaled local forecasts. Frame grids may be explicitly subsampled for display; city selection uses the returned grid and advertises that resolution. No fabricated geography, weather basemap downloads, or implied surface accuracy.

## Exact HTTP/JSON contract (version 1)

All responses use JSON except static assets. API errors have `{ "error": { "code": "invalid_request", "message": "Human-readable explanation" } }` with an appropriate non-2xx status. Reject cross-origin mutation and invalid Host headers; bind only to `127.0.0.1`. JSON numbers must be finite; a non-finite forecast fails the job, never becomes invented zero values. IDs are server-generated and job/model lookup is allowlisted. Bound in-memory job retention and return a clear capacity error rather than retaining records without limit; cancelled jobs are reclaimable like completed and failed jobs.

### `GET /api/config`

Returns `{ "schema_version": 1, "models": Model[], "defaults": { "model_id": "toy_tl63", "duration_hours": 1, "seed": 0 }, "duration_hours": [1,3,6,12,24], "output_interval_hours": 1, "level_hpa": 850, "fields": Field[], "limits": { "max_scenarios": 2 }, "provenance": Provenance }`.

`Model` is `{ "id": string, "label": string, "kind": "toy"|"production", "stochastic": boolean, "available": boolean, "unavailable_reason": string|null, "checkpoint": string }`. `checkpoint` is a catalog-relative path or `bundled:tl63_stochastic_mini.pkl`, not an absolute user path. All seven entries appear, even when missing/incompatible.

`Field` is `{ "id": "temperature"|"wind_speed"|"specific_humidity", "label": string, "unit": "°C"|"m/s"|"g/kg" }`.

`Provenance` is `{ "engine": "MLX", "initial_condition": "ERA5 reanalysis", "initial_time": ISO8601, "forcing": "Historical sea-surface temperature and sea-ice held constant", "level_hpa": 850, "warnings": string[] }`. Job provenance adds `model_id`, `model_kind`, `checkpoint`, `seed`, and `display_grid` (a human-readable resolution/subsampling description).

### `GET /api/initial`

Returns `{ "grid": Grid, "frame": Frame, "provenance": Provenance }` for the historical starting conditions, available without submitting a forecast job. `frame` is the same shape as a frame response below: `{ "index": 0, "valid_time": "1959-01-02T00:00:00Z", "lead_hours": 0, "kind": "initial_reanalysis", "fields": { "temperature": number[], "wind_speed": number[], "specific_humidity": number[] } }`. Values are 850 hPa temperature in °C, wind speed in m/s, and specific humidity in g/kg, flattened in the grid's latitude-major order.

The producer loads the bundled toy checkpoint's coordinate metadata to establish its TL63 grid, then loads the packaged ERA5 reanalysis regridded to those coordinates with `demo.load_data`. This is the starting reanalysis data, not a forecast, not a direct measurement, and not an inference result from that toy checkpoint. No model is loaded and no forecast job is run to serve this endpoint. The snapshot is cached per server process. The response's `provenance` is the experiment's baseline provenance metadata; its MLX engine label describes the engine used for future predictions, not computation performed by this initial-data endpoint.

### `POST /api/jobs`

Accept exactly `{ "model_id": string, "duration_hours": 1|3|6|12|24, "seed": integer }`, seed in `[0, 4294967295]`; reject unknown fields/types (including boolean integers). Reply `202` with a `Job`. Deterministic models retain the submitted seed in provenance but label it as having no stochastic effect. Scenarios are independent jobs; the UI submits at most two and labels them A/B. A queued job is not falsely described as running.

### `POST /api/jobs/clear`

Accept exactly `{}` and return `200` with `{ "cleared_count": integer }`. Cancel only jobs that are still queued; atomically set their `status` and `phase` to `"cancelled"` and remove their queued work items so job capacity is immediately reusable. A job already transitioned to `running` continues without interruption, and completed jobs and their frames remain available. A cancelled job reports 0 completed steps, its original total steps, no available frames, and `error: null`. Apply the same Host/Origin, JSON content-type, body-size, and duplicate-field protections as other mutations.

### `GET /api/jobs/{job_id}`

Return `Job`: `{ "id": string, "status": "queued"|"running"|"completed"|"failed"|"cancelled", "phase": "queued"|"loading"|"encoding"|"forecasting"|"completed"|"failed"|"cancelled", "progress": { "completed_steps": integer, "total_steps": integer, "fraction": number }, "request": { "model_id": string, "duration_hours": integer, "seed": integer }, "available_frames": integer, "frame_times": ISO8601[], "grid": Grid|null, "provenance": Provenance, "error": { "code": string, "message": string }|null }`.

Steps are completed hourly output intervals, not fabricated wall-clock estimates; fraction is 0 while loading/encoding and reaches 1 only on completion. `frame_times` enumerates only available frames, including the initial reanalysis at index 0; `available_frames` is its length. `Grid` is `{ "latitudes": number[], "longitudes": number[], "level_hpa": 850, "layout": "latitude-major", "display_stride": { "latitude": integer, "longitude": integer } }`. Longitudes use `[-180,180)`, both coordinate arrays are ascending, and field arrays are flattened in `[latitude][longitude]` order. Return the exact sampled coordinates, not an equispaced substitute. Limit visualization to approximately 96 longitudes × 48 latitudes by integer striding; no forecast-value interpolation.

### `GET /api/jobs/{job_id}/frames/{index}`

Return `{ "index": integer, "valid_time": ISO8601, "lead_hours": integer, "kind": "initial_reanalysis"|"prediction", "fields": { "temperature": number[], "wind_speed": number[], "specific_humidity": number[] } }`. Initial frame has lead 0; future frames have leads 1…duration. An unavailable frame is a clear non-2xx response. There is no endpoint that pretends an unfinished forecast exists.

## Browser interactions and ownership

`web/index.html`, `web/app.js`, and `web/styles.css` own the application shell, initial weather, status/error/provenance, polling, cached frame requests, shared timeline/playback, city time-series charts, and the optional A/B controls. No account, external map service, or external library is required. The starting screen shows London's nearest-grid starting weather; no forecast starts automatically. **Predict next hour** launches the first job. After it succeeds, expose city exploration, wind, and **Predict 3 hours**. Keep A/B, model, seed, and clear-queued-jobs controls under collapsed **Advanced**. Disable seed editing for deterministic models and explain why. Keep controls usable during slow loading, show honest model setup/forecast phases, and surface actionable server failures.

The guided shell prefers the available `deterministic_2_8_deg` production checkpoint; when unavailable, it uses the available toy model and clearly identifies it as a toy. This browser choice does not change the `GET /api/config` defaults above; they remain the API defaults independently of the guided lesson. After a successful prediction, toggle between **Starting weather** and **1 hour later**; show the actual starting and predicted values plus their numeric temperature delta for the same London grid cell. The starting frame remains historical reanalysis and the later frame is a future model prediction, never later observed weather.

`web/globe.js` and `web/globe.css` own a drag-rotatable, zoomable canvas globe and color legend, with a visible drag/zoom hint and accessible controls. Export `createGlobe(container)` returning `{ setFrame({grid, frame, field, range}), setLocation({name, latitude, longitude}|null, {focus:true} = {}), resize(), destroy() }`. `setLocation` renders a geographically projected marker and label only while the location is on the front hemisphere; `null` clears it. `focus: true` centers the projection on the city without changing zoom. `range` is `{min:number,max:number}` supplied for both scenarios and for the initial/target frames; field is a contract field ID. The globe module imports its stylesheet through a link in `index.html`, not a JS CSS import. `setFrame` accepts `frame: null` to render an honest empty state. Plot actual sampled grid cells on the visible hemisphere; show field/unit labels and common color scales. No predictions are generated in JavaScript.

The shell uses one shared frame index/time and field selector for A/B, and city charts show actual returned lead-time values. If a scenario has not produced the selected frame, show waiting/unavailable rather than substituting another time. Initial and prediction maps use a common color scale so the two frames are directly comparable. A three-hour run is a real forecast; identify its target as **3 hours later** / **Prediction +3h returned**, and report the returned change over those three hours rather than labeling it as one hour. Distinguish the initial reanalysis and predictions by text and chart styling, not color alone. City labels include coordinates and nearest-grid caveat. Play/pause and timeline scrubbing operate only on available data. Side-by-side is optional until B is enabled; disabling B must not silently erase A.

Visual direction: a compact scientific WeatherLab console in deep navy/slate with warm-white text and restrained cyan/amber accents. Place scenario controls in a compact left rail, keep the globe workspace dominant, use a compact shared timeline and legible city charts, and collapse verbose provenance in accessible details. Preserve visible historical/toy warnings. The Earth view uses genuine bundled offline geographic outlines and graticules over contiguous sample cells, with each cell retaining its exact returned weather value; no fabricated continents or interpolated weather. Maintain responsive layouts and visible keyboard focus without changing inference or interaction behavior.

## Acceptance / implementation advice

1. On a prepared Apple Silicon Mac, double-click opens a local page without Terminal; all browser assets work offline.
2. All six cataloged production models plus toy are listed with honest availability. Every enabled model calls the verified real MLX pathway; toy and historical caveats remain visible.
3. A requested run produces real forecast frames, progress, provenance, and actionable failures; no fake output appears before completion of a step.
4. Temperature, wind speed, and specific humidity render on a rotatable/zoomable globe with playback; a selected city has matching hourly evolution charts.
5. Duration and seed controls generate independently tracked A/B jobs with synchronized times and common color scales; deterministic seeds are not falsely advertised as variable scenarios.
6. Main verifies API validation/security, real forecast jobs and finite arrays, frontend/browser interactions, and double-click launcher behavior. Edit workers run no tests, builds, linters, formatters, or runtime checks.

Implementation slices are disjoint: server/job API (`neuralgcm/local_app.py`); globe (`web/globe.js`, `web/globe.css`); shell/city/scenarios (`web/index.html`, `web/app.js`, `web/styles.css`); launcher and user setup docs (`launch/`, existing installation docs). The advisor owns this contract and integration only. Changes to MLX/backend numerical modules are out of scope.
