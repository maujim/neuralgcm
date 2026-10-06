# Local NeuralGCM explorer — mini PRD

## Goal and boundaries

Let friends and family double-click a macOS app and explore genuine local MLX atmospheric predictions in a browser, without a cloud account, terminal, or npm. The only requested views are an interactive globe, temperature/wind/humidity playback, city weather evolution, forecast duration and stochastic seed controls, and two scenarios side by side. Attachment #1 is unavailable; no additional features are assumed.

This is an educational historical experiment, **not live weather or a public forecast service**. Initial conditions are the bundled ERA5 reanalysis sample from 1959-01-02; reanalysis is not a direct observation. Ocean forcing is held at that historical sample. The bundled mini model is explicitly a toy. Production checkpoints remain research models and do not turn the historical sample into today's weather.

## Architecture and release gate

Implement only after Main reports **MLX verified**. A Python standard-library HTTP server (`neuralgcm/local_app.py`) binds to loopback and serves plain ES modules/CSS from `web/`; no frontend build or external CDN. One background inference worker serializes jobs to avoid competing GPU allocations. The browser polls status and lazily requests completed frames. It never downloads checkpoints or sends data off-device. A double-clickable macOS `.app` under `launch/` starts the prepared local Python environment and opens the browser; missing prerequisites are shown in a GUI dialog, never hidden behind a terminal. Distribution must include a prepared compatible environment and checkpoints; the launcher must clearly document that source files alone are not a self-contained Python runtime.

Use the verified `neuralgcm.mlx.PressureLevelModel` only: `from_checkpoint`, `demo.load_data(model.data_coords)`, `data_from_xarray`, `encode` with the requested JAX seed, and real `unroll`/`decode`, materializing MLX results. JAX constructs/traces the reference graph; numerical inference executes on MLX. No synthetic frames, substitute models, JAX inference fallback, or interpolated future predictions. Display raw initial reanalysis separately from decoded future predictions. Support every downloaded production checkpoint compatible with this verified pressure-level API; incompatible/unverified models remain visible with a specific disabled reason rather than masquerading as supported. Main supplies the verified support list.

Checkpoint IDs map only to these trusted local files under `models/`: `deterministic_0_7_deg` → `v1/deterministic_0_7_deg.pkl`, `deterministic_1_4_deg` → `v1/deterministic_1_4_deg.pkl`, `deterministic_2_8_deg` → `v1/deterministic_2_8_deg.pkl`, `stochastic_1_4_deg` → `v1/stochastic_1_4_deg.pkl`, `stochastic_precip_2_8_deg` → `v1_precip/stochastic_precip_2_8_deg.pkl`, and `stochastic_evap_2_8_deg` → `v1_precip/stochastic_evap_2_8_deg.pkl`. `toy_tl63` uses `demo.load_checkpoint_tl63_stochastic()`. Never accept browser-supplied filesystem paths or arbitrary pickles. Downloads are a separate, pre-existing setup operation.

## Weather semantics

All displayed fields are **850 hPa lower-atmosphere values**, approximately 1.5 km altitude, not surface/2 m weather; this pressure surface can lie below mountainous terrain. Label this persistently, including city charts. `temperature` is Celsius (`K − 273.15`); `wind_speed` is horizontal speed in m/s (`sqrt(u² + v²)`); `specific_humidity` is g/kg (`kg/kg × 1000`), **not relative humidity %**. Display nearest model-grid values for cities, not station measurements or downscaled local forecasts. Frame grids may be explicitly subsampled for display; city selection uses the returned grid and advertises that resolution. No fabricated geography, weather basemap downloads, or implied surface accuracy.

## Exact HTTP/JSON contract (version 1)

All responses use JSON except static assets. API errors have `{ "error": { "code": "invalid_request", "message": "Human-readable explanation" } }` with an appropriate non-2xx status. Reject cross-origin mutation and invalid Host headers; bind only to `127.0.0.1`. JSON numbers must be finite; a non-finite forecast fails the job, never becomes invented zero values. IDs are server-generated and job/model lookup is allowlisted. Limit in-memory finished jobs and return a clear capacity error rather than unbounded retention.

### `GET /api/config`

Returns `{ "schema_version": 1, "models": Model[], "defaults": { "model_id": "toy_tl63", "duration_hours": 1, "seed": 0 }, "duration_hours": [1,3,6,12,24], "output_interval_hours": 1, "level_hpa": 850, "fields": Field[], "limits": { "max_scenarios": 2 }, "provenance": Provenance }`.

`Model` is `{ "id": string, "label": string, "kind": "toy"|"production", "stochastic": boolean, "available": boolean, "unavailable_reason": string|null, "checkpoint": string }`. `checkpoint` is a catalog-relative path or `bundled:tl63_stochastic_mini.pkl`, not an absolute user path. All seven entries appear, even when missing/incompatible.

`Field` is `{ "id": "temperature"|"wind_speed"|"specific_humidity", "label": string, "unit": "°C"|"m/s"|"g/kg" }`.

`Provenance` is `{ "engine": "MLX", "initial_condition": "ERA5 reanalysis", "initial_time": ISO8601, "forcing": "Historical sea-surface temperature and sea-ice held constant", "level_hpa": 850, "warnings": string[] }`. Job provenance adds `model_id`, `model_kind`, `checkpoint`, `seed`, and `display_grid` (a human-readable resolution/subsampling description).

### `POST /api/jobs`

Accept exactly `{ "model_id": string, "duration_hours": 1|3|6|12|24, "seed": integer }`, seed in `[0, 4294967295]`; reject unknown fields/types (including boolean integers). Reply `202` with a `Job`. Deterministic models retain the submitted seed in provenance but label it as having no stochastic effect. Scenarios are independent jobs; the UI submits at most two and labels them A/B. A queued job is not falsely described as running.

### `GET /api/jobs/{job_id}`

Return `Job`: `{ "id": string, "status": "queued"|"running"|"completed"|"failed", "phase": "queued"|"loading"|"encoding"|"forecasting"|"completed"|"failed", "progress": { "completed_steps": integer, "total_steps": integer, "fraction": number }, "request": { "model_id": string, "duration_hours": integer, "seed": integer }, "available_frames": integer, "frame_times": ISO8601[], "grid": Grid|null, "provenance": Provenance, "error": { "code": string, "message": string }|null }`.

Steps are completed hourly output intervals, not fabricated wall-clock estimates; fraction is 0 while loading/encoding and reaches 1 only on completion. `frame_times` enumerates only available frames, including the initial reanalysis at index 0; `available_frames` is its length. `Grid` is `{ "latitudes": number[], "longitudes": number[], "level_hpa": 850, "layout": "latitude-major", "display_stride": { "latitude": integer, "longitude": integer } }`. Longitudes use `[-180,180)`, both coordinate arrays are ascending, and field arrays are flattened in `[latitude][longitude]` order. Return the exact sampled coordinates, not an equispaced substitute. Limit visualization to approximately 96 longitudes × 48 latitudes by integer striding; no forecast-value interpolation.

### `GET /api/jobs/{job_id}/frames/{index}`

Return `{ "index": integer, "valid_time": ISO8601, "lead_hours": integer, "kind": "initial_reanalysis"|"prediction", "fields": { "temperature": number[], "wind_speed": number[], "specific_humidity": number[] } }`. Initial frame has lead 0; future frames have leads 1…duration. An unavailable frame is a clear non-2xx response. There is no endpoint that pretends an unfinished forecast exists.

## Browser interactions and ownership

`web/index.html`, `web/app.js`, and `web/styles.css` own the application shell, configuration, A/B controls, city search over a bundled named-city list, status/error/provenance, polling, cached frame requests, shared timeline/playback, and city time-series charts. No account, external map service, or external library is required. Disable seed editing for deterministic models and explain why. Keep controls usable during slow loading, show honest model setup/forecast phases, and surface actionable server failures.

`web/globe.js` and `web/globe.css` own a drag-rotatable, zoomable canvas globe and color legend, with a visible drag/zoom hint and accessible controls. Export `createGlobe(container)` returning `{ setFrame({grid, frame, field, range}), resize(), destroy() }`. `range` is `{min:number,max:number}` supplied identically for both scenarios; field is a contract field ID. The globe module imports its stylesheet through a link in `index.html`, not a JS CSS import. `setFrame` accepts `frame: null` to render an honest empty state. Plot actual sampled grid cells on the visible hemisphere; show field/unit labels and common color scales. No predictions are generated in JavaScript.

The shell uses one shared frame index/time and field selector for A/B, and city charts show actual returned lead-time values. If a scenario has not produced the selected frame, show waiting/unavailable rather than substituting another time. Distinguish the initial reanalysis and predictions by text and chart styling, not color alone. City labels include coordinates and nearest-grid caveat. Play/pause and timeline scrubbing operate only on available data. Side-by-side is optional until B is enabled; disabling B must not silently erase A.

## Acceptance / implementation advice

1. On a prepared Apple Silicon Mac, double-click opens a local page without Terminal; all browser assets work offline.
2. All six cataloged production models plus toy are listed with honest availability. Every enabled model calls the verified real MLX pathway; toy and historical caveats remain visible.
3. A requested run produces real forecast frames, progress, provenance, and actionable failures; no fake output appears before completion of a step.
4. Temperature, wind speed, and specific humidity render on a rotatable/zoomable globe with playback; a selected city has matching hourly evolution charts.
5. Duration and seed controls generate independently tracked A/B jobs with synchronized times and common color scales; deterministic seeds are not falsely advertised as variable scenarios.
6. Main verifies API validation/security, real forecast jobs and finite arrays, frontend/browser interactions, and double-click launcher behavior. Edit workers run no tests, builds, linters, formatters, or runtime checks.

Implementation slices are disjoint: server/job API (`neuralgcm/local_app.py`); globe (`web/globe.js`, `web/globe.css`); shell/city/scenarios (`web/index.html`, `web/app.js`, `web/styles.css`); launcher and user setup docs (`launch/`, existing installation docs). The advisor owns this contract and integration only. Changes to MLX/backend numerical modules are out of scope.
