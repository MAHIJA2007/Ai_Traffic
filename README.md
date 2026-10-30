# 🚦 Flow28 · AI-Based Traffic Prediction System with MLOps

**Project 28** · K.Mahija (2400080129) · Sk.Anjum (2400080003) · P.Hasitha (2400080005)

> **Problem statement.** Traffic demand changes by hour, weekday, season, weather, events, and location, so a static estimate cannot respond to changing patterns. This project builds a complete, reproducible MLOps lifecycle around a real four-junction traffic dataset: validate the data, engineer time-series features, train and evaluate a model, version every experiment in a model registry, promote only a measurably better champion, serve next-hour and day-ahead forecasts through an API and a fifteen-module operator workspace, monitor input drift, and demonstrate reviewed retraining plus offline/edge failover — all without physical roadside hardware.

![Flow28 Traffic Command Deck](docs/screenshots/dashboard.png)

---

## Why this project is different

Most student traffic-prediction projects stop at a notebook that prints an RMSE. **Flow28** is a working operations product wrapped around the model:

| Signature idea | What it actually does |
| --- | --- |
| **Command-deck interface** | A custom, CSS-only control-room aesthetic: animated signal lamps and road, junction dials, a schematic SVG city map with animated flow lanes, a live alert ticker, KPI cards, PSI bars, score bars, a device rack, and a dark neon theme. No JavaScript, no UI framework beyond Streamlit — every pixel is styled in [`traffic_mlops/ui.py`](traffic_mlops/ui.py). |
| **Four generated themes** | Command cyan, Night run, Amber watch, and Deep field each swap the palette **and** the background artwork at runtime — see the [theme system](#theme-system-with-generated-backgrounds). Status colours stay semantic, so a theme never changes what red/amber/green mean. |
| **Fifteen-module workspace** | The dashboard is not one page. Access, monitoring, weather, alerts, map, forecast, control, incident vision, corridor advisory, assistant, edge/5G, analytics, MLOps, and simulation are separate modules with a shared design language — see the [feature map](#the-fifteen-module-workspace). |
| **Advisory engine, not a black box** | [`traffic_mlops/advisory.py`](traffic_mlops/advisory.py) turns the current virtual readings into a signal plan, per-device recommendations, ranked corridor plans, weather guidance, alert categories, and a scene-triage score. Every number is rule-based and unit-tested, so the UI can explain *why* it recommends something. |
| **Day-ahead recursive outlook** | Beyond the next-hour forecast, the champion is rolled forward hour by hour (6–48 h). Each prediction becomes the next step's input, and the chart shows a ±RMSE uncertainty band plus a p95 congestion advisory. |
| **Registry that cannot cheat** | A new model is registered every run, but the `champion` alias moves **only** when holdout RMSE improves on the *same* dataset. Dataset changes reset the comparison baseline. |
| **Offline edge serving** | The promoted champion is mirrored to an edge cache as a versioned `skops` artifact. If the simulated cloud link drops, the dashboard and API keep forecasting from the cache while clearly labeling the fallback. |
| **Software-only digital twin** | Sensors, weather, 5G link quality, edge latency, sensor faults, offline mode, signal phases, and controllable devices are simulated reproducibly from historical junction profiles — labeled *virtual* everywhere. |
| **Grounded AI assistant** | Answers are computed from the dataset, the registered champion, the advisory engine, and the current virtual telemetry — no external LLM calls, no invented numbers. |

---

## The fifteen-module workspace

The reference brief for this build was a smart-farming dashboard. Every one of its fifteen features was rebuilt as a traffic-operations equivalent rather than copied as a farming screen:

| Reference feature | Flow28 module | What it does here |
| --- | --- | --- |
| Login / farmer profile | ◈ **Operator access** | Session profile — operator name, ID, role, assigned corridor, network coverage, shift — plus a role-capability matrix. Clearly labeled session-only: no password, no identity service, no personal record stored. |
| Main dashboard | ◉ **Dashboard** | Hero status strip, live ticker, eight KPIs (corridors, records, fleet flow, peak occupancy, spare capacity, weather index, champion, network health), junction dial cards, observed baseline, the latest simulated alerts with their tick, and next-best actions. |
| Live sensor monitoring | ▣ **Live sensors** | Per-junction virtual flow, speed, occupancy, temperature/humidity, rain, wind, visibility, light, and three trend charts over virtual ticks. |
| AI crop monitoring | ◫ **Incident vision** | Upload a photo or capture from the camera; the frame is measured (brightness, contrast, edge density, dark share, saturation) and triaged into a severity band with a confidence %, findings, a pre-file checklist, and a model-based congestion-risk estimate. Reports persist with the triage attached. |
| Smart irrigation (auto/manual) | ☰ **Smart control** | Rule-based signal plan in automatic mode, or a manual green-phase override (15–90 s) that is clamped and explained. A five-device rack (adaptive signal, ramp meter, VMS, lighting, depot barrier) with recommendations and per-device state/mode control. |
| Weather monitoring | ☁ **Weather monitor** | Six weather KPIs, a 0–100 weather index in four bands, trend charts, rule-based operator suggestions, and the latest weather for every junction. |
| AI farming assistant | ◇ **AI assistant** | Grounded Q&A about flow, forecasts, congestion, weather, devices, alerts, and corridor plans — each answer quoting the readings that produced it. |
| Edge AI status | ≋ **Edge & 5G** | Champion metadata, edge-cache status and files, and a **run-local-inference** button that measures real latency on this host, plus the simulated 5G link. |
| 5G network monitoring | ≋ **Edge & 5G** | Signal strength, upload/download, and latency per junction, with link health folded into fleet status and offline fallback. |
| Alerts & notifications | △ **Alert centre** | Six alert categories with counts, filters by severity/category/junction, a current-tick or unacknowledged-only toggle, and persisted acknowledgement. |
| Crop recommendation | ✦ **Corridor advisory** | Six candidate plans (adaptive retiming, ramp metering, traveller messaging, incident patrol, lighting response, demand broadcast) scored 0–100 % and ranked, with adoption logged as an operator action. |
| Analytics | ≡ **Analytics & reports** | Five tabs spanning the historical baseline, virtual telemetry, forecasts, incidents, and operations history. |
| Farm map (green/amber/red zones) | ▦ **Junction map** | Schematic SVG city map with animated flow lanes and green/amber/red zones derived from simulated occupancy, plus zone counts and headroom. |
| Device control | ☰ **Smart control** | Folded into the device rack: state, mode, recommendation match, and an update path that writes the new state and an operator action. |
| Reports | ≡ **Analytics & reports** | An export pack of eight files: daily and weekly Markdown digests plus CSV exports for observations, sensor history, forecasts, incidents, operator actions, and device states. |

### Screenshot gallery

| | |
| --- | --- |
| ![Operator access and profile](docs/screenshots/access.png) | ![Live sensor monitoring](docs/screenshots/sensors.png) |
| *Operator access — session profile and role matrix* | *Live sensors — telemetry and trend charts* |
| ![Weather monitoring](docs/screenshots/weather.png) | ![Alert centre](docs/screenshots/alerts.png) |
| *Weather monitor — index, trends, suggestions* | *Alert centre — categories and acknowledgement* |
| ![Schematic junction map](docs/screenshots/map.png) | ![Demand forecast and outlook](docs/screenshots/forecast.png) |
| *Junction map — schematic zones* | *Forecast & outlook — recursive outlook* |
| ![Smart control](docs/screenshots/control.png) | ![Incident vision](docs/screenshots/incidents.png) |
| *Smart control — signal plan and device rack* | *Incident vision — scene triage and filing* |
| ![Corridor advisory](docs/screenshots/corridor.png) | ![Edge and 5G](docs/screenshots/edge.png) |
| *Corridor advisory — ranked plans* | *Edge & 5G — local inference and link* |
| ![Traffic AI assistant](docs/screenshots/assistant.png) | ![Analytics and reports](docs/screenshots/analytics.png) |
| *AI assistant — grounded answers* | *Analytics & reports — exports* |
| ![MLOps and data](docs/screenshots/mlops.png) | ![Simulation lab](docs/screenshots/lab.png) |
| *MLOps & data — champion and drift* | *Simulation lab — 5G, edge, failover* |

---

## Theme system with generated backgrounds

The deck ships four appearance options. Switch them at runtime in the sidebar under **Appearance → Deck theme**, or link straight to one:

| Theme | Accent | Background | Link |
| --- | --- | --- | --- |
| **Command cyan** | cyan `#22d3ee` | signal grid, no photo | [`/?theme=command`](http://localhost:8501/?theme=command) |
| **Night run** | indigo `#8b9bff` | after-dark expressway | [`/?theme=night-run`](http://localhost:8501/?theme=night-run) |
| **Amber watch** | sodium amber `#f59e0b` | dusk arterial | [`/?theme=amber-watch`](http://localhost:8501/?theme=amber-watch) |
| **Deep field** | lime `#a3e635` | green corridor | [`/?theme=deep-field`](http://localhost:8501/?theme=deep-field) |

| Night run background | Amber watch background | Deep field background |
| --- | --- | --- |
| ![Night run artwork](assets/themes/night-run.jpg) | ![Amber watch artwork](assets/themes/amber-watch.jpg) | ![Deep field artwork](assets/themes/deep-field.jpg) |

What a switch actually does:

- **Repaints the palette.** Every surface, border, heading, and accent in the deck reads from CSS custom properties (`--accent`, `--ink`, `--panel`, `--panel-2`, `--text`, `--muted`, `--line`, the three accent stops and their RGB triplets). One `:root` override recolours the whole interface — sidebar, hero, KPI cards, junction dials, the SVG map, device rack, and tables.
- **Swaps the background layer stack.** Grid lines, accent glows, a dimming veil, and — for the three photo themes — the generated artwork, all composited in one `background-image` list with per-layer sizes and repeats.
- **Retints the charts.** `ui.style_fig()` reads the active theme, so Plotly's colorway, axis grid, hover cards, and legend follow the same palette instead of a hardcoded one.
- **Keeps status meaning fixed.** Mint = good, amber = warn, rose = bad, violet = sensor are literals in `TONE`; only the brand accent moves, so changing theme never changes what a colour *means*.

Two session controls sit beside the picker:

- **Background depth** (55–95) — the veil alpha behind the artwork. Lower lets the city show through, higher pushes it back for projector readability. Photo themes only; Command cyan has no artwork to dim.
- **Interface motion** — stops the ticker, signal lamps, driving cars, and map flow lanes. A `prefers-reduced-motion` media query applies the same suppression automatically.

The three photo backgrounds are generated raster assets in [`assets/themes/`](assets/themes) (1280 px wide, re-compressed for web). They are embedded as base64 data URIs at render time, so the deck issues no extra image requests and keeps working offline; only the active theme's image is embedded. Theme choice, depth, and motion are session state — like the operator profile, they reset when the browser session ends, and they are never written to disk.

---

## How the project works

```mermaid
flowchart LR
  A[Four-junction traffic CSV] --> B[Validate timestamps, junctions, counts]
  B --> C[Time-series features + chronological split]
  C --> D[Train Random Forest pipeline]
  D --> E[MLflow run + model registry version]
  E --> F{Holdout RMSE better than champion?}
  F -->|yes| G[Move champion alias + refresh edge cache]
  F -->|no| H[Keep as registered candidate]
  G --> I[FastAPI + Streamlit serving]
  B --> K[PSI drift monitor]
  K --> L[Review, retrain, or investigate]
  L --> C
  I --> M[Virtual telemetry + advisory engine]
  M --> N[Workspace: alerts, control, advisory, reports]
  N --> J[(SQLite predictions, incidents, actions, devices, telemetry, alerts)]
```

1. **Data validation.** `data/traffic.csv` is cleaned and normalised: timestamps parsed, junctions standardised, vehicle counts validated, duplicate junction-hours removed, and continuous hourly histories enforced per junction.
2. **Feature engineering.** For every junction the pipeline builds `lag_1`, `lag_2`, `lag_3`, `lag_24`, `lag_168`, `rolling_mean_24`, cyclical `hour`, `weekday`, and `month` encodings, plus `temperature_c` and `is_event`. The target is the **next** hour's vehicle count. The train/test split is chronological (80/20) — never a random shuffle.
3. **Training.** A scikit-learn `Pipeline` one-hot encodes the junction and feeds numeric features to a `RandomForestRegressor` (120 trees, depth 20, `min_samples_leaf=3`, `max_features=0.6`, `random_state=42`). Parameters, metrics (RMSE, MAE, R²), dataset identity, signature, and an example input are logged to MLflow.
4. **Registry and promotion.** Every run produces a new model version. The `champion` alias is assigned only when the new holdout RMSE beats the incumbent on the same dataset; otherwise the model is stored as a candidate. The champion is also written to `edge_cache/` for offline use.
5. **Serving.** `traffic_mlops/prediction.py` loads the champion once and caches it, forces single-threaded inference for low latency, and serves both single-step and recursive multi-step forecasts. The same code path powers the Streamlit pages and the FastAPI service.
6. **Simulation.** `traffic_mlops/simulation.py` derives flow from each junction's historical hour profile, then derives occupancy from capacity, speed from occupancy, and weather (including a diurnal temperature curve), link, and edge readings from seeded distributions. A fresh workspace is seeded with 24 virtual ticks across three simulated days, and faults, cloud outages, and device states are inputs, so a demo is reproducible.
7. **Advisory.** `traffic_mlops/advisory.py` converts the latest readings into a signal plan (derived and overridable), device recommendations, ranked corridor plans, weather guidance, alert categories, and the frame-triage score. All of it is deterministic and rule-based, so it can be tested and explained.
8. **Operation.** Every operator-facing value is either observed data (labeled as source) or simulated (labeled as virtual). Forecasts, incident reports (with triage), device changes, acknowledgements, and operator actions persist in SQLite; the analytics pages and report exports read them back.

### Model at a glance

| Item | Value |
| --- | --- |
| Champion | **v4**, `TrafficFlowPredictor@champion` |
| Holdout RMSE / MAE / R² | **4.73 / 2.94 / 0.969** vehicles per hour |
| Validation | Last 20 % of each junction's timeline, chronological |
| Serving latency | ≈ 53 ms per single-row prediction after warm-up; ≈ 2.5 s for a 24-step recursive outlook |
| Drift metric | Population Stability Index, review threshold 0.2 |

---

## Quick start

### Local (Windows PowerShell)

```powershell
python -m venv .traffic-venv
.\.traffic-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m traffic_mlops.cli bootstrap   # validate data, train, register, promote champion
streamlit run app.py
```

Open **http://localhost:8501**. Start the API and the MLflow UI in two more activated terminals:

```powershell
uvicorn traffic_mlops.api:app --reload
mlflow ui --backend-store-uri sqlite:///mlflow_project28.db --host 127.0.0.1 --port 5000
```

API docs: **http://localhost:8000/docs** · MLflow: **http://localhost:5000**

### Local (macOS / Linux)

```bash
python3 -m venv .traffic-venv
source .traffic-venv/bin/activate
pip install -r requirements.txt
python -m traffic_mlops.cli bootstrap
streamlit run app.py
```

### Docker Compose

```bash
docker compose up --build
```

| Service | URL | Notes |
| --- | --- | --- |
| Dashboard | http://localhost:8501 | Streamlit |
| API | http://localhost:8000/docs | FastAPI |
| MLflow | http://localhost:5000 | Tracking server |

Volumes `mlflow_data` and `app_data` persist tracking artifacts and application history.

### Five-minute demo script

1. **Appearance** — in the sidebar, switch between Command cyan, Night run, Amber watch, and Deep field, then drag **Background depth** to show the artwork moving behind the deck.
2. **Operator access** — set the operator name, role, and corridor, then save the session profile.
3. **Dashboard** — read the hero status, the ticker, and the eight KPIs, then skim the next-best actions.
4. **Simulation lab** — press **Generate next virtual minute**; occupancy, weather, 5G, and edge readings update and are persisted.
5. **Live sensors / Weather monitor** — inspect the new tick and the weather index guidance.
6. **Alert centre** — filter by category, then acknowledge an alert and confirm it is marked reviewed.
7. **Forecast & outlook** — run the next-hour forecast, then **Project demand curve** for the 24-hour outlook with the uncertainty band.
8. **Smart control** — switch the signal plan to manual, set a green phase, apply the device recommendations, and record an operator action.
9. **Incident vision** — upload a road photo, review the triage score and checklist, and file the report.
10. **Corridor advisory** — read the ranked plans and adopt the top one.
11. **Edge & 5G** — run a local inference and read the measured latency, then flip the cloud link offline and confirm the edge fallback.
12. **MLOps & data** — run the drift check, read the PSI bars, then train and register a new version (the champion moves only if RMSE improves).
13. **Analytics & reports** — download the weekly Markdown digest and the sensor-history CSV.

---

## API reference

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service status and champion readiness |
| `GET` | `/model` | Champion metadata: version, run id, RMSE/MAE/R², dataset, status |
| `POST` | `/predict` | Next-hour forecast; persists the result |
| `POST` | `/forecast/horizon` | Recursive 1–48 hour outlook with peak statistics |
| `POST` | `/train` | Train, evaluate, register a version, consider promotion |
| `GET` | `/monitor` | PSI drift check over recent features |
| `GET` | `/predictions?limit=100` | Persisted forecast history |
| `POST` | `/simulation/tick` | Generate virtual sensor/weather/network/alert records |
| `GET` | `/simulation/telemetry` | Persisted virtual telemetry ticks |
| `GET` | `/simulation/alerts` | Persisted virtual alerts |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"junction":"Junction 1","vehicles":220,"temperature_c":20,"is_event":0,"offline":false}'

curl -X POST http://localhost:8000/forecast/horizon \
  -H "Content-Type: application/json" \
  -d '{"junction":"Junction 1","hours":24,"offline":false}'

curl -X POST http://localhost:8000/simulation/tick \
  -H "Content-Type: application/json" \
  -d '{"cloud_online":true,"failed_junction":null}'
```

Forecast output is a model estimate for planning support — not a traffic-control instruction.

---

## MLOps details

- **Tracking backend.** Local by default: SQLite `mlflow_project28.db` with artifacts under `mlartifacts/project28/`. The earlier starter database `mlflow.db` is left untouched.
- **Registry integrity.** Model versions are immutable; only the alias changes. If the dataset tag differs from the champion's, the comparison baseline resets so metrics from different datasets are never compared.
- **Fast loading.** `prediction._load_pipeline` loads the `skops` artifact directly with an explicit trusted-type allow-list (≈1 s) instead of the MLflow PyFunc wrapper (tens of seconds on a 100 MB forest), caches it with `lru_cache`, and falls back to `mlflow.sklearn.load_model` if the artifact format differs.
- **Offline mode.** `load_champion(offline=True)` and the simulation cloud toggle force host-local inference from `edge_cache/`, and every response is marked `offline: true`.
- **Drift monitoring.** `GET /monitor` and the MLOps page compare older vs recent feature windows with PSI; the current dataset produces a `rolling_mean_24` PSI of ≈0.59, comfortably above the review threshold — a deliberate demonstration of drift detection.
- **Model boundary.** Only the demand model is trained. Signal plans, device recommendations, corridor scores, weather guidance, and scene triage are transparent rule-based logic in `advisory.py`, kept separate so neither is mistaken for the other.
- **CI.** GitHub Actions (`.github/workflows/ci.yml`) installs the requirements, runs `pytest`, byte-compiles the app, then smoke-tests `generate-data` → `train` → `monitor` against a throwaway SQLite tracking store.

---

## Data and attribution

The bundled CSV is derived from Kaggle's [Traffic Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset), contributed by `fedesoriano`. The Kaggle data card reports 48,120 hourly vehicle-count observations for four junctions and states educational-use-only / source-confidential terms. Use this copy for this academic project and follow the source page's terms for any redistribution.

| Property | Value |
| --- | --- |
| Rows | 48,120 hourly observations |
| Columns | `timestamp`, `junction`, `vehicles` (no weather fields) |
| Junctions 1–3 | 14,592 rows each, continuous hourly history |
| Junction 4 | 4,344 rows, starts later than the others |
| Window | 1 Nov 2015 → 30 Jun 2017 |

`data/raw/kaggle_junction_traffic.zip` is the downloaded archive; `data/traffic.csv` is the prepared training file. Rebuild it with:

```bash
python scripts/prepare_junction_data.py
```

### Bring your own data

Required columns: `timestamp`, `junction`, `vehicles`. Optional: `temperature_c`, `is_event`. Provide a continuous hourly history of at least **168 hours per junction** so all lag features are available. Upload a CSV from the sidebar for a session, or replace `data/traffic.csv` for the API and CLI. The uploader validates the schema and reports missing hours.

The source CSV has no weather fields; the normaliser supplies neutral defaults for model compatibility, and the simulator generates weather, network, and sensor values that are labeled **virtual** in every view.

---

## Testing and verification

```bash
python -m pytest -q
```

`tests/test_pipeline.py` covers schema validation and deduplication, time-series feature construction, PSI drift behaviour, recursive horizon ordering, and horizon input validation. `tests/test_advisory.py` covers alert categorisation, weather index bands and suggestions, the signal plan (including override clamping), corridor ranking bounds, device recommendations, and the image triage (dark vs bright frames, confidence, signature, non-image rejection). Current status:

- **13 tests passing** (≈10 s on the bundled data)
- CLI `bootstrap`, `train`, and `monitor` verified end to end
- API smoke-tested: `/health`, `/model`, `/predict`, `/forecast/horizon`, `/simulation/tick`, `/monitor`, `/predictions`, `/openapi.json`
- All fifteen modules rendered and exercised in a real browser with no Streamlit exceptions

---

## Configuration and persistence

| Variable | Purpose |
| --- | --- |
| `MLFLOW_TRACKING_URI` | Optional tracking backend; defaults to project-local `mlflow_project28.db` |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | Credentials for a remote tracking server |
| `TRAFFIC_APP_DB` | SQLite file for predictions, incidents, actions, devices, telemetry, alerts (default `traffic_app.db`) |
| `TRAFFIC_EDGE_CACHE` | Directory for the versioned local champion used in offline mode (default `edge_cache/`) |

SQLite tables: `predictions`, `incident_reports` (with triage JSON), `operator_actions`, `virtual_telemetry`, `virtual_alerts` (with acknowledgement), `devices` (five seeded virtual controllers).

### Optional DagsHub tracking

```powershell
pip install -r requirements-dagshub.txt
$env:DAGSHUB_REPO_OWNER = "MAHIJA2007"
$env:DAGSHUB_REPO_NAME = "AI_Based_Traffic"
$env:DAGSHUB_USER_TOKEN = "your-dagshub-token"
python -m traffic_mlops.cli train
```

Never commit tokens. For Compose with DagsHub, copy `.env.example` to `.env`, set `MLFLOW_TRACKING_URI=https://dagshub.com/<owner>/<repository>.mlflow`, and keep `.env` private.

---

## Project structure

```text
app.py                        Streamlit command deck (15 modules)
traffic_mlops/ui.py           Design system: theme registry, hero, dials, map, charts, racks
traffic_mlops/data.py         Loading, validation, normalisation, demo data
traffic_mlops/features.py     Junction-aware time-series feature pipeline
traffic_mlops/training.py     Chronological training, evaluation, registry promotion
traffic_mlops/prediction.py   Cached champion loading, edge cache, horizon forecasting
traffic_mlops/monitoring.py   PSI drift monitoring
traffic_mlops/simulation.py   Virtual sensors, weather, network, alerts, signal timing
traffic_mlops/advisory.py     Signal plans, device recommendations, corridor scores,
                              weather index, alert categories, image triage
traffic_mlops/history.py      SQLite persistence: predictions, incidents, actions,
                              devices, telemetry, alerts
traffic_mlops/api.py          FastAPI service
traffic_mlops/cli.py          bootstrap / train / monitor / generate-data
scripts/prepare_junction_data.py  Prepare the four-junction CSV
tests/test_pipeline.py        Pipeline and forecasting tests
tests/test_advisory.py        Advisory-engine and triage tests
data/traffic.csv              Ready-to-run dataset
edge_cache/                   Versioned champion artifact for offline serving
assets/themes/                Generated background artwork for the three photo themes
docs/screenshots/             Interface captures used in this README
docker-compose.yml            Dashboard + API + MLflow
```

### Workflow commands

```bash
python -m traffic_mlops.cli bootstrap                 # first-time setup
python -m traffic_mlops.cli generate-data             # reproducible synthetic fallback
python -m traffic_mlops.cli train                     # train on data/traffic.csv
python -m traffic_mlops.cli train --data my.csv       # train on another dataset
python -m traffic_mlops.cli monitor                   # run the PSI drift check
```

---

## Scope, limitations, and safe use

- All telemetry is **simulated** and labeled virtual. The project demonstrates an MLOps lifecycle; it does not control real signals or ingest live road data.
- The incident-vision score is a **heuristic scene triage** computed from frame brightness, contrast, edge density, dark share, and saturation. It is not a trained image classifier; the photo itself is not interpreted beyond those measurements, and the accompanying risk figure is the demand model compared with a junction's historical 95th percentile.
- Signal plans, corridor scores, device recommendations, and weather suggestions are rule-based. They are decision support, not control commands, and manual overrides are clamped to a safe range before being logged.
- Device states and alert acknowledgements persist locally in SQLite. They simulate a device fleet; no real hardware is addressed.
- Operator name, ID, region, and junction are session profile fields, not authentication. Before any public deployment, add HTTPS, authentication, secret management, durable storage, and restricted CORS/network access.
- Drift detection flags distribution change; it does not prove accuracy loss. Always confirm with observed outcomes before retraining or deploying.

---

## Team

**Project title:** AI-Based Traffic Prediction System with MLOps
**Project number:** 28

| Student ID | Name |
| --- | --- |
| 2400080129 | K.Mahija |
| 2400080003 | Sk.Anjum |
| 2400080005 | P.Hasitha |
