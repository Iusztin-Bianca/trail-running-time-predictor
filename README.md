# Trail Running Time Predictor

Machine learning web application that predicts trail running race completion times from GPX files, trained on real Strava data.

[![Monthly Training](https://github.com/Iusztin-Bianca/trail-running-time-predictor/actions/workflows/monthly_training.yml/badge.svg)](https://github.com/Iusztin-Bianca/trail-running-time-predictor/actions/workflows/monthly_training.yml) ![Python](https://img.shields.io/badge/python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.127-green) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-red)

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript |
| Backend | FastAPI, Python 3.12 |
| ML | scikit-learn, Ridge Regression, XGBoost, SHAP |
| Storage | Azure Blob Storage |
| CI/CD | GitHub Actions |
| Containerization | Docker |

## Demo

![App screenshot](docs/screenshots/main_page.PNG)
![Prediction result](docs/screenshots/prediction_modal.PNG)

## How It Works

### Inference (Prediction)
1. User uploads a GPX file
2. Route is split into segments (of maximum 1000m) by terrain type (uphill / downhill / flat)
3. For each segment, 16 features are extracted (gradient, distance, elevation, energy cost, etc.)
4. Model predicts time for each segment independently
5. Segment times are summed → total predicted race time

### ML Pipeline
- **Training data**: personal activities from Strava application (trail runs with elevation ≥ 150m)
- **Approach**: segment-level regression — each segment is one training observation, dramatically increasing dataset size
- **Model**: Ridge Regression (chosen over XGBoost due to better generalization on small datasets)
- **Retraining**: automated monthly via GitHub Actions → new model saved to Azure Blob Storage + committed to repo as fallback
  **Feature importance (SHAP analysis - the 5 most significant features)**

| Rank | Feature | SHAP Value |
|---|---|---|
| 1 | `segment_energy_cost` | 101.6 — dominant predictor, captures combined gradient + distance cost |
| 2 | `elevation_loss_m` | 63.1 — descents significantly impact pace |
| 3 | `downhill_cost` | 26.1 — braking effort on steep descents |
| 4 | `elevation_gain_m` | 24.2 — climbing effort |
| 5 | `is_race` | 22.9 — race effort shifts pace considerably |

### Feature Engineering

Each segment is described by 17 features:

**Segment geometry**
| Feature | Description |
|---|---|
| `segment_distance_m` | Segment length in meters |
| `segment_time_sec` | Segment duration in seconds (target during training) |
| `segment_pace_mps` | Average pace in m/s |
| `cumulative_distance` | Total distance from activity start to end of segment |
| `cumulative_elevation` | Total elevation gain from activity start to end of segment |

**Elevation & gradient**
| Feature | Description |
|---|---|
| `elevation_gain_m` | Elevation gained within the segment (max(0, Δalt)) |
| `elevation_loss_m` | Elevation lost within the segment (max(0, -Δalt)) |
| `avg_gradient` | Mean gradient = Δelevation / segment_distance |
| `std_gradient` | Standard deviation of gradient in segment - measures terrain irregularity |
| `max_uphill_gradient` | Steepest uphill point (uphill segments only) |
| `max_downhill_gradient` | Steepest downhill point (downhill segments only) |
| `avg_elevation` | Mean altitude of the segment (affects air density / fatigue) |

**Effort type**
| Feature | Description |
|---|---|
| `is_race` | 1 if predicting a race effort, 0 otherwise |
| `is_easy` | 1 if predicting a recovery run, 0 otherwise |

**Biomechanical cost**
| Feature | Formula | Description |
|---|---|---|
| `uphill_cost` | `distance × (1 + 6 × gradient)` | Extra effort penalty for climbing |
| `downhill_cost` | `distance × (1 + 6 × \|gradient\|)` | Braking effort penalty for descending |
| `segment_energy_cost` | Minetti et al. (2002) | Metabolic energy cost (J/kg) per segment |

The Minetti formula models the metabolic cost of running on a slope:

$$E = (155.4g^5 - 30.4g^4 - 43.3g^3 + 46.3g^2 + 19.5g + 3.6) \times d$$

where `g` is the signed gradient (positive = uphill, negative = downhill) and `d` is segment distance in meters.

## Project Structure









