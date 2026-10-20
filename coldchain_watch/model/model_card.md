# Model Card — ColdChain Watch Pakistan
Generated: 2026-10-19 21:06 UTC

## Purpose
Predict external air temperature at Karachi for 1h, 6h, and 24h ahead.
Predictions are compared against FAO cold-storage thresholds per crop
to generate heat alerts for warehouse managers.

## Training Data
- Source 1: Open-Meteo Historical API (CC BY 4.0)
- Source 2: NASA POWER Hourly API (NASA Open Data)
- Location: Karachi, Pakistan (24.8607°N, 67.0011°E)
- Period: 2 years of hourly data
- Features: 36 engineered features from Gold layer

## Declared Baseline
**Same hour last week** (`temp_lag_168h`) — declared before results were seen.

## Train / Test Split
- Time-based split: most recent 8 weeks held out as test set
- No random shuffle — future data never seen during training
- Scalers/imputers fit on training data only

## Results

| Horizon | Model MAE | Baseline MAE | Improvement | Beats Baseline |
|---------|-----------|--------------|-------------|----------------|
| 1h | 0.221°C | 1.092°C | +79.8% | ✅ Yes |
| 6h | 0.472°C | 2.370°C | +80.1% | ✅ Yes |
| 24h | 0.511°C | 1.042°C | +50.9% | ✅ Yes |

## Errors by Group

### Horizon: 1h

```
 group_type       group_value  n_rows    mae  max_error
     season            autumn     455 0.2242     1.1786
     season           monsoon     889 0.2193     4.5550
hour_bucket afternoon (12–17)     336 0.1964     0.9475
hour_bucket   evening (18–23)     336 0.1553     0.9407
hour_bucket    morning (6–11)     336 0.2849     4.5550
hour_bucket       night (0–5)     336 0.2473     1.1956
 is_flagged             False    1332 0.2194     4.5550
 is_flagged              True      12 0.3914     0.9226
```

### Horizon: 6h

```
 group_type       group_value  n_rows    mae  max_error
     season            autumn     450 0.5210     4.5239
     season           monsoon     889 0.4465     3.8958
hour_bucket afternoon (12–17)     336 0.3221     1.0778
hour_bucket   evening (18–23)     331 0.4148     3.0885
hour_bucket    morning (6–11)     336 0.3785     1.9176
hour_bucket       night (0–5)     336 0.7699     4.5239
 is_flagged             False    1327 0.4693     4.5239
 is_flagged              True      12 0.7198     1.1094
```

### Horizon: 24h

```
 group_type       group_value  n_rows    mae  max_error
     season            autumn     432 0.5853     4.3061
     season           monsoon     889 0.4749     3.5823
hour_bucket afternoon (12–17)     330 0.4346     3.3392
hour_bucket   evening (18–23)     331 0.3614     1.5524
hour_bucket    morning (6–11)     330 0.7640     4.3061
hour_bucket       night (0–5)     330 0.4845     3.1048
 is_flagged             False    1309 0.5112     4.3061
 is_flagged              True      12 0.4850     1.9115
```

## Crop Thresholds (FAO)
| Crop | Safe Max Temp | Source |
|------|--------------|--------|
| Potato | 4°C | FAO Postharvest Compendium |
| Tomato | 13°C | FAO Postharvest Compendium |
| Onion | 2°C | FAO Postharvest Compendium |

## Limitations
- Model predicts **external ambient temperature**, not internal cold-store temperature.
- Actual spoilage depends on insulation quality, door-opening frequency,
  pre-cooling efficiency — none of which are in the public data.
- Karachi external temperature **always** exceeds potato and onion thresholds.
  Heat alerts for these crops indicate the energy cost of maintaining cold-chain,
  not an imminent spoilage event.
- Model trained on Karachi only. Do not apply to other cities without retraining.
- Predictions beyond 24h are not provided — uncertainty grows too large.

## Honest AI Use
See AI_use_log.md for details of all AI-assisted steps.