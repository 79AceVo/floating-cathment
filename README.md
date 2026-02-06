# floating-catchment

Modular Python library for **Floating Catchment Area (FCA)** spatial accessibility methods.

Each method is a standalone function you can call independently, with pluggable distance-decay functions and flexible input formats.

## Methods

| Function | Method | Reference |
|----------|--------|-----------|
| `two_sfca()` | Two-Step FCA | Luo & Wang (2003) |
| `e2sfca()` | Enhanced 2SFCA | Luo & Qi (2009) |
| `three_sfca()` | Three-Step FCA | Wan et al. (2012) |
| `m2sfca()` | Modified 2SFCA | Delamater (2013) |
| `kd2sfca()` | Kernel Density 2SFCA | Dai (2010) |

## Installation

```bash
pip install -e .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import floating_catchment as fca

# Load supply (hospitals) and demand (population centres)
supply_gdf = fca.load_points("hospitals.csv", id_col="id")
demand_gdf = fca.load_points("population.csv", id_col="id")

# Build a cost matrix (Euclidean distance)
cost = fca.euclidean_distance_matrix(supply_gdf, demand_gdf)

# Or load a pre-computed travel-time matrix (ESRI OD Cost Matrix)
cost = fca.load_cost_matrix("od_cost.csv", fmt="long",
                            origin_col="OriginID",
                            dest_col="DestinationID",
                            cost_col="Total_Minutes")

# Run any FCA method
scores = fca.two_sfca(supply_gdf["capacity"], demand_gdf["population"],
                      cost, threshold=30)

# Visualise
demand_gdf["accessibility"] = scores
fig = fca.plot_accessibility(demand_gdf, supply_gdf=supply_gdf)
m   = fca.interactive_map(demand_gdf, supply_gdf=supply_gdf)
```

## Module Reference

### FCA Methods

All methods share the same core signature:

```python
scores = method(supply, demand, cost_matrix, threshold, ...)
```

- **`supply`** — `pd.Series` of capacity values, indexed by supply location ID
- **`demand`** — `pd.Series` of population values, indexed by demand location ID
- **`cost_matrix`** — `pd.DataFrame` with demand IDs as rows and supply IDs as columns
- **`threshold`** — catchment size (same units as the cost matrix)
- Returns a `pd.Series` of accessibility scores indexed by demand location

#### `two_sfca(supply, demand, cost_matrix, threshold, decay_fn="binary", **decay_kwargs)`

Classic two-step approach. Step 1 computes a supply-to-demand ratio within each supply catchment; step 2 sums ratios for each demand location. Pass any decay function name to override the default binary (step) weighting.

#### `e2sfca(supply, demand, cost_matrix, zones, weights)`

Enhanced variant with concentric distance zones. `zones` is a sorted list of breakpoints (e.g. `[10, 20, 30]`) and `weights` is the corresponding weight for each zone (e.g. `[1.0, 0.68, 0.22]`).

#### `three_sfca(supply, demand, cost_matrix, threshold, decay_fn="gaussian", **decay_kwargs)`

Adds a selection-probability step (step 0) that models competition — the probability that a demand location selects a particular supply is proportional to supply capacity weighted by distance decay.

#### `m2sfca(supply, demand, cost_matrix, threshold, decay_fn="gaussian", **decay_kwargs)`

Normalises distance-decay weights per demand location so they sum to 1, accounting for suboptimal spatial configurations.

#### `kd2sfca(supply, demand, cost_matrix, threshold, kernel="epanechnikov", **kernel_kwargs)`

Uses a kernel density function as the decay weight. Default is the Epanechnikov kernel; any function from the `decay` module can be used.

### Distance Decay Functions (`floating_catchment.decay`)

All decay functions have the signature `fn(d, d0, **params) -> np.ndarray`:

| Function | Formula (within catchment) | Parameters |
|----------|---------------------------|------------|
| `binary(d, d0)` | 1 | — |
| `linear(d, d0)` | 1 − d/d₀ | — |
| `gaussian(d, d0, beta=1.0)` | exp(−d²/(β·d₀²)) | `beta`: shape |
| `epanechnikov(d, d0)` | ¾(1 − (d/d₀)²) | — |
| `butterworth(d, d0, n=2)` | 1/(1 + (d/d₀)^2n) | `n`: filter order |
| `power(d, d0, alpha=1.5)` | 1 − (d/d₀)^α | `alpha`: exponent |
| `zonal(d, zones, weights)` | step-wise weights per zone | `zones`, `weights` |

All return 0 for distances beyond the threshold.

### Catchment (`floating_catchment.catchment`)

| Function | Description |
|----------|-------------|
| `euclidean_distance_matrix(supply_gdf, demand_gdf)` | Pairwise Euclidean distances from point geometries |
| `filter_cost_matrix(cost_matrix, threshold)` | Set entries above threshold to `inf` |

### IO (`floating_catchment.io`)

| Function | Description |
|----------|-------------|
| `load_points(source, *, lat_col, lon_col, crs, id_col)` | Load from CSV path, DataFrame, or GeoDataFrame |
| `load_cost_matrix(source, *, fmt, origin_col, dest_col, cost_col)` | Load ESRI OD long format or wide matrix |

The cost matrix loader supports:
- **`fmt="long"`** — three-column table (OriginID, DestinationID, Total_Cost), the default ESRI OD Cost Matrix export
- **`fmt="wide"`** — pre-pivoted matrix with demand as rows and supply as columns

### Visualisation (`floating_catchment.viz`)

| Function | Description |
|----------|-------------|
| `plot_accessibility(gdf, column, *, supply_gdf, ...)` | Static matplotlib choropleth map |
| `interactive_map(gdf, column, *, supply_gdf, ...)` | Interactive Folium/Leaflet map |

Both accept optional `supply_gdf` to overlay supply locations.

## Demo Notebook

See [`notebooks/demo.ipynb`](notebooks/demo.ipynb) for a full walkthrough with synthetic data covering all five methods, decay function plots, comparison charts, and both map types.

## Running Tests

```bash
pytest
```

78 tests covering all modules: decay functions, catchment computation, IO loading, all 5 FCA methods, and visualisation.

## Project Structure

```
src/floating_catchment/
├── __init__.py          # Public API
├── catchment.py         # Distance matrix + threshold filtering
├── decay.py             # 7 distance-decay functions
├── io.py                # CSV / GeoDataFrame / ESRI OD matrix loading
├── viz.py               # Static + interactive maps
└── methods/
    ├── two_sfca.py      # 2SFCA
    ├── e2sfca.py        # E2SFCA
    ├── three_sfca.py    # 3SFCA
    ├── m2sfca.py        # M2SFCA
    └── kd2sfca.py       # KD2SFCA
tests/                   # 78 tests
notebooks/demo.ipynb     # Interactive demo
```

## References

- Dai, D. (2010). Black residential segregation, disparities in spatial access to health care facilities, and late-stage breast cancer diagnosis in metropolitan Detroit. *Health & Place*, 16(5), 1038–1052.
- Delamater, P. L. (2013). Spatial accessibility in suboptimally configured health care systems: A modified two-step floating catchment area (M2SFCA) metric. *Health & Place*, 24, 30–43.
- Luo, W., & Qi, Y. (2009). An enhanced two-step floating catchment area (E2SFCA) method for measuring spatial accessibility to primary care physicians. *Health & Place*, 15(4), 1100–1107.
- Luo, W., & Wang, F. (2003). Measures of spatial accessibility to health care in a GIS environment: synthesis and a case study in the Chicago region. *Environment and Planning B*, 30(6), 865–884.
- Wan, N., Zou, B., & Sternberg, T. (2012). A three-step floating catchment area method for analyzing spatial access to health services. *International Journal of Geographical Information Science*, 26(6), 1073–1089.

## License

MIT
