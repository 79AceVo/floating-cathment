"""
Data loading utilities.

Handles:

- **Point data** from CSV (with lat/lon columns) or GeoDataFrames
  (shapefiles, GeoJSON, GeoPackage, etc.).
- **Cost / travel-time matrices** from ESRI OD Cost Matrix exports
  (long format: OriginID, DestinationID, Total_Cost) or from
  wide-format CSV / Excel files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
#  Point data
# --------------------------------------------------------------------------- #

def load_points(
    source: Union[PathLike, pd.DataFrame, gpd.GeoDataFrame],
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    crs: str = "EPSG:4326",
    id_col: str | None = None,
) -> gpd.GeoDataFrame:
    """Load point data from various formats into a GeoDataFrame.

    Parameters
    ----------
    source
        - A file path to CSV, GeoJSON, Shapefile, GeoPackage, etc.
        - A plain ``DataFrame`` with coordinate columns.
        - An existing ``GeoDataFrame`` (returned as-is after optional
          index adjustment).
    lat_col, lon_col
        Column names for latitude / longitude when loading from CSV or
        a plain DataFrame.  Ignored when *source* is already spatial.
    crs
        Coordinate reference system string.  Only used when building
        geometry from lat/lon columns.
    id_col
        Optional column to use as the index.

    Returns
    -------
    GeoDataFrame
        With a ``geometry`` column of Point objects.
    """
    if isinstance(source, gpd.GeoDataFrame):
        gdf = source.copy()
    elif isinstance(source, pd.DataFrame):
        gdf = _df_to_geodf(source, lat_col, lon_col, crs)
    else:
        path = Path(source)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            gdf = _df_to_geodf(df, lat_col, lon_col, crs)
        else:
            # GeoJSON, Shapefile, GPKG, etc. — let geopandas handle it
            gdf = gpd.read_file(path)
    if id_col and id_col in gdf.columns:
        gdf = gdf.set_index(id_col)
    return gdf


def _df_to_geodf(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    crs: str,
) -> gpd.GeoDataFrame:
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)


# --------------------------------------------------------------------------- #
#  Cost / travel-time matrix
# --------------------------------------------------------------------------- #

def load_cost_matrix(
    source: Union[PathLike, pd.DataFrame],
    *,
    fmt: str = "long",
    origin_col: str = "OriginID",
    dest_col: str = "DestinationID",
    cost_col: str = "Total_Cost",
) -> pd.DataFrame:
    """Load a cost / travel-time matrix.

    Parameters
    ----------
    source
        File path (CSV or Excel) or an already-loaded DataFrame.
    fmt : ``"long"`` | ``"wide"``
        - ``"long"``:  Three-column table (origin, destination, cost),
          the default ESRI OD Cost Matrix export format.
        - ``"wide"``:  Pre-pivoted matrix where rows = demand (origins)
          and columns = supply (destinations).
    origin_col, dest_col, cost_col
        Column names for the long format.

    Returns
    -------
    DataFrame
        Wide-format matrix with demand IDs as row index and supply IDs
        as column index.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        if path.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)

    if fmt == "wide":
        return df

    # Long → wide pivot
    if fmt == "long":
        matrix = df.pivot(index=origin_col, columns=dest_col, values=cost_col)
        matrix = matrix.fillna(np.inf)
        return matrix

    raise ValueError(f"Unsupported format: {fmt!r}. Use 'long' or 'wide'.")
