"""
Visualization helpers — static (matplotlib) and interactive (Folium) maps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import folium
from folium.plugins import MarkerCluster


def plot_accessibility(
    gdf: gpd.GeoDataFrame,
    column: str = "accessibility",
    *,
    title: str = "Spatial Accessibility Index",
    cmap: str = "RdYlGn",
    figsize: tuple[int, int] = (12, 8),
    legend: bool = True,
    supply_gdf: gpd.GeoDataFrame | None = None,
    supply_marker: str = "^",
    supply_color: str = "blue",
    ax: plt.Axes | None = None,
    **plot_kwargs: Any,
) -> Figure:
    """Create a static choropleth-style map of accessibility scores.

    Parameters
    ----------
    gdf : GeoDataFrame
        Demand locations with an accessibility score column.
    column : str
        Name of the column containing the accessibility index.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    figsize : tuple
        Figure size in inches.
    legend : bool
        Whether to include a colour-bar legend.
    supply_gdf : GeoDataFrame, optional
        If provided, supply locations are overlaid as markers.
    supply_marker, supply_color
        Marker style for supply locations.
    ax : Axes, optional
        Existing matplotlib axes to draw on.
    **plot_kwargs
        Forwarded to ``GeoDataFrame.plot()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    gdf.plot(
        column=column,
        cmap=cmap,
        legend=legend,
        ax=ax,
        markersize=60,
        edgecolor="black",
        linewidth=0.3,
        **plot_kwargs,
    )

    if supply_gdf is not None:
        supply_gdf.plot(
            ax=ax,
            marker=supply_marker,
            color=supply_color,
            markersize=80,
            edgecolor="black",
            linewidth=0.5,
            label="Supply",
        )
        ax.legend()

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def interactive_map(
    gdf: gpd.GeoDataFrame,
    column: str = "accessibility",
    *,
    supply_gdf: gpd.GeoDataFrame | None = None,
    supply_label_col: str | None = None,
    demand_label_col: str | None = None,
    tiles: str = "cartodbpositron",
    zoom_start: int = 11,
    cmap_name: str = "RdYlGn",
    radius: int = 10,
) -> folium.Map:
    """Create an interactive Folium map of accessibility scores.

    Parameters
    ----------
    gdf : GeoDataFrame
        Demand locations with accessibility scores.  Must be in
        EPSG:4326 (or will be reprojected).
    column : str
        Column name for accessibility values.
    supply_gdf : GeoDataFrame, optional
        Supply locations to overlay.
    supply_label_col : str, optional
        Column in *supply_gdf* to use for marker popups.
    demand_label_col : str, optional
        Column in *gdf* to use for circle popups.
    tiles : str
        Folium tileset name.
    zoom_start : int
        Initial zoom level.
    cmap_name : str
        Matplotlib colourmap used to colour circles.
    radius : int
        Circle marker radius in pixels.

    Returns
    -------
    folium.Map
    """
    # Ensure WGS 84
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    if supply_gdf is not None and supply_gdf.crs and supply_gdf.crs.to_epsg() != 4326:
        supply_gdf = supply_gdf.to_crs(epsg=4326)

    # Centre on demand centroid
    center = [gdf.geometry.y.mean(), gdf.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=zoom_start, tiles=tiles)

    # Colour normalisation
    cmap = plt.colormaps[cmap_name]
    values = gdf[column]
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        vmax = vmin + 1  # avoid division by zero

    for idx, row in gdf.iterrows():
        normed = (row[column] - vmin) / (vmax - vmin)
        rgba = cmap(normed)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        )
        popup_text = f"ID: {idx}<br>{column}: {row[column]:.4f}"
        if demand_label_col and demand_label_col in gdf.columns:
            popup_text = f"{row[demand_label_col]}<br>" + popup_text
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=radius,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.8,
            popup=popup_text,
        ).add_to(m)

    # Supply markers
    if supply_gdf is not None:
        fg = folium.FeatureGroup(name="Supply Locations")
        for idx, row in supply_gdf.iterrows():
            label = str(idx)
            if supply_label_col and supply_label_col in supply_gdf.columns:
                label = str(row[supply_label_col])
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=label,
                icon=folium.Icon(color="blue", icon="plus", prefix="fa"),
            ).add_to(fg)
        fg.add_to(m)
        folium.LayerControl().add_to(m)

    return m
