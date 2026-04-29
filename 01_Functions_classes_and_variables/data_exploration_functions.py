

## Libraries
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
import tempfile
import webbrowser

def plot_gdf_points(
    gdf,
    size_col=None,
    color_col=None,
    size_scale=50,
    cmap="viridis",
    offset_ratio=0.05,
    alpha=0.7,
    edgecolor="k",
    linewidth=0.3,
    normalize_size=False,
    log_size=False,
    add_colorbar=True,
    figsize=(8, 8),
    ax=None,
    title=None
):


    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x = gdf.geometry.x
    y = gdf.geometry.y

    if size_col is not None:
        sizes = gdf[size_col].astype(float)

        if log_size:
            sizes = np.log1p(sizes)

        if normalize_size:
            sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min())

        sizes = sizes * size_scale
    else:
        sizes = size_scale

    if color_col is not None:
        colors = gdf[color_col]
    else:
        colors = "blue"

    sc = ax.scatter(
        x, y,
        s=sizes,
        c=colors,
        cmap=cmap if color_col is not None else None,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth
    )

    if color_col is not None and add_colorbar:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(color_col)

    xmin, ymin, xmax, ymax = gdf.total_bounds
    x_range = xmax - xmin
    y_range = ymax - ymin

    ax.set_xlim(xmin - offset_ratio * x_range,
                xmax + offset_ratio * x_range)

    ax.set_ylim(ymin - offset_ratio * y_range,
                ymax + offset_ratio * y_range)

    ax.set_aspect("equal")

    if title:
        ax.set_title(title)

    return ax
    
def show_gdf_folium(gdf, fields=['block_id'], zoom_start=13):
    if gdf.empty:
        print("GeoDataFrame is empty.")
        return
    gdf_4326 = gdf.to_crs(epsg=4326)
    m = folium.Map(location=[0, 0], zoom_start=zoom_start)
    folium.GeoJson(
        gdf_4326,
        name="GeoDataFrame",
        tooltip=folium.GeoJsonTooltip(fields=fields,
                                      aliases=[f.replace("_", " ").title() for f in fields])
    ).add_to(m)
    folium.LayerControl().add_to(m)
    tmp_file = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
    m.save(tmp_file.name)
    webbrowser.open(tmp_file.name)
    return m