

## Libraries
import numpy as np
import pandas as pd

import geopandas as gpd
import matplotlib.pyplot as plt


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



def make_att_table(df, inner_ring, outer_rings, treated_col, suffix):
    att_col = f'att_{suffix}'
    se_col = f'se_att_{suffix}'

    temp_list = []

    temp_s = df[df[treated_col] == inner_ring]['treated_inner_ring']

    temp_df = pd.DataFrame({
        'ring': [inner_ring],
        att_col: [float(temp_s.mean())],
        se_col: [float(temp_s.sem())]
    })

    temp_list.append(temp_df)

    for ring in outer_rings:
        temp_s = df[df[treated_col] == ring]['treated_outer_rings']

        temp_df = pd.DataFrame({
            'ring': [ring],
            att_col: [float(temp_s.mean())],
            se_col: [float(temp_s.sem())]
        })

        temp_list.append(temp_df)

    return pd.concat(temp_list, ignore_index=True)

def plot_att_row(
    df,
    ring_name,
    true_effect_col,
    att_dict,
    figsize=(8, 5)
    ):

    # wybór wiersza
    row = df[df["ring"] == ring_name].iloc[0]

    # true effect
    true_effect = row[true_effect_col]

    models = list(att_dict.keys())
    x = np.arange(len(models))

    atts = []
    lower = []
    upper = []

    for model_name, (att_col, se_col) in att_dict.items():
        att = row[att_col]
        se = row[se_col]

        ci_low = att - 1.96 * se
        ci_high = att + 1.96 * se

        atts.append(att)
        lower.append(ci_low)
        upper.append(ci_high)

    plt.figure(figsize=figsize)

    # confidence intervals
    for i in range(len(models)):
        plt.vlines(
            x=i,
            ymin=lower[i],
            ymax=upper[i],
            color="black",
            linewidth=1
        )

    # point estimates
    plt.scatter(
        x,
        atts,
        color="black",
        zorder=3
    )

    # true effect line
    plt.axhline(
        y=true_effect,
        color="red",
        linestyle="--",
        linewidth=1.5
    )
    # zero line
    plt.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=1
    )
    
    plt.xticks(x, models, rotation=90)
    plt.ylabel("ATT")
    plt.title(ring_name)

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_att_by_param(
    df,
    param_col,
    ring_value,
    models_dict,
    true_effect_col="true_effect"
):

    d = df[df["ring"] == ring_value].copy()
    d = d.sort_values(param_col)

    x = d[param_col].values

    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models_dict)))

    # =========================
    # MODELS
    # =========================
    for i, (model_name, (att_col, se_col)) in enumerate(models_dict.items()):

        att = d[att_col].values
        se = d[se_col].values

        color = colors[i]

        # central ATT line
        plt.plot(
            x, att,
            label=model_name,
            color=color,
            linewidth=2.5
        )

        # confidence band (fill)
        plt.fill_between(
            x,
            att - se,
            att + se,
            color=color,
            alpha=0.1
        )

        # SE boundaries (dotted)
        plt.plot(
            x, att + se,
            color=color,
            linestyle=":",
            linewidth=1,
            alpha=0.7
        )

        plt.plot(
            x, att - se,
            color=color,
            linestyle=":",
            linewidth=1,
            alpha=0.7
        )

    # =========================
    # TRUE EFFECT
    # =========================
    if true_effect_col in d.columns:
        plt.plot(
            x,
            d[true_effect_col],
            "r--",
            linewidth=2,
            label="True effect"
        )

    # =========================
    # ZERO LINE
    # =========================
    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    # =========================
    # STYLE
    # =========================
    plt.xlabel(param_col)
    plt.ylabel("ATT")
    plt.title(f"ATT vs {param_col} | {ring_value}")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()

def plot_rmse_att(
    df,
    parameter_col,
    att_true_col,
    model_cols,
    title_arg,
    y_axis_title,
    rotate_x_labels=False
):

    rmse_df = (
        df.groupby(parameter_col)
        .apply(
            lambda g: pd.Series({
                f"rmse_{col}": np.sqrt(
                    np.mean(
                        ((g[col] - g[att_true_col]) / g[att_true_col]) ** 2
                    )
                )
                for col in model_cols
            })
        )
        .reset_index()
    )

    # Plot
    plt.figure(figsize=(9, 5))

    for col in model_cols:
        plt.plot(
            rmse_df[parameter_col],
            rmse_df[f"rmse_{col}"],
            marker="o",
            label=col
        )

    plt.xlabel(parameter_col)
    plt.ylabel(y_axis_title)
    plt.title(title_arg)

    if rotate_x_labels:
        plt.xticks(rotation=90)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()



def compute_significance_share(
    df,
    parameter_col,
    model_dict
):


    df_flagged = df.copy()

    se_cols_to_drop = []

    # =========================================================
    # 1. Create flags + mask ATT if not significant
    # =========================================================
    for model_name, (att_col, se_col) in model_dict.items():

        flag_col = f"{model_name}_is_sig"

        is_sig = (df_flagged[att_col] - df_flagged[se_col]) > 0

        df_flagged[flag_col] = is_sig

        # mask ATT if not significant
        df_flagged.loc[~is_sig, att_col] = np.nan

        se_cols_to_drop.append(se_col)

    # drop SE columns
    df_flagged = df_flagged.drop(columns=se_cols_to_drop)

    # =========================================================
    # 2. Aggregate shares
    # =========================================================
    results = []

    grouped = df_flagged.groupby(parameter_col)

    for param_value, g in grouped:

        row = {
            parameter_col: param_value,
            "n": len(g)
        }

        for model_name in model_dict.keys():
            flag_col = f"{model_name}_is_sig"
            row[f"{model_name}_sig_share"] = g[flag_col].mean()

        results.append(row)

    df_sig = (
        pd.DataFrame(results)
        .sort_values(parameter_col)
        .reset_index(drop=True)
    )

    return df_sig, df_flagged