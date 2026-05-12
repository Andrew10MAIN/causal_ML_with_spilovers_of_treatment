import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from libpysal.weights import DistanceBand
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

def generate_spillovers(coords, gdf, mean_tau_treated,
                        distance_ring_number,
                        spacing,
                        scalars,
                        noise_scale=0.1,
                        seed=None):


    if seed is not None:
        np.random.seed(seed)

    if len(scalars) != distance_ring_number:
        raise ValueError("Długość 'scalars' musi być równa distance_ring_number")

    treated_coords = coords[gdf["T"] == 1]
    tree = cKDTree(treated_coords)

    dist, _ = tree.query(coords, k=1)

    n = len(coords)
    spill = np.zeros(n)

    for i in range(distance_ring_number):
        lower = i * spacing
        upper = (i + 1) * spacing

        ring_mask = (dist > lower) & (dist <= upper)

        spill[ring_mask] = np.random.normal(
            loc=scalars[i] * mean_tau_treated,
            scale=noise_scale,
            size=ring_mask.sum()
        )

    return spill


def return_spatial_geo_df(
    n_x: int,
    n_y: int,
    spacing: int,

    treated_scope_x_start: int,
    treated_scope_x_end: int,
    treated_scope_y_start: int,
    treated_scope_y_end: int,

    ATT_target: float,

    y_spatial_autocorelation_scope_x_start: int,
    y_spatial_autocorelation_scope_x_end: int,
    y_spatial_autocorelation_scope_y_start: int,
    y_spatial_autocorelation_scope_y_end: int,

    rho: float,

    spatial_confounder_scope_x_start: int,
    spatial_confounder_scope_x_end: int,
    spatial_confounder_scope_y_start: int,
    spatial_confounder_scope_y_end: int,

    lambda_cs: float,

    max_treatment_spillover_distance: float,
    logistic_distance_decay: bool,


    understimated_treatment_spillover_distance: float = None,
    overestimated_treatment_spillover_distance: float = None,

    nonspatial_confounders_contribution_to_Y: float = 1.0,
    spatial_confounder_contribution_to_Y: float = 1.0,

    epsilon_distribution_mean: float = 0.0,
    epsilon_distribution_standard_error: float = 1.0,
):

    # =========================================================
    # VALIDATION
    # =========================================================
    if understimated_treatment_spillover_distance is not None:
        if understimated_treatment_spillover_distance > max_treatment_spillover_distance:
            raise ValueError(
                "understimated_treatment_spillover_distance "
                "cannot be greater than "
                "max_treatment_spillover_distance"
            )

    if overestimated_treatment_spillover_distance is not None:
        if overestimated_treatment_spillover_distance < max_treatment_spillover_distance:
            raise ValueError(
                "overestimated_treatment_spillover_distance "
                "cannot be smaller than "
                "max_treatment_spillover_distance"
            )


    # =========================================================
    # 0. GRID
    # =========================================================
    points = [
        Point(i * spacing, j * spacing)
        for i in range(n_x)
        for j in range(n_y)
    ]

    gdf = gpd.GeoDataFrame(
        geometry=points,
        crs="EPSG:3857"
    ).reset_index()

    gdf = gdf.rename(columns={"index": "unit_id"})
    gdf["unit_id"] += 1000

    gdf["x"] = gdf.geometry.x.astype(int)
    gdf["y"] = gdf.geometry.y.astype(int)

    n = len(gdf)

    coords = np.array(
        list(zip(gdf["x"], gdf["y"]))
    )

    # =========================================================
    # 1. SPATIAL WEIGHTS MATRIX
    # =========================================================
    W = DistanceBand(
        coords,
        threshold=spacing + 1,
        binary=True,
        silence_warnings=True
    )

    W.transform = "R"

    W_sparse = W.sparse

    # =========================================================
    # 2. INNER TREATMENT RING (T)
    # =========================================================
    x_vals = np.arange(
        treated_scope_x_start * spacing,
        (treated_scope_x_end + 1) * spacing,
        spacing
    )

    y_vals = np.arange(
        treated_scope_y_start * spacing,
        (treated_scope_y_end + 1) * spacing,
        spacing
    )

    gdf["T"] = (
        gdf["x"].isin(x_vals) &
        gdf["y"].isin(y_vals)
    ).astype(int)

    mask_T_inner = gdf["T"] == 1

    # =========================================================
    # 3. NON-SPATIAL CONFOUNDERS
    # =========================================================
    C1 = np.random.normal(0, 1, n)
    C2 = np.random.normal(0, 1, n)
    C3 = np.random.normal(0, 1, n)

    fC = (
        0.6 * C1 +
        0.5 * C2 +
        0.4 * C3 +
        0.1 * C1**2 -
        0.1 * C2 * C3
    )

    # =========================================================
    # 4. SPATIAL CONFOUNDER
    # =========================================================
    eta = np.random.normal(0, 1, n)

    mask_cs = (
        gdf["x"].between(
            spatial_confounder_scope_x_start * spacing,
            spatial_confounder_scope_x_end * spacing
        ) &
        gdf["y"].between(
            spatial_confounder_scope_y_start * spacing,
            spatial_confounder_scope_y_end * spacing
        )
    )

    Cs = eta.copy()

    for _ in range(5):
        Cs = lambda_cs * W_sparse.dot(Cs) + eta

    Cs[~mask_cs] = eta[~mask_cs]

    # =========================================================
    # 5. PROPENSITY
    # =========================================================
    gdf["propensity"] = (
        1.2 * fC +
        0.8 * Cs +
        np.random.normal(0, 0.5, n)
    )

    # =========================================================
    # 6. DISTANCE TO TREATMENT
    # =========================================================
    treated_coords = coords[mask_T_inner]

    tree = cKDTree(treated_coords)

    distances, _ = tree.query(coords)

    # =========================================================
    # 7. TOTAL TREATMENT AREA (T_tot)
    # =========================================================
    mask_outer_ring = (
        (distances > 0) &
        (distances <= max_treatment_spillover_distance)
    )

    gdf["T_tot"] = (
        mask_T_inner |
        mask_outer_ring
    ).astype(int)

    # =========================================================
    # 8. DECAY FUNCTION
    # =========================================================
    decay = np.zeros(n)

    # inner ring
    decay[mask_T_inner] = 1.0

    # outer ring
    mask_outer_only = (
        mask_outer_ring &
        (~mask_T_inner)
    )

    if logistic_distance_decay:

        x = (
            distances[mask_outer_only] /
            max_treatment_spillover_distance
        )

        decay_outer = 1 / (
            1 + np.exp(10 * (x - 0.5))
        )

        decay[mask_outer_only] = decay_outer

    else:
        # linear decay
        decay_outer = (
            1 -
            distances[mask_outer_only] /
            max_treatment_spillover_distance
        )

        decay[mask_outer_only] = decay_outer

    decay = np.clip(decay, 0, 1)

    # =========================================================
    # 9. ITE
    # =========================================================
    tau_base = np.zeros(n)

    mask_total_treated = gdf["T_tot"] == 1

    tau_base[mask_total_treated] = np.random.normal(
        loc=(
            1 +
            0.3 * C1[mask_total_treated] +
            0.2 * C2[mask_total_treated]
        ),
        scale=0.2
    )

    # ATT calibration ONLY for inner ring
    tau_base[mask_T_inner] += (
        ATT_target -
        tau_base[mask_T_inner].mean()
    )

    # final tau
    tau = np.zeros(n)

    # inner ring -> full treatment
    tau[mask_T_inner] = tau_base[mask_T_inner]

    # outer ring -> decayed treatment
    tau[mask_outer_only] = (
        tau_base[mask_outer_only] *
        decay[mask_outer_only]
    )

    # =========================================================
    # 10. SPILLOVER MULTIPLIER
    # =========================================================
    spill = np.zeros(n)

    spill[mask_total_treated] = decay[mask_total_treated]

    # =========================================================
    # 11. BASE Y
    # =========================================================
    epsilon = np.random.normal(
        epsilon_distribution_mean,
        epsilon_distribution_standard_error,
        n
    )

    Y_base = (
        nonspatial_confounders_contribution_to_Y * fC +
        spatial_confounder_contribution_to_Y * Cs +
        tau +
        epsilon
    )

    # =========================================================
    # 12. SPATIAL Y
    # =========================================================
    mask_y = (
        gdf["x"].between(
            y_spatial_autocorelation_scope_x_start * spacing,
            y_spatial_autocorelation_scope_x_end * spacing
        ) &
        gdf["y"].between(
            y_spatial_autocorelation_scope_y_start * spacing,
            y_spatial_autocorelation_scope_y_end * spacing
        )
    )

    W2 = W_sparse.copy().tolil()

    for i in range(n):

        if not mask_y.iloc[i]:
            W2[i, :] = 0

    W2 = W2.tocsr()

    Y = spsolve(
        identity(n) - rho * W2,
        Y_base
    )

    # =========================================================
    # 13. HELPER FUNCTION FOR CATEGORICAL RINGS
    # =========================================================
    def create_ring_categories(
        distances_array,
        inner_mask,
        max_distance,
        spacing_value
    ):

        categories = np.full(
            len(distances_array),
            "control",
            dtype=object
        )

        categories[inner_mask] = "treated_inner_ring"

        n_rings = int(
            np.ceil(max_distance / spacing_value)
        )

        for ring in range(1, n_rings + 1):

            lower = (ring - 1) * spacing_value
            upper = ring * spacing_value

            mask_ring = (
                (~inner_mask) &
                (distances_array > lower) &
                (distances_array <= upper) &
                (distances_array <= max_distance)
            )

            categories[mask_ring] = (
                f"treated_outer_ring{ring}"
            )

        return categories

    # =========================================================
    # 14. T_tot_cat
    # =========================================================
    gdf["T_tot_cat"] = create_ring_categories(
        distances_array=distances,
        inner_mask=mask_T_inner.values,
        max_distance=max_treatment_spillover_distance,
        spacing_value=spacing
    )

    # =========================================================
    # 15. UNDER / OVER ESTIMATED CATEGORIES
    # =========================================================
    if understimated_treatment_spillover_distance is not None:

        gdf["T_tot_cat_underestim"] = (
            create_ring_categories(
                distances_array=distances,
                inner_mask=mask_T_inner.values,
                max_distance=understimated_treatment_spillover_distance,
                spacing_value=spacing
            )
        )

    if overestimated_treatment_spillover_distance is not None:

        gdf["T_tot_cat_overerestim"] = (
            create_ring_categories(
                distances_array=distances,
                inner_mask=mask_T_inner.values,
                max_distance=overestimated_treatment_spillover_distance,
                spacing_value=spacing
            )
        )

    # =========================================================
    # 16. ODR VARIABLES
    # =========================================================
    if overestimated_treatment_spillover_distance is not None:

        odr_max_distance = (
            overestimated_treatment_spillover_distance
        )

    else:

        odr_max_distance = (
            max_treatment_spillover_distance
        )

    n_odr = int(
        np.ceil(odr_max_distance / spacing)
    )

    for ring in range(1, n_odr + 1):

        lower = (ring - 1) * spacing
        upper = ring * spacing

        gdf[f"ODR_{ring}"] = (
            (
                (~mask_T_inner) &
                (distances > lower) &
                (distances <= upper) &
                (distances <= odr_max_distance)
            )
        ).astype(int)

    # =========================================================
    # 17. OUTPUT
    # =========================================================
    gdf["C1"] = C1
    gdf["C2"] = C2
    gdf["C3"] = C3

    gdf["Cs"] = Cs

    gdf["tau"] = tau
    gdf["tau_base"] = tau_base

    gdf["spill"] = spill
    gdf["distance_to_treatment"] = distances
    gdf["decay"] = decay

    gdf["Y"] = Y
    gdf["Y_ns"] = Y_base

    return gdf