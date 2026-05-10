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

def return_spatial_geo_df(n_x: int,  
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
    distance_ring_number: int,
    distance_ring_ate_scalars: list,
    nonspatial_confounders_contribution_to_Y: float,
    spatial_confounder_contribution_to_Y: float,
    epsilon_distribution_mean: float,
    epsilon_distribution_standard_error: float):


    spatial_confounder_scope_x_start_spacing = spatial_confounder_scope_x_start * spacing
    spatial_confounder_scope_x_end_spacing = spatial_confounder_scope_x_end * spacing
    spatial_confounder_scope_y_start_spacing = spatial_confounder_scope_y_start * spacing
    spatial_confounder_scope_y_end_spacing = spatial_confounder_scope_y_end * spacing

    y_spatial_autocorelation_scope_x_start_spacing = y_spatial_autocorelation_scope_x_start * spacing
    y_spatial_autocorelation_scope_x_end_spacing = y_spatial_autocorelation_scope_x_end * spacing
    y_spatial_autocorelation_scope_y_start_spacing = y_spatial_autocorelation_scope_y_start * spacing
    y_spatial_autocorelation_scope_y_end_spacing = y_spatial_autocorelation_scope_y_end * spacing

    y_start, y_end, y_step = (treated_scope_y_start)*spacing, (treated_scope_y_end)*spacing, spacing
    x_start, x_end, x_step = (treated_scope_x_start)*spacing, (treated_scope_x_end)*spacing, spacing

    y_vals = np.arange(y_start, y_end + y_step, y_step)
    x_vals = np.arange(x_start, x_end + x_step, x_step)



    origin_x, origin_y = 0, 0

    points = []

    for i in range(n_x):
        for j in range(n_y):
            x = origin_x + i * spacing
            y = origin_y + j * spacing
            points.append(Point(x, y))

    gdf = gpd.GeoDataFrame(geometry=points)

    gdf.set_crs(epsg=3857, inplace=True)
    gdf2 = gdf.reset_index() 
    gdf2 = gdf2.rename(columns = {'index': 'unit_id'})
    gdf2['unit_id'] = 1000 + gdf2['unit_id']
    gdf2["x"] = gdf2.geometry.x.astype(int)
    gdf2["y"] = gdf2.geometry.y.astype(int)
    gdf = gdf2.copy()
    n = len(gdf)

    coords = np.array(list(zip(gdf["x"], gdf["y"])))

    # -------------------------
    # 1. DITANCE MATRIX (100m)
    # -------------------------
    W = DistanceBand(coords, threshold=101, binary=True, silence_warnings=True)
    W.transform = "R"  # row-standardization

    W_sparse = W.sparse

    # -------------------------
    # 2. TREATMENT (PREDEFINED)
    # -------------------------
    treated_mask = (
        gdf['y'].isin(y_vals) &
        gdf['x'].isin(x_vals)
    )

    gdf["T"] = treated_mask.astype(int)

    # -------------------------
    # 3. NONSPATIAL CONFOUNDERS 
    # -------------------------
    C1 = np.random.normal(0,1,n)
    C2 = np.random.normal(0,1,n)
    C3 = np.random.normal(0,1,n)


    fC = (
        0.6*C1 + 0.5*C2 + 0.4*C3 +
        0.1*C1**2 - 0.1*C2*C3
    )

    # -------------------------
    # 4. SPATIAL CONFOUNDER 
    # -------------------------
    eta = np.random.normal(0,1,n)


    mask_cs = (
        (gdf["y"].between(spatial_confounder_scope_y_start_spacing,spatial_confounder_scope_y_end_spacing)) &
        (gdf["x"].between(spatial_confounder_scope_x_start_spacing,spatial_confounder_scope_x_end_spacing))
    )  

    Cs = eta.copy()
    for _ in range(5):
        Cs = lambda_cs * W_sparse.dot(Cs) + eta

    Cs[~mask_cs] = eta[~mask_cs]

    # -------------------------
    # 5. "PROPENSITY" 
    # -------------------------
    T_star = 1.2*fC + 0.8*Cs + np.random.normal(0,0.5,n)

    gdf["propensity"] = T_star

    # -------------------------
    # 6. ITE (only for treated)
    # -------------------------
    tau = np.zeros(n)

    # tau[gdf["T"]==1] = np.random.normal(
    #     loc = 1 + 0.3*C1[gdf["T"]==1] + 0.2*C2[gdf["T"]==1],
    #     scale = 0.2
    # )
    tau[gdf["T"]==1] = np.random.normal(
        loc = 1 + 0.3*C1[gdf["T"]==1] + 0.2*C2[gdf["T"]==1],
        scale = 0.2
    )

    mean_tau_treated = tau[gdf["T"]==1].mean()

    tau[gdf["T"]==1] += (ATT_target - mean_tau_treated)

    mean_tau_treated = tau[gdf["T"]==1].mean()


    # -------------------------
    # 7. SPILLOVERS (rings)
    # -------------------------


    spill = generate_spillovers(
        coords=coords,
        gdf=gdf,
        mean_tau_treated=mean_tau_treated,
        distance_ring_number=distance_ring_number,
        spacing=spacing,
        scalars=distance_ring_ate_scalars,
        seed=42
    )


    # -------------------------
    # 8. BASE Y (without spatial autocorrelation)
    # -------------------------
    epsilon = np.random.normal(epsilon_distribution_mean, epsilon_distribution_standard_error , n)

    Y_base = (
        nonspatial_confounders_contribution_to_Y*fC +
        spatial_confounder_contribution_to_Y*Cs +
        tau * gdf["T"] +
        spill +
        epsilon
    )

    # -------------------------
    # 9. SPATIAL AUTOCORRELATION OF Y (LOCAL)
    # -------------------------

    mask_y = (
        (gdf["y"].between(y_spatial_autocorelation_scope_y_start_spacing,y_spatial_autocorelation_scope_y_end_spacing)) &
        (gdf["x"].between(y_spatial_autocorelation_scope_x_start_spacing,y_spatial_autocorelation_scope_x_end_spacing))
    )

    W2 = W_sparse.copy().tolil()

    for i in range(n):
        if not mask_y.iloc[i]:
            W2[i,:] = 0

    W2 = W2.tocsr()

    I = identity(n)

    Y = spsolve(I - rho * W2, Y_base)


    gdf["C1"] = C1
    gdf["C2"] = C2
    gdf["C3"] = C3
    gdf["Cs"] = Cs

    gdf["tau"] = tau
    gdf["spill"] = spill
    gdf["Y"] = Y
    gdf["Y_ns"] = Y_base 
    return gdf


def return_spatial_geo_df2(
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

    # 🔴 NOWE PARAMETRY
    max_treatment_spillover_distance: float,
    logistic_distance_decay: bool,

    nonspatial_confounders_contribution_to_Y: float,
    spatial_confounder_contribution_to_Y: float,

    epsilon_distribution_mean: float,
    epsilon_distribution_standard_error: float,
):

    # =========================
    # 0. GRID
    # =========================
    points = [
        Point(i * spacing, j * spacing)
        for i in range(n_x)
        for j in range(n_y)
    ]

    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:3857").reset_index()
    gdf = gdf.rename(columns={"index": "unit_id"})
    gdf["unit_id"] += 1000
    gdf["x"] = gdf.geometry.x.astype(int)
    gdf["y"] = gdf.geometry.y.astype(int)

    n = len(gdf)
    coords = np.array(list(zip(gdf["x"], gdf["y"])))

    # =========================
    # 1. W
    # =========================
    W = DistanceBand(coords, threshold=spacing + 1, binary=True, silence_warnings=True)
    W.transform = "R"
    W_sparse = W.sparse

    # =========================
    # 2. TREATMENT
    # =========================
    x_vals = np.arange(treated_scope_x_start * spacing,
                       (treated_scope_x_end + 1) * spacing,
                       spacing)

    y_vals = np.arange(treated_scope_y_start * spacing,
                       (treated_scope_y_end + 1) * spacing,
                       spacing)

    gdf["T"] = (
        gdf["x"].isin(x_vals) &
        gdf["y"].isin(y_vals)
    ).astype(int)

    # =========================
    # 3. NON-SPATIAL C
    # =========================
    C1 = np.random.normal(0, 1, n)
    C2 = np.random.normal(0, 1, n)
    C3 = np.random.normal(0, 1, n)

    fC = (
        0.6 * C1 + 0.5 * C2 + 0.4 * C3 +
        0.1 * C1**2 - 0.1 * C2 * C3
    )

    # =========================
    # 4. SPATIAL C
    # =========================
    eta = np.random.normal(0, 1, n)

    mask_cs = (
        gdf["x"].between(spatial_confounder_scope_x_start * spacing,
                         spatial_confounder_scope_x_end * spacing) &
        gdf["y"].between(spatial_confounder_scope_y_start * spacing,
                         spatial_confounder_scope_y_end * spacing)
    )

    Cs = eta.copy()
    for _ in range(5):
        Cs = lambda_cs * W_sparse.dot(Cs) + eta

    Cs[~mask_cs] = eta[~mask_cs]

    # =========================
    # 5. PROPENSITY
    # =========================
    gdf["propensity"] = 1.2 * fC + 0.8 * Cs + np.random.normal(0, 0.5, n)

    # =========================
    # 6. ITE
    # =========================
    tau = np.zeros(n)

    mask_T = gdf["T"] == 1
    tau[mask_T] = np.random.normal(
        loc=1 + 0.3 * C1[mask_T] + 0.2 * C2[mask_T],
        scale=0.2
    )

    # dopasowanie ATT
    tau[mask_T] += (ATT_target - tau[mask_T].mean())
    mean_tau_treated = tau[mask_T].mean()

    # =========================
    # 7. SPILLOVER
    # =========================
    treated_coords = coords[mask_T]
    tree = cKDTree(treated_coords)
    distances, _ = tree.query(coords)

    spill = np.zeros(n)
    mask_control = ~mask_T

    if logistic_distance_decay:
        # ---- logistic ----
        x = distances / max_treatment_spillover_distance
        decay = 1 / (1 + np.exp(10 * (x - 0.5)))

        spill[mask_control] = mean_tau_treated * decay[mask_control]

    else:
        # ---- pseudo rings ----
        n_rings = int(max_treatment_spillover_distance / spacing)
        rings = np.ceil(distances / spacing).astype(int)

        for r in range(1, n_rings + 1):
            mask_r = (rings == r) & mask_control
            weight = (n_rings - r + 1) / (n_rings + 1)
            spill[mask_r] = mean_tau_treated * weight

    # cutoff
    spill[distances > max_treatment_spillover_distance] = 0

    # =========================
    # 8. Y base
    # =========================
    epsilon = np.random.normal(
        epsilon_distribution_mean,
        epsilon_distribution_standard_error,
        n
    )

    Y_base = (
        nonspatial_confounders_contribution_to_Y * fC +
        spatial_confounder_contribution_to_Y * Cs +
        tau * gdf["T"] +
        spill +
        epsilon
    )

    # =========================
    # 9. SPATIAL Y
    # =========================
    mask_y = (
        gdf["x"].between(y_spatial_autocorelation_scope_x_start * spacing,
                         y_spatial_autocorelation_scope_x_end * spacing) &
        gdf["y"].between(y_spatial_autocorelation_scope_y_start * spacing,
                         y_spatial_autocorelation_scope_y_end * spacing)
    )

    W2 = W_sparse.copy().tolil()

    for i in range(n):
        if not mask_y.iloc[i]:
            W2[i, :] = 0

    W2 = W2.tocsr()

    Y = spsolve(identity(n) - rho * W2, Y_base)

    # =========================
    # 10. OUTPUT
    # =========================
    gdf["C1"] = C1
    gdf["C2"] = C2
    gdf["C3"] = C3
    gdf["Cs"] = Cs
    gdf["tau"] = tau
    gdf["spill"] = spill
    gdf["Y"] = Y
    gdf["Y_ns"] = Y_base

    return gdf

def return_spatial_geo_df3(
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

    # spillover / external treatment ring
    max_treatment_spillover_distance: float,
    logistic_distance_decay: bool,

    nonspatial_confounders_contribution_to_Y: float,
    spatial_confounder_contribution_to_Y: float,

    epsilon_distribution_mean: float,
    epsilon_distribution_standard_error: float,
):

    # =========================================================
    # IMPORTS
    # =========================================================


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
    # 13. OUTPUT
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