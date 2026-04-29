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

    y_start, y_end, y_step = (treated_scope_x_start)*spacing, (treated_scope_x_end)*spacing, spacing
    x_start, x_end, x_step = (treated_scope_y_start)*spacing, (treated_scope_y_end)*spacing, spacing

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