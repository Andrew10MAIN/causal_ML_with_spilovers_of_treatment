def main():
    import sys
    from pathlib import Path
    import sqlite3
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    
    BASE_DIR = Path(__file__).resolve().parent.parent  
    sys.path.append(str(BASE_DIR))

    from src.data.simulation import generate_dict_of_combs
    #from src.data.exploration import *
    from src.data.operation import import_non_spatial_data_frame
    from src.models.modelling import make_treatment_effects_df, process_single_key
    from src.utils.config import  base_pairs, third_values, replacing_dict_0_ring, replacing_dict_odr_ring, load_config

    from joblib import Parallel, delayed




    config = load_config()
    np.random.seed(config["experiment"]["seed"])
    db_path = BASE_DIR / config["data"]["output_path1"] / config["data"]["output_path2"] / config["data"]["output_path3"]
    
    #connection_link_var = str(db_path)


    dict_of_combs = generate_dict_of_combs(base_pairs, third_values)

    dict_of_gdfs = {}

    conn = sqlite3.connect(db_path)

    for name in dict_of_combs:

        df = import_non_spatial_data_frame(
            conn=conn,
            df_to_return_name=name
        )

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["x"], df["y"]),
            crs="EPSG:3857"
        )

        dict_of_gdfs[name] = gdf

    conn.close()

    list_of_output_dfs = Parallel(
        n_jobs=-1,
        backend="loky",
        verbose=10
    )(
        delayed(process_single_key)(i, replacing_dict_0_ring, replacing_dict_odr_ring, dict_of_gdfs, dict_of_combs)
        for i in dict_of_gdfs.keys()
    )

    df_results = pd.concat(list_of_output_dfs, ignore_index=True)

    df_results.to_csv(BASE_DIR / config["results"]["results_path1"] / config["results"]["results_path2"]/ config["results"]["results_path3"] )


if __name__ == "__main__":
    main()