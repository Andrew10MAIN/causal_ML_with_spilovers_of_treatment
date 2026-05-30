def main():
    
    import os

    import sys
    from pathlib import Path
    import sqlite3
    import numpy as np


    BASE_DIR = Path(__file__).resolve().parent.parent  
    sys.path.append(str(BASE_DIR))


    from src.data.simulation import generate_dict_of_combs, return_spatial_geo_df
    from src.utils.config import  base_pairs, third_values, load_config
    
    config = load_config()
    #db_path = config["data"]["output_path"]
    # db_path = Path(config["data"]["output_path"])
    # db_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(config["experiment"]["seed"])

    dict_of_combs = generate_dict_of_combs(base_pairs, third_values)

    dict_of_output_gdf = {}
    for i in dict_of_combs.keys():
        single_gdf_output = return_spatial_geo_df(

            n_x = config["simulation"]["n_x"],  
            n_y = config["simulation"]["n_y"],    
            spacing = 100, 

            treated_scope_x_start = dict_of_combs[i][0],
            treated_scope_x_end = dict_of_combs[i][1],
            treated_scope_y_start = config["simulation"]["treated_scope_y_start"],
            treated_scope_y_end = config["simulation"]["treated_scope_y_end"],
            treated_last_row_length = None,
        
            ATT_target = config["simulation"]["ATT_target"],
            y_ns_to_att_ratio = dict_of_combs[i][2],

            y_spatial_autocorelation_scope_x_start = 2,
            y_spatial_autocorelation_scope_x_end = 10,
            y_spatial_autocorelation_scope_y_start = 2,
            y_spatial_autocorelation_scope_y_end = 10,

            rho = config["simulation"]["rho"],

            spatial_confounder_scope_x_start = 0,
            spatial_confounder_scope_x_end = 0,
            spatial_confounder_scope_y_start = 0,
            spatial_confounder_scope_y_end = 0,
            

            lambda_cs = 0.0,
            max_treatment_spillover_distance = config["spillovers"]["max_distance"],
            understimated_treatment_spillover_distance=None,
            overestimated_treatment_spillover_distance=None,
            
            logistic_distance_decay = config["spillovers"]["logistic_decay"],

            nonspatial_confounders_contribution_to_Y=1.0,
            spatial_confounder_contribution_to_Y=0.7,
            
            epsilon_distribution_mean = 0.025,
            epsilon_distribution_standard_error = 0.2)
        
        single_gdf_output2 = single_gdf_output.drop(columns = ['tau_base', 
                                                            # 'spill','T_tot',
                            'distance_to_treatment',
                            'decay','C3',
                            'Cs','geometry',
                            'propensity',
                            #'T_tot_cat_underestim'
                            ]).copy()
        dict_of_output_gdf[i] = single_gdf_output2



    db_path = BASE_DIR / config["data"]["output_path1"] / config["data"]["output_path2"] / config["data"]["output_path3"]

    conn = sqlite3.connect(db_path)

    for k in dict_of_output_gdf.keys():
        dict_of_output_gdf[k].to_sql(
            k,
            conn,
            if_exists="replace",
            index=False
        )

    conn.close()


if __name__ == "__main__":
    main()