

## Libraries

import pandas as pd
import numpy as np
import geopandas as gpd

from itertools import product

from libpysal.weights import Queen, DistanceBand, higher_order
from libpysal.weights.spatial_lag import lag_spatial
import sqlite3
import shapely.wkb



def import_non_spatial_data_frame(
        connection_link: str,
        df_to_return_name: str
                                  ):
    conn = sqlite3.connect(connection_link)
    df_to_return = pd.read_sql(f"SELECT * FROM {df_to_return_name}", conn)
    #df_to_return['year'] = df_to_return['year'].values.astype('datetime64[M]')
    conn.close()
    return df_to_return


def import_spatial_point_data_frame(
        connection_link_lnx: str,
        layer_name: str,
        crs_param: int
                                  ):
    df = gpd.read_file(
    connection_link_lnx,
    layer=layer_name
        )
    # df["GEOMETRY"] = df["geometry"]
    # df["geometry"] = df["GEOMETRY"].apply(shapely.wkb.loads)
    # df = df.drop(columns = ['GEOMETRY'])
    gdf_to_return = gpd.GeoDataFrame(df, geometry="geometry")
    gdf_to_return  = gdf_to_return.set_crs(epsg=crs_param)
    return gdf_to_return


