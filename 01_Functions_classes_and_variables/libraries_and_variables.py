

## Libraries
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap, to_hex

## Variables
### Colors
prez_blue  = (2/255, 87/255, 163/255, 1.0)
prez_blue2 = (2/255, 87/255, 163/255, 0.69)
prez_blue3 = (2/255, 87/255, 163/255, 0.35)
prez_red   = (227/255, 26/255, 28/255, 1.0)
prez_red2  = (227/255, 26/255, 28/255, 0.69)
prez_red3  = (227/255, 26/255, 28/255, 0.35)
prez_yellow = (224/255, 184/255, 24/255, 1.0)
prez_grey  = (210/255, 214/255, 216/255, 1.0)
prez_green = (27/255, 158/255, 119/255, 1.0)
prez_purple = (117/255, 107/255, 177/255, 1.0)
prez_brown  = (140/255, 81/255, 10/255, 1.0)
prez_white = (1, 1, 1)
prez_black = "#000000"

base_rgb_blue = prez_blue[:3]
cmap_custom_blue = LinearSegmentedColormap.from_list(
    'custom_blue',
    [(1, 1, 1), base_rgb_blue]
)


base_rgb_red = prez_red[:3]
cmap_custom_red = LinearSegmentedColormap.from_list(
    'custom_red',
    [(1, 1, 1), base_rgb_red]
)

base_rgb_yellow = (224/255, 184/255, 24/255)
cmap_custom_yellow = LinearSegmentedColormap.from_list(
    'custom_yellow', 
    [(1,1,1), base_rgb_yellow]  
)


base_rgb_grey = prez_grey[:3]
cmap_custom_grey = LinearSegmentedColormap.from_list(
    'custom_grey',
    [(1, 1, 1), base_rgb_grey]
)

cmap_div_white = LinearSegmentedColormap.from_list(
    "blue_white_red",
    [
        (0.0, (0/255, 70/255, 180/255)),
        (0.5, (1, 1, 1)),
        (1.0, (180/255, 0/255, 0/255))
    ]
)
