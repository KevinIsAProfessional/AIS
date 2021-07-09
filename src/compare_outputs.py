# compare_outputs.py

import pandas as pd
# import geopandas as gpd

def compare_outputs(old_path, new_path): 
    
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)

    cols = ['huc12',
            'prediction_ensemble',
            'prediction_proba',
            'prediction_uncertainty',
            'MESS']

    old = old[cols]
    new = new[cols]

    print((old == new).mean())

