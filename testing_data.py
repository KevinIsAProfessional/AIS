#!/usr/bin/env python3

import ee
ee.Initialize()

import pandas as pd
import configparser

def main():
    config = configparser.ConfigParser()
    config.read('aisconfig.ini')

    
    state_abbrev = config["WHENWHERE"]["STATE_ABBREVIATION"]
    start_year = int(config['WHENWHERE']['START_YEAR'])
    end_year = int(config['WHENWHERE']['END_YEAR'])
    covariate_folder = config["GEEPATHS"]["ASSETID"]
    output_path = config["LOCALPATHS"]["TESTING_DATA"] 

    HUC_clip = ee.FeatureCollection("USGS/WBD/2017/HUC12").filter(ee.Filter.eq('states',state_abbrev))
    years = range(start_year, end_year)
    images = list(map(lambda x: ee.Image(covariate_folder + "covariates" + str(x)), years))



    mean_image = ee.Image(ee.ImageCollection.fromImages(images).mean())

    huc_level_covariates = mean_image.reduceRegions(**{
        'collection': HUC_clip,
        'reducer': ee.Reducer.mean(),
        'crs': 'EPSG:4326',
        'scale': 100,
        'tileScale': 16})

    pd.DataFrame([x['properties'] for x in huc_level_covariates.getInfo()['features']]).to_csv(output_path)

    print(".csvs written to ", output_path)


if __name__ == "__main__":
    main()
