#!/usr/bin/env python3
import ee
ee.Initialize()

import os
import sys
import configparser
import pandas
from src.gee_funs import reduce_HUCS, embed_date



# Check if covariate assets exist in GEE
def covariates_exist(years, assetId):
    command = 'earthengine ls ' + assetId
    result = os.popen(command).read().split('\n')
    assets = list(map(lambda path: path.split('/')[-1], result))
    expected_assets = list(map(lambda year: 'covariates' + str(year), years))

    all_accounted_for = True
    for item in expected_assets:
        if item not in assets:
            print("Expected asset " + item + " not found.")
            all_accounted_for = False
        else:
            print(item, " found")

    if all_accounted_for:
        print("All required covariates found.")
    else:
        print("Required covariate data missing. Please locate requred covariates and try again.")
        sys.exit(1)
    

def main():
    config = configparser.ConfigParser()
    config.read('aisconfig.ini')

    state_abbrev = config["WHENWHERE"]["STATE_ABBREVIATION"]
    start_year = int(config['WHENWHERE']['START_YEAR'])
    end_year = int(config['WHENWHERE']['END_YEAR'])
    covariate_folder = config["GEEPATHS"]["ASSETID"]
    thinned_asset_path = config["GEEPATHS"]["AIS_THINNED_POINT_PATH"]
    trainingdata = config["LOCALPATHS"]["TRAINING_DATA"]

    HUC_clip = ee.FeatureCollection("USGS/WBD/2017/HUC12").filter(ee.Filter.eq('states',state_abbrev))
    years = range(start_year, end_year)
    points_file = ee.FeatureCollection(thinned_asset_path).map(embed_date)

    covariates_exist(years, covariate_folder)

    images = list(map(lambda x: ee.Image(covariate_folder + "covariates" + str(x)), years))

    if not os.path.exists(trainingdata):
        print("WARNING: TRAINING_DATA variable in aisconfig.ini is set to a directory that does not exist.")
        print("         Please either create the direcory or specify a valid path in aisconfig.ini.")
        print("         Exiting")   
        sys.exit(1)

    for i in range(len(years)):
        current_year = start_year + i
        
        print("Starting ", current_year)
        
        data = reduce_HUCS(images[i], points_file, HUC_clip)

        my_csv = pandas.DataFrame([x['properties'] for x in data.getInfo()['features']])

        if my_csv.empty:
            print("WARNING: " + thinned_asset_path + " contains no data points for " + str(current_year))
            print("No .csv created")
        else:
            my_csv.to_csv((trainingdata) + str(current_year) + '.csv', index=False)
            print("Finished", current_year)

    print("All files exported to " + trainingdata)


if __name__ == "__main__":
    main()
