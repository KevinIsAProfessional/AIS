#!/usr/bin/env python3
import ee
ee.Initialize()

import os
import sys
import configparser




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
    
# Spatially thin presence/absence data

# Reduce regions from existing images

def main():
    config = configparser.ConfigParser()
    config.read('aisconfig.ini')

    start_year = int(config['WHENWHERE']['START_YEAR'])
    end_year = int(config['WHENWHERE']['END_YEAR'])
    covariate_folder = config["GEEPATHS"]["ASSETID"]

    years = range(start_year, end_year)

    covariates_exist(years, covariate_folder)
    print("I'm not broken, hooray!")

if __name__ == "__main__":
    main()
