#!/usr/bin/env python3

import ee 
ee.Initialize()

import configparser
import time
from src.build_annual_cube import build_all_cubes 

def main():
    config = configparser.ConfigParser()
    config.read('aisconfig.ini')

    start_year = int(config['WHENWHERE']['START_YEAR'])
    end_year = int(config['WHENWHERE']['END_YEAR'])

    banded_images_list = build_all_cubes(start_year, end_year) 
    state_geometry = ee.FeatureCollection("TIGER/2016/States").filter(ee.Filter.eq('NAME',config["WHENWHERE"]["STATE"])).geometry()

    # Export each image within the for loop
    years_strings = [str(y) for y in range(start_year, end_year)]
    for i,y in zip(range(len(years_strings)), years_strings):
        print(y, ": STARTING.")
        img = ee.Image(banded_images_list.get(ee.Number(i)))
        export = ee.batch.Export.image.toAsset(
            image = img,
            description = 'covariates' + y,
            assetId = config["GEEPATHS"]["ASSETID"] + y,
            region = ee.Geometry(state_geometry),
            scale =  100,
            maxPixels = 1e13)
        export.start()
        
        status = export.status()['state']
        if status == 'READY':
            print(y,": SUBMITTED TO SERVER.")
        else:
            print(y,": ERROR. Status: ",status)
            print("Check your GEE tasks for more information.")
        print("")

    print("\nAll requests submitted. Please be patient while GEE processes your images.")
    print("You can get more information on the status of your requests in your GEE tasks.")

if __name__ == "__main__":
    main()
