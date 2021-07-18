# GOAL: Get covariate files for each huc in montana
#  covariate files by year
#  Columns: HucId, mean of each covariate by huc (15 cols)
#  Rows: one for each HUC in montana (4000ish)
#  Files: one for each year (2002-2013/18ish)
#  We use the covariate files, reduce pixel groups into huc level averages for each covariate band
#  reduceRegion will calculate a mean for each band by default 


import ee
ee.Initialize()
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read('../aisconfig.ini')

start_year = int(config['WHENWHERE']['START_YEAR'])
end_year = int(config['WHENWHERE']['END_YEAR'])
covariate_folder = config['GEEPATHS']['ASSETID']
trainingdata = config['LOCALPATHS']['TRAININGDATA']


def reduce_huc_covariates(img,HUC_clip):
    reduced_image = img.reduceRegions(**{
                              'collection': HUC_clip,
                              'reducer': ee.Reducer.mean(),
                              'crs': 'EPSG:4326',
                              'scale': 100,
                              'tileScale': 16}).map(lambda y:y.set({'Time': img.get("system:time_start")}))
    return reduced_image


years = range(start_year, end_year)
HUC_clip = ee.FeatureCollection("USGS/WBD/2017/HUC12").filter(ee.Filter.eq('states',"MT"))

# COVARIATE IMAGES 
images = list(map(lambda x: ee.Image(covariate_folder + str(x)), years))

for i in range(len(years)):
    print("Starting", start_year+i)

    data = reduce_huc_covariates(images[i],HUC_clip) 

    # export as a GEE asset (rather than a local file)
    export = ee.batch.Export.table.toAsset(
            collection = data,
            description = 'covariates_huc' + str(i),
            assetId = covariate_folder + str(i))

    export.start()
    
    # export as local file
    # my_csv = pd.DataFrame([x['properties'] for x in data.getInfo()['features']])
    # my_csv.to_csv((trainingdata) + str(2002+i) + '_huc.csv', index=False) 
   
    print("Finished", start_year+i) 

