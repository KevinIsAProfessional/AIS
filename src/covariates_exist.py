# covariates_exist.py
import os

# return boolean whether all covariate assets exist in GEE for given years list
def covariates_exist(years, assetId):
    command = 'earthengine ls ' + assetId
    result = os.popen(command).read().split('\n')
    assets = list(map(lambda path: path.split('/')[-1], result))
    expected_assets = list(map(lambda year: 'covariates' + str(year), years))
    print('assets_found: ', assets)
    print('expected_assets: ', expected_assets)
    return all(item in assets for item in expected_assets)

