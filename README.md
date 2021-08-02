# AIS
Aquatic Invasive Species (AIS) habitat suitability prediction toolset

---

## Dependencies
You will need access to a Google Earth Engine account in order to use this package.
In addidtion, the following packages are required:
* python 3
* earthengine-api 
* pandas
* geopandas
* numpy

---

## Setup 
#### Conda
It is recommended that you set up a conda environment prior to using this package.
If you choose not to, continue down to [Earthengine authentication](#earthengine-authentication). 
```
% conda create -n gee python=3
% source activate gee
% conda install -c conda-forge earthengine-api
% conda install -c anaconda pandas
% conda install -c anaconda geopandas
% conda install -c anaconda numpy
```

#### Earthengine authentication
Make sure you have access to an Earth Engine account and have installed Python earthengine-api. 
Run `earthengine authenticate` and follow the prompts.\
You should now have an environment variable set up an authentication key, which allows you to directly initialize ee without authenticating it every time.

---

## Workflow

The two goals of this software are to produce a prediction visualization and produce a feature importance histogram.
In order to do both of these, we need a set of training data. In order to get training data, we need both covariate assets and AIS presence/absence data.\
With this in mind, the workflow is structured as follows:
1. Generate covariate assets in GEE
2. Spatially thin user supplied presence/absence data
3. Combine covariate assets and thinned presence/absence data to create training data files
4. Train and run a machine learning model that outputs a prediction visualization and a feature importance graphic


The following sections will walk you through running the pipeline from start to finish.
For this section, it is assumed that your environment is properly set up.

#### Make Covariates: ./make_covariates.py
Required config variables:
* START_YEAR: First year of interest range
* END_YEAR: Last year of interest range
* STATE: US state where your presence/absence data exists
* ASSETID: GEE path to where the covariate files will be exported

In order to create training data for the machine learning model, you will need annual covariate data assets stored in GEE.
Generating the covariate files takes a while (> 1 hour), but should only need to be done once per state. \
To generate a set of files, set variables in aisconfig.ini, then run `./make_covariates`.

NOTES:
* You should only use years for which you have data in your presence/absence datasets.
* (7/28/21) The only state currently supported is Montana.
* Your GEE paths will (at least should) always start with `/users/<gee_username>/`
* You can check the status of your covariate assets in the `tasks` tab in your GEE console.

#### Spatially Thin Data
I'll fill this in when I've finished the code

#### Make Training Data: ./make_training_data.py
Required config variables:
* START_YEAR: First year of interest range
* END_YEAR: Last year of interest range
* STATE: US state where your presence/absence data exists
* ASSETID: GEE path where the covariate files will be exported to
* AIS_THINNED_POINT_PATH: GEE path to your thinned point data asset
* TRAININGDATA: Local directory to where your training files will be exported

Training data files are .csv files that combine covariate data and presence/absence data. \
These files are used to train the Machine Learning model. They require GEE covariate assets for each year in your interest range,
and a GEE asset of thinned presence/absence data.\
When this step is done, you will have a collection of .csv files in the folder specified by TRAININGDATA in aisconfig.ini. The files are named `2002.csv`, `2003.csv`, etc for each year in your interest range.\
To generate training data, set variables in aisconfig.ini and run `./make_training_data.py`

NOTES:
* The program will overwrite any files in the given folder that share the name of the file being exported.
* If your presence/absence asset does not contain any points for a given year, it will display a warning and skip that year.
* (8/2/2021) The only state currently supported is Montana.

#### Make Testing Data: ./make_testing_data.py
I'll also fill this one in when I've finished the code. 

#### Machine Learning: ./ml_script.py
I'm just about done with this part, actually.

---

## Contributors 
Sean Carter\
Leif Howard\
Myles Stokowski\
Kevin Christensen
