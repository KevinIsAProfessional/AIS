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
2. Combine covariate assets and user supplied presence/absence data to create training data files
3. Train and run a machine learning model that outputs a prediction visualization and a feature importance graphic


The following sections will walk you through running the pipeline from start to finish.
For this section, it is assumed that your environment is properly set up.

#### Covariates
Required config variables:
* START_YEAR: First year of interest range
* END_YEAR: Last year of interest range
* STATE: US state where your presence/absence data exists
* ASSETID: GEE path where the covariate files will be exported to. 

In order to create training data for the machine learning model, you will need annual covariate data assets stored in GEE.
Generating the covariate files takes a while (> 1 hour), but should only need to be done once per state. \
To generate a set of files, set variables in aisconfig.ini, then run `./yearly_covariates`.

NOTES:
* You should only use years for which you have data in your presence/absence datasets.
* (7/28/21) The only state currently supported is Montana.
* Your GEE paths will (at least should) always start with `/users/<gee_username>/`
* You can check the status of your covariate assets in the `tasks` tab in your GEE console.


#### Training Data
I'll fill this in when I've finished the code.

#### Machine Learning
I'll also fill this one in when I've finished the code. 


---

## Contributors 
Sean Carter\
Leif Howard\
Myles Stokowski\
Kevin Christensen\
