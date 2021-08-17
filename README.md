 AIS
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

###### Protip: GEE = Google Earth Engine
######         AIS = Aquatic Invasive Species

The two goals of this software are to produce a prediction visualization and produce a ranking of environmental covariates that describes their relative importance in predicting aquatic invasive species habitat.
In order to do both of these, we need a set of training data that describes the environmental conditions where species have been found. In order to get said training data, we need both covariate assets and AIS presence/absence data.\
With this in mind, the workflow is structured as follows:
1. Generate covariate assets - raster images that indicate environmental conditions, primarily observed by remote sensing satellite imagery - in GEE
2. Spatially thin user supplied presence/absence data
3. Combine covariate assets and thinned presence/absence data to create training data files
4. Train and run a machine learning model that outputs a prediction visualization and a feature importance graphic


The following sections will walk you through running the pipeline from start to finish.
For this section, it is assumed that your environment is properly set up. In addition, please ensure that your presence / absence dataset has the following required columns, with each column name matching exactly:
| NEEDED| NEEDED|

### Config: aisconfig.ini
* STATE: The US state that contains your presence/absence data points
* STATE_ABBREVIATION: The two letter abbreviation for your chosen state. e.g. Montana = MT
* START_YEAR: First year of interest range
* END_YEAR: Last year of interest range

* GEE_PATH: The GEE path to your Earth Engine user directory. Must end in a forward slash. e.g. `users/kjchristensen93/`
* AIS_POINT_PATH: The GEE path to your (unthinned) presence/absence asset. This is the path to an asset, not to a directory. It must NOT end in a forward slash.  
* AIS_THINNED_POINT_PATH: The GEE path to your thinned presence/absence asset. This is a path to an asset, not to a directory. It mush NOT end in a forward slash.
* ASSETID: GEE path to where the covariate files will be exported. This is a directory, it must end in a forward slash.

* TRAINING_DATA: Local directory where, once created, the training csvs containing covariate information your training files  will be exported. This happens during the `./training_data`. 
* TESTING_DATA: Local path, including filename, to where `./testing_data.py` will export it's output file. This is a randomly selected portion of your data to be used for model validation. e.g. `./datasets/training_data/<your-filename>.csv`
* VISUALIZATION_DATA: Local path, including filename, to where `./ml_script.py` will export it's output file. e.g. `./datasets/visualizations/<your-filename>.csv`
* HUC_STATE: Local path, including filename, of the .geojson file containing HUC data. (8/3/21) We will provide a .geojson for Montana.


#### Make Covariates: ./make_covariates.py
Required config variables:\
START_YEAR, END_YEAR, STATE, ASSETID

In order to create training data for the machine learning model, you will need annual covariate data assets stored in GEE.
Generating the covariate files takes a while (> 1 hour), but should only need to be done once per state. \
To generate a set of files, set variables in aisconfig.ini, then run `./make_covariates`.

NOTES:
* You should only use years for which you have data in your presence/absence datasets.
* Your GEE paths will (at least should) always start with `/users/<gee_username>/`
* You can check the status of your covariate assets in the `tasks` tab in your GEE console.

#### Spatially Thin Data
I'll fill this in when I've finished the code

#### Make Training Data: ./training_data.py
Required config variables:\
START_YEAR, END_YEAR, STATE_ABBREVIATION, ASSETID, AIS_THINNED_POINT_PATH, TRAININGDATA

Training data files are .csv files that combine covariate data and presence/absence data. \
These files are used to train the Machine Learning model. They require GEE covariate assets for each year in your interest range,
and a GEE asset of thinned presence/absence data.\
When this step is done, you will have a collection of .csv files in the folder specified by TRAININGDATA in aisconfig.ini. The files are named `2002.csv`, `2003.csv`, etc for each year in your interest range.\
To generate training data, set variables in aisconfig.ini and run `./training_data.py`

NOTES:
* The program will overwrite any files in the given folder that share the name of the file being exported.
* If your presence/absence asset does not contain any points for a given year, it will display a warning and skip that year.

#### Make Testing Data: ./testing_data.py
Required config variables:\
START_YEAR, END_YEAR, STATE_ABBREVIATION, ASSETID, TESTING_DATA

Testing data files take the training data files and average the data over a range of years. \
Testing data files are used by a trained Machine Learning model to produce a prediction file. \
To generate testing data, set variables in aisconfig.ini and run `./testing_data.py`\
You may wish to create multiple testing files from ranges in your training data. To do this, you will need to manually edit START_YEAR, END_YEAR, and TESTING_DATA in aisconfig.ini and run the program multiple times.\
Note that if you don't edit TESTING_DATA, your testing data will be overwritten each time your run the program.


#### Machine Learning: ./ml_script.py
Required config variables: \
TRAINING_DATA, TESTING_DATA, VISUALIZATION_DATA, HUC_STATE

The machine learning portion takes a folder full of training data, a single testing data file, and a huc .geojson, and returns a prediction file in the form of a .csv.
The script also outputs a histogram of feature importances as determined by the ML model.
To run, set the variables in aisconfig.ini and run ./run_model.
If you have split your testing data into multiple files, you will have to run the model multiple times, changing TESTING_DATA and VISUALIZATION_DATA in aisconfig each time.
If you do not change VISUALIZATION_DATA, your prediction .csv will be overwritten each time you run the model.

---

## Contributors 
Sean Carter\
Leif Howard\
Myles Stokowski\
Kevin Christensen
