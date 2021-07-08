import pandas as pd
import glob
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import *
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import *
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE, SelectFromModel


# return a df with all the trainingglob data concatenated, extra columns
# removed, and avg_presence reduced to 1s and 0s
def concat_training_csvs(trainingglob):
    dfs = []
    for file in glob.glob(trainingglob):
        dfs.append(pd.read_csv(file))

    df = pd.DataFrame(pd.concat(dfs, ignore_index=True))

    to_drop = ['Points', 'areaacres', 'areasqkm',
               'gnis_id','huc12', 'humod','hutype',
               'loaddate', 'metasource', 'name',
               'noncontr00','noncontrib','shape_area',
               'shape_leng', 'sourcedata','sourcefeat',
               'sourceorig','states','tnmid', 'tohuc', 'Time']

    df['Avg_Presence'] = [1 if x > 0 else 0 for x in df['Avg_Presence']]

    return df.drop(columns = to_drop)

### ML STARTS HERE!!!!! ###

# Ensemble modeling class
class Ensemble():
    def __init__(self, models = []):
        self.models = models
        self.accs_dict = None
        self.rocs_dict = None
        self.model_names = [m.__class__.__name__ for m in models]
        self.weights = []
        
    def fit_all(self, X_train, y_train):
        for m in self.models:
            print("Fitting", m.__class__.__name__)
            m.fit(X_train, y_train)
            print(m.__class__.__name__, 'fit.')
        
    # set weights properties for each model
    # FIXME returns either accs or rocs based on metric param
    # but always sets both properties - maybe a strange divergent behavior?
    def evaluate_all(self, X_test, y_true, metric = 'acc'):
        accs = [accuracy_score(y_true, m.predict(X_test)) for m in self.models]
        accs_dict = dict(zip(self.model_names, accs))
        
        # reciever operating score (like accuracy score)
        rocs = [roc_auc_score(y_true, m.predict(X_test)) for m in self.models]
        rocs_dict = dict(zip(self.model_names, rocs))
        
        self.rocs_dict = rocs_dict
        self.accs_dict = accs_dict
        
        if metric == 'acc':
            return accs_dict
        
        return rocs_dict
        
    def get_weights(self):
        return self.rocs_dict
    
    def get_model_names(self):
        return self.model_names
        

# Function for testing classifiers for ideal hyperparameters. Returns the 
# state that results in the highest model accuracy. Grid search would be ideal
# but computationally intensive.
# TODO: currently specific to testing random forest max_depth, could be 
# generalized but might not be worth it because tweaking hyperparams may not
# lead to big increases in model accuracy
def test_rf_hyperparams(X_train, X_test, y_train, y_test):
    state = 0
    accs = []

    while state < 100:
        rf = RandomForestClassifier(max_depth = state)
        rf.fit(X_train, y_train)
        
        accs.append(accuracy_score(y_test, rf.predict(X_test)))
        
        state += 1

    return accs.index(max(accs))


# NOTE: negative = extrapolating, positive = similar to training data
def MESS(train_df, pred_df):
#     Let min_i be the minimum value of variable V_i
#     over the reference point set, and similarly for max_i.
    mins = dict(X_train.min())
    maxs = dict(X_train.max())
    
    def calculate_s(column):
        # Let f_i be the percent of reference points 
        # whose value of variable V_i is smaller than p_i.

        # First store training values
        values = train_df[column]

        # Find f_i as above
        sims = []
        for element in np.array(pred_df[column]):
            f = np.count_nonzero((values < element))/values.size

            # Find Similarity:
            if f == 0:
                sim = ((element - mins[column]) / (maxs[column] - mins[column])) * 100

            elif ((f > 0) & (f <= 50)):
                sim = 2 * f

            elif ((f > 50) & (f < 100)):
                sim = 2 * (100-f)

            elif f == 100:
                sim = ((maxs[column] - element) / (maxs[column] - mins[column])) * 100

            sims.append(sim)


        return sims
    
    # Embedd Dataframe with sim values
    sim_df = pd.DataFrame()
    for c in pred_df.columns:
        sim_df[c] = calculate_s(c)
    
    
    minimum_similarity = sim_df.min(axis=1)
    
    return minimum_similarity


# return a decade df with unnecessary data removed
# decade: path to a decade csv
# X_train: used to drop unecessary cols
def preprocess_decade(decade, X_train): 
    
    decade_df = pd.read_csv(decade, index_col = 'huc12')
    
    # drop columns in predictions input that are not in the training data
    drops = [c for c in decade_df if c not in X_train.columns]
    
    # NOTE: may need to drop rows where HUCS have NA data, which happened
    # previously when a HUC was in Canada and LST data was not in GEE
    bad_hucs = decade_df.index[decade_df.Max_LST_Annual.isna()]

    return decade_df.drop(bad_hucs).drop(columns = drops)

#############################################################################
# END FUNCTIONS, START SCRIPT
#############################################################################

# TODO delete
trainingglob = './datasets/training/*.csv'
decade1 = ('./datasets/decade/decade1.csv')
decade2 =('./datasets/decade/decade2.csv')
decade1_pred = ('./datasets/decade/decade1_pred.csv')
decade2_pred = ('./datasets/decade/decade2_pred.csv')
HUC_state = ('./datasets/hucs/MT_HUCS.geojson')


# merge training data into one df 
df = concat_training_csvs(trainingglob)

# split into training and testing data
X_train, X_test, y_train, y_test = \
    train_test_split(df.drop(columns = ['Avg_Presence']),
                     df['Avg_Presence'], 
                     random_state = 73)

# initialize ensemble of models (tree methods seem far stronger)
mlp = MLPClassifier(max_iter = 1000, random_state = 73)
logit = LogisticRegression(max_iter = 10000)
rf = RandomForestClassifier()
brt = GradientBoostingClassifier()
dt = DecisionTreeClassifier()

# Construct ensemble object
ensemble = Ensemble([mlp, brt, dt, rf, logit]) 
ensemble.fit_all(X_train,y_train)

print(ensemble.evaluate_all(X_test, y_test, metric = 'roc'))
#y_pred = ensemble.evaluate_ensemble(X_test, y_test)

#accuracy_score(y_test,y_pred)

## This chunk of code builds the ensemble 

# This stores the weights that each model should have when voting
# NOTE: main use of ensemble
weights = [ensemble.get_weights()[c] for c in ensemble.get_model_names()]

# Here is where we might turn off different models due to lower accuracy score. 
# I don't know how the models will predict EBT
# Might Turn off ANN
# TODO could automate turning off models based on weight threshold, e.g.
# weights = [x if x > 0.5 else 0 for x in weights]
weights[2] = 0

#Also turn off Logistic Regression and Decision tree
# weights[1] = 0
# weights[-1] = 0

vc_names = [('RF', rf), ('Logit', logit), 
            ('ANN', mlp), ('BRT', brt), ('DT', dt)]

vc = VotingClassifier(estimators=vc_names, 
                      voting='soft', 
                      weights = weights)
vc.fit(X_train, y_train)
print(accuracy_score(y_test, vc.predict(X_test)))

# Check out confusion matrix
confusion_matrix(y_test, vc.predict(X_test))
# Project Model to Environmental Space

first_decade = preprocess_decade(decade1, X_train)
# second_decade = preprocess_decade(decade2, X_train)



## GET PREDICTION UNCERTAINTY
predictions_first = []
#predictions_second = []
# TODO add toggle for which models (e.g. turning off Logistic and MLP)
for ind,model in enumerate(vc.estimators_):
    # Don't use Logistic Regression or MLP
    if ind in [1,2]:
        continue
    # TODO predicting presence/absence?
    predict_prob = [a[0] for a in model.predict_proba(first_decade)]
    predictions_first.append(predict_prob)
    
    #predict_prob = [a[0] for a in model.predict_proba(second_decade)]
    #predictions_second.append(predict_prob)

## \GET PREDICTION UNCERTAINTY
    
first_decade_pred = pd.DataFrame({
    'huc12':pd.Series(first_decade.index),
    'prediction_ensemble':vc.predict(first_decade),
    'prediction_proba':[a[0] for a in vc.predict_proba(first_decade)],
    'prediction_uncertainty': np.std(np.array(predictions_first), axis=0),
    'MESS': MESS(X_train,first_decade)})
#second_decade_pred = pd.DataFrame({'huc12':pd.Series(second_decade.index),
                                  #'prediction_ensemble':vc.predict(second_decade),
                                  #'prediction_proba':[a[0] for a in vc.predict_proba(second_decade)],
                                  #'prediction_uncertainty': np.std(np.array(predictions_second), axis=0),
                                  #'MESS': MESS(X_train,second_decade)})

#Merge these predictions with the Shape File, which needs to be downloaded from the USGS website
# This needs to be downloaded from the USGS website
hucs = gpd.read_file(HUC_state)
hucs['huc12']=hucs['huc12'].astype('int64')

#drops = ['MESS_mean', 'Most_Dissimilar_Variable_majority',
       #'FIRST_Decade_QC_mean', 'Second_Decade_QC_mean', 'FIrst_Decade_QC_sum',
       #'FIRST_DECADE_QC_sum', 'FIRST_DECADE_QC_max', 'SECOND_DECADE_QCmax']
#hucs.drop(columns=drops,inplace=True)

hucs_pred_first = hucs.merge(first_decade_pred,on='huc12')
#hucs_pred_second = hucs.merge(second_decade_pred,on='huc12')

# TODO probably better to keep as/write as geojson rather than CSV
decade_1 = pd.DataFrame(hucs_pred_first)

decade_1.to_csv(decade1_pred, index=False)

#decade_2 = pd.DataFrame(hucs_pred_second)

#decade_2.to_csv(decade2_pred, index=False)

# Get Feature Importances
# NOTE: next question managers will ask is why (what are the predictions based on)

#This will be the backbone of the Community-level "top predictors" analysis.

# Function that runs the "Drop Column" Feature importance technique 
# I actually have these in a separate .py file which would be much cleaner. 

# Short function to create sorted data frame of feature importances
def make_imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df

def drop_col(model, X_train, y_train, random_state = 42):
    #Clone the model
    model_clone = clone(model)
    #Reset random state
    model_clone.random_state = random_state
    #Train and score the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    #Store importances
    importances = []
    
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col,axis=1),y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
        importances.append(benchmark_score - drop_col_score)
        
    importances_df = make_imp_df(X_train.columns, importances)
    return importances_df


# Do a bit of manipulation and run all these feature importance techniques
vc_names = [('RF', rf), ('Logit', logit), ('ANN', mlp), ('BRT', brt), ('DT', dt)]
def make_dict():
    return dict(zip([tup[0] for tup in vc_names], [None]))


rfe_dict = make_dict()
perm_dict = make_dict()
drop_dict = make_dict()

for alg in vc.named_estimators:
    dict_name = alg
    clf = vc.named_estimators[dict_name]
    
    if dict_name == 'ANN':
        continue
    
    print("Considering", clf)
    
    # Find Recursive feature elimination for each classifier
    rfe_selector = RFE(estimator=clf, n_features_to_select=3, step=1, verbose=5)
    rfe_selector.fit(X_train,y_train)
    rfe_support = rfe_selector.get_support()
    rfe_features = X_train.loc[:,rfe_support].columns.tolist()
    
    # Add to rfe_dict
    rfe_dict[dict_name] = rfe_features
    
    
    # //========================================================
    # Find Permuation importance for each classifier
    perm_imp = permutation_importance(clf, X_train, y_train,
                          n_repeats=30,
                          random_state = 0)
    
    perm_imp['feature'] = X_train.columns
    perm_features = pd.DataFrame(perm_imp['feature'],perm_imp['importances_mean'],columns = ['Feature']) \
                            .sort_index(ascending=False)['Feature'].values[:3]
    
    # Add permutation features to dict
    perm_dict[dict_name] = perm_features
    
    
    
    #//========================================================
    # Find Drop Columns importance for each classifier
    drop_col_feats = drop_col(clf, X_train, y_train, random_state = 10)
    drop_col_three = drop_col_feats.sort_values('feature_importance',ascending = False)['feature'][:3]
    
    drop_dict[dict_name] = drop_col_three
    
    
print("Done")




def mark_true(series):
    return [True if feature in series else False for feature in X_train.columns]

def rename_dict(dictionary, tek_name):
    return_names = []
    return_lists = []
    
    for item in dictionary.items():
        return_names.append(tek_name + str(item[0]))
        return_lists.append(mark_true(list(item[1])))
        
    return dict(zip(return_names, return_lists))
        

#We end up with a dataframe that says, for any model / feature importance technique, whether or not a feature ended up in the top N (i.e. whether or not that feature is important).
features_df = pd.concat([pd.DataFrame(rename_dict(perm_dict,'PERM_')).reset_index(drop=True),
                        pd.DataFrame(rename_dict(rfe_dict,'RFE_')),
                        pd.DataFrame(rename_dict(drop_dict, 'DROP_'))],axis=1)


features_df['Feature'] = X_train.columns
features_df['Total'] = np.sum(features_df, axis=1)
features_df.sort_values(['Total','Feature'] , ascending=False,inplace=True)
features_df

fig, ax = plt.subplots(figsize=(8,4))
plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left" )
plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left" )

# plt.xticks(rotation=75)
plt.bar(features_df['Feature'], features_df['Total'])
plt.title("Variable importances")
plt.ylabel("Number of times a variable appeared in top 3")
