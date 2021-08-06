#!/usr/bin/env python3

import configparser
import pandas  as pd
import matplotlib.pyplot as plt
from numpy import sum
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE

from src.ml_funs import *

def main():
    config = configparser.ConfigParser()
    config.read('aisconfig.ini')

    trainingdata = config['LOCALPATHS']['TRAINING_DATA']
    trainingglob = trainingdata + '*.csv'
    decade1 = config['LOCALPATHS']['TESTING_DATA']
    #decade2 = config['LOCALPATHS']['DECADE2']
    decade1_pred = config['LOCALPATHS']['VISUALIZATION_DATA']
    #decade2_pred = config['LOCALPATHS']['DECADE2_PRED']
    HUC_state = config['LOCALPATHS']['HUC_STATE'] 


    # merge training data into one df 
    df = concat_training_csvs(trainingglob)

    # split into training and testing data
    X_train, X_test, y_train, y_test = \
        train_test_split(df.drop(columns = ['Avg_Presence']),
                         df['Avg_Presence'], 
                         random_state = 73)
        
    # build and train the voting classifier
    vc = build_voting_classifier(X_train, X_test, y_train, y_test)

    # print confusion matrix and accuracy score
    print_vc_accuracy(vc, X_test, y_test)

    # read in decade csvs and drop unecessary rows and cols
    first_decade = preprocess_decade(decade1, X_train)

    # get predictions dfs
    first_decade_predictions = get_predictions(vc, first_decade, X_train)

    # write predictions dfs merged with geometry
    write_predictions(first_decade_predictions, HUC_state, decade1_pred)


    # Get Feature Importances
    # Do a bit of manipulation and run all these feature importance techniques

    vc_names = vc.estimators

    rfe_dict = make_dict(vc_names)
    perm_dict = make_dict(vc_names)
    drop_dict = make_dict(vc_names)

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
            

    #We end up with a dataframe that says, for any model / feature importance technique, 
    # whether or not a feature ended up in the top N (i.e. whether or not that feature is important).
    features_df = pd.concat([pd.DataFrame(rename_dict(perm_dict,'PERM_', X_train)).reset_index(drop=True),
                            pd.DataFrame(rename_dict(rfe_dict,'RFE_', X_train)),
                            pd.DataFrame(rename_dict(drop_dict, 'DROP_', X_train))],axis=1)


    features_df['Feature'] = X_train.columns
    features_df['Total'] = sum(features_df, axis=1)
    features_df.sort_values(['Total','Feature'] , ascending=False,inplace=True)
    features_df

    fig, ax = plt.subplots(figsize=(8,4))
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left" )
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left" )

    # plt.xticks(rotation=75)
    plt.bar(features_df['Feature'], features_df['Total'])
    plt.title("Variable importances")
    plt.ylabel("Number of times a variable appeared in top 3")
    plt.show()

if __name__ == "__main__":
    main()
