'''
Created on Mar 5, 2017

@author: leem42
'''
    
    
import numpy as np
import pandas as pd
import pickle
import sys
from numpy import log
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles


def main():

    #######################################
    #    Open our data set                #
    #######################################    
    train_file = open('train_with_nearest.obj', 'rb')
    train = pickle.load(train_file)
          
    train_file = open('response_train_updated.obj', 'rb')
    response = pickle.load(train_file)
      

      
    train_file = open('test_with_nearest.obj', 'rb')
    test = pickle.load(train_file)
      
    train_file = open('response_test_updatedt.obj', 'rb')
    test_response = pickle.load(train_file)  
     
    train = train.drop("ParticipantBarcode",axis=1)
    train = train.drop("original_gene_symbol",axis=1) 
    train = train.drop("locations",axis=1)
    train['response'] = response
    
    test = test.drop("ParticipantBarcode",axis=1)
    test = test.drop("original_gene_symbol",axis=1) 
    test = test.drop("locations",axis=1)

    ####################################
    #  Convert response to 0,1 values  #
    ####################################
    
    Y = (np.array(response) > 600).astype(int)
    for i in range(len(Y)):
        if(Y[i] == 0):
            Y[i]= -1

    test_response  = (np.array(test_response) > 600).astype(int)
    for i in range(len(test_response)):
        if(test_response[i] == 0):
            test_response[i]= -1        
    
    train['response'] = Y
    test['response'] = test_response
    
    ####################################
    #  Initializa a_i = 0 for all points #
    ####################################
    
    alpha = [1.0 / len(train.iloc[:,0])] * len(train.iloc[:,0])
    train['alpha'] = alpha
    
    features = train.columns
    train = train.iloc[range(1000),:]
    
    features = features.drop("alpha").drop("response")
    
    df = pd.DataFrame.from_items([('A', [1, 1, 1, 1]), ('B', [1, 1, 0,0])])
    df['response'] = [1,1,0,0]
    df['alpha'] = [0.25,0.25,0.25,0.25]
    print df
    
    weights,splits = learn_booster(train, 10, features)
    print weights
#     print train.shape
#     trained_root = open('trained_weights.obj', 'wb')
#     pickle.dump(weights,trained_root)
 
    test = test.iloc[range(1000),:]
     
    #### TO RUN ON TRAIN REMOVE ALPHA
#     train = train.drop("alpha",axis=1)
#     error = classifyData(test, weights, splits)
#     print error
      
    ##### 
    # Professional sklearn attempt
    #####
    
    
#     bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
#                          algorithm="SAMME",
#                          )
#     train = train.drop('response',axis=1).drop('alpha',axis=1)
#     bdt.fit(train,Y[0:1000])
#     test = test.drop("response",axis=1)
#     print train.shape
#     print test.shape
#     output = sum(np.abs(bdt.predict(test) - test_response[0:1000]))
#     print output
    
def learn_booster(X, T, features):
    
    weights = {}
    splits = {}
    
    for feature in features:
        sorted_features = np.sort(X[feature])
        additional_vector = sorted_features.copy()
        additional_vector = additional_vector[1:]
        sorted_features = sorted_features[0:len(sorted_features) - 1]
        #### to_split_on is potential values to splot with
        ### May get many of the same feature, so we choose only unique splits
        to_split_on = set(np.divide(np.add(sorted_features,additional_vector),2.0) )
        #### get the best feature by its best possible min error
        val_split, minError = min_error_split(X, to_split_on, feature)
        splits[feature] = val_split
        weights[feature] = 1
    
    for iteration in range(T):
        chosenFeature, chosenError = find_best_feature(X, features,splits)
        weights[chosenFeature] = 0.5 * np.log((1 - chosenError) / chosenError)
        X = recompute_alphas(X, weights, splits, feature)
    return weights,splits
        
def classifyData(X,weights,splits):
    
    response = X['response']
    X = X.drop('response',axis=1)
    error = 0
    for i in range(len(X.iloc[:,0])):
        row = X.iloc[i,:]
        computed_response  = 0
        for key in splits.keys():
            value = splits[key]
            computed_response+= (row[key] >= value) * weights[key]
        computed_response = np.sign(computed_response)
        error+= abs(response.iloc[i] - computed_response)  
        
    return error
          
        
def recompute_alphas(X, weights, splits,feature):
    response = X['response']
    split = splits[feature]
    computed_values = (X[feature] >= split).astype(int)
    VectorOfError = np.abs(response - computed_values)
    together = zip(VectorOfError,X['alpha'])
    feature_index = X.columns.tolist().index(feature)
    for i, value in enumerate(together):
        error = value[0]
        value = value[1]
        weight = weights[feature]
        if(error  == 0):
            X.iloc[i,feature_index] = value * np.exp(weight * -1)
        else:
            X.iloc[i,feature_index] = value * np.exp(weight)
    return X
##############################################
#    Find the best feature to first split on #
#    For a specific subtree                  #
##############################################
def find_best_feature(X,FeaturesToUse, splits):
    min_feature_error = 999999999999
    feature_name = 0
    for feature in FeaturesToUse:
        
        #### get the best feature by its best possible min error
        error = classification_error(X, splits[feature], feature)
        if(error < min_feature_error):
            min_feature_error = error
            feature_name = feature
    return feature_name, min_feature_error
        
# calculate the classification error
def classification_error(X,split,feature):
    response = X['response']
    classification = X[feature]
    classification = (classification >= split).astype(float)
    classification[classification == 0]  = -1
    classification = classification / 2.0
    error = sum(X['alpha'] *abs( (classification - response)))
    error = error / sum(X['alpha'])

    return error
    
# calculate the classification errors for each value in the ToSplitOn parameter
def min_error_split(X,ToSplitOn,feature):
    minError = 999999999999
    minErrorIndex = 0
    response = X['response']
    
    for i, val in enumerate(ToSplitOn):
        error = 0
        classification = X[feature]
        classification = (classification >= val).astype(int)
        classification[classification == 0]  = -1
        error = sum(abs(classification - response)) / 2.0
        if(error < minError):
            minError = error
            minErrorIndex = i
    
    val_of_split = X[feature].iloc[minErrorIndex]
    return val_of_split, minError
    
if __name__ == '__main__':
    main()