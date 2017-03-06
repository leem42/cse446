'''
Created on Mar 5, 2017

@author: leem42
'''
    
    
import numpy as np
import pandas as pd
import pickle
import sys
from numpy import log
       
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
    
    Y = (response > 600).astype(int)
    test_response  = (test_response > 600).astype(int)
        
    train['response'] = Y
    test['response'] = test_response
    
    ####################################
    #  Initializa a_i = 0 for all points #
    ####################################
    
    alpha = [1.0 / len(train.iloc[:,0])] * len(train.iloc[:,0])
    train['alpha'] = alpha
    
    features = train.columns
    features = features.remove("alpha").remove("response")
    weights,splits = learn_booster(train, 100, features)
    
    error = classifyData(train, weights, splits)
    
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
    
    for i in range(T):
        chosenFeature, chosenError = find_best_feature(X, features)
        weights[chosenFeature] = 0.5 * np.log((1 - chosenError) / chosenError)
        X = recompute_alphas(X, weights, splits, feature)
    return weights,splits
        
def classifyData(X,weights,splits):
    
    response = X['response']
    X= X.drop('response',axis=1)
    X = X.drop('alpha',axis=1)
    error = 0
    for i in range(len(X.iloc[:,0])):
        row = X.iloc[i,:]
        computed_response  = 0
        for key,value in splits:
            computed_response+= (row[key] >= value) * weights[key]
        computed_response =( np.sign(computed_response) + 1) / 2
        error+= abs(response[i] - computed_response)  
        
    return error
          
        
def recompute_alphas(X, weights, splits,feature):
    response = X['response']
    split = splits[feature]
    computed_values = (X[feature] >= split).astype(int)
    VectorOfError = np.abs(response - computed_values)
    
    for i,error, value in enumerate(VectorOfError,X['alpha']):
        weight = weights[feature]
        if(error  == 0):
            X[feature].iloc[i] = value * np.exp(weight * -1)
        else:
            X[feature].iloc[i] = value * np.exp(weight)
    
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
    error = 0
    classification = X[feature]
    classification = (classification >= split).astype(int)
    error = sum(abs(X['alpha'] * (classification - response)))
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
        error = sum(abs(classification - response))
        if(error < minError):
            minError = error
            minErrorIndex = i
    
    val_of_split = X[feature].iloc[minErrorIndex]
    return val_of_split, minError
    
if __name__ == '__main__':
    main()