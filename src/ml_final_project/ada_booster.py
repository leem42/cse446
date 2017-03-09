'''
Created on Mar 5, 2017

@author: Michael Lee, Jonathan Wolf

ada_booster program implements a binary classification using depth-one decision trees. 
The program learns by fitting itself to a data set with given output labels and adjusts dynamically
to focus on data points that it missclassified. This implementation focuses on data with continuous values 
for its feature.

This class implements the algorithm described at [1]

Command Line Instructions
----------
    To run ada_booster on  on a test data set do as follows:
    python ada_booster.py TRAIN TEST
    
    TRAIN
        This is the train data being used to learn the algorithm, this data set must be a file in the local directory 
        that is in .csv format. A note on the format, this data set must contain the output for each observation in a
         column named 'response', so TRAIN['response'] would return a column of the response variables for each 
         observation.
    
    TEST
        This is the data to make predictions on, if there is a column 'response' then the program will print out its 
        accuracy on this table, otherwise it will only print out it predictions.

References

----------
    
    [1] https://courses.cs.washington.edu/courses/cse446/17wi/slides/boosting-kNNclassification-annotated.pdf
    (Slide 2)
    
'''
    
    
import numpy as np
import pandas as pd
import pickle
import sys
from numpy import log
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    #  Convert response to -1,1 values  #
    ####################################
    
    Y = (np.array(response) >= 238).astype(int)
    for i in range(len(Y)):
        if(Y[i] == 0):
            Y[i]= -1

    test_response  = (np.array(test_response) >= 238).astype(int)
    for i in range(len(test_response)):
        if(test_response[i] == 0):
            test_response[i]= -1        
    
    train['response'] = Y
    test['response'] = test_response
    
    ###################################
    #  Initializa a_i = 1/N for all points #
    ####################################
    
    alpha = [1.0 / len(train.iloc[:,0])] * len(train.iloc[:,0])
    train['alpha'] = alpha
    
    features = train.columns
    features = features.drop("alpha").drop("response")
    
    T  = [1,2,3,4,5,10,20,50,100,200,500]
    bestT = None
    errorForT = 99999999999999999999
    split = None
    bestSplit = {}
    bestWeights = {}
    for t in T :
        result = {}
        error = 0
        for subset in range(5):
            train_subset, validation = train_test_split(train, test_size = 0.2)
            weights,splits = learn_booster(train_subset, t, features)
            error+= classifyData(validation, weights, splits)
            result = weights
            split = splits
        print "T is : " + str(t)
        average_error = error / 5.0
        print "Error for t is:" + str(average_error)
        if(average_error < errorForT ):
            bestT = t
            bestWeights = result
            bestSplit = split
            errorForT = average_error
        ##### Save resulting weight objects
        trained_weight = open('trained_weight_' + str(T) + '.obj', 'wb')
        pickle.dump(result,trained_weight)

    print "Best value of t and its error below"
    print bestT
    print errorForT
    ### TO RUN ON TRAIN REMOVE ALPHA
    
    error = classifyData(test, bestWeights, bestSplit)
    print error
    
    ##### 
    # Professional sklearn attempt
    #####
    
    
#     bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
#                          algorithm="SAMME",
#                          )
#     train = train.drop('response',axis=1)
#     bdt.fit(train,Y)
#     test = test.drop("response",axis=1)
#     output = sum(np.abs(bdt.predict(test) - test_response)) / 2
#     print output
   
   
    
def learn_booster(X, T, features):
    '''
    Parameters
    ----------
    X:
        The data matrix being trained on
    
    T:
        The number of iterations to run on the learn booster
        
    Features:
        The name of the features used in data matrix X

    Returns
    ----------
    Weights:
        Dictionary, where the keys are the features to use and the values are the values to split the feature on.
        A single key-value pair then represents a single decision tree 
        
    Splits:
        

    '''
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
        weights[feature] = 0
    
    for iteration in range(T):
        start_time = time.time()
        chosenFeature, chosenError = find_best_feature(X, features,splits)
        print chosenError
        weights[chosenFeature] = 0.5 * np.log((1 - chosenError) / chosenError)
        X = recompute_alphas(X, weights, splits, feature)
        normalized_alpha = X['alpha']
        normalized_alpha = normalized_alpha / sum(normalized_alpha)
        X['alpha'] = normalized_alpha
     
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
            classify = row[key] >= value
            if(classify == 0):
                classify = -1
            computed_response+= (classify) * weights[key]
        computed_response = np.sign(computed_response)
        error+= abs(response.iloc[i] - computed_response) / 2.0
        
    return error
          
        
def recompute_alphas(X, weights, splits,feature):
    response = X['response']
    split = splits[feature]
    computed_values = (X[feature] >= split).astype(int)
    VectorOfError = np.abs(response - computed_values)
    together = zip(VectorOfError,X['alpha'])
    new_alpha = []
    feature_index = X.columns.tolist().index(feature)
    for i, value in enumerate(together):
        error = value[0]
        value = value[1]
        weight = weights[feature]
        if(error  == 0):
            new_alpha.append(value * np.exp(weight * -1))
        else:
            new_alpha.append(value * np.exp(weight * 1))
    X['alpha'] = new_alpha
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
    error = sum(X['alpha'] *abs( (classification - response))) / 2.0
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