'''
Created on Mar 5, 2017

@author: Michael Lee, Jonothan Wolf

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
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
import time
from sklearn.decomposition import PCA


def main():

    #######################################
    #    Open our data set                #
    #######################################    
    
    train_file = open('train_data_simple.obj', 'rb')
    train = pickle.load(train_file)
    
    test_file = open('test_simple_2,obj', 'rb')
    test = pickle.load(test_file)
    
    Y = train['response']
    train = train.drop('response',axis=1)
    tree = DecisionTreeClassifier(max_depth=1)
    tree=tree.fit(train,Y)

    features = train.columns
    features = features.drop("response")
    print features
    T  = [5,10,20,50,100,200,500]
#     train = train.iloc[range(1000),:]
#     test = test.iloc[range(500),:]
#     train[train.response == -1] = 0
#     test[test.response == -1] = 0
#     weights = learn_booster(train, 10, features)
#     error = classifyData(test, weights)
#     print error
#     sys.exit(0)
#     bestT = None
#     errorForT = 99999999999999999999
#     bestWeights = {}
#     train = train.iloc[range(500),:]
#     for t in T :
#         result = {}
#         error = 0
#         print "T is : " + str(t)
#         for subset in range(2):
#             train_subset, validation = train_test_split(train, test_size = 0.2)
#             weights = learn_booster(train_subset, t, features)
#             error+= classifyData(validation, weights)
#             print error
#             result = weights
#         average_error = error / 2.0
#         print "Error for t is:" + str(average_error)
#         if(average_error < errorForT ):
#             bestT = t
#             bestWeights = result
#             errorForT = average_error
#         ##### Save resulting weight objects
#         trained_weight = open('new_trained_weight_' + str(T) + '.obj', 'wb')
#         pickle.dump(result,trained_weight)
# 
#     print "Best value of t and its error below"
#     print bestT
#     print errorForT
#     ### TO RUN ON TRAIN REMOVE ALPHA
#     
#     error = classifyData(test, bestWeights)
#     print error
    
    ##### 
    # Professional sklearn attempt
    #####
    
    
#     bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
#                          algorithm="SAMME",
#                          )
#     Y = train['response']
#     train = train.drop('response',axis=1)
#     bdt.fit(train,Y)
#     test_response = test['response']
#     test = test.drop("response",axis=1)
# #     print bdt.predict(test)
#     output = sum(np.abs(bdt.predict(test) - test_response)) / 2
#     print output
#     pca = PCA().fit(train,Y)
#     
    print
    
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
        

    '''
    weights = {}
    alpha = [1.0 / len(X.iloc[:,0])] * len(X.iloc[:,0])
    for iteration in range(T):
        #############################################
        # Step One: Generate Next Best Weak Learner #
        #############################################
        start = time.time()
        chosenFeature, value_of_split, chosenError = find_best_feature_with_alpha(X,features,alpha)
        end = time.time()
        print 'time to get best feature: ' +  str(end - start)
        ################################################################
        # Step Two: Compute weight for the new learner and update alpha#
        ################################################################
        if(chosenFeature not in weights):
            weights[chosenFeature] = []
        new_weight = (0.5 * np.log((1 - chosenError) / chosenError))
        weights[chosenFeature].append((0.5 * np.log((1 - chosenError) / chosenError), value_of_split))
        alpha = recompute_alphas(X, new_weight,value_of_split, chosenFeature,alpha)
        alpha = alpha / np.sum(alpha)

    return weights


def find_best_feature_with_alpha(X,features,alpha):
    min_feature_error = 999999999999
    feature_index = 0
    value_of_split = None
    for feature in features:
        #### Sort features into new columns
        start = time.time()
        sorted_features = np.sort(X[feature])
        additional_vector = sorted_features.copy()
        additional_vector = additional_vector[1:]
        sorted_features = sorted_features[0:len(sorted_features) - 1]
        #### to_split_on is potential values to splot with
        ### May get many of the same feature, so we choose only unique splits
        
        to_split_on = set(np.divide(np.add(sorted_features,additional_vector),2.0) )  
        end = time.time()
#         print 'time to get sort: ' +  str(end - start)
#         print 'length of split ' + str(len(to_split_on))
        #### get the best feature by its best possible min error
        val_split, min_split_error = min_error_split(X, to_split_on, feature,alpha)
#         print feature
#         print min_split_error
        if(min_split_error < min_feature_error):
            min_feature_error = min_split_error
            value_of_split = val_split
            feature_index = feature
    if(feature_index == 0):
        print 'hi'
 
    return feature_index, value_of_split, min_feature_error   
      
def classifyData(X,weights):
    response = X['response']
    X = X.drop('response',axis=1)
    error = 0
    for i in range(len(X.iloc[:,0])):
        row = X.iloc[i,:]
        computed_response  = 0
        for key in weights.keys():
            for value in weights[key]:
                weight = value[0]
                split = value[1]
                classify = row[key] >= split
                if(classify == 0):
                    classify = -1 
                computed_response+= (classify) * weight
        computed_response = np.sign(computed_response)
        error+= abs(response.iloc[i] - computed_response) 
    return error / 2.0

def recompute_alphas(X, new_weight,last_split,feature,alpha):
    response = X['response']
    split = last_split
    computed_values = (X[feature] >= split).astype(int)
    computed_values[computed_values == 0] = -1
    VectorOfError = np.abs(response - computed_values) / 2.0
    together = zip(VectorOfError,alpha)
    new_alpha = []
    weight  = new_weight
    for i, value in enumerate(together):
        error = value[0]
        value = value[1]
        if(error  == 0):
            new_alpha.append(value * np.exp(weight * -1))
        else:
            new_alpha.append(value * np.exp(weight * 1))

    
     
    return new_alpha

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
def min_error_split(X,ToSplitOn,feature,alpha):
    minError = 999999999999
    minErrorIndex = 0
    response = X['response']
    for i, val in enumerate(ToSplitOn):
        error = 0
        classification = X[feature]
        classification = (classification >= val).astype(int)
        classification[classification == 0]  = -1
        error = alpha*np.abs(classification - response)  / 2.0
        error = sum(error)
        error = error / sum(alpha)
        if(error < minError):
            minError = error
            minErrorIndex = i
#             
#     def my_func(v,split):
#         return (v >= split).astype(int)
#     
#     def set_to_minus_one(v):
#         v[v == 0] = -1
#         return v
#     
#     def get_error(v,alpha,response):
#         error = alpha*np.abs(classification - response) / 2.0
#         error = sum(error)
#         error = error / sum(alpha)
#         
#     classification = np.tile(X[feature], (len(ToSplitOn),1))
#     ToSplitOn = np.array(ToSplitOn)
#     classification = np.apply_along_axis(my_func, 0, ToSplitOn,classification)
#     classification = np.apply_along_axis(set_to_minus_one, 1, classification)
#     classification = np.apply_along_axis(my_func, 1,classification, alpha,response)
#     
#     best = np.argmin(classification)

    val_of_split = X[feature].iloc[minErrorIndex]
    return val_of_split, minError
    
if __name__ == '__main__':
    main()
