'''
Created on Mar 2, 2017

@author: leem42
'''
import matplotlib
import pylab
import numpy as np
import pandas as pd
import math, sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge 
from sklearn import tree
import pickle
import re
import pprint

class Node(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.feature_index = None
        self.feature_value = None
        self.classification = None
    
    def add_point(self,point):
        self.points.append(point)

    def add_left(self, data):
        self.left = Node(data)
        
    def add_right(self, data):
        self.right = Node(data)
    
def decision_tree(X, FeaturesToUse):
        current = Node()
        examples = X['response']
        classification = sum(examples)
        if( classification == 0):
            current.classification = 0
            return current
        elif(classification == len(examples)):
            current.classification = 1
            return current
        if(len(FeaturesToUse) == 0):    #### REPLACE WITH ACTUAL
            if(classification > len(examples) * 0.5):
                current.classification = 1
            else:
                current.classifcation = 0
            return current
        # recursive case
        feature_index, value_of_split = find_best_feature(X,FeaturesToUse)
        column_name = X.columns[feature_index]
        FeaturesToUse = FeaturesToUse.remove(feature_index)
        # Divide X into two chunks: less than split and greater or equal to the split
        left_data = X[X.column_name < value_of_split]    
        right_data = X[X.column_name >=value_of_split]
        current.feature_index = feature_index
        current.feature_value = value_of_split
        current.left = decision_tree(left_data, FeaturesToUse)
        current.right = decision_tree(right_data, FeaturesToUse) 
        return current
##############################################
#    Find the best feature to first split on #
#    For a specific subtree                  #
##############################################
def find_best_feature(X,FeaturesToUse):
    min_feature_error = 999999999
    feature_index = 0
    value_of_split = None
    for feature in FeaturesToUse:
        #### Sort features into new columns
        sorted_features = np.sort(X.iloc[:,feature])
        additional_vector = sorted_features.copy()
        additional_vector = additional_vector[1:]
        sorted_features = sorted_features[0:len(sorted_features) - 1]
        #### to_split_on is potential values to splot with
        to_split_on = np.divide(np.add(sorted_features,additional_vector),2) 
        #### get the best feature by its best possible min error
        val_split, min_split_error = min_error_split(X, to_split_on, feature)
        if(min_split_error < min_feature_error):
            min_feature_error = min_split_error
            value_of_split = val_split
    
    return feature_index, value_of_split
            
        

# calculate the classification errors for each value in the ToSplitOn parameter
def min_error_split(X,ToSplitOn,feature):
    minError = 999999999999
    minErrorIndex = 0
    response = X['response']
    
    for i in range(len(ToSplitOn)):
        error = 0
        val = ToSplitOn[i]
        classification = X.iloc[:,feature]
        classification = classification[classification < val]
        ones = classification.count(1)
        zeros = classification.count(0)
        error = 0
        if(ones >= zeros):
            error = sum(response - np.ones(len(X.iloc[:,0])))
        else:
            error = sum(response - np.zeros(len(X.iloc[:,0])))
        if(error < minError):
            minError = error
            minErrorIndex = i
    
    val_of_split = X.iloc[minErrorIndex,feature]
    return val_of_split, minError
            
            
def get_majority_class(current, Y): 
    left_class_zero = np.zeros(len(current.points))  
    error_left_zero = np.sum(np.abs(Y[current.points] - left_class_zero))
    left_class_one = np.ones(len(current.points))  
    error_left_one = np.sum(np.abs(Y[current.points] - left_class_one))
    decision = 0 if error_left_zero > error_left_one else 1
    return decision
            
def traverse_tree_for_point(observation, root, Y):
    pass
        
    
    
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

    
    Y = np.zeros((len(train.iloc[:,0]),))
    for i in range(len(train.iloc[:,0])):
        Y[i] = response[i] > 600
    for i in range(len(test_response[:])):
        test_response[i] = test_response[i] > 600
    
    
    #######################################
    #    Decision Tree Learner            #
    #######################################  
    FeaturesToUse = range(len(train.iloc[0,:]))
    root = decision_tree(train.copy(), FeaturesToUse)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #######################################
    #    The Dream                       #
    #######################################  
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train, Y)
    error = sum(np.square(test_response - clf.predict(test))) 
    
    print error
    print len(test_response)



if __name__ == '__main__':
    main()