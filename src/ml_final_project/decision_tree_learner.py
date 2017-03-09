'''
Created on Mar 2, 2017

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
    
    [1] http://www.cs.princeton.edu/courses/archive/spr07/cos424/papers/mitchell-dectrees.pdf
    (Page 56)
    
'''



from __future__ import division
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
from sklearn.linear_model.sgd_fast import Classification


TRAINING_DATA = pickle.load(open('train_data_simple.obj', 'rb'))
TRAINING_DATA[TRAINING_DATA.response == -1] = 0
MAX_DEPTH = 10

class Node(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.feature_index = None
        self.feature_value = None
        self.classification = None
    
    def print_me(self,level):
        if(self == None):
            print "None"
        out = str(self.feature_index) + '  Value: ' + str(self.feature_value) 
        print "\t"*level+str(out)+"\n"
        if(self.left == None):
            print "\t"*level+str("Leaf : ") + str(self.classification)+"\n"
        if(self.right == None):
            print "\t"*level+str("Leaf : ") + str(self.classification) + "\n"
        if(self.left != None):
            self.left.print_me(level+1)
        if(self.right != None):
            self.right.print_me(level + 1)
        


def simpleBuildTree(level):
    if(level == 0):
        return None
    root = Node()
    root.left = simpleBuildTree(level - 1)
    return root

####################################
#    Build Decision Tree          #
####################################
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
        if(len(FeaturesToUse) == 1):    #
            if(classification >= len(examples) * 0.5):
                current.classification = 1
            else:
                current.classification = 0
            return current
        # recursive case
        feature_index, value_of_split = find_best_feature(X,FeaturesToUse)
        column_name = feature_index
        print column_name
        FeaturesToUse = FeaturesToUse.drop(column_name)
#         Divide X into two chunks: less than split and greater or equal to the split
        print "Feature: " + str(column_name)
        print "Value of Feature: " + str(value_of_split)
        left_data = X[X[column_name] <  value_of_split]    
        right_data = X[X[column_name] >= value_of_split]
        go_left = True
        go_right =True
        if(left_data.shape[0] == 0):
            current.left = Node()
            classification =sum(TRAINING_DATA['response'])
            decision = None
            if(classification < 0.5 * len(X.iloc[:,0])):
                decision = 0
            else:
                decision = 1
            go_left = False
            current.left.classification = decision
        if(right_data.shape[0] == 0):
            current.right = Node()
            classification =sum(TRAINING_DATA['response'])
            decision = None
            if(classification < 0.5 * len(X.iloc[:,0])):
                decision = 0
            else:
                decision = 1
            current.right.feature_index
            current.right.classification = decision
            go_right = False
        current.feature_index = feature_index
        current.feature_value = value_of_split
        if(go_left):
            current.left = decision_tree(left_data, FeaturesToUse)
        if(go_right):
            current.right = decision_tree(right_data, FeaturesToUse) 

        return current
    
##############################################
#    Find the best feature to first split on #
#    For a specific subtree                  #
##############################################
def find_best_feature(X,FeaturesToUse):
    min_feature_error = 999999999999
    feature_index = 0
    value_of_split = None
    SansResponse = FeaturesToUse.drop("response")
    for feature in SansResponse:
        #### Sort features into new columns
        sorted_features = np.sort(X[feature])
        additional_vector = sorted_features.copy()
        additional_vector = additional_vector[1:]
        sorted_features = sorted_features[0:len(sorted_features) - 1]
        #### to_split_on is potential values to splot with
        ### May get many of the same feature, so we choose only unique splits
        to_split_on = set(np.divide(np.add(sorted_features,additional_vector),2.0) )
        #### get the best feature by its best possible min error
        val_split, min_split_error = min_error_split(X, to_split_on, feature)
        if(min_split_error < min_feature_error):
            min_feature_error = min_split_error
            value_of_split = val_split
            feature_index = feature
    if(feature_index == 0):
        print
    return feature_index, value_of_split
            
        

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
            
def classify_data(X,root):
    error = 0
    if(root.left is None and root.right is None):
        if(X.shape[0] == 0):
            return 0
        classification = root.classification
        computed_response = np.ones(len(X.iloc[:,0])) * classification
        error = sum(np.abs(X['response'] - computed_response))
        return error
     
    feature_value = root.feature_value
    column_name = root.feature_index
    left_data = X[X[column_name] < feature_value]    
    right_data = X[X[column_name] >= feature_value]
    error+=classify_data(left_data, root.left)
    error+=classify_data(right_data, root.right)
    return error 
        
def main():

    #######################################
    #    Open our data set                #
    #######################################    
    train_file = open('train_data_simple.obj', 'rb')
    train = pickle.load(train_file)
    train[train.response == -1] = 0
    
    test_file = open('test_simple_2,obj', 'rb')
    test = pickle.load(test_file)
    test[test.response == -1] = 0

    #######################################
    #    Test Tree Learner            #
    ####################################### 
#     df = pd.DataFrame.from_items([('A', [1, 1, 1, 1]), ('B', [1, 1, 1,1])])
#     df['response'] = [1,1,0,0]
#     print df

    #######################################
    #    Decision Tree Learner   Train    #
    #######################################  
    train = train.iloc[:,:]
    
#     root = decision_tree(train, train.columns)
#     trained_root = open('trained_root_diff.obj', 'wb')
#     pickle.dump(root,trained_root)
#         
    root = open('trained_root_diff.obj', 'rb')
    root = pickle.load(root)
    #######################################
    #    Decision Tree Classify            #
    #######################################  
    error = classify_data(train, root)
    print error
    test = test.iloc[:,:]
    error = classify_data(test, root)
    print error
    print root.print_me(0)
    print len(train.iloc[:,0])
    #######################################
    #    The Dream                       #
    #######################################  

    
#     clf = tree.DecisionTreeClassifier()
#     clf = clf.fit(train, Y)
#     error = sum(np.abs(test_response - clf.predict(test))) 
#      
#     print error



if __name__ == '__main__':
    main()
