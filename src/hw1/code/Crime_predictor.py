'''
Created on Jan 16, 2017

@author: leem42

   
                 
'''


from __future__ import division
from json.encoder import INFINITY

import pandas as pd 
import sys, os, numpy, matplotlib.pyplot, pickle, math



def main(args):
    
    lamda = 18.75
    weights = pickle.load(open("weights.obj","rb"))

    df_train = pd.read_table('crime-train.txt').drop("ViolentCrimesPerPop",axis=1)
    response = pd.read_table('crime-train.txt').iloc[0:,0]
    diff = 10e-5
    
#     print weights
    while( diff > 10e-6 ):
        diff = None
        for j in range(len(df_train.columns) ):
            # accessing feature j of design matrix df_train
            a_j = 2 * numpy.sum(df_train[df_train.columns[j]].values**2)
            fast_j = numpy.sum(df_train.iloc[:,j].values*(response[:] - numpy.dot(df_train, weights) + weights[j] * df_train.iloc[:,j].values))
            fast_j*=2
            weight_old= weights[j]
            weights[j] = soft(fast_j / a_j, lamda / a_j)
            diff = max(diff,abs(weights[j] - weight_old))
     
    print  str(weights[df_train.columns.get_loc('agePct12t29')])
    print  str(weights[df_train.columns.get_loc('pctWSocSec')])
    print  str(weights[df_train.columns.get_loc('PctKids2Par')])
    print  str(weights[df_train.columns.get_loc('PctIlleg')])
    print  str(weights[df_train.columns.get_loc('HousVacant')])
    print
    print weights
    print max(weights), df_train.columns[numpy.where(weights == max(weights))[0][0]]
    print min(weights), df_train.columns[numpy.where(weights == min(weights))[0][0]]
    
    test_matrix = pd.read_table('crime-test.txt').drop("ViolentCrimesPerPop",axis=1)
    response_test = pd.read_table('crime-test.txt').iloc[0:,0]    
    squared_error = sum( (numpy.dot(test_matrix,weights) - response_test[:])**2)
    pickle.dump(weights, open("weights.obj","wb"))

def soft(a,b):    
    return numpy.sign(a) * max(abs(a) - b, 0)

if __name__ == '__main__':
    main(sys.argv) 