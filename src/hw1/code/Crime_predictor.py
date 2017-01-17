'''
Created on Jan 16, 2017

@author: leem42

                 
'''


from __future__ import division
from json.encoder import INFINITY

import pandas as pd 
import sys, os, numpy, matplotlib.pyplot, pickle



def main(args):
    
    lamda = 600
 #  response = args[2]
 #  matrix = args[3]
    weights = numpy.array(numpy.random.normal(size=96), dtype=float)
    output = 0

    df_train = pd.read_table('crime-test.txt')
    
    index = 0
    
    diff = 10e-5
    while( diff > 10e-6 ):
        diff = None
        print weights
        for j in range(len(df_train.columns)):
            # accessing feature j of design matrix df_train
            a_j = 2 * numpy.sum(df_train[df_train.columns[j]].values)
            c_j = 0
            ## at column j, accessing each value (ie. the ith value)
            for i,value in enumerate(df_train[df_train.columns[j]].values):
                ### multiple our jth-weight times the data point at i
                y_i =  sum(numpy.transpose(weights) *numpy.array(df_train.iloc[i,0:])) 
                sub_j = numpy.transpose(numpy.delete(weights,j)) * numpy.delete(numpy.array(df_train.iloc[i,0:]),j)   
                total = value * ( y_i - sub_j + weights[j]*value)     
                c_j+= value
            c_j*=2
            weight_old= weights[j]
            weights[j] = soft(c_j / a_j, lamda / a_j)
            diff = max(diff,abs(weights[j] - weight_old))
            print abs(weights[j] - weight_old)
        print weights
        print 'iteration ' + str(diff)

     
    print weights
    print 'agePct12t29 = ' + str(weights[df_train.columns.get_loc('agePct12t29')])
    print 'pctWSocSec = ' + str(weights[df_train.columns.get_loc('pctWSocSec')])
    print 'PctKids2Par = ' + str(weights[df_train.columns.get_loc('PctKids2Par')])
    print 'PctIlleg = ' + str(weights[df_train.columns.get_loc('PctIlleg')])
    print 'HousVacant = ' + str(weights[df_train.columns.get_loc('HousVacant')])

#    pickle.dump(weights, open("weights.obj","wb"))


def soft(a,b):    
    return numpy.sign(a)*(abs(a) - b)
        

if __name__ == '__main__':
    main(sys.argv) 