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
    response = args[2]
    matrix = args[3]
    weights = numpy.random.normal(size=96)
    output = 0

    df_train = pd.read_table('crime-test.txt')
    df_test = pd.read_table('crime-train.txt')
    
    diff = INFINITY
    while( diff > 10e-6):
        for j in range(len(df_train.values)):
            # accessing feature j of design matrix df_train
            a_j = 2 * sum(float(x) for x in df_train[df_train.columns[j]].values)
            c_j = 0
            ## at column j, accessing each value (ie. the ith value)
            for value in df_train[df_train.columns[j]].values:
                ### multiple our jth-weight times the data point at i
                mid = (weights[j] * value) - (numpy.transpose(weights) * df_train.iloc[0,0:]) 
                mid = mid  + weights[j]*value ### right multiplier
                c_j+= (value * mid)                
            c_j*=2
            weights[j] = soft(c_j / a_j, lamda / a_j)
            diff = max(diff,abs(weights[j]))
            
    print weights
    pickle.dumps(weights, "weights.obj")


def soft(a,b):    
    return numpy.sign(a)*(abs(a) - b)
        

if __name__ == '__main__':
    main(sys.argv) 