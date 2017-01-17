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
    weights = numpy.random.normal(size=96)
    output = 0

    df_train = pd.read_table('crime-test.txt')
    df_test = pd.read_table('crime-train.txt')
    
    index = 0
    
    diff = INFINITY * -1
    while( index < 1 ):
        for j in range(len(df_train.columns)):
            # accessing feature j of design matrix df_train
            a_j = 2 * sum(float(x) for x in df_train[df_train.columns[j]].values)
            c_j = 0
            ## at column j, accessing each value (ie. the ith value)
            for i,value in enumerate(df_train[df_train.columns[j]].values):
                ### multiple our jth-weight times the data point at i
                y_i =  sum(numpy.transpose(weights) * df_train.iloc[i,0:]) 
                sub_j = numpy.transpose(numpy.delete(weights,j)) * numpy.delete(numpy.array(df_train.iloc[i,0:]),j)   
                total = value * ( y_i - sub_j + weights[j]*value)     
                c_j+= value
            c_j*=2
            weight_old= weights[j]
            weights[j] = soft(c_j / a_j, lamda / a_j)
            diff = max(diff,abs(weights[j] - weight_old))
        if(diff < 10e-6):
            diff = INFINITY * -1
            index+=1
            lamda = lamda / 2
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