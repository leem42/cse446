'''
Created on Jan 16, 2017

@author: leem42

   
                 
'''


from __future__ import division
from json.encoder import INFINITY

import pandas as pd 
import sys, os, numpy, matplotlib.pyplot, pickle, math



def main(args):
    print 'hi'
    lamda = 600

   # weights = pickle.load(open("weights.obj","rb"))
    weights = numpy.random.normal(size=96)

    df_train = pd.read_table('crime-train.txt')
#     for i in range(len(df_train.columns)):
#         df_train[df_train.columns[i]]/=math.sqrt(sum(df_train[df_train.columns[i]]**2))

    diff = 10e-5
    while( diff > 10e-6 ):
        diff = None
        for j in range(len(df_train.columns)):
            # accessing feature j of design matrix df_train
            a_j = 2 * numpy.sum(df_train[df_train.columns[j]].values**2)
            c_j = 0
            ## at column j, accessing each value (ie. the ith value)
            for i,value in enumerate(df_train[df_train.columns[j]].values):
                ### multiple our jth-weight times the data point at i
                y_i =  numpy.matrix(weights) * numpy.matrix(df_train.iloc[i,0:]).T
                sub_j = numpy.delete(numpy.matrix(weights),j) * numpy.delete(numpy.matrix(df_train.iloc[i,0:]),j).T
                total = value * (y_i - sub_j)   
                c_j+= total
            c_j*=2
            print c_j
            weight_old= weights[j]
            weights[j] = soft(c_j / a_j, lamda / a_j)
            diff = max(diff,abs(weights[j] - weight_old))
        print weights
        print 'iteration ' + str(diff)

     
    print weights
    print 'agePct12t29 = ' + str(weights[df_train.columns.get_loc('agePct12t29')])
    print 'pctWSocSec = ' + str(weights[df_train.columns.get_loc('pctWSocSec')])
    print 'PctKids2Par = ' + str(weights[df_train.columns.get_loc('PctKids2Par')])
    print 'PctIlleg = ' + str(weights[df_train.columns.get_loc('PctIlleg')])
    print 'HousVacant = ' + str(weights[df_train.columns.get_loc('HousVacant')])

    pickle.dump(weights, open("weights.obj","wb"))


def soft(a,b):    
    return numpy.sign(a) * max(abs(a) - b, 0)
        

if __name__ == '__main__':
    main(sys.argv) 