'''
Created on Jan 16, 2017

@author: leem42

   
                 
'''


from __future__ import division
from json.encoder import INFINITY

import pandas as pd 
import sys, os, numpy, matplotlib.pyplot, pickle, math



def main(args):
    lamda = 300
    print lamda
    weights = pickle.load(open("weights.obj","rb"))
    df_train = pd.read_table('crime-train.txt').drop("ViolentCrimesPerPop",axis=1)
    response = pd.read_table('crime-train.txt').iloc[0:,0]
    weights = numpy.zeros(95)
    diff = 10e-5
    
    while( diff > 10e-6 ):
        diff = None
        max_c = 0
        for j in range(len(df_train.columns) ):
            # accessing feature j of design matrix df_train
            a_j = 2 * numpy.sum(df_train[df_train.columns[j]].values**2)
            c_j = 0
            fast_j = numpy.sum(df_train.iloc[:,j].values*(response[:] - numpy.dot(df_train, weights) + weights[j] * df_train.iloc[:,j].values))
#             for i,value in enumerate(df_train[df_train.columns[j]].values):
#                 ## multiple our jth-weight times the data point at i
#                 #sub_j = numpy.dot(numpy.delete(weights,j ), numpy.delete(df_train.iloc[i,0:].as_matrix(),j))
#                 y_i = response[j]
#                 sub_j = numpy.dot(weights, df_train.iloc[i,0:].values) 
#                 total = value * (y_i - sub_j + value * weights[j]) 
#                 c_j+= total
            c_j*=2
            fast_j*=2
            print 'fast ' + str(fast_j)
#             print 'c_j ' + str(c_j)
            weight_old= weights[j]
            weights[j] = soft(fast_j / a_j, lamda / a_j)
            diff = max(diff,abs(weights[j] - weight_old))
        print weights
        print 'c has a max ' + str(max_c)
        print 'iteration ' + str(diff)

     
    print 'agePct12t29 = ' + str(weights[df_train.columns.get_loc('agePct12t29')])
    print 'pctWSocSec = ' + str(weights[df_train.columns.get_loc('pctWSocSec')])
    print 'PctKids2Par = ' + str(weights[df_train.columns.get_loc('PctKids2Par')])
    print 'PctIlleg = ' + str(weights[df_train.columns.get_loc('PctIlleg')])
    print 'HousVacant = ' + str(weights[df_train.columns.get_loc('HousVacant')])

    pickle.dump(weights, open("weights.obj","wb"))


def soft(a,b):    
    #print abs(a) - b
    return numpy.sign(a) * max(abs(a) - b, 0)
        

if __name__ == '__main__':
    main(sys.argv) 