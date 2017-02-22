'''
Created on Jan 16, 2017

@author: leem42

Crime Predictor Program performs a lasso regression on a data set in csv format. It accepts four arguments such it can
be run from the command line as follows:
    python Crime_predictor.py 600 weights.obj crime-train.txt
    
    -600 is value of lamda
    -weights.obj represents the initial weights used for lamda, use 'zero' to set guess to all zeros, and 'normal' to set
    guess to normal distribution'
    -crime-train.txt represents the matrix X that will be read over
    
    output: weights.obj
                 
'''


from __future__ import division
from json.encoder import INFINITY
import pandas as pd 
import sys, os, numpy, matplotlib.pyplot, pickle, math


def main(args):

        print 'error: wrong number of parameters'
        print 'arg1 = lamda, ie. 600'
        print 'arg2= initial value of weights, use zeros to set guess as all zeros, normal for guess '
        print '      from normal distribution or a .obj file'
        print "arg3 = initial matrix X to use, ie. 'crime-train.txt' or 'crime-test.txt' "
        print "example:"
        print "python Crime_predictor.py 600 normal crime-train.txt"
        print 'algorithm prints our final solution for vector of weights w and also outputs the object in a pickled form'
        print 'ie. weights.obj will be outputted'
        sys.exit(0)
    
    lamda = float(args[1])
    if(args[2][len(args[2]) - 3:] == 'obj'):
        weights = pickle.load(open(args[2],"rb"))
    elif(args[2] == 'zeros'):
        weights = numpy.zeros(95)
    elif(args[2] == 'normal'):
        weights = numpy.random.normal(size=95)
    else:
        print 'Error: Either a .obj file, zeros or normal was not specified for initial guess'
        sys.exit(0)
        
    df_train = pd.read_table('crime-train.txt').drop("ViolentCrimesPerPop",axis=1)
    response = pd.read_table('crime-train.txt').iloc[0:,0]
    diff = 10e-5
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
     
    print weights
    pickle.dump(weights, open("weights.obj","wb"))

def soft(a,b):    
    return numpy.sign(a) * max(abs(a) - b, 0)

if __name__ == '__main__':
    main(sys.argv) 