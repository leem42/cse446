'''
Created on Jan 30, 2017

@author: leem42
'''

import sys
import numpy as np
import pandas as pd
import math
import matplotlib
import pylab
from numpy import size
from scipy.odr.odrpack import Output

def main():
    eta= 0.00001
    train = pd.read_csv('HW2_training_data.csv').drop("label", axis=1)
    train['bias'] = np.ones(len(train.iloc[:,0]))
    response = pd.read_csv('HW2_training_data.csv').iloc[0:,0]
    
    test = pd.read_csv('HW2_testing_data.csv').drop("label", axis=1)
    test['bias'] = np.ones(len(test.iloc[:,0]))
    test_response = pd.read_csv('HW2_testing_data.csv').iloc[0:,0]
    
    weights = np.zeros(len(train.columns))
    
    avg_loss = 0
    index = 1
    norms = []
    losses = []
    
    
    for iteration in range(10):
            for i in range(len(response)):
                actual = response[i] 
                x_i = train.iloc[i,:]
                prior = weights.copy()
                weights = weights - eta * (x_i * ([response[i]] * 9 - prob_exp(weights,x_i)))
                avg_loss+= (np.dot(prior, x_i) - actual) ** 2
                if(index >= 100 and index % 100 == 0):
                    losses.append(avg_loss * 1.0 / index)
                    classifyPatients(test,weights, test_response, index)
                if(index % 500 == 0):
                    norms.append(np.linalg.norm(weights))
                index+=1

    print weights
    x = range(0,5000,100)
    matplotlib.pyplot.scatter(x,losses)
    matplotlib.pyplot.title("Average Loss For Eta = " + str(eta))
    matplotlib.pyplot.show()
     
    x_norms = range(0,10)
    matplotlib.pyplot.scatter(x_norms,norms)
    matplotlib.pyplot.title("Norms for W with Eta = " + str(eta))
    matplotlib.pyplot.show()

def classifyPatients(matrix, weights, actual, index):
    classification = np.dot(matrix,weights) * -1
    classification = 1 + math.e ** classification
    classification = np.round(1.0 / classification)
    classification = abs(actual - classification)
    print  sum(classification)
   
    
def indicator(a):
    if(a == 1):
        return 1 
    else:
        return 0

def prob_exp(a,b):
    dot = np.dot(a.T,b) * -1
    denom = 1 + (math.e ** dot)
    return 1.0 / (denom)
        

if __name__ == '__main__':
    main()