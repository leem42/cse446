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

def main():
    eta= 0.00001
    train = pd.read_csv('HW2_training_data.csv').drop("label", axis=1)
    response = pd.read_csv('HW2_training_data.csv').iloc[0:,0]
    weights = np.zeros(len(train.columns))
    avg_loss = 0
    index = 1
    norms = []
    losses = []
    for iteration in range(10):
            for i in range(len(response)):
                actual = response[i]
                x_i = train.iloc[i,:]
                weightsCopy = weights.copy()
                for j in range(len(train.columns)):
                    x_ij = train.iloc[i,j]
                    partial_j = weights[j] - eta * x_ij * (indicator(actual) - prob_exp(weights, x_i))
                    weights[j] = partial_j 
                avg_loss+= (np.dot(weightsCopy, x_i) - actual) ** 2
                index+=1
                if(index >= 100 and index % 100 == 0):
                    losses.append(avg_loss / index)
                if(index % 500 == 0):
                    norms.append(np.linalg.norm(weights))
    
    print weights
    x = range(0,5000,100)
    matplotlib.pyplot.scatter(x,losses)
    matplotlib.pyplot.title("Average Loss For Eta = " + str(eta))
    matplotlib.pyplot.show()
     
    x_norms = range(0,10)
    matplotlib.pyplot.scatter(x_norms,norms)
    matplotlib.pyplot.title("Norms for W with Eta = " + str(eta))
    matplotlib.pyplot.show()
    
    
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