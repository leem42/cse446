'''
Created on Jan 30, 2017

@author: leem42
'''

import sys
import numpy as np
import pandas as pd
import math

def main():
    eta= 0.00001
    train = pd.read_csv('HW2_training_data.csv').drop("label", axis=1)
    response = pd.read_csv('HW2_training_data.csv').iloc[0:,0]
    weights = np.zeros(len(train.columns))
    avg_loss = 0
    index = 1
    
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
                    print 'iteration ' + str(index)
                    print avg_loss / index
                if(index % 500 == 0):
                    print np.linalg.norm(weights)
    
    print weights
    
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