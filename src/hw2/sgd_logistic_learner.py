'''
Created on Jan 30, 2017

@author: leem42
'''

import sys
import numpy as np
import pandas as pd
import math

def main():
    eta= 0.8
    train = pd.read_csv('HW2_training_data.csv').drop("label", axis=1)
    response = pd.read_csv('HW2_training_data.csv').iloc[0:,0]
    weights = np.zeros(len(train.columns))
    
    random_i = 0
    
    for iteration in range(10):
            for i in range(len(response)):
                for j in range(len(train.columns)):
                    random_i = np.random.randint(len(train.columns))
                    x_ij = train.iloc[random_i,j]
                    actual = response[random_i]
                    x_i = train.iloc[random_i,:]
                    partial_j = weights[j] - x_ij * (indicator(actual) - prob_exp(weights, x_i))
                    weights[j] = partial_j
             
    print weights
    
def indicator(a):
    if(a == 1):
        return 1 
    else:
        return 0

def prob_exp(a,b):
    dot = np.dot(a.T,b) * -1
    denom = 1 + math.exp(dot)
    return 1.0 / (denom)
        

if __name__ == '__main__':
    main()