'''
Created on Jan 30, 2017

@author: leem42
'''

import numpy as np
import pandas as pd

def main():
        
    eta= 0.00001
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
                    partial_j = -2 * x_ij * (actual - np.dot(weights,x_i)) * eta
                    weights[j] = partial_j
    print weights

if __name__ == '__main__':
    main()