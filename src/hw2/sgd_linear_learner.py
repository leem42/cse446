'''
Created on Jan 30, 2017

@author: leem42
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
            for j in range(len(train.columns)):
                x_ij = train.iloc[i,j]
                partial_j = -2 * x_ij * (actual - np.dot(weights,x_i)) 
                weights[j] = weights[j] + ( partial_j * eta)
            avg_loss+= (np.dot(weights, x_i) - actual) ** 2
            index+=1
            #if(index >= 100 and index % 100 == 0):
              #  print 'iteration ' + str(index)
             #   print avg_loss / index
            if(index % 500 == 0):
                print np.linalg.norm(weights)

    print index
    print weights

if __name__ == '__main__':
    main()