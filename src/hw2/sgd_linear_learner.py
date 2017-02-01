'''
Created on Jan 30, 2017

@author: leem42
'''

import matplotlib
import pylab
import numpy as np
import pandas as pd


def main():
        
    eta= 0.001
    train = pd.read_csv('HW2_training_data.csv').drop("label", axis=1)
    response = pd.read_csv('HW2_training_data.csv').iloc[0:,0]
    weights = np.zeros(len(train.columns))
    avg_loss = 0
    losses = []
    norms = []
    index = 1
    for iteration in range(10):
        for i in range(len(response)):
            actual = response[i]
            x_i = train.iloc[i,:]
            prior = weights.copy()
            for j in range(len(train.columns)):
                x_ij = train.iloc[i,j]
                partial_j = -2 * x_ij * (actual - np.dot(weights,x_i)) 
                weights[j] = weights[j] + ( partial_j * eta)
            avg_loss+= (np.dot(prior, x_i) - actual) ** 2
            if(index >= 100 and index % 100 == 0):
                losses.append(avg_loss / index)
            if(index % 500 == 0):
                norms.append(np.linalg.norm(weights))
            index+=1
 
    x = range(0,5000,100)
    matplotlib.pyplot.scatter(x,losses)
    matplotlib.pyplot.title("Average Loss For Eta = " + str(eta))
    matplotlib.pyplot.show()    
    
    x_norms = range(0,10)
    matplotlib.pyplot.scatter(x_norms,norms)
    matplotlib.pyplot.title("Norms for W with Eta = " + str(eta))
    matplotlib.pyplot.show()
        
if __name__ == '__main__':
    main()