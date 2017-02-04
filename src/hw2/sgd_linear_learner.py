'''
Created on Jan 30, 2017

@author: leem42
'''

import matplotlib
import pylab
import numpy as np
import pandas as pd
import math, sys

def main(args):
    if(len(args) != 3):
        print 'Error: Incorrect Parameters Entered'
        print 'Please run program like the following:'
        print '    python sgd_linear_learner.py 0.8 10'
        print 'first argument, (0.8 in the ex.)  is value for step size'
        print 'second argument, (10 in the ex.) is value for number of iterations'
        print 'After execution program will display three graphs: Average Loss, Magnitude Of Weights, and SSE'
        sys.exit(0)
        
    eta= float(args[1])
    ITERATIONS = int(args[2])
    
    train = pd.read_csv('HW2_training_data.csv').drop("label", axis=1)
    train['bias'] = np.ones(len(train.iloc[:,0])) 
    response = pd.read_csv('HW2_training_data.csv').iloc[0:,0]
    
    test = pd.read_csv('HW2_testing_data.csv').drop("label", axis=1)
    test['bias'] = np.ones(len(test.iloc[:,0]))
    test_response = pd.read_csv('HW2_testing_data.csv').iloc[0:,0]
    
    weights = np.zeros(len(train.columns))
    avg_loss = 0
    losses = []
    norms = []
    SSE = []
    index = 1
    
    
    for iteration in range(ITERATIONS):
        for i in range(len(response)):
            actual = response[i]
            x_i = train.iloc[i,:]
            weights = weights + (-2 * eta * x_i * ([actual] * 9 - np.round(np.dot(weights,x_i))))
            avg_loss+= (np.round(np.dot(weights, x_i)) - actual) ** 2
            if(index >= 100 and index % 100 == 0):
                losses.append(avg_loss / index)
                SSE.append(classifyPatients(test,weights, test_response, index))
            if(index % 500 == 0):
                norms.append(np.linalg.norm(weights))
            index+=1
    
    x = range(0,500 * ITERATIONS,100)
    matplotlib.pyplot.scatter(x,losses)
    matplotlib.pyplot.title("Average Loss For Eta = " + str(eta))
    matplotlib.pyplot.show()    
    
    x_norms = range(0,ITERATIONS)
    matplotlib.pyplot.scatter(x_norms,norms)
    matplotlib.pyplot.title("Norms for W with Eta = " + str(eta))
    matplotlib.pyplot.show()

    x = range(0,500 * ITERATIONS,100)
    matplotlib.pyplot.scatter(x,SSE)
    matplotlib.pyplot.title("SSE For Linear With Eta = " + str(eta))
    matplotlib.pyplot.show()


def classifyPatients(matrix, weights, actual, index):
    classification = np.dot(matrix,weights)
    classification = sum(actual - np.round(classification) )
    return classification
   



if __name__ == '__main__':
    main(sys.argv)