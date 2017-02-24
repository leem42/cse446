'''
Created on Feb 21, 2017

@author: leem42
'''

import matplotlib
import pylab
import numpy as np
import pandas as pd
import math, sys
from sklearn.model_selection import train_test_split

def main(args):
    args=[0,"0.0001","5"]
    if(len(args) != 3):
        print 'Error: Incorrect Parameters Entered'
        print 'Please run program like the following:'
        print '    python regression_learner.py 0.8 10'
        print 'first argument, [0.8 in the ex.] is value for step size'
        print 'second argument, [10 in the ex.]is value for number of iterations'
        print 'After execution program will display three graphs: Average Loss, Magnitude Of Weights, and SSE'
        sys.exit(0)
        
    eta= float(args[1])
    ITERATIONS = int(args[2])
    
    #Get Data
    df = pd.read_csv('response_modified.csv')
    train, test = train_test_split(df, test_size = 0.3)

    train['bias'] = np.ones(len(train.iloc[:,0])) 
    response = train.iloc[:,2]
    train =  train.drop("normalized_count", axis=1)
    train = train.drop("ParticipantBarcode",axis=1)
    train = train.drop("original_gene_symbol",axis=1)
    
    
    
    
    test['bias'] = np.ones(len(test.iloc[:,0]))
    test_response = test.iloc[:,2]
    test = test.drop("normalized_count",axis=1)
    test = test.drop("ParticipantBarcode",axis=1)
    test = test.drop("original_gene_symbol",axis=1)
    test.iloc[:,7] = pd.to_numeric(test.iloc[:,7], errors='coerce')
    
    weights = np.zeros(len(train.columns))
    avg_loss = 0
    losses = []
    norms = []
    SSE = []
    index = 1
    for iteration in range(ITERATIONS):
        print iteration
        for i in range(len(response)):
            actual = response.iloc[i]
            x_i = train.iloc[i,:].astype(float)
            weights = weights + (-.01 * eta * x_i * ([actual] * 15 - np.round(np.dot(weights,x_i))))
            print weights
            print np.dot(weights, x_i), actual
            avg_loss+= (np.round(np.dot(weights, x_i)) - actual) ** 2
            if(index >= 100 and index % 100 == 0):
                losses.append(avg_loss / index)
                SSE.append(error(test,weights, test_response, index))
            if(index % 500 == 0):
                print index
                norms.append(np.linalg.norm(weights))
            index+=1
 
   
    x = range(0,len(response) * ITERATIONS,100)
    x = x[0:len(x)-1]
    matplotlib.pyplot.scatter(x,losses)
    matplotlib.pyplot.title("Average Loss For Eta = " + str(eta))
    matplotlib.pyplot.show()    
    matplotlib.pyplot.savefig("avg_loss.png")
    
    x = range(0,len(response) * ITERATIONS,100)
    x = x[0:len(x)-1]
    matplotlib.pyplot.scatter(x,SSE)
    matplotlib.pyplot.title("SSE For Linear With Eta = " + str(eta))
    matplotlib.pyplot.show()
    matplotlib.pyplot.savefig("sse.png")
    
    x_norms = range(0,ITERATIONS)
    x_norms = x_norms[0:len(x_norms)-1]
    matplotlib.pyplot.scatter(x_norms,norms)
    matplotlib.pyplot.title("Norms for W with Eta = " + str(eta))
    matplotlib.pyplot.show()
    matplotlib.pyplot.savefig("norms.png")

   


def error(matrix, weights, actual, index):
    classification = np.dot(matrix,weights)
    classification = sum(np.abs(actual - classification))
    return classification

if __name__ == '__main__':
    main(sys.argv)