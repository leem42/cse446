'''
Created on Feb 21, 2018

@author: leem42
'''

import matplotlib
import pylab
import numpy as np
import pandas as pd
import math, sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

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
    df = pd.read_csv('polished_output.csv')
    train, test = train_test_split(df, test_size = 0.2)
    
#     train['bias'] = np.ones(len(train.iloc[:,0])) 
    response = train.iloc[:,2]
    train =  train.drop("normalized_count", axis=1)
    train = train.drop("ParticipantBarcode",axis=1)
    train = train.drop("original_gene_symbol",axis=1)
    train = train.drop("pathologic_stage",axis=1)
    train = train.drop("BMI",axis=1)
#     train.iloc[:,8] = pd.to_numeric(train.iloc[:,8], errors='coerce')
    
    
#     output = (train.iloc[:,8] == 0)
# 
#     stage_one = (train.iloc[:,8] == 'Stage I')
#     train['stage_one'] = stage_one.astype(int)
#     
#     stage_oneA = (train.iloc[:,8] == 'Stage IA')
#     train['stage_oneA'] = stage_oneA.astype(int)
#     
#     stage_two= (train.iloc[:,8] == 'Stage II')
#     train['stage_two'] = stage_two.astype(int)
#     
#     stage_twoA= (train.iloc[:,8] == 'Stage IIA')
#     train['stage_twoA'] = stage_twoA.astype(int)
# 
#     stage_twoB= (train.iloc[:,8] == 'Stage IIB')
#     train['stage_twoB'] = stage_twoB.astype(int)
# 
#     stage_twoC= (train.iloc[:,8] == 'Stage IIC')
#     train['stage_twoC'] = stage_twoC.astype(int)
# 
#     stage_three= (train.iloc[:,8] == 'Stage III')
#     train['stage_three'] = stage_three.astype(int)
#     
#     stage_threeA= (train.iloc[:,8] == 'Stage IIIA')
#     train['stage_threeA'] = stage_threeA.astype(int)
#     
#     stage_threeB= (train.iloc[:,8] == 'Stage IIIB')
#     train['stage_threeB'] = stage_threeB.astype(int)
#     
#     stage_threeC= (train.iloc[:,8] == 'Stage IIIC')
#     train['stage_threeC'] = stage_threeC.astype(int)
# 
#     stage_four= (train.iloc[:,8] == 'Stage IV')
#     train['stage_four'] = stage_four.astype(int)
# 
#     stage_fourA= (train.iloc[:,8] == 'Stage IV')
#     train['stage_fourA'] = stage_fourA.astype(int)
#     
#     train.iloc[:,8] = output.astype(int)


#     test['bias'] = np.ones(len(test.iloc[:,0]))
    test_response = test.iloc[:,2]
    test = test.drop("ParticipantBarcode",axis=1)
    test = test.drop("normalized_count",axis=1)
    test = test.drop("original_gene_symbol",axis=1)
    test = test.drop("pathologic_stage",axis=1)
    test = test.drop("BMI",axis=1)
#     output = (test.iloc[:,8] == 0)
# 
#     stage_one = (test.iloc[:,8] == 'Stage I')
#     test['stage_one'] = stage_one.astype(int)
#     
#     stage_oneA = (test.iloc[:,8] == 'Stage IA')
#     test['stage_oneA'] = stage_oneA.astype(int)
#     
#     stage_two= (test.iloc[:,8] == 'Stage II')
#     test['stage_two'] = stage_two.astype(int)
#     
#     stage_twoA= (test.iloc[:,8] == 'Stage IIA')
#     test['stage_twoA'] = stage_twoA.astype(int)
# 
#     stage_twoB= (test.iloc[:,8] == 'Stage IIB')
#     test['stage_twoB'] = stage_twoB.astype(int)
# 
#     stage_twoC= (test.iloc[:,8] == 'Stage IIC')
#     test['stage_twoC'] = stage_twoC.astype(int)
# 
#     stage_three= (test.iloc[:,8] == 'Stage III')
#     test['stage_three'] = stage_three.astype(int)
#     
#     stage_threeA= (test.iloc[:,8] == 'Stage IIIA')
#     test['stage_threeA'] = stage_threeA.astype(int)
#     
#     stage_threeB= (test.iloc[:,8] == 'Stage IIIB')
#     test['stage_threeB'] = stage_threeB.astype(int)
#     
#     stage_threeC= (test.iloc[:,8] == 'Stage IIIC')
#     test['stage_threeC'] = stage_threeC.astype(int)
# 
#     stage_four= (test.iloc[:,8] == 'Stage IV')
#     test['stage_four'] = stage_four.astype(int)
# 
#     test['stage_fourA'] = (test.iloc[:,8] == 'Stage IV').astype(int)
# 
#     test.iloc[:,8] = output.astype(int)
    
    clf7 = Ridge(alpha= 200, normalize=True,fit_intercept=True)
    clf6 = Ridge(alpha=15,normalize=True,fit_intercept=True)
    clf5 = Ridge(alpha=10,normalize=True,fit_intercept=True)   
    clf4 = Ridge(alpha=5,normalize=True,fit_intercept=True)
    clf3 = Ridge(alpha=1,normalize=True,fit_intercept=True)
    clf2 = Ridge(alpha=0.5,normalize=True,fit_intercept=True)
    clf = Ridge(alpha=0.05,normalize=True,fit_intercept=True)
    
    weights =  [clf.fit(train, response).coef_, 
                clf2.fit(train, response).coef_,
                clf3.fit(train, response).coef_, 
                clf4.fit(train, response).coef_,
                clf5.fit(train, response).coef_,
                clf6.fit(train, response).coef_,
                clf7.fit(train, response).coef_]

    error = []
    for value in weights:
        error.append(sum(np.abs(test_response - np.dot(test,value))))
    print error
    x = ['.05','.5','.1','.5','10','15','20']
    matplotlib.pyplot.scatter(x,error)
    matplotlib.pyplot.title("Absolute Error Vs Lamda")
    matplotlib.pyplot.show()

def error(matrix, weights, actual, index):
    classification = np.dot(matrix,weights)
    classification = sum(np.abs(actual - classification))
    return classification

if __name__ == '__main__':
    main(sys.argv)