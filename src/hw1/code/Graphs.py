'''
Created on Jan 16, 2017

@author: leem42
'''

import matplotlib.pyplot, pickle
import pandas as pd


estimator = pickle.load( open("weights.obj","rb"))

df_train = pd.read_table('crime-test.txt')
df_test = pd.read_table('crime-train.txt')

# 2. (10 points) The regularization paths (in one plot) for the coefficients for input variables agePct12t29,
# pctWSocSec, PctKids2Par, PctIlleg, and HousVacant — use log(lamda) instead of lamda.

# 3. (4 points) A plot of log(lamda) against the squared error in the training data.


# 4. (4 points) A plot of log(lamda) against the squared error in the test data.

# 5. (3 points) A plot of lambda against the number of nonzero coefficients.

# 6. (2 points) A brief commentary on the task of selecting lambda




if __name__ == '__main__':
    pass