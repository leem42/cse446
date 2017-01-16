'''
Created on Jan 16, 2017

@author: leem42
'''

import pandas as pd 
import sys, os


from json.encoder import INFINITY

def main(args):
    
    
    lamda = args[1]
    response = args[2]
    matrix = args[3]
    initial_weights = args[4]
    output = 0

    df_train = pd.read_table('crime-test.txt')
    df_test = pd.read_table('crime-train.txt')


    diff = INFINITY
    
    

if __name__ == '__main__':
    main(sys.argv)