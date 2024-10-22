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
import pickle
import re
import pprint

def main(args):
    args=[0,"0.0001","5"]
    if(len(args) != 3):
        print 'Error: Incorrect Parameters Entered'
        print 'Please run program like the following:'
        print '    python regression_learner.py 0.10 10'
        print 'first argument, [0.10 in the ex.] is value for step size'
        print 'second argument, [10 in the ex.]is value for number of iterations'
        print 'After execution program will display three graphs: Average Loss, Magnitude Of Weights, and SSE'
        sys.exit(0)
        
#     eta= float(args[1])
#     ITERATIONS = int(args[2])
#     
# #     #Get Data
#     df = pd.read_csv('polished_output.csv')
#     train, test = train_test_split(df, test_size = 0.2)
#      
#     train['bias'] = np.ones(len(train.iloc[:,0])) 
#     response = train.iloc[:,2]
#     train =  train.drop("normalized_count", axis=1)
# #     train = train.drop("ParticipantBarcode",axis=1)
# #     train = train.drop("original_gene_symbol",axis=1)      
#        
#     output = (train.iloc[:,10] == 0)
#   
#     stage_one = (train.iloc[:,10] == 'Stage I')
#     train['stage_one'] = stage_one.astype(int)
#       
#     stage_oneA = (train.iloc[:,10] == 'Stage IA')
#     train['stage_oneA'] = stage_oneA.astype(int)
#       
#     stage_two= (train.iloc[:,10] == 'Stage II')
#     train['stage_two'] = stage_two.astype(int)
#       
#     stage_twoA= (train.iloc[:,10] == 'Stage IIA')
#     train['stage_twoA'] = stage_twoA.astype(int)
#   
#     stage_twoB= (train.iloc[:,10] == 'Stage IIB')
#     train['stage_twoB'] = stage_twoB.astype(int)
#   
#     stage_twoC= (train.iloc[:,10] == 'Stage IIC')
#     train['stage_twoC'] = stage_twoC.astype(int)
#   
#     stage_three= (train.iloc[:,10] == 'Stage III')
#     train['stage_three'] = stage_three.astype(int)
#       
#     stage_threeA= (train.iloc[:,10] == 'Stage IIIA')
#     train['stage_threeA'] = stage_threeA.astype(int)
#       
#     stage_threeB= (train.iloc[:,10] == 'Stage IIIB')
#     train['stage_threeB'] = stage_threeB.astype(int)
#       
#     stage_threeC= (train.iloc[:,10] == 'Stage IIIC')
#     train['stage_threeC'] = stage_threeC.astype(int)
#   
#     stage_four= (train.iloc[:,10] == 'Stage IV')
#     train['stage_four'] = stage_four.astype(int)
#   
#     stage_fourA= (train.iloc[:,10] == 'Stage IV')
#     train['stage_fourA'] = stage_fourA.astype(int)
#       
#     train.iloc[:,10] = output.astype(int)
#    
#    
#     test['bias'] = np.ones(len(test.iloc[:,0]))
#     test_response = test.iloc[:,2]
# #     test = test.drop("ParticipantBarcode",axis=1)
#     test = test.drop("normalized_count",axis=1)
# #     test = test.drop("original_gene_symbol",axis=1)
#     output = (test.iloc[:,10] == 0)
#   
#     stage_one = (test.iloc[:,10] == 'Stage I')
#     test['stage_one'] = stage_one.astype(int)
#       
#     stage_oneA = (test.iloc[:,10] == 'Stage IA')
#     test['stage_oneA'] = stage_oneA.astype(int)
#       
#     stage_two= (test.iloc[:,10] == 'Stage II')
#     test['stage_two'] = stage_two.astype(int)
#       
#     stage_twoA= (test.iloc[:,10] == 'Stage IIA')
#     test['stage_twoA'] = stage_twoA.astype(int)
#   
#     stage_twoB= (test.iloc[:,10] == 'Stage IIB')
#     test['stage_twoB'] = stage_twoB.astype(int)
#   
#     stage_twoC= (test.iloc[:,10] == 'Stage IIC')
#     test['stage_twoC'] = stage_twoC.astype(int)
#   
#     stage_three= (test.iloc[:,10] == 'Stage III')
#     test['stage_three'] = stage_three.astype(int)
#       
#     stage_threeA= (test.iloc[:,10] == 'Stage IIIA')
#     test['stage_threeA'] = stage_threeA.astype(int)
#       
#     stage_threeB= (test.iloc[:,10] == 'Stage IIIB')
#     test['stage_threeB'] = stage_threeB.astype(int)
#       
#     stage_threeC= (test.iloc[:,10] == 'Stage IIIC')
#     test['stage_threeC'] = stage_threeC.astype(int)
#   
#     stage_four= (test.iloc[:,10] == 'Stage IV')
#     test['stage_four'] = stage_four.astype(int)
#   
#     test['stage_fourA'] = (test.iloc[:,10] == 'Stage IV').astype(int)
#   
#     test.iloc[:,10] = output.astype(int)
#      
#     locations = pd.read_table('chromosome_location.txt')
#     ####################################
#     #    preprocess chromsome          #
#     ####################################
#     num_rows = len(locations.iloc[:,0])
#     for row in range(num_rows):
#         new_location = locations.iloc[row,0].split("chr")[1] + locations.iloc[row,3]
#         locations.iloc[row,3] = new_location
#      
#     ####################################
#     #Add Gene Location to each patient #
#     ####################################
#     samples = len(train.iloc[:,0])
#     genes = pd.read_table('total_genes.txt')
#     gene_locations= []
#     for row in range(0,samples):
#         gene = train.iloc[row,1]
#         relative_loc = get_gene_relaitve(gene, genes)
#         actual_location = get_actual_location(relative_loc, locations)
#         chromosome = actual_location[0]
#         start = actual_location[1]
#         end = actual_location[2]
#         info = [chromosome,start,end]
#         gene_locations.append(info)
#     train['locations'] = gene_locations
#      
#     samples = len(test.iloc[:,0])
#     gene_locations_test = []
#     for row in range(0,samples):
#         gene = test.iloc[row,1]
#         relative_loc = get_gene_relaitve(gene, genes)
#         actual_location = get_actual_location(relative_loc, locations)
#         chromosome = actual_location[0]
#         start = actual_location[1]
#         end = actual_location[2]
#         info = [chromosome,start,end]
#         gene_locations_test.append(info)
#     test['locations'] = gene_locations_test
    
    #######################################
    #    Save our data set                #
    #######################################

    
    #### Save the training data table and its response #####
#     train_file = open('train2.obj', 'wb')
#     pickle.dump(train,train_file)
#     
#     train_response_file = open('response2.obj','wb')
#     pickle.dump(response, train_response_file)
#     
#     #### Save the testing data and its response ####
#     test_file = open('test_response2.obj','wb')
#     pickle.dump(test,test_file)
#     
#     test_response_file = open('test2.obj', 'wb')
#     pickle.dump(test_response,test_response_file)
    
    ######################################
#         Open our data set                #
    ######################################        
#     train_file = open('train2.obj', 'rb')
#     train = pickle.load(train_file)
#           
#     train_file = open('response2.obj', 'rb')
#     response = pickle.load(train_file)
#       
#     train_file = open('test_response2.obj', 'rb')
#     test = pickle.load(train_file)
#     
#     train_file = open('test2.obj', 'rb')
#     test_response = pickle.load(train_file)
      
    ###################################
    ##### Find Nearest Mutation     ##
    ##################################
#     pkl_file = open('patient_mutations.obj', 'rb')
#     mutations = pickle.load(pkl_file)
#     samples = len(train.iloc[:,0])
#     new_mutations = [0] * samples
#     train['NearestMutation'] = new_mutations
      
#     missing_cr = set()
#     missing_id = set()   
#      
#     new_response = []
#     for row in range(0,samples):
#         id_name = train.iloc[row,0]
#         info = train.iloc[row,:]['locations']
#         chromosome = info[0].split('chr')[1]
#         start = info[1]
#         end = info[2]
#         nearest = get_nearest(id_name,chromosome,start,end,mutations,train,row)
#         if(nearest == 'No ID'):
#             missing_id.add(id_name)
#         else:
#             new_response.append(response.iloc[row])
#             missing_cr.add(id_name)
#             train.iloc[row,25] = nearest
#     response = new_response 
#     
#     ### Will just remove the ones with no id
#     print '>>> No ID'
#     for value in missing_id:
#         train = train[train.ParticipantBarcode != value]
#     
#     print
#     print '>>> No Chromosome'
#     for value in missing_cr:
#         print value, pprint.pprint(mutations[value])
#   
#       
#     samples = len(test.iloc[:,0])
#     new_mutations = [0] * samples
#     test['NearestMutation'] = new_mutations
#     test_new_response = []
#     missing_cr = set()
#     missing_id = set()
#     for row in range(0,samples):   
#         id_name = test.iloc[row,0]
#         info = test.iloc[row,:]['locations']
#         chromosome = info[0].split('chr')[1]
#         start = info[1]
#         end = info[2]
#         nearest = get_nearest(id_name,chromosome,start,end,mutations,test,row)
#         if(nearest == 'No ID'):
#             missing_id.add(id_name)
#         else:
#             missing_cr.add(id_name)
#             test_new_response.append(test_response.iloc[row])
#             test.iloc[row,25]= nearest
#             print row
#       
#     test_response = test_new_response
#     ### Will just remove the ones with no id
#     print '>>> No ID'
#     for value in missing_id:
#         test = test[test.ParticipantBarcode != value]
#       
#     print
#     print '>>> No Chromosome'
#     for value in missing_cr:
#         print value, pprint.pprint(mutations[value])
        
#######################################
#    Save our data set                #
#######################################
    
#     train_file = open('train_with_nearest.obj', 'wb')
#     pickle.dump(train,train_file)
#      
#     train_file = open('response_train_updated.obj', 'wb')
#     pickle.dump(response,train_file)
#      
# #     #### Save the testing data and its response ####
#     test_file = open('test_with_nearest.obj','wb')
#     pickle.dump(test,test_file)
#      
#     test_file = open('response_test_updatedt.obj','wb')
#     pickle.dump(test_response,test_file)
#     sys.exit(0)
#     #######################################
#     #    Open our data set                #
#     #######################################        
    train_file = open('train_with_nearest.obj', 'rb')
    train = pickle.load(train_file)
          
    train_file = open('response_train_updated.obj', 'rb')
    response = pickle.load(train_file)
      
    train_file = open('test_with_nearest.obj', 'rb')
    test = pickle.load(train_file)
      
    train_file = open('response_test_updatedt.obj', 'rb')
    test_response = pickle.load(train_file)  
     
    train = train.drop("ParticipantBarcode",axis=1)
    train = train.drop("original_gene_symbol",axis=1) 
    train = train.drop("locations",axis=1)
    
    test = test.drop("ParticipantBarcode",axis=1)
    test = test.drop("original_gene_symbol",axis=1) 
    test = test.drop("locations",axis=1)

    
    clf7 = Ridge(alpha= 200)
    clf6 = Ridge(alpha=15)
    clf5 = Ridge(alpha=10)   
    clf4 = Ridge(alpha=5)
    clf3 = Ridge(alpha=1)
    clf2 = Ridge(alpha=0.5)
    clf = Ridge(alpha=0.05)
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

    predictions = zip(test_response, np.dot(test,weights[0]))
    print 'hi'
    for value in predictions:
        print value

def get_gene_relaitve(name,table):
    if (name == 'MGC16121'):
        return 'Xq26.3'
    if(name == 'C20orf111'):
        return '20q13.12'
    if (name == 'FLJ39653'):
        return '4p15.32'
    if(name == 'LOC728392'):
        return '17p13.2'
    return table.iloc[table['symbol'][table['symbol'] == name].index[0],:]['location']
  

### returns the [chromosome, start, end] end position
def get_actual_location(relative,table):
    out = ''
    copy = []
    if('-' in relative):
        split = relative.split("-")
        head = split[0]
        tail =  re.split('p|q',head)[0] + split[1]
        start = table.iloc[table.iloc[:,3][table.iloc[:,3] == head].index[0],:]
        end = table.iloc[table.iloc[:,3][table.iloc[:,3] == tail].index[0],:][2]
        return [start[0],start[1],end]
    else:
        out = table.iloc[table.iloc[:,3][table.iloc[:,3] == relative].index[0],:]
        return  [out[0],out[1],out[2]]

def get_nearest(id,chromosome,start,end,mutations,table,row):
    mid  = 0
    if(id in mutations):
        if(chromosome in mutations[id]):
            mid = np.array([start] * len(mutations[id][chromosome]))
        else:
            return 1000000000
    else:
        return 'No ID'
    
    list_mutations = np.array(sorted(mutations[id][chromosome]))
    shortest = np.min(list_mutations - mid)
    return shortest
    

def error(matrix, weights, actual, index):
    classification = np.dot(matrix,weights)
    classification = sum(np.abs(actual - classification))
    return classification

if __name__ == '__main__':
    main(sys.argv)