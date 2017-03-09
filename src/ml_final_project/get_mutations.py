'''
Created on Mar 5, 2017

@author: Michael Lee, Jonathan Wolf

get_mutations is a program to properly format the data gathered from the ISB to attain a list of mutations 
for each patient, it goes through the data set and makes it a python dictionary.
References

----------
        
'''
import pandas as pd
import pickle
def main():

    mutations = pd.read_csv('mutations.csv')
    outgoing_mutations = {}    
    rows = len(mutations.iloc[:,0])
    
#############################################################
#    For Each patient in the data table gather the mutation listed and its
#    location in their genome
#
#    Result dictionary will contain infomration:
#    patient_id { chromosome: [list of mutations for specifed chromosome] }
###################################################
    for i in range(rows):
        patient = mutations.iloc[i,0]
        chromsome = mutations.iloc[i,2]
        actualLocation = mutations.iloc[i,11].split(":")[1]
        x =  len(actualLocation)
        actualLocation = actualLocation[0:x - 3]
        
        if '_' in actualLocation:
            actualLocation = actualLocation.split('_')[0]
        if 'd' in actualLocation:
            actualLocation = int(actualLocation[0:len(actualLocation) - 1]) 
        else:
            actualLocation = int(actualLocation)
            
        if(patient not in outgoing_mutations):
            outgoing_mutations[patient] = {}
            
        if(chromsome not in outgoing_mutations[patient]):
            outgoing_mutations[chromsome] = {}
            
        if(chromsome not in outgoing_mutations[patient]):
            outgoing_mutations[patient][chromsome] = []
            
        outgoing_mutations[patient][chromsome].append(actualLocation)
    
    output = open('patient_mutations.obj', 'wb')
    pickle.dump(outgoing_mutations, output)       
    
if __name__ == '__main__':
    main()