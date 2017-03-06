'''
Created on Mar 4, 2017

@author: leem42
'''

import pickle

class Node(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.feature_index = None
        self.feature_value = None
        self.classification = None
    
    def print_me(self,level):
        if(self == None):
            print "None"
        out = str(self.feature_index) + '  Value: ' + str(self.feature_value) 
        print "\t"*level+str(out)+"\n"
        if(self.left == None):
            print "\t"*level+str("None")+"\n"
        elif(self.right == None):
            print "\t"*level+str("None")+"\n"
        else:
            self.left.print_me(level+1)
            self.right.print_me(level + 1)

def main():
    
    train_file = open('trained_root.obj', 'rb')
    root = pickle.load(train_file)
    
    root.print_me(0)
    


if __name__ == '__main__':
    main()