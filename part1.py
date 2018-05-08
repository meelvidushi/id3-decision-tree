import math
import numpy as np
from collections import Counter

#-------------------------------------------------------------------------
'''
    Part 1: Decision Tree (with Discrete Attributes)
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `pytest -v test1.py` in the terminal.
'''
#-----------------------------------------------
class Node:
    '''
        Decision Tree Node (with discrete attributes)
    '''
    def __init__(self, X, Y, i=None, C=None, isleaf=False, p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C = C
        self.isleaf = isleaf
        self.p = p

#-----------------------------------------------
class Tree(object):
    '''
        Decision Tree using ID3 Algorithm
    '''
    #--------------------------
    @staticmethod
    def entropy(Y):
        '''
            Compute the entropy of a list of values.
        '''
        pass

    #--------------------------
    @staticmethod
    def conditional_entropy(Y, X):
        '''
            Compute the conditional entropy of Y given X.
        '''
        pass

    #--------------------------
    @staticmethod
    def information_gain(Y, X):
        '''
            Compute the information gain of Y after splitting over attribute X.
        '''
        pass

    #--------------------------
    @staticmethod
    def best_attribute(X, Y):
        '''
            Find the best attribute to split the node.
        '''
        pass

    #--------------------------
    @staticmethod
    def split(X, Y, i):
        '''
            Split the node based on the i-th attribute.
        '''
        pass

    #--------------------------
    @staticmethod
    def stop1(Y):
        '''
            Stop condition: all instances have the same label.
        '''
        pass

    #--------------------------
    @staticmethod
    def stop2(X):
        '''
            Stop condition: all instances have the same attribute values.
        '''
        pass

    #--------------------------
    @staticmethod
    def most_common(Y):
        '''
            Get the most-common label from the list Y.
        '''
        pass

    #--------------------------
    @staticmethod
    def build_tree(t):
        '''
            Recursively build tree nodes.
        '''
        pass

    #--------------------------
    @staticmethod
    def train(X, Y):
        '''
            Train a decision tree.
        '''
        pass

    #--------------------------
    @staticmethod
    def inference(t, x):
        '''
            Infer the label of a single instance using the decision tree.
        '''
        pass

    #--------------------------
    @staticmethod
    def predict(t, X):
        '''
            Predict labels for a dataset.
        '''
        pass

    #--------------------------
    @staticmethod
    def load_dataset(filename='data1.csv'):
        '''
            Load dataset from a CSV file.
        '''
        pass
