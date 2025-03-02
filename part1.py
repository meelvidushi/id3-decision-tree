import math
import numpy as np
from collections import Counter

#-----------------------------------------------
class Node:
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
        counter = Counter(Y)
        total = len(Y)
        e = -sum((count/total) * math.log2(count/total) for count in counter.values() if count > 0)
        return e

    #--------------------------
    @staticmethod
    def conditional_entropy(Y, X):
        '''
            Compute the conditional entropy of Y given X.
        '''
        counter = Counter(X)
        total = len(X)
        ce = sum((count/total) * Tree.entropy(Y[X == value]) for value, count in counter.items())
        return ce
    
    #--------------------------
    @staticmethod
    def information_gain(Y, X):
        '''
            Compute the information gain of Y after splitting over attribute X.
        '''
        return Tree.entropy(Y) - Tree.conditional_entropy(Y, X)

    #--------------------------
    @staticmethod
    def best_attribute(X, Y):
        '''
            Find the best attribute to split the node.
        '''
        gains = [Tree.information_gain(Y, X[i]) for i in range(X.shape[0])]
        return np.argmax(gains)

    #--------------------------
    @staticmethod
    def split(X, Y, i):
        '''
            Split the node based on the i-th attribute.
        '''
        values = np.unique(X[i])
        C = {}
        for v in values:
            mask = (X[i] == v)
            C[v] = Node(X[:, mask], Y[mask])
        return C

    #--------------------------
    @staticmethod
    def stop1(Y):
        '''
            Stop condition: all instances have the same label.
        '''
        return len(set(Y)) == 1

    #--------------------------
    @staticmethod
    def stop2(X):
        '''
            Stop condition: all instances have the same attribute values.
        '''
        return np.all(X == X[:, [0]])

    #--------------------------
    @staticmethod
    def most_common(Y):
        '''
            Get the most-common label from the list Y.
        '''
        return Counter(Y).most_common(1)[0][0]

    #--------------------------
    @staticmethod
    def build_tree(t):
        '''
            Recursively build tree nodes.
        '''
        if Tree.stop1(t.Y):
            t.isleaf = True
            t.p = t.Y[0]
            return

        if Tree.stop2(t.X):
            t.isleaf = True
            t.p = Tree.most_common(t.Y)
            return

        t.i = Tree.best_attribute(t.X, t.Y)
        t.C = Tree.split(t.X, t.Y, t.i)

        for child in t.C.values():
            Tree.build_tree(child)

    #--------------------------
    @staticmethod
    def train(X, Y):
        '''
            Train a decision tree.
        '''
        root = Node(X, Y)
        Tree.build_tree(root)
        return root

    #--------------------------
    @staticmethod
    def inference(t, x):
        '''
            Infer the label of a single instance using the decision tree.
        '''
        if t.isleaf:
            return t.p

        value = x[t.i]
        if value in t.C:
            return Tree.inference(t.C[value], x)
        return t.p

    #--------------------------
    @staticmethod
    def predict(t, X):
        '''
            Predict labels for a dataset.
        '''
        return np.array([Tree.inference(t, X[:, i]) for i in range(X.shape[1])])

    #--------------------------
    @staticmethod
    def load_dataset(filename='data1.csv'):
        '''
            Load dataset from a CSV file.
        '''
        data = np.genfromtxt(filename, delimiter=',', dtype=str)
        Y = data[1:, 0]
        X = data[1:, 1:].T
        return X, Y