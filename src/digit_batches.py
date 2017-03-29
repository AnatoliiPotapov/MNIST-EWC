import numpy as np
from itertools import product
import random
import gzip
import pickle
np.random.seed(seed=42)

def get_digit(X, y, digit):
    """ Get subset of dataset which corresponds to particular digit. """
    Xd = X[y == digit,]
    yd = y[y == digit]
    return [Xd, yd]
    
def batch_generator(X, y, batch_size):
    """ Yields new batch with .next() """
    l = X.shape[0]
    def another():
        indexes = np.random.randint(low=0, high=l, size=batch_size)
        return [
            X.take(indexes, axis=0),
            y[indexes]
        ]
            
    while True:
        yield another()
    
class Digits_batches(object):
    """ Loads data and initializes batch generators. """
    
    def __init__(self, spec_list, batch_size = 128):
        
        with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
            (X_train, y_train), (X_val, y_val), (X_test, y_test) =  pickle.load(f)
        
        self.train = (X_train, y_train)
        self.val = (X_val, y_val)
        self.test = (X_test, y_test)
        
        self.data = []
        
        for batch_spec in spec_list:
            
            Xl, yl = [], []
            for d in batch_spec:
                X, y = get_digit(X_train, y_train, d)
                Xl.append(X)
                yl.append(y)
        
            self.data.append([np.concatenate(Xl, axis=0), np.concatenate(yl, axis=0)])       
            self.batches = [batch_generator(D[0], D[1], batch_size=batch_size) for D in self.data]

    
    
    