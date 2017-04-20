import numpy as np
from itertools import product
import random
import gzip
import pickle
np.random.seed(seed=42)

def old_permutation(X, n, n_perm = 250):
    """ Old version of image permutations. """
    from_ind = [np.random.randint(low=0, high=784, size=n_perm) for i in range(n)]
    to_ind = [np.random.randint(low=0, high=784, size=n_perm) for i in range(n)]
    output = []
    for i in range(n):
        a = from_ind[i]
        b = to_ind[i]
        Xn = np.copy(X)
        for j in range(n_perm):
            tmp = Xn[:,a[j]]
            Xn[:,a[j]] = Xn[:,b[j]]
            Xn[:,b[j]] = tmp 
        output.append(Xn)
    return output

def permutation(X, n, n_perm = 250):
    """ New version of image permutations. """
    perms = random.sample(list(product(range(768), range(768))), n*n_perm)
    output = []
    for i in range(n):
        p = perms[i*n_perm:(i+1)*n_perm]
        Xn = np.copy(X)
        for j in range(n_perm):
            Xn[:,p[j][0]] = Xn[:,p[j][1]] 
            Xn[:,p[j][1]] = Xn[:,p[j][0]]  
        output.append(Xn)
    return output

def random_permutation(X, n, n_perm = 250):
    """ New version of image permutations. """
    Xn = np.copy(X)
    for i in range(X.shape[0]):
        np.random.shuffle(Xn[i,:])
    return [Xn]

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
        
def reformat(dataset, labels):
    labels = (np.arange(10) == labels[:,None]).astype(np.float32)
    return dataset, labels
        
class Permutation_batches(object):
    """ Loads data and initializes batch generators. """
    
    def __init__(self, batch_size, n, n_perm=250, version='new', one_hot=False):
        
        if version=='new':
            self.permutation = permutation
        if version=='old':
            self.permutation = old_permutation
        if version=='inv':
            self.permutation = random_permutation

        self.n = n
        self.n_perm = n_perm
        self.batch_size = batch_size

        with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
            (X_train, y_train), (X_val, y_val), (X_test, y_test) =  pickle.load(f)
        
        if one_hot==True:
            self.train = reformat(X_train, y_train)
            self.val = reformat(X_val, y_val)
            self.test = reformat(X_test, y_test)
        else:    
            self.train = (X_train, y_train)
            self.val = (X_val, y_val)
            self.test = (X_test, y_test)
        
        self.permutations = self.permutation(X_train, n, n_perm)
        self.batches = [batch_generator(X, self.train[1], batch_size=batch_size) for X in self.permutations]
       
        

    
    
    
    