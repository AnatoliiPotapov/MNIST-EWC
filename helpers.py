import os 
import sys
from urllib import urlretrieve
import numpy as np


def maybe_download(url, data_root, filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(data_root, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename) 
        filename, _ = urlretrieve(url + filename, dest_filename)
        print('\nDownload Complete!')
    
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
            'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename


def one_hot(dataset, labels, num_labels = 10):
    """ Transform X, y to X, y* where y* is one_hot encoded vector."""
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels


from ipywidgets import IntProgress
from IPython.display import display
import time


class Progress(object):
    """ Renders progress bar with time remaining in minutes and seconds."""
    def __init__(self, max):
        self.p = IntProgress(max = max)
        self.p.description = 'Running'
        self.stime = time.time()
        self.last_delta = 0
        self.max = max
        display(self.p)
    
    def update(self, step):
        self.p.value = step
        delta = int(time.time() - self.stime)
        if delta % 5 == 0 and delta > self.last_delta:
            self.last_delta = delta
            time_remaining = int(delta * (self.max - step) / step)
            self.p.description = '{0}:{1}'.format(time_remaining / 60, time_remaining % 60 )                   
            
            
var = 10
n_perm = 250

from itertools import product
import random


def old_permutation(X, n):
    """ Old version of image permutations. """
    from_ind = [np.random.randint(low=0, high=784, size=n_perm) for i in range(var)]
    to_ind = [np.random.randint(low=0, high=784, size=n_perm) for i in range(var)]
    output = []
    for i in range(n):
        a = from_ind[i]
        b = to_ind[i]
        Xn = np.copy(X)
        for j in range(n_perm):
            tmp = Xn[:,a[i]]
            Xn[:,a[i]] = Xn[:,b[i]]
            Xn[:,b[i]] = tmp 
        output.append(Xn)
    return output


def permutation(X, n):
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


    
    
    
    
    