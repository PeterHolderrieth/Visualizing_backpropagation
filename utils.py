import numpy as np 

def give_ranks_of_array(array):
    shape=array.shape
    array_flat=array.flatten()
    temp = array_flat.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array_flat))
    return ranks.reshape(shape)