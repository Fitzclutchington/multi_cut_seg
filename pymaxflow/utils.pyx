from itertools import combinations
import numpy as np
from scipy.misc import comb
from sklearn.linear_model import LogisticRegression

cimport numpy as np
cimport cython

cdef void c_weight_addition(np.ndarray[dtype=np.int32_t, ndim=1, negative_indices=False] graph_indices,
                            np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False]r_weights,
                            boundary_weight_dict,
                            np.ndarray[dtype=np.int32_t, ndim=1, negative_indices=False]labels
                                        ,int alpha,int beta,int width, int height):
    
    cdef int left, right, up, down, node, size
    cdef int N = graph_indices.size
    cdef float cost = 0
    for i in range(N):
        node = graph_indices[i]      
        left = node-1
        right = node+1
        up = node-width
        down = node+width
        
        size = width*height
        if left > -1:
            if node % width != 0:
                if labels[left] != alpha and labels[left] != beta:
                    try:
                        cost = boundary_weight_dict[(node,left)]
                    except KeyError:
                        cost = boundary_weight_dict[(left,node)]                    
                    r_weights[node][alpha] += cost
                    r_weights[node][beta] += cost
                    

        if right < size:
            if right % width != 0:
                if labels[right] != alpha and labels[right] != beta:
                    try:
                        cost = boundary_weight_dict[(node,right)]
                    except KeyError:
                        cost = boundary_weight_dict[(right,node)]                    
                    r_weights[node][alpha] += cost
                    r_weights[node][beta] += cost

        if up > -1:        
            if labels[up] != alpha and labels[up] != beta:
                try:
                    cost = boundary_weight_dict[(node,up)]
                except KeyError:
                    cost = boundary_weight_dict[(up,node)]                    
                r_weights[node][alpha] += cost
                r_weights[node][beta] += cost

        if down < size:        
            if labels[down] != alpha and labels[down] != beta:
                try:                
                    cost = boundary_weight_dict[(node,down)]
                except KeyError:
                    cost = boundary_weight_dict[(down,node)]                    
                r_weights[node][alpha] += cost
                r_weights[node][beta] += cost

cdef void c_weight_addition_with_count(np.ndarray[dtype=np.int32_t, ndim=1, negative_indices=False] graph_indices,
                            np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False]r_weights,
                            boundary_weight_dict,
                            np.ndarray[dtype=np.int32_t, ndim=1, negative_indices=False]labels
                                        ,int alpha,int beta,int width, int height, int count):
    
    cdef int left, right, up, down, node, size
    cdef int N = graph_indices.size
    cdef float cost = 0
    for i in range(N):
        node = graph_indices[i]      
        left = node-1
        right = node+1
        up = node-width
        down = node+width
        
        size = width*height
        if left > -1:
            if node % width != 0:
                if labels[left] != alpha and labels[left] != beta:
                    try:
                        cost = boundary_weight_dict[(node,left)][count]
                    except KeyError:
                        cost = boundary_weight_dict[(left,node)][count]                    
                    r_weights[node][alpha] += cost
                    r_weights[node][beta] += cost
                    

        if right < size:
            if right % width != 0:
                if labels[right] != alpha and labels[right] != beta:
                    try:
                        cost = boundary_weight_dict[(node,right)][count]
                    except KeyError:
                        cost = boundary_weight_dict[(right,node)][count]                    
                    r_weights[node][alpha] += cost
                    r_weights[node][beta] += cost

        if up > -1:        
            if labels[up] != alpha and labels[up] != beta:
                try:
                    cost = boundary_weight_dict[(node,up)][count]
                except KeyError:
                    cost = boundary_weight_dict[(up,node)][count]                    
                r_weights[node][alpha] += cost
                r_weights[node][beta] += cost

        if down < size:        
            if labels[down] != alpha and labels[down] != beta:
                try:                
                    cost = boundary_weight_dict[(node,down)][count]
                except KeyError:
                    cost = boundary_weight_dict[(down,node)][count]                    
                r_weights[node][alpha] += cost
                r_weights[node][beta] += cost

def calculate_boundary_stats(brush_strokes,num_samples):
    '''
    returns mean and std of the difference between random samples from 
    brush strokes
    '''
    num_classes = len(brush_strokes)
    com_list = combinations(range(num_classes),2)
    means = np.zeros((comb(num_classes,2)))
    std = np.zeros((comb(num_classes,2)))
    for j,com in enumerate(com_list):
        diffs = np.zeros((num_samples ** 2))
        a_samples = np.random.choice(brush_strokes[com[0]][:,0],num_samples,replace=False)
        b_samples = np.random.choice(brush_strokes[com[1]][:,0],num_samples,replace=False)
        for i in range(num_samples):
            start = i*num_samples
            end = start +num_samples
            diffs[start:end] = np.abs(np.subtract(np.roll(a_samples,i),b_samples))
        means[j] = np.mean(diffs)
        std[j] = np.std(diffs)

    return (means,std)


def calculate_boundary_stats_lgr(brush_strokes,num_samples):
    '''
    returns mean and std of the difference between random samples from
    brush strokes
    '''
    num_classes = len(brush_strokes)
    com_list = combinations(range(num_classes),2)
    sample_mat = np.zeros((comb(num_classes,2) * num_samples**2))
    labels_mat = np.zeros((comb(num_classes,2) * num_samples**2))
    for j,com in enumerate(com_list):
        diffs = np.zeros((num_samples ** 2))
        a_samples = np.random.choice(brush_strokes[com[0]][:,0],num_samples,replace=False)
        b_samples = np.random.choice(brush_strokes[com[1]][:,0],num_samples,replace=False)
        for i in range(num_samples):
            start = i*num_samples
            end = start +num_samples
            diffs[start:end] = np.abs(np.subtract(np.roll(a_samples,i),b_samples))
        start = j*num_samples**2
        end = start + num_samples**2
        sample_mat[start:end] = diffs
        labels_mat[start:end] = j
            
    lgr = LogisticRegression()
    lgr.fit(sample_mat.reshape((sample_mat.size,1)),labels_mat)
    return lgr
        
def non_graph_weight_addition(graph_indices,r_weights,boundary_weight_dict,labels,alpha,beta,width,height):
    """Computes n!"""
    return c_weight_addition(graph_indices,r_weights,boundary_weight_dict,labels,alpha,beta,width,height)

def non_graph_weight_addition_with_count(graph_indices,r_weights,boundary_weight_dict,labels,alpha,beta,width,height,count):
    """Computes n!"""
    return c_weight_addition_with_count(graph_indices,r_weights,boundary_weight_dict,labels,alpha,beta,width,height,count)


    