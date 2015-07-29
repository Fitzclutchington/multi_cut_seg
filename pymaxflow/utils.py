from itertools import combinations
import numpy as np
from scipy.misc import comb
def non_graph_weight_addition(graph_indices,r_weights,boundary_weight_dict,labels,alpha,beta,width,height,count):
    for i in graph_indices:
                
        left = i-1
        right = i+1
        up = i-width
        down = i+width
        
        size = width*height
        if left > -1:
            if i % width != 0:
                if labels[left] != alpha and labels[left] != beta:
                    try:
                        r_weights[i][alpha] += boundary_weight_dict[(i,left)][count]
                        r_weights[i][beta] += boundary_weight_dict[(i,left)][count]
                    except KeyError:
                        r_weights[i][alpha] += boundary_weight_dict[(left,i)][count]
                        r_weights[i][beta] += boundary_weight_dict[(left,i)][count]

        if right < size:
            if right % width != 0:
                if labels[right] != alpha and labels[right] != beta:
                    try:
                        r_weights[i][alpha] += boundary_weight_dict[(i,right)][count]
                        r_weights[i][beta] += boundary_weight_dict[(i,right)][count]
                    except KeyError:
                        r_weights[i][alpha] += boundary_weight_dict[(right,i)][count]
                        r_weights[i][beta] += boundary_weight_dict[(right,i)][count]

        if up > -1:        
            if labels[up] != alpha and labels[up] != beta:
                try:
                    r_weights[i][alpha] += boundary_weight_dict[(i,up)][count]
                    r_weights[i][beta] += boundary_weight_dict[(i,up)][count]
                except KeyError:
                    r_weights[i][alpha] += boundary_weight_dict[(up,i)][count]
                    r_weights[i][beta] += boundary_weight_dict[(up,i)][count]

        if down < size:        
            if labels[down] != alpha and labels[down] != beta:
                try:
                    r_weights[i][alpha] += boundary_weight_dict[(i,down)][count]
                    r_weights[i][beta] += boundary_weight_dict[(i,down)][count]
                except KeyError:
                    r_weights[i][alpha] += boundary_weight_dict[(down,i)][count]
                    r_weights[i][beta] += boundary_weight_dict[(down,i)][count]

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
        
        
        



    