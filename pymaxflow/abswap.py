import numpy as np
from scipy.misc import imread, imsave
from pymaxflow import PyGraph
import sys
import time
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import math

def neighbor_cost_boykov(p1, p2, alpha):
    pdiff = np.absolute(np.subtract(p1,p2))
    e_term = np.negative((np.multiply(pdiff,pdiff)))/(2 * alpha * alpha)
    cost = np.exp(e_term)
    return cost

def neighbor_cost_boykov_scalar(p1, p2, alpha):
    pdiff = p1 - p2
    e_term = -(pdiff * pdiff)/(2 * alpha * alpha)
    cost = math.exp(e_term)
    return cost

if len(sys.argv) < 3:
    print "usage: python abswap.py image brush"
    exit()

num_objs = 4

actual_img = imread(sys.argv[1], True)
brush_img = imread(sys.argv[2])


im_shape = actual_img.shape
width = im_shape[0]
height = im_shape[1]

actual_img = actual_img.reshape((width*height,1))
brush_img = brush_img.reshape((width*height,3))

indices = np.arange(actual_img.size).reshape(im_shape).astype(np.int32)

labels = np.random.randint(0,num_objs,(width*height))
regional_weights = np.zeros((width*height,num_objs))

graph_size = width*height

obj0samples = []
obj1samples = []
obj2samples = []
obj3samples = []

#masks for samples and intial labeling
red_band = brush_img[:,0]
green_band = brush_img[:,1]
blue_band = brush_img[:,2]

red_mask = np.logical_and(red_band > 200,green_band < 200)
blue_mask = np.logical_and(blue_band > 200,green_band < 200)
green_mask = np.logical_and(green_band > 200,red_band < 200)
yellow_mask_1 = np.logical_and(red_band > 200,green_band > 200)
yellow_mask = np.logical_and(yellow_mask_1,blue_band<200)

labels[red_mask] = 0
labels[yellow_mask] = 1
labels[blue_mask] = 2
labels[green_mask] = 3

t1 = time.time()
obj0samples = actual_img[red_mask]
obj1samples = actual_img[blue_mask]
obj2samples = actual_img[green_mask]
obj3samples = actual_img[yellow_mask]
print "time to get samples " + str(time.time() - t1)

sample_mat = np.concatenate((obj0samples,obj1samples, obj2samples, obj3samples))    
sample_mat = sample_mat.reshape((sample_mat.size,1))
label_mat = np.array([0]*obj0samples.size + [1]*obj1samples.size + [2]*obj2samples.size + [3]*obj3samples.size)

lgr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
lgr.fit(sample_mat,label_mat)

t1 = time.time()
regional_weights = lgr.predict_log_proba(actual_img)
print "time to calculate regional weights " + str(time.time() - t1)
print regional_weights.shape

# get adjacent edges
# right edges = (left_node[i],right_node[i])
left_nodes = indices[:, :-1].ravel()
right_nodes = indices[:, 1:].ravel()
side_weights = neighbor_cost_boykov(left_nodes,right_nodes,100)

#down edges = (up_node[i],down_node[i]) 
down_nodes = indices[1:, :].ravel()
up_nodes = indices[:-1,:].ravel()
vert_weights = neighbor_cost_boykov(down_nodes,up_nodes,100)

print left_nodes.shape
print up_nodes.shape
v1 = np.concatenate((left_nodes,up_nodes))
v1 = v1.reshape((v1.size))

v2 = np.concatenate((right_nodes, down_nodes))
v2 = v2.reshape((v2.size))

boundary_weights = np.concatenate((side_weights,vert_weights))
boundary_weights = boundary_weights.reshape((boundary_weights.size)).astype(np.float32)

t1= time.time()
boundary_weight_dict = {}
for i in range(v1.shape[0]):
    boundary_weight_dict[(v1[i],v2[i])] = boundary_weights[i]
    
print time.time() - t1

for alpha,beta in combinations(range(num_objs),2):

    alpha_beta_mask = np.logical_or(labels == alpha, labels == beta)
    graph_indices = indices.reshape((indices.size))[alpha_beta_mask]
    graph_size = graph_indices.size

    node_map = np.full((actual_img.size,1),-1)
    node_map[alpha_beta_mask] = np.array(range(0,graph_indices.size)).reshape((graph_indices.size,1))

    v1_mask = np.in1d(v1,graph_indices)
    v2_mask = np.in1d(v2,graph_indices)
    edge_mask = np.logical_and(v1_mask,v2_mask)
    
    e1 = np.take(node_map,v1[edge_mask]).astype(np.int32)
    e2 = np.take(node_map,v2[edge_mask]).astype(np.int32)
    r_weights = regional_weights
    g = PyGraph(graph_size, graph_size * 4)

    g.add_node(graph_size)
    
    g.add_edge_vectorized(e1,e2,boundary_weights[edge_mask],boundary_weights[edge_mask])
    
    # this needs speed up
    for i in graph_indices:
        
        left = i-1
        right = i+1
        up = i-width
        down = i+width

        if left > -1:
            if left % width != width -1:
                if labels[left] != alpha or labels[left] != beta:
                    try:
                        r_weights[i][alpha] += boundary_weight_dict[(i,left)]
                        r_weights[i][beta] += boundary_weight_dict[(i,left)]
                    except KeyError:
                        r_weights[i][alpha] += boundary_weight_dict[(left,i)]
                        r_weights[i][beta] += boundary_weight_dict[(left,i)]

        if right < actual_img.size:
            if right % width != 0:
                if labels[right] != alpha or labels[right] != beta:
                    try:
                        r_weights[i][alpha] += boundary_weight_dict[(i,right)]
                        r_weights[i][beta] += boundary_weight_dict[(i,right)]
                    except KeyError:
                        r_weights[i][alpha] += boundary_weight_dict[(right,i)]
                        r_weights[i][beta] += boundary_weight_dict[(right,i)]

        if up > -1:        
            if labels[up] != alpha or labels[up] != beta:
                try:
                    r_weights[i][alpha] += boundary_weight_dict[(i,up)]
                    r_weights[i][beta] += boundary_weight_dict[(i,up)]
                except KeyError:
                    r_weights[i][alpha] += boundary_weight_dict[(up,i)]
                    r_weights[i][beta] += boundary_weight_dict[(up,i)]

        if down < actual_img.size:        
            if labels[down] != alpha or labels[down] != beta:
                try:
                    r_weights[i][alpha] += boundary_weight_dict[(i,down)]
                    r_weights[i][beta] += boundary_weight_dict[(i,down)]
                except KeyError:
                    r_weights[i][alpha] += boundary_weight_dict[(down,i)]
                    r_weights[i][beta] += boundary_weight_dict[(down,i)]
   
    g.add_tweights_vectorized(np.array(range(graph_size)).astype(np.int32),r_weights[:,alpha][alpha_beta_mask].astype(np.float32),r_weights[:,beta][alpha_beta_mask].astype(np.float32))

    t1=time.time()
    g.maxflow()
    print "time for maxflow " + str(time.time() -t1)

    out = g.what_segment_vectorized()
    
    # this needs speed up
    for i, label in enumerate(out):
        index = graph_indices[i]
        if label == 0:
            labels[index] = alpha
        else:
            labels[index] = beta

imsave('test_ct_cython.png',labels.reshape((width,height)))