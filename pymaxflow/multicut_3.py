import numpy as np
from scipy.misc import imread, imsave, comb
from pymaxflow import PyGraph
import sys
import time
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import math
from utils import calculate_boundary_stats, calculate_boundary_stats_lgr,non_graph_weight_addition, non_graph_weight_addition_with_count

def neighbor_cost_boykov(p1, p2, alpha=.001):
    pdiff = np.subtract(p1,p2)
    e_term = np.negative((np.multiply(pdiff,pdiff)))/(2.0 * alpha * alpha)
    cost = np.exp(e_term)
    return cost

def t_link_cost(std, mean, pixel):
    sqrt2xmean = math.sqrt(mean*2)
    d = pixel - mean
    e_term = np.negative(np.multiply(d,d))/(2 * std * std)
    e = np.exp(e_term)
    g = e / (std * sqrt2xmean)
    inf_mask = g==0
    g[inf_mask] = .000001
    return np.abs(np.log(g))

def neighbor_cost_boykov_scalar(p1, p2, alpha=100):
    pdiff = p1 - p2
    e_term = -(pdiff * pdiff)/(2 * alpha * alpha)
    cost = math.exp(e_term)
    return cost

if len(sys.argv) < 4:
    print "usage: python abswap.py image brush output cost_weight boundary regional"
    exit()

bound_dict = {'0':'boykov',
              '1':'gaussian',
              '2':'lgr'}

reg_dict = {'0':'gaussian',
            '1':'lgr'}

num_objs = 3

actual_img = imread(sys.argv[1], True)
brush_img = imread(sys.argv[2])
output_file = sys.argv[3]
cost_weight = float(sys.argv[4])
boundary_option = int(sys.argv[5])
regional_option = int(sys.argv[6])

im_shape = actual_img.shape
width = im_shape[0]
height = im_shape[1]

actual_img = actual_img.reshape((width*height,1))
brush_img = brush_img.reshape((width*height,3))

indices = np.arange(actual_img.size).reshape(im_shape).astype(np.int32)

labels = np.random.randint(0,num_objs,(width*height)).astype(np.int32)
regional_weights = np.zeros((width*height,num_objs))

graph_size = width*height

obj0samples = []
obj1samples = []
obj2samples = []


#masks for samples and intial labeling
red_band = brush_img[:,0]
green_band = brush_img[:,1]
blue_band = brush_img[:,2]

red_mask = np.logical_and(red_band == 255, green_band == 0)
blue_mask = np.logical_and(blue_band == 255,green_band == 0)
green_mask = np.logical_and(green_band == 255,red_band == 0)


labels[green_mask] = 0
labels[blue_mask] = 1
labels[red_mask] = 2

#order matters so that initial labeling and probabilities are correct
t1 = time.time()

obj0samples = actual_img[green_mask]
obj1samples = actual_img[blue_mask]
obj2samples = actual_img[red_mask]

brush_strokes = [obj0samples,obj1samples,obj2samples]

means, stds = calculate_boundary_stats(brush_strokes,50)
combs = int(comb(num_objs,2))

if regional_option == 0:
    samples = [obj0samples, obj1samples, obj2samples]
    mean = []
    std = []

    for sample in samples:
        mean.append(np.mean(sample))
        std.append(np.std(sample))
  
    regional_weights = np.zeros((actual_img.size,3))

    regional_weights[:,0] = t_link_cost(std[0],mean[0],actual_img).T
    regional_weights[:,1] = t_link_cost(std[1],mean[1],actual_img).T
    regional_weights[:,2] = t_link_cost(std[2],mean[2],actual_img).T
    

else:
    sample_mat = np.concatenate((obj0samples,obj1samples, obj2samples))    
    sample_mat = sample_mat.reshape((sample_mat.size,1))
    label_mat = np.array([0]*obj0samples.size + [1]*obj1samples.size + [2]*obj2samples.size )

    lgr = LogisticRegression()#solver='newton-cg')#,multi_class='multinomial')
    lgr.fit(sample_mat,label_mat)

    t1 = time.time()
    regional_weights = np.abs(lgr.predict_log_proba(actual_img))

# get adjacent edges
# right edges = (left_node[i],right_node[i])
left_nodes = indices[:, :-1].ravel()
right_nodes = indices[:, 1:].ravel()

#for i in range(len(means):

#down edges = (up_node[i],down_node[i]) 
down_nodes = indices[1:, :].ravel()
up_nodes = indices[:-1,:].ravel()



v1 = np.concatenate((left_nodes,up_nodes))
v1 = v1.reshape((v1.size))

v2 = np.concatenate((right_nodes, down_nodes))
v2 = v2.reshape((v2.size))

if boundary_option == 0:
    side_weights = neighbor_cost_boykov(actual_img[left_nodes],actual_img[right_nodes])
    vert_weights = neighbor_cost_boykov(actual_img[down_nodes],actual_img[up_nodes])
    boundary_weights = np.concatenate((side_weights,vert_weights))
    boundary_weights = boundary_weights.reshape((boundary_weights.size)).astype(np.float32) * cost_weight
elif boundary_option == 1:
    boundary_weights = np.zeros((v1.size,combs))
    diffs = np.abs(np.subtract(actual_img[v1],actual_img[v2]))
    for i in range(combs):
        boundary_weights[:,i] = t_link_cost(stds[i],means[i],diffs).T
    boundary_weights = boundary_weights.astype(np.float32) * cost_weight
else:
    lgr_bound = calculate_boundary_stats_lgr(brush_strokes,50)
    boundary_weights = np.abs(lgr_bound.predict_log_proba(np.abs(np.subtract(actual_img[v1],actual_img[v2])))).astype(np.float32) * cost_weight


boundary_weight_dict = {}
for i in range(v1.shape[0]):
    boundary_weight_dict[(v1[i],v2[i])] = boundary_weights[i]


for count, (alpha,beta) in enumerate(combinations(range(num_objs),2)):

    alpha_beta_mask = np.logical_or(labels == alpha, labels == beta)
    graph_indices = indices.reshape((indices.size))[alpha_beta_mask].astype(np.int32)
    graph_size = graph_indices.size

    node_map = np.full((actual_img.size,1),-1)
    node_map[alpha_beta_mask] = np.array(range(0,graph_indices.size)).reshape((graph_indices.size,1))

    v1_mask = np.in1d(v1,graph_indices)
    v2_mask = np.in1d(v2,graph_indices)
    edge_mask = np.logical_and(v1_mask,v2_mask)
    
    e1 = np.take(node_map,v1[edge_mask]).astype(np.int32)
    e2 = np.take(node_map,v2[edge_mask]).astype(np.int32)
    r_weights = regional_weights.astype(np.float32)

    g = PyGraph(graph_size, graph_size * 4)

    g.add_node(graph_size)
    
    if boundary_option == 0:
        g.add_edge_vectorized(e1,e2,boundary_weights[edge_mask],boundary_weights[edge_mask])
        non_graph_weight_addition(graph_indices,r_weights,boundary_weight_dict,labels,alpha,beta,width,height)
    else:
        g.add_edge_vectorized(e1,e2,boundary_weights[:,count][edge_mask],boundary_weights[:,count][edge_mask])
        non_graph_weight_addition_with_count(graph_indices,r_weights,boundary_weight_dict,labels,alpha,beta,width,height,count)
    
    # this needs speed up

    g.add_tweights_vectorized(np.arange(graph_size).astype(np.int32),r_weights[:,alpha][alpha_beta_mask].astype(np.float32),r_weights[:,beta][alpha_beta_mask].astype(np.float32))

    
    g.maxflow()
    

    out = g.what_segment_vectorized()
    # this needs speed up
    for i, label in enumerate(out):
        index = graph_indices[i]
        if label == 1:
            labels[index] = alpha
        else:
            labels[index] = beta

labels = labels.reshape((width,height))

red_mask = labels == 2
blue_mask = labels == 1
green_mask = labels == 0
np.savetxt('labels', labels)

seg_im = np.zeros((width,height,3))

seg_im[red_mask] = [255,0,0]
seg_im[blue_mask] = [0,0,255]
seg_im[green_mask] = [0,0,0]
imsave(output_file,seg_im)
