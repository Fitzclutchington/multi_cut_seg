import numpy as np
from scipy.misc import imread, imsave, comb
from pymaxflow import PyGraph
import sys
import time
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import math
from utils import non_graph_weight_addition,non_graph_weight_addition_with_count, calculate_boundary_stats#, calculate_boundary_stats_lgr
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

def neighbor_cost_boykov(p1, p2, alpha=100):
    pdiff = np.subtract(p1,p2)
    e_term = np.negative((np.multiply(pdiff,pdiff)))/(2 * alpha * alpha)
    cost = np.exp(e_term)
    return cost

def neighbor_cost_boykov_scalar(p1, p2, alpha=100):
    pdiff = p1 - p2
    e_term = -(pdiff * pdiff)/(2 * alpha * alpha)
    cost = math.exp(e_term)
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

if len(sys.argv) < 4:
    print "usage: python abswap.py image brush output alpha"
    exit()

num_objs = 4

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

####################################################
#                                                  #
# Set up initial labelling                         #
#                                                  #
####################################################

obj0samples = []
obj1samples = []
obj2samples = []
obj3samples = []

#masks for samples and intial labeling
red_band = brush_img[:,0]
green_band = brush_img[:,1]
blue_band = brush_img[:,2]

red_mask = np.logical_and(red_band == 255, green_band == 0)
blue_mask = np.logical_and(blue_band == 255,green_band == 0)
green_mask = np.logical_and(green_band == 255,red_band == 0)
yellow_mask_1 = np.logical_and(red_band == 255,green_band == 255)
yellow_mask = np.logical_and(yellow_mask_1,blue_band==0)


labels[yellow_mask] = 0
labels[red_mask] = 1
labels[blue_mask] = 2
labels[green_mask] = 3

#order matters so that initial labeling and probabilities are correct
obj0samples = actual_img[yellow_mask]
obj1samples = actual_img[red_mask]
obj2samples = actual_img[blue_mask]
obj3samples = actual_img[green_mask]

brush_strokes = [obj0samples,obj1samples,obj2samples,obj3samples]

means, stds = calculate_boundary_stats(brush_strokes,100)
print means
print stds
combs = int(comb(num_objs,2))

##############################################
#                                            #
#  Calculate regional weights                #
#                                            #
##############################################

if regional_option == 0:
    samples = [obj0samples, obj1samples, obj2samples, obj3samples]
    mean = []
    std = []

    for sample in samples:
        mean.append(np.mean(sample))
        std.append(np.std(sample))

    print mean
    print std   
    regional_weights = np.zeros((actual_img.size,num_objs))

    for i in range(num_objs):
        regional_weights[:,i] = t_link_cost(std[i],mean[i],actual_img).T

else:
    sample_mat = np.concatenate((obj0samples,obj1samples, obj2samples,obj3samples))    
    sample_mat = sample_mat.reshape((sample_mat.size,1))
    label_mat = np.array([0]*obj0samples.size + [1]*obj1samples.size + [2]*obj2samples.size + [3]*obj3samples.size)

    lgr = LogisticRegression()#solver='newton-cg')#,multi_class='multinomial')
    lgr.fit(sample_mat,label_mat)

    t1 = time.time()
    regional_weights = np.abs(lgr.predict_log_proba(actual_img)) 

##############################################
#                                            #
#  Set up boundary weights                   #
#                                            #
##############################################

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
    lgr_bound = calculate_boundary_stats_lgr(brush_strokes,100)
    boundary_weights = lgr_bound.predict_log_proba(np.abs(np.subtract(actual_img[v1],actual_img[v2]))).astype(np.float32) * cost_weight


# Dictionary used to add to terminal edges
boundary_weight_dict = {}
for i in range(v1.shape[0]):
    boundary_weight_dict[(v1[i],v2[i])] = boundary_weights[i]

###############################
#                             #
# Graph Construction          #
#                             #
###############################
reps = 3
while reps > 0:
    reps -= 1
    for count, (alpha,beta) in enumerate(combinations(range(num_objs),2)):
        alpha_beta_mask = np.logical_or(labels == alpha, labels == beta)
        #graph indices are the pixels locations that are included in the present ab graph
        graph_indices = indices.reshape((indices.size))[alpha_beta_mask].astype(np.int32)
        graph_size = graph_indices.size
        
        # node_map[i] is the node_id of the pixel i
        node_map = np.full((actual_img.size),-1).astype(np.int32)
        # fill node map with node values
        node_map[alpha_beta_mask] = np.array(range(graph_size)).reshape((graph_size))
        

        # find edges that belong to the current graph
        #v1 = left and up, v2 = right and down
        # in1d(a,b) returns true if b is in a (b can be an array)
        v1_mask = np.in1d(v1,graph_indices)
        v2_mask = np.in1d(v2,graph_indices)
        edge_mask = np.logical_and(v1_mask,v2_mask)
        
        # get all values in node_map from the indicies of v1,v2 
        # e1 and e2 are the values of nodes in graphs that are neighbors
        e1 = np.take(node_map,v1[edge_mask]).astype(np.int32)
        e2 = np.take(node_map,v2[edge_mask]).astype(np.int32)

        r_weights = regional_weights.astype(np.float32)
        
        g = PyGraph(graph_size, graph_size * 4)

        g.add_node(graph_size)
        
        
        t1 = time.time()
        if boundary_option == 0:
            g.add_edge_vectorized(e1,e2,boundary_weights[edge_mask],boundary_weights[edge_mask])
            non_graph_weight_addition(graph_indices,r_weights,boundary_weight_dict,labels,alpha,beta,width,height)
        else:
            g.add_edge_vectorized(e1,e2,boundary_weights[:,count][edge_mask],boundary_weights[:,count][edge_mask])
            non_graph_weight_addition_with_count(graph_indices,r_weights,boundary_weight_dict,labels,alpha,beta,width,height,count)
        print 'time to evaluate non neighbor weights ' + str(time.time() - t1)
        
        g.add_tweights_vectorized(np.array(range(graph_size)).astype(np.int32),r_weights[:,alpha][alpha_beta_mask],r_weights[:,beta][alpha_beta_mask])
        
        
        g.maxflow()
        

        out = g.what_segment_vectorized()
        print sum(out)
        # this needs speed up
        for i, label in enumerate(out):
            index = graph_indices[i]
            if label == 1:
                labels[index] = alpha
            else:
                labels[index] = beta

black_mask = labels == 0
grey_mask = labels == 1
greyer_mask = labels == 2
white_mask = labels == 3
np.savetxt('labels', labels)

labels[black_mask] = 0
labels[grey_mask] = 50
labels[greyer_mask] = 100
labels[white_mask] = 255
imsave(output_file,labels.reshape((width,height)))