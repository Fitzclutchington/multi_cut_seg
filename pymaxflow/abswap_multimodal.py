import numpy as np
from scipy.misc import imread, imsave, comb
from pymaxflow import PyGraph
import sys
import time
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import math
from utils import non_graph_weight_addition,non_graph_weight_addition_with_count, calculate_boundary_stats, calculate_boundary_stats_lgr
import json
import dicom

def neighbor_cost_boykov(p1, p2, alpha=100):
    pdiff = np.subtract(p1,p2)
    e_term = np.negative((np.multiply(pdiff,pdiff)))/(2 * alpha * alpha)
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

if len(sys.argv) < 2:
    print "usage: python abswap.py json_config"
    exit()

num_objs = 2

f = open(sys.argv[1],'r')
task = json.load(f)
f.close()

training_image = task['training_images']
modalities = [m for m in task['modalities']]
training_image_index = task['training_image_index']
case = task['case']
prefix = task['prefix']
outdir =  case + '/' + task['outdir']
dicomdir = task['dicomdir']
regional_method = task['regional_weights']
boundary_method = task['boundary_weights']

dirs = [case +"/"+dicomdir+"/" + m for m in modalities]
num_modalities = len(dirs)
mmdirs = []

for d in dirs:
    fil = d+'/IMG00'+ str(training_image_index)+'.dcm'
    mmdirs.append(fil)

brush_img = imread(str(training_image))

im_shape = brush_img.shape
width = im_shape[0]
height = im_shape[1]
im_size = width*height
brush_img = brush_img.reshape((im_size,3))

indices = np.arange(im_size).reshape((width,height)).astype(np.int32)

labels = np.random.randint(0,num_objs,(width*height)).astype(np.int32)
regional_weights = np.zeros((width*height,num_objs))

image_mat = np.zeros((im_size,num_modalities))
for i,img in enumerate(mmdirs):
    image_mat[:,i] = dicom.read_file(str(img)).pixel_array.reshape((im_size))

####################################################
#                                                  #
# Set up initial labelling                         #
#                                                  #
####################################################

samples0 = []
samples1 = []

#masks for samples and intial labeling
red_band = brush_img[:,0]
green_band = brush_img[:,1]
blue_band = brush_img[:,2]

red_mask = np.logical_and(red_band == 255, green_band == 0)
blue_mask = np.logical_and(blue_band == 255,green_band == 0)
green_mask = np.logical_and(green_band == 255,red_band == 0)
yellow_mask_1 = np.logical_and(red_band == 255,green_band == 255)
yellow_mask = np.logical_and(yellow_mask_1,blue_band==0)


labels[blue_mask] = 0
labels[red_mask] = 1

for img in mmdirs:
    current_im = dicom.read_file(str(img)).pixel_array.reshape((im_size))
    samples0.append(current_im[blue_mask])
    samples1.append(current_im[red_mask])

#setup array to pass to fit
obj0samples = np.zeros((samples0[0].size,num_modalities))
obj1samples = np.zeros((samples1[0].size,num_modalities))

for i in range(num_modalities):
    obj0samples[:,i] = samples0[i]
    obj1samples[:,i] = samples1[i]

print type(samples0[0])   

##############################################
#                                            #
#  Calculate regional weights                #
#                                            #
##############################################

if regional_method == 'gaussian':
    samples = [obj0samples, obj1samples, obj2samples, obj3samples]
    mean = []
    std = []

    for sample in samples:
        mean.append(np.mean(sample))
        std.append(np.std(sample))
   
    regional_weights = np.zeros((actual_img.size,num_objs))

    for i in range(num_objs):
        regional_weights[:,i] = t_link_cost(std[i],mean[i],actual_img).T

else:
    sample_mat = np.concatenate((obj0samples,obj1samples))    
    #sample_mat = sample_mat.reshape((sample_mat.size,1))
    label_mat = np.array([0]*samples0[0].shape[0] + [1]*samples1[0].shape[0])
    
    print sample_mat.shape
    print label_mat.shape
    lgr = LogisticRegression()#solver='newton-cg')#,multi_class='multinomial')
    lgr.fit(sample_mat,label_mat)
    regional_weights = np.abs(lgr.predict_log_proba()) 


##############################################
#                                            #
#  Set up boundary weights                   #
#                                            #
##############################################

brush_strokes = [obj0samples,obj1samples]

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

if boundary_option == 'boykov':
    side_weights = neighbor_cost_boykov(actual_img[left_nodes],actual_img[right_nodes])
    vert_weights = neighbor_cost_boykov(actual_img[down_nodes],actual_img[up_nodes])
    boundary_weights = np.concatenate((side_weights,vert_weights))
    boundary_weights = boundary_weights.reshape((boundary_weights.size)).astype(np.float32) * cost_weight

elif boundary_option == 'gaussian':
    means, stds = calculate_boundary_stats(brush_strokes,100)
    combs = int(comb(num_objs,2))

    boundary_weights = np.zeros((v1.size,combs))
    diffs = np.abs(np.subtract(actual_img[v1],actual_img[v2]))
    for i in range(combs):
        boundary_weights[:,i] = t_link_cost(stds[i],means[i],diffs).T
    boundary_weights = boundary_weights.astype(np.float32) #* cost_weight

else:
    lgr_bound = calculate_boundary_stats_lgr(brush_strokes,100)
    boundary_weights = np.abs(lgr_bound.predict_log_proba(np.abs(np.subtract(actual_img[v1],actual_img[v2])))).astype(np.float32) * cost_weight

# Dictionary used to add to terminal edges
boundary_weight_dict = {}
for i in range(v1.shape[0]):
    boundary_weight_dict[(v1[i],v2[i])] = boundary_weights[i]


###############################
#                             #
# Graph Construction          #
#                             #
###############################
reps = 1
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
        
        
        
        if boundary_option == 0:
            g.add_edge_vectorized(e1,e2,boundary_weights[edge_mask],boundary_weights[edge_mask])
            non_graph_weight_addition(graph_indices,r_weights,boundary_weight_dict,labels,alpha,beta,width,height)
        else:
            g.add_edge_vectorized(e1,e2,boundary_weights[:,count][edge_mask],boundary_weights[:,count][edge_mask])
            non_graph_weight_addition_with_count(graph_indices,r_weights,boundary_weight_dict,labels,alpha,beta,width,height,count)
        
        
        g.add_tweights_vectorized(np.array(range(graph_size)).astype(np.int32),r_weights[:,alpha][alpha_beta_mask],r_weights[:,beta][alpha_beta_mask])
        
        
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
yellow_mask = labels == 0
red_mask = labels == 1
blue_mask = labels == 2
green_mask = labels == 3
np.savetxt('labels', labels)

seg_im = np.zeros((width,height,3))
seg_im[blue_mask] = [0,0,0]
seg_im[red_mask] = [255,0,0]
#seg_im[blue_mask] = [0,0,255]
#seg_im[green_mask] = [0,255,0]
save_loc = outdir + '_' + regional_method + '_' + boundary_method
imsave(save_loc,seg_im)
