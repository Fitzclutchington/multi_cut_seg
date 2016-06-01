import numpy as np
from scipy.misc import imread, imsave, comb
from scipy.stats import multivariate_normal
from pymaxflow import PyGraph
import sys
import time
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GMM
from itertools import combinations
import math
from utils import non_graph_weight_addition, non_graph_weight_addition_with_count, calculate_boundary_stats, calculate_boundary_stats#lgr
import json
import dicom
import glob

def mask_list_bmp(brush_img):
    red_band = brush_img[:,0]
    green_band = brush_img[:,1]
    blue_band = brush_img[:,2]

    red_mask = np.logical_and(red_band == 255, green_band == 0)
    blue_mask = np.logical_and(blue_band == 255,green_band == 0)
    green_mask = np.logical_and(green_band == 255,red_band == 0)
    yellow_mask_1 = np.logical_and(red_band == 255,green_band == 255)
    yellow_mask = np.logical_and(yellow_mask_1,blue_band==0)
    
    print red_mask.shape
    return [yellow_mask, red_mask, blue_mask, green_mask]

def t_link_cost(std, mean, pixel):
    sqrt2xmean = math.sqrt(mean*2)
    d = pixel - mean
    e_term = np.negative(np.multiply(d,d))/(2 * std * std)
    e = np.exp(e_term)
    g = e / (std * sqrt2xmean)
    inf_mask = g==0
    g[inf_mask] = .000001
    return np.abs(np.log(g))

def neighbor_cost_prob(regional_weights,v1,v2,num_comb,com_list):
    ''' This function creates a matrix of boundary weights of size M X N
    where M is the number of neighbors and N is the number of combinations
    of classes.  Boundary weights are calculated multiplying the probaility
    that pixel i is in class x and the probability that pixel j is in class y'''

    num_rows = v1.shape[0]
    boundary_weights = np.zeros((num_rows,num_comb))
    probs_v1 = regional_weights[v1]
    probs_v2 = regional_weights[v2]

    for i,com in enumerate(com_list):
        class_i = com[0]
        class_j = com[1]
        prob_i_j = np.add(probs_v1[:,class_i],probs_v2[:,class_j])
        prob_j_i = np.add(probs_v1[:,class_j],probs_v2[:,class_i])
        boundary_weights[:,i] = np.abs(np.add(prob_j_i,prob_i_j))
    print boundary_weights
    return boundary_weights

def neighbor_cost_boykov(p1, p2, alpha=10):
    '''calculates neighbor cost based on equations from Boykov's paper'''
    print p1.shape
    pdiff = np.subtract(p1,p2)
    psquare = np.sqrt(np.square(pdiff))
    e_term = np.negative((np.multiply(psquare,psquare)))/(2 * alpha * alpha)
    cost = np.exp(e_term)
    print cost
    return cost.astype(np.float32).reshape(cost.size)       

if len(sys.argv) < 2:
    print "usage: python abswap.py json_config"
    exit()

f = open(sys.argv[1],'r')
task = json.load(f)
f.close()

num_objs = task['object_num']
training_images = [p for p in task['training_images']]
modalities = [m for m in task['modalities']]
training_image_indices = [m for m in task['training_image_indices']]
case = task['case']
outdir =  case + '/' + task['outdir']
dicomdir = task['dicomdir']
regional_method = task['regional_weights']
boundary_method = task['boundary_weights']
img_type = task['img_type']
num_comb = int(comb(num_objs,2))
com_list = combinations(range(num_objs),2)
dirs = [case +"/"+dicomdir+"/" + m for m in modalities]
num_modalities = len(dirs)

print dirs
mmdirs = []

if img_type == "dcm":
    for d in dirs:
        fil = d + '/*.dcm'
        lis = glob.glob(fil)
        lis.sort()
        mmdirs.append(lis)
else:
    for d in dirs:
        fil = d + '/*.bmp'
        lis = glob.glob(fil)
        lis.sort()
        mmdirs.append(lis)


mmimgs= []
for i in range(len(mmdirs[0])):
    mmimgs.append([m[i] for m in mmdirs])


####################################################
#                                                  #
# Set up initial labelling                         #
#                                                  #
####################################################
object_list = [ [] for i in range(num_objs)]
for i in object_list:
    for j in range(num_modalities):
        i.append([])


for i,img in enumerate(training_images):
    if img_type == "dcm":
        brush_img = dicom.read_file(img)
        brush_img = brush_img.pixel_array
        width = brush_img.Columns
        height = brush_img.Rows
    else:
        brush_img = imread(img)
        width = brush_img.shape[0]
        height = brush_img.shape[1]
    im_size = width*height
    brush_img = brush_img.reshape((im_size,3))
    #masks for samples and intial labeling    

    if img_type == "dcm":
        mask_list = [brush_img== num for num in range(num_objs)]
    else:
        mask_list = mask_list_bmp(brush_img)
    
    #print mask_list[0]
    train_index = training_image_indices[i] - 1
    actual_imgs = mmimgs[train_index]
    
    
    for k,img in enumerate(actual_imgs):
        if img_type == "dcm":
            current_im = dicom.read_file(str(img)).pixel_array
        else:
            # True designates that we want a gray scale image
            current_im = imread(str(img),True)
        current_im = current_im.reshape((im_size))
        for obj_num,obj in enumerate(object_list):
            obj[k].extend(current_im[mask_list[obj_num].reshape((im_size))])


samples = [np.array(obj).T for obj in object_list]
for i,sample in enumerate(samples):
    print 'Sample size of ' + str(i) + ' = ' + str(len(sample))

##############################################
#                                            #
#  Calculate regional weights Stats          #
#                                            #
##############################################

if regional_method == 'gaussian':
    gaussian_mean = []
    gaussian_std = []

    for sample in samples:
        gaussian_mean.append(np.mean(sample))
        gaussian_std.append(np.std(sample))

else:
    # todo fix for new samples
    sample_mat = np.vstack(samples)    
    label_mat = []
    for i,samp in enumerate(samples):
        label_mat.extend([i]*samp.shape[0])

    lgr = LogisticRegression()#solver='newton-cg')#,multi_class='multinomial')
    lgr.fit(sample_mat,label_mat)

##############################################
#                                            #
#  Define neighbors                          #
#                                            #
##############################################
indices = np.arange(im_size).reshape((height,width)).astype(np.int32)

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

##############################################
#                                            #
#  Cut Graph                                 #
#                                            #
##############################################


for im_num,im_slice in enumerate(mmimgs):

    labels = np.random.randint(0,num_objs,(width*height)).astype(np.int32)

    image_mat = np.zeros((im_size,num_modalities))
    for i,img in enumerate(im_slice):
        if img_type =="dcm":
            image_mat[:,i] = dicom.read_file(str(img)).pixel_array.reshape((im_size))
        else:
            image_mat[:,i] = imread(str(img),True).reshape((im_size))
    
    ##############################################
    #                                            #
    #  Calculate regional weights                #
    #                                            #
    ##############################################

    if regional_method == 'gaussian':
        regional_weights = np.zeros((im_size,num_objs))

        for i in range(num_objs):
            regional_weights[:,i] = t_link_cost(gaussian_std[i],gaussian_mean[i],image_mat).T

    else:
        regional_weights = np.abs(lgr.predict_log_proba(image_mat)) 
    print regional_weights
    ##############################################
    #                                            #
    #  Set up boundary weights                   #
    #                                            #
    ##############################################

    print "regional weights size = " + str(regional_weights.shape)
    if boundary_method == 'prob':
        boundary_weights = neighbor_cost_prob(regional_weights,v1,v2,num_comb,com_list)
    else:
        boundary_weights = neighbor_cost_boykov(image_mat[v1],image_mat[v2])
    
   
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
        for count, (alpha,beta) in enumerate(com_list):
            alpha_beta_mask = np.logical_or(labels == alpha, labels == beta)
            #graph indices are the pixels locations that are included in the present ab graph
            graph_indices = indices.reshape((indices.size))[alpha_beta_mask].astype(np.int32)
            graph_size = graph_indices.size
            
            # node_map[i] is the node_id of the pixel i
            node_map = np.full((im_size),-1).astype(np.int32)
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
            
            
           
            if boundary_method == 'boykov':
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

    labels = labels.reshape((height,width))
    black_mask = labels == 0
    red_mask = labels == 1
    blue_mask = labels == 2
    green_mask = labels == 3
    purp_mask = labels == 4
    np.savetxt('labels', labels)

    seg_im = np.zeros((height,width,3))
    seg_im[black_mask] = [0,0,0]
    seg_im[red_mask] = [255,0,0]
    seg_im[blue_mask] = [0,0,255]
    seg_im[green_mask] = [0,255,0]
    seg_im[purp_mask] = [255,0,255]
    save_loc = outdir + '/GT_' + regional_method + '_' + boundary_method +'_{0:04d}'.format(im_num+1) + '.png'
    imsave(save_loc,seg_im)
    print save_loc
