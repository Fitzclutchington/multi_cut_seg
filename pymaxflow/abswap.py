import numpy as np
from scipy.misc import imread, imsave
from pymaxflow import PyGraph
import sys
import time
from sklearn.linear_model import LogisticRegression


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

labels = np.zeros((width*height),dtype=np.uint8)
regional_weights = np.zeros((width*height,num_objs))

graph_size = width*height

obj0samples = []
obj1samples = []
obj2samples = []
obj3samples = []


# assign labels and collect samples based on brush image
t1 = time.time()
for i, cur_pix in enumerate(brush_img):
    #label 0
    if cur_pix[0] > 250 and cur_pix[1] < 10:
        obj0samples.append(actual_img[i])
        labels[i] = 0
    #label 1
    elif cur_pix[0] < 10  and cur_pix[1] > 250:
        obj1samples.append(actual_img[i])
        labels[i] = 1
    #label 2
    elif cur_pix[0] < 10  and cur_pix[2] > 250:
        obj2samples.append(actual_img[i])
        labels[i] = 2
    #label 3
    elif cur_pix[0] > 250 and cur_pix[1] > 250:
        obj3samples.append(actual_img[i])
        labels[i] = 3
    # assign random label
    else:
        labels[i] = np.random.randint(0,4)
print "time to establish initial labelling " + str(time.time() - t1)

sample_mat = np.array(obj0samples + obj1samples + obj2samples + obj3samples)
sample_mat = sample_mat.reshape((sample_mat.shape[0],1))
label_mat = np.array([0]*len(obj0samples) + [1]*len(obj1samples) + [2]*len(obj2samples) + [3]*len(obj3samples))


lgr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
lgr.fit(sample_mat,label_mat)

t1 = time.time()
regional_weights = lgr.predict_log_proba(actual_img)
print "time to calculate regional weights " + str(time.time() - t1)

# get adjacent edges
# right edges = (left_node[i],right_node[i])
left_nodes = indices[:, :-1].ravel()
right_nodes = indices[:, 1:].ravel()


#down edges = (up_node[i],down_node[i]) 
down_node = indices[1:, :].ravel()
up_node = indices[:-1,:].ravel()

#for alpha,beta in combinations(range(num_objs),2):
    # construct graph:
    # We need all pixels that are labeled with alpha and beta
    # add them as nodes using pygraph.add_node
    # add edges if neighbors, else add weight to t_edges
    # maxflow
    # use what_segment to determine new labels
