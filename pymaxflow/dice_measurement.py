import numpy as np
import dicom
import sys
import glob
from scipy.misc import imread, imsave, comb

if len(sys.argv) < 3:
    print "usage: python dice_measurement.py case class"
    exit()

case = sys.argv[1]
multi_class = int(sys.argv[2])
GT_dir = case+'/DCM/GT'
seg_dir = case + '/test'

fil = GT_dir + '/*.dcm'
GT_files = glob.glob(fil)
GT_files.sort()
 
fil = seg_dir + '/*.png'
seg_files = glob.glob(fil)
seg_files.sort()

if multi_class == 0:
    record_file = case + '/dice.txt'
else:
    record_file = case + '/dice_class.txt'
f =open(record_file,'w')

for i,ims in enumerate(zip(GT_files,seg_files)):

    gt_img = dicom.read_file(ims[0])
    

    width = gt_img.Columns
    height = gt_img.Rows
    
    gt_img = gt_img.pixel_array.reshape(width*height)
    seg_img = imread(ims[1]).reshape(width*height,3)
    gt_mask = gt_img != 0
    
    if multi_class == 0:
        if gt_mask.sum() > 0:
            seg_mask = seg_img != [0,0,0]
            seg_mask = np.any(seg_mask,axis=1)
                       

            intersection = np.logical_and(seg_mask,gt_mask)

            dice = 2. * intersection.sum() /(seg_mask.sum() + gt_mask.sum())
            f.write(ims[1] + ' has dice coeffecient ' + str(dice) +'\n')
    else:
        if gt_mask.sum() > 0:
            class1_mask = gt_img == 1
            class2_mask = gt_img == 2
            class3_mask = gt_img == 3
            class4_mask = gt_img == 4
            
            class_mask = [class1_mask,class2_mask,class3_mask,class4_mask]

            seg1_mask = np.any((seg_img == [255,0,0]),axis=1)
            seg2_mask = np.any((seg_img == [0,0,255]),axis=1)
            seg3_mask = np.any((seg_img == [0,255,0]),axis=1)
            seg4_mask = np.any((seg_img == [255,0,255]),axis=1)

            seg_mask = [seg1_mask,seg2_mask,seg3_mask,seg4_mask]
            
            f.write('slice ' + str(i))
            for j,mask in enumerate(zip(class_mask,seg_mask)):
                intersection = np.logical_and(mask[0],mask[1])
                dice = 2. * intersection.sum() /(mask[0].sum() + mask[1].sum())
                f.write(' class ' + str(j)+ ": " + str(dice))

            f.write('\n')















    
