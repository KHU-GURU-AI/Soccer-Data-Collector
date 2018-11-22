import mxnet as mx
import numpy as np
import os
#import urllib
import cv2
#import matplotlib.pyplot as plt
from PIL import Image
import logging
import imageio
logging.basicConfig(level=logging.DEBUG)
import time
import sys

if not os.path.exists('deep3d-0050.params'):
        print("error")
model = mx.model.FeedForward.load('deep3d', 50, mx.gpu(1))
model.aux_params['bn_pool3_moving_var'] = model.aux_params['bn_pool3_moving_inv_var']
model.aux_params['bn_pool2_moving_var'] = model.aux_params['bn_pool2_moving_inv_var']
model.aux_params['bn_pool4_moving_var'] = model.aux_params['bn_pool4_moving_inv_var']
model.aux_params['bn_pool1_moving_var'] = model.aux_params['bn_pool1_moving_inv_var']
del model.aux_params['bn_pool1_moving_inv_var']
del model.aux_params['bn_pool2_moving_inv_var']
del model.aux_params['bn_pool3_moving_inv_var']
del model.aux_params['bn_pool4_moving_inv_var']


#use video (.avi or .mp4)
#start = time.time()
cap = cv2.VideoCapture('fifaonline4.mp4')
# current directory my video load (soccer video input)
i=0
time_list = []
while(cap.isOpened()):
#    if i%25 == 0:
#        time_list.append(time.time()-start)
    ret , frame = cap.read()
    if ret == False:
        break
    shape = (384,160)
    img = cv2.resize(frame,shape)
    X = img.astype(np.float32).transpose((2,0,1))
    X = X.reshape((1,)+X.shape)
    test_iter = mx.io.NDArrayIter({'left': X, 'left0':X})
    Y = model.predict(test_iter)
         
    right = np.clip(Y.squeeze().transpose((1,2,0)), 0, 255).astype(np.uint8)
    right = Image.fromarray(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    left = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    left.save('svideo_result_left/svideo_left_'+str(i)+'.png') # save left image(PIL)
    right.save('svideo_result_right/svideo_right_'+str(i)+'.png') # save predict right image (PIL)
    #generate gif (imageio open library)
#images = []
#images.append(imageio.imread('result/test3_left_'+str(i)+'.png'))
#images.append(imageio.imread('result/test3_right_'+str(i)+'.png'))
#imageio.mimsave('gif/test3_'+str(i)+'.gif',images,duration=0.01)
    i += 1
print(time_list)
cap.release()


#in folder use image
'''
file_list = os.listdir('15-deg-left')
#my image folder name (soccer image )
print(file_list)
for fname in file_list:
    shape = (384, 160)
    print(fname)
    img = cv2.imread('15-deg-left/'+fname)
    raw_shape = (img.shape[1], img.shape[0])
    img = cv2.resize(img, shape)

    X = img.astype(np.float32).transpose((2,0,1))
    X = X.reshape((1,)+X.shape)
    test_iter = mx.io.NDArrayIter({'left': X, 'left0':X})
    Y = model.predict(test_iter)
    
    #generate right image(PIL)
    right = np.clip(Y.squeeze().transpose((1,2,0)), 0, 255).astype(np.uint8)
    right = Image.fromarray(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    left = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))




    #generate gif
    sname = fname.split('.')
    right.save('result/'+ sname[0] +'_right.png')
    left.save('result/'+sname[0] + '_left.png')
#images = []
#images.append(imageio.imread('result/'+ sname[0] +'_right.png'))
#images.append(imageio.imread('result/'+ sname[0] +'_left.png'))
#imageio.mimsave('gif/'+sname[0] + '.gif',images,duration=0.01)
'''
