import mxnet as mx
import numpy as np
import os
import cv2
from PIL import Image
import logging
from os.path import isfile, join
import argparse
#import imageio
logging.basicConfig(level=logging.DEBUG)


def make_right(video,imagepath) :

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

    left_path = imagepath+'_left/'
    right_path = imagepath + '_right/'

    if not (os.path.isdir(left_path)):
        os.mkdir(left_path)
    if not (os.path.isdir(right_path)):
        os.mkdir(right_path)

    vname = video.split('.')
    #use video (.avi or .mp4)
    cap = cv2.VideoCapture(video)
    # current directory my video load (soccer video input)
    i=0
    while(cap.isOpened()):
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
        if(i < 10 ):
            left.save(left_path + vname[0] + '_left_000000' +str(i)+'.png') # save left image(PIL)
            right.save(right_path + vname[0] + '_right_000000' +str(i)+'.png') # save predict right image (PIL)
        elif i < 100 :
            left.save(left_path + vname[0] + '_left_00000' + str(i) + '.png')  # save left image(PIL)
            right.save(right_path + vname[0] + '_right_00000' + str(i) + '.png')  # save predict right image (PIL)
        elif i < 1000:
            left.save(left_path + vname[0] + '_left_0000' + str(i) + '.png')  # save left image(PIL)
            right.save(right_path + vname[0] + '_right_0000' + str(i) + '.png')  # save predict right image (PIL)
        elif i < 10000:
            left.save(left_path + vname[0] + '_left_000' + str(i) + '.png')  # save left image(PIL)
            right.save(right_path + vname[0] + '_right_000' + str(i) + '.png')  # save predict right image (PIL)
        else :
            left.save(left_path + vname[0] + '_left_00' + str(i) + '.png')  # save left image(PIL)
            right.save(right_path + vname[0] + '_right_00' + str(i) + '.png')  # save predict right image (PIL)
        
        #generate gif (imageio open library)
        #images = []
        #images.append(imageio.imread('result/test3_left_'+str(i)+'.png'))
        #images.append(imageio.imread('result/test3_right_'+str(i)+'.png'))
        #imageio.mimsave('gif/test3_'+str(i)+'.gif',images,duration=0.01)
        i += 1
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

def merge(imagepath,vname):
    left_path = imagepath+'_left/'
    file_list = os.listdir(left_path)
    for fname in file_list:
        images = []
        img = Image.open(left_path + fname)
        images.append(img)
        sname = fname.split('_')
        img2 = Image.open(imagepath + '_right/' + vname + '_right_' + sname[len(sname)-1])
        images.append(img2)
        new_img = Image.new("RGB", (384 * 2, 160), 'white')
        width = 0
        for index in images:
            new_img.paste(index, (width, 0))
            width += 384
        new_img.save(imagepath + '/'+ vname + '_result_' + sname[len(sname)-1])


def convert_frames_to_video(imagepath, output, fps):
    frame_array = []
    files = [f for f in os.listdir(imagepath) if isfile(join(imagepath, f))]
    files.sort()
# width = 0
#height = 0
    for i in range(len(files)):
        filename = imagepath + '/' + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape

        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'XVID'), fps, (384*2, 160))

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def main(video,imagepath,output):
    if not os.path.isdir(imagepath):
        os.mkdir(imagepath)
    make_right(video,imagepath)
    
    v_name = video.split('.')
    output += '/'+ v_name[0] + '_result.avi'
    merge(imagepath,v_name[0])
    fps = 26.0
    convert_frames_to_video(imagepath, output, fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert movie to 3D')
    parser.add_argument('video', type=str, default='test.avi', help='convert video(current path)')
    parser.add_argument('imagepath', type=str, default='imagepath', help='save image path(empty folder or no exist folder')
    parser.add_argument('output', type=str, default='.', help='Output 3d video path(empty folder or no exist folder')
    args = parser.parse_args()

    main(args.video,args.imagepath,args.output)
