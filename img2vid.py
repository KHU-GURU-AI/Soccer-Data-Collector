import cv2
import numpy as np
import os

from os.path import isfile, join

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    files.sort()
    #for sorting the file names properly
    # files.sort(key = lambda x: int(x[5:-4]))

    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    # out = cv2.VideoWriter(pathOut, -1, fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        # frame_array[i] = cv2.cvtColor(frame_array[i], cv2.COLOR_RGB2BGR)
        out.write(frame_array[i])
    out.release()

def main():
    #os.chdir('/')
    pathIn= '/home/kym/IdeaProjects/img100/'
    pathOut = '/home/kym/IdeaProjects/video.avi'
    fps = 50.0
    convert_frames_to_video(pathIn, pathOut, fps)

if __name__=="__main__":
    main()