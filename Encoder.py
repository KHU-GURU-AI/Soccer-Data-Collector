"""
Encoder

video to image sequence
"""

import os
import sys
import cv2

class Encoder():
    def __init__(self, video_path, save_path='same', fps=60):
        self.video_path = video_path
        if save_path == 'same':
            self.save_path = video_path.split('.')[0]
        else:
            self.save_path = save_path
        self.fps = fps

        # check save directory 
        if os.path.exists(self.save_path):
            if not os.path.isdir(self.save_path):
                raise Exception("save_path is not directoy.")
        else:
            os.mkdir(self.save_path, mode=0o0755)
    
    def encoding(self):
        video = cv2.VideoCapture(self.video_path)

        global_frames = 0
        while video.isOpened():
            _, frame = video.read()
            if global_frames % self.fps == 0:
                cv2.imwrite(os.path.join(self.save_path, "frame_{}.jpg".format(global_frames)), frame)
                
            global_frames += 1
        
        video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    en = Encoder("/home/kairos/Videos/4K Video Downloader/2017 U-20 월드컵 대표팀 평가전- U-20 대표팀 vs 전북현대모터스 Full.mp4")
    en.encoding()
