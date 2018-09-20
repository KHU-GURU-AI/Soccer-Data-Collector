"""
Encoder

video to image sequence
"""

import os
import sys

import cv2

"""
Encoder

video to images of frames
"""
class Encoder():
    def __init__(self, video_dir, save_path=None, sec=1):
        self.video_dir = video_dir
        if save_path is None:
            self.save_path = os.path.join(video_dir, "real")
        else:
            self.save_path = save_path
        self.sec = sec

        self.videos = [os.path.join(video_dir, file_name) for file_name in os.listdir(video_dir)]

        # check save directory 
        if os.path.exists(self.save_path):
            if not os.path.isdir(self.save_path):
                raise Exception("save_path {} is not directoy.".format(self.save_path))
        else:
            os.mkdir(self.save_path, mode=0o0755)
    
    def encoding(self, video_path, file_header=None):
        if file_header is None:
            file_header = video_path

        video = cv2.VideoCapture(video_path)
        fps = video.get(5)
        max_frame = video.get(7)

        print("[{}] fps: {}, max_frame: {}".format(file_header, fps, max_frame))

        spend_time = 0
        cur_frame = 0
        while cur_frame <= max_frame:
            video.set(1, cur_frame)
            _, frame = video.read()
            cv2.imwrite(os.path.join(self.save_path, \
                                        "{}_{}.jpg".format(file_header, spend_time)), frame)
            spend_time += 1
            cur_frame = spend_time * fps


        video.release()
        cv2.destroyAllWindows()

    def encode_all(self, report=True):
        for i, file_name in enumerate(self.videos):
            if report:
                print("Start {}".format(file_name))
            self.encoding(file_name, i)


if __name__ == '__main__':
    en = Encoder("/home/kairos/Videos/4K Video Downloader/")
    en.encode_all()
