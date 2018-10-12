"""
Encoder

video to image sequence
"""

import os
import sys
import threading
import argparse

import cv2

"""
Encoder

video to images of frames
"""
class Encoder(threading.Thread):
    def __init__(self, video_path, save_path, interval=1):
        threading.Thread.__init__(self)

        self.video_path = os.path.abspath(video_path)
        assert os.path.exists(self.video_path)
        assert os.path.isfile(self.video_path)

        if save_path is None:
            self.save_path = os.path.join(video_path, "real")
        else:
            self.save_path = os.path.abspath(save_path)

        self.interval = interval
    
    def run(self):
        file_name = self.video_path.split('/')[-1][:-4]
        save_path = os.path.join(self.save_path, file_name)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        video = cv2.VideoCapture(self.video_path)
        fps = video.get(5)
        max_frame = video.get(7)

        # print("[{:.10s}] fps: {:.2f}, max_frame: {:f}".format(file_name, fps, max_frame))

        spend_time = 0
        cur_frame = 0
        while cur_frame <= max_frame:
            video.set(1, cur_frame)
            _, frame = video.read()
            cv2.imwrite(os.path.join(save_path, \
                                     "{}.jpg".format(spend_time)), frame)

            if spend_time % 100 == 0:
                print("[{:<.30}] \t Progress : {:.2f} %".format(file_name, 100*spend_time/(max_frame/fps)))

            spend_time += self.interval
            cur_frame = spend_time * fps * self.interval

        video.release()
        cv2.destroyAllWindows()


class Video_to_Images():
    def __init__(self, video_dir, save_dir, interval=1, workers=1):
        self.video_dir = video_dir
        self.save_dir = save_dir
        # check save directory 
        if os.path.exists(self.save_dir):
            assert os.path.isdir(self.save_dir)
        else:
            os.mkdir(self.save_dir, mode=0o0755)
        self.interval = interval
        self.workers = workers

        self.videos = [os.path.join(video_dir, file_name) for file_name in os.listdir(self.video_dir)]
        self.video_count = len(self.videos)

        self.worker_list = []

    def run(self):
        cur = 0

        for _ in range(self.workers):
            w = Encoder(self.videos[cur], self.save_dir, self.interval)
            w.start()
            self.worker_list.append(w)
            cur += 1
        
        while cur > self.video_count:
            for i, w in enumerate(self.worker_list):
                if not w.isAlive():
                    w = Encoder(self.videos[cur], self.save_dir, self.interval)
                    w.start()
                    self.worker_list[i] = w
                    cur += 1
                    break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('video_dir', type=str)
    parser.add_argument('--save_dir', type=str, default='./real')
    parser.add_argument('-i', '--interval', type=int, default=1)
    parser.add_argument('-w', '--workers', type=int, default=1)
    args = parser.parse_args()

    v2i = Video_to_Images(args.video_dir, args.save_dir, interval=args.interval, workers=args.workers)
    v2i.run()
