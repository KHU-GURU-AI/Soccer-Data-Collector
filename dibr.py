import os
import numpy as np
import cv2
#import 뎁스 이미지 받아오는 라이브러리
"""
deep3d 에서 sym 부분을 받아와야함 
토치 사용할 예정 mxnet...?
"""
import data
import argparse
import logging
import mxnet as mx


def dibr(model_path , ctx, source , fname):
    from parse import anaglyph, sbs

    logging.basicConfig(level=logging.DEBUG)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DIBR')
    parser.add_argument('model_path', type=str, help='아직모름')
    parser.add_argument('--ctx',type=str,help='gpu id ')
    parser.add_argument('--source',type=str,help='none')
    parser.add_argument('--output',type=str,help='none')
    args = parser.parse_args()

    dibr(args.model_path,mx.gpu(args.ctx), args.source, args.output)