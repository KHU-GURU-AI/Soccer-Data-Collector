from mxnet.recordio import *

"""
전부 수정 예정 
"""
def split(frame, reshape=(256,144), vert=True, clip=None):
    if vert == True:
        lframe = frame[:, :frame.shape[1]/2]
        rframe = frame[:, frame.shape[1]/2:]
    elif vert == False:
        lframe = frame[:frame.shape[0]/2, :]
        rframe = frame[frame.shape[0]/2:, :]
    else:
        lframe = frame
        rframe = frame
    if clip is not None:
        lframe = lframe[clip[1]:clip[3], clip[0]:clip[2]]
        rframe = rframe[clip[1]:clip[3], clip[0]:clip[2]]
    if reshape is not None:
        lframe = cv2.resize(lframe, reshape)
        rframe = cv2.resize(rframe, reshape)
    return lframe, rframe
