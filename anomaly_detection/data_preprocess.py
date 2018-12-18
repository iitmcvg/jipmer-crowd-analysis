import os
import numpy as np
import cv2
import glob
import scipy.io
import torch 

video = '/home/saivinay/Documents/jipmer-crowd-analysis/anomaly_detection/videoplayback'

def make_segment():
    cap = cv2.VideoCapture(video)
    flag = 0
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if flag==0:
            video_segment = frame
            flag = 1

        if ret==False:
            print(video_segment.shape)
            return torch.from_numpy(video_segment)

        video_segment = np.concatenate((video_segment, frame), axis = 2 )
        i+=1
        print(i, ret)
        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
    return torch.from_numpy(video_segment)

if __name__ == "__main__":
    
    video_segment = make_segment()
    