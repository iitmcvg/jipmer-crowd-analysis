import os
import numpy as np
import cv2
import glob
import scipy.io
import torch 
import argparse

parser = argparse.ArgumentParser(description="Inputs to code")
parser.add_argument("--video_path", type = str, default = "videoplayback")
args = parser.parse_args()

video = args.video_path

def make_segment():
    cap = cv2.VideoCapture(video)
    flag = 0
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if flag==0:
            video_segment = frame
            flag = 1
            print(frame.shape)

        if ret==False:
            print(video_segment.shape, False)
            video_segment = np.expand_dims(video_segment, axis = 0 )
            video_segment = np.expand_dims(video_segment, axis = 0 )
            video_segment = np.swapaxes(video_segment, 3, 4)
            video_segment = np.swapaxes(video_segment, 2, 3)
            video_segment = np.resize(video_segment,(231, 1, 3, 176, 320))
            video_segment = np.swapaxes(video_segment, 1, 2)
                         
            return (torch.from_numpy(video_segment)).type('torch.FloatTensor')

        video_segment = np.concatenate((video_segment, frame), axis = 2 )

        
        # a = a.type('torch.DoubleTensor')
        i+=1
        # print(i, ret)
        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    video_segment = np.expand_dims(video_segment, axis = 0 )
    video_segment = np.expand_dims(video_segment, axis = 0 )
    video_segment = np.swapaxes(video_segment, 3, 4)
    video_segment = np.swapaxes(video_segment, 2, 3)
    video_segment = np.resize(video_segment,(231, 1, 3, 176, 320))
    video_segment = np.swapaxes(video_segment, 1, 2)
    return (torch.from_numpy(video_segment)).type('torch.FloattTensor')

if __name__ == "__main__":
    
    video_segment = make_segment()
    video_segment = np.expand_dims(video_segment, axis = 0 )
    video_segment = np.expand_dims(video_segment, axis = 0 )
    video_segment = np.swapaxes(video_segment, 3, 4)
    video_segment = np.swapaxes(video_segment, 2, 3)
    video_segment = np.resize(video_segment,(231, 1, 3, 176, 320))
    video_segment = np.swapaxes(video_segment, 1, 2)            


    print(video_segment.shape)

    