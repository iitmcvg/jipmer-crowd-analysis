import os
import numpy as np
import cv2
import glob
import scipy.io
import torch 
import argparse

parser = argparse.ArgumentParser(description="Inputs to code")
<<<<<<< HEAD
parser.add_argument("--video_path", type = str, default = "/home/saivinay/Documents/jipmer-crowd-analysis/anomaly_detection/videoplayback")
parser.add_argument("--normal_videos", type = str, default = "/home/saivinay/Documents/jipmer-crowd-analysis/anomaly_detection/dataset/normal_videos/")
parser.add_argument("--anomaly_videos", type = str, default = "/home/saivinay/Documents/jipmer-crowd-analysis/anomaly_detection/dataset/anomaly_videos/")

=======
parser.add_argument("--video_path", type = str, default = "videoplayback")
>>>>>>> 32709fa60cd3ffe57a7efbf570fa2e02994ba2b3
args = parser.parse_args()

# video = args.video_path

def videos_array(path, array = []):
    for i in glob.glob(os.path.join(path,"*")):
        # print(i)
        array.append(i)
    return array

def make_segment(video):
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
    video_segment = np.resize(video_segment,(30, 1, 3, 160, 160))
    video_segment = np.swapaxes(video_segment, 1, 2)
    return (torch.from_numpy(video_segment)).type('torch.FloattTensor')

if __name__ == "__main__":
    
    anomaly_videos = videos_array(args.anomaly_videos)
    video_segment = make_segment(anomaly_videos[0])
    video_segment = np.expand_dims(video_segment, axis = 0 )
    video_segment = np.expand_dims(video_segment, axis = 0 )
    video_segment = np.swapaxes(video_segment, 3, 4)
    video_segment = np.swapaxes(video_segment, 2, 3)
    video_segment = np.resize(video_segment,(30, 1, 3, 160, 160))
    video_segment = np.swapaxes(video_segment, 1, 2)            

    print(video_segment.shape)

    