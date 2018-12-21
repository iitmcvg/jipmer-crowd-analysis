import os
import numpy as np
import cv2
import glob
import scipy.io
import torch 
import argparse

parser = argparse.ArgumentParser(description="Inputs to code")

parser.add_argument("--video_path", type = str, default = "/home/saivinay/Documents/jipmer-crowd-analysis/anomaly_detection/videoplayback")
parser.add_argument("--normal_videos", type = str, default = "/home/saivinay/Documents/jipmer-crowd-analysis/anomaly_detection/dataset/normal_videos/")
parser.add_argument("--anomaly_videos", type = str, default = "/home/saivinay/Documents/jipmer-crowd-analysis/anomaly_detection/dataset/anomaly_videos/")

args = parser.parse_args()

# video = args.video_path

def videos_array(path, array = []):
    for i in glob.glob(os.path.join(path,"*")):
        # print(i)
        array.append(i)
    return array


def make_segment1(video):       # making videos of 16 frames each from a video            
    
    video_segments = []
    cap = cv2.VideoCapture(video)
    
    i=0

    while (cap.isOpened()):
        for j in range(16):
            ret, frame = cap.read()

            if ret == False:
                print(video_segments[i].shape)
                return video_segments
 
            if j == 0:
                video_segments.append(frame)
            else:
                video_segments[i] = np.concatenate((video_segments[i], frame), axis = 2 )
        
        # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # print(i, ret)
        
        video_segments[i] = np.expand_dims(video_segments[i], axis = 0 )
        video_segments[i] = np.expand_dims(video_segments[i], axis = 0 )
        video_segments[i] = np.swapaxes(video_segments[i], 3, 4)
        video_segments[i] = np.swapaxes(video_segments[i], 2, 3)
        video_segments[i] = np.resize(video_segments[i],(16, 1, 3, 160, 160))
        video_segments[i] = np.swapaxes(video_segments[i], 1, 2)
        video_segments[i] = torch.from_numpy(video_segments[i]).type('torch.FloatTensor')               
        
        print(video_segments[i].shape)
        print(i)
        i+=1


def make_segment2(video):            # using the total video
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
            video_segment = np.resize(video_segment,(30, 1, 3, 160, 160))
            video_segment = np.swapaxes(video_segment, 1, 2)
                         
            return (torch.from_numpy(video_segment)).type('torch.FloatTensor')

        video_segment = np.concatenate((video_segment, frame), axis = 2 )

        
        # a = a.type('torch.DoubleTensor')
        i+=1

        print(i, ret)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_segment = np.expand_dims(video_segment, axis = 0 )
    video_segment = np.expand_dims(video_segment, axis = 0 )
    video_segment = np.swapaxes(video_segment, 3, 4)
    video_segment = np.swapaxes(video_segment, 2, 3)
    video_segment = np.resize(video_segment,(30, 1, 3, 160, 160))
    video_segment = np.swapaxes(video_segment, 1, 2)
    return (torch.from_numpy(video_segment)).type('torch.FloattTensor')

if __name__ == "__main__":

    anomaly_videos = videos_array(args.anomaly_videos)
    make_segment1(anomaly_videos[0])
    
    # video_segment = make_segment(anomaly_videos[0])
    # video_segment = np.expand_dims(video_segment, axis = 0 )
    # video_segment = np.expand_dims(video_segment, axis = 0 )
    # video_segment = np.swapaxes(video_segment, 3, 4)
    # video_segment = np.swapaxes(video_segment, 2, 3)
    # video_segment = np.resize(video_segment,(30, 1, 3, 160, 160))
    # video_segment = np.swapaxes(video_segment, 1, 2)            

    # print(video_segment.shape)

