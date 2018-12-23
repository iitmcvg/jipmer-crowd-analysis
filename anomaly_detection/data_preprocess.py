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

def videos_array(path):                                 # make an array of video paths 
    array = []
    for i in glob.glob(os.path.join(path,"*")):
        # print(i)
        array.append(i)
    return array


def frames(video):                                      # calculating the number of frames in a video
    cap = cv2.VideoCapture(video)
    num_frames = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==False:
            break
        num_frames += 1
    # print(num_frames)
    return num_frames


def make_segments(video):                                    
    num_frames = frames(video)
    num_frames_per_segment = num_frames//30
    num_16segment_frames  = num_frames_per_segment//16
    video_segments = [[]]
    # print(num_frames, n)
    cap = cv2.VideoCapture(video)
    print(num_frames, num_16segment_frames, num_frames_per_segment)

    while(cap.isOpened()):
        for i in range(30):
            for j in range(num_16segment_frames):
                print(j, i)
                temp = []
                for k in range(16):
                    ret, frame = cap.read()                 # i is iterator of segments from the complete video
                    print(ret)                              # j is iterator of 16 frames segments in a segmented video(30 segments) from a complete video
                    if ret==False:                          # k is iterator for each frame
                        return video_segments, num_16segment_frames
                    if k == 0:
                        video_segments[i].append(frame)     ############## To be corrected
                    else:
                        video_segments[i][j] = np.concatenate((video_segments[i][j], frame), axis = 2 )
                video_segments[i][j] = np.expand_dims(video_segments[i][j], axis = 0 )
                video_segments[i][j] = np.expand_dims(video_segments[i][j], axis = 0 )
                video_segments[i][j] = np.swapaxes(video_segments[i][j], 3, 4)
                video_segments[i][j] = np.swapaxes(video_segments[i][j], 2, 3)
                video_segments[i][j] = np.resize(video_segments[i][j],(16, 1, 3, 160, 160))
                video_segments[i][j] = np.swapaxes(video_segments[i][j], 1, 2)
                video_segments[i][j] = torch.from_numpy(video_segments[i][j]).type('torch.FloatTensor')
                print(video_segments[i][j].shape, num_16segment_frames)        

    
    return video_segments, num_16segment_frames

        



if __name__ == "__main__":


    anomaly_videos = videos_array(args.anomaly_videos)
    normal_videos  = videos_array(args.normal_videos)
    video_segments = make_segments(normal_videos[1])
    print(i, k, len(video_segments), len(video_segments[0]), len(video_segments[0][0]), len(video_segments[0][0][0])  )
    # print(video_segments)

'''
# def from_video_segments(video_segment):                 # takes a video segment and makes it into smaller segments of 16 frames                
#                                                         # each to pass to the model
#     from_segments = []                                  
#     cap = cv2.VideoCapture(video_segment)                             
    
#     i=0

#     while (cap.isOpened()):
#         for j in range(16):
#             ret, frame = cap.read()
#             if ret == False:
#                 print(from_segments[i].shape)
#                 return from_segments

#             if j == 0:
#                 from_segments.append(frame)
#             else:
#                 from_segments[i] = np.concatenate((from_segments[i], frame), axis = 2 )
        
#         # cv2.imshow("frame", frame)
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break
#         print(i, ret)
        
#         from_segments[i] = np.expand_dims(from_segments[i], axis = 0 )
#         from_segments[i] = np.expand_dims(from_segments[i], axis = 0 )
#         from_segments[i] = np.swapaxes(from_segments[i], 3, 4)
#         from_segments[i] = np.swapaxes(from_segments[i], 2, 3)
#         from_segments[i] = np.resize(from_segments[i],(16, 1, 3, 160, 160))
#         from_segments[i] = np.swapaxes(from_segments[i], 1, 2)
#         from_segments[i] = torch.from_numpy(from_segments[i]).type('torch.FloatTensor')               
        
#         print(from_segments[i].shape, "from segments")
#         print(i, "from segments")
#         i+=1
#         return from_segments
'''