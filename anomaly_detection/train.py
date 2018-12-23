import torch
import os
import numpy as np
import argparse
from torch.autograd import Variable
import data_preprocess
from os.path import exists

import psuedo3dresnet
from psuedo3dresnet import P3D63, P3D131, P3D199
from data_preprocess import frames, make_segments, videos_array, 

parser = argparse.ArgumentParser(description="Inputs to the code")
parser.add_argument("--anomaly_videos_path",type = str ,default="/home/saivinay/Documents/jipmer-crowd-analysis/anomaly_detection/dataset/anomaly_videos",help="path of anomaly videos")
parser.add_argument("--normal_videos_path",type = str ,default="/home/saivinay/Documents/jipmer-crowd-analysis/anomaly_detection/dataset/normal_videos",help="path of normal videos")
parser.add_argument("--no_of_videos", type = int, default = 1, help = "the number of videos of each anomaly and normal that are used for training atmost = total no of videos of each type")
parser.add_argument("--num_epochs",type=float,default=1000,help="num of epochs for training")
parser.add_argument("--learning_rate",type=float,default=0.001,help="learning rate for training")
parser.add_argument("--ckpt_savedir",type = str,default='/home/saivinay/Documents/jipmer-crowd-analysis/anomaly_detection/checkpoints/',help="path to save checkpoints")
parser.add_argument("--load_ckpt",type = str,default='/home/saivinay/Documents/jipmer-crowd-analysis/anomaly_detection/checkpoints/',help="path to load checkpoints ")
parser.add_argument("--save_freq",type = int,default=50,help="save frequency")
parser.add_argument("--display_step",type = int,default=1,help="display frequency")
parser.add_argument("--summary_freq",type = int,default=50,help="summary writer frequency")

parser.add_argument("--log_directory",type = str,default='./log_dir',help="path to tensorboard log")
parser.add_argument("--batch_size",type=int,default=8,help="Batch Size")
parser.add_argument("--no_iterations",type=int,default=50000,help="number of iterations for training")
args = parser.parse_args()


lambda_1 = torch.tensor(8e-5, requires_grad = False)
lambda_2 = torch.tensor(8e-5, requires_grad = False)


class Loss(torch.nn.Module):
    
    def __init__(self, lambda_1, lambda_2, max_anomaly_score = 0, max_normal_score = 0 , anomaly_scores=[]):
        super(Loss, self).__init__()

    def rankingLoss(self):
        return max(0, 1 - self.max_anomaly_score + self.max_normal_score)


    def temporal_smoothness(self):
        n = len(self.anomaly_scores)
        sum_ = 0
        for i in range(n-1):
            sum_ += torch.pow(torch.pow(self.anomaly_scores[i+1], 2) - torch.pow(self.anomaly_scores[i], 2) , 2)

        return sum_


    def sparcity(self):
        return torch.sum(self.anomaly_scores)

    def forward(self):
        loss =  self.rankingLoss() + self.lambda_1*self.temporal_smoothness() + self.lambda_2*self.sparcity() 
        return loss




def Train():

    model = P3D199(pretrained=False, modality='RGB')

    if exists(args.load_ckpt):
        checkpoint = torch.load(args.load_ckpt)
        model.load_state_dict(checkpoint)

    optimizer = torch.optim.Adagrad(model.parameters(), lr = args.learning_rate, momentum=0.9)

    for i in range(args.num_epochs):

        anomaly_videos = videos_array(args.anomaly_videos_path)
        normal_videos = videos_array(args.normal_videos_path)
        optimizer.zero_grad()        

        for j in range(len(videos_array(args.anomaly_videos_path))):
            anomaly_scores = []
            normal_scores = []
            video_segments_anomaly, num_16segment_frames1 = make_segments(anomaly_videos[j])
            video_segments_normal, num_16segment_frames2 = make_segments(normal_videos[j])
            
            for k in range(30):
                temp1 = []
                temp2 = []
                for l in range(num_16segment_frames1):
                    anomaly_score = model(video_segments_anomaly[k][l])
                    temp1.append(anomaly_score)
                for l in range(num_16segment_frames2):
                    normal_score = model(video_segments_normal[k][l])
                    temp2.append(normal_score)
            
                anomaly_score = sum(temp1)/float(len(temp1))
                normal_score = sum(temp2)/float(len(temp2))

                anomaly_scores.append(anomaly_score)
                normal_scores.append(normal_score)    

            max_anomaly_score = max(anomaly_scores)
            max_normal_score = max(normal_scores)
    
        loss = Loss(lambda_1, lambda_2, max_anomaly_score, max_normal_score, anomaly_scores)
        print("===> Epoch[{}]  Loss: {:.4f}".format(i, loss.item()))
        
        loss.backwards()
        optimizer.step()


    torch.save(model.state_dict(), args.ckpt_savedir)


if __name__=="__main__":
    Train()