import torch
import os
import numpy as np
import argparse
from torch.autograd import Variable
import psuedo3dresnet
import data_preprocess

from psuedo3dresnet import P3D63, P3D131, P3D199
from data_preprocess import video_segment

parser = argparse.ArgumentParser(description="Inputs to the code")
parser.add_argument("--batch_size",type=int,default=8,help="Batch Size")
parser.add_argument("--log_directory",type = str,default='./log_dir',help="path to tensorboard log")
parser.add_argument("--ckpt_savedir",type = str,default='/home/saivinay/Documents/jipmer-crowd-analysis/checkpoints/',help="path to save checkpoints")
parser.add_argument("--load_ckpt",type = str,default='/home/saivinay/Documents/jipmer-crowd-analysis/checkpoints/',help="path to load checkpoints from")
parser.add_argument("--save_freq",type = int,default=50,help="save frequency")
parser.add_argument("--display_step",type = int,default=1,help="display frequency")
parser.add_argument("--summary_freq",type = int,default=50,help="summary writer frequency")
parser.add_argument("--no_iterations",type=int,default=50000,help="number of iterations for training")
parser.add_argument("--learning_rate",type=float,default=0.001,help="learning rate for training")
parser.add_argument("--num_epochs",type=float,default=1000,help="num of epochs for training")
args = parser.parse_args()


def rankingLoss(max_anomaly_score, max_normal_score):
    return max(0, 1 - max_anomaly_score + max_normal_score)

def temporal_smoothness(anomaly_scores = []):
    n = len(anomaly_scores)
    sum_ = 0
    for i in range(n-1):
        sum_ += torch.pow(torch.pow(anomaly_scores[i+1], 2) - torch.pow(anomaly_scores[i], 2) , 2)

    return sum_

def sparcity(anomaly_scores = []):
    return torch.sum(anomaly_scores)

lambda_1  = Variable(torch.random(1), requires_grad = True) 
lambda_2 = Variable(torch.random(1), requires_grad = True) 

def totol_loss(lambda_1, lambda_2):
    loss =  rankingLoss + lambda_1*temporal_smoothness + lambda_2*sparcity 
    return loss


def Train():
    model = P3D199(pretrained=False, modality='RGB')

    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    loss = totol_loss(lambda_1, lambda_2)

    for i in range(args.num_epochs):

        anomaly_scores = []
        normal_scores = []
        num_videos = 0  # for keeping count of a certain type of video
        
        while(True):

            output = model(video_segment)
            
            if num_videos<40:
                anomaly_scores.append(output)
            else:
                normal_scores.append(output)
            
            

        max_anomaly_score = max(anomaly_scores)
        max_normal_score = max(normal_scores)

        loss_ = loss
        loss_.backwards()
        optimizer.step()






if __name__=="__main__":
    Train()