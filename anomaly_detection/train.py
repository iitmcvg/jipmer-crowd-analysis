import torch
import os
import numpy as np
import argparse
from torch.autograd import Variable
import psuedo3dresnet
import data_preprocess
from os.path import exists
from psuedo3dresnet import P3D63, P3D131, P3D199
from data_preprocess import make_segment

parser = argparse.ArgumentParser(description="Inputs to the code")
parser.add_argument("--batch_size",type=int,default=8,help="Batch Size")
parser.add_argument("--log_directory",type = str,default='./log_dir',help="path to tensorboard log")
parser.add_argument("--ckpt_savedir",type = str,default='/home/saivinay/Documents/jipmer-crowd-analysis/anomaly_detection/checkpoints/',help="path to save checkpoints")
parser.add_argument("--load_ckpt",type = str,default='/home/saivinay/Documents/jipmer-crowd-analysis/anomaly_detection/checkpoints/',help="path to load checkpoints from")
parser.add_argument("--save_freq",type = int,default=50,help="save frequency")
parser.add_argument("--display_step",type = int,default=1,help="display frequency")
parser.add_argument("--summary_freq",type = int,default=50,help="summary writer frequency")
parser.add_argument("--no_iterations",type=int,default=50000,help="number of iterations for training")
parser.add_argument("--learning_rate",type=float,default=0.001,help="learning rate for training")
parser.add_argument("--num_epochs",type=float,default=1000,help="num of epochs for training")
args = parser.parse_args()


lambda_1 = torch.randn(1, dtype=torch.float)
lambda_2 = torch.randn(1, dtype=torch.float)

# lambda_1  = Variable(torch.random(1), requires_grad = True) 
# lambda_2 = Variable(torch.random(1), requires_grad = True) 



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

    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

    for i in range(args.num_epochs):

        anomaly_scores = []
        normal_scores = []
        num_videos = 0           # for keeping count of a certain type of video
        optimizer.zero_grad()
        while(True):

            output = model(make_segment())
            
            if num_videos<40:
                anomaly_scores.append(output)
                num_videos +=1
            else:
                normal_scores.append(output)
            

        max_anomaly_score = max(anomaly_scores)
        max_normal_score = max(normal_scores)
    
        loss = Loss(lambda_1, lambda_2, max_anomaly_score, max_normal_score, anomaly_scores)
        print("===> Epoch[{}]  Loss: {:.4f}".format(i, loss.item()))
        
        loss.backwards()
        optimizer.step()


    torch.save(model.state_dict(), args.ckpt_savedir)


if __name__=="__main__":
    Train()