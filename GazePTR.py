import torch
import torch.nn as nn 
import numpy as np
import math
import copy
from IVModule import Backbone, Transformer, PoseTransformer, TripleDifferentialProj, PositionalEncoder


def ep0(x):
    return x.unsqueeze(0)       
        
class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        # used for origin.
        transIn = 128
        convDims = [64, 128, 256, 512]

        # norm only used for gaze estimation.
        self.bnorm = Backbone(1, transIn, convDims)
        
        # MLP for gaze estimation
        self.MLP_n_dir = nn.Linear(transIn, 2)
        
        module_list = []
        for i in range(len(convDims)):
            module_list.append(nn.Linear(transIn, 2))
        self.MLPList_n = nn.ModuleList(module_list)
                   
        # Loss function     
        self.loss_op_re = nn.L1Loss()
    

    def forward(self, x_in, train=True):

        # feature [outFeatureNum, Batch, transIn], MLfeatgure: list[x1, x2...]
        feature_n, feature_list_n = self.bnorm(x_in['norm_face'])
 
        # Get feature for different task
        # [5, 128] [1. 5, 128]
        feature_n_dir = feature_n.squeeze()

        # estimate gaze from fused feature
        gaze = self.MLP_n_dir(feature_n_dir)
        # zone = self.MLP_o_zone(feature_o_zone)

        # for loss caculation
        loss_gaze_n = []
        if train:
            for i, feature in enumerate(feature_list_n):
                loss_gaze_n.append(self.MLPList_n[i](feature))
        
        return gaze, None, None, loss_gaze_n

    def loss(self, x_in, label):

        gaze, _, _, loss_gaze_n = self.forward(x_in)

        loss1 = 2 * self.loss_op_re(gaze, label.normGaze)

        loss2 = 0
        # for zone in zones:
        #    loss2 += (0.2/3) * self.loss_op_cls(zone, label.zone.view(-1))

        loss3 = 0
        # for pred in loss_gaze_o:
        #    loss3 += self.loss_op_re(pred, label.originGaze)

        loss4 = 0
        for pred in loss_gaze_n:
            loss4 += self.loss_op_re(pred, label.normGaze)
        loss = loss1 + loss2 + loss3 + loss4

        return loss, [loss1, loss2, loss3, loss4]


if __name__ == '__main__':
    x_in = {'origin': torch.zeros([5, 3, 224, 224]).cuda(), 
        'norm': torch.zeros([5, 3, 224, 224]).cuda(),
        'pos': torch.zeros(5, 2, 6).cuda()
        }

    model = Model()
    model = model.to('cuda')
    print(model)
    a = model(x_in)
    print(a)
