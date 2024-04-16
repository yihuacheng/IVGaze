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

        # origin produces two features, one for gaze zone, one for gaze direction.
        self.borigin = Backbone(2, transIn, convDims)
        
        # norm only used for gaze estimation.
        self.bnorm = Backbone(1, transIn, convDims)
        
        self.ptrans = PoseTransformer(transIn, 3)

        # Proj
        self.proj = TripleDifferentialProj()

        # 3 * (30 * 2 + 1)          
        self.PoseEncoder = PositionalEncoder(30, True)

        self.MLP_Pos_x = nn.Sequential(
                        nn.Linear(183, 128),
                        nn.ReLU(inplace=True),
            )

        self.MLP_Pos_y = nn.Sequential(
                        nn.Linear(183, 128),
                        nn.ReLU(inplace=True),
            )

        self.MLP_Pos_z = nn.Sequential(
                        nn.Linear(183, 128),
                        nn.ReLU(inplace=True),
            )

        self.MLP_Pos = [self.MLP_Pos_x, self.MLP_Pos_y, self.MLP_Pos_z]


        # Transformer to combine gaze point and feature
        self.tripleTrans = Transformer(
                input_dim = 128,
                length = 3,
                layer_num=2,
                nhead=4
            )
                
        self.zoneTrans = Transformer(
                input_dim = 128,
                length = 2,
                layer_num=2,
                nhead=4
            )
         
        # MLP for gaze estimation
        class_num = 122
    
        self.MLP_o_dir = nn.Linear(transIn, 2)
        self.MLP_n_dir = nn.Linear(transIn, 2)

        
        module_list = []
        for i in range(len(convDims)):
            module_list.append(nn.Linear(transIn, 2))
        self.MLPList_o  = nn.ModuleList(module_list) 
        
        module_list = []
        for i in range(len(convDims)):
            module_list.append(nn.Linear(transIn, 2))
        self.MLPList_n = nn.ModuleList(module_list)
                   
        self.MLP_o_dir2 = nn.Linear(transIn, 2)
        self.MLP_n_dir2 = nn.Linear(transIn, 2)

        self.MLP_o_zone = nn.Linear(transIn, class_num)
        self.MLP_o_zone2 = nn.Linear(transIn, class_num)
        self.MLP_o_zone3 = nn.Linear(transIn, class_num)

        # Loss function     
        self.loss_op_re = nn.L1Loss()
        self.loss_op_cls = nn.CrossEntropyLoss()
    

    def forward(self, x_in, train=True):

        # feature [outFeatureNum, Batch, transIn], MLfeatgure: list[x1, x2...]

        # Extract feature from both two images
        feature_o, feature_list_o= self.borigin(x_in['origin_face'])

        feature_n, feature_list_n = self.bnorm(x_in['norm_face'])
 
        # Get feature for different task
        # [5, 128] [1. 5, 128]
        feature_o_zone = feature_o[0,:]
        feature_o_dir = feature_o[1,:]

        feature_n_dir = feature_n.squeeze()

        zone1 = self.MLP_o_zone(feature_o_zone)
  
        # Fuse two direction feature and input it into transformer
        features_dir = torch.cat([ep0(feature_o_dir), ep0(feature_n_dir)], 0)
        features = self.ptrans(features_dir, x_in['pos'])

        # Get fused feature
        # feature_o_dir2 = features[0, :]
        feature_n_dir2 = features[1, :]
            
        # estimate gaze from fused feature
        gaze = self.MLP_n_dir2(feature_n_dir2)
        # zone = self.MLP_o_zone(feature_o_zone)

        # Proj
        gaze_proj = self.proj(gaze.detach(), x_in['gaze_origin'], x_in['norm_cam'])
        gaze_proj_list = []
        # gaze_proj_list.append(ep0(feature_o_zone))

        for i, gaze2d in enumerate(gaze_proj):
            gaze_proj_list.append(ep0(self.MLP_Pos[i](self.PoseEncoder.encode(gaze2d))))

        triple_feature = torch.cat(gaze_proj_list, 0)
        triple_feature = self.tripleTrans(triple_feature)
        zone2 = self.MLP_o_zone2(triple_feature)
         
        feature_o_zone2 = torch.cat([ep0(feature_o_zone), ep0(triple_feature)], 0) 
        feature_o_zone2 = self.zoneTrans(feature_o_zone2)
        zone3 = self.MLP_o_zone3(feature_o_zone2)
      
        zone = [zone1, zone2, zone3]

        # for loss caculation
        loss_gaze_o = []
        loss_gaze_n = []
        if train:
            loss_gaze_n.append(self.MLP_n_dir(feature_n_dir))
            loss_gaze_o.append(self.MLP_o_dir(feature_o_dir))
            
            for i, feature in enumerate(feature_list_o):
                loss_gaze_o.append(self.MLPList_o[i](feature))

            for i, feature in enumerate(feature_list_n):
                loss_gaze_n.append(self.MLPList_n[i](feature))
        
        else:
            zone = zone2.max(1)[1]

        return gaze, zone, loss_gaze_o, loss_gaze_n

    def loss(self, x_in, label):

        gaze, zones, loss_gaze_o, loss_gaze_n = self.forward(x_in)

        loss1 = 2 * self.loss_op_re(gaze, label.normGaze)

        loss2 = 0
        for zone in zones:
            loss2 += (0.2/3) * self.loss_op_cls(zone, label.zone.view(-1))

        loss3 = 0
        for pred in loss_gaze_o:
            loss3 += self.loss_op_re(pred, label.originGaze)

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
