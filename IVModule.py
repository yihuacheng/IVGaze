import torch
import torch.nn as nn 
import numpy as np
import math
import copy
from resnet import resnet18

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def ep0(x):
    return x.unsqueeze(0)

class TripleDifferentialProj(nn.Module):

    def __init__(self):
        super(TripleDifferentialProj, self).__init__()

        # one 3D point in screen plane
        point = torch.Tensor([0, 0, 0])

        # normal vector of screen plane
        normal = torch.Tensor([0, 0, 1])

    def forward(self, gaze, origin, norm_mat = None):
        # inputted gaze is [pitch, yaw]  

        gaze3d = self.gazeto3d(gaze)
        if norm_mat != None:
            gaze3d = torch.einsum('acd,ad->ac', norm_mat, gaze3d)

        gazex = self.gazeto2dPlus(gaze3d, origin, 0)
        gazey = self.gazeto2dPlus(gaze3d, origin, 1)
        gazez = self.gazeto2dPlus(gaze3d, origin, 2)
        gaze = [gazex, gazey, gazez] 
        return gaze

    def gazeto3d(self, point):
        # Yaw Pitch, Here   
        x = -torch.cos(point[:, 1]) * torch.sin(point[:, 0])
        y = -torch.sin(point[:, 1])
        z = -torch.cos(point[:, 1]) * torch.cos(point[:, 0])
        gaze = torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)], 1)
        return gaze

    def gazeto2dPlus(self, gaze, origin, plane: int):

        assert plane < 3, 'plane should be 0(x), 1(y) or 2(z)'
        length = origin[:, plane]
        g_len = gaze[:, plane]
        scale = -length / g_len
        gaze = torch.einsum('ik, i->ik', gaze, scale)
        point = origin + gaze
        return point
    
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class PoseTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        return src + pos
        

    def forward(self, src, pos):
                # src_mask: Optional[Tensor] = None,
                # src_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos
        

    def forward(self, src, pos):
                # src_mask: Optional[Tensor] = None,
                # src_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class conv1x1(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(conv1x1, self).__init__()
        
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

        self.bn = nn.BatchNorm2d(out_planes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 

    def forward(self, feature):
        output = self.conv(feature)
        output = self.bn(output)
        output = self.avgpool(output)
        output = output.squeeze()
 
        return output
        


class MLEnhance(nn.Module):

    def __init__(self, input_nums, hidden_num):

        super(MLEnhance, self).__init__()

        # input_nums: [64, 128, 256, 512]
        length = len(input_nums)

        self.input_nums = input_nums
        self.hidden_num = hidden_num
        
        self.length = length

        layerList = []

        for i in range(length):
            layerList.append(self.__build_layer(input_nums[i], hidden_num))

        self.layerList = nn.ModuleList(layerList)


    def __build_layer(self, input_num, hidden_num):
        layer = conv1x1(input_num, hidden_num)
        
        return layer

    def forward(self, feature_list):

        out_feature_list = []

        out_feature_gather =[]

        for i, feature in enumerate(feature_list):
            result = self.layerList[i](feature)

            # Dim [B, C] -> [1, B, C]
            out_feature_list.append(result)

            out_feature_gather.append(ep0(result))

            
        # [L, B, C]
        feature = torch.cat(out_feature_gather, 0)
        return feature, out_feature_list
        
class PositionalEncoder():
    # encode low-dim, vec to high-dims.

    def __init__(self, number_freqs, include_identity=False):
        freq_bands = torch.pow(2, torch.linspace(0., number_freqs - 1, number_freqs))
        self.embed_fns = []
        self.output_dim = 0

        if include_identity:
            self.embed_fns.append(lambda x:x)
            self.output_dim += 1

        for freq in freq_bands:
            for transform_fns in [torch.sin, torch.cos]:
                self.embed_fns.append(lambda x, fns=transform_fns, freq=freq: fns(x*freq))
                self.output_dim += 1

    def encode(self, vecs):
        # inputs: [B, N]
        # outputs: [B, N*number_freqs*2]
        return torch.cat([fn(vecs) for fn in self.embed_fns], -1)

    def getDims(self):
        return self.output_dim

class PoseTransformer(nn.Module):

    def __init__(self, input_dim, pos_length, nhead=8, hidden_dim=512, layer_num = 6, 
                pos_freq=30, pos_ident=True, pos_hidden=128, dropout=0.1):

        super(PoseTransformer, self).__init__()
        # input feature + added token

        # The input feature should be [L, Batch, Input_dim]
        encoder_layer = PoseTransformerEncoderLayer(
                  input_dim, 
                  nhead = nhead, 
                  dim_feedforward = hidden_dim, 
                  dropout=dropout)

        encoder_norm = nn.LayerNorm(input_dim) 
        self.encoder = TransformerEncoder(encoder_layer, num_layers = layer_num, norm = encoder_norm)

        self.pos_embedding = PositionalEncoder(pos_freq, pos_ident)

        out_dim = pos_length * (pos_freq * 2 + pos_ident)

        self.pos_encode = nn.Sequential(
            nn.Linear(out_dim, pos_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(pos_hidden, pos_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(pos_hidden, pos_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(pos_hidden, input_dim)
        )
       
    def forward(self, feature, pos_feature):

        """
        Inputs:
            feature: [length, batch, dim1]
            pos_feature: [batch, length, dim2]

        Outputs:
            feature: [batch, length, dim1]


        """

        # feature: [Length, Batch, Dim]
        pos_feature = self.pos_embedding.encode(pos_feature)
        pos_feature = self.pos_encode(pos_feature)
        pos_feature = pos_feature.permute(1, 0, 2)

        # feature [Length, batch, dim]
        feature = self.encoder(feature, pos_feature)
        # feature = feature.permute(1, 0, 2)

        return feature

class Transformer(nn.Module):

    def __init__(self, input_dim, nhead=8, hidden_dim=512, layer_num = 6, pred_num=1, length=4, dropout=0.1):

        super(Transformer, self).__init__()

        self.pnum = pred_num
        # input feature + added token
        # self.length = length + 1
        self.length = length

        # The input feature should be [L, Batch, Input_dim]
        encoder_layer = TransformerEncoderLayer(
                  input_dim,
                  nhead = nhead,
                  dim_feedforward = hidden_dim,
                  dropout=dropout)

        encoder_norm = nn.LayerNorm(input_dim)

        self.encoder = TransformerEncoder(encoder_layer, num_layers = layer_num, norm = encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(pred_num, 1, input_dim))

        self.token_pos_embedding = nn.Embedding(pred_num, input_dim)
        self.pos_embedding = nn.Embedding(length, input_dim)


    def forward(self, feature, num = 1):

        # feature: [Length, Batch, Dim]
        batch_size = feature.size(1)

        # cls_num, 1, Dim -> cls_num, Batch_size, Dim

        feature_list = []

        for i in range(num):

            cls = self.cls_token[i, :, :].repeat((1, batch_size, 1))

            feature_in = torch.cat([cls, feature], 0)

            # position
            position = torch.from_numpy(np.arange(self.length)).cuda()
            pos_feature = self.pos_embedding(position)

            token_position = torch.Tensor([i]).long().cuda()
            token_pos_feature = self.token_pos_embedding(token_position)

            pos_feature = torch.cat([pos_feature, token_pos_feature], 0)

            # feature [Length, batch, dim]
            feature_out = self.encoder(feature_in, pos_feature)

            # [batch, dim, length]
            # feature = feature.permute(1, 2, 0)

            # get the first dimension, [pnum, batch, dim]
            feature_out = feature_out[0:1, :, :]
            feature_list.append(feature_out)

        return torch.cat(feature_list, 0).squeeze()
      
class Backbone(nn.Module):

    def __init__(self, outFeatureNum=1, transIn=128, convDims=[64, 128, 256, 512]):

        super(Backbone, self).__init__()

        self.base_model = resnet18(pretrained=True, input_dim = convDims)

        self.transformer = Transformer(input_dim = transIn, nhead = 8, hidden_dim=512, 
                                layer_num=6, pred_num = outFeatureNum, length = len(convDims))

        # convert multi-scale feature 
        self.mle = MLEnhance(input_nums = convDims, hidden_num = transIn)

        self.feed = nn.Linear(transIn, 2)
            
        self.loss_op = nn.L1Loss()

        self.loss_layerList = []

        self.outFeatureNum = outFeatureNum

        # for i in range(len(convDims)): 
        #    self.loss_layerList.append(nn.Linear(tranIn, 2))

    def forward(self, image):

        x1, x2, x3, x4 = self.base_model(image) 

        feature, feature_list = self.mle([x1, x2, x3, x4])

        #t_gaze = []
        # for t_feature in feature_list:
        #    t_gaze.append(self.loss_layerList(t_feature))

        feature = self.transformer(feature, self.outFeatureNum)
         
        # gaze = self.feed(feature)
        
        return feature, feature_list

