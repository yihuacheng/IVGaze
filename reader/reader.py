import os
import cv2 
import torch
import random
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import copy

def gazeto2d(gaze):
    yaw = -np.arctan2(-gaze[0], -gaze[2])
    pitch = -np.arcsin(-gaze[1])
    return np.array([yaw, pitch])

def Decode_MPII(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[5], line[6]
    anno.gaze2d, anno.head2d = line[7], line[8]
    return anno

def Decode_IVOrigin(line):
    anno = edict()
    anno.face = line[0]
    anno.name = line[0]
    anno.gaze = line[1]
    anno.placeholder = line[2]
    anno.zone = line[3]
    # anno.target = line[4]
    anno.origin = line[5]
    return anno

def Decode_IVNorm(line):
    anno = edict()
    anno.face = line[0]
    anno.name = line[0]
    anno.gaze = line[1]
    anno.head = line[2]
    anno.zone = line[3]
    anno.origin = line[4]
    anno.norm = line[6]
    return anno


def Decode_Dict():
    mapping = edict()
    mapping.ivorigin = Decode_IVOrigin
    mapping.ivnorm = Decode_IVNorm
    return mapping

def long_substr(str1, str2):
    substr = ''
    for i in range(len(str1)):
        for j in range(len(str1)-i+1):
            if j > len(substr) and (str1[i:i+j] in str2):
                substr = str1[i:i+j]
    return len(substr)

def Get_Decode(name):
    mapping = Decode_Dict()
    keys = list(mapping.keys())
    name = name.lower()
    score = [long_substr(name, i) for i in keys]
    key  = keys[score.index(max(score))]
    return mapping[key]
    

class commonloader(Dataset): 

  def __init__(self, dataset):

    # Read source data
    self.source = edict() 
    self.source.origin = edict()
    self.source.norm = edict()

    # Read origin data
    origin = dataset.origin

    self.source.origin.root = origin.image 
    self.source.origin.line = self.__readlines(origin.label, origin.header)
    self.source.origin.decode = Get_Decode(origin.name)
    
    # Read norm data
    norm = dataset.norm

    # self.source.norm = copy.deepcopy(dataset.norm)
    self.source.norm.root = norm.image 
    self.source.norm.line = self.__readlines(norm.label, norm.header)
    self.source.norm.decode = Get_Decode(norm.name)
    
    # build transforms
    self.transforms = transforms.Compose([
        transforms.ToTensor()
    ])


  def __readlines(self, filename, header=True):

    data = []
    if isinstance(filename, list):
      for i in filename:
        with open(i) as f: line = f.readlines()
        if header: line.pop(0)
        data.extend(line)

    else:
      with open(filename) as f: data = f.readlines()
      if header: data.pop(0)
    return data


  def __len__(self):
    assert len(self.source.origin.line) == len(self.source.norm.line), 'Two files are not aligned.' 
    return len(self.source.origin.line) 


  def __getitem__(self, idx):

    # ------------------Read origin-----------------------
    line = self.source.origin.line[idx]
    line = line.strip().split(" ")

    # decode the info
    anno = self.source.origin.decode(line)

    # read image
    origin_img = cv2.imread(os.path.join(self.source.origin.root, anno.face))
    origin_img = self.transforms(origin_img)

    origin_cam_mat = np.diag((1, 1, 1))
    origin_cam_mat = torch.from_numpy(origin_cam_mat).type(torch.FloatTensor)

    origin_z_axis = torch.Tensor([0, 0, 1]).type(torch.FloatTensor)

    zone = int(anno.zone)
    zone = torch.Tensor([zone]).type(torch.long)

    # read label
    origin_label = gazeto2d(np.array(anno.gaze.split(",")).astype("float"))
    origin_label = torch.from_numpy(origin_label).type(torch.FloatTensor)

    gaze_origin = np.array(anno.origin.split(",")).astype("float")
    gaze_origin = torch.from_numpy(gaze_origin).type(torch.FloatTensor)

    name = anno.name

    # --------------------read norm------------------------
    line = self.source.norm.line[idx]
    line = line.strip().split(" ")

    # decode the info
    anno = self.source.norm.decode(line)

    # read image
    norm_img = cv2.imread(os.path.join(self.source.norm.root, anno.face))
    norm_img = self.transforms(norm_img)

    # camera position.
    norm_mat = np.fromstring(anno.norm, sep=',')
    norm_mat = cv2.Rodrigues(norm_mat)[0]  
    
     # Camera rotation. Label = R * prediction
    inv_mat = np.linalg.inv(norm_mat)
    z_axis = inv_mat[:, 2].flatten()
    
    norm_cam_mat = torch.from_numpy(inv_mat).type(torch.FloatTensor)
    z_axis = torch.from_numpy(z_axis).type(torch.FloatTensor)

    # read label
    norm_label = gazeto2d(np.array(anno.gaze.split(",")).astype("float"))
    norm_label = torch.from_numpy(norm_label).type(torch.FloatTensor)

    assert name == anno.name, 'Data is not aligned'

    pos =  torch.concat([torch.unsqueeze(origin_z_axis, 0), torch.unsqueeze(z_axis, 0)] ,0)

    # ---------------------------------------------------
    data = edict()
    data.origin_face = origin_img
    data.origin_cam = origin_cam_mat 
    data.norm_face = norm_img
    data.norm_cam = norm_cam_mat 
    data.pos = pos
    data.name = anno.name
    data.gaze_origin = gaze_origin

    label = edict()
    label.originGaze = origin_label
    label.normGaze = norm_label
    label.zone = zone
    
    return data, label


def loader(source, batch_size, shuffle=False,  num_workers=0):

  dataset = commonloader(source)

  print(f"-- [Read Data]: Total num: {len(dataset)}")

  print(f"-- [Read Data]: Source: {source.norm.label}")

  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load

if __name__ == "__main__":
  
  path = './p00.label'
# d = loader(path)
# print(len(d))
# (data, label) = d.__getitem__(0)

