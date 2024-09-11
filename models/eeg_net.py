import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import os, time
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch.nn as nn

# ------------------------------------------------------------------

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

# ------------------------------------------------------------------

class EEGNet_Torch(nn.Module):
  
    # https://torcheeg.readthedocs.io/en/latest/_modules/torcheeg/models/cnn/eegnet.html
    
    def __init__(self,
                 chunk_size: int = 151,
                 num_electrodes: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        
        super(EEGNet_Torch, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        # -----------------------------

        self.b1_conv1 = nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False)

        self.b1_bn1 = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)

        self.b1_conv2 = Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False)
        
        self.b1_bn2 = nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3)
        
        self.b1_act1 = nn.ELU()
        self.b1_avgpool = nn.AvgPool2d((1, 4), stride=4)
        self.b1_do1 = nn.Dropout(p=dropout)

        self.b2_conv1 = nn.Conv2d(self.F1 * self.D,
                        self.F1 * self.D, (1, self.kernel_2),
                        stride=1,
                        padding=(0, self.kernel_2 // 2),
                        bias=False,
                        groups=self.F1 * self.D)
        
        self.b2_conv2 = nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1)
        # in_chn   F1*D = 8*2=16
        # out_chn  16
        # kernel   
        self.b2_bn1 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        self.b2_act1 = nn.ELU()
        self.b2_avgpool = nn.AvgPool2d((1, 8), stride=8)
        self.b2_do1 = nn.Dropout(p=dropout)

        self.b3_flatten = nn.Flatten()
        self.b3_lin = nn.Linear(self.F2 * self.feature_dim, num_classes, bias=False)

    @property
    def feature_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            x = self.b1_conv1(x)
            x = self.b1_bn1(x)
            x = self.b1_conv2(x)
            x = self.b1_bn2(x)
            x = self.b1_act1(x)
            x = self.b1_avgpool(x)
            x = self.b1_do1(x)
            x = self.b2_conv1(x)
            x = self.b2_conv2(x)
            x = self.b2_bn1(x)
            x = self.b2_act1(x)
            x = self.b2_avgpool(x)
            x = self.b2_do1(x)
            
        return x.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.b1_conv1(x)
        x = self.b1_bn1(x)
        x = self.b1_conv2(x)
        x = self.b1_bn2(x)
        x = self.b1_act1(x)
        x = self.b1_avgpool(x)
        x = self.b1_do1(x)
        x = self.b2_conv1(x)
        x = self.b2_conv2(x)
        x = self.b2_bn1(x)
        x = self.b2_avgpool(x)
        x = self.b2_do1(x)
        x = self.b3_flatten(x)
        x = self.b3_lin(x)
        return x

    def get_shapes_and_layers(self, x: torch.Tensor):

        shapes = []

        layer_names = ['b1_conv1', 'b1_conv2', 'b2_conv1', 'b2_conv2', 'b3_flatten', 'b3_lin']

        x = x.unsqueeze(1)

        x = self.b1_conv1(x)
        shapes.append(x.shape)

        x = self.b1_bn1(x)
        x = self.b1_conv2(x)
        shapes.append(x.shape)

        x = self.b1_bn2(x)
        x = self.b1_act1(x)
        x = self.b1_avgpool(x)
        x = self.b1_do1(x)
        x = self.b2_conv1(x)
        shapes.append(x.shape)

        x = self.b2_conv2(x)
        shapes.append(x.shape)

        x = self.b2_bn1(x)
        x = self.b2_avgpool(x)
        x = self.b2_do1(x)
        x = self.b3_flatten(x)
        shapes.append(x.shape)

        x = self.b3_lin(x)
        shapes.append(x.shape)

        return shapes, layer_names
    


# ------------------------------------------------------------------

class EEGNet_Torch_1_ConvLayer1(nn.Module):
  
    def __init__(self,
                 chunk_size: int = 151,
                 num_electrodes: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        
        super(EEGNet_Torch_1_ConvLayer, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        # -----------------------------

        self.b1_conv1 = nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False)
        
        self.b1_act1 = nn.ELU()
        self.b1_avgpool = nn.AvgPool2d((1, 4), stride=4)

        self.b3_flatten = nn.Flatten()
        self.b3_lin = nn.Linear(self.F2 * self.feature_dim, num_classes, bias=False)

    @property
    def feature_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            x = self.b1_conv1(x)
            x = self.b1_act1(x)
            x = self.b1_avgpool(x)
            
        return x.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.b1_conv1(x)
        x = self.b1_act1(x)
        x = self.b1_avgpool(x)
        x = self.b3_flatten(x)
        x = self.b3_lin(x)
        return x

    def get_shapes_and_layers(self, x: torch.Tensor):

        shapes = []

        layer_names = ['b1_conv1', 'b1_conv2', 'b2_conv1', 'b2_conv2', 'b3_flatten', 'b3_lin']

        x = x.unsqueeze(1)

        x = self.b1_conv1(x)
        shapes.append(x.shape)

        # print(x.shape)

        x = self.b1_act1(x)
        x = self.b1_avgpool(x)
        shapes.append(x.shape)
        # print(x.shape)

        x = self.b3_flatten(x)
        shapes.append(x.shape)
        # print(x.shape)

        x = self.b3_lin(x)
        shapes.append(x.shape)
        # print(x.shape)

        return shapes, layer_names
    






import torch
import torch.nn as nn

class EEGNet_Torch_1_ConvLayer(nn.Module):
  
    def __init__(self,
                 chunk_size: int = 151,
                 num_electrodes: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        
        super(EEGNet_Torch_1_ConvLayer, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        # -----------------------------
        self.b1_conv1 = nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False)
        self.b1_act1 = nn.ELU()
        self.b1_avgpool = nn.AvgPool2d((1, 4), stride=4)
        self.b3_flatten = nn.Flatten()
        # self.b3_lin = nn.Linear(self.F1 * (self.num_electrodes // 4), num_classes, bias=False)
        self.b3_lin = nn.Linear(self.feature_dim, num_classes, bias=False)


    @property
    def feature_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            x = self.b1_conv1(x)
            x = self.b1_act1(x)
            x = self.b1_avgpool(x)
        return x.view(x.size(0), -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.b1_conv1(x)
        x = self.b1_act1(x)
        x = self.b1_avgpool(x)
        x = self.b3_flatten(x)
        x = self.b3_lin(x)
        return x

    def get_shapes_and_layers(self, x: torch.Tensor):
        shapes = []
        layer_names = ['b1_conv1', 'b1_act1', 'b1_avgpool', 'b3_flatten', 'b3_lin']

        x = x.unsqueeze(1)
        x = self.b1_conv1(x)
        shapes.append(x.shape)
        # print(x.shape)

        x = self.b1_act1(x)
        x = self.b1_avgpool(x)
        shapes.append(x.shape)
        # print(x.shape)

        x = self.b3_flatten(x)
        shapes.append(x.shape)
        # print(x.shape)

        x = self.b3_lin(x)
        shapes.append(x.shape)
        # print(x.shape)

        return shapes, layer_names



















# ------------------------------------------------------------------

# class EEGNet3(nn.Module):
#     def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5,
#                  kernLength=60, F1=8, D=2, F2=None, norm_rate=0.25):
#         super(EEGNet3, self).__init__()

#         if F2 is None:
#             F2 = F1 * D

#         self.conv1 = nn.Conv2d(1, F1, (Chans, kernLength), padding=(0, kernLength // 2), bias=False)

#         self.batchnorm1 = nn.BatchNorm2d(F1)

#         ####
#         self.depthwiseConv = nn.Conv2d(F1, F1 * D, (1, 1), groups=F1, bias=False)

#         self.batchnorm2 = nn.BatchNorm2d(F1 * D)
#         self.activation = nn.ELU()
#         self.pooling1 = nn.AvgPool2d((1, 4))
#         self.dropout1 = nn.Dropout2d(dropoutRate)

#         ####
#         self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)

#         self.batchnorm3 = nn.BatchNorm2d(F2)
#         self.pooling2 = nn.AvgPool2d((1, 8))
#         self.dropout2 = nn.Dropout2d(dropoutRate)
#         self.flatten = nn.Flatten()

#         ####
#         self.dense = nn.Linear(F2 * (Samples // 32), nb_classes)

#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):

#         x = x.unsqueeze(1)

#         x = self.conv1(x)
#         x = self.batchnorm1(x)
#         x = self.depthwiseConv(x)
#         x = self.batchnorm2(x)
#         x = self.activation(x)
#         x = self.pooling1(x)
#         x = self.dropout1(x)
#         x = self.separableConv(x)
#         x = self.batchnorm3(x)
#         x = self.activation(x)
#         x = self.pooling2(x)
#         x = self.dropout2(x)
#         x = self.flatten(x)
#         x = self.dense(x)
#         x = self.softmax(x)

#         return x

# # ------------------------------------------------------------------


# # ------------------------------------------------------------------

# class EEGNet_Torch2(nn.Module):
  
#     def __init__(self,
#                  chunk_size: int = 151,
#                  num_electrodes: int = 60,
#                  F1: int = 8,
#                  F2: int = 16,
#                  D: int = 2,
#                  num_classes: int = 2,
#                  kernel_1: int = 64,
#                  kernel_2: int = 16,
#                  dropout: float = 0.25):
        
#         super(EEGNet_Torch, self).__init__()
#         self.F1 = F1
#         self.F2 = F2
#         self.D = D
#         self.chunk_size = chunk_size
#         self.num_classes = num_classes
#         self.num_electrodes = num_electrodes
#         self.kernel_1 = kernel_1
#         self.kernel_2 = kernel_2
#         self.dropout = dropout

#         self.block1 = nn.Sequential(
#             nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
#             nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
#             Conv2dWithConstraint(self.F1,
#                                  self.F1 * self.D, (self.num_electrodes, 1),
#                                  max_norm=1,
#                                  stride=1,
#                                  padding=(0, 0),
#                                  groups=self.F1,
#                                  bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
#             nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

#         self.block2 = nn.Sequential(
#             nn.Conv2d(self.F1 * self.D,
#                       self.F1 * self.D, (1, self.kernel_2),
#                       stride=1,
#                       padding=(0, self.kernel_2 // 2),
#                       bias=False,
#                       groups=self.F1 * self.D),
#             nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
#             nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
#             nn.Dropout(p=dropout))
        
#         self.lin = nn.Linear(self.F2 * self.feature_dim, num_classes, bias=False)

#     @property
#     def feature_dim(self):
#         with torch.no_grad():
#             mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
#             mock_eeg = self.block1(mock_eeg)
#             mock_eeg = self.block2(mock_eeg)

#         return mock_eeg.shape[3]

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         r'''
#         Args:
#             x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 60, 151]`. Here, :obj:`n` corresponds to the batch size, :obj:`60` corresponds to :obj:`num_electrodes`, and :obj:`151` corresponds to :obj:`chunk_size`.

#         Returns:
#             torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
#         '''

#         x = x.unsqueeze(1)
#         x = self.block1(x)
#         x = self.block2(x)
#         x = x.flatten(start_dim=1)
#         x = self.lin(x)

#         return x
