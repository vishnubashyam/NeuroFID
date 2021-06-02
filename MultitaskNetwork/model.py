import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model_3d(nn.Module):
    def __init__(self, base_filters, extra_features):
        super(Model_3d, self).__init__()
        self.base_filters = base_filters
        self.extra_features = extra_features

        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.7)
        self.dropout3d = nn.Dropout3d(0.3)

        self.Sigmoid = nn.Sigmoid()

        self.block1 = self.conv_layer(1, base_filters)
        self.block2 = self.conv_layer(base_filters, base_filters*2)
        self.block3 = self.conv_layer(base_filters*2, base_filters*4)
        self.block4 = self.conv_layer(base_filters*4, base_filters*8)
        self.block5 = self.conv_layer(base_filters*8, base_filters*8)
        self.block6 = self.conv_layer(base_filters*8, base_filters*16)
        self.block7 = self.conv_layer(base_filters*16, base_filters*16)

        self.fc1 = nn.Linear((base_filters*16 ), base_filters*4)
        self.batch_norm = nn.BatchNorm1d(base_filters*4)
        self.fc2 = nn.Linear(base_filters*4,1)



    def conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=1),
        nn.LeakyReLU(),
        nn.BatchNorm3d(out_c),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        x = self.block3(x)
        x = self.dropout3d(x)

        x = self.block4(x)
        x = self.dropout3d(x)

        x = self.block5(x)
        x = self.dropout3d(x)

        x = self.block6(x)
        x = self.dropout3d(x)

        x = self.block7(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        # x = self.batch_norm(x)
        x = self.fc2(x)
        return x
