import torch
import torch.nn as nn
import pandas
from torchvision import datasets, models, transforms
import cv2
import numpy as np
import PIL
import nibabel as nib


class Dataset_3d(torch.utils.data.Dataset):
  def __init__(self, list_IDs, data_dir, transforms=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.data_dir = data_dir
        self.transforms = transforms

  def __len__(self):
        return self.list_IDs.shape[0]


  def __getitem__(self, index):
        id = self.list_IDs['ID'][index]
        labels = self.list_IDs[labels][index]
        mask = self.list_IDs['mask'][index]

        X = nib.load(self.data_dir + str(id)).get_fdata().reshape(1,182,218,182)
        Y = np.array(label, dtype = np.int32)


        if self.transforms:
            X = self.transforms(X)

        X = torch.from_numpy(X).float()
        X = (X/ X.max())
        Y = torch.from_numpy(Y)
        aux_data = torch.from_numpy(aux_data).float()
        return X, aux_data, Y
