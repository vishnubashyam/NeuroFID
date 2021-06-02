import torch
import torch.nn as nn
import pandas
import cv2
import numpy as np
import PIL
import nibabel as nib


class Dataset_3d(torch.utils.data.Dataset):
  def __init__(self, list_IDs, data_dir, transforms=None):
        self.list_IDs = list_IDs
        self.data_dir = data_dir
        self.transforms = transforms

  def __len__(self):
        return self.list_IDs.shape[0]

  def __getitem__(self, index):
        subject_id = self.list_IDs['File_name'][index]
        labels = self.list_IDs['AGE'][index]

        img = nib.load(self.data_dir + str(subject_id)).get_fdata().reshape(1,182,218,182)
        Y = np.array(labels, dtype = np.float32)


        if self.transforms:
            img = self.transforms(img)

        img = torch.from_numpy(img).float()
        img = (img/ img.max())
        Y = torch.from_numpy(Y).float()
        return img, Y
