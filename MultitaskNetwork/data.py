import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas
import numpy as np
import PIL
import nibabel as nib
from sklearn.preprocessing import StandardScaler

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
        img = img[:,:,18:-18,:]
        Y = np.array(labels, dtype = np.float32)

        if self.transforms:
            img = self.transforms(img)

        img = torch.from_numpy(img).float()
        img = (img - img.mean())/img.std()
        Y = torch.from_numpy(Y).float()
        return img, Y



class Dataset_ROI(torch.utils.data.Dataset):
    def __init__(self, df, targets, transforms=None):
        self.df = df
        self.targets = targets
        self.labels = self.process_targets(self.df, self.targets)
        self.labels_raw = self.labels.values
        self.data = self.df.iloc[:,self.df.columns.str.contains('MUSE')].fillna(0).values
        self.data_enc = self.scale_features(self.data)
        self.per_target_heads, self.target_types = self.get_per_target_heads()
        self.labels_enc, self.masks = self.encode_labels()

    def scale_features(self, data):
        scl = StandardScaler()
        return scl.fit_transform(data)

    def process_numerical(self, input_series):
        scaler = StandardScaler()
        series_out = scaler.fit_transform(input_series.values.reshape(-1, 1))
        return series_out, scaler

    def process_categorical(self, input_series):
        input_series = input_series.astype('category').cat.codes
        input_series = input_series.astype('float')
        input_series[input_series==-1.0] = np.nan
        return input_series

    def encode_categorical(self, labels, target_size):
        if target_size>1:
            test_dat = torch.Tensor(labels)
            loss_mask = ~torch.isnan(test_dat)
            output_tensor = torch.zeros((test_dat.shape[0])).long()
            output_tensor[loss_mask] = ((test_dat[loss_mask]).long())
        else:
            output_tensor = torch.Tensor(labels)
            loss_mask = ~torch.isnan(output_tensor)
            
        return output_tensor, loss_mask
    
    def encode_labels(self):
        gather_enc = []
        gather_masks = []
        for i in range(self.labels_raw.shape[1]):
            encoded, masks = self.encode_categorical(self.labels_raw[:,i], self.per_target_heads[i])
            gather_enc.append(encoded)    
            gather_masks.append(masks) 
        return gather_enc, gather_masks

    def process_targets(self, df, targets):
        included = []
        for col in targets:
            target = (list(col.keys())[0])
            target_type = (list(col.values())[0]['Type'])
            if target_type == 'Numerical':
                df[target] = self.process_numerical(df[target])[0]
            if target_type == 'Categorical':
                df[target] = self.process_categorical(df[target]).values
            included.append(target)
        return df[included]
    
    def get_per_target_heads(self):
        ## Get the number of output nodes needed for each task
        per_target_heads = []
        target_type = []
        for target in self.targets:
            if list(target.values())[0]['Type'] == 'Categorical':
                per_target_heads.append(len(self.labels[(list(target.keys())[0])].dropna().unique()))
                target_type.append('Categorical')
            else:
                per_target_heads.append(1)
                target_type.append('Numerical')
        return per_target_heads, target_type

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        labels = []
        masks = []
        for i in range(len(self.labels_enc)):
            labels.append(self.labels_enc[i][index])
            masks.append(self.masks[i][index])

        
        features = self.data_enc[index]
        features = torch.from_numpy(features).float()

        return features, labels, masks
