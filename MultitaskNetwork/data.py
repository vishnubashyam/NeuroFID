import torch
import torch.nn as nn
import pandas
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
        self.labels = self.process_targets(self.df, targets).values
        self.data = self.df.iloc[:,self.df.columns.str.contains('MUSE')].values


    def process_numerical(self, nput_series):
        scaler = StandardScaler()
        series_out = scaler.fit_transform(input_series.values.reshape(-1, 1))
        return series_out, scaler

    def process_categorical(self, input_series):
        input_series = input_series.astype('category').cat.codes
        input_series = input_series.astype('float')
        input_series[input_series==-1.0] = np.nan
        return input_series


    def process_targets(self, df, targets):
        included = []
        for col in targets:
            target = (list(col.keys())[0])
            target_type = (list(col.values())[0]['Type'])
            print(target)
            if target_type == 'Numerical':
                df[target] = process_numerical(df[target])[0]
            if target_type == 'Categorical':
                df[target] = process_categorical(df[target]).values
            included.append(target)
        return df[included]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        
        labels = self.labels[index]
        labels = torch.from_numpy(labels).float()
        
        features = self.data[index]
        features = torch.from_numpy(features).float()

        return features, labels