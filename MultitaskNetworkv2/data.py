from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List, Any

import nibabel as nib
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pytorch_lightning import LightningDataModule
from rising.loading import DataLoader



class Dataset_3d(torch.utils.data.Dataset):
	def __init__(
		self,
		csv: pd.DataFrame,
		targets: List,
		data_dir: Path,
		mode: str = "train",
		transforms=None,
	):
		self.csv = csv
		self.targets = targets
		self.data_dir = data_dir
		self.transforms = transforms
		self.mode = mode

		self.labels = self.process_targets(self.csv, self.targets)
		self.labels_raw = self.labels.values
		self.per_target_heads, self.target_types = self.get_per_target_heads()

		self.labels_enc, self.masks = self.encode_labels()

	def __len__(self) -> int:
		return self.csv.shape[0]

	def load_image(
		self,
		scan_path: Path,
		vol_size: Optional[Tuple[int, int, int]] = None,
		percentile: Optional[float] = 0.01,
	) -> torch.Tensor:

		img_path = self.data_dir / (scan_path)
		# print('Path: ' + str(img_path))
		assert img_path.exists()

		img = nib.load(img_path).get_fdata()
		img = img.reshape(1, 182, 218, 182)

		# Temporary fix to make MRI data square
		img = img[:, :, 18:-18, :]

		img = torch.from_numpy(img).float()

		# Normalize
		if percentile:
			p_low = np.quantile(img, percentile)
			p_high = np.quantile(img, 1 - percentile)
			img = (img - p_low) / (p_high - p_low)

		return img

	def process_categorical(self, input_series):
		input_series = input_series.astype('category').cat.codes
		ouput_series = input_series.astype('float')
		ouput_series[ouput_series==-1.0] = np.nan
		return ouput_series

	def process_numerical(self, input_series):
		scaler = MinMaxScaler()
		series_out = scaler.fit_transform(input_series.values.reshape(-1, 1))
		return series_out, scaler

	def process_targets(self, df, targets):
		included = []
		for col in targets:
			target = (list(col.keys())[0])
			target_type = (list(col.values())[0]['Type'])
            # if target_type == 'Numerical':
            #     df[target] = self.process_numerical(df[target])[0]
			if target_type == 'Categorical':
				df[target] = self.process_categorical(df[target]).values
			if target_type == 'Numerical':
				df[target] = self.process_numerical(df[target])[0]				
			included.append(target)
		return df[included]

	def encode_mask(self, labels):
		output_tensor = torch.Tensor(labels)
		loss_mask = ~torch.isnan(output_tensor)
		return output_tensor, loss_mask

	def encode_labels(self):
		gather_enc = []
		gather_masks = []
		for i in range(self.labels_raw.shape[1]):
			encoded, masks = self.encode_mask(self.labels_raw[:,i].astype('float'))
			gather_enc.append(encoded)    
			gather_masks.append(masks) 
		return gather_enc, gather_masks

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

	def __getitem__(self, index) -> Dict[str, torch.Tensor]:
		subject_id = self.csv["MRID"][index] + '_T1_BrainAligned.nii.gz'
		labels = self.labels_raw[index]

		masks = []
		for i in range(len(self.labels_enc)):
			masks.append(self.masks[i][index])

		img = self.load_image(scan_path=Path(subject_id))
		if self.transforms:
			img = self.transforms(img)

		Y = np.array(labels, dtype=np.float32)
		# Y = torch.from_numpy(Y).int()
		Y = torch.from_numpy(Y).float()

		return {"data": img, "label": Y, "masks": masks}


class Scans3dDM(LightningDataModule):
	def __init__(
		self,
		targets: List,
		data_dir: Path,
		train_df: pd.DataFrame,
		validation_df: pd.DataFrame,
		test_df: pd.DataFrame,
		vol_size: Union[None, int, Tuple[int, int, int]] = 182,
		batch_size: int = 2,
		num_workers: Optional[int] = 4,
		train_transforms=None,
		valid_transforms=None,
		**kwargs_dataloader,
	):
		super().__init__()
		# path configurations

		# self.vol_size = (
		# 	(vol_size, vol_size, vol_size)
		# 	if isinstance(vol_size, int)
		# 	else vol_size)
		self.targets = targets
		self.train_df = train_df
		self.validation_df = validation_df
		self.test_df = test_df
		self.data_dir = data_dir

		# other configs
		self.batch_size = batch_size
		self.kwargs_dataloader = kwargs_dataloader
		self.num_workers = num_workers

		# need to be filled in setup()
		self.train_dataset = None
		self.valid_dataset = None
		self.test_dataset = None
		self.train_transforms = train_transforms
		self.valid_transforms = valid_transforms

	@property
	def dl_defaults(self) -> Dict[str, Any]:
		return dict(
			batch_size=self.batch_size,
			num_workers=self.num_workers)

	def setup(self, *_, **__) -> None:
		"""Prepare datasets"""
		self.train_dataset = Dataset_3d(
			data_dir=self.data_dir,
			csv=self.train_df,
			targets=self.targets,
			mode='train')
		self.valid_dataset = Dataset_3d(
			data_dir=self.data_dir,
			csv=self.validation_df,
			targets=self.targets,
			mode='validation')
		self.test_dataset = Dataset_3d(
			data_dir=self.data_dir,
			csv=self.test_df,
			targets=self.targets,
			mode='test')

	def train_dataloader(self) -> DataLoader:
		return DataLoader(
			self.train_dataset,
			shuffle=True,
			batch_transforms=self.train_transforms,
			**self.dl_defaults,
			**self.kwargs_dataloader,
		)

	def val_dataloader(self) -> DataLoader:
		return DataLoader(
			self.valid_dataset,
			shuffle=False,
			batch_transforms=self.valid_transforms,
			**self.dl_defaults,
			**self.kwargs_dataloader,
		)

	def test_dataloader(self) -> DataLoader:
		return DataLoader(
			self.test_dataset,
			shuffle=False,
			batch_transforms=self.valid_transforms,
			**self.dl_defaults,
			**self.kwargs_dataloader,
		)
