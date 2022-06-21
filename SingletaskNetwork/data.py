from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any

import nibabel as nib
import torch
import numpy as np
import pandas as pd

from pytorch_lightning import LightningDataModule
from rising.loading import DataLoader
from monai.data import NibabelReader


class Dataset_3d(torch.utils.data.Dataset):
	def __init__(
		self,
		csv: pd.DataFrame,
		data_dir: Path,
		mode: str = "train",
		transforms=None,
	):
		self.csv = csv
		self.data_dir = data_dir
		self.transforms = transforms
		self.mode = mode
		self.reader = NibabelReader()

	def __len__(self) -> int:
		return self.csv.shape[0]

	def load_image(
		self,
		scan_path: Path,
		vol_size: Optional[Tuple[int, int, int]] = None,
		percentile: Optional[float] = 0.01,
	) -> torch.Tensor:
		"""_summary_

		Args:
			scan_path (Path): _description_
			vol_size (Optional[Tuple[int, int, int]], optional): _description_. Defaults to None.
			percentile (Optional[float], optional): _description_. Defaults to 0.01.

		Returns:
			torch.Tensor: _description_
		"""
		img_path = self.data_dir / (scan_path)
		# print('Path: ' + str(img_path))
		assert img_path.exists()

		img = nib.load(img_path).get_fdata()
		img = img.reshape(1, 182, 218, 182)

		# Temporary fix to make MRI data square
		img = img[:, :, 18:-18, :]

		img = torch.from_numpy(img).to(dtype=torch.float16)

		# Normalize
		if percentile:
			p_low = np.quantile(img, percentile)
			p_high = np.quantile(img, 1 - percentile)
			img = (img - p_low) / (p_high - p_low)

		return img

	def __getitem__(self, index) -> Dict[str, torch.Tensor]:
		subject_id = self.csv["MRID"][index] + '_T1_BrainAligned.nii.gz'
		labels = self.csv["Label"][index]

		img = self.load_image(scan_path=Path(subject_id))

		if self.transforms:
			img = self.transforms(img)

		Y = np.array(labels, dtype=np.float32)
		# Y = torch.from_numpy(Y).int()
		Y = torch.from_numpy(Y).float()

		return {"data": img, "label": Y}



class Scans3dDM(LightningDataModule):
	def __init__(
		self,
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
			transforms=self.train_transforms,
			mode='train')
		self.valid_dataset = Dataset_3d(
			data_dir=self.data_dir,
			csv=self.validation_df,
			transforms=self.valid_transforms,
			mode='validation')
		self.test_dataset = Dataset_3d(
			data_dir=self.data_dir,
			csv=self.test_df,
			transforms=self.valid_transforms,
			mode='test')

	def train_dataloader(self) -> DataLoader:
		return DataLoader(
			self.train_dataset,
			shuffle=True,
			**self.dl_defaults,
			**self.kwargs_dataloader,
		)

	def val_dataloader(self) -> DataLoader:
		return DataLoader(
			self.valid_dataset,
			shuffle=False,
			**self.dl_defaults,
			**self.kwargs_dataloader,
		)

	def test_dataloader(self) -> DataLoader:
		return DataLoader(
			self.test_dataset,
			shuffle=False,
			**self.dl_defaults,
			**self.kwargs_dataloader,
		)
