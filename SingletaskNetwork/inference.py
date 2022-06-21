from dis import pretty_flags
import numpy as np
import os
from time import gmtime, strftime

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from torchvision.models.feature_extraction import create_feature_extractor
import torch 

from data import Dataset_3d
from transforms import get_val_transforms
from model import LitBrainMRI

# argparser = parser.create_parser()
# args = argparser.parse_args()


experiment_checkpoints = Path('/cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/SingletaskNetwork/ModelCheckpoints_LSO')
current_experiment_folder = experiment_checkpoints / "Age_REG"

data_dir = Path('/cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/DataPrep/Preprocessing/BrainAligned')
df = pd.read_csv('/cbica/home/bashyamv/comp_space/1_Projects/15_NeuroFID/NeuroFID/DataPrep/Preprocessing/Lists/LSO_Training_Lists/Age_REG/Age_MAIN_REG.csv').iloc[:50]
# experiment_name = f'{args.experiment}_{args.model_name}_{str(args.model_size)}'

# print(experiment_name)



dataset = Dataset_3d(df, data_dir=data_dir, transforms=get_val_transforms())

model = LitBrainMRI.load_from_checkpoint('./ModelCheckpoints/last.ckpt').net
model = model.cuda()
trainer = pl.Trainer(
    gpus=[0],
    precision=16,
    benchmark=True,
)
data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=12)

# pred = trainer.predict(
#     model=model,
#     dataloaders=data_loader,
#     return_predictions=True,
# )

# print(model)

return_nodes = {
    "avgpool": "avgpool"
}
feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

predictions = torch.Tensor().cuda()
for i, data in enumerate(data_loader):
    with torch.no_grad():

        pred = feature_extractor(data['data'].cuda())['avgpool']
        predictions = torch.cat((predictions, pred), dim=0)
    print(i)

predictions = predictions.cpu().detach().numpy()
print(predictions.reshape((-1,predictions.shape[1])))