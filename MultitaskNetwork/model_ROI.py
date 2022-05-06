import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model_ROI(nn.Module):
    def __init__(self, task_types, per_target_heads, num_features):
        super(Model_ROI, self).__init__()
        self.task_types = task_types
        self.number_of_tasks = len(self.task_types)
        self.num_features = num_features
        self.per_target_heads = per_target_heads
        
        self.backbone = self.build_backbone()
        self.multihead = self.build_multihead()
        
            
    def build_backbone(self):
        backbone = nn.Sequential(
              nn.Linear(self.num_features, 100),
              nn.ReLU(inplace=True),
              nn.Linear(100, 10))
        return backbone
        
    def build_multihead(self):
        self.heads = nn.ModuleList([])
        for nout, task_type in zip(self.per_target_heads, self.task_types):
            
            if list(task_type.values())[0]['Type'] == 'Categorical':
                self.heads.append(nn.Sequential(
                    nn.Linear(10, nout * 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(nout * 2, nout),
                    nn.Softmax()))
                
            if list(task_type.values())[0]['Type'] == 'Numerical':                
                self.heads.append(nn.Sequential(
                    nn.Linear(10, nout * 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(nout * 2, nout)))

        return self.heads
        
        
    def forward(self, x):
        common_features = self.backbone(x)  # compute the shared features
        outputs = []
        for head in self.heads:
            outputs.append(head(common_features))       
        return outputs

