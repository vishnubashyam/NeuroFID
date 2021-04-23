import torch
import torchvision
from glob import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas
from torchvision import datasets, models, transforms
import cv2
import numpy as np
import PIL
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score



def train(max_epochs, fold, training_generator, validation_generator, network, optimizer, criterion):

# Loop over epochs
    for epoch in range(1, max_epochs):
        # Training
        epoch_loss = 0
        acc = 0.	# Accuracy
        length = 0
        iter = 0


        for local_batch,aux_data, local_labels in training_generator:
            network.train(True)

            # Transfer to GPU
            local_batch, aux_data, local_labels = local_batch.cuda(), aux_data.cuda() , local_labels.cuda()

            optimizer.zero_grad()

            # with torch.cuda.amp.autocast(enabled=True):

            output = network(local_batch, aux_data)
            SR_flat = output.view(output.size(0),-1)
            GT_flat = local_labels.view(local_labels.size(0),-1).float()
            loss = criterion(SR_flat, GT_flat)

            #################
            #MIXED PRECISION TRAINING#
            #################

            # scaler.step(optimizer)
            # scaler.scale(loss).backward()
            # scaler.update()
            loss.backward()
            iter+=1

            optimizer.step()
            acc += (np.mean((torch.where(SR_flat>0.5, 1,0)==GT_flat).detach().cpu().numpy()))
            # print(acc)
            length += local_batch.size(0)
            if iter%50==0:
                print(acc/iter)
                print(loss)



        acc = acc/(iter)



        print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f ' % (
    					  epoch+1, max_epochs, \
    					  epoch_loss,\
    					  acc))
        best_net = network.state_dict()
        torch.save(best_net,net_path+ 'Fold_'+str(fold)+'_Epoch_' +str(epoch) + "_model.pkl")


        # Validation
        predictions=torch.Tensor()
        with torch.set_grad_enabled(False):
            for local_batch,aux_data, local_labels in validation_generator:
                network.train(False)
                network.eval()
                # Transfer to GPU

                epoch_loss = 0
                acc = 0.	# Accuracy
                length = 0


                local_batch, aux_data, local_labels = local_batch.cuda(), aux_data.cuda() , local_labels.cuda()

                output = network(local_batch, aux_data)
                predictions = torch.cat((predictions, output.to('cpu')), dim=0)
                # acc += (1-loss)
                
                length += local_batch.size(0)

            # acc = acc/length


            val_df['Pred'] = predictions.numpy().reshape(-1)
            auc= roc_auc_score(val_df['amyloidStatus'], val_df['Pred'])

            val_df.to_csv("./weights/val_predictions"+'Fold_'+str(fold)+'_Epoch_' +str(epoch)+ "_AUC_"+str(auc)+".csv", index=False)

            # auc = metrics.auc(fpr, tpr)
            print('[Validation] AUC: %.4f'%(auc))
