import torch
import torch.nn as nn
import numpy as np


class Trainer:
    def __init__(self, max_epochs, fold, training_generator, validation_generator, network, optimizer, criterion):
        self.max_epochs = max_epochs
        self.fold = fold
        self.training_generator = training_generator
        self.validation_generator = validation_generator
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = 0

    def __print_save_epoch(self, epoch, loss):
        print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f ' % (
                epoch+1, max_epochs, \
                epoch_loss,\
                acc))
        best_net = network.state_dict()
        torch.save(best_net,net_path+ 'Fold_'+str(fold)+'_Epoch_' +str(epoch) + "_model.pkl")


    def __train_epoch(self):

        iter = 0
        for img_batch, labels in self.training_generator:

            # Enable Dropout and BatchNorm
            self.network.train(True)

            # Transfer to GPU
            img_batch, labels = img_batch.cuda(), labels.cuda()

            # Clear Gradients from Previous Pass
            self.optimizer.zero_grad()

            # Forward Pass and Loss Calculation
            output = self.network(img_batch)
            loss = self.criterion(output, labels)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            iter+=1


    def __train_epoch_mixed_precision(self, scaler):
                
        iter = 0
        for img_batch, labels in self.training_generator:

            # Enable Dropout and BatchNorm
            self.network.train(True)

            # Transfer to GPU
            img_batch, labels = img_batch.cuda(), labels.cuda()

            #Clear Gradients from Previous Pass
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                # Forward Pass and Loss Calculation
                output = self.network(img_batch)
                loss = self.criterion(output, labels)
            
            # Backpropagation
            scaler.step(self.optimizer)
            scaler.scale(loss).backward()
            scaler.update()
            iter+=1

            optimizer.step()


    def __validation_epoch(self):

        # Validation
        predictions=torch.Tensor()

        with torch.set_grad_enabled(False):
            for img_batch, labels in self.validation_generator:
                #Turn Off Dropout
                network.eval()

                # Transfer to GPU
                img_batch, labels = img_batch.cuda(), labels.cuda()

                output = network(img_batch, aux_data)
                predictions = torch.cat((predictions, output.to('cpu')), dim=0)


    def train_loop(self):

        # Loop over epochs
        for epoch in range(1, self.max_epochs):

            self.epoch = epoch
            # Training           
            self.__train_epoch()
            self.__validation_epoch()
            self.__print_save_epoch()