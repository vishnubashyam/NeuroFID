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
        pass
        # print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f ' % (
        #         epoch+1, max_epochs, \
        #         epoch_loss,\
        #         acc))
        # best_net = network.state_dict()
        # torch.save(best_net,net_path+ 'Fold_'+str(fold)+'_Epoch_' +str(epoch) + "_model.pkl")


    def __train_epoch(self):

        iter = 0
        epoch_loss = []
        for img_batch, labels in self.training_generator:

            # Enable Dropout and BatchNorm
            self.network.train(True)

            # Transfer to GPU
            img_batch, labels = img_batch.cuda(), labels.cuda()

            # Clear Gradients from Previous Pass
            self.optimizer.zero_grad()

            # Forward Pass and Loss Calculation
            output = self.network(img_batch)
            loss = self.criterion(output.view(-1), labels)

            # Backpropagation
            loss.backward()
            epoch_loss.append(loss.item())

            print(np.mean(epoch_loss))
            self.optimizer.step()
            iter+=1
        return epoch_loss


    def __train_epoch_mixed_precision(self, scaler):
                
        iter = 0
        epoch_loss = []
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
                loss = self.criterion(output.view(-1), labels)
            
            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            epoch_loss.append(loss.item())
            print(np.mean(epoch_loss))
            iter+=1
        return epoch_loss

    def __get_predictions(self, generator):

        predictions=torch.Tensor()
        labels_all=torch.Tensor()

        with torch.set_grad_enabled(False):
            for img_batch, labels in generator:
                #Turn Off Dropout
                self.network.eval()

                # Transfer to GPU
                img_batch, labels = img_batch.cuda(), labels.cuda()

                output = self.netwoexirk(img_batch, labels)
                predictions = torch.cat((predictions, output.to('cpu')), dim=0)
                labels_all = torch.cat((labels_all, labels.to('cpu')), dim=0)

        return predictions, labels_all


    def __validation_epoch(self):

        # Validation
        predictions, labels = self.__get_predictions(self.validation_generator)

        val_mae_epoch = torch.mean(torch.abs(predictions - labels))
        print("Validation MAE: ", round(val_mae_epoch, 3))

        return val_mae_epoch   

    def train_loop(self, amp = False):

        epochs_loss_train = []
        epochs_mae_val = []

        # Loop over epochs
        for epoch in range(1, self.max_epochs+1):
            self.epoch = epoch
            print('--Epoch: ' + str(epoch) + '--')
            # Training           
            if amp:
                scaler = torch.cuda.amp.GradScaler()
                loss = self.__train_epoch_mixed_precision(scaler)
            else:
                loss = self.__train_epoch()

            epochs_loss_train.append(loss)
            val_mae = self.__validation_epoch()
            epochs_mae_val.append(val_mae)

        # Save results into class varibles
        self.epochs_loss_train = epochs_loss_train
        self.epochs_mae_val = epochs_mae_val
