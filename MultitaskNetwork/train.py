import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class Trainer:
    def __init__(self, max_epochs, fold, training_generator, 
                validation_generator, network, optimizer, criterion,
                output_path, device):
        self.max_epochs = max_epochs
        self.fold = fold
        self.training_generator = training_generator
        self.validation_generator = validation_generator
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = 0
        self.output_path = output_path
        self.device = device
        self.saved_model_path = []


    def __train_epoch(self):

        iter = 0
        epoch_loss = []
        for img_batch, labels in self.training_generator:

            # Enable Dropout and BatchNorm
            self.network.train(True)

            # Transfer to GPU
            img_batch, labels = img_batch.cuda(), labels.cuda()

            # Clear Gradients from Previous Pass
            for param in self.network.parameters():
                param.grad = None

            # Forward Pass and Loss Calculation
            output = self.network(img_batch)
            loss = self.criterion(output.view(-1), labels)

            # Backpropagation
            loss.backward()
            epoch_loss.append(loss.item())
            if iter%50==0:
                print(f'Iteration: {iter}  Loss: {np.mean(epoch_loss)}')
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
            if iter%1==0:
                print(f'Iteration: {iter}  Loss: {np.mean(epoch_loss)}')
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

                output = self.network(img_batch)
                predictions = torch.cat((predictions, output.to('cpu')), dim=0)
                labels_all = torch.cat((labels_all, labels.to('cpu')), dim=0)

        return predictions, labels_all


    def __validation_epoch(self):

        # Validation
        predictions, labels = self.__get_predictions(self.validation_generator)

        val_mae_epoch = float(torch.mean(torch.abs(predictions - labels)).cpu().numpy())
        print("Validation MAE: ", round(val_mae_epoch, 3))

        return val_mae_epoch   

    def train_loop(self, validation = True, save_models = True, amp = False):

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

            if validation:
                epochs_loss_train.append(loss)
                val_mae = self.__validation_epoch()
                epochs_mae_val.append(val_mae)

                if save_models:
                    torch.save(
                        self.network.state_dict(),
                        self.output_path + 'Fold_'+str(self.fold)+'_Epoch_' + \
                        str(epoch) +'_valScore_' +str(round(val_mae, 2))+ "_model.pkl")
                    self.saved_model_path.append(
                        str(self.output_path + 'Fold_'+str(self.fold)+'_Epoch_' + \
                        str(epoch) +'_valScore_' +str(round(val_mae, 2))+ "_model.pkl"))

            # Save results into class varibles
        if validation:
            self.epochs_loss_train = epochs_loss_train
            self.epochs_mae_val = epochs_mae_val


    def test_loop(self, test_generator, test_set, prediction_output_path):
        best_epoch = np.argmin(self.epochs_mae_val)
        print(best_epoch)
        print(self.epochs_mae_val)
        best_model = self.saved_model_path[best_epoch]
        print(f'\nEpoch {best_epoch+1} performed best on the validation set\nGenerating Predictions')
        self.network.load_state_dict(torch.load(best_model))
        predictions, labels = self.__get_predictions(test_generator)
        out_data = pd.DataFrame(predictions.numpy())
        out_data_label = pd.DataFrame(labels.numpy()) # CV
        df = pd.concat([out_data,out_data_label], axis=1)
        df['file_name'] = test_set.list_IDs.file_name
        df.to_csv(prediction_output_path, index=False)
        print(f'Predictions saved to {prediction_output_path}')