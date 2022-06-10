import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class Trainer:
    def __init__(self, max_epochs, fold, training_generator, 
                validation_generator, network, optimizer, criterion,
                output_path, device, target_types, targets):
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
        self.target_types = target_types
        self.targets = targets
        self.loss_weights = torch.ones(18)
        
    def __train_epoch(self):

        iter = 0
        epoch_loss = []
        for img_batch, labels_tmp, masks_tmp in self.training_generator:

            # Enable Dropout and BatchNorm
            self.network.train(True)

            # Transfer to GPU
            img_batch = img_batch.cuda()
            labels, masks = [], []
            for label, mask in zip(labels_tmp, masks_tmp):
                labels.append(label.cuda())
                masks.append(mask.cuda())

            # Clear Gradients from Previous Pass
            for param in self.network.parameters():
                param.grad = None

            # Forward Pass and Loss Calculation
            output = self.network(img_batch)

            # Multitask Loss Calculation
            losses = torch.tensor(0.0, requires_grad=True).cuda()
            
            for target_type, label, pred, mask, weight in zip(self.target_types, labels, output, masks, self.loss_weights):
                # print(target_type)
                # print(label[mask])
                # print(pred[mask])
                if target_type == 'Numerical':
                    loss_tmp = self.criterion[target_type](pred[mask].view(-1), label[mask].float())
                else:
                    loss_tmp = self.criterion[target_type](pred[mask], label[mask].long())

                if not loss_tmp.isnan():
                    losses += loss_tmp * weight
            
            # print(losses)
            # Backpropagation
            losses.backward()
            epoch_loss.append(losses.item())
            if iter%50==0:
                print(f'Iteration: {iter}  Loss: {np.mean(epoch_loss)}')
            self.optimizer.step()
            iter+=1
        return epoch_loss


    def __get_predictions(self, generator):

        predictions=torch.Tensor()
        labels_all=torch.Tensor()
        masks_all=torch.Tensor()

        with torch.set_grad_enabled(False):
            total_loss = []

            for img_batch, labels_tmp, masks_tmp in generator:
                #Turn Off Dropout
                self.network.eval()

                # Transfer to GPU
                img_batch = img_batch.cuda()
                labels, masks = [], []
                for label, mask in zip(labels_tmp, masks_tmp):
                    labels.append(label.cuda())
                    masks.append(mask.cuda())


                output = self.network(img_batch)
                losses = torch.tensor(0.0, requires_grad=False).cuda()

                for target_type, label, pred, mask, weight in zip(self.target_types, labels, output, masks, self.loss_weights):
                    if target_type == 'Numerical':
                        loss_tmp = self.criterion[target_type](pred[mask].view(-1), label[mask].float())
                    else:
                        loss_tmp = self.criterion[target_type](pred[mask], label[mask].long())

                    if not loss_tmp.isnan():
                        losses += loss_tmp * weight
                total_loss.append(losses.item())       

                predictions, labels_all, masks_all = {}, {}, {}
                for target, label, pred, mask in zip(self.targets, labels, output, masks):
                    labels_all[list(target.keys())[0]] = label.to('cpu').numpy()
                    predictions[list(target.keys())[0]] = pred.to('cpu').numpy()
                    masks_all[list(target.keys())[0]] = mask.to('cpu').numpy()

        return predictions, labels_all, masks_all, total_loss


    def __validation_epoch(self):

        # Validation
        predictions, labels, masks, loss = self.__get_predictions(self.validation_generator)

        val_loss = np.mean(loss)
        print("Validation Loss: ", str(val_loss))

        return val_loss   

    def train_loop(self, validation = True, save_models = True, amp = False):

        epochs_loss_train = []
        epochs_mae_val = []

        # Loop over epochs
        for epoch in range(1, self.max_epochs+1):
            self.epoch = epoch
            print('--Epoch: ' + str(epoch) + '--')
            # Training           
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