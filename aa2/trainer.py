
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data 
import torch.optim as optim 
torch

class Trainer:


    def __init__(self, dump_folder="/tmp/aa2_models/"):
        self.dump_folder = dump_folder
        os.makedirs(dump_folder, exist_ok=True)
        self.epoch = 10
        self.device = torch.device('cuda:1')


    def save_model(self, epoch, model, optimizer, loss, scores, hyperparamaters, model_name):
        # epoch = epoch
        # model =  a train pytroch model
        # optimizer = a pytorch Optimizer
        # loss = loss (detach it from GPU)
        # scores = dict where keys are names of metrics and values the value for the metric
        # hyperparamaters = dict of hyperparamaters
        # model_name = name of the model you have trained, make this name unique for each hyperparamater.  I suggest you name them:
        # model_1, model_2 etc 
        #  
        #
        # More info about saving and loading here:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

        save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'hyperparamaters': hyperparamaters,
                        'loss': loss,
                        'scores': scores,
                        'model_name': model_name
                        }

        torch.save(save_dict, os.path.join(self.dump_folder, model_name + ".pt"))


    def load_model(self,path):
        
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        hyperparamaters = checkpoint['hyperparamaters']
        loss = checkpoint['loss']
        scores = checkpoint['scores']
        model_name = checkpoint['model_name']
        

        return epoch, model_state_dict, optimizer_state_dict, hyperparamaters, loss, scores, model_name
        # Finish this function so that it loads a model and return the appropriate variables
        pass
#ΔΙΚΟ ΜΟΥ

    def train(self, train_X, train_y, val_X, val_y, model_class, hyperparamaters):
        self.batch_size = hyperparamaters['batch_size']
        self.learn_rate = hyperparamaters['learning_rate']        
        hidden_size = hyperparamaters['hidden_size']
        num_layers = hyperparamaters['num_layers']
        emb_size = hyperparamaters['emb_size']
        model_name = hyperparamaters['model_name']
        epochs = hyperparamaters['epochs']
        del hyperparamaters['model_name']
        
        train_loader = self.load_batches(train_X, train_y, batch_size = self.batch_size)
        val_loader = self.load_batches(val_X, val_y, batch_size = self.batch_size)        
        model = model_class(train_X.shape[2], hidden_size, num_layers, emb_size,out_size=6).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learn_rate)        
        loss = nn.L1Loss()
        
        #opt = optim.Adam(model.parameters(), lr=self.learn_rate, weight_decay=1e-5)
        
        model.to(self.device)
        #model.train()
          
        train_loss = 0
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                model.train()
                X_batch, labels = batch
                optimizer.zero_grad()
                X_batch, labels = X_batch.to(self.device), labels.to(self.device)
                out = model(X_batch.float())
                l = loss(out, labels.float())
                total_loss += l
                l.backward()
                optimizer.step() 
                
                # creating lists where we will have
                model.eval()
                val_correct_labels = []
                val_predictions = []
                for val_batch in val_loader:
                    X_val_batch, val_labels = val_batch
                    X_val_batch = X_val_batch.float()
                    X_val_batch, val_labels = X_val_batch.to(device), val_labels.to(device)
                    model.zero_grad()
                    out = model(X_val_batch)
                    for i in range(out.shape[0]):
                        predict_sent = out[i].tolist()
                        label_sent = val_labels[i].tolist()
                        for j in range(len(predict_sent)):
                            predict_tok = round(predict_sent[j])
                            label_tok = label_sent[j]
                            val_correct_labels.append(label_tok)
                            val_predictions.append(predict_tok)
        scores = {}
        accuracy = accuracy_score(val_correct_labels, val_predictions, normalize=True)
        scores['accuracy'] = accuracy
        recall = recall_score(val_correct_labels, val_predictions, average='weighted')
        scores['recall'] = recall
        precision = precision_score(val_correct_labels, val_predictions, average='weighted')
        scores['precision'] = precision
        f = f1_score(val_correct_labels, val_predictions, average='weighted')
        scores['f1_score'] = f
        print("{}: Total loss in epoch {} is: {}      |      F1 score in validation is: {}".format(model_name, epochs, total_loss, f))
        e += 1
        self.save_model(epochs, model, optimizer, total_loss, scores, hyperparamaters, model_name)
 
            
        
        # Finish this function so that it set up model then trains and saves it.
        pass


    
    def test(self, test_X, test_y, model_class, best_model_path):
        # Finish this function so that it loads a model, test is and print results.
        pass
    
    
    
    def load_batches(self, X, y, batch_size):
        data_set = torch.utils.data.TensorDataset(X,y)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size = batch_size)
        return data_loader
    
