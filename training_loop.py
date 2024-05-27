import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import contextlib
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from eval import weighted_c_index
# from loss import standard_log_likelihood
class Trainer:
    def __init__(self, model, model_type, device, seed=0):
        
        self.device = device
        self.model = model.to(self.device)
        self.model_type = model_type
        self.seed = seed

        # Set random seed:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def train(self, epochs, train_loader, val_loader, loss_fn, out_dir, learning_rate=0.001, log=True):
        """
        Train model:

        Inputs:
        1. train_loader:
        2. val_loader:
        3. test_loader:
        4. epochs: Number of epochs for the training session.
        5. out_dir: 
        6. learning_rate: Learning rate for optimizer.
        7. log: if True, tensorboard logging is done

        Outputs:
        train_loss = List of loss at every epoch

        """
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.out_dir = out_dir
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.log = log
        self.max_valid = -99
        self.stop_flag = 0

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

        metrics = {
            "loss_train": [],
            "loss_val": [],
            "c_index": [],
        }

        tr_time = train_loader.dataset[:][2]
        tr_time_unique = torch.unique(tr_time.flatten())
        eval_time = [int(np.percentile(tr_time_unique, 25)), int(np.percentile(tr_time_unique, 50)), int(np.percentile(tr_time_unique, 75))]
        
        va_time = val_loader.dataset[:][2]

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            (loss_train, preds_train, labels_train, times_train) = self.step(
                train_loader,
                is_train=True,
                log=log
            )
            
            metrics["loss_train"] = loss_train
            

            (loss_val, preds_val, labels_val, times_val) = self.step(
                val_loader,
                is_train=False,
                log=log
            )

            c_index = self.evaluation(labels_train, labels_val, times_train, times_val, preds_val, eval_time)


            metrics["loss_val"] = loss_val
            metrics["c_index"] = c_index

            print(f"Training loss: {loss_train:.3f}")
            print(f"Validation loss: {loss_val:.3f}")

        return metrics
    
    def step(self, dataloader, is_train, log):
        if is_train:
            self.model.train()
            context = contextlib.nullcontext()
        else:
            self.model.eval()
            context = torch.no_grad()
        
        with context:
            batch_loss = 0
            predictions, labels_all, times = torch.tensor([]), torch.tensor([]), torch.tensor([])
            
            for batch, data in enumerate(dataloader):
                x = data[0].to(self.device)
                labels = data[1].to(self.device).long()
                time = data[2].to(self.device)
                mask1 = data[3].to(self.device)
                mask2 = data[4].to(self.device)
            
                if is_train:
                    # Clear gradient:
                    self.optimizer.zero_grad()
            
                # Compute prediction with forward pass
                logits = self.model(x).to(self.device)
                outputs = F.softmax(logits, dim=1) # shape [batch_size,num_diseases*num_Category]
                if torch.isnan(outputs).any():
                    import pdb
                    pdb.set_trace()
                    print('Found NaN')

                outputs = outputs.reshape(-1, mask1.shape[1], mask1.shape[2]) # shape [128,num_diseases,num_Category]
                
                #UNTIL HERE OK! outputs [128,36]
                # Compute loss:
                if self.model_type == 'deep':
                    loss_val = self.loss_fn(outputs, labels, mask1, mask2, time)
                    # log_likelihood = standard_log_likelihood(outputs, labels, mask1)
                elif self.model_type == 'deep_time':
                    loss_val = self.loss_fn(outputs, time, labels)

                if is_train:
                    # Backward pass:
                    # loss_val.backward()
                    loss_val.backward()

                    # Update parameters
                    self.optimizer.step()

                # Compute performance evaluation
                # pred = outputs.data.max(1)[1].cpu()
                # predictions = torch.cat([predictions, pred])
                predictions = torch.cat([predictions, outputs.cpu()])
                labels_all = torch.cat([labels_all, labels.cpu()])
                times = torch.cat([times, time.cpu()])
                # Track loss over batch
                batch_loss += loss_val.item()
            
            loss = batch_loss / (batch+1)
            
            return loss, predictions, labels_all, times
        
    def evaluation(self, tr_label, va_label, tr_time, va_time, preds, eval_time):
        if self.model_type == 'deep':
            results = np.zeros([preds.shape[1], len(eval_time)])
            for t, t_time in enumerate(eval_time):
                eval_horizon = int(t_time)
                risk = preds[:,:,:(eval_horizon+1)].sum(dim=2) # risk score until eval_time
                for k in range(preds.shape[1]): # loop through diseases
                    results[k,t] = weighted_c_index(tr_time[:,k].numpy(), (tr_label[:,k]).to(dtype=torch.int32).numpy(), risk[:,k], va_time[:,k].numpy(), (va_label[:,k]).to(dtype=torch.int32).numpy(), eval_horizon)
        
            tmp_valid = np.mean(results)

            if tmp_valid > self.max_valid:
                self.stop_flag = 0
                self.max_valid = tmp_valid
                print('Updated average C-index = ' + str('%.4f' %(tmp_valid)))
            else:
                self.stop_flag +=1

            return self.max_valid
        elif self.model_type == 'deep_time':
            return 0
