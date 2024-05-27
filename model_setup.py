from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
import torch.nn as nn
from training_loop import Trainer

class BaseModel: 
    def __init__(self, args):
        self.args = args
        self.model = None
    
    def train(self, X_train, y_train):
        pass

    def evaluate(self, X_test, y_test):
        pass

class CoxModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = CoxPHSurvivalAnalysis()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        score = self.model.score(X_test, y_test)
        return score
    
    def predict(self, X):
        pred = self.model.predict(X)
        return pred

class RandomForestModel(BaseModel):   
    def __init__(self, args):
        super().__init__(args)
        self.model = RandomSurvivalForest(random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        pred = self.model.predict(X)
        return pred

def choose_model(args, device=None, num_feat=None, num_diseases=None, time_span=None):
    if args.model_type == 'cox':
        model = CoxModel(args)
    elif args.model_type == 'random_forest':
        model = RandomForestModel(args)
    elif args.model_type == 'deep':
        fcnnsurv_model = FCNNSurv(args, num_feat, num_diseases,time_span)
        model = DeepModel(args, device, fcnnsurv_model)
    elif args.model_tupe == 'deep_time':
        fcnnsurv_model = FCNNSurv_time(args, num_feat, num_diseases)
        model = DeepModel(args, device, fcnnsurv_model)
    return model

# class FCNNSurv(nn.Module):
#     def __init__(self, args):
#         super().__init__(args)


# class Deep(nn.Module):
class DeepModel(BaseModel):
    def __init__(self, args, device, torch_model):
        super().__init__(args)
        self.model = torch_model
        self.trainer = Trainer(self.model, args.model_type, device)
        self.epochs = args.epochs
        self.out_dir = args.out_dir
        self.learning_rate = args.learning_rate
        self.log = args.log # Tensorboard log flag

    def train(self, train_dataloader, val_dataloader, loss_fn):
        metrics = self.trainer.train(self.epochs, train_dataloader, val_dataloader, loss_fn, out_dir = self.out_dir, learning_rate = self.learning_rate, log = self.log)
        return metrics
    
    def predict(self, X_test):
        return self.model(X_test)

class FCNNSurv(nn.Module):
    def __init__(self, args, num_feat, num_diseases, time_span):
        super(FCNNSurv, self).__init__()
        self.layer1 = nn.Linear(num_feat, num_feat*3)
        self.dropout1 = nn.Dropout(p=0.4)
        self.layer2 = nn.Linear(num_feat*3, num_feat*5)
        self.dropout2 = nn.Dropout(p=0.4)
        self.layer3 = nn.Linear(num_feat*5, num_feat*3)
        self.dropout3 = nn.Dropout(p=0.4)
        self.layer4 = nn.Linear(num_feat*3, num_diseases*time_span)
        #THINK OF OUTPUT SIZE (NUM_DISEASES*NUM_CATEGORY?)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout1(self.relu(x))
        x = self.layer2(x)
        x = self.dropout2(self.relu(x))
        x = self.layer3(x)
        x = self.dropout3(self.relu(x))
        x = self.layer4(x)
        return x
    

class FCNNSurv_time(nn.Module):
    def __init__(self, args, num_feat, num_diseases):
        super(FCNNSurv, self).__init__()
        self.layer1 = nn.Linear(num_feat, num_feat*3)
        self.dropout1 = nn.Dropout(p=0.4)
        self.layer2 = nn.Linear(num_feat*3, num_feat*5)
        self.dropout2 = nn.Dropout(p=0.4)
        self.layer3 = nn.Linear(num_feat*5, num_feat*3)
        self.dropout3 = nn.Dropout(p=0.4)
        self.layer4 = nn.Linear(num_feat*3, num_diseases)
        #THINK OF OUTPUT SIZE (NUM_DISEASES*NUM_CATEGORY?)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout1(self.relu(x))
        x = self.layer2(x)
        x = self.dropout2(self.relu(x))
        x = self.layer3(x)
        x = self.dropout3(self.relu(x))
        x = self.relu(self.layer4(x))
        return x