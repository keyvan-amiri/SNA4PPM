import gc

import os
import xgboost as xgb
import joblib
import optuna

from torchmetrics import AUROC, R2Score
from torchmetrics.classification import Accuracy, BinaryAccuracy

from sklearn.metrics import r2_score, roc_auc_score

import pandas as pd
import numpy as np

from autogluon.tabular import TabularPredictor

from sklearn.preprocessing import OrdinalEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from rtdl_revisiting_models import MLP, ResNet, FTTransformer, CategoricalEmbeddings
import rtdl_num_embeddings
from rtdl_num_embeddings import PeriodicEmbeddings, LinearReLUEmbeddings, LinearEmbeddings

# import delu
import time

import catboost as cb
from torchcontrib.optim import SWA
import lightgbm as lgbm

import sklearn
from copy import deepcopy
import category_encoders as ce
import math 
from focal_loss import SparseCategoricalFocalLoss

import pickle
import zipfile

from tabular_learning.tabm_reference import Model, make_parameter_groups
from torch import Tensor

import warnings
warnings.filterwarnings("ignore") 

def get_model(model_name):
    if model_name in ["LightGBMHolzmueller", "LightGBMModelHuertas", "LightGBMModelHuertasTuneMinLeaf", "LightGBMModelExperiment", "LightGBMModel1024BinsHuertasTuneMinLeaf", "LightGBMModel50000Bins", "LightGBMModelAllCat", "MLPModelMish", "XGBoostSmallData", 'LightGBMModelHuertas2', "MLPModelLongTrain"]:
        return eval(model_name)
    if model_name=="ResNetPLR":
        return ResNetPLR
    if model_name=="GLM":
        return LinearModel
    if model_name=="XGBoost": # 
        return XGBModel
    if model_name=="XGBModel1024Bins":   
        return XGBModel1024Bins
    if model_name=="XGBModelDepth1": 
        return XGBModelDepth1
    if model_name=="XGBModelDepth2": 
        return XGBModelDepth2
    if model_name=="XGBModelDepth3": 
        return XGBModelDepth3
    if model_name=="XGBModelDepth4": 
        return XGBModelDepth4
    if model_name=="XGBModelDepth5": 
        return XGBModelDepth5
    if model_name=="XGBModelDepth20": 
        return XGBModelDepth20
    if model_name=="XGBModelExact": 
        return XGBModelExact
    if model_name=="XGBoostLossguided": 
        return XGBoostLossguided
    if model_name=="XGBoostHolzmueller": 
        return XGBoostHolzmueller
    if model_name=="AutoGluon":
        return AutoGluonModel
    if model_name=="MLPContLinear":
        return MLPContLinear
    if model_name=="MLPContReLU":
        return MLPContReLU
    if model_name=="MLPContReLUTarget":
        return MLPContReLUTarget
    if model_name=="MLPContDoubleReLUTarget":
        return MLPContDoubleReLUTarget
    if model_name=="MLPContWeightedSum":
        return MLPContWeightedSum
    if model_name=="MLP":
        return MLPModel
    if model_name=="ResNet":
        return ResNetModel
    if model_name=="FTTransformer":
        return FTTransformerModel
    if model_name=="CatBoost":
        return CatBoostModel
    if model_name=="CatBoostModel1024Bins":
        return CatBoostModel1024Bins
    if model_name=="CatBoostLossguided":
        return CatBoostLossguided
    if model_name in ["MLP-PLR", "MLP-PLR-minmax", "MLP-PLR-notransform"]:
        return MLPPLR   
    if model_name=="MLPPLRHighEmbedding":
        return MLPPLRHighEmbedding   
    if model_name=="MLPPLRFixedArchitecture": 
        return MLPPLRFixedArchitecture   
    if model_name=="MLPPLRFixedArchitectureTuneSeed": 
        return MLPPLRFixedArchitectureTuneSeed
    if model_name=="MLPPLRFeatureDropout": 
        return MLPPLRFeatureDropout # 
    if model_name=="MLPPLRStopInterpol": 
        return MLPPLRStopInterpol 
    if model_name=="TabMStopInterpol": 
        return TabMStopInterpol 
    if model_name=="MLPStopInterpol": 
        return MLPStopInterpol 
    if model_name=="Trompt":
        return TromptModel   
    if model_name=="LightGBM":
        return LightGBMModel  
    if model_name=="LightGBMModelDepthLimit":
        return LightGBMModelDepthLimit  
    if model_name=="LightGBMModel1024Bins":
        return LightGBMModel1024Bins  
    if model_name=="LightGBMModelNomindataleaf":
        return LightGBMModelNomindataleaf  
    if model_name=="RealMLP":
        return RealMLP  
    if model_name=="AMFormer":
        return AMFormer  
    if model_name=="TabM":
        return TabM  
    if model_name=="TabMmini":
        return TabMmini  
    else:
        raise ValueError(f"Model '{model_name}' not implemented.")


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y) + self.eps)

class OrdinalMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        # return self.mse(yhat.abs(),y.abs())
        return ((yhat-y).abs()**2).mean()

class BaseModel:
    '''
    Possible splittings
        - model agnostic vs. model-dependent params
        - build vs fit (before/after seeing data)
        - tunable hyperparams vs rest
    Currently: Put all in one at init
    
    Required in params: task_type, task_type, cont_indices, cat_indices, cat_cardinalities, device, d_out,
    
    Each model needs to define the following functions: 
        init
        fit
        predict 
        get_default_hyperparameters
        get_optuna_hyperparameters
        
    '''
    def __init__(self, params):    
        
        self.params = params
        
        # Task specific parameters
        # self.task_type = params["task_type"]
        # self.cont_indices = params["cont_indices"]
        # self.cat_indices = params["cat_indices"]
        # self.cat_cardinalities = params["cat_cardinalities"]
        # self.d_out = params["d_out"]  
        # self.device = params["device"] 
        # self.save_path = params["save_path"] 


############################################################
############################################################
############################################################


class ResNetModel(BaseModel):
    def __init__(self, params):
        '''
        params:
            n_blocks=2,
            d_block=192,
            d_hidden=None,
            d_hidden_multiplier=2.0,
            dropout1=0.15,
            dropout2=0.0,
            d_embedding=8,
            learning_rate=0.001
        device: ["cuda", "cpu"]
        '''
        super().__init__(params)
        
        # Model-specific fixed parameters
        self.params["n_cont_features"] = len(self.params["cont_indices"])
        self.params["n_cat_features"] = len(self.params["cat_indices"])
        # self.batch_size = params["batch_size"]
        # self.val_batch_size = params["val_batch_size"]
        # self.epochs = params["epochs"]
        # self.patience = params["patience"]
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters(self.params["large_dataset"])       
        
        self.resnet = ResNet(
            d_in=self.params["n_cont_features"] + self.params["n_cat_features"]*self.params["hyperparameters"]["d_embedding"],
            d_out=self.params["d_out"],
            n_blocks=self.params["hyperparameters"]["n_blocks"],
            d_block=self.params["hyperparameters"]["d_block"],
            d_hidden=self.params["hyperparameters"]["d_hidden"],
            d_hidden_multiplier=self.params["hyperparameters"]["d_hidden_multiplier"],
            dropout1=self.params["hyperparameters"]["dropout1"],
            dropout2=self.params["hyperparameters"]["dropout2"],
        )                
        self.resnet.to(self.params["device"])
        
        if self.params["n_cat_features"]>0:
            self.embedding_layer = CategoricalEmbeddings(self.params["cat_cardinalities"],d_embedding=self.params["hyperparameters"]["d_embedding"])
            self.embedding_layer.to(self.params["device"])

    
    def forward(self,X):
        if self.params["n_cat_features"]>0:
            X_cont = X[:,self.params["cont_indices"]]
            X_cat = X[:,self.params["cat_indices"]].to(torch.int)
            
            X_cat_embed = self.embedding_layer(X_cat)
            X_cont_catembed = torch.cat([X_cont, X_cat_embed.flatten(1, -1)],dim=1)
            res = self.resnet(X_cont_catembed)
        else:
            res = self.resnet(X)
        
        return res
        
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):
        '''
        
        Partially copied from https://github.com/naszilla/tabzilla/blob/main/TabZilla/models/basemodel_torch.py
        
        '''
        # Create the save path
        if not os.path.exists(self.params["save_path"]):
            os.makedirs(self.params["save_path"])
        
        # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X_train.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        if eval_set is not None:
            X_val = eval_set[0][0]
            y_val = eval_set[0][1]
            
            X_val = torch.tensor(X_val.values).float()
            y_val = torch.tensor(y_val.values).reshape((y_val.shape[0], ))

            eval_set = [(X_val,y_val)]            
        
        X_train = torch.tensor(X_train.values).float()
        y_train = torch.tensor(y_train.values).reshape((y_train.shape[0], )) 
        
        if self.params["n_cat_features"]>0:
            # Define optimizer
            optimizer = optim.AdamW(
                list(self.resnet.parameters())+list(self.embedding_layer.parameters()), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
            # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
        else:
            optimizer = optim.AdamW(
                self.resnet.parameters(), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
            # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
            
         
        from torcheval.metrics import BinaryAUROC 

        reset_metric = True
        # Define loss function
        if self.params["task_type"] == "regression":
            loss_func = nn.MSELoss()
            if self.params["eval_metric"]=="rmse":
                eval_func = RMSELoss()
                eval_direction = "minimize"
                reset_metric = False
            elif self.params["eval_metric"]=="r2":
                eval_func = R2Score().to(self.params["device"])
                eval_direction = "maximize"
            elif self.params["eval_metric"]=="mae":
                eval_func = nn.L1Loss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.MSELoss()
                eval_direction = "minimize"
                reset_metric = False
            y_train = y_train.float()
            y_val = y_val.float()
        elif self.params["task_type"] == "multiclass":
            loss_func = nn.CrossEntropyLoss()
            if self.params["eval_metric"]=="mlogloss":
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False
        else:
            loss_func = nn.BCEWithLogitsLoss()
            if self.params["eval_metric"] in ["auc", "gini"]:
                eval_func = AUROC(task="binary").to(self.params["device"])
                eval_direction = "maximize"
            else:
                eval_func = nn.BCEWithLogitsLoss()
                eval_direction = "minimize"
                reset_metric = False
                
            y_train = y_train.float()
            y_val = y_val.float()

        # Define data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=False
        )
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.params["val_batch_size"], shuffle=True
        )
        
        # Start training loop
        if eval_direction=="minimize":
            min_val = float("inf")
        else:
            min_val = -float("inf")

        min_val_idx = 0


        history = {}
        history["loss"] = []
        history["val_loss"] = []            
        history["eval_metric"] = []
        history["eval_metric_val"] = []            
        start_time = time.time()
        for epoch in range(self.params["epochs"]):
            # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            self.resnet.train()
            if self.params["n_cat_features"]>0:
                self.embedding_layer.train()
            for i, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.params["device"])
                out = self.forward(batch_X)
                
                if (
                    self.params["task_type"] == "regression"
                    or self.params["task_type"] == "binary"
                ):
                    # out = out.squeeze()
                    out = out.reshape((batch_X.shape[0], ))
                
                loss = loss_func(out, batch_y.to(self.params["device"]))
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            history["loss"].append(loss.detach().cpu())
            
            # Early Stopping
            val_eval = 0.0
            val_loss = 0.0
            val_dim = 0
            predictions = []
            true = []
            with torch.no_grad():
                self.resnet.eval()
                if self.params["n_cat_features"]>0:
                    self.embedding_layer.eval()
                for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                    if reset_metric:
                        eval_func.reset()
                    batch_val_X = batch_val_X.to(self.params["device"])
                    batch_val_y = batch_val_y.to(self.params["device"])
                    
                    out = self.forward(batch_val_X)
    
                    if (
                        self.params["task_type"] == "regression"
                        or self.params["task_type"] == "binary"
                    ):
                        #out = out.squeeze()
                        out = out.reshape((batch_val_X.shape[0], ))
                    
                    val_loss += loss_func(out, batch_val_y)
                    # Store predictions and batch order to compute metrics outside batches as some metrics are non-cumulative
                    predictions.append(out.cpu().detach())
                    true.append(batch_val_y.cpu().detach())
                    # val_eval += eval_func(out, batch_val_y)
                    val_dim += 1

                # val_eval /= val_dim
                if reset_metric:
                    eval_func.reset()
                val_eval = eval_func(torch.concatenate(predictions).to(self.params["device"]),
                                     torch.concatenate(true).to(self.params["device"]), 
                                     )
                del true, predictions                
                
                val_loss /= val_dim
                history["val_loss"].append(val_loss.detach().cpu())
                history["eval_metric_val"].append(val_eval.detach().cpu())
                
                print("Epoch %d, Val Loss: %.5f, Val Metric: %.5f" % (epoch, val_loss, val_eval))
                
                if eval_direction=="minimize":
                    condition = val_eval < min_val
                else:
                    condition = val_eval > min_val
    
                if condition:
                    min_val = val_eval.detach().cpu()
                    min_val_idx = epoch
                    
                    # Save the currently best model
                    torch.save(self.resnet.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_ResNet.pt")
                    if self.params["n_cat_features"]>0:
                        torch.save(self.embedding_layer.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_Embeddings.pt")
                
                if min_val_idx + self.params["patience"] < epoch:
                    # print(
                    #     "Validation loss has not improved for %d steps!"
                    #     % self.params["patience"]
                    # )
                    print(
                        f"Validation loss has not improved for {self.params['patience']} steps after {epoch} epochs!"
                    )
    
                    print("Early stopping applies.")
                    break
                
                # scheduler.step(val_loss)
    
                runtime = time.time() - start_time
                # if runtime > time_limit:
                #     print(
                #         f"Runtime has exceeded time limit of {time_limit} seconds. Stopping fit."
                #     )
                #     break
    
                # torch.cuda.empty_cache()
    
        # Load best model
        state_dict_resnet = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_ResNet.pt")
        self.resnet.load_state_dict(state_dict_resnet)
        if self.params["n_cat_features"]>0:
            state_dict_embeddings = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_Embeddings.pt")
            self.embedding_layer.load_state_dict(state_dict_embeddings)
        torch.cuda.empty_cache()
        gc.collect()

        self.history = history
        
        return history        
        
        
    def predict(self, X):
       # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        X = torch.tensor(X.values).float()
        
        test_dataset = TensorDataset(X)
        test_loader = DataLoader( 
            dataset=test_dataset,
            batch_size=self.params["val_batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.resnet.eval()
        if self.params["n_cat_features"]>0:
            self.embedding_layer.eval()

        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                if self.params["task_type"]=="binary":
                    preds = torch.sigmoid(self.forward(batch_X[0].to(self.params["device"]))).cpu().detach().numpy()
                elif self.params["task_type"]=="multiclass":
                    preds = torch.nn.functional.softmax((self.forward(batch_X[0].to(self.params["device"]))), dim=1).cpu().detach().numpy()
                else:
                    preds = self.forward(batch_X[0].to(self.params["device"])).cpu().detach().numpy()                       
                predictions.append(preds)
        
        return np.concatenate(predictions)
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, large_dataset=False, dataset_name="", **kwargs):
        
        if dataset_name == "bnp-paribas-cardif-claims-management":
            max_embedding = 128
        else:
            max_embedding = 512
        
        params = {
            "n_blocks": trial.suggest_int("n_blocks",1,8), # Original paper: (A) UniformInt[1, 8], (B) UniformInt[1, 16], Kadra: first
            "d_block": trial.suggest_int("d_block",64,1024), # Original paper: Not used, default settings for FTTransformer are: [96, 128, 192, 256, 320, 384][n_blocks - 1], Kadra: [64,1024]
            "d_hidden": trial.suggest_categorical("d_hidden",[None]), # Original paper: (A) UniformInt[64, 512], (B) UniformInt[64, 1024], Kadra: don't use
            "d_hidden_multiplier": trial.suggest_int("d_hidden_multiplier",1,4), # Original paper: [2.0]), (A,B) Uniform[1, 4], Kadra: same
            "dropout1": trial.suggest_float("dropout1",0.,0.5), # Original paper:  (A,B) Uniform[0, 0.5], Kadra: same
            "dropout2": trial.suggest_float("dropout2",0.,0.5), # Original paper:  (A,B) {0, Uniform[0, 0.5]}, Kadra: same
            "d_embedding": trial.suggest_int("d_embedding",4,max_embedding), # Original paper: only for one dataset: UniformInt[64, 512], Kadra: same; we adapted lower bound to 4 as not all datasets need high dimensions for cat features
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),  # Original paper: LogUniform[1e-5, 1e-2]
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True) # Original paper: ?, Kadra: 1e-6, 1e-3
            
        }
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self, large_dataset=False):
        # if large_dataset:
        #     learning_rate = 0.00001
        #     n_blocks = 4
        #     weight_decay=0.000001
        # else:
        #     learning_rate = 0.001
        #     weight_decay=0.0001
        #     n_blocks = 2
        learning_rate = 0.0001
        weight_decay=0.00001
        params = {
            "n_blocks": 2, #2
            "d_block": 192, # 
            "d_hidden": None, # 
            "d_hidden_multiplier": 2.0, # 
            "dropout1": 0.25, # 
            "dropout2": 0.0, # 
            "d_embedding": 8, # 
            "learning_rate": learning_rate, # 0.001
            "weight_decay": weight_decay
            
        }

        return params    

############################################################
############################################################
############################################################

class MLPModel(BaseModel):
    def __init__(self, params):
        ''' '''
        super().__init__(params)
        
        # Model-specific fixed parameters
        self.params["n_cont_features"] = len(self.params["cont_indices"])
        self.params["n_cat_features"] = len(self.params["cat_indices"])
        # self.batch_size = params["batch_size"]
        # self.val_batch_size = params["val_batch_size"]
        # self.epochs = params["epochs"]
        # self.patience = params["patience"]
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters(self.params["large_dataset"])       
        
        self.mlp = MLP(
            d_in=self.params["n_cont_features"] + self.params["n_cat_features"]*self.params["hyperparameters"]["d_embedding"],
            d_out=self.params["d_out"],
            n_blocks=self.params["hyperparameters"]["n_blocks"],
            d_block=self.params["hyperparameters"]["d_block"],
            dropout=self.params["hyperparameters"]["dropout"],
        )                
        self.mlp.to(self.params["device"])
        
        if self.params["n_cat_features"]>0:
            self.embedding_layer = CategoricalEmbeddings(self.params["cat_cardinalities"],d_embedding=self.params["hyperparameters"]["d_embedding"])
            self.embedding_layer.to(self.params["device"])

    
    def forward(self,X):
        if self.params["n_cat_features"]>0:
            X_cont = X[:,self.params["cont_indices"]]
            X_cat = X[:,self.params["cat_indices"]].to(torch.int)
            
            X_cat_embed = self.embedding_layer(X_cat)
            X_cont_catembed = torch.cat([X_cont, X_cat_embed.flatten(1, -1)],dim=1)
            res = self.mlp(X_cont_catembed)
        else:
            res = self.mlp(X)
        
        return res
        
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):
        '''
        
        Partially copied from https://github.com/naszilla/tabzilla/blob/main/TabZilla/models/basemodel_torch.py
        
        '''
        # Create the save path
        if not os.path.exists(self.params["save_path"]):
            os.makedirs(self.params["save_path"])
        
        # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X_train.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        if eval_set is not None:
            X_val = eval_set[0][0]
            y_val = eval_set[0][1]
            
            X_val = torch.tensor(X_val.values).float()
            y_val = torch.tensor(y_val.values).reshape((y_val.shape[0], ))

            eval_set = [(X_val,y_val)]            
        
        X_train = torch.tensor(X_train.values).float()
        y_train = torch.tensor(y_train.values).reshape((y_train.shape[0], )) 
        
        if self.params["n_cat_features"]>0:
            # Define optimizer
            optimizer = optim.AdamW(
                list(self.mlp.parameters())+list(self.embedding_layer.parameters()), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
            # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
        else:
            optimizer = optim.AdamW(
                self.mlp.parameters(), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
            # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
            
         
        from torcheval.metrics import BinaryAUROC 

        reset_metric = True
        # Define loss function
        if self.params["task_type"] == "regression":
            loss_func = nn.MSELoss()
            if self.params["eval_metric"]=="rmse":
                eval_func = RMSELoss()
                eval_direction = "minimize"
                reset_metric = False
            elif self.params["eval_metric"]=="r2":
                eval_func = R2Score().to(self.params["device"])
                eval_direction = "maximize"
            elif self.params["eval_metric"]=="mae":
                eval_func = nn.L1Loss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.MSELoss()
                eval_direction = "minimize"
                reset_metric = False
            y_train = y_train.float()
            y_val = y_val.float()
        elif self.params["task_type"] == "multiclass":
            loss_func = nn.CrossEntropyLoss()
            if self.params["eval_metric"]=="mlogloss":
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False
        else:
            loss_func = nn.BCEWithLogitsLoss()
            if self.params["eval_metric"] in ["auc", "gini"]:
                eval_func = AUROC(task="binary").to(self.params["device"])
                eval_direction = "maximize"
            else:
                eval_func = nn.BCEWithLogitsLoss()
                eval_direction = "minimize"
                reset_metric = False
                
            y_train = y_train.float()
            y_val = y_val.float()

        # Define data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=False
        )
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.params["val_batch_size"], shuffle=True
        )
        
        # Start training loop
        if eval_direction=="minimize":
            min_val = float("inf")
        else:
            min_val = -float("inf")

        min_val_idx = 0


        history = {}
        history["loss"] = []
        history["val_loss"] = []            
        history["eval_metric"] = []
        history["eval_metric_val"] = []            
        start_time = time.time()
        for epoch in range(self.params["epochs"]):
            # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            self.mlp.train()
            if self.params["n_cat_features"]>0:
                self.embedding_layer.train()
            for i, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.params["device"])
                out = self.forward(batch_X)
                
                if (
                    self.params["task_type"] == "regression"
                    or self.params["task_type"] == "binary"
                ):
                    # out = out.squeeze()
                    out = out.reshape((batch_X.shape[0], ))
                
                loss = loss_func(out, batch_y.to(self.params["device"]))
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            history["loss"].append(loss.detach().cpu())
            
            # Early Stopping
            val_eval = 0.0
            val_loss = 0.0
            val_dim = 0
            predictions = []
            true = []
            with torch.no_grad():
                self.mlp.eval()
                if self.params["n_cat_features"]>0:
                    self.embedding_layer.eval()
                for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                    if reset_metric:
                        eval_func.reset()
                    batch_val_X = batch_val_X.to(self.params["device"])
                    batch_val_y = batch_val_y.to(self.params["device"])
                    
                    out = self.forward(batch_val_X)
    
                    if (
                        self.params["task_type"] == "regression"
                        or self.params["task_type"] == "binary"
                    ):
                        #out = out.squeeze()
                        out = out.reshape((batch_val_X.shape[0], ))
                    
                    val_loss += loss_func(out, batch_val_y)
                    # Store predictions and batch order to compute metrics outside batches as some metrics are non-cumulative
                    predictions.append(out.cpu().detach())
                    true.append(batch_val_y.cpu().detach())
                    # val_eval += eval_func(out, batch_val_y)
                    val_dim += 1

                # val_eval /= val_dim
                if reset_metric:
                    eval_func.reset()
                val_eval = eval_func(torch.concatenate(predictions).to(self.params["device"]),
                                     torch.concatenate(true).to(self.params["device"]), 
                                     )
                del true, predictions                
                
                val_loss /= val_dim
                history["val_loss"].append(val_loss.detach().cpu())
                history["eval_metric_val"].append(val_eval.detach().cpu())
                
                print("Epoch %d, Val Loss: %.5f, Val Metric: %.5f" % (epoch, val_loss, val_eval))
                
                if eval_direction=="minimize":
                    condition = val_eval < min_val
                else:
                    condition = val_eval > min_val
    
                if condition:
                    min_val = val_eval.detach().cpu()
                    min_val_idx = epoch
                    
                    # Save the currently best model
                    torch.save(self.mlp.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_mlp.pt")
                    if self.params["n_cat_features"]>0:
                        torch.save(self.embedding_layer.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_Embeddings.pt")
                
                if min_val_idx + self.params["patience"] < epoch:
                    # print(
                    #     "Validation loss has not improved for %d steps!"
                    #     % self.params["patience"]
                    # )
                    print(
                        f"Validation loss has not improved for {self.params['patience']} steps after {epoch} epochs!"
                    )
    
                    print("Early stopping applies.")
                    break
                
                # scheduler.step(val_loss)
    
                runtime = time.time() - start_time
                # if runtime > time_limit:
                #     print(
                #         f"Runtime has exceeded time limit of {time_limit} seconds. Stopping fit."
                #     )
                #     break
    
                # torch.cuda.empty_cache()
    
        # Load best model
        state_dict_mlp = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_mlp.pt")
        self.mlp.load_state_dict(state_dict_mlp)
        if self.params["n_cat_features"]>0:
            state_dict_embeddings = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_Embeddings.pt")
            self.embedding_layer.load_state_dict(state_dict_embeddings)
        torch.cuda.empty_cache()
        gc.collect()

        self.history = history
        
        return history        
        
        
    def predict(self, X):
       # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        X = torch.tensor(X.values).float()
        
        test_dataset = TensorDataset(X)
        test_loader = DataLoader( 
            dataset=test_dataset,
            batch_size=self.params["val_batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.mlp.eval()
        if self.params["n_cat_features"]>0:
            self.embedding_layer.eval()

        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                if self.params["task_type"]=="binary":
                    preds = torch.sigmoid(self.forward(batch_X[0].to(self.params["device"]))).cpu().detach().numpy()
                elif self.params["task_type"]=="multiclass":
                    preds = torch.nn.functional.softmax((self.forward(batch_X[0].to(self.params["device"]))), dim=1).cpu().detach().numpy()
                else:
                    preds = self.forward(batch_X[0].to(self.params["device"])).cpu().detach().numpy()                       
                predictions.append(preds)
        
        return np.concatenate(predictions)
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, large_dataset=False, dataset_name="", **kwargs):
        
        if dataset_name == "bnp-paribas-cardif-claims-management":
            max_embedding = 128
        else:
            max_embedding = 512
        
        params = {
            "n_blocks": trial.suggest_int("n_blocks",1,6), # Original paper: (A) UniformInt[1, 8], (B) UniformInt[1, 16], Kadra: first
            "d_block": trial.suggest_int("d_block",64,1024), # Original paper: Not used, default settings for FTTransformer are: [96, 128, 192, 256, 320, 384][n_blocks - 1], Kadra: [64,1024]
            "dropout": trial.suggest_float("dropout",0.,0.5), # Original paper:  (A,B) Uniform[0, 0.5], Kadra: same
            "d_embedding": trial.suggest_int("d_embedding",4,max_embedding), # Original paper: only for one dataset: UniformInt[64, 512], Kadra: same; we adapted lower bound to 4 as not all datasets need high dimensions for cat features
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),  # Original paper: LogUniform[1e-5, 1e-2]
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True) # Original paper: ?, Kadra: 1e-6, 1e-3
            
        }
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self, large_dataset=False):
        # if large_dataset:
        #     learning_rate = 0.00001
        #     n_blocks = 4
        #     weight_decay=0.000001
        # else:
        #     learning_rate = 0.001
        #     weight_decay=0.0001
        #     n_blocks = 2
        learning_rate = 0.0001
        weight_decay=0.00001
        params = {
            "n_blocks": 2, #2
            "d_block": 128, # 
            "dropout": 0.25, # 
            "d_embedding": 8, # 
            "learning_rate": learning_rate, # 0.001
            "weight_decay": weight_decay
            
        }

        return params    
        


#########################################################       
#########################################################       
#########################################################       

class FTTransformerModel(BaseModel):
    def __init__(self, params):
        '''
        params:
            n_blocks=2,
            d_block=192,
            d_hidden=None,
            d_hidden_multiplier=2.0,
            dropout1=0.15,
            dropout2=0.0,
            d_embedding=8,
            learning_rate=0.001
        device: ["cuda", "cpu"]
        '''
        super().__init__(params)
        
        # Model-specific fixed parameters
        self.params["n_cont_features"] = len(self.params["cont_indices"])
        self.params["n_cat_features"] = len(self.params["cat_indices"])
        # self.batch_size = params["batch_size"]
        # self.val_batch_size = params["val_batch_size"]
        # self.epochs = params["epochs"]
        # self.patience = params["patience"]
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters(self.params["large_dataset"]) 
        else:
            self.params["hyperparameters"]["_is_default"] = False
        
        torch.backends.cudnn.benchmark = True
        if self.params["hyperparameters"]["_is_default"]:
            self.FTTransformer = FTTransformer(
                **self.params["hyperparameters"],
                # d_in=self.params["n_cont_features"] + self.params["n_cat_features"]*self.params["hyperparameters"]["d_embedding"],
                n_cont_features=self.params["n_cont_features"],
                cat_cardinalities=self.params["cat_cardinalities"],
                d_out=self.params["d_out"],
                # n_blocks=self.params["hyperparameters"]["n_blocks"],
                # d_block=self.params["hyperparameters"]["d_block"],
                # d_hidden=self.params["hyperparameters"]["d_hidden"],
                # d_hidden_multiplier=self.params["hyperparameters"]["d_hidden_multiplier"],
                # dropout1=self.params["hyperparameters"]["dropout1"],
                # dropout2=self.params["hyperparameters"]["dropout2"],
                linformer_kv_compression_ratio=0.2,           # <---
                linformer_kv_compression_sharing='headwise'
            )                
        else:
            self.FTTransformer = FTTransformer(
                # **self.params["hyperparameters"],
                # d_in=self.params["n_cont_features"] + self.params["n_cat_features"]*self.params["hyperparameters"]["d_embedding"],
                n_cont_features=self.params["n_cont_features"],
                cat_cardinalities=self.params["cat_cardinalities"],
                d_out=self.params["d_out"],
                n_blocks=self.params["hyperparameters"]["n_blocks"],
                d_block=self.params["hyperparameters"]["d_block"],
                attention_dropout=self.params["hyperparameters"]["attention_dropout"],
                ffn_d_hidden_multiplier=self.params["hyperparameters"]["ffn_d_hidden_multiplier"],
                ffn_dropout=self.params["hyperparameters"]["ffn_dropout"],
                residual_dropout=self.params["hyperparameters"]["residual_dropout"],
                attention_n_heads=self.params["hyperparameters"]["attention_n_heads"],
                linformer_kv_compression_ratio=0.2,           # <---
                linformer_kv_compression_sharing='headwise'
            )                
            
        self.FTTransformer.to(self.params["device"])
        
        # if self.params["n_cat_features"]>0:
        #     self.embedding_layer = CategoricalEmbeddings(self.params["cat_cardinalities"],d_embedding=self.params["hyperparameters"]["d_embedding"])
        #     self.embedding_layer.to(self.params["device"])

    
    def forward(self,X):
        if len(self.params["cont_indices"])==0:
            X_cont = None
        else:
            X_cont = X[:,self.params["cont_indices"]]
        if self.params["n_cat_features"]>0:
            X_cat = X[:,self.params["cat_indices"]].to(torch.int)
        else:
            X_cat = None
            
        #     X_cat_embed = self.embedding_layer(X_cat)
        #     X_cont_catembed = torch.cat([X_cont, X_cat_embed.flatten(1, -1)],dim=1)
        #     res = self.resnet(X_cont_catembed)
        # else:
        #     res = self.resnet(X)
        
        return self.FTTransformer(X_cont, X_cat)#.squeeze(-1)

        
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):
        '''
        
        Partially copied from https://github.com/naszilla/tabzilla/blob/main/TabZilla/models/basemodel_torch.py
        
        '''

        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4'
        # torch.backends.cudnn.benchmark = True
        
        # Create the save path
        if not os.path.exists(self.params["save_path"]):
            os.makedirs(self.params["save_path"])
        
        # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X_train.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        if eval_set is not None:
            X_val = eval_set[0][0]
            y_val = eval_set[0][1]
            
            X_val = torch.tensor(X_val.values).float()
            y_val = torch.tensor(y_val.values).reshape((y_val.shape[0], ))

            eval_set = [(X_val,y_val)]            
        
        X_train = torch.tensor(X_train.values).float()
        y_train = torch.tensor(y_train.values).reshape((y_train.shape[0], )) 
        
        # if self.params["n_cat_features"]>0:
        #     # Define optimizer
        #     if self.params["hyperparameters"]
        #     optimizer = optim.AdamW(
        #         list(self.resnet.parameters())+list(self.embedding_layer.parameters()), 
        #         lr=self.params["hyperparameters"]["learning_rate"], 
        #         weight_decay=self.params["hyperparameters"]["weight_decay"]
        #     )
        #     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
        #     # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
        if self.params["hyperparameters"]["_is_default"]:
            optimizer = self.FTTransformer.make_default_optimizer()
        else:
            optimizer = self.FTTransformer.make_default_optimizer()
            for group in optimizer.param_groups:
                group['lr'] = self.params["hyperparameters"]["learning_rate"]
            optimizer.param_groups[1]["weight_decay"] = self.params["hyperparameters"]["weight_decay"]

            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
            # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
            
         
        from torcheval.metrics import BinaryAUROC 

        reset_metric = True
        # Define loss function
        if self.params["task_type"] == "regression":
            loss_func = nn.MSELoss()
            if self.params["eval_metric"]=="rmse":
                eval_func = RMSELoss()
                eval_direction = "minimize"
                reset_metric = False
            elif self.params["eval_metric"]=="r2":
                eval_func = R2Score().to(self.params["device"])
                eval_direction = "maximize"
            elif self.params["eval_metric"]=="mae":
                eval_func = nn.L1Loss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.MSELoss()
                eval_direction = "minimize"
                reset_metric = False
            y_train = y_train.float()
            y_val = y_val.float()
        elif self.params["task_type"] == "multiclass":
            loss_func = nn.CrossEntropyLoss()
            if self.params["eval_metric"]=="mlogloss":
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False
        else:
            loss_func = nn.BCEWithLogitsLoss()
            if self.params["eval_metric"] in ["auc", "gini"]:
                eval_func = AUROC(task="binary").to(self.params["device"])
                eval_direction = "maximize"
            else:
                eval_func = nn.BCEWithLogitsLoss()
                eval_direction = "minimize"
                reset_metric = False
                
            y_train = y_train.float()
            y_val = y_val.float()

        # Define data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=False
        )
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.params["val_batch_size"], shuffle=True
        )
        
        # Start training loop
        if eval_direction=="minimize":
            min_val = float("inf")
        else:
            min_val = -float("inf")

        min_val_idx = 0
        # if eval_direction=="minimize":
        #     early_stopping = delu.tools.EarlyStopping(patience, mode="min")
        # else:
        #     early_stopping = delu.tools.EarlyStopping(patience, mode="max")
        #         early_stopping.update(val_eval.detach().cpu())
        #         if early_stopping.should_stop():
        #             print(f"Validation loss has not improved for {self.params['patience']} steps after {epoch} epochs!")
        #             break

        print(f"total_params = {np.round(sum(p.numel() for p in self.FTTransformer.parameters())/1000000,3)}M")
        
        history = {}
        history["loss"] = []
        history["val_loss"] = []            
        history["eval_metric"] = []
        history["eval_metric_val"] = []            
        start_time = time.time()
        for epoch in range(self.params["epochs"]):
            # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            self.FTTransformer.train()
            for i, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.params["device"])
                out = self.forward(batch_X)
                
                if (
                    self.params["task_type"] == "regression"
                    or self.params["task_type"] == "binary"
                ):
                    # out = out.squeeze()
                    out = out.reshape((batch_X.shape[0], ))
                
                loss = loss_func(out, batch_y.to(self.params["device"]))
                history["loss"].append(loss.detach().cpu())
            
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            ###################


            #############
            
            # Early Stopping
            val_eval = 0.0
            val_loss = 0.0
            val_dim = 0
            predictions = []
            true = []
            with torch.no_grad():
                self.FTTransformer.eval()
                for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                    if reset_metric:
                        eval_func.reset()
                    batch_val_X = batch_val_X.to(self.params["device"])
                    batch_val_y = batch_val_y.to(self.params["device"])
                    
                    out = self.forward(batch_val_X)

                    if (
                        self.params["task_type"] == "regression"
                        or self.params["task_type"] == "binary"
                    ):
                        #out = out.squeeze()
                        out = out.reshape((batch_val_X.shape[0], ))
                    
                    val_loss += loss_func(out, batch_val_y)
                    
                    # Store predictions and batch order to compute metrics outside batches as some metrics are non-cumulative
                    predictions.append(out.cpu().detach())
                    true.append(batch_val_y.cpu().detach())
                    # val_eval += eval_func(out, batch_val_y)
                    val_dim += 1
                
                # print(y_val.shape,torch.concatenate(predictions).shape)
                # val_eval /= val_dim
                if reset_metric:
                    eval_func.reset()
                val_eval = eval_func(torch.concatenate(predictions).to(self.params["device"]),
                                     torch.concatenate(true).to(self.params["device"]), 
                                     )
                del true, predictions
                
                val_loss /= val_dim
                history["val_loss"].append(val_loss.detach().cpu())
                history["eval_metric_val"].append(val_eval.detach().cpu())
                
                print("Epoch %d, Val Loss: %.5f, Val Metric: %.5f" % (epoch, val_loss, val_eval))
                
                if eval_direction=="minimize":
                    condition = val_eval < min_val
                else:
                    condition = val_eval > min_val
    
                if condition:
                    min_val = val_eval.detach().cpu()
                    min_val_idx = epoch
                    
                    # Save the currently best model
                    torch.save(self.FTTransformer.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_FTTransformer.pt")
                
                if min_val_idx + self.params["patience"] < epoch:
                    # print(
                    #     "Validation loss has not improved for %d steps!"
                    #     % self.params["patience"]
                    # )
                    print(
                        f"Validation loss has not improved for {self.params['patience']} steps after {epoch} epochs!"
                    )
    
                    print("Early stopping applies.")
                    break
                
                # scheduler.step(val_loss)
    
                runtime = time.time() - start_time
                # if runtime > time_limit:
                #     print(
                #         f"Runtime has exceeded time limit of {time_limit} seconds. Stopping fit."
                #     )
                #     break
    
                # torch.cuda.empty_cache()
    
        # Load best model
        state_dict_fttransformer = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_FTTransformer.pt")
        self.FTTransformer.load_state_dict(state_dict_fttransformer)
        
        torch.cuda.empty_cache()
        gc.collect()
        return history        
        
    def predict(self, X):
       # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        X = torch.tensor(X.values).float()
        
        test_dataset = TensorDataset(X)
        test_loader = DataLoader( 
            dataset=test_dataset,
            batch_size=self.params["val_batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=False
        )
        
        self.FTTransformer.eval()

        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                if self.params["task_type"]=="binary":
                    preds = torch.sigmoid(self.forward(batch_X[0].to(self.params["device"]))).cpu().detach().numpy()
                elif self.params["task_type"]=="multiclass":
                    preds = torch.nn.functional.softmax((self.forward(batch_X[0].to(self.params["device"]))), dim=1).cpu().detach().numpy()
                else:
                    preds = self.forward(batch_X[0].to(self.params["device"])).cpu().detach().numpy()                       
                predictions.append(preds)
        
        return np.concatenate(predictions)
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, large_dataset=False, dataset_name="", **kwargs):
        params = FTTransformer.get_default_kwargs()
        if dataset_name=="santander-value-prediction-challenge":
            params["n_blocks"] = 1
        if large_dataset:
            params["n_blocks"] = 1
    
        params["attention_dropout"] = trial.suggest_float("attention_dropout",0.0,0.5)
        params["ffn_dropout"] = trial.suggest_float("ffn_dropout",0.,0.5)
        params["residual_dropout"] = trial.suggest_float("residual_dropout",0.,0.2)
        params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        params["d_block"] = trial.suggest_int("d_block", 8, 512, 8) # Original paper: Feature embedding size UniformInt[64, 512]
        params["_is_default"] = False
        
        # params = {
        #     "attention_dropout": trial.suggest_float("attention_dropout",0.0,0.5), # Original paper: [0,0.5]
        #     # "ffn_d_hidden_multiplier": trial.suggest_float("d_ffn_factor",2/3,8/3), # Original paper: [2/3,8/3]
        #     # "d_block": trial.suggest_categorical("d_block",[96, 128, 192, 256, 320, 384]), # Original paper: Feature embedding size UniformInt[64, 512]  # [96, 128, 192, 256, 320, 384][n_blocks - 1]
        #     # "d_block": trial.suggest_int("d_block",64,512,8), # Original paper: Feature embedding size UniformInt[64, 512]  # [96, 128, 192, 256, 320, 384][n_blocks - 1]            
        #     "ffn_dropout": trial.suggest_float("ffn_dropout",0.,0.5), # Original paper:  (A,B) Uniform[0, 0.5]
        #     # "n_blocks": trial.suggest_int("n_blocks",1,3), # Original paper: (A) UniformInt[1, 4], (B) UniformInt[1, 6]
        #     "residual_dropout": trial.suggest_float("residual_dropout",0.,0.2), # Original paper:  (A,B) {0, Uniform[0, 0.5]}
        #     # "attention_n_heads": 8,
        #     "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),  # Original paper: LogUniform[1e-5, 1e-2]
        #     "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True), # Original paper: [1e-6, 1e-3]
        #     "_is_default": False
        # }

        

        return params
    
    @classmethod
    def get_default_hyperparameters(self, large_dataset=False):
        # Default hyperparameters
        # {'n_blocks': 3,
        #  'd_block': 192,
        #  'attention_n_heads': 8,
        #  'attention_dropout': 0.2,
        #  'ffn_d_hidden': None,
        #  'ffn_d_hidden_multiplier': 1.3333333333333333,
        #  'ffn_dropout': 0.1,
        #  'residual_dropout': 0.0,
         # 'learning_rate': 1e-4,
         # 'weight_decay': 1e-5,
        #  '_is_default': True}

        params = FTTransformer.get_default_kwargs()
        
        if large_dataset:
            print("Apply Large dataset configs")
            params["n_blocks"] = 1
            # params["d_block"] = 64
        
        return params    





#################################################################
#################################################################
#################################################################


class MLPPLR(BaseModel):
    def __init__(self, params):
        '''
        params:
            n_blocks=2,
            d_block=192,
            dropout=0.15,
            d_embedding=8,
            d_embedding_num=8,
            learning_rate=0.001
        device: ["cuda", "cpu"]
        '''
        super().__init__(params)
        
        # Model-specific fixed parameters
        self.params["n_cont_features"] = len(self.params["cont_indices"])
        self.params["n_cat_features"] = len(self.params["cat_indices"])
        
        # self.batch_size = params["batch_size"]
        # self.val_batch_size = params["val_batch_size"]
        # self.epochs = params["epochs"]
        # self.patience = params["patience"]
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters(large_dataset=self.params["large_dataset"],cat_cardinalities=self.params["cat_cardinalities"])       

        self.mlp = MLP(
            d_in=self.params["n_cont_features"]*self.params["hyperparameters"]["d_embedding_num"] + self.params["n_cat_features"]*self.params["hyperparameters"]["d_embedding"],
            d_out=self.params["d_out"],
            n_blocks=self.params["hyperparameters"]["n_blocks"],
            d_block=self.params["hyperparameters"]["d_block"],
            dropout=self.params["hyperparameters"]["dropout"],
            
        )                
        self.mlp.to(self.params["device"])
        
        if self.params["n_cat_features"]>0:
            self.embedding_layer = CategoricalEmbeddings(self.params["cat_cardinalities"],d_embedding=self.params["hyperparameters"]["d_embedding"])
            self.embedding_layer.to(self.params["device"])

        if self.params["n_cont_features"]>0:
            self.embedding_layer_cont = PeriodicEmbeddings(self.params["n_cont_features"], 
                                                           frequency_init_scale = self.params["hyperparameters"]["frequency_init_scale"],
                                                           d_embedding=self.params["hyperparameters"]["d_embedding_num"], lite=self.params["hyperparameters"]["lite"])
            self.embedding_layer_cont.to(self.params["device"])
            

    
    def forward(self,X):
        if self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            X_cont = X[:,self.params["cont_indices"]]
            X_cat = X[:,self.params["cat_indices"]].to(torch.int)
            
            X_cat_embed = self.embedding_layer(X_cat)
            X_cont_embed = self.embedding_layer_cont(X_cont)
            X_contembed_catembed = torch.cat([X_cont_embed.flatten(1, -1), X_cat_embed.flatten(1, -1)],dim=1)
            res = self.mlp(X_contembed_catembed)
        elif not self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            X_cont = X
            
            X_cont_embed = self.embedding_layer_cont(X_cont)
            res = self.mlp(X_cont_embed.flatten(1, -1))        
        elif self.params["n_cat_features"]>0 and not self.params["n_cont_features"]>0:
            X_cat = X.to(torch.int)
            
            X_cat_embed = self.embedding_layer(X_cat)
            res = self.mlp(X_cat_embed.flatten(1, -1))      
        
        return res
        
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):
        '''
        
        Partially copied from https://github.com/naszilla/tabzilla/blob/main/TabZilla/models/basemodel_torch.py
        
        '''
        # Create the save path
        if not os.path.exists(self.params["save_path"]):
            os.makedirs(self.params["save_path"])
        
        # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X_train.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        if eval_set is not None:
            X_val = eval_set[0][0]
            y_val = eval_set[0][1]
            
            X_val = torch.tensor(X_val.values).float()
            y_val = torch.tensor(y_val.values).reshape((y_val.shape[0], ))

            eval_set = [(X_val,y_val)]            
        
        X_train = torch.tensor(X_train.values).float()
        y_train = torch.tensor(y_train.values).reshape((y_train.shape[0], ))      
            
        if self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            optimizer = optim.AdamW(
                list(self.mlp.parameters())+list(self.embedding_layer_cont.parameters())+list(self.embedding_layer_cont.parameters()), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )                
        elif not self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            optimizer = optim.AdamW(
                list(self.mlp.parameters())+list(self.embedding_layer_cont.parameters()), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )        
        elif self.params["n_cat_features"]>0 and not self.params["n_cont_features"]>0:
            # Define optimizer
            optimizer = optim.AdamW(
                list(self.mlp.parameters())+list(self.embedding_layer.parameters()), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
            # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
            
         
        from torcheval.metrics import BinaryAUROC 

        reset_metric = True
        # Define loss function  
        if self.params["task_type"] == "regression":
            # if len(np.unique(y_train.numpy()))<10:
            #     loss_func = OrdinalMSELoss()
            # else:
            loss_func = nn.MSELoss()    
            if self.params["eval_metric"]=="rmse":
                eval_func = RMSELoss()
                eval_direction = "minimize"
                reset_metric = False
            elif self.params["eval_metric"]=="r2":
                eval_func = R2Score().to(self.params["device"])
                eval_direction = "maximize"
            elif self.params["eval_metric"]=="mae":
                eval_func = nn.L1Loss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.MSELoss()
                eval_direction = "minimize"
                reset_metric = False
            y_train = y_train.float()
            y_val = y_val.float()
        elif self.params["task_type"] == "multiclass":
            loss_func = nn.CrossEntropyLoss()
            if self.params["eval_metric"]=="mlogloss":
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False
            elif self.params["eval_metric"]=="mAccuracy":
                eval_func = Accuracy(task="multiclass", num_classes=self.params["d_out"]).to(self.params["device"])
                eval_direction = "maximize"
                # reset_metric = False
            else:
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False                
        else:
            loss_func = nn.BCEWithLogitsLoss()
            if self.params["eval_metric"] in ["auc", "gini"]:
                eval_func = AUROC(task="binary").to(self.params["device"])
                eval_direction = "maximize"
            elif self.params["eval_metric"]=="Accuracy":
                eval_func = BinaryAccuracy().to(self.params["device"])   
                eval_direction = "maximize"
                # reset_metric = False
            else:
                eval_func = nn.BCEWithLogitsLoss()
                eval_direction = "minimize"
                reset_metric = False
                
            y_train = y_train.float()
            y_val = y_val.float()

        # Define data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=False
        )
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.params["val_batch_size"], shuffle=True
        )
        
        # Start training loop
        if eval_direction=="minimize":
            min_val = float("inf")
        else:
            min_val = -float("inf")

        min_val_idx = 0
        
        # print(f"total_params = {np.round(sum(p.numel() for p in self.mlp.parameters())/1000000,3)}M")
        
        history = {}
        history["loss"] = []
        history["val_loss"] = []            
        history["eval_metric"] = []
        history["eval_metric_val"] = []            
        start_time = time.time()
        for epoch in range(self.params["epochs"]):
            # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            self.mlp.train()
            if self.params["n_cat_features"]>0:
                self.embedding_layer.train()
            if self.params["n_cont_features"]>0:
                self.embedding_layer_cont.train()
            for i, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.params["device"])
                out = self.forward(batch_X)
                
                if (
                    self.params["task_type"] == "regression"
                    or self.params["task_type"] == "binary"
                ):
                    # out = out.squeeze()
                    out = out.reshape((batch_X.shape[0], ))
                
                loss = loss_func(out, batch_y.to(self.params["device"]))
                history["loss"].append(loss.detach().cpu())
            
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            
            
            # Early Stopping
            val_eval = 0.0
            val_loss = 0.0
            val_dim = 0
            predictions = []
            true = []
            with torch.no_grad():
                self.mlp.eval()
                if self.params["n_cat_features"]>0:
                    self.embedding_layer.eval()
                if self.params["n_cont_features"]>0:
                    self.embedding_layer_cont.eval()                    
                for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                    if reset_metric:
                        eval_func.reset()
                    batch_val_X = batch_val_X.to(self.params["device"])
                    batch_val_y = batch_val_y.to(self.params["device"])
                    
                    out = self.forward(batch_val_X)
    
                    if (
                        self.params["task_type"] == "regression"
                        or self.params["task_type"] == "binary"
                    ):
                        #out = out.squeeze()
                        out = out.reshape((batch_val_X.shape[0], ))
                    
                    val_loss += loss_func(out, batch_val_y)
                    # Store predictions and batch order to compute metrics outside batches as some metrics are non-cumulative
                    predictions.append(out.cpu().detach())
                    true.append(batch_val_y.cpu().detach())
                    # val_eval += eval_func(out, batch_val_y)
                    val_dim += 1
                
                
                # val_eval /= val_dim
                if reset_metric:
                    eval_func.reset()
                val_eval = eval_func(torch.concatenate(predictions).to(self.params["device"]),
                                     torch.concatenate(true).to(self.params["device"]), 
                                     )
                del true, predictions        
                
                val_loss /= val_dim
                history["val_loss"].append(val_loss.detach().cpu())
                history["eval_metric_val"].append(val_eval.detach().cpu())
                
                print("Epoch %d, Val Loss: %.5f, Val Metric: %.5f" % (epoch, val_loss, val_eval))
                
                if eval_direction=="minimize":
                    condition = val_eval < min_val
                else:
                    condition = val_eval > min_val
    
                if condition:
                    min_val = val_eval.detach().cpu()
                    min_val_idx = epoch
                    
                    # Save the currently best model
                    torch.save(self.mlp.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_ResNet.pt")
                    if self.params["n_cat_features"]>0:
                        torch.save(self.embedding_layer.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_Embeddings.pt")
                    if self.params["n_cont_features"]>0:
                        torch.save(self.embedding_layer_cont.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_Embeddings_Cont.pt")
                        
                    
                if min_val_idx + self.params["patience"] < epoch:
                    # print(
                    #     "Validation loss has not improved for %d steps!"
                    #     % self.params["patience"]
                    # )
                    print(
                        f"Validation loss has not improved for {self.params['patience']} steps after {epoch} epochs!"
                    )
    
                    print("Early stopping applies.")
                    break
                
                # scheduler.step(val_loss)
    
                runtime = time.time() - start_time
                # if runtime > time_limit:
                #     print(
                #         f"Runtime has exceeded time limit of {time_limit} seconds. Stopping fit."
                #     )
                #     break
    
                # torch.cuda.empty_cache()
    
        # Load best model
        state_dict_resnet = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_ResNet.pt")
        self.mlp.load_state_dict(state_dict_resnet)
        if self.params["n_cat_features"]>0:
            state_dict_embeddings = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_Embeddings.pt")
            self.embedding_layer.load_state_dict(state_dict_embeddings)
        if self.params["n_cont_features"]>0:
            state_dict_embeddings_cont = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_Embeddings_Cont.pt")
            self.embedding_layer_cont.load_state_dict(state_dict_embeddings_cont)

        
        torch.cuda.empty_cache()
        gc.collect()
        return history        
        
        
    def predict(self, X):
       # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        X = torch.tensor(X.values).float()
        
        test_dataset = TensorDataset(X)
        test_loader = DataLoader( 
            dataset=test_dataset,
            batch_size=self.params["val_batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.mlp.eval()
        if self.params["n_cat_features"]>0:
            self.embedding_layer.eval()
        if self.params["n_cont_features"]>0:
            self.embedding_layer_cont.eval()              
        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                if self.params["task_type"]=="binary":
                    preds = torch.sigmoid(self.forward(batch_X[0].to(self.params["device"]))).cpu().detach().numpy()
                elif self.params["task_type"]=="multiclass":
                    preds = torch.nn.functional.softmax((self.forward(batch_X[0].to(self.params["device"]))), dim=1).cpu().detach().numpy()
                else:
                    preds = self.forward(batch_X[0].to(self.params["device"])).cpu().detach().numpy()                       
                predictions.append(preds)
        
        return np.concatenate(predictions)
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, large_dataset=False, cat_cardinalities=[], **kwargs):
        params = {
            "n_blocks": trial.suggest_int("n_blocks",1,8), # Original paper: (A) UniformInt[1, 8], (B) UniformInt[1, 16], Kadra: first
            "d_block": trial.suggest_int("d_block",1,1024), # Original paper: Not used, default settings for FTTransformer are: [96, 128, 192, 256, 320, 384][n_blocks - 1], Kadra: [64,1024]
            "dropout": trial.suggest_float("dropout",0.,0.5), # Original paper:  (A,B) Uniform[0, 0.5], Kadra: same
            "d_embedding": trial.suggest_int("d_embedding",1,512), # Original paper: only for one dataset: UniformInt[64, 512], Kadra: same; we adapted lower bound to 4 as not all datasets need high dimensions for cat features
            "d_embedding_num": trial.suggest_int("d_embedding_num",1,128), # Original paper: only for one dataset: UniformInt[64, 512], Kadra: same; we adapted lower bound to 4 as not all datasets need high dimensions for cat features
            "frequency_init_scale": trial.suggest_float("frequency_init_scale", 0.01, 10., log=True), # Original paper:  0.01, 100. - but in docu they recommend to set to max 10 
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 0.005, log=True),  # Original paper: LogUniform[1e-5, 1e-2]
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True), # Original paper: ?, Kadra: 1e-6, 1e-3
            # "frequency_init_scale": 0, 
            "lite": True,
            
        }
        
        # params["d_embedding"] = params["d_embedding_num"]
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self, large_dataset=False, cat_cardinalities=[]):
        # if large_dataset:
        #     learning_rate = 0.00001
        #     n_blocks = 4
        #     weight_decay=0.000001
        # else:
        #     learning_rate = 0.001
        #     weight_decay=0.0001
        #     n_blocks = 2
        learning_rate = 0.001
        weight_decay=0.0001
        params = {
            "n_blocks": 2, #2
            "d_block": 192, # 
            "dropout": 0.25, # 
            "d_embedding": 8, # 
            "d_embedding_num": 8, # 
            "lite": False,
            "frequency_init_scale": 0.01,
            "learning_rate": learning_rate, # 0.001
            "weight_decay": weight_decay
            
        }
        return params    



############################################
############################################
############################################
class MLPContLinear(BaseModel):
    def __init__(self, params):
        '''
        params:
            n_blocks=2,
            d_block=192,
            dropout=0.15,
            d_embedding=8,
            d_embedding_num=8,
            learning_rate=0.001
        device: ["cuda", "cpu"]
        '''
        super().__init__(params)
        
        # Model-specific fixed parameters
        self.params["n_cont_features"] = len(self.params["cont_indices"])
        self.params["n_cat_features"] = len(self.params["cat_indices"])
        
        # self.batch_size = params["batch_size"]
        # self.val_batch_size = params["val_batch_size"]
        # self.epochs = params["epochs"]
        # self.patience = params["patience"]
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters(self.params["large_dataset"])       

        self.mlp = MLP(
            d_in=self.params["n_cont_features"]*self.params["hyperparameters"]["d_embedding_num"] + self.params["n_cat_features"]*self.params["hyperparameters"]["d_embedding"],
            d_out=self.params["d_out"],
            n_blocks=self.params["hyperparameters"]["n_blocks"],
            d_block=self.params["hyperparameters"]["d_block"],
            dropout=self.params["hyperparameters"]["dropout"],
            
        )                
        self.mlp.to(self.params["device"])
        
        if self.params["n_cat_features"]>0:
            self.embedding_layer = CategoricalEmbeddings(self.params["cat_cardinalities"],d_embedding=self.params["hyperparameters"]["d_embedding"])
            self.embedding_layer.to(self.params["device"])

        if self.params["n_cont_features"]>0:
            self.embedding_layer_cont = LinearEmbeddings(self.params["n_cont_features"], 
                                                         d_embedding=self.params["hyperparameters"]["d_embedding_num"])
            self.embedding_layer_cont.to(self.params["device"])
            

    
    def forward(self,X):
        if self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            X_cont = X[:,self.params["cont_indices"]]
            X_cat = X[:,self.params["cat_indices"]].to(torch.int)
            
            X_cat_embed = self.embedding_layer(X_cat)
            X_cont_embed = self.embedding_layer_cont(X_cont)
            X_contembed_catembed = torch.cat([X_cont_embed.flatten(1, -1), X_cat_embed.flatten(1, -1)],dim=1)
            res = self.mlp(X_contembed_catembed)
        elif not self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            X_cont = X
            
            X_cont_embed = self.embedding_layer_cont(X_cont)
            res = self.mlp(X_cont_embed.flatten(1, -1))        
        elif self.params["n_cat_features"]>0 and not self.params["n_cont_features"]>0:
            X_cat = X.to(torch.int)
            
            X_cat_embed = self.embedding_layer(X_cat)
            res = self.mlp(X_cat_embed.flatten(1, -1))      
        
        return res
        
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):
        '''
        
        Partially copied from https://github.com/naszilla/tabzilla/blob/main/TabZilla/models/basemodel_torch.py
        
        '''
        # Create the save path
        if not os.path.exists(self.params["save_path"]):
            os.makedirs(self.params["save_path"])
        
        # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X_train.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        if eval_set is not None:
            X_val = eval_set[0][0]
            y_val = eval_set[0][1]
            
            X_val = torch.tensor(X_val.values).float()
            y_val = torch.tensor(y_val.values).reshape((y_val.shape[0], ))

            eval_set = [(X_val,y_val)]            
        
        X_train = torch.tensor(X_train.values).float()
        y_train = torch.tensor(y_train.values).reshape((y_train.shape[0], ))      
            
        if self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            optimizer = optim.AdamW(
                list(self.mlp.parameters())+list(self.embedding_layer_cont.parameters())+list(self.embedding_layer_cont.parameters()), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )                
        elif not self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            optimizer = optim.AdamW(
                list(self.mlp.parameters())+list(self.embedding_layer_cont.parameters()), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )        
        elif self.params["n_cat_features"]>0 and not self.params["n_cont_features"]>0:
            # Define optimizer
            optimizer = optim.AdamW(
                list(self.mlp.parameters())+list(self.embedding_layer.parameters()), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
            # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
            
         
        from torcheval.metrics import BinaryAUROC 

        reset_metric = True
        # Define loss function
        if self.params["task_type"] == "regression":
            loss_func = nn.MSELoss()
            if self.params["eval_metric"]=="rmse":
                eval_func = RMSELoss()
                eval_direction = "minimize"
                reset_metric = False
            elif self.params["eval_metric"]=="r2":
                eval_func = R2Score().to(self.params["device"])
                eval_direction = "maximize"
            elif self.params["eval_metric"]=="mae":
                eval_func = nn.L1Loss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.MSELoss()
                eval_direction = "minimize"
                reset_metric = False
            y_train = y_train.float()
            y_val = y_val.float()
        elif self.params["task_type"] == "multiclass":
            loss_func = nn.CrossEntropyLoss()
            if self.params["eval_metric"]=="mlogloss":
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False                
        else:
            loss_func = nn.BCEWithLogitsLoss()
            if self.params["eval_metric"] in ["auc", "gini"]:
                eval_func = AUROC(task="binary").to(self.params["device"])
                eval_direction = "maximize"
            else:
                eval_func = nn.BCEWithLogitsLoss()
                eval_direction = "minimize"
                reset_metric = False
                
            y_train = y_train.float()
            y_val = y_val.float()

        # Define data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=False
        )
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.params["val_batch_size"], shuffle=True
        )
        
        # Start training loop
        if eval_direction=="minimize":
            min_val = float("inf")
        else:
            min_val = -float("inf")

        min_val_idx = 0
        
        # print(f"total_params = {np.round(sum(p.numel() for p in self.mlp.parameters())/1000000,3)}M")
        
        history = {}
        history["loss"] = []
        history["val_loss"] = []            
        history["eval_metric"] = []
        history["eval_metric_val"] = []            
        start_time = time.time()
        for epoch in range(self.params["epochs"]):
            # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            self.mlp.train()
            if self.params["n_cat_features"]>0:
                self.embedding_layer.train()
            if self.params["n_cont_features"]>0:
                self.embedding_layer_cont.train()
            for i, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.params["device"])
                out = self.forward(batch_X)
                
                if (
                    self.params["task_type"] == "regression"
                    or self.params["task_type"] == "binary"
                ):
                    # out = out.squeeze()
                    out = out.reshape((batch_X.shape[0], ))
                
                loss = loss_func(out, batch_y.to(self.params["device"]))
                history["loss"].append(loss.detach().cpu())
            
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            
            
            # Early Stopping
            val_eval = 0.0
            val_loss = 0.0
            val_dim = 0
            predictions = []
            true = []
            with torch.no_grad():
                self.mlp.eval()
                if self.params["n_cat_features"]>0:
                    self.embedding_layer.eval()
                if self.params["n_cont_features"]>0:
                    self.embedding_layer_cont.eval()                    
                for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                    if reset_metric:
                        eval_func.reset()
                    batch_val_X = batch_val_X.to(self.params["device"])
                    batch_val_y = batch_val_y.to(self.params["device"])
                    
                    out = self.forward(batch_val_X)
    
                    if (
                        self.params["task_type"] == "regression"
                        or self.params["task_type"] == "binary"
                    ):
                        #out = out.squeeze()
                        out = out.reshape((batch_val_X.shape[0], ))
                    
                    val_loss += loss_func(out, batch_val_y)
                    # Store predictions and batch order to compute metrics outside batches as some metrics are non-cumulative
                    predictions.append(out.cpu().detach())
                    true.append(batch_val_y.cpu().detach())
                    # val_eval += eval_func(out, batch_val_y)
                    val_dim += 1
                
                
                # val_eval /= val_dim
                if reset_metric:
                    eval_func.reset()
                val_eval = eval_func(torch.concatenate(predictions).to(self.params["device"]),
                                     torch.concatenate(true).to(self.params["device"]), 
                                     )
                del true, predictions        
                
                val_loss /= val_dim
                history["val_loss"].append(val_loss.detach().cpu())
                history["eval_metric_val"].append(val_eval.detach().cpu())
                
                print("Epoch %d, Val Loss: %.5f, Val Metric: %.5f" % (epoch, val_loss, val_eval))
                
                if eval_direction=="minimize":
                    condition = val_eval < min_val
                else:
                    condition = val_eval > min_val
    
                if condition:
                    min_val = val_eval.detach().cpu()
                    min_val_idx = epoch
                    
                    # Save the currently best model
                    torch.save(self.mlp.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_ResNet.pt")
                    if self.params["n_cat_features"]>0:
                        torch.save(self.embedding_layer.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_Embeddings.pt")
                    if self.params["n_cont_features"]>0:
                        torch.save(self.embedding_layer_cont.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_Embeddings_Cont.pt")
                        
                    
                if min_val_idx + self.params["patience"] < epoch:
                    # print(
                    #     "Validation loss has not improved for %d steps!"
                    #     % self.params["patience"]
                    # )
                    print(
                        f"Validation loss has not improved for {self.params['patience']} steps after {epoch} epochs!"
                    )
    
                    print("Early stopping applies.")
                    break
                
                # scheduler.step(val_loss)
    
                runtime = time.time() - start_time
                # if runtime > time_limit:
                #     print(
                #         f"Runtime has exceeded time limit of {time_limit} seconds. Stopping fit."
                #     )
                #     break
    
                # torch.cuda.empty_cache()
    
        # Load best model
        state_dict_resnet = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_ResNet.pt")
        self.mlp.load_state_dict(state_dict_resnet)
        if self.params["n_cat_features"]>0:
            state_dict_embeddings = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_Embeddings.pt")
            self.embedding_layer.load_state_dict(state_dict_embeddings)
        if self.params["n_cont_features"]>0:
            state_dict_embeddings_cont = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_Embeddings_Cont.pt")
            self.embedding_layer_cont.load_state_dict(state_dict_embeddings_cont)

        
        torch.cuda.empty_cache()
        gc.collect()
        return history        
        
        
    def predict(self, X):
       # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        X = torch.tensor(X.values).float()
        
        test_dataset = TensorDataset(X)
        test_loader = DataLoader( 
            dataset=test_dataset,
            batch_size=self.params["val_batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.mlp.eval()
        if self.params["n_cat_features"]>0:
            self.embedding_layer.eval()
        if self.params["n_cont_features"]>0:
            self.embedding_layer_cont.eval()              
        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                if self.params["task_type"]=="binary":
                    preds = torch.sigmoid(self.forward(batch_X[0].to(self.params["device"]))).cpu().detach().numpy()
                elif self.params["task_type"]=="multiclass":
                    preds = torch.nn.functional.softmax((self.forward(batch_X[0].to(self.params["device"]))), dim=1).cpu().detach().numpy()
                else:
                    preds = self.forward(batch_X[0].to(self.params["device"])).cpu().detach().numpy()                       
                predictions.append(preds)
        
        return np.concatenate(predictions)
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, large_dataset=False, **kwargs):
        params = {
            "n_blocks": trial.suggest_int("n_blocks",1,8), # Original paper: (A) UniformInt[1, 8], (B) UniformInt[1, 16], Kadra: first
            "d_block": trial.suggest_int("d_block",1,1024), # Original paper: Not used, default settings for FTTransformer are: [96, 128, 192, 256, 320, 384][n_blocks - 1], Kadra: [64,1024]
            "dropout": trial.suggest_float("dropout",0.,0.5), # Original paper:  (A,B) Uniform[0, 0.5], Kadra: same
            "d_embedding": trial.suggest_int("d_embedding",1,512), # Original paper: only for one dataset: UniformInt[64, 512], Kadra: same; we adapted lower bound to 4 as not all datasets need high dimensions for cat features
            "d_embedding_num": trial.suggest_int("d_embedding_num",1,128), # Original paper: only for one dataset: UniformInt[64, 512], Kadra: same; we adapted lower bound to 4 as not all datasets need high dimensions for cat features
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 0.005, log=True),  # Original paper: LogUniform[1e-5, 1e-2]
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True), # Original paper: ?, Kadra: 1e-6, 1e-3
            
        }
        
        # params["d_embedding"] = params["d_embedding_num"]
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self, large_dataset=False):
        # if large_dataset:
        #     learning_rate = 0.00001
        #     n_blocks = 4
        #     weight_decay=0.000001
        # else:
        #     learning_rate = 0.001
        #     weight_decay=0.0001
        #     n_blocks = 2
        learning_rate = 0.001
        weight_decay=0.0001
        params = {
            "n_blocks": 2, #2
            "d_block": 192, # 
            "dropout": 0.25, # 
            "d_embedding": 8, # 
            "d_embedding_num": 8, # 
            "learning_rate": learning_rate, # 0.001
            "weight_decay": weight_decay
            
        }
        return params    


#################################################################
#################################################################
#################################################################

class MLPContReLU(BaseModel):
    def __init__(self, params):
        '''
        params:
            n_blocks=2,
            d_block=192,
            dropout=0.15,
            d_embedding=8,
            d_embedding_num=8,
            learning_rate=0.001
        device: ["cuda", "cpu"]
        '''
        super().__init__(params)
        
        # Model-specific fixed parameters
        self.params["n_cont_features"] = len(self.params["cont_indices"])
        self.params["n_cat_features"] = len(self.params["cat_indices"])
        
        # self.batch_size = params["batch_size"]
        # self.val_batch_size = params["val_batch_size"]
        # self.epochs = params["epochs"]
        # self.patience = params["patience"]
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters(self.params["large_dataset"])       

        self.mlp = MLP(
            d_in=self.params["n_cont_features"]*self.params["hyperparameters"]["d_embedding_num"] + self.params["n_cat_features"]*self.params["hyperparameters"]["d_embedding"],
            d_out=self.params["d_out"],
            n_blocks=self.params["hyperparameters"]["n_blocks"],
            d_block=self.params["hyperparameters"]["d_block"],
            dropout=self.params["hyperparameters"]["dropout"],
            
        )                
        self.mlp.to(self.params["device"])
        
        if self.params["n_cat_features"]>0:
            self.embedding_layer = CategoricalEmbeddings(self.params["cat_cardinalities"],d_embedding=self.params["hyperparameters"]["d_embedding"])
            self.embedding_layer.to(self.params["device"])

        if self.params["n_cont_features"]>0:
            self.embedding_layer_cont = LinearReLUEmbeddings(self.params["n_cont_features"], 
                                                         d_embedding=self.params["hyperparameters"]["d_embedding_num"])
            self.embedding_layer_cont.to(self.params["device"])
            

    def forward(self,X):
        if self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            X_cont = X[:,self.params["cont_indices"]]
            X_cat = X[:,self.params["cat_indices"]].to(torch.int)
            
            X_cat_embed = self.embedding_layer(X_cat)
            X_cont_embed = self.embedding_layer_cont(X_cont)
            X_contembed_catembed = torch.cat([X_cont_embed.flatten(1, -1), X_cat_embed.flatten(1, -1)],dim=1)
            res = self.mlp(X_contembed_catembed)
        elif not self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            X_cont = X
            
            X_cont_embed = self.embedding_layer_cont(X_cont)
            res = self.mlp(X_cont_embed.flatten(1, -1))        
        elif self.params["n_cat_features"]>0 and not self.params["n_cont_features"]>0:
            X_cat = X.to(torch.int)
            
            X_cat_embed = self.embedding_layer(X_cat)
            res = self.mlp(X_cat_embed.flatten(1, -1))      
        
        return res
        
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):
        '''
        
        Partially copied from https://github.com/naszilla/tabzilla/blob/main/TabZilla/models/basemodel_torch.py
        
        '''
        # Create the save path
        if not os.path.exists(self.params["save_path"]):
            os.makedirs(self.params["save_path"])
        
        # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X_train.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        if eval_set is not None:
            X_val = eval_set[0][0]
            y_val = eval_set[0][1]
            
            X_val = torch.tensor(X_val.values).float()
            y_val = torch.tensor(y_val.values).reshape((y_val.shape[0], ))

            eval_set = [(X_val,y_val)]            
        
        X_train = torch.tensor(X_train.values).float()
        y_train = torch.tensor(y_train.values).reshape((y_train.shape[0], ))      
            
        if self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            optimizer = optim.AdamW(
                list(self.mlp.parameters())+list(self.embedding_layer_cont.parameters())+list(self.embedding_layer_cont.parameters()), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )                
        elif not self.params["n_cat_features"]>0 and self.params["n_cont_features"]>0:
            optimizer = optim.AdamW(
                list(self.mlp.parameters())+list(self.embedding_layer_cont.parameters()), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )        
        elif self.params["n_cat_features"]>0 and not self.params["n_cont_features"]>0:
            # Define optimizer
            optimizer = optim.AdamW(
                list(self.mlp.parameters())+list(self.embedding_layer.parameters()), 
                lr=self.params["hyperparameters"]["learning_rate"], 
                weight_decay=self.params["hyperparameters"]["weight_decay"]
            )
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
            # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
            
         
        from torcheval.metrics import BinaryAUROC 

        reset_metric = True
        # Define loss function
        if self.params["task_type"] == "regression":
            loss_func = nn.MSELoss()
            if self.params["eval_metric"]=="rmse":
                eval_func = RMSELoss()
                eval_direction = "minimize"
                reset_metric = False
            elif self.params["eval_metric"]=="r2":
                eval_func = R2Score().to(self.params["device"])
                eval_direction = "maximize"
            elif self.params["eval_metric"]=="mae":
                eval_func = nn.L1Loss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.MSELoss()
                eval_direction = "minimize"
                reset_metric = False
            y_train = y_train.float()
            y_val = y_val.float()
        elif self.params["task_type"] == "multiclass":
            loss_func = nn.CrossEntropyLoss()
            if self.params["eval_metric"]=="mlogloss":
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False                
        else:
            loss_func = nn.BCEWithLogitsLoss()
            if self.params["eval_metric"] in ["auc", "gini"]:
                eval_func = AUROC(task="binary").to(self.params["device"])
                eval_direction = "maximize"
            else:
                eval_func = nn.BCEWithLogitsLoss()
                eval_direction = "minimize"
                reset_metric = False
                
            y_train = y_train.float()
            y_val = y_val.float()

        # Define data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=False
        )
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.params["val_batch_size"], shuffle=True
        )
        
        # Start training loop
        if eval_direction=="minimize":
            min_val = float("inf")
        else:
            min_val = -float("inf")

        min_val_idx = 0
        
        # print(f"total_params = {np.round(sum(p.numel() for p in self.mlp.parameters())/1000000,3)}M")
        
        history = {}
        history["loss"] = []
        history["val_loss"] = []            
        history["eval_metric"] = []
        history["eval_metric_val"] = []            
        start_time = time.time()
        for epoch in range(self.params["epochs"]):
            # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            self.mlp.train()
            if self.params["n_cat_features"]>0:
                self.embedding_layer.train()
            if self.params["n_cont_features"]>0:
                self.embedding_layer_cont.train()
            for i, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.params["device"])
                out = self.forward(batch_X)
                
                if (
                    self.params["task_type"] == "regression"
                    or self.params["task_type"] == "binary"
                ):
                    # out = out.squeeze()
                    out = out.reshape((batch_X.shape[0], ))
                
                loss = loss_func(out, batch_y.to(self.params["device"]))
                history["loss"].append(loss.detach().cpu())
            
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            
            
            # Early Stopping
            val_eval = 0.0
            val_loss = 0.0
            val_dim = 0
            predictions = []
            true = []
            with torch.no_grad():
                self.mlp.eval()
                if self.params["n_cat_features"]>0:
                    self.embedding_layer.eval()
                if self.params["n_cont_features"]>0:
                    self.embedding_layer_cont.eval()                    
                for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                    if reset_metric:
                        eval_func.reset()
                    batch_val_X = batch_val_X.to(self.params["device"])
                    batch_val_y = batch_val_y.to(self.params["device"])
                    
                    out = self.forward(batch_val_X)
    
                    if (
                        self.params["task_type"] == "regression"
                        or self.params["task_type"] == "binary"
                    ):
                        #out = out.squeeze()
                        out = out.reshape((batch_val_X.shape[0], ))
                    
                    val_loss += loss_func(out, batch_val_y)
                    # Store predictions and batch order to compute metrics outside batches as some metrics are non-cumulative
                    predictions.append(out.cpu().detach())
                    true.append(batch_val_y.cpu().detach())
                    # val_eval += eval_func(out, batch_val_y)
                    val_dim += 1
                
                
                # val_eval /= val_dim
                if reset_metric:
                    eval_func.reset()
                val_eval = eval_func(torch.concatenate(predictions).to(self.params["device"]),
                                     torch.concatenate(true).to(self.params["device"]), 
                                     )
                del true, predictions        
                
                val_loss /= val_dim
                history["val_loss"].append(val_loss.detach().cpu())
                history["eval_metric_val"].append(val_eval.detach().cpu())
                
                print("Epoch %d, Val Loss: %.5f, Val Metric: %.5f" % (epoch, val_loss, val_eval))
                
                if eval_direction=="minimize":
                    condition = val_eval < min_val
                else:
                    condition = val_eval > min_val
    
                if condition:
                    min_val = val_eval.detach().cpu()
                    min_val_idx = epoch
                    
                    # Save the currently best model
                    torch.save(self.mlp.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_ResNet.pt")
                    if self.params["n_cat_features"]>0:
                        torch.save(self.embedding_layer.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_Embeddings.pt")
                    if self.params["n_cont_features"]>0:
                        torch.save(self.embedding_layer_cont.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_Embeddings_Cont.pt")
                        
                    
                if min_val_idx + self.params["patience"] < epoch:
                    # print(
                    #     "Validation loss has not improved for %d steps!"
                    #     % self.params["patience"]
                    # )
                    print(
                        f"Validation loss has not improved for {self.params['patience']} steps after {epoch} epochs!"
                    )
    
                    print("Early stopping applies.")
                    break
                
                # scheduler.step(val_loss)
    
                runtime = time.time() - start_time
                # if runtime > time_limit:
                #     print(
                #         f"Runtime has exceeded time limit of {time_limit} seconds. Stopping fit."
                #     )
                #     break
    
                # torch.cuda.empty_cache()
    
        # Load best model
        state_dict_resnet = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_ResNet.pt")
        self.mlp.load_state_dict(state_dict_resnet)
        if self.params["n_cat_features"]>0:
            state_dict_embeddings = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_Embeddings.pt")
            self.embedding_layer.load_state_dict(state_dict_embeddings)
        if self.params["n_cont_features"]>0:
            state_dict_embeddings_cont = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_Embeddings_Cont.pt")
            self.embedding_layer_cont.load_state_dict(state_dict_embeddings_cont)

        
        torch.cuda.empty_cache()
        gc.collect()
        return history        
        
        
    def predict(self, X):
       # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        X = torch.tensor(X.values).float()
        
        test_dataset = TensorDataset(X)
        test_loader = DataLoader( 
            dataset=test_dataset,
            batch_size=self.params["val_batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.mlp.eval()
        if self.params["n_cat_features"]>0:
            self.embedding_layer.eval()
        if self.params["n_cont_features"]>0:
            self.embedding_layer_cont.eval()              
        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                if self.params["task_type"]=="binary":
                    preds = torch.sigmoid(self.forward(batch_X[0].to(self.params["device"]))).cpu().detach().numpy()
                elif self.params["task_type"]=="multiclass":
                    preds = torch.nn.functional.softmax((self.forward(batch_X[0].to(self.params["device"]))), dim=1).cpu().detach().numpy()
                else:
                    preds = self.forward(batch_X[0].to(self.params["device"])).cpu().detach().numpy()                       
                predictions.append(preds)
        
        return np.concatenate(predictions)
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, large_dataset=False, **kwargs):
        params = {
            "n_blocks": trial.suggest_int("n_blocks",1,8), # Original paper: (A) UniformInt[1, 8], (B) UniformInt[1, 16], Kadra: first
            "d_block": trial.suggest_int("d_block",1,1024), # Original paper: Not used, default settings for FTTransformer are: [96, 128, 192, 256, 320, 384][n_blocks - 1], Kadra: [64,1024]
            "dropout": trial.suggest_float("dropout",0.,0.5), # Original paper:  (A,B) Uniform[0, 0.5], Kadra: same
            "d_embedding": trial.suggest_int("d_embedding",1,512), # Original paper: only for one dataset: UniformInt[64, 512], Kadra: same; we adapted lower bound to 4 as not all datasets need high dimensions for cat features
            "d_embedding_num": trial.suggest_int("d_embedding_num",1,128), # Original paper: only for one dataset: UniformInt[64, 512], Kadra: same; we adapted lower bound to 4 as not all datasets need high dimensions for cat features
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 0.005, log=True),  # Original paper: LogUniform[1e-5, 1e-2]
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True), # Original paper: ?, Kadra: 1e-6, 1e-3
            
        }
        
        # params["d_embedding"] = params["d_embedding_num"]
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self, large_dataset=False):
        # if large_dataset:
        #     learning_rate = 0.00001
        #     n_blocks = 4
        #     weight_decay=0.000001
        # else:
        #     learning_rate = 0.001
        #     weight_decay=0.0001
        #     n_blocks = 2
        learning_rate = 0.001
        weight_decay=0.0001
        params = {
            "n_blocks": 2, #2
            "d_block": 192, # 
            "dropout": 0.25, # 
            "d_embedding": 8, # 
            "d_embedding_num": 8, # 
            "learning_rate": learning_rate, # 0.001
            "weight_decay": weight_decay
            
        }
        return params    


#################################################################
#################################################################
#################################################################


class TabM(BaseModel):
    def __init__(self, params):
        '''License: https://github.com/yandex-research/tabm/blob/main/LICENSE '''
        super().__init__(params)

        self.arch_type = 'tabm'
        self.bins = None
        
        # Model-specific fixed parameters
        self.params["n_cont_features"] = len(self.params["cont_indices"])
        self.params["n_cat_features"] = len(self.params["cat_indices"])
        # self.batch_size = params["batch_size"]
        # self.val_batch_size = params["val_batch_size"]
        # self.epochs = params["epochs"]
        # self.patience = params["patience"]
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters(self.params["large_dataset"])       
        
        # self.model = Model(
        #     n_num_features=self.params["n_cont_features"],
        #     cat_cardinalities=self.params["cat_cardinalities"],
        #     n_classes=self.params["d_out"],
        #     backbone={
        #         'type': 'MLP',
        #         'n_blocks': self.params["hyperparameters"]["n_blocks"],
        #         'd_block': self.params["hyperparameters"]["d_block"],
        #         'dropout': self.params["hyperparameters"]["dropout"],
        #     },
        #     bins=bins,
        #     num_embeddings=(
        #         None
        #         if bins is None
        #         else {
        #             'type': 'PiecewiseLinearEmbeddings',
        #             'd_embedding': 16,
        #             'activation': False,
        #             'version': 'B',
        #         }
        #     ),
        #     arch_type=arch_type,
        #     k=32,
        # )
        
        
        # self.model.to(self.params["device"])
        
        # if self.params["n_cat_features"]>0:
        #     self.embedding_layer = CategoricalEmbeddings(self.params["cat_cardinalities"],d_embedding=self.params["hyperparameters"]["d_embedding"])
        #     self.embedding_layer.to(self.params["device"])

    
    def forward(self,X):
        if self.params["n_cat_features"]>0:
            X_cont = X[:,self.params["cont_indices"]]
            X_cat = X[:,self.params["cat_indices"]].long()
            
            # X_cat_embed = self.embedding_layer(X_cat)
            # X_cont_catembed = torch.cat([X_cont, X_cat_embed.flatten(1, -1)],dim=1)
            # print(X_cont.dtype,X_cat.dtype)
            res = self.model(X_cont,X_cat)
        else:
            res = self.model(X,None)
        
        return res
        
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):
        
        
        if self.bins is not None:
            if self.params["n_cont_features"]>0:
                self.bins = rtdl_num_embeddings.compute_bins(torch.Tensor(X_train.iloc[:,self.params["cont_indices"]].values))
            else:
                self.bins = None
        self.model = Model(
            n_num_features=self.params["n_cont_features"],
            cat_cardinalities=self.params["cat_cardinalities"],
            n_classes=self.params["d_out"],
            backbone={
                'type': 'MLP',
                'n_blocks': self.params["hyperparameters"]["n_blocks"],
                'd_block': self.params["hyperparameters"]["d_block"],
                'dropout': self.params["hyperparameters"]["dropout"],
            },
            bins=self.bins,
            num_embeddings=(
                None
                if self.bins is None
                else {
                    'type': 'PiecewiseLinearEmbeddings',
                    'd_embedding': self.params["hyperparameters"]["d_embedding"],
                    'activation': False,
                    'version': 'B',
                }
            ),
            arch_type=self.arch_type,
            k=32,
        )
        
        
        self.model.to(self.params["device"])

        
        ######################################################
        
        '''
        Partially copied from https://github.com/naszilla/tabzilla/blob/main/TabZilla/models/basemodel_torch.py
        
        '''
        # Create the save path
        if not os.path.exists(self.params["save_path"]):
            os.makedirs(self.params["save_path"])
        
        # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X_train.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        if eval_set is not None:
            X_val = eval_set[0][0]
            y_val = eval_set[0][1]
            
            X_val = torch.tensor(X_val.values).float()
            y_val = torch.tensor(y_val.values).reshape((y_val.shape[0], ))

            eval_set = [(X_val,y_val)]            
        
        X_train = torch.tensor(X_train.values).float()
        y_train = torch.tensor(y_train.values).reshape((y_train.shape[0], )) 
        
        # if self.params["n_cat_features"]>0:
        #     # Define optimizer
        #     optimizer = optim.AdamW(
        #         list(self.model.parameters())+list(self.embedding_layer.parameters()), 
        #         lr=self.params["hyperparameters"]["learning_rate"], 
        #         weight_decay=self.params["hyperparameters"]["weight_decay"]
        #     )
        #     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
        #     # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
        # else:
        optimizer = optim.AdamW(
            make_parameter_groups(self.model), # self.model.parameters(), 
            lr=self.params["hyperparameters"]["learning_rate"], 
            weight_decay=self.params["hyperparameters"]["weight_decay"]
        )
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
            # optimizer = SWA(base_optimizer, swa_start=2, swa_freq=2, swa_lr=self.params["hyperparameters"]["learning_rate"])
        
        evaluation_mode = torch.inference_mode
         
        from torcheval.metrics import BinaryAUROC 

        reset_metric = True
        # Define loss function
        if self.params["task_type"] == "regression":
            base_loss_fn = nn.MSELoss()
            if self.params["eval_metric"]=="rmse":
                eval_func = RMSELoss()
                eval_direction = "minimize"
                reset_metric = False
            elif self.params["eval_metric"]=="r2":
                eval_func = R2Score().to(self.params["device"])
                eval_direction = "maximize"
            elif self.params["eval_metric"]=="mae":
                eval_func = nn.L1Loss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.MSELoss()
                eval_direction = "minimize"
                reset_metric = False
            y_train = y_train.float()
            y_val = y_val.float()
        elif self.params["task_type"] == "multiclass":
            base_loss_fn = nn.CrossEntropyLoss()
            if self.params["eval_metric"]=="mlogloss":
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False
            else:
                eval_func = nn.CrossEntropyLoss()
                eval_direction = "minimize"
                reset_metric = False
        else:
            base_loss_fn = nn.BCEWithLogitsLoss()
            if self.params["eval_metric"] in ["auc", "gini"]:
                eval_func = AUROC(task="binary").to(self.params["device"])
                eval_direction = "maximize"
            else:
                eval_func = nn.BCEWithLogitsLoss()
                eval_direction = "minimize"
                reset_metric = False
                
            y_train = y_train.float()
            y_val = y_val.float()

        # Define data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=False
        )
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.params["val_batch_size"], shuffle=True
        )
        
        # Start training loop
        if eval_direction=="minimize":
            min_val = float("inf")
        else:
            min_val = -float("inf")

        min_val_idx = 0


        def loss_func(y_pred: Tensor, y_true: Tensor) -> Tensor:
            # TabM produces k predictions per object. Each of them must be trained separately.
            # (regression)     y_pred.shape == (batch_size, k)
            # (classification) y_pred.shape == (batch_size, k, n_classes)
            # k = y_pred.shape[-1 if self.params["task_type"] == 'regression' else -2]
            if self.params["task_type"]=="multiclass":
                k = y_pred.shape[-2]
                return base_loss_fn(y_pred.flatten(0,1), y_true.repeat_interleave(k))
            else:
                k = y_pred.shape[-2]
                return base_loss_fn(y_pred.flatten(), y_true.repeat_interleave(k))
            # return base_loss_fn(y_pred.flatten(), y_true.repeat_interleave(k))
        
        history = {}
        history["loss"] = []
        history["val_loss"] = []            
        history["eval_metric"] = []
        history["eval_metric_val"] = []            
        start_time = time.time()
        for epoch in range(self.params["epochs"]):
            # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            self.model.train()
            # if self.params["n_cat_features"]>0:
            #     self.embedding_layer.train()
            for i, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.params["device"])
                out = self.forward(batch_X)
                
                # if (
                #     self.params["task_type"] == "regression"
                #     or self.params["task_type"] == "binary"
                # ):
                    # out = out.squeeze()
                    # out = out.reshape((batch_X.shape[0], ))
                
                loss = loss_func(out, batch_y.to(self.params["device"]))
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            history["loss"].append(loss.detach().cpu())
            
            # Early Stopping
            val_eval = 0.0
            val_loss = 0.0
            val_dim = 0
            predictions = []
            true = []
            with torch.no_grad():
                self.model.eval()
                # if self.params["n_cat_features"]>0:
                #     self.embedding_layer.eval()
                for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                    if reset_metric:
                        eval_func.reset()
                    batch_val_X = batch_val_X.to(self.params["device"])
                    batch_val_y = batch_val_y.to(self.params["device"])
                    
                    out = self.forward(batch_val_X)
    
                    # if (
                    #     self.params["task_type"] == "regression"
                    #     or self.params["task_type"] == "binary"
                    # ):
                        #out = out.squeeze()
                        # out = out.reshape((batch_val_X.shape[0], ))
                    
                    val_loss += loss_func(out, batch_val_y)
                    # Store predictions and batch order to compute metrics outside batches as some metrics are non-cumulative
                    predictions.append(out.cpu().detach())
                    true.append(batch_val_y.cpu().detach())
                    # val_eval += eval_func(out, batch_val_y)
                    val_dim += 1

                # val_eval /= val_dim
                if reset_metric:
                    eval_func.reset()
                
                y_pred = torch.concatenate(predictions)
                y_pred = y_pred.mean(1)

                if not self.params["task_type"]=="multiclass":
                    y_pred = y_pred.flatten()
                
                val_eval = eval_func(y_pred.to(self.params["device"]),
                                     torch.concatenate(true).to(self.params["device"]), 
                                     )
                del true, predictions, y_pred                
                
                val_loss /= val_dim
                history["val_loss"].append(val_loss.detach().cpu())
                history["eval_metric_val"].append(val_eval.detach().cpu())
                
                print("Epoch %d, Val Loss: %.5f, Val Metric: %.5f" % (epoch, val_loss, val_eval))
                
                if eval_direction=="minimize":
                    condition = val_eval < min_val
                else:
                    condition = val_eval > min_val
    
                if condition:
                    min_val = val_eval.detach().cpu()
                    min_val_idx = epoch
                    
                    # Save the currently best model
                    torch.save(self.model.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_TabM.pt")
                    # if self.params["n_cat_features"]>0:
                        # torch.save(self.embedding_layer.state_dict(), self.params["save_path"] + f"/{self.params['exp_name']}_Embeddings.pt")
                
                if min_val_idx + self.params["patience"] < epoch:
                    # print(
                    #     "Validation loss has not improved for %d steps!"
                    #     % self.params["patience"]
                    # )
                    print(
                        f"Validation loss has not improved for {self.params['patience']} steps after {epoch} epochs!"
                    )
    
                    print("Early stopping applies.")
                    break
                
                # scheduler.step(val_loss)
    
                runtime = time.time() - start_time
                # if runtime > time_limit:
                #     print(
                #         f"Runtime has exceeded time limit of {time_limit} seconds. Stopping fit."
                #     )
                #     break
    
                # torch.cuda.empty_cache()
    
        # Load best model
        state_dict_model = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_TabM.pt")
        self.model.load_state_dict(state_dict_model)
        # if self.params["n_cat_features"]>0:
        #     state_dict_embeddings = torch.load(self.params["save_path"]+f"/{self.params['exp_name']}_Embeddings.pt")
            # self.embedding_layer.load_state_dict(state_dict_embeddings)
        torch.cuda.empty_cache()
        gc.collect()

        self.history = history
        
        return history        
        
        
    def predict(self, X):
       # Prepare data
        if self.params["n_cat_features"]>0:
            for col in self.params["cat_indices"]:
                if X.iloc[:,col].dtype in [str, "object", "category"]:
                    raise ValueError(f"Feature {col} required ordinal encoding")
        
        X = torch.tensor(X.values).float()
        
        test_dataset = TensorDataset(X)
        test_loader = DataLoader( 
            dataset=test_dataset,
            batch_size=self.params["val_batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.model.eval()
        # if self.params["n_cat_features"]>0:
        #     self.embedding_layer.eval()

        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                if self.params["task_type"]=="binary":
                    preds = torch.sigmoid(self.forward(batch_X[0].to(self.params["device"])))
                    # preds = self.forward(batch_X[0].to(self.params["device"]))
                    
                elif self.params["task_type"] in ["multiclass"]:
                    preds = torch.nn.functional.softmax((self.forward(batch_X[0].to(self.params["device"]))), dim=2)
                else:
                    preds = self.forward(batch_X[0].to(self.params["device"]))                       
                predictions.append(preds)
        if self.params["task_type"] in ["multiclass"]:
            y_pred = torch.concatenate(predictions).mean(1).cpu().detach().numpy()
        else:
            y_pred = torch.concatenate(predictions).mean(1).flatten().cpu().detach().numpy()
        
        return y_pred
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, large_dataset=False, dataset_name="", **kwargs):


        params = {
            "k": 32,
            "n_blocks": trial.suggest_int("n_blocks",1,5), 
            "d_block": trial.suggest_int("d_block",64,1024), #
            "d_embedding": 16, # NOT USED!
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True), 
            
        }

        params["use_dropout"] = trial.suggest_categorical("use_dropout", [True, False])
        if params["use_dropout"]: 
            params["dropout"] = trial.suggest_float("dropout",0.,0.5) 
        else:
            params["dropout"] = 0.

        params["use_decay"] = trial.suggest_categorical("use_decay", [True, False])
        if params["use_decay"]: 
            params["weight_decay"] = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True) 
        else:
            params["weight_decay"] = 0.


        
        return params
    
    @classmethod
    def get_default_hyperparameters(self, large_dataset=False):
        params = {
            "k": 32, #2
            "n_blocks": 3, #2
            "d_block": 512, # 
            "dropout": 0.1, # 
            "d_embedding": 16, # NOT USED!
            "learning_rate": 0.002, # 0.001
            "weight_decay": 0.0003
            
        }

        return params

class TabMStopInterpol(TabM):
    def __init__(self, params):
        ''' '''
        super().__init__(params)
         
    def get_uniques(self, X):
        self.train_values = {}
        for column in X.columns:
            self.train_values[column] = np.array(X[column].unique())

    
    def replace_with_closest_efficient(self, X):
        X_modified = X.copy()
    
        # self.train_values = {}
        for column in X.columns:
            # self.train_values[column] = np.array(X_train[column].unique())
            
            # Build a KDTree for fast nearest-neighbor lookup
            tree = KDTree(self.train_values[column].reshape(-1, 1))
            
            # Identify values in X_test that are not in X_train
            test_values = np.array(X[column])
            distances, indices = tree.query(test_values.reshape(-1, 1))
            
            # Replace X_test values with their closest match in X_train
            X_modified[column] = self.train_values[column][indices]
        
        return X_modified
    
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):

        self.cont_indices = list(set(range(len(X_train.columns)))-set(self.params["cat_indices"]))
        self.get_uniques(X_train.iloc[:,self.cont_indices])
        
        X_val = eval_set[0][0]
        X_val.iloc[:,self.cont_indices] = self.replace_with_closest_efficient(X_val.iloc[:,self.cont_indices])
 
        super().fit(X_train, y_train, [(X_val, eval_set[0][1])])


    def predict(self, X):
       X.iloc[:,self.cont_indices] = self.replace_with_closest_efficient(X.iloc[:,self.cont_indices])
       return super().predict(X)

class TabMmini(TabM):
    def __init__(self, params):
        '''License: https://github.com/yandex-research/tabm/blob/main/LICENSE '''
        super().__init__(params)

        self.arch_type = 'tabm-mini'
        self.bins = "determine"

    @classmethod
    def get_optuna_hyperparameters(self, trial, large_dataset=False, dataset_name="", **kwargs):


        params = {
            "k": 32,
            "n_blocks": trial.suggest_int("n_blocks",1,5), # Original paper: (A) UniformInt[1, 8], (B) UniformInt[1, 16], Kadra: first
            "d_block": trial.suggest_int("d_block",64,1024), # Original paper: Not used, default settings for FTTransformer are: [96, 128, 192, 256, 320, 384][n_blocks - 1], Kadra: [64,1024]
            "d_embedding": trial.suggest_int("d_block",8,32),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),  # Original paper: LogUniform[1e-5, 1e-2]
            
        }

        params["use_dropout"] = trial.suggest_categorical("use_dropout", [True, False])
        if params["use_dropout"]: 
            params["dropout"] = trial.suggest_float("dropout",0.,0.5) # Original paper:  (A,B) {0, Uniform[0, 0.5]}, Kadra: same
        else:
            params["dropout"] = 0.

        params["use_decay"] = trial.suggest_categorical("use_decay", [True, False])
        if params["use_decay"]: 
            params["weight_decay"] = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True) # Original paper: ?, Kadra: 1e-6, 1e-3
        else:
            params["weight_decay"] = 0.


        
        return params
    
#################################################################
#################################################################
#################################################################


    
#################################################################
#################################################################
#################################################################

    
class AutoGluonModel(BaseModel):
    def __init__(self, params):
        '''
        Model-specific Parameters in params:
            time_limit
            presets
            num_cpus
        '''
        super().__init__(params)
        
        
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = None
        elif self.params["eval_metric"] in ["auc", "gini", "mauc"]:
            self.params["eval_metric"] = "roc_auc"
        elif self.params["eval_metric"]=="rmse":
            self.params["eval_metric"] = "root_mean_squared_error"
        elif self.params["eval_metric"]=="ams":
            self.params["eval_metric"] = "log_loss"
        elif self.params["eval_metric"]=="mae":
            self.params["eval_metric"] = "mean_absolute_error"
        elif self.params["eval_metric"]=="mlogloss":
            self.params["eval_metric"] = "log_loss"
        elif self.params["eval_metric"]=="logloss":
            self.params["eval_metric"] = "log_loss"
            
        
        if "num_cpus" not in params:
            self.params["num_cpus"] = None
                
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters()                            
        
    def fit(self, 
            X_train, y_train, 
            eval_set=None,
           ):
        
        label = y_train.name
        data = pd.concat([X_train,y_train],axis=1)
        
        self.model = TabularPredictor(label, eval_metric=self.params["eval_metric"],
                                      path=f"./logs/AutoGluon/{self.params['dataset_name']}_{self.params['exp_name']}",

                                     )
        self.model.fit(data, 
                       # ag_args_fit={'num_gpus': 1},
                       time_limit=self.params["time_limit"], 
                       presets=self.params["presets"],
                       ag_args_fit={'num_cpus': self.params["num_cpus"]},
                      )

        
        
    def predict(self, X):
        if self.params["task_type"]=="binary":
            pred = self.model.predict_proba(X).iloc[:,1].values            
        if self.params["task_type"]=="multiclass":
            pred = self.model.predict_proba(X).values            
        else:
            pred = self.model.predict(X).values
            
        return pred    
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, **kwargs):
        params = {}        
        
        return params    
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {}        
        
        return params    
    

class XGBModel(BaseModel):
    def __init__(self, params):
        '''
        Model specific hyperparameters: n_estimators, eval_metric
        '''
        
        super().__init__(params)
        
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = None
        # Use rmse for r2 as objective is same
        elif self.params["eval_metric"]=="r2":
            self.params["eval_metric"] = "rmse"
        elif self.params["eval_metric"]=="ams":
            self.params["eval_metric"] = "logloss"
        elif self.params["eval_metric"] in ["gini"]:
            self.params["eval_metric"] = "auc"
        elif self.params["eval_metric"] in ["mauc"]:
            self.params["eval_metric"] = "mlogloss"
        elif self.params["eval_metric"] in ["Accuracy"]:
            self.params["eval_metric"] = "error"
        elif self.params["eval_metric"] in ["mAccuracy"]:
            self.params["eval_metric"] = "merror"
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters()               
        
        if self.params["task_type"] == "regression":
            self.model = xgb.XGBRegressor(
                **self.params["hyperparameters"],
                booster = "gbtree",
                tree_method = "hist",
                enable_categorical=True,
                sampling_method="gradient_based",
                n_estimators = self.params["n_estimators"],
                early_stopping_rounds=self.params["patience"],
                device=self.params["device"],
                max_bin = self.params["max_bin"],
                eval_metric = self.params["eval_metric"],
            )
        elif self.params["task_type"] in ["binary", "multiclass"]:
            self.model = xgb.XGBClassifier(
                **self.params["hyperparameters"],
                booster = "gbtree",
                tree_method = "hist", 
                enable_categorical=True,
                sampling_method="gradient_based",
                n_estimators = self.params["n_estimators"],
                early_stopping_rounds=self.params["patience"],
                device=self.params["device"],
                max_bin = self.params["max_bin"],
                eval_metric = self.params["eval_metric"],
            )
        
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):

        # if self.params["dataset_name"]=="bnp-paribas-cardif-claims-management":
        # self.params["cat_indices"] = 
        
        self.cat_col_names = X_train.iloc[:,self.params["cat_indices"]].columns
        self.cat_dtypes = {}
        if eval_set is not None:
            # X_val = eval_set[0][0]  
            # y_val = eval_set[0][1]
            for num, (X_val, y_val) in enumerate(eval_set):
                for col in self.cat_col_names:
                    if X_train.loc[:,col].dtype!="category":
                        # X_train.loc[:,col] = X_train[col].astype(object)
                        # X_val.loc[:,col] = X_val[col].astype(object)
                        # u_cats = list(X_train[col].unique())+["nan"] #np.unique(list(X_train[col].unique())+list(X_val[col].unique())+["nan"]).tolist()
                        # self.cat_dtypes[col] = pd.CategoricalDtype(categories=u_cats)
                        # X_train.loc[:,col] = X_train.loc[:,col].astype(self.cat_dtypes[col])
                        # X_val.loc[:,col] = X_val.loc[:,col].astype(self.cat_dtypes[col])
    
                        self.cat_dtypes[col] = pd.CategoricalDtype(categories=list(X_train[col].unique())+["nan"])
                        X_train[col] = X_train[col].astype(self.cat_dtypes[col])
                        X_val[col] = X_val[col].astype(self.cat_dtypes[col])           

                eval_set[num] = (X_val,y_val)
        else:
            for col in self.cat_col_names:
                if X_train.loc[:,col].dtype!="category":
                    self.cat_dtypes[col] = pd.CategoricalDtype(categories=list(X_train[col].unique())+["nan"])
                    X_train[col] = X_train[col].astype(self.cat_dtypes[col])
            eval_set = [(X_train,y_train)]

        if "sample_weights" in self.params["hyperparameters"]:
            sample_weights = self.params["hyperparameters"]["sample_weights"]
        else:
            sample_weights = None
        
        h = self.model.fit(X_train, y_train, 
            eval_set=eval_set,
            verbose=50,
            sample_weight = sample_weights
                          )

        
        
    def predict(self, X):
        for col in self.cat_col_names:
            if X.loc[:,col].dtype!="category":
                X[col] = X[col].astype(self.cat_dtypes[col])

        self.model.set_params(device="cpu")
        
        if self.params["task_type"]=="regression":
            pred = self.model.predict(X)            
        elif self.params["task_type"]=="binary":
            pred = self.model.predict_proba(X)[:,1]            
        elif self.params["task_type"]=="multiclass":
            pred = self.model.predict_proba(X)         
        
        self.model.set_params(device="cuda") 
        
        return pred    
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, dataset_name = "", **kwargs):
        if dataset_name == "bnp-paribas-cardif-claims-management":
            max_depth = 5
        else:        
            max_depth = 11
        params = {
            "eta": trial.suggest_float("eta", 1e-3, 0.7, log=True), # From Shwartz et al - reduced min threshold as 1e-3 is already really low and only increases training time unnecessarily
            "max_depth": trial.suggest_int("max_depth", 1, max_depth, log=False), # Increased depth to allow learning higher-order interactions in one tree 
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5,1.0, log=False), # From Shwartz et al
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5,1.0, log=False), # From Shwartz et al
            "subsample": trial.suggest_float("subsample", 0.5,1.0, log=False), # From Shwartz et al
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8,1e2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1,4, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8,7, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 1e2, log=True)
        }       
        

        
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {
            "eta": 0.3, # From XGB Docu
            "max_depth": 6, # From XGB Docu 
            "colsample_bylevel": 1, # From XGB Docu
            "colsample_bytree": 1, # From XGB Docu
            "subsample": 1., # From XGB Docu
            "min_child_weight": 1, # From XGB Docu
            "reg_alpha": 0., # From XGB Docu
            "reg_lambda": 1, # From XGB Docu
            "gamma": 0, # From XGB Docu
            }        
        
        return params    

class XGBoostSmallData(XGBModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters()
        
        super().__init__(params)
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, dataset_name = "", **kwargs):
        if dataset_name == "bnp-paribas-cardif-claims-management":
            max_depth = 5
        else:        
            max_depth = 11
        params = {
            "eta": trial.suggest_float("eta", 1e-3, 0.7, log=True), # From Shwartz et al - reduced min threshold as 1e-3 is already really low and only increases training time unnecessarily
            "max_depth": trial.suggest_int("max_depth", 1, max_depth, log=False), # Increased depth to allow learning higher-order interactions in one tree 
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5,1.0, log=False), # From Shwartz et al
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5,1.0, log=False), # From Shwartz et al
            "subsample": trial.suggest_float("subsample", 0.5,1.0, log=False), # From Shwartz et al
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8,1e2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1,4, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8,7, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 1e2, log=True)
        }
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {
            "eta": 0.3, # From XGB Docu
            "max_depth": 6, # From XGB Docu 
            "colsample_bylevel": 1, # From XGB Docu
            "colsample_bytree": 1, # From XGB Docu
            "subsample": 1., # From XGB Docu
            "min_child_weight": 0., # From XGB Docu
            "reg_alpha": 0., # From XGB Docu
            "reg_lambda": 1, # From XGB Docu
            "gamma": 0, # From XGB Docu
            }        
        
        return params    


class XGBoostHolzmueller(XGBModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters(task_type=params["task_type"])
        
        params["n_estimators"] = 1000
        params["patience"] = 300
        params["max_bin"] = 256

        super().__init__(params)
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, **kwargs):
        ### !!! NOT USED
        params = {
            "max_depth": trial.suggest_int("max_depth", 1,11, log=False), 
            "eta": trial.suggest_float("eta", 1e-5, 0.7, log=True), 
           "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5,1.0, log=False), 
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5,1.0, log=False),
            "subsample": trial.suggest_float("subsample", 0.5,1.0, log=False), 
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8,1e2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1,4, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8,7, log=True),
            
            "learning_rate": 0.05,
            "subsample": 0.7,
            "colsample_bytree": 1.0, 
            "colsample_bylevel": 1.0, 
            "min_child_weight": 2.0,
            "lambda": 0.0,
        #     "n_estimators": 1000,
        #     "max_bin": 256,
        #     "early_stopping_rounds": 300,
        }        
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self, task_type="binary", **kwargs):
        if task_type in ["binary", "multiclass", "classification"]:
            params = {
                "max_depth": 6, #5
                "learning_rate": 0.08, #0.08
                "subsample": 0.65,
                "colsample_bytree": 1.0, 
                "colsample_bylevel": 0.9, 
                "min_child_weight": 5e-6,
                "lambda": 0.0,
                # "tree_method": "hist",
                # "n_estimators": 1000,
                # "max_bin": 256,
                # "early_stopping_rounds": 300,
            }        
        elif task_type in ["regression"]:
            params = {
                "max_depth": 9, 
                "learning_rate": 0.05,
                "subsample": 0.7,
                "colsample_bytree": 1.0, 
                "colsample_bylevel": 1.0, 
                "min_child_weight": 2.0,
                "lambda": 0.0,
                # "tree_method": "hist",
                # "n_estimators": 1000,
                # "max_bin": 256,
                # "early_stopping_rounds": 300,
            }        
            
        return params    


class XGBoostLossguided(XGBModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters()
        super().__init__(params)
        

    
    @classmethod
    def get_optuna_hyperparameters(self, trial, n_features=1, large_dataset=False, **kwargs):
        params = {
            "grow_policy": "lossguide",
            "eta": trial.suggest_float("eta", 1e-3, 0.7, log=True), # From Shwartz et al - reduced min threshold as 1e-3 is already really low and only increases training time unnecessarily
            # "max_depth": trial.suggest_int("max_depth", 1, max_depth, log=False), # Increased depth to allow learning higher-order interactions in one tree 
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5,1.0, log=False), # From Shwartz et al
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5,1.0, log=False), # From Shwartz et al
            "subsample": trial.suggest_float("subsample", 0.5,1.0, log=False), # From Shwartz et al
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8,1e2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1,4, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8,7, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 1e2, log=True)
        }       
        
        # params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [20, 50, 100, 500, 1000, 2000])

        params["max_depth"] = trial.suggest_int("max_depth", 1, 11)
        params["max_leaves"] = trial.suggest_int("max_leaves", 2*params["max_depth"], np.max([2*params["max_depth"],(2**params["max_depth"])-1]))
        

        
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {
            "grow_policy": "lossguide",
            # "learning_rate": 0.08, #0.08
            "max_depth": 20, #5
            "max_leaves": 64
            # "l2_leaf_reg": 5, 
            # "bagging_temperature": 1,
            # "leaf_estimation_iterations": 1
        }        
        
        return params    

class XGBModel1024Bins(XGBModel):
    def __init__(self, params):
        '''
        Model specific hyperparameters: n_estimators, eval_metric
        '''
        params["max_bin"] = 1024
        super().__init__(params)


class XGBModelDepth1(XGBModel):
    def __init__(self, params):
        '''
        Model specific hyperparameters: n_estimators, eval_metric
        '''
        super().__init__(params)

    @classmethod
    def get_optuna_hyperparameters(self, trial, dataset_name = "", **kwargs):
        params = {
            "eta": trial.suggest_float("eta", 1e-3, 0.7, log=True), # From Shwartz et al - reduced min threshold as 1e-3 is already really low and only increases training time unnecessarily
            "max_depth": 1, # trial.suggest_int("max_depth", 1, max_depth, log=False), # Increased depth to allow learning higher-order interactions in one tree 
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5,1.0, log=False), # From Shwartz et al
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5,1.0, log=False), # From Shwartz et al
            "subsample": trial.suggest_float("subsample", 0.5,1.0, log=False), # From Shwartz et al
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8,1e2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1,4, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8,7, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 1e2, log=True)
        }       
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {
            "eta": 0.3, # From XGB Docu
            "max_depth": 1, # From XGB Docu 
            "colsample_bylevel": 1, # From XGB Docu
            "colsample_bytree": 1, # From XGB Docu
            "subsample": 1., # From XGB Docu
            "min_child_weight": 1, # From XGB Docu
            "reg_alpha": 0., # From XGB Docu
            "reg_lambda": 1, # From XGB Docu
            "gamma": 0, # From XGB Docu
            }                
        return params    

class XGBModelDepth2(XGBModel):
    def __init__(self, params):
        '''
        Model specific hyperparameters: n_estimators, eval_metric
        '''
        super().__init__(params)

    @classmethod
    def get_optuna_hyperparameters(self, trial, dataset_name = "", **kwargs):
        params = {
            "eta": trial.suggest_float("eta", 1e-3, 0.7, log=True), # From Shwartz et al - reduced min threshold as 1e-3 is already really low and only increases training time unnecessarily
            "max_depth": 2, # trial.suggest_int("max_depth", 1, max_depth, log=False), # Increased depth to allow learning higher-order interactions in one tree 
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5,1.0, log=False), # From Shwartz et al
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5,1.0, log=False), # From Shwartz et al
            "subsample": trial.suggest_float("subsample", 0.5,1.0, log=False), # From Shwartz et al
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8,1e2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1,4, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8,7, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 1e2, log=True)
        }       
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {
            "eta": 0.3, # From XGB Docu
            "max_depth": 2, # From XGB Docu 
            "colsample_bylevel": 1, # From XGB Docu
            "colsample_bytree": 1, # From XGB Docu
            "subsample": 1., # From XGB Docu
            "min_child_weight": 1, # From XGB Docu
            "reg_alpha": 0., # From XGB Docu
            "reg_lambda": 1, # From XGB Docu
            "gamma": 0, # From XGB Docu
            }                
        return params    

class XGBModelDepth3(XGBModel):
    def __init__(self, params):
        '''
        Model specific hyperparameters: n_estimators, eval_metric
        '''
        super().__init__(params)

    @classmethod
    def get_optuna_hyperparameters(self, trial, dataset_name = "", **kwargs):
        params = {
            "eta": trial.suggest_float("eta", 1e-3, 0.7, log=True), # From Shwartz et al - reduced min threshold as 1e-3 is already really low and only increases training time unnecessarily
            "max_depth": 3, # trial.suggest_int("max_depth", 1, max_depth, log=False), # Increased depth to allow learning higher-order interactions in one tree 
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5,1.0, log=False), # From Shwartz et al
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5,1.0, log=False), # From Shwartz et al
            "subsample": trial.suggest_float("subsample", 0.5,1.0, log=False), # From Shwartz et al
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8,1e2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1,4, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8,7, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 1e2, log=True)
        }       
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {
            "eta": 0.3, # From XGB Docu
            "max_depth": 3, # From XGB Docu 
            "colsample_bylevel": 1, # From XGB Docu
            "colsample_bytree": 1, # From XGB Docu
            "subsample": 1., # From XGB Docu
            "min_child_weight": 1, # From XGB Docu
            "reg_alpha": 0., # From XGB Docu
            "reg_lambda": 1, # From XGB Docu
            "gamma": 0, # From XGB Docu
            }                
        return params    

class XGBModelDepth4(XGBModel):
    def __init__(self, params):
        '''
        Model specific hyperparameters: n_estimators, eval_metric
        '''
        super().__init__(params)

    @classmethod
    def get_optuna_hyperparameters(self, trial, dataset_name = "", **kwargs):
        params = {
            "eta": trial.suggest_float("eta", 1e-3, 0.7, log=True), # From Shwartz et al - reduced min threshold as 1e-3 is already really low and only increases training time unnecessarily
            "max_depth": 4, # trial.suggest_int("max_depth", 1, max_depth, log=False), # Increased depth to allow learning higher-order interactions in one tree 
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5,1.0, log=False), # From Shwartz et al
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5,1.0, log=False), # From Shwartz et al
            "subsample": trial.suggest_float("subsample", 0.5,1.0, log=False), # From Shwartz et al
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8,1e2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1,4, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8,7, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 1e2, log=True)
        }       
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {
            "eta": 0.3, # From XGB Docu
            "max_depth": 4, # From XGB Docu 
            "colsample_bylevel": 1, # From XGB Docu
            "colsample_bytree": 1, # From XGB Docu
            "subsample": 1., # From XGB Docu
            "min_child_weight": 1, # From XGB Docu
            "reg_alpha": 0., # From XGB Docu
            "reg_lambda": 1, # From XGB Docu
            "gamma": 0, # From XGB Docu
            }                
        return params    

class XGBModelDepth5(XGBModel):
    def __init__(self, params):
        '''
        Model specific hyperparameters: n_estimators, eval_metric
        '''
        super().__init__(params)

    @classmethod
    def get_optuna_hyperparameters(self, trial, dataset_name = "", **kwargs):
        params = {
            "eta": trial.suggest_float("eta", 1e-3, 0.7, log=True), # From Shwartz et al - reduced min threshold as 1e-3 is already really low and only increases training time unnecessarily
            "max_depth": 5, # trial.suggest_int("max_depth", 1, max_depth, log=False), # Increased depth to allow learning higher-order interactions in one tree 
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5,1.0, log=False), # From Shwartz et al
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5,1.0, log=False), # From Shwartz et al
            "subsample": trial.suggest_float("subsample", 0.5,1.0, log=False), # From Shwartz et al
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8,1e2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1,4, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8,7, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 1e2, log=True)
        }       
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {
            "eta": 0.3, # From XGB Docu
            "max_depth": 5, # From XGB Docu 
            "colsample_bylevel": 1, # From XGB Docu
            "colsample_bytree": 1, # From XGB Docu
            "subsample": 1., # From XGB Docu
            "min_child_weight": 1, # From XGB Docu
            "reg_alpha": 0., # From XGB Docu
            "reg_lambda": 1, # From XGB Docu
            "gamma": 0, # From XGB Docu
            }                
        return params    




class XGBModelDepth20(XGBModel):
    def __init__(self, params):
        '''
        Model specific hyperparameters: n_estimators, eval_metric
        '''
        super().__init__(params)

    @classmethod
    def get_optuna_hyperparameters(self, trial, dataset_name = "", **kwargs):
        params = {
            "eta": trial.suggest_float("eta", 1e-3, 0.7, log=True), # From Shwartz et al - reduced min threshold as 1e-3 is already really low and only increases training time unnecessarily
            "max_depth": 20, # trial.suggest_int("max_depth", 1, max_depth, log=False), # Increased depth to allow learning higher-order interactions in one tree 
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5,1.0, log=False), # From Shwartz et al
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5,1.0, log=False), # From Shwartz et al
            "subsample": trial.suggest_float("subsample", 0.5,1.0, log=False), # From Shwartz et al
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8,1e2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1,4, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8,7, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 1e2, log=True)
        }       
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {
            "eta": 0.3, # From XGB Docu
            "max_depth": 20, # From XGB Docu 
            "colsample_bylevel": 1, # From XGB Docu
            "colsample_bytree": 1, # From XGB Docu
            "subsample": 1., # From XGB Docu
            "min_child_weight": 1, # From XGB Docu
            "reg_alpha": 0., # From XGB Docu
            "reg_lambda": 1, # From XGB Docu
            "gamma": 0, # From XGB Docu
            }                
        return params    



class XGBModelExact(XGBModel):
    def __init__(self, params):
        '''
        Model specific hyperparameters: n_estimators, eval_metric
        '''
        super().__init__(params)
        
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = None
        # Use rmse for r2 as objective is same
        elif self.params["eval_metric"]=="r2":
            self.params["eval_metric"] = "rmse"
        elif self.params["eval_metric"]=="ams":
            self.params["eval_metric"] = "logloss"
        elif self.params["eval_metric"] in ["gini"]:
            self.params["eval_metric"] = "auc"
        elif self.params["eval_metric"] in ["mauc"]:
            self.params["eval_metric"] = "mlogloss"
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters()               
        
        if self.params["task_type"] == "regression":
            self.model = xgb.XGBRegressor(
                **self.params["hyperparameters"],
                booster = "gbtree",
                tree_method = "exact",
                enable_categorical=True,
                sampling_method="gradient_based",
                n_estimators = self.params["n_estimators"],
                early_stopping_rounds=self.params["patience"],
                device=self.params["device"],
                max_bin = self.params["max_bin"],
                eval_metric = self.params["eval_metric"],
            )
        elif self.params["task_type"] in ["binary", "multiclass"]:
            self.model = xgb.XGBClassifier(
                **self.params["hyperparameters"],
                booster = "gbtree",
                tree_method = "exact", 
                enable_categorical=True,
                sampling_method="gradient_based",
                n_estimators = self.params["n_estimators"],
                early_stopping_rounds=self.params["patience"],
                device=self.params["device"],
                max_bin = self.params["max_bin"],
                eval_metric = self.params["eval_metric"],
            )

class CatBoostModel(BaseModel):
    def __init__(self, params):
        '''
        Model specific hyperparameters: iterations, patience, device
        If multiclass, additionally num_classes has to be given
        '''
        
        super().__init__(params)
        # Not tunable parameters
        if self.params["device"] == "cuda":
            self.params["cb_task_type"] = 'GPU'
        else:
            self.params["cb_task_type"] = None
            
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = None
        elif self.params["eval_metric"] in ["auc", "gini", "mauc"]:
            self.params["eval_metric"] = "AUC"
        elif self.params["eval_metric"]=="rmse":
            self.params["eval_metric"] = "RMSE"
        elif self.params["eval_metric"]=="rmsle":
            self.params["eval_metric"] = "MSLE"
        elif self.params["eval_metric"]=="r2":
            self.params["eval_metric"] = "R2"
        elif self.params["eval_metric"]=="ams":
            self.params["eval_metric"] = "Logloss"
        elif self.params["eval_metric"]=="mlogloss":
            self.params["eval_metric"] = "MultiClass"
        elif self.params["eval_metric"]=="logloss":
            self.params["eval_metric"] = "Logloss"
        elif self.params["eval_metric"]=="mae":
            self.params["eval_metric"] = "MAE"
        elif self.params["eval_metric"]=="norm_gini":
            self.params["eval_metric"] = "NormalizedGini"
        elif self.params["eval_metric"]=="multilogloss":
            self.params["eval_metric"] = "MultiLogloss"

        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters()       
        
        if self.params["task_type"] == "regression":
            self.model = cb.CatBoostRegressor(
                **self.params["hyperparameters"],
                border_count = self.params["border_count"], 
                iterations = self.params["iterations"], 
                eval_metric = self.params["eval_metric"], 
                od_type = "Iter", 
                od_wait = self.params["patience"], 
                task_type = self.params["cb_task_type"],
                verbose=50,
                gpu_ram_part = (0.95/self.params["seeds_parallel"]),
                train_dir='./logs/catboost/'
            )
        elif self.params["task_type"] == "binary":
            self.model = cb.CatBoostClassifier(
                **self.params["hyperparameters"],
                border_count = self.params["border_count"], 
                iterations = self.params["iterations"], 
                eval_metric = self.params["eval_metric"], 
                od_type = "Iter", 
                od_wait = self.params["patience"], 
                task_type = self.params["cb_task_type"],
                verbose=50,
                gpu_ram_part = (0.95/self.params["seeds_parallel"]),
                train_dir='./logs'
            )
        elif self.params["task_type"] == "multiclass":
            self.model = cb.CatBoostClassifier(
                border_count = self.params["border_count"], 
                classes_count=self.params["num_classes"],
                **self.params["hyperparameters"],
                iterations = self.params["iterations"], 
                eval_metric = self.params["eval_metric"], 
                od_type = "Iter", 
                od_wait = self.params["patience"], 
                task_type = self.params["cb_task_type"],
                verbose=50,
                gpu_ram_part = (0.95/self.params["seeds_parallel"]),
                train_dir='./logs'
            )
        elif self.params["task_type"] == "multilabel":
            self.model = cb.CatBoostClassifier(
                border_count = self.params["border_count"], 
                # classes_count=self.params["num_classes"],
                loss_function='MultiLogloss',
                **self.params["hyperparameters"],
                iterations = self.params["iterations"], 
                eval_metric = self.params["eval_metric"], 
                od_type = "Iter", 
                od_wait = self.params["patience"], 
                task_type = self.params["cb_task_type"],
                verbose=50,
                gpu_ram_part = (0.95/self.params["seeds_parallel"]),
                train_dir='./logs'
            )
            
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):
        X_train_use = X_train.copy()
        y_train_use = y_train.copy()
        self.cat_col_names = list(X_train_use.iloc[:,self.params["cat_indices"]].columns)
        self.cat_dtypes = {}
        if eval_set is not None:
            X_val_use = eval_set[0][0].copy()
            y_val_use = eval_set[0][1].copy()
        
            for col in self.cat_col_names:
                X_train_use[col] = X_train_use[col].astype(str).fillna("nan")
                X_val_use[col] = X_val_use[col].astype(str).fillna("nan")      
            eval_set = [(X_val_use,y_val_use)]
        else:
            for col in self.cat_col_names:
                X_train_use[col] = X_train_use[col].astype(str).fillna("nan")
            eval_set = [(X_train_use,y_train_use)]

        h = self.model.fit(
            X_train_use, y_train_use, 
            eval_set=eval_set,
            cat_features=self.cat_col_names,
            use_best_model=True
        )
    
    def predict(self, X):
        X_use = X.copy()
        for col in self.cat_col_names:
            X_use.loc[:,col] = X_use.loc[:,col].astype(str).fillna("nan")

        if self.params["task_type"]=="regression":
            pred = self.model.predict(X_use)
        elif self.params["task_type"]=="binary":
            pred = self.model.predict_proba(X_use)[:,1]            
        elif self.params["task_type"]=="multiclass":
            pred = self.model.predict_proba(X_use)            
        
        return pred    
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, n_features=1, large_dataset=False, **kwargs):
        # Limit max_depth for too large datasets
        if large_dataset:
            max_depth = 8
        else:
            max_depth = 11
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
            "depth": trial.suggest_int("depth", 2, max_depth), # Max depth set to 11 because 12 fails for santander value dataset on A6000
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            # "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
            # "max_leaves": trial.suggest_categorical("max_leaves", [5,10,15,20,25,30,35,40,45,50,55,60]),
            # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5,1),
        }        
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {
            # "learning_rate": 0.08, #0.08
            # "depth": 5, #5
            # "l2_leaf_reg": 5,
            # "bagging_temperature": 1,
            # "leaf_estimation_iterations": 1
        }        
        
        return params    

class CatBoostLossguided(CatBoostModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters()
        
        super().__init__(params)
        

    
    @classmethod
    def get_optuna_hyperparameters(self, trial, n_features=1, large_dataset=False, **kwargs):

        params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [20, 50, 100, 500, 1000, 2000])

        params["max_depth"] = trial.suggest_int("max_depth", 1, 11)
        params["num_leaves"] = trial.suggest_int("num_leaves", 2*params["max_depth"], np.max([2*params["max_depth"], 128]))

        params.update({
            "grow_policy": "lossguide",
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
            "depth": trial.suggest_int("depth", 2, max_depth), # Max depth set to 11 because 12 fails for santander value dataset on A6000
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            # "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
            # "max_leaves": trial.suggest_categorical("max_leaves", [5,10,15,20,25,30,35,40,45,50,55,60]),
            # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5,1),
        })
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {
            "grow_policy": "lossguide",
            # "learning_rate": 0.08, #0.08
            "depth": 16, #5
            "num_leaves": 64
            # "l2_leaf_reg": 5, 
            # "bagging_temperature": 1,
            # "leaf_estimation_iterations": 1
        }        
        
        return params    
        
  
        
class CatBoostModel1024Bins(CatBoostModel):
    def __init__(self, params):
        '''
        Model specific hyperparameters: iterations, patience, device
        If multiclass, additionally num_classes has to be given
        '''
        params["border_count"] = 1024
        super().__init__(params)




class LightGBMModel(BaseModel):
    def __init__(self, params):
        
        super().__init__(params)
        # Not tunable parameters
        # Do not use GPU here, overwrite existing parameter
        self.params["device"] = "cpu"

        self.feval = None
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = None
        elif self.params["eval_metric"] in ["auc", "gini"]:
            self.params["eval_metric"] = "auc"
        elif self.params["eval_metric"]=="r2":
            self.params["eval_metric"] = "rmse"
        elif self.params["eval_metric"] in ["logloss", "ams"]:
            self.params["eval_metric"] = "binary"
        elif self.params["eval_metric"] == "mlogloss":
            self.params["eval_metric"] = "multiclass"

        elif self.params["eval_metric"]=="rmsle":
            def rmsle_lgbm(y_pred, data):
            
                y_true = np.array(data.get_label())
                score = np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))
            
                return 'rmsle', score, False
            self.params["eval_metric"] = "custom"
            self.feval = rmsle_lgbm
        
        
            
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters()
            
    def fit(self, 
            X_train, y_train, 
            eval_set,
            ):
        # X_train_use = X_train.copy()
        # y_train_use = y_train.copy()
        
        
        self.cat_col_names = X_train.iloc[:,self.params["cat_indices"]].columns.tolist()
        self.cat_dtypes = {}
        if eval_set is not None:
            X_val = eval_set[0][0]
            y_val = eval_set[0][1]

            # Todo: generalize this exception for high-cardinality as described in https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html
            if self.params["dataset_name"]=="albert":
                cat_indices = self.params["cat_indices"].copy() 
                
                self.columns_to_remove = []
                for index in self.params["cat_indices"]:
                    if X_train.iloc[:, index].nunique() > 1000:
                        self.columns_to_remove.append(index)
                
                # Remove columns from cat_indices
                X_train.iloc[:,self.columns_to_remove] = X_train.iloc[:,self.columns_to_remove].astype(str).astype(float)        
                X_val.iloc[:,self.columns_to_remove] = X_val.iloc[:,self.columns_to_remove].astype(str).astype(float)        
                self.params["cat_indices"] = [index for index in self.params["cat_indices"] if index not in self.columns_to_remove]
                self.cat_col_names = X_train.iloc[:,self.params["cat_indices"]].columns.tolist()

                # X_train.iloc[:,self.params["cat_indices"]] = X_train.iloc[:,self.params["cat_indices"]].astype(str).astype(float)        
                # X_val.iloc[:,self.params["cat_indices"]] = X_val.iloc[:,self.params["cat_indices"]].astype(str).astype(float)        
                # self.params["cat_indices"] = [] 
                # self.cat_col_names = []
                
            for col in self.cat_col_names:
                if X_train.loc[:,col].dtype!="category":
                    # X_train.loc[:,col] = X_train[col].astype(object)
                    # X_val.loc[:,col] = X_val[col].astype(object)
                    # u_cats = list(X_train[col].unique())+["nan"] #np.unique(list(X_train[col].unique())+list(X_val[col].unique())+["nan"]).tolist()
                    # self.cat_dtypes[col] = pd.CategoricalDtype(categories=u_cats)
                    # X_train.loc[:,col] = X_train.loc[:,col].astype(self.cat_dtypes[col])
                    # X_val.loc[:,col] = X_val.loc[:,col].astype(self.cat_dtypes[col])

                    self.cat_dtypes[col] = pd.CategoricalDtype(categories=list(X_train[col].unique())+["nan"])
                    X_train[col] = X_train[col].astype(self.cat_dtypes[col])
                    X_val[col] = X_val[col].astype(self.cat_dtypes[col])            
                    
            eval_set = [(X_val,y_val)]
        else: 
            for col in self.cat_col_names:
                if X_train.loc[:,col].dtype!="category":
                    self.cat_dtypes[col] = pd.CategoricalDtype(categories=list(X_train[col].unique())+["nan"])
                    X_train[col] = X_train[col].astype(self.cat_dtypes[col])
            eval_set = [(X_train,y_train)]
            
        params = {
            **self.params["hyperparameters"],
            "n_estimators": self.params["n_estimators"],
            "objective": self.params["task_type"],
            "boosting_type": "gbdt",
            "num_class": self.params["d_out"],
            "metric": self.params["eval_metric"],
            "verbosity": -1,
            
        }

        dtrain = lgbm.Dataset(X_train, y_train, categorical_feature=self.cat_col_names)
        if eval_set is not None:
            X_val_use = eval_set[0][0].copy()
            y_val_use = eval_set[0][1].copy()
            dvalid = lgbm.Dataset(X_val_use, y_val_use, reference=dtrain, categorical_feature=self.cat_col_names)
        else:
            dvalid = None

        if self.params["patience"] != None:
            callbacks = [lgbm.early_stopping(stopping_rounds=self.params["patience"])]
        else:
            callbacks = None

        self.model = lgbm.train(params, dtrain, valid_sets=dvalid, callbacks=callbacks, feval=self.feval)
    
    def predict(self, X):
        if self.params["dataset_name"]=="albert":
            X.iloc[:,self.columns_to_remove] = X.iloc[:,self.columns_to_remove].astype(str).astype(float)        
        #     self.params["cat_indices"] = [index for index in self.params["cat_indices"] if index not in columns_to_remove]
        #     self.cat_col_names = X_train.iloc[:,self.params["cat_indices"]].columns.tolist()
        
        for col in self.cat_col_names:
            if X.loc[:,col].dtype!="category":
                X[col] = X[col].astype(self.cat_dtypes[col])
        dpred = lgbm.Dataset(X, categorical_feature=self.params["cat_indices"])

        pred = self.model.predict(X, num_iteration=self.model.best_iteration)         
        
        return pred    
    
    @classmethod
    # copied from catboost (to be adapted)
    def get_optuna_hyperparameters(self, trial, n_features=1, **kwargs):
        # Tuning parameter ranges defined based on the descriptions in https://lightgbm.readthedocs.io/en/latest/Parameters.html and https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#add-more-computational-resources as well as the TabR paper
        
        
        
        # # Limit max_depth for too large datasets
        # if n_features>3000:
        #     max_depth = 9
        # else:
        #     max_depth = 11
        params = {}
        params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [20, 50, 100, 500, 1000, 2000])

        infinite_depth = trial.suggest_categorical("infinite_depth", [True, False])
        if infinite_depth:
            params["max_depth"] = -1
            params["num_leaves"] = trial.suggest_int("num_leaves", 2, 2047)
        else:
            params["max_depth"] = trial.suggest_int("max_depth", 1, 11)
            params["num_leaves"] = trial.suggest_int("num_leaves", 2*params["max_depth"], np.max([2*params["max_depth"],(2**params["max_depth"])-1]))

        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
            # "num_leaves": trial.suggest_int("num_leaves", 2, (2**params["max_depth"])-1), # Max depth set to 11 because 12 fails for santander value dataset on A6000
            # "min_data_in_leaf": trial.suggest_float("lambda_l2", 0.1, 10., log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0001, 10., log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.0001,100.0, log=True),
        })        
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        """ porto seguro expert configuration
        params = {
            "learning_rate": 0.1,
                  "num_leaves": 15,
                  "max_bin": 256,
                  "feature_fraction": 0.6,
                  "verbosity": 1,
                  "drop_rate": 0.1,
                  "is_unbalance": False,
                  "max_drop": 50,
                  "min_child_samples": 10,
                  "min_child_weight": 150,
                  "min_split_gain": 0,
                  "subsample": 0.9,
                  "num_iterations": 10000
        }"""

        # default params of LightGBM module
        params = {
            # "learning_rate": 0.1,  Hyperparameters from expert porto-seguro solution
            # "num_leaves": 31,
            # "max_bin": 255,
            # "feature_fraction": 1.0,
            # "verbosity": 1,
            # "drop_rate": 0.1,
            # "is_unbalance": False,
            # "max_drop": 50,
            # "min_child_samples": 20,
            # "min_child_weight": 1e-3,
            # "min_split_gain": 0,
            # "subsample": 1,
            # "num_iterations": 100
        }
        
        return params  


class LightGBMAutoGluon(LightGBMModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters()

        super().__init__(params)
        
    @classmethod
    # copied from catboost (to be adapted)
    def get_optuna_hyperparameters(self, trial, n_features=1, **kwargs):
        params = {}
        params["max_bin"] = 1024
        params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [20, 50, 100, 500, 1000, 2000])

        infinite_depth = trial.suggest_categorical("infinite_depth", [True, False])
        if infinite_depth:
            params["max_depth"] = -1
            params["num_leaves"] = trial.suggest_int("num_leaves", 2, 2047)
        else:
            params["max_depth"] = trial.suggest_int("max_depth", 1, 11)
            params["num_leaves"] = trial.suggest_int("num_leaves", 2*params["max_depth"], np.max([2*params["max_depth"],(2**params["max_depth"])-1]))

        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0001, 10., log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.0001,100.0, log=True),
        })        
        
        return params

    @classmethod
    def get_default_hyperparameters(self):

        # default params of LightGBM module
        params = {
            # "max_depth": 5, 
            # "max_bin": 255,
            # "feature_fraction": 1.0,
            # "verbosity": 1,
            # "drop_rate": 0.1,
            # "is_unbalance": False,
            # "max_drop": 50,
            # "min_child_samples": 20,
            # "min_child_weight": 1e-3,
            # "min_split_gain": 0,
            # "subsample": 1,
            # "num_iterations": 100
        }
        params["max_bin"] = 1024

        return params  

class LightGBMHolzmueller(LightGBMModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters(task_type=params["task_type"])
        params["n_estimators"] = 1000
        params["max_bin"] = 255
        params["patience"] = 300
        super().__init__(params)
        
    @classmethod
    # NOT CHANGED FROM ORIGINAL METHOD YET
    def get_optuna_hyperparameters(self, trial, n_features=1, **kwargs):
        params = {}
        params["max_bin"] = 1024
        params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [20, 50, 100, 500, 1000, 2000])

        infinite_depth = trial.suggest_categorical("infinite_depth", [True, False])
        if infinite_depth:
            params["max_depth"] = -1
            params["num_leaves"] = trial.suggest_int("num_leaves", 2, 2047)
        else:
            params["max_depth"] = trial.suggest_int("max_depth", 1, 11)
            params["num_leaves"] = trial.suggest_int("num_leaves", 2*params["max_depth"], np.max([2*params["max_depth"],(2**params["max_depth"])-1]))

        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0001, 10., log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.0001,100.0, log=True),
        })        
        
        return params

    @classmethod
    def get_default_hyperparameters(self, task_type="binary"):
        if task_type in ["binary", "multiclass", "classification"]:
            params = {
                "num_leaves": 50, 
                "learning_rate": 0.04,
                "subsample": 0.75,
                "verbosity": 1,
                "colsample_bytree": 1.0,
                "min_data_in_leaf": 40,
                "min_sum_hessian_in_leaf": 1e-7,
                "bagging_fraction": 1.,
            }
        elif task_type in ["regression"]:
            params = {
                "num_leaves": 100, 
                "learning_rate": 0.05,
                "subsample": 0.7,
                "verbosity": 1,
                "colsample_bytree": 1.0,
                "min_data_in_leaf": 3,
                "min_sum_hessian_in_leaf": 1e-7,
                "bagging_fraction": 1.,
            }
        return params  

class LightGBMModelAllCat(LightGBMModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters()
        
        super().__init__(params)
        
    @classmethod
    # copied from catboost (to be adapted)
    def get_optuna_hyperparameters(self, trial, n_features=1, **kwargs):
        params = {}
        params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [20, 50, 100, 500, 1000, 2000])

        infinite_depth = trial.suggest_categorical("infinite_depth", [True, False])
        if infinite_depth:
            params["max_depth"] = -1
            params["num_leaves"] = trial.suggest_int("num_leaves", 2, 2047)
        else:
            params["max_depth"] = trial.suggest_int("max_depth", 1, 11)
            params["num_leaves"] = trial.suggest_int("num_leaves", 2*params["max_depth"], np.max([2*params["max_depth"],(2**params["max_depth"])-1]))

        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0001, 10., log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.0001,100.0, log=True),
        })        
        
        params["max_cat_to_onehot"] = 2

        return params

    @classmethod
    def get_default_hyperparameters(self):

        # default params of LightGBM module
        params = {
            # "max_depth": 5, 
            # "max_bin": 255,
            # "feature_fraction": 1.0,
            # "verbosity": 1,
            # "drop_rate": 0.1,
            # "is_unbalance": False,
            # "max_drop": 50,
            # "min_child_samples": 20,
            # "min_child_weight": 1e-3,
            # "min_split_gain": 0,
            # "subsample": 1,
            # "num_iterations": 100
        }
        params["max_cat_to_onehot"] = 2

        return params  


class LightGBMModel1024Bins(LightGBMModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters()

        super().__init__(params)
        
    @classmethod
    # copied from catboost (to be adapted)
    def get_optuna_hyperparameters(self, trial, n_features=1, **kwargs):
        params = {}
        params["max_bin"] = 1024
        params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [20, 50, 100, 500, 1000, 2000])

        infinite_depth = trial.suggest_categorical("infinite_depth", [True, False])
        if infinite_depth:
            params["max_depth"] = -1
            params["num_leaves"] = trial.suggest_int("num_leaves", 2, 2047)
        else:
            params["max_depth"] = trial.suggest_int("max_depth", 1, 11)
            params["num_leaves"] = trial.suggest_int("num_leaves", 2*params["max_depth"], np.max([2*params["max_depth"],(2**params["max_depth"])-1]))

        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0001, 10., log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.0001,100.0, log=True),
        })        
        
        return params

    @classmethod
    def get_default_hyperparameters(self):

        # default params of LightGBM module
        params = {
            # "max_depth": 5, 
            # "max_bin": 255,
            # "feature_fraction": 1.0,
            # "verbosity": 1,
            # "drop_rate": 0.1,
            # "is_unbalance": False,
            # "max_drop": 50,
            # "min_child_samples": 20,
            # "min_child_weight": 1e-3,
            # "min_split_gain": 0,
            # "subsample": 1,
            # "num_iterations": 100
        }
        params["max_bin"] = 1024

        return params  


class LightGBMModel50000Bins(LightGBMModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters()

        super().__init__(params)
        
    @classmethod
    # copied from catboost (to be adapted)
    def get_optuna_hyperparameters(self, trial, n_features=1, **kwargs):
        params = {}
        params["max_bin"] = 50000
        params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [20, 50, 100, 500, 1000, 2000])

        infinite_depth = trial.suggest_categorical("infinite_depth", [True, False])
        if infinite_depth:
            params["max_depth"] = -1
            params["num_leaves"] = trial.suggest_int("num_leaves", 2, 2047)
        else:
            params["max_depth"] = trial.suggest_int("max_depth", 1, 11)
            params["num_leaves"] = trial.suggest_int("num_leaves", 2*params["max_depth"], np.max([2*params["max_depth"],(2**params["max_depth"])-1]))

        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0001, 10., log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.0001,100.0, log=True),
        })        
        
        return params

    @classmethod
    def get_default_hyperparameters(self):

        # default params of LightGBM module
        params = {
            # "max_depth": 5, 
            # "max_bin": 255,
            # "feature_fraction": 1.0,
            # "verbosity": 1,
            # "drop_rate": 0.1,
            # "is_unbalance": False,
            # "max_drop": 50,
            # "min_child_samples": 20,
            # "min_child_weight": 1e-3,
            # "min_split_gain": 0,
            # "subsample": 1,
            # "num_iterations": 100
        }
        params["max_bin"] = 50000

        return params  

class LightGBMModelExperiment(LightGBMModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters()

        super().__init__(params)
        
    @classmethod
    # copied from catboost (to be adapted)
    def get_optuna_hyperparameters(self, trial, n_features=1, **kwargs):
        params = {}
        params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [1,2,5,8,10])

        params["max_depth"] = -1
        params["num_leaves"] = trial.suggest_int("num_leaves", 2, 16)

        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0001, 10., log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.),
            # "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.0001,100.0, log=True),
            "bagging_freq": 1,
            "min_data_per_group": 1, 
            "cat_l2": 0,
            "cat_smooth": 0,
            "max_cat_to_onehot": 100,

        })        
          
        return params

    @classmethod
    def get_default_hyperparameters(self):

        # default params of LightGBM module
        params = {
            "enable_bundle": False,
            "extra_trees": True,
            "num_leaves": 4, 
            "eta": 0.05,
            "min_data_in_leaf": 1, 
            "feature_fraction": 0.5,
            "bagging_fraction": 0.5,
            "bagging_freq": 1,
            "min_data_per_group": 1, 
            "cat_l2": 0,
            "cat_smooth": 0,
            "max_cat_to_onehot": 100,
            
            # "max_drop": 50,
            # "min_child_samples": 20,
            # "min_child_weight": 1e-3,
            # "min_split_gain": 0,
            # "subsample": 1,
            # "num_iterations": 100
        }
        
        return params  

class LightGBMModel1024BinsHuertasTuneMinLeaf(LightGBMModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters()

        super().__init__(params)
        
    @classmethod
    # copied from catboost (to be adapted)
    def get_optuna_hyperparameters(self, trial, n_features=1, **kwargs):
        params = {}
        params["max_bin"] = 1024
        params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [1,2,5,8,10])

        params["max_depth"] = -1
        params["num_leaves"] = trial.suggest_int("num_leaves", 2, 16)

        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0001, 10., log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.),
            # "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.0001,100.0, log=True),
            "bagging_freq": 1,
            "min_data_per_group": 1, 
            "cat_l2": 0,
            "cat_smooth": 0,
            "max_cat_to_onehot": 100,

        })        
          
        return params

    @classmethod
    def get_default_hyperparameters(self):

        # default params of LightGBM module
        params = {
            "max_bin": 1024,
            "extra_trees": True,
            "num_leaves": 4, 
            "eta": 0.05,
            "min_data_in_leaf": 1, 
            "feature_fraction": 0.5,
            "bagging_fraction": 0.5,
            "bagging_freq": 1,
            "min_data_per_group": 1, 
            "cat_l2": 0,
            "cat_smooth": 0,
            "max_cat_to_onehot": 100,
            
            # "max_drop": 50,
            # "min_child_samples": 20,
            # "min_child_weight": 1e-3,
            # "min_split_gain": 0,
            # "subsample": 1,
            # "num_iterations": 100
        }
        
        return params  

class LightGBMModelHuertasTuneMinLeaf(LightGBMModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters()

        super().__init__(params)
        
    @classmethod
    # copied from catboost (to be adapted)
    def get_optuna_hyperparameters(self, trial, n_features=1, **kwargs):
        params = {}
        params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [1,2,5,8,10])

        params["max_depth"] = -1
        params["num_leaves"] = trial.suggest_int("num_leaves", 2, 16)

        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0001, 10., log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.),
            # "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.0001,100.0, log=True),
            "bagging_freq": 1,
            "min_data_per_group": 1, 
            "cat_l2": 0,
            "cat_smooth": 0,
            "max_cat_to_onehot": 100,

        })        
          
        return params

    @classmethod
    def get_default_hyperparameters(self):

        # default params of LightGBM module
        params = {
            "extra_trees": True,
            "num_leaves": 4, 
            "eta": 0.05,
            "min_data_in_leaf": 1, 
            "feature_fraction": 0.5,
            "bagging_fraction": 0.5,
            "bagging_freq": 1,
            "min_data_per_group": 1, 
            "cat_l2": 0,
            "cat_smooth": 0,
            "max_cat_to_onehot": 100,
            
            # "max_drop": 50,
            # "min_child_samples": 20,
            # "min_child_weight": 1e-3,
            # "min_split_gain": 0,
            # "subsample": 1,
            # "num_iterations": 100
        }
        
        return params  


class LightGBMModelHuertas2(LightGBMModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters()

        super().__init__(params)
        
    @classmethod
    # copied from catboost (to be adapted)
    def get_optuna_hyperparameters(self, trial, n_features=1, **kwargs):
        params = {}
        params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [1,2,5,8,10])

        params["max_depth"] = -1
        params["num_leaves"] = trial.suggest_int("num_leaves", 2, 16)

        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0001, 10., log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.),
            # "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.0001,100.0, log=True),
            "bagging_freq": 1,
            "min_data_per_group": 1, 
            "cat_l2": 0,
            "cat_smooth": 0,
            "max_cat_to_onehot": 100,
            "extra_trees": True,

        })        
          
        return params

    @classmethod
    def get_default_hyperparameters(self):

        # default params of LightGBM module
        params = {
            "extra_trees": True,
            "num_leaves": 4, 
            "eta": 0.05,
            "min_data_in_leaf": 1, 
            "feature_fraction": 0.5,
            "bagging_fraction": 0.5,
            "bagging_freq": 1,
            "min_data_per_group": 1, 
            "cat_l2": 0,
            "cat_smooth": 0,
            "max_cat_to_onehot": 100,
            
            # "max_drop": 50,
            # "min_child_samples": 20,
            # "min_child_weight": 1e-3,
            # "min_split_gain": 0,
            # "subsample": 1,
            # "num_iterations": 100
        }
        
        return params  


class LightGBMModelHuertas(LightGBMModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters()

        super().__init__(params)
        
    @classmethod
    # copied from catboost (to be adapted)
    def get_optuna_hyperparameters(self, trial, n_features=1, **kwargs):
        params = {}
        params["min_data_in_leaf"] = 1 #trial.suggest_categorical("min_data_in_leaf", [1,2,5,8,10])

        params["max_depth"] = -1
        params["num_leaves"] = trial.suggest_int("num_leaves", 2, 16)

        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0001, 10., log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.),
            # "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.0001,100.0, log=True),
            "bagging_freq": 1,
            "min_data_per_group": 1, 
            "cat_l2": 0,
            "cat_smooth": 0,
            "max_cat_to_onehot": 100,

        })        
          
        return params

    @classmethod
    def get_default_hyperparameters(self):

        # default params of LightGBM module
        params = {
            "extra_trees": True,
            "num_leaves": 4, 
            "eta": 0.05,
            "min_data_in_leaf": 1, 
            "feature_fraction": 0.5,
            "bagging_fraction": 0.5,
            "bagging_freq": 1,
            "min_data_per_group": 1, 
            "cat_l2": 0,
            "cat_smooth": 0,
            "max_cat_to_onehot": 100,
            
            # "max_drop": 50,
            # "min_child_samples": 20,
            # "min_child_weight": 1e-3,
            # "min_split_gain": 0,
            # "subsample": 1,
            # "num_iterations": 100
        }
        
        return params  

class LightGBMModelNomindataleaf(LightGBMModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters()

        super().__init__(params)
        
    @classmethod
    # copied from catboost (to be adapted)
    def get_optuna_hyperparameters(self, trial, n_features=1, **kwargs):
        params = {}
        params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [1,2,5,8,10,15,20])

        infinite_depth = trial.suggest_categorical("infinite_depth", [True, False])
        if infinite_depth:
            params["max_depth"] = -1
            params["num_leaves"] = trial.suggest_int("num_leaves", 2, 2047)
        else:
            params["max_depth"] = trial.suggest_int("max_depth", 1, 11)
            params["num_leaves"] = trial.suggest_int("num_leaves", 2*params["max_depth"], np.max([2*params["max_depth"],(2**params["max_depth"])-1]))

        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0001, 10., log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.),
            # "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.0001,100.0, log=True),
        })        
          
        return params

    @classmethod
    def get_default_hyperparameters(self):

        # default params of LightGBM module
        params = {
            "min_data_in_leaf": 1, 
            # "min_sum_hessian_in_leaf": 1e-10
            # "max_depth": 5, 
            # "max_bin": 255,
            # "feature_fraction": 1.0,
            # "verbosity": 1,
            # "drop_rate": 0.1,
            # "is_unbalance": False,
            # "max_drop": 50,
            # "min_child_samples": 20,
            # "min_child_weight": 1e-3,
            # "min_split_gain": 0,
            # "subsample": 1,
            # "num_iterations": 100
        }
        
        return params  


class LightGBMModelDepthLimit(LightGBMModel):
    def __init__(self, params):
        # Tunable hyperparameters
        if "hyperparameters" not in params:
            params["hyperparameters"] = self.get_default_hyperparameters()

        super().__init__(params)
        
    @classmethod
    # copied from catboost (to be adapted)
    def get_optuna_hyperparameters(self, trial, n_features=1, **kwargs):
        params = {}
        params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [20, 50, 100, 500, 1000, 2000])

        params["max_depth"] = trial.suggest_int("max_depth", 1, 11)
        params["num_leaves"] = trial.suggest_int("num_leaves", 2*params["max_depth"], np.max([2*params["max_depth"],(2**params["max_depth"])-1]))

        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0001, 10., log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.0001,100.0, log=True),
        })        
        
        return params

    @classmethod
    def get_default_hyperparameters(self):

        # default params of LightGBM module
        params = {
            "max_depth": 5, 
            # "max_bin": 255,
            # "feature_fraction": 1.0,
            # "verbosity": 1,
            # "drop_rate": 0.1,
            # "is_unbalance": False,
            # "max_drop": 50,
            # "min_child_samples": 20,
            # "min_child_weight": 1e-3,
            # "min_split_gain": 0,
            # "subsample": 1,
            # "num_iterations": 100
        }
        
        return params  

from sklearn.linear_model import LinearRegression, LogisticRegression

class LinearModel(BaseModel):
    def __init__(self, params):
        '''
        Model specific hyperparameters: n_estimators, eval_metric
        '''
        
        super().__init__(params)
        
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = None
        # Use rmse for r2 as objective is same
        elif self.params["eval_metric"]=="r2":
            self.params["eval_metric"] = "rmse"
        elif self.params["eval_metric"]=="ams":
            self.params["eval_metric"] = "logloss"
        elif self.params["eval_metric"]=="gini":
            self.params["eval_metric"] = "auc"
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters()               
        
        if self.params["task_type"] == "regression":
            self.model = LinearRegression(
                # **self.params["hyperparameters"],
                # booster = "gbtree",
                # tree_method = "hist",
                # enable_categorical=True,
                # sampling_method="gradient_based",
                # n_estimators = self.params["n_estimators"],
                # early_stopping_rounds=self.params["patience"],
                # device=self.params["device"],
                # eval_metric = self.params["eval_metric"],
            )
        elif self.params["task_type"] in ["binary", "multiclass"]:
            self.model = LogisticRegression(
                # **self.params["hyperparameters"],
                # booster = "gbtree",
                # tree_method = "hist", 
                # enable_categorical=True,
                # sampling_method="gradient_based",
                # n_estimators = self.params["n_estimators"],
                # early_stopping_rounds=self.params["patience"],
                # device=self.params["device"],
                # eval_metric = self.params["eval_metric"],
            )
        
    def fit(self, 
            X_train, y_train, 
            eval_set,
           ):

        # Todo: Implement categorical data treatment

        # if self.params["dataset_name"]=="bnp-paribas-cardif-claims-management":
        # self.params["cat_indices"] = 
        
        self.cat_col_names = X_train.iloc[:,self.params["cat_indices"]].columns
        self.cat_dtypes = {}
        if eval_set is not None:
            X_val = eval_set[0][0]
            y_val = eval_set[0][1]
        
            for col in self.cat_col_names:
                if X_train.loc[:,col].dtype!="category":
                    # X_train.loc[:,col] = X_train[col].astype(object)
                    # X_val.loc[:,col] = X_val[col].astype(object)
                    # u_cats = list(X_train[col].unique())+["nan"] #np.unique(list(X_train[col].unique())+list(X_val[col].unique())+["nan"]).tolist()
                    # self.cat_dtypes[col] = pd.CategoricalDtype(categories=u_cats)
                    # X_train.loc[:,col] = X_train.loc[:,col].astype(self.cat_dtypes[col])
                    # X_val.loc[:,col] = X_val.loc[:,col].astype(self.cat_dtypes[col])

                    self.cat_dtypes[col] = pd.CategoricalDtype(categories=list(X_train[col].unique())+["nan"])
                    X_train[col] = X_train[col].astype(self.cat_dtypes[col])
                    X_val[col] = X_val[col].astype(self.cat_dtypes[col])            
                    
            eval_set = [(X_val,y_val)]
        else:
            for col in self.cat_col_names:
                if X_train.loc[:,col].dtype!="category":
                    self.cat_dtypes[col] = pd.CategoricalDtype(categories=list(X_train[col].unique())+["nan"])
                    X_train[col] = X_train[col].astype(self.cat_dtypes[col])
            eval_set = [(X_train,y_train)]


        X = pd.concat([X_train,X_val])
        y = pd.concat([y_train,y_val])
        
        h = self.model.fit(X, y)
        
    def predict(self, X):

        if self.params["task_type"]=="regression":
            pred = self.model.predict(X)            
        elif self.params["task_type"]=="binary":
            pred = self.model.predict_proba(X)[:,1]            
        elif self.params["task_type"]=="multiclass":
            pred = self.model.predict_proba(X)         
        
        return pred    
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, dataset_name = "", **kwargs):
        params = {
            # "eta": trial.suggest_float("eta", 1e-3, 0.7, log=True), # From Shwartz et al - reduced min threshold as 1e-3 is already really low and only increases training time unnecessarily
            # "max_depth": trial.suggest_int("max_depth", 1, max_depth, log=False), # Increased depth to allow learning higher-order interactions in one tree 
            # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5,1.0, log=False), # From Shwartz et al
            # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5,1.0, log=False), # From Shwartz et al
            # "subsample": trial.suggest_float("subsample", 0.5,1.0, log=False), # From Shwartz et al
            # "reg_alpha": trial.suggest_float("reg_alpha", 1e-8,1e2, log=True),
            # "reg_lambda": trial.suggest_float("reg_lambda", 1,4, log=True),
            # "gamma": trial.suggest_float("gamma", 1e-8,7, log=True),
            # "min_child_weight": trial.suggest_float("min_child_weight", 1, 1e2, log=True)
        }       
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self):
        params = {
            # "eta": 0.3, # From XGB Docu
            # "max_depth": 6, # From XGB Docu 
            # "colsample_bylevel": 1, # From XGB Docu
            # "colsample_bytree": 1, # From XGB Docu
            # "subsample": 1., # From XGB Docu
            # "min_child_weight": 1, # From XGB Docu
            # "reg_alpha": 0., # From XGB Docu
            # "reg_lambda": 1, # From XGB Docu
            # "gamma": 0, # From XGB Docu
            }        
        
        return params    
    

from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier, RealMLP_TD_Regressor

class RealMLP(BaseModel):
    def __init__(self, params):
        
        super().__init__(params)

        if "eval_metric" not in self.params:
            self.params["eval_metric"] = None
        # Use rmse for r2 as objective is same
        elif self.params["eval_metric"]=="r2":
            self.params["eval_metric"] = "rmse"
        elif self.params["eval_metric"]=="ams":
            self.params["eval_metric"] = "logloss"
        elif self.params["eval_metric"]=="gini":
            self.params["eval_metric"] = "auc"
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters(self.params["task_type"])               
        
        if self.params["task_type"] == "regression":
            self.model = RealMLP_TD_Regressor(
                **self.params["hyperparameters"]
            )
        elif self.params["task_type"] in ["binary", "classification", "multiclass"]:
            self.model = RealMLP_TD_Classifier(
                **self.params["hyperparameters"],
                
            )
        
    def fit(self, 
            X_train, y_train, 
            eval_set,
            ):
        torch.set_float32_matmul_precision('medium') # 'medium' | 'high'

        X_val, y_val = eval_set[0]
        
        # # Fill missing values of continuous columns with mean 
        
        if len(self.params["cat_indices"])!=X_train.shape[1]:
            cont_indices = np.array([i for i in range(X_train.shape[1]) if i not in self.params["cat_indices"]])
            self.cont_col_names = X_train.iloc[:,cont_indices].columns
            self.na_train_means = X_train[self.cont_col_names].mean()
            if X_val.isna().sum().sum()>0:
                X_val[self.cont_col_names] = X_val[self.cont_col_names].fillna(self.na_train_means)
            if X_train.isna().sum().sum()>0:
                X_train[self.cont_col_names] = X_train[self.cont_col_names].fillna(self.na_train_means)
        
        if len(self.params["cat_indices"])>0:
            self.cat_col_names = X_train.iloc[:,self.params["cat_indices"]].columns
        else:
            self.cat_col_names = None
        self.cat_dtypes = {}  

        self.model.fit(X_train, y_train, 
                       X_val=X_val, y_val=y_val, 
                       cat_col_names=self.cat_col_names)
        
    def predict(self, X):
        # # Fill missing values of continuous columns with mean 
        if len(self.params["cat_indices"])!=X.shape[1]:
            if X.isna().sum().sum()>0:
                X[self.cont_col_names] = X[self.cont_col_names].fillna(self.na_train_means)
        
        if self.params["task_type"] =="regression":
            pred = self.model.predict(X)
        elif self.params["task_type"] in ["classification", "multiclass"]:  
            pred = self.model.predict_proba(X)
        elif self.params["task_type"] in ["binary"]:  
            pred = self.model.predict_proba(X)[:,1]
        return pred    
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, dataset_name = "", **kwargs):


        def weighted_categorical(trial, name, options, probabilities):
            """
            Suggest a categorical parameter with weighted probabilities.
            """
            # Ensure probabilities sum to 1
            probabilities = np.array(probabilities) / sum(probabilities)
            # Sample based on weighted probabilities
            chosen_option = np.random.choice(np.array(options,dtype="object"), p=probabilities)
            # Log the choice into the trial
            trial.set_user_attr(name, chosen_option)
            return chosen_option
        
        params = {
            "num_emb_type": trial.suggest_categorical("num_emb_type", [None, "pbld", "pl", "plr"]),
            # "add_front_scale": trial.suggest_categorical("add_front_scale", [True, False]),
            "add_front_scale": weighted_categorical(trial, "add_front_scale", [True, False], [0.6, 0.4]),
            "lr": trial.suggest_loguniform("lr", 2e-2, 3e-1),
            # "p_drop": trial.suggest_categorical("p_drop", [0, 0.15, 0.3]),
            "p_drop": weighted_categorical(trial, "p_drop", [0, 0.15, 0.3], [0.3, 0.5, 0.2]),
            "act": trial.suggest_categorical("act", ["relu", "silu", "mish"]),
            "hidden_sizes": weighted_categorical(trial, "hidden_sizes", [[256, 256, 256], [64, 64, 64, 64, 64], [512]], [0.6, 0.2, 0.2]),
            "wd": trial.suggest_categorical("wd", [0, 2e-2]),
            "weight_init_gain": trial.suggest_loguniform("weight_init_mode", 0.05, 0.5),

        }       

        if kwargs["task_type"] != "regression":
            params["ls_eps"] = weighted_categorical(trial, "ls_eps", [0, 0.1], [0.3, 0.7])
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self, task_type):
        if task_type == "regression":
            params = dict(
                hidden_sizes=[256] * 3,
                max_one_hot_cat_size=9, embedding_size=8,
                weight_param='ntk', weight_init_mode='std',
                bias_init_mode='he+5', bias_lr_factor=0.1,
                act='mish', use_parametric_act=True, act_lr_factor=0.1,
                wd=2e-2, wd_sched='flat_cos', bias_wd_factor=0.0,
                block_str='w-b-a-d', p_drop=0.15, p_drop_sched='flat_cos',
                add_front_scale=True, scale_lr_factor=6.0,
                tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
                num_emb_type='pbld', plr_sigma=0.1, plr_hidden_1=16, plr_hidden_2=4, plr_lr_factor=0.1,
                clamp_output=True, normalize_output=True,
                lr=0.2,
                n_epochs=256, lr_sched='coslog4', opt='adam', sq_mom=0.95
            )
        elif task_type in ["binary", "classification", "multiclass"]:
            params = dict(
                hidden_sizes=[256] * 3,
                max_one_hot_cat_size=9, embedding_size=8,
                weight_param='ntk', bias_lr_factor=0.1,
                act='selu', use_parametric_act=True, act_lr_factor=0.1,
                block_str='w-b-a-d', p_drop=0.15, p_drop_sched='flat_cos',
                add_front_scale=True,
                scale_lr_factor=6.0,
                bias_init_mode='he+5', weight_init_mode='std',
                wd=2e-2, wd_sched='flat_cos', bias_wd_factor=0.0,
                use_ls=True, ls_eps=0.1,
                num_emb_type='pbld', plr_sigma=0.1, plr_hidden_1=16, plr_hidden_2=4, plr_lr_factor=0.1,
                lr=4e-2,
                tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
                n_epochs=256, lr_sched='coslog4', opt='adam', sq_mom=0.95
            )
        
        return params    


class RealMLP2(BaseModel):
    def __init__(self, params):
        
        super().__init__(params)

        if "eval_metric" not in self.params:
            self.params["eval_metric"] = None
        # Use rmse for r2 as objective is same
        elif self.params["eval_metric"]=="r2":
            self.params["eval_metric"] = "rmse"
        elif self.params["eval_metric"]=="ams":
            self.params["eval_metric"] = "logloss"
        elif self.params["eval_metric"]=="gini":
            self.params["eval_metric"] = "auc"
        
        # Tunable hyperparameters
        if "hyperparameters" not in self.params:
            self.params["hyperparameters"] = self.get_default_hyperparameters(self.params["task_type"])               
        
        if self.params["task_type"] == "regression":
            self.model = RealMLP_TD_Regressor(
                **self.params["hyperparameters"]
            )
        elif self.params["task_type"] in ["binary", "classification", "multiclass"]:
            self.model = RealMLP_TD_Classifier(
                **self.params["hyperparameters"],
                
            )
        
    def fit(self, 
            X_train, y_train, 
            eval_set,
            ):
        torch.set_float32_matmul_precision('medium') # 'medium' | 'high'
        
        if len(self.params["cat_indices"])>0:
            self.cat_col_names = X_train.iloc[:,self.params["cat_indices"]].columns
        else:
            self.cat_col_names = None
        self.cat_dtypes = {}

        self.model.fit(X_train, y_train, 
                       X_val=eval_set[0][0], y_val=eval_set[0][1], 
                       cat_col_names=self.cat_col_names)
        
    def predict(self, X):
        if self.params["task_type"] =="regression":
            pred = self.model.predict(X)
        elif self.params["task_type"] in ["classification", "multiclass"]:  
            pred = self.model.predict_proba(X)
        elif self.params["task_type"] in ["binary"]:  
            pred = self.model.predict_proba(X)[:,1]
        return pred    
    
    @classmethod
    def get_optuna_hyperparameters(self, trial, dataset_name = "", **kwargs):


        def weighted_categorical(trial, name, options, probabilities):
            """
            Suggest a categorical parameter with weighted probabilities.
            """
            # Ensure probabilities sum to 1
            probabilities = np.array(probabilities) / sum(probabilities)
            # Sample based on weighted probabilities
            chosen_option = np.random.choice(np.array(options,dtype="object"), p=probabilities)
            # Log the choice into the trial
            trial.set_user_attr(name, chosen_option)
            return chosen_option
        
        params = {
            "num_emb_type": trial.suggest_categorical("num_emb_type", [None, "pbld", "pl", "plr"]),
            # "add_front_scale": trial.suggest_categorical("add_front_scale", [True, False]),
            "add_front_scale": weighted_categorical(trial, "add_front_scale", [True, False], [0.6, 0.4]),
            "lr": trial.suggest_loguniform("lr", 2e-2, 3e-1),
            # "p_drop": trial.suggest_categorical("p_drop", [0, 0.15, 0.3]),
            "p_drop": weighted_categorical(trial, "p_drop", [0, 0.15, 0.3], [0.3, 0.5, 0.2]),
            "act": trial.suggest_categorical("act", ["relu", "silu", "mish"]),
            "hidden_sizes": weighted_categorical(trial, "hidden_sizes", [[256, 256, 256], [64, 64, 64, 64, 64], [512]], [0.6, 0.2, 0.2]),
            "wd": trial.suggest_categorical("wd", [0, 2e-2]),
            "weight_init_gain": trial.suggest_loguniform("weight_init_mode", 0.05, 0.5),

        }       

        if kwargs["task_type"] != "regression":
            params["ls_eps"] = weighted_categorical(trial, "ls_eps", [0, 0.1], [0.3, 0.7])
        
        return params
    
    @classmethod
    def get_default_hyperparameters(self, task_type):
        if task_type == "regression":
            params = {}
        elif task_type in ["binary", "classification", "multiclass"]:
            params = {}
        
        return params    
