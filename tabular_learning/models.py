import gc

import os
import joblib
import optuna

from torchmetrics import AUROC, R2Score
from torchmetrics.classification import Accuracy, BinaryAccuracy

from sklearn.metrics import r2_score, roc_auc_score

import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from rtdl_revisiting_models import MLP, ResNet, FTTransformer, CategoricalEmbeddings
import rtdl_num_embeddings
from rtdl_num_embeddings import PeriodicEmbeddings, LinearReLUEmbeddings, LinearEmbeddings

# import delu
import time

import catboost as cb
from torchcontrib.optim import SWA
# import lightgbm as lgbm

import sklearn
from copy import deepcopy
# import category_encoders as ce
import math 

import pickle
import zipfile

from tabular_learning.tabm_reference import Model, make_parameter_groups
from torch import Tensor

import warnings
warnings.filterwarnings("ignore") 

def get_model(model_name):

    if model_name=="CatBoost":
        return CatBoostModel
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
                train_dir=self.params["save_path"],
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
                train_dir=self.params["save_path"]
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
                train_dir=self.params["save_path"]
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
                train_dir=self.params["save_path"]
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

