import os
import yaml

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
warnings.filterwarnings("ignore", message="Failed to load image Python extension:*")
import logging

import random
import numpy as np
import pickle
import time

import pandas as pd
import numpy as np

# from models import get_model
# from utils import set_seed, get_metric
from tabular_learning.datasets import TabularDataset

import optuna
import joblib
import ray
import gc

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error, log_loss, mean_absolute_error, f1_score,accuracy_score,balanced_accuracy_score, matthews_corrcoef, root_mean_squared_error, mean_absolute_percentage_error


def set_seed(seed=42):
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def get_config(model_name):
    base_dir = "tabular_learning/configs/"
    if model_name in ["XGBoost", "XGBModel1024Bins", "XGBModelExact", "XGBModelDepth20", "XGBModelDepth1", "XGBModelDepth2", "XGBModelDepth3", "XGBModelDepth4", "XGBModelDepth5", "XGBoostLossguided", "XGBoostHolzmueller", "XGBoostSmallData"]:
        with open(base_dir+'xgb_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        configs["model"]["model_name"] = model_name
    if model_name == "MLPContLinear":
        with open(base_dir+'mlp_contlinear_config.yaml', 'r') as file: 
            configs = yaml.safe_load(file)
    if model_name == "MLPContReLU":
        with open(base_dir+'mlp_contrelu_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
    if model_name in ["MLP", "MLPModelMish", "MLPModelLongTrain"]:
        with open(base_dir+'mlp_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        configs["model"]["model_name"] = model_name
    if model_name == "ResNet":
        with open(base_dir+'resnet_config.yaml', 'r') as file: 
            configs = yaml.safe_load(file)
    if model_name in ["CatBoost", "CatBoostModel1024Bins", "CatBoostLossguided"]:
        with open(base_dir+'catboost_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        configs["model"]["model_name"] = model_name
    if model_name in ["MLP-PLR", "MLPPLRHighEmbedding", "MLPPLRFixedArchitecture", "MLPPLRFixedArchitectureTuneSeed", "MLPPLRFeatureDropout", "MLPPLRStopInterpol", "MLP-PLR-minmax", "MLP-PLR-notransform"]:
        with open(base_dir+'mlpplr_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        configs["model"]["model_name"] = model_name
        if model_name == "MLP-PLR-minmax":# , 
            configs["dataset"]["num_scaler"] = "minmax" 
        elif model_name == "MLP-PLR-notransform":
            configs["dataset"]["num_scaler"] = None 
    if model_name == "TabMStopInterpol":
        with open(base_dir+'tabm_stopinterpol_config.yaml', 'r') as file:
            configs = yaml.safe_load(file) 
    if model_name == "MLPStopInterpol":
        with open(base_dir+'mlp_stopinterpol_config.yaml', 'r') as file:
            configs = yaml.safe_load(file) 
    if model_name == "FTTransformer":
        with open(base_dir+'fttransformer_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
    if model_name in ["LightGBM", "LightGBMModelDepthLimit", "LightGBMModel1024Bins", "LightGBMModelNomindataleaf", "LightGBMHolzmueller", "LightGBMModelHuertas", "LightGBMModelHuertasTuneMinLeaf", "LightGBMModelExperiment", "LightGBMModel1024BinsHuertasTuneMinLeaf", "LightGBMModel50000Bins", "LightGBMModelAllCat", "LightGBMModelHuertas2"]:
        with open(base_dir+'lightgbm_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        configs["model"]["model_name"] = model_name
    if model_name == "TabM":
        with open(base_dir+'tabm_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
    if model_name == "TabMmini":
        with open(base_dir+'tabm-mini_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
    if model_name == "RealMLP":
        with open(base_dir+'realmlp_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
    if model_name == "TabPFNv2":
        with open(base_dir+'tabpfnv2_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)

    return configs 

def get_metric(eval_metric_name):
    if eval_metric_name=="r2":
        return r2_score, "maximize"
    elif eval_metric_name=="auc":
        return roc_auc_score, "maximize"
    elif eval_metric_name=="mauc":
        return lambda y_true, y_pred: roc_auc_score(y_true, y_pred, multi_class='ovo', labels=list(range(y_pred.shape[1]))), "maximize"
    elif eval_metric_name=="mae":
        return lambda y_true, y_pred, **kwargs: mean_absolute_error(y_true, y_pred), "minimize"
    elif eval_metric_name=="mse":
        return lambda y_true, y_pred, **kwargs: mean_squared_error(y_true, y_pred), "minimize"
    elif eval_metric_name=="rmse":
        return lambda y_true, y_pred, **kwargs: np.sqrt(mean_squared_error(y_true, y_pred)), "minimize"
    elif eval_metric_name=="rmsle":
        return lambda y_true, y_pred, **kwargs: np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), "minimize"
    elif eval_metric_name=="rmsse":
        return r2_score, "maximize"
    elif eval_metric_name=="gini":
        return lambda y_true, y_pred: (2*roc_auc_score(y_true, y_pred))-1, "maximize"
    elif eval_metric_name=="logloss":
        return log_loss, "minimize"
    elif eval_metric_name=="mlogloss":
        return lambda y_true, y_pred: log_loss(y_true, y_pred, labels=list(range(y_pred.shape[1]))), "minimize"
    elif eval_metric_name=="mae":
        return mean_absolute_error, "minimize"
    elif eval_metric_name=="norm_gini":
        return normalized_gini, "maximize"
    elif eval_metric_name=="multilabel":
        return multilabel_log_loss, "minimize"
    elif eval_metric_name=="Accuracy":
        return lambda y_true, y_pred: accuracy_score(y_true, np.round(y_pred)), "maximize"
    elif eval_metric_name=="mAccuracy":
        return lambda y_true, y_pred: accuracy_score(y_true, np.argmax(y_pred,axis=1)), "maximize"
    else:
        raise ValueError(f"Metric '{eval_metric_name}' not implemented.")    

def get_results(configs):
    os.environ["CUDA_VISIBLE_DEVICES"] = configs["model"]["gpus"]
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # warnings.filterwarnings("ignore")
    # warnings.filterwarnings("ignore")

    if configs["model"]["model_name"] != "TabPFNv2": 
        from tabular_learning.models import get_model


    exp_name = configs["exp_name"]
    seed = configs["seed"]
    model_name = configs["model"]["model_name"]
    preprocess_type = configs["dataset"]["preprocess_type"]
    n_trials = configs["hpo"]["n_trials"]

    if configs["dataset"]["fe_type"] is not None and configs["dataset"]["fe_order"] not in [None, -1]:   
        fe_type = str(configs["dataset"]["fe_order"]) + "-" + configs["dataset"]["fe_type"]
    else:
        fe_type = configs["dataset"]["fe_type"]

    set_seed(seed)

    if not "eval_metric_name" in configs["dataset"]:
        configs["dataset"]["eval_metric_name"] = None
    
    dataset = TabularDataset(configs["dataset"]["dataset_name"], 
                             val_strategy=configs["dataset"]["val_strategy"], 
                             eval_metric_name=configs["dataset"]["eval_metric_name"]) 

    if configs["hpo"]["n_trials"] > 0:
        exp_name += f'_tuned_{configs["hpo"]["n_trials"]}trials'

        if configs["hpo"]["ensemble_best_trials"] is not None: # Todo: Suboptimal, as ensemble = False does not suffice to deactivate ensembling
            configs["hpo"]["ensemble"] = True
        else:
            configs["hpo"]["ensemble"] = False
            
    else:
        configs["hpo"]["ensemble"] = False
    
    if not os.path.exists(f'results/{dataset.dataset_name}/{fe_type}_{preprocess_type}/{model_name}/{exp_name}/{exp_name}_seed{seed}.pickle'):
        if not os.path.exists(f'results/{dataset.dataset_name}/{fe_type}_{preprocess_type}/{model_name}/{exp_name}/'):
            os.makedirs(f'results/{dataset.dataset_name}/{fe_type}_{preprocess_type}/{model_name}/{exp_name}/')

        results = {}
        
        print(f"Train model {model_name}")
        results["performance"] = {}
        results["performance"]["Train"] = {}
        results["performance"]["Val"] = {}
        results["performance"]["Test"] = {}
        results["predictions"] = {}
        results["times"] = {}
        if configs["hpo"]["ensemble_best_trials"] == "auto" or type(configs["hpo"]["ensemble_best_trials"])==int:
            results["performance_ens"] = {}
            results["performance_ens"]["Train"] = {}
            results["performance_ens"]["Val"] = {}
            results["performance_ens"]["Test"] = {}
            results["predictions_ens"] = {}

        configs["model"]["save_path"] = f'results/{dataset.dataset_name}/{fe_type}_{preprocess_type}/{model_name}/{exp_name}/'
        configs["model"]["exp_name"] = configs["exp_name"]
        configs["model"]["seed"] = configs["seed"]

        if model_name=="AutoGluon":
            ##### NOT IMPLEMENTED !!!
            configs["hpo"]["ensemble"] = False

            # if model_name =="TabPFNv2":
            #     from models_tabpfn import TabPFNv2 as model_class
            # else:
            from tabular_learning.models import get_model 
            model_class = get_model(model_name)
            
            model = model_class(configs["model"])
            model.fit(pd.concat([dataset.X_train,dataset.X_val]),pd.concat([dataset.y_train,dataset.y_val]))

            y_train_pred = model.predict(dataset.X_train)
            y_val_pred = model.predict(dataset.X_val)
            y_test_pred = model.predict(dataset.X_test)

            # Apply dataset-specific postprocessing
            if "minimalistic" in dataset.preprocess_states:
                y_train_eval = dataset.minimalistic_postprocessing(dataset.X_train, dataset.y_train)
                y_val_eval = dataset.minimalistic_postprocessing(dataset.X_val, dataset.y_val)
                y_test_eval = dataset.minimalistic_postprocessing(dataset.X_test, dataset.y_test)
                
                y_train_pred = dataset.minimalistic_postprocessing(dataset.X_train, y_train_pred)
                y_val_pred = dataset.minimalistic_postprocessing(dataset.X_val, y_val_pred)
                y_test_pred = dataset.minimalistic_postprocessing(dataset.X_test, y_test_pred)
            else:
                y_train_eval = dataset.y_train.copy()
                y_val_eval = dataset.y_val.copy()
                y_test_eval = dataset.y_test.copy()
            if "expert" in dataset.preprocess_states:
                y_train_eval = dataset.expert_postprocessing(dataset.X_train, y_train_eval)
                y_val_eval = dataset.expert_postprocessing(dataset.X_val, y_val_eval)
                y_test_eval = dataset.expert_postprocessing(dataset.X_test, y_test_eval)
                
                y_train_pred = dataset.expert_postprocessing(dataset.X_train, y_train_pred)
                y_val_pred = dataset.expert_postprocessing(dataset.X_val, y_val_pred)
                y_test_pred = dataset.expert_postprocessing(dataset.X_test, y_test_pred)
            
            results["performance"]["Train"] = dataset.eval_metric(y_train_eval,y_train_pred)
            results["performance"]["Val"] = model.model.leaderboard()["score_val"][0]
            results["performance"]["Test"] = dataset.eval_metric(y_test_eval,y_test_pred)
            results["predictions"] = [y_train_pred, y_val_pred, y_test_pred]
            results["model_specific_outputs"] = {"leaderboard": model.model.leaderboard()}

        else:
            repeat_seeds = [configs["seed"]*i for i in range(1, len(dataset.split_indices)+1)]
            
            if configs["model"]["device"] in ["gpu", "cuda"]:
                parallel_tasks = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
                    
                print(f"Use {parallel_tasks} GPUs and parallelize {configs['model']['seeds_parallel']} seeds on each GPU")
                ray.init(
                    runtime_env={"working_dir": ".", 
                                       "excludes": [
                                           "AutogluonModels/",
                                            "cfg/",
                                            "data/",
                                           "enable_estimates/",
                                           "logs/",
                                           "results/",
                                           "tabicl/",
                                           "utils/",
                                           "test.ipynb",
                                           "*.sh",
                                           "*.ipynb",
                                        ],
                                     },
                    num_cpus=(configs["model"]["seeds_parallel"]*parallel_tasks), # Each parallel seed uses own CPU-core 
                    num_gpus=parallel_tasks # Use all available GPUs as previously specified
                ) 
    
                run_seed_parallel = run_seed.options(num_cpus=1, # Each GPU uses one CPU
                                                     num_gpus=1/configs["model"]["seeds_parallel"]) # Each GPU trains seeds_parallel seeds
            else:
                ray.init(runtime_env={"working_dir": ".", 
                                       "excludes": [
                                           "AutogluonModels/",
                                            "cfg/",
                                            "data/",
                                           "enable_estimates/",
                                           "logs/",
                                           "results/",
                                           "tabicl/",
                                           "utils/",
                                           "test.ipynb",
                                           "*.sh",
                                           "*.ipynb",
                                        ],
                                     })
                parallel_tasks = 0
                run_seed_parallel = run_seed.options(num_cpus=np.trunc(configs["model"]["num_threads"]/configs["model"]["seeds_parallel"]), # Each seed uses X CPUs
                                                     num_gpus=0) # Each GPU trains seeds_parallel seeds


            result_by_trial = [run_seed_parallel.remote(
                                               dataset=dataset,
                                               num_seed=num_seed, 
                                               seed=seed, 
                                               seed_configs=configs) for num_seed, seed in enumerate(repeat_seeds)]
            
            result_by_trial = ray.get(result_by_trial)
            
            for num_seed, result_by_seed in enumerate(result_by_trial):
                results[f"seed_{num_seed}"] = result_by_seed
                results["performance"]["Train"][f"seed_{num_seed}"] = result_by_seed["performance"]["Train"]
                results["performance"]["Val"][f"seed_{num_seed}"] = result_by_seed["performance"]["Val"]
                results["performance"]["Test"][f"seed_{num_seed}"] = result_by_seed["performance"]["Test"]
                results["predictions"][f"seed_{num_seed}"] = result_by_seed["predictions"]
                results["times"][f"seed_{num_seed}"] = result_by_seed["times"]
                if configs["hpo"]["ensemble"]:
                    results["performance_ens"]["Train"][f"seed_{num_seed}"] = result_by_seed["performance_ens"]["Train"]
                    results["performance_ens"]["Val"][f"seed_{num_seed}"] = result_by_seed["performance_ens"]["Val"]
                    results["performance_ens"]["Test"][f"seed_{num_seed}"] = result_by_seed["performance_ens"]["Test"]
                    results["predictions_ens"][f"seed_{num_seed}"] = result_by_seed["predictions_ens"]

            ray.shutdown()
            
        with open(configs["model"]["save_path"]+f'{exp_name}_seed{seed}.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        
    else:
        print(f'Results at "results/{dataset.dataset_name}/{fe_type}_{preprocess_type}/{model_name}/{exp_name}/{exp_name}_seed{seed}.pickle" already exist and are loaded')
        with open(f'results/{dataset.dataset_name}/{fe_type}_{preprocess_type}/{model_name}/{exp_name}/{exp_name}_seed{seed}.pickle', 'rb') as handle:
            results = pickle.load(handle)

    return results


@ray.remote(num_cpus=1, num_gpus=1) 
def run_seed(dataset,
             num_seed, 
             seed,
             seed_configs):

    print(f"Start Training for seed iteration {num_seed}")

#######################################
    train_index, val_index, test_index = dataset.split_indices[num_seed]
    
    X_train = dataset.X.loc[train_index]
    X_val = dataset.X.loc[val_index]
    X_test = dataset.X.loc[test_index]
    
    y_train = dataset.y.loc[train_index]
    y_val = dataset.y.loc[val_index]
    y_test = dataset.y.loc[test_index]



    cat_indices = pd.Series(dataset.cat_indices).copy().values.tolist()
    
    if seed_configs["dataset"]["fe_type"] is not None:
       X_train, X_val, X_test, y_train, y_val, y_test, cat_indices = dataset.FE_operation(
            X_train, X_val,  X_test, y_train, y_val,  y_test, cat_indices, seed_configs["dataset"]["fe_type"], seed_configs["dataset"]["fe_order"], seed=seed)    
   
    if seed_configs["dataset"]["preprocess_type"]=="minimalistic":
        X_train, X_val, X_test, y_train, y_val, y_test, cat_indices = dataset.minimalistic_preprocessing(
            X_train, X_val,  X_test, y_train, y_val,  y_test, cat_indices)

    # elif preprocess_type=="openfe":
    #     dataset.load_data()
    #     dataset.minimalistic_preprocessing(dataset.X_train, dataset.X_val,  dataset.X_test, 
                                           # dataset.y_train, dataset.y_val,  dataset.y_test)
    #     dataset.openfe_preprocessing(dataset.X_train, dataset.X_val,  dataset.X_test, 
                        #                    dataset.y_train, dataset.y_val,  dataset.y_test, 
                        # overwrite_existing=seed_configs["dataset"]["overwrite_existing"])
    
    else:
        print(f"No preprocessing applied (either because none is selected or because preprocess_type={preprocess_type} is not implemented)")

    # Apply model-specific preprocessing
    if seed_configs["model"]["model_name"] in ["MLP", "ResNet", "ResNetPLR", "FTTransformer", "MLP-PLR", "Trompt", "MLPContLinear", "MLPContReLU", "MLPContReLUTarget", "MLPContDoubleReLUTarget", "MLPContWeightedSum", "TabM", "TabMmini", "MLPPLRFixedArchitecture", "MLPPLRFixedArchitectureTuneSeed", "MLPPLRFeatureDropout", "MLPPLRStopInterpol", "TabMStopInterpol", "MLPStopInterpol", "MLPPLRHighEmbedding", "MLP-PLR-minmax", "MLP-PLR-notransform", "MLPModelMish", "MLPModelLongTrain"]:  
        if "num_scaler" not in seed_configs["dataset"]:
            seed_configs["dataset"]["num_scaler"] = "quantile"
    
        X_train, X_val, X_test, y_train, y_val, y_test, cat_indices = dataset.neuralnet_preprocessing(
            X_train, X_val,  X_test, y_train, y_val,  y_test, cat_indices, num_scaler = seed_configs["dataset"]["num_scaler"])
        
        cat_cardinalities = (np.array([X_train.iloc[:,cat_indices].max(),
                                       X_val.iloc[:,cat_indices].max(),
                                       X_test.iloc[:,cat_indices].max()]).max(axis=0)+1).tolist()
    else:
        cat_cardinalities = list(X_train.iloc[:,cat_indices].nunique())
        
    print(f"{dataset.dataset_name} - Train has {X_train.shape[0]} samples and {X_train.shape[1]} features of which {len(cat_indices)} are categorical")
    # Update dataset-specific parameters
    seed_configs["model"].update({
        # Dataset-specific Parameters
        "dataset_name": dataset.dataset_name,
        "task_type": dataset.task_type,
        "cont_indices": [i for i in range(X_train.shape[1]) if i not in cat_indices],
        "cat_indices": cat_indices,
        "cat_cardinalities": cat_cardinalities,
        "d_out": 1 if dataset.task_type in ["regression", "binary"] else dataset.num_classes,
        "sample_size": X_train.shape[0],
        "large_dataset": dataset.large_dataset,
        "eval_metric": dataset.eval_metric_name
    })   
    if dataset.task_type in ["multiclass", "classification"]:
        seed_configs["model"].update({
            "num_classes": dataset.num_classes
        })   

#######################################

    
    exp_name = seed_configs["exp_name"]+f"_{num_seed}"
    seed_configs["seed"] = seed
    
    set_seed(seed_configs["seed"])
    
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore")
    
    res = {}
    res["performance"] = {}
    res["times"] = {}
    
    seed_configs["model"]["save_path"] += f"/seed_{num_seed}"
    if not os.path.exists(seed_configs["model"]["save_path"]):
        os.makedirs(seed_configs["model"]["save_path"])

    n_trials = seed_configs["hpo"]["n_trials"]
    
    if n_trials > 0:
        start = time.time()
        print(f"Run HPO for {n_trials} trials")

        study = tune_hyperparameters(
            dataset,
            X_train.copy(), X_val.copy(), X_test.copy(), 
            y_train.copy(), y_val.copy(), y_test.copy(),
            seed_configs,
            num_seed
        )                    
        
        seed_configs["model"]["hyperparameters"] = study.best_params
        
        end = time.time()
        res["times"]["mean_trial_time"] = (end-start)/60/n_trials

        print(f'Mean time per trial: {res["times"]["mean_trial_time"]}')

        X_train_seed = X_train.copy()
        X_val_seed = X_val.copy()
        X_test_seed = X_test.copy()
        
        y_train_pred_seed, y_val_pred_seed, y_test_pred_seed = study.best_trial.user_attrs["predictions"]
        
        if "neuralnet" in dataset.preprocess_states:
            y_train_seed = dataset.neuralnet_postprocessing(X_train_seed, y_train)
            y_val_seed = dataset.neuralnet_postprocessing(X_val_seed, y_val)
            y_test_seed = dataset.neuralnet_postprocessing(X_test_seed, y_test)
        else:
            y_train_seed = y_train.copy()
            y_val_seed = y_val.copy()
            y_test_seed = y_test.copy()
        if "minimalistic" in dataset.preprocess_states:
            y_train_seed = dataset.minimalistic_postprocessing(X_train_seed, y_train_seed)
            y_val_seed = dataset.minimalistic_postprocessing(X_val_seed, y_val_seed)
            y_test_seed = dataset.minimalistic_postprocessing(X_test_seed, y_test_seed)
        if "expert" in dataset.preprocess_states:
            y_train_seed = dataset.expert_postprocessing(X_train_seed, y_train_seed)
            y_val_seed = dataset.expert_postprocessing(X_val_seed, y_val_seed)
            y_test_seed = dataset.expert_postprocessing(X_test_seed, y_test_seed)
         
        if seed_configs["hpo"]["ensemble"]:
            if dataset.valid_type in ["5CV", "10CV"]:
                y_val_seed_copy = y_val_seed.copy()
                y_val_seed = pd.concat([y_train_seed, y_val_seed_copy], axis=0)
                y_train_seed = pd.concat([y_train_seed, y_val_seed_copy], axis=0)
            if seed_configs["hpo"]["ensemble_best_trials"] == "auto" and n_trials>=3: 
                # Obtain best val performances and predictions from trials
                trials = study.get_trials()
                trial_performances = [trials[i].values for i in range(n_trials)]
                if dataset.eval_metric_direction=="maximize": 
                    trial_performances = np.array([i[0] if i is not None else -np.inf  for i in trial_performances])
                if dataset.eval_metric_direction=="minimize": 
                    trial_performances = np.array([i[0] if i is not None else np.inf  for i in trial_performances])
                
                best_trial_performances_idx = np.argsort(trial_performances)
                if dataset.eval_metric_direction=="maximize": 
                    best_trial_performances_idx = np.argsort(trial_performances)[::-1]
                
                val_trial_predictions = np.array([trials[i].user_attrs["predictions"][1] for i in range(n_trials)])
    
                curr_best_trial_perf = np.round(trial_performances[best_trial_performances_idx[0]],4)
                print(f"Best performance prior ensembling: {curr_best_trial_perf}")

                # Limit max. no. of trials for ensembling to 10 as deploying too many models would be impractical in real applications
                max_trials = np.min([11,n_trials])
                
                # Get ensemble performances
                hpo_ensembles = [dataset.eval_metric(y_val_seed,val_trial_predictions[best_trial_performances_idx[:used_trials]].mean(axis=0)) for used_trials in range(2,max_trials)]
                
                if dataset.eval_metric_direction=="maximize": 
                    best_ensemble = np.argmax(hpo_ensembles)
                else:
                    best_ensemble = np.argmin(hpo_ensembles)
                
                y_val_pred_seed_ens = val_trial_predictions[best_trial_performances_idx[:2+best_ensemble]].mean(axis=0)
                    
                ensemble_perf = dataset.eval_metric(y_val_seed,y_val_pred_seed_ens)
                
                if dataset.eval_metric_direction=="maximize": 
                    condition = np.round(ensemble_perf,4)>curr_best_trial_perf
                else:
                    condition = np.round(ensemble_perf,4)<curr_best_trial_perf
                
                if condition:
                
                    for used_trials in range(2,2+best_ensemble+1):
                        y_val_pred_seed_ens = val_trial_predictions[best_trial_performances_idx[:2+best_ensemble]].mean(axis=0)
                        
                        ensemble_perf = dataset.eval_metric(y_val_seed,y_val_pred_seed_ens)
                        
                        if used_trials==2+best_ensemble:
                            print(f"Final Ensemble using top {used_trials} HP settings: {ensemble_perf}")
                        else:
                            print(f"Ensemble using top {used_trials} HP settings: {ensemble_perf}")
                    
                    y_train_pred_seed_ens = np.array([trials[i].user_attrs["predictions"][0] for i in range(n_trials)])[best_trial_performances_idx[:2+best_ensemble]].mean(axis=0)
                    y_val_pred_seed_ens = np.array([trials[i].user_attrs["predictions"][1] for i in range(n_trials)])[best_trial_performances_idx[:2+best_ensemble]].mean(axis=0)
                    y_test_pred_seed_ens = np.array([trials[i].user_attrs["predictions"][2] for i in range(n_trials)])[best_trial_performances_idx[:2+best_ensemble]].mean(axis=0)
                else:
                    print(f"Using top HP settings does not improve the ensemble")
                    y_train_pred_seed_ens, y_val_pred_seed_ens, y_test_pred_seed_ens = study.best_trial.user_attrs["predictions"]
                    
                print("--------------------")            
                
            elif type(seed_configs["hpo"]["ensemble_best_trials"])==2: 
                trials = study.get_trials()

                trial_performances = [trials[i].values for i in range(n_trials)]
                if dataset.eval_metric_direction=="maximize": 
                    trial_performances = np.array([i[0] if i is not None else -np.inf  for i in trial_performances])
                    best_trial_performances = np.argsort(trial_performances)[::-1]

                if dataset.eval_metric_direction=="minimize": 
                    trial_performances = np.array([i[0] if i is not None else np.inf  for i in trial_performances])
                    best_trial_performances = np.argsort(trial_performances)
                
                y_train_pred_seed_ens = np.array([trials[i].user_attrs["predictions"][0] for i in range(n_trials)])[best_trial_performances].mean(axis=0)
                y_val_pred_seed_ens = np.array([trials[i].user_attrs["predictions"][1] for i in range(n_trials)])[best_trial_performances].mean(axis=0)
                y_test_pred_seed_ens = np.array([trials[i].user_attrs["predictions"][2] for i in range(n_trials)])[best_trial_performances].mean(axis=0)
                
            
            else: 
                print("Not enough trials for ensembling - disable ensembling over trials.")
                seed_configs["hpo"]["ensemble"] = False

            res["performance_ens"] = {}
            res["performance_ens"]["Train"] = dataset.eval_metric(y_train_seed,y_train_pred_seed_ens)
            res["performance_ens"]["Val"] = dataset.eval_metric(y_val_seed,y_val_pred_seed_ens)
            res["performance_ens"]["Test"] = dataset.eval_metric(y_test_seed,y_test_pred_seed_ens)
            res["predictions_ens"] = [y_train_pred_seed_ens, y_val_pred_seed_ens, y_test_pred_seed_ens]    
            
    
    else:
        if dataset.valid_type in ["5CV", "10CV"]:
            if dataset.valid_type == "5CV":
                n_folds = 5
            elif dataset.valid_type == "10CV":
                n_folds = 10
            
            # Concatenate X_train and X_val for cross-validation
            X_full = pd.concat([X_train, X_val], axis=0)
            y_full = pd.concat([y_train, y_val], axis=0)
            if dataset.dataset_name in ["lymph5CV", "lymph5CV_resetsplit"]:
            #     X_full = pd.concat([X_full, X_full], axis=0)   
            #     y_full = pd.concat([y_full, y_full], axis=0)
                X_full = X_full.reset_index(drop=True)
                y_full = y_full.reset_index(drop=True)
            
            # Determine prediction shape based on task type
            if dataset.task_type == "regression":
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed_configs["seed"])
                splits = kf.split(X_full)
                prediction_type = "vector"
                prediction_columns = [0]
            elif dataset.task_type == "binary":
                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_configs["seed"])
                splits = kf.split(X_full, y_full)
                prediction_type = "vector"
                prediction_columns = [0]
            elif dataset.task_type == "multiclass":
                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_configs["seed"])
                splits = kf.split(X_full, y_full)
                prediction_type = "matrix"
                num_classes = len(np.unique(y_full)) if len(y_full.shape) == 1 else y_full.shape[1]
                prediction_columns = range(num_classes)
            else:
                raise ValueError(f"Unsupported task type: {dataset.task_type}")
            
            # Initialize predictions
            if prediction_type == "vector":
                y_train_pred_seed = pd.Series(0, index=X_full.index, dtype=float)
                y_val_pred_seed = pd.Series(0, index=X_full.index, dtype=float)
                y_test_pred_seed = pd.Series(0, index=X_test.index, dtype=float)
            elif prediction_type == "matrix":
                y_train_pred_seed = pd.DataFrame(0, index=X_full.index, columns=prediction_columns, dtype=float)
                y_val_pred_seed = pd.DataFrame(0, index=X_full.index, columns=prediction_columns, dtype=float)
                y_test_pred_seed = pd.DataFrame(0, index=X_test.index, columns=prediction_columns, dtype=float)
            
            # Counters for aggregation
            train_counts = pd.Series(0, index=X_full.index, dtype=int)
            
            # To store fold-specific predictions and scores
            fold_train_predictions = []
            fold_val_predictions = []
            fold_test_predictions = []
            
            # To store performance results
            res = {"performance": {}, "times": {}, "predictions": {}}
            train_scores_folds = []
            val_scores_folds = []
            test_scores_folds = []
            
            start = time.time()
            
            for train_index, val_index in splits:
                # Split data for the current fold
                X_train_cv, X_val_cv = X_full.iloc[train_index], X_full.iloc[val_index]
                y_train_cv, y_val_cv = y_full.iloc[train_index], y_full.iloc[val_index]

                # Initialize and train the model
                # if seed_configs["model"]["model_name"] =="TabPFNv2":
                #     from models_tabpfn import TabPFNv2 as model_class
                # else:
                from tabular_learning.models import get_model 
                model_class = get_model(seed_configs["model"]["model_name"])
                
                model = model_class(params=seed_configs["model"])
                model.fit(X_train_cv.copy(), y_train_cv.copy(), [(X_val_cv.copy(), y_val_cv.copy())])
            
                # Predict based on task type
                if dataset.task_type in ["regression", "binary"]:
                    y_train_cv_pred = pd.Series(model.predict(X_train_cv).ravel(), index=X_train_cv.index)
                    y_val_cv_pred = pd.Series(model.predict(X_val_cv).ravel(), index=X_val_cv.index)
                    y_test_fold_pred = pd.Series(model.predict(X_test).ravel(), index=X_test.index)
                elif dataset.task_type == "multiclass":
                    y_train_cv_pred = pd.DataFrame(model.predict(X_train_cv), index=X_train_cv.index, columns=prediction_columns)
                    y_val_cv_pred = pd.DataFrame(model.predict(X_val_cv), index=X_val_cv.index, columns=prediction_columns)
                    y_test_fold_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=prediction_columns)
            
                # Update out-of-fold predictions for val
                y_val_pred_seed.loc[y_val_cv.index] = y_val_cv_pred
            
                # Accumulate predictions for train samples
                y_train_pred_seed.loc[y_train_cv.index] += y_train_cv_pred
                train_counts.loc[y_train_cv.index] += 1
            
                # Aggregate test predictions
                y_test_pred_seed += y_test_fold_pred
            
                # Store fold-specific predictions
                fold_train_predictions.append(y_train_cv_pred)
                fold_val_predictions.append(y_val_cv_pred)
                fold_test_predictions.append(y_test_fold_pred)
            
                # Evaluate performance for the current fold
                train_score = dataset.eval_metric(y_train_cv, y_train_cv_pred)
                val_score = dataset.eval_metric(y_val_cv, y_val_cv_pred)
                test_score = dataset.eval_metric(y_test, y_test_fold_pred)
            
                train_scores_folds.append(train_score)
                val_scores_folds.append(val_score)
                test_scores_folds.append(test_score)
            
            # Average the train predictions where applicable
            if dataset.task_type in ["regression", "binary"]:
                y_train_pred_seed /= train_counts
            elif dataset.task_type == "multiclass":
                y_train_pred_seed = y_train_pred_seed.div(train_counts, axis=0)
            
            # Final test predictions (average over folds)
            y_test_pred_seed /= kf.get_n_splits()
            
            # Postprocessing for final predictions
            if "neuralnet" in dataset.preprocess_states:
                y_train_seed = dataset.neuralnet_postprocessing(X_full, y_full)
                y_val_seed = dataset.neuralnet_postprocessing(X_full, y_full)
                y_test_seed = dataset.neuralnet_postprocessing(X_test, y_test)
                y_train_pred_seed = dataset.neuralnet_postprocessing(X_full, y_train_pred_seed)
                y_val_pred_seed = dataset.neuralnet_postprocessing(X_full, y_val_pred_seed)
                y_test_pred_seed = dataset.neuralnet_postprocessing(X_test, y_test_pred_seed)
            else:
                y_train_seed = y_full.copy()
                y_val_seed = y_full.copy()
                y_test_seed = y_test.copy()
            if "minimalistic" in dataset.preprocess_states:
                y_train_seed = dataset.minimalistic_postprocessing(X_full, y_train_seed)
                y_val_seed = dataset.minimalistic_postprocessing(X_full, y_val_seed)
                y_test_seed = dataset.minimalistic_postprocessing(X_test, y_test_seed)
                y_train_pred_seed = dataset.minimalistic_postprocessing(X_full, y_train_pred_seed)
                y_val_pred_seed = dataset.minimalistic_postprocessing(X_full, y_val_pred_seed)
                y_test_pred_seed = dataset.minimalistic_postprocessing(X_test, y_test_pred_seed)
            else:
                y_train_seed = y_full.copy()
                y_val_seed = y_full.copy()
                y_test_seed = y_test.copy()
            if "expert" in dataset.preprocess_states:
                y_train_seed = dataset.expert_postprocessing(X_full, y_train_seed)
                y_val_seed = dataset.expert_postprocessing(X_full, y_val_seed)
                y_test_seed = dataset.expert_postprocessing(X_test_seed, y_test_seed)
                
                y_train_pred_seed = dataset.expert_postprocessing(X_full, y_train_pred_seed)
                y_val_pred_seed = dataset.expert_postprocessing(X_full, y_val_pred_seed)
                y_test_pred_seed = dataset.expert_postprocessing(X_test_seed, y_test_pred_seed)
                
            # Final performance evaluation
            # res["performance"]["Train"] = dataset.eval_metric(y_full, y_train_pred_seed)
            # res["performance"]["Val"] = dataset.eval_metric(y_full, y_val_pred_seed)
            # res["performance"]["Test"] = dataset.eval_metric(y_test, y_test_pred_seed)
            
            # # Store aggregated predictions
            # res["predictions"] = [y_train_pred_seed, y_val_pred_seed, y_test_pred_seed]
            
            # Store fold-specific predictions
            res["fold_predictions"] = {
                "train": fold_train_predictions,
                "val": fold_val_predictions,
                "test": fold_test_predictions,
            }
            
            end = time.time()
            res["times"]["train_time"] = (end - start) / 60
            res["times"]["test_time"] = (end - start) / 60
        else:
            # Train model
            # if seed_configs["model"]["model_name"] =="TabPFNv2":
            #     from models_tabpfn import TabPFNv2 as model_class
            # else:
            from tabular_learning.models import get_model 
            model_class = get_model(seed_configs["model"]["model_name"])
            
            start = time.time()
            model = model_class(params=seed_configs["model"])
    
            model.fit(X_train.copy(), y_train.copy(),
                      [(X_val.copy(), y_val.copy())],
                     )
            
            end = time.time()
            res["times"]["train_time"] = (end-start)/60
    
            start = time.time()
            y_train_pred_seed = model.predict(X_train)
            y_val_pred_seed = model.predict(X_val)
            y_test_pred_seed = model.predict(X_test)
            
            end = time.time()
            res["times"]["test_time"] = (end-start)/60
            
            print(f'Fit+Predict Time: {res["times"]["train_time"]+res["times"]["test_time"]}')
     
            # Apply model-specific postprocessing
            if "neuralnet" in dataset.preprocess_states:
                y_train_seed = dataset.neuralnet_postprocessing(X_train, y_train)
                y_val_seed = dataset.neuralnet_postprocessing(X_val, y_val)
                y_test_seed = dataset.neuralnet_postprocessing(X_test, y_test)
                y_train_pred_seed = dataset.neuralnet_postprocessing(X_train, y_train_pred_seed)
                y_val_pred_seed = dataset.neuralnet_postprocessing(X_val, y_val_pred_seed)
                y_test_pred_seed = dataset.neuralnet_postprocessing(X_test, y_test_pred_seed)
            else:
                y_train_seed = y_train.copy()
                y_val_seed = y_val.copy()
                y_test_seed = y_test.copy()
                
            if "minimalistic" in dataset.preprocess_states:
                y_train_seed = dataset.minimalistic_postprocessing(X_train, y_train_seed)
                y_val_seed = dataset.minimalistic_postprocessing(X_val, y_val_seed)
                y_test_seed = dataset.minimalistic_postprocessing(X_test, y_test_seed)
                y_train_pred_seed = dataset.minimalistic_postprocessing(X_train, y_train_pred_seed)
                y_val_pred_seed = dataset.minimalistic_postprocessing(X_val, y_val_pred_seed)
                y_test_pred_seed = dataset.minimalistic_postprocessing(X_test, y_test_pred_seed)
            if "expert" in dataset.preprocess_states:
                y_train_seed = dataset.expert_postprocessing(X_train, y_train_seed)
                y_val_seed = dataset.expert_postprocessing(X_val, y_val_seed)
                y_test_seed = dataset.expert_postprocessing(X_test, y_test_seed)
                
                y_train_pred_seed = dataset.expert_postprocessing(X_train, y_train_pred_seed)
                y_val_pred_seed = dataset.expert_postprocessing(X_val, y_val_pred_seed)
                y_test_pred_seed = dataset.expert_postprocessing(X_test, y_test_pred_seed)
                
    # Specific implementation for the Higgs-Boson dataset
    res["performance"]["Train"] = dataset.eval_metric(y_train_seed,y_train_pred_seed)
    res["performance"]["Val"] = dataset.eval_metric(y_val_seed,y_val_pred_seed)
    res["performance"]["Test"] = dataset.eval_metric(y_test_seed,y_test_pred_seed)
    res["predictions"] = [y_train_pred_seed, y_val_pred_seed, y_test_pred_seed]

    print(f'Val Performance seed no. {num_seed}: {res["performance"]["Val"]}') 
    if seed_configs["hpo"]["ensemble"]:
        print(f'Val Ensemble Performance seed {num_seed}: {res["performance_ens"]["Val"]}')
    
    return res
 


def tune_hyperparameters(
    dataset,
    X_train, X_val, X_test, 
    y_train, y_val, y_test,
    configs,
    num_seed): # External


    exp_name = configs["exp_name"]
    
    if not os.path.exists(configs["model"]["save_path"]):
        os.makedirs(configs["model"]["save_path"])
    
    set_seed(configs["seed"])

    eval_metric, eval_metric_direction = get_metric(dataset.eval_metric_name)

    def objective(trial, study):
        # if configs["model"]["model_name"] =="TabPFNv2":
        #     from models_tabpfn import TabPFNv2 as model_class
        # else:
        from tabular_learning.models import get_model 
        model_class = get_model(configs["model"]["model_name"])
    
        configs["model"]["hyperparameters"] = model_class.get_optuna_hyperparameters(
            trial,
            n_features = X_train.shape[1],
            large_dataset = configs["model"]["large_dataset"],
            dataset_name = configs["model"]["dataset_name"],
            sample_size = configs["model"]["sample_size"],
            task_type = dataset.task_type,
        ) 
        
        print(configs["model"]["hyperparameters"])
        if dataset.valid_type in ["5CV", "10CV"]:
            if dataset.valid_type == "5CV":
                n_splits = 5
            elif dataset.valid_type == "10CV":
                n_splits = 10
                
            # Concatenate X_train and X_val
            X_full = pd.concat([X_train, X_val], axis=0)
            y_full = pd.concat([y_train, y_val], axis=0)
            
            if dataset.dataset_name in ["lymph5CV", "lymph5CV_resetsplit"]:
                X_full = X_full.reset_index(drop=True)
                y_full = y_full.reset_index(drop=True)
            
            # Determine prediction shape based on task type
            if dataset.task_type == "regression":
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=configs["seed"])
                splits = kf.split(X_full)
                prediction_type = "vector"
                prediction_columns = [0]
            elif dataset.task_type == "binary":
                kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=configs["seed"])
                splits = kf.split(X_full, y_full)
                prediction_type = "vector"
                prediction_columns = [0]
            elif dataset.task_type == "multiclass":
                kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=configs["seed"])
                splits = kf.split(X_full, y_full)
                prediction_type = "matrix"
                num_classes = len(np.unique(y_full)) if len(y_full.shape) == 1 else y_full.shape[1]
                prediction_columns = range(num_classes)
            else:
                raise ValueError(f"Unsupported task type: {dataset.task_type}")
            
            # Initialize predictions
            if prediction_type == "vector":
                train_predictions = pd.Series(0, index=X_full.index, dtype=float)
                val_predictions = pd.Series(0, index=X_full.index, dtype=float)
                test_predictions = pd.Series(0, index=X_test.index, dtype=float)
            elif prediction_type == "matrix":
                train_predictions = pd.DataFrame(0, index=X_full.index, columns=prediction_columns, dtype=float)
                val_predictions = pd.DataFrame(0, index=X_full.index, columns=prediction_columns, dtype=float)
                test_predictions = pd.DataFrame(0, index=X_test.index, columns=prediction_columns, dtype=float)

            
            # Counters for aggregation
            train_counts = pd.Series(0, index=X_full.index, dtype=int)
            
            # To store fold-specific scores and predictions
            train_scores_folds = []
            val_scores_folds = []
            test_scores_folds = []
            
            fold_train_predictions = []
            fold_val_predictions = []
            fold_test_predictions = []
            
            start = time.time()
            
            for train_index, val_index in splits:
                # Use iloc for indexing to maintain pandas structure
                X_train_cv, X_val_cv = X_full.iloc[train_index], X_full.iloc[val_index]
                y_train_cv, y_val_cv = y_full.iloc[train_index], y_full.iloc[val_index]
                    
                model = model_class(params=configs["model"])
                model.fit(X_train_cv, y_train_cv, [(X_val_cv, y_val_cv)])
            
                
                # Predict based on task type
                if dataset.task_type in ["regression", "binary"]:
                    y_train_cv_pred = pd.Series(model.predict(X_train_cv).ravel(), index=X_train_cv.index)
                    y_val_cv_pred = pd.Series(model.predict(X_val_cv).ravel(), index=X_val_cv.index)
                    y_test_fold_pred = pd.Series(model.predict(X_test).ravel(), index=X_test.index)
                elif dataset.task_type == "multiclass":
                    y_train_cv_pred = pd.DataFrame(model.predict(X_train_cv), index=X_train_cv.index, columns=prediction_columns)
                    y_val_cv_pred = pd.DataFrame(model.predict(X_val_cv), index=X_val_cv.index, columns=prediction_columns)
                    y_test_fold_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=prediction_columns)

                # Update out-of-fold predictions for val
                val_predictions.loc[y_val_cv.index] = y_val_cv_pred
            
                # Accumulate predictions for train samples
                train_predictions.loc[y_train_cv.index] += y_train_cv_pred
                train_counts.loc[y_train_cv.index] += 1
            
                # Aggregate test predictions
                test_predictions += y_test_fold_pred
            
                # Store fold-specific predictions
                fold_train_predictions.append(y_train_cv_pred)
                fold_val_predictions.append(y_val_cv_pred)
                fold_test_predictions.append(y_test_fold_pred)
            
                # Evaluate fold performance
                train_score = eval_metric(y_train_cv, y_train_cv_pred)
                val_score = eval_metric(y_val_cv, y_val_cv_pred)
                test_score = eval_metric(y_test, y_test_fold_pred)
            
                train_scores_folds.append(train_score)
                val_scores_folds.append(val_score)
                test_scores_folds.append(test_score)
            
            # Average the train predictions where applicable
            if dataset.task_type in ["regression", "binary"]:
                train_predictions /= train_counts
            elif dataset.task_type == "multiclass":
                train_predictions = train_predictions.div(train_counts, axis=0)
            
            # Final test predictions (average over folds)
            test_predictions /= kf.get_n_splits()
            
            # Apply model-specific postprocessing
            if "neuralnet" in dataset.preprocess_states:
                y_train_eval = dataset.neuralnet_postprocessing(X_full, y_full)
                y_val_eval = dataset.neuralnet_postprocessing(X_full, y_full)
                y_test_eval = dataset.neuralnet_postprocessing(X_test, y_test)
                
                y_train_pred = dataset.neuralnet_postprocessing(X_full, train_predictions)
                y_val_pred = dataset.neuralnet_postprocessing(X_full, val_predictions)
                y_test_pred = dataset.neuralnet_postprocessing(X_test, test_predictions)
            else:
                y_train_eval = y_full.copy()
                y_val_eval = y_full.copy()
                y_test_eval = y_test.copy()
                y_train_pred = train_predictions
                y_val_pred = val_predictions
                y_test_pred = test_predictions
            # Apply expert-specific preprocessing
            if "minimalistic" in dataset.preprocess_states:
                y_train_eval = dataset.minimalistic_postprocessing(X_train, y_train_eval)
                y_val_eval = dataset.minimalistic_postprocessing(X_val, y_val_eval)
                y_test_eval = dataset.minimalistic_postprocessing(X_test, y_test_eval)
                
                y_train_pred = dataset.minimalistic_postprocessing(X_train, y_train_pred)
                y_val_pred = dataset.minimalistic_postprocessing(X_val, y_val_pred)
                y_test_pred = dataset.minimalistic_postprocessing(X_test, y_test_pred)
            if "expert" in dataset.preprocess_states:
                y_train_eval = dataset.expert_postprocessing(X_train, y_train_eval)
                y_val_eval = dataset.expert_postprocessing(X_val, y_val_eval)
                y_test_eval = dataset.expert_postprocessing(X_test, y_test_eval)
                
                y_train_pred = dataset.expert_postprocessing(X_train, y_train_pred)
                y_val_pred = dataset.expert_postprocessing(X_val, y_val_pred)
                y_test_pred = dataset.expert_postprocessing(X_test, y_test_pred)
            
            # Evaluate overall train and val scores based on aggregated predictions
            final_train_score = eval_metric(y_train_eval, y_train_pred)
            final_val_score = eval_metric(y_val_eval, y_val_pred)
            final_test_score = eval_metric(y_test_eval, y_test_pred)
            
            end = time.time()
            train_time = (end - start) / 60
            
            # Save aggregated predictions and scores
            trial.set_user_attr("train_predictions", y_train_pred)
            trial.set_user_attr("val_predictions", y_val_pred)
            trial.set_user_attr("test_predictions", y_test_pred)
            
            trial.set_user_attr("train_performance", final_train_score)
            trial.set_user_attr("val_performance", final_val_score)
            trial.set_user_attr("test_performance", final_test_score)
            
            # Save fold-specific scores for detailed analysis
            trial.set_user_attr("fold_train_scores", train_scores_folds)
            trial.set_user_attr("fold_val_scores", val_scores_folds)
            trial.set_user_attr("fold_test_scores", test_scores_folds)
            
            # Save fold-specific predictions for detailed analysis
            trial.set_user_attr("fold_train_predictions", fold_train_predictions)
            trial.set_user_attr("fold_val_predictions", fold_val_predictions)
            trial.set_user_attr("fold_test_predictions", fold_test_predictions)
            
            trial.set_user_attr("train_time", train_time)
            trial.set_user_attr("test_time", None)

            trial.set_user_attr("predictions", [y_train_pred,y_val_pred,y_test_pred])
            
            val_score = np.mean(val_scores_folds)
            
        else:
        
            start = time.time()
            model = model_class(params=configs["model"])

            model.fit(X_train, y_train,
                      [(X_val, y_val)],
                      )
            
            end = time.time()
            train_time = (end-start)/60
    
            start = time.time()
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            end = time.time()
            test_time = (end-start)/60
        
            # Apply model-specific postprocessing
            if "neuralnet" in dataset.preprocess_states:
                y_train_eval = dataset.neuralnet_postprocessing(X_train, y_train)
                y_val_eval = dataset.neuralnet_postprocessing(X_val, y_val)
                y_test_eval = dataset.neuralnet_postprocessing(X_test, y_test)
                
                y_train_pred = dataset.neuralnet_postprocessing(X_train, y_train_pred)
                y_val_pred = dataset.neuralnet_postprocessing(X_val, y_val_pred)
                y_test_pred = dataset.neuralnet_postprocessing(X_test, y_test_pred)
            else:
                y_train_eval = y_train.copy()
                y_val_eval = y_val.copy()
                y_test_eval = y_test.copy()
            # Apply expert-specific preprocessing
            if "minimalistic" in dataset.preprocess_states:
                y_train_eval = dataset.minimalistic_postprocessing(X_train, y_train_eval)
                y_val_eval = dataset.minimalistic_postprocessing(X_val, y_val_eval)
                y_test_eval = dataset.minimalistic_postprocessing(X_test, y_test_eval)
                
                y_train_pred = dataset.minimalistic_postprocessing(X_train, y_train_pred)
                y_val_pred = dataset.minimalistic_postprocessing(X_val, y_val_pred)
                y_test_pred = dataset.minimalistic_postprocessing(X_test, y_test_pred)
            if "expert" in dataset.preprocess_states:
                    y_train_eval = dataset.expert_postprocessing(X_train, y_train_eval)
                    y_val_eval = dataset.expert_postprocessing(X_val, y_val_eval)
                    y_test_eval = dataset.expert_postprocessing(X_test, y_test_eval)
                    
                    y_train_pred = dataset.expert_postprocessing(X_train, y_train_pred)
                    y_val_pred = dataset.expert_postprocessing(X_val, y_val_pred)
                    y_test_pred = dataset.expert_postprocessing(X_test, y_test_pred)

            
            try:
                train_score = eval_metric(y_train_eval,y_train_pred)
                val_score = eval_metric(y_val_eval,y_val_pred)
                test_score = eval_metric(y_test_eval,y_test_pred)
            except: 
                if eval_metric_direction=="minimize":
                    train_score = 1e5
                    val_score = 1e5
                    test_score = 1e5
                elif eval_metric_direction=="maximize":
                    train_score = -1e5
                    val_score = -1e5
                    test_score = -1e5
        
            trial.set_user_attr("predictions", [y_train_pred,y_val_pred,y_test_pred])
            trial.set_user_attr("train_performance", train_score)
            trial.set_user_attr("test_performance", test_score)
            trial.set_user_attr("train_time", train_time)
            trial.set_user_attr("test_time", test_time)
    
        if (trial.number % study.user_attrs["save_interval"])==0:
            joblib.dump(study, study.user_attrs["save_path"])

        return val_score

    def wrapped_objective(trial):
        return objective(trial, study)

    if not os.path.exists(f'{configs["model"]["save_path"]}/{exp_name}_study.pkl'):

        # Create a study object and optimize the objective function
        sampler = optuna.samplers.TPESampler(seed=configs["seed"],
                                             n_startup_trials=configs["hpo"]["n_startup_trials"],
                                             multivariate=True,
                                             warn_independent_sampling=False
                                            ) 

        study = optuna.create_study(direction=eval_metric_direction,
                                    sampler=sampler,
                                   )
        study.set_user_attr("save_path", f'{configs["model"]["save_path"]}/{exp_name}_study.pkl')
        study.set_user_attr("save_interval", configs["hpo"]["save_interval"])
        
        study.optimize(wrapped_objective, 
                       n_trials=configs["hpo"]["n_trials"], 
                       gc_after_trial=True)

        joblib.dump(study, study.user_attrs["save_path"])

        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)

    else:
        print(f"Results '{configs['model']['save_path']}/{exp_name}_study.pkl' already exist and will be loaded.")  
        
        study = joblib.load(f'{configs["model"]["save_path"]}/{exp_name}_study.pkl')
        study.set_user_attr("save_path", f'{configs["model"]["save_path"]}/{exp_name}_study.pkl')
        study.set_user_attr("save_interval", configs["hpo"]["save_interval"])
        try:
            print(f"Best trial until now: {study.best_trial.value} with parameters: {study.best_trial.params}")
        except:
            print("No trials finished yet")
        if configs["hpo"]["n_trials"]>len(study.trials):
            study.optimize(wrapped_objective, 
                           n_trials=configs["hpo"]["n_trials"]-len(study.trials), 
                           gc_after_trial=True)
            joblib.dump(study, f'{configs["model"]["save_path"]}/{exp_name}_study.pkl')
    
            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)            
            
    
    return study
