import os
import pickle 
import yaml
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold, RepeatedKFold, train_test_split 

# from dataset_utils import dataset_maps
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, OrdinalEncoder, StandardScaler, LabelEncoder, MinMaxScaler, PowerTransformer

################### FROM UTILS
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
    if model_name in ["XGBoost", "XGBModel1024Bins", "XGBModelExact", "XGBModelDepth20", "XGBModelDepth1", "XGBModelDepth2", "XGBModelDepth3", "XGBModelDepth4", "XGBModelDepth5", "XGBoostLossguided", "XGBoostHolzmueller", "XGBoostSmallData"]:
        with open('configs/xgb_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        configs["model"]["model_name"] = model_name
    if model_name == "MLPContLinear":
        with open('configs/mlp_contlinear_config.yaml', 'r') as file: 
            configs = yaml.safe_load(file)
    if model_name == "MLPContReLU":
        with open('configs/mlp_contrelu_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
    if model_name in ["MLP", "MLPModelMish", "MLPModelLongTrain"]:
        with open('configs/mlp_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        configs["model"]["model_name"] = model_name
    if model_name == "ResNet":
        with open('configs/resnet_config.yaml', 'r') as file: 
            configs = yaml.safe_load(file)
    if model_name in ["CatBoost", "CatBoostModel1024Bins", "CatBoostLossguided"]:
        with open('configs/catboost_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        configs["model"]["model_name"] = model_name
    if model_name in ["MLP-PLR", "MLPPLRHighEmbedding", "MLPPLRFixedArchitecture", "MLPPLRFixedArchitectureTuneSeed", "MLPPLRFeatureDropout", "MLPPLRStopInterpol", "MLP-PLR-minmax", "MLP-PLR-notransform"]:
        with open('configs/mlpplr_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        configs["model"]["model_name"] = model_name
        if model_name == "MLP-PLR-minmax":# , 
            configs["dataset"]["num_scaler"] = "minmax" 
        elif model_name == "MLP-PLR-notransform":
            configs["dataset"]["num_scaler"] = None 
    if model_name == "TabMStopInterpol":
        with open('configs/tabm_stopinterpol_config.yaml', 'r') as file:
            configs = yaml.safe_load(file) 
    if model_name == "MLPStopInterpol":
        with open('configs/mlp_stopinterpol_config.yaml', 'r') as file:
            configs = yaml.safe_load(file) 
    if model_name == "FTTransformer":
        with open('configs/fttransformer_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
    if model_name in ["LightGBM", "LightGBMModelDepthLimit", "LightGBMModel1024Bins", "LightGBMModelNomindataleaf", "LightGBMHolzmueller", "LightGBMModelHuertas", "LightGBMModelHuertasTuneMinLeaf", "LightGBMModelExperiment", "LightGBMModel1024BinsHuertasTuneMinLeaf", "LightGBMModel50000Bins", "LightGBMModelAllCat", "LightGBMModelHuertas2"]:
        with open('configs/lightgbm_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        configs["model"]["model_name"] = model_name
    if model_name == "TabM":
        with open('configs/tabm_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
    if model_name == "TabMmini":
        with open('configs/tabm-mini_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
    if model_name == "GRANDE":
        with open('configs/grande_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
    if model_name == "RealMLP":
        with open('configs/realmlp_config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
    if model_name == "TabPFNv2":
        with open('configs/tabpfnv2_config.yaml', 'r') as file:
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



##################

class TabularDataset():
    ''' '''
    def __init__(self, dataset_name, val_strategy="5CV-grouped", eval_metric_name=None, seed=42):
        self.dataset_name = dataset_name
        self.split_type = "custom"#dataset_maps[dataset_name]["split"]
        self.task_type = "regression"#dataset_maps[dataset_name]["task_type"]
        self.heavy_tailed = False
        self.preprocess_states = []
        self.x_scaled = False
        self.large_dataset = False
        self.val_strategy = val_strategy

        if "5CV" in dataset_name:
            self.valid_type = "5CV"
        elif "10CV" in dataset_name:
            self.valid_type = "10CV"
        else:
            self.valid_type = "single_split"

        # if self.split_type == "custom":
        self.X = pd.read_csv(f"data/processed/{dataset_name}/{dataset_name}_two_TS.csv", low_memory=False)
        self.y = self.X['rem_time']/60/60/24
        self.X = self.X.drop('rem_time',axis=1)

        ### Fix weird feature names for some models (LightGBM)
        import re
        def make_safe(name):
            # keep letters, digits, and underscores; replace everything else with '_'
            return re.sub(r'[^0-9A-Za-z_]', '_', name)
        self.X.columns = [make_safe(col) for col in self.X.columns ]
        
        self.y_col = self.y.name

        if self.task_type== "multiclass" or self.task_type== "binary":
            self.target_label_enc = LabelEncoder()
            self.y = pd.Series(self.target_label_enc.fit_transform(self.y),index=self.y.index, name=self.y_col)

            if self.task_type== "multiclass":
                self.num_classes = self.y.nunique()

        self.split_indices = self.get_splits(self.X, self.y, seed)

        # if dataset_name == "HelpDesk":
        drop_cols = ["case_concept_name", "prefix_length", "set", # identifiers
         "start", "end", "enabled_time", # used for feature extraction
         "next_proc", "next_wait" # alternative targets
        ]
        self.X = self.X.drop(drop_cols, axis=1)
        self.cat_indices = np.where(self.X.dtypes=="object")[0].tolist()
        
        if eval_metric_name is None:
            if self.task_type == "binary":
                self.eval_metric_name = "logloss"  
            elif self.task_type == "multiclass":
                self.eval_metric_name = "mlogloss"
            elif self.task_type == "regression":
                self.eval_metric_name = "rmse"
        else:
            self.eval_metric_name  = eval_metric_name
            if self.eval_metric_name=="Accuracy" and self.task_type in ["multiclass", "classification"]:
                self.eval_metric_name  = "mAccuracy"
                
        self.eval_metric, self.eval_metric_direction = get_metric(self.eval_metric_name)

        # Ensure proper binary feature handling
        for num, col in enumerate(self.X.columns):
            # Encode binary categorical features   
            if self.X[col].nunique()==2:
                value_1 = self.X[col].dropna().unique()[0]
                self.X[col] = (self.X[col]==value_1).astype(float)
                # X_val[col] = (X_val[col]==value_1).astype(float)
                # X_test[col] = (X_test[col]==value_1).astype(float)
                if num in self.cat_indices:
                    self.cat_indices.remove(num)

        
    
    def minimalistic_preprocessing(self, 
                                   X_train, X_val,  X_test, 
                                   y_train, y_val, y_test, cat_indices, 
                                   scaler=None, one_hot_encode=False, use_test=False):

        print("Apply minimalistic preprocessing")
        
        # Encode binary cat features as numeric
        for col in X_train.columns[X_train.nunique()==2]:
            if X_train[col].dtype in [str, "O", "category", "object"]:
                le = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
                X_train[col] = le.fit_transform(X_train[[col]])
                mode = X_train[col].mode()[0]

                if len(X_test[col].unique())==2:
                    X_val[col] = le.transform(X_val[[col]])
                    X_test[col] = le.transform(X_test[[col]])
                else:
                    X_val[col] = le.transform(X_val[[col]])
                    X_val[col] = X_val[col].fillna(mode)
                    
                    X_test[col] = le.transform(X_test[[col]])
                    X_test[col] = X_test[col].fillna(mode)


        cat_indices_new = pd.Series(cat_indices).copy().values.tolist()
        
        
        # Define categorical feature types
        cat_indices_new += list(np.where(X_train.dtypes=="O")[0]) 
        cat_indices_new += list(np.where(X_train.dtypes=="object")[0]) 
        cat_indices_new += list(np.where(X_train.dtypes=="category")[0]) 
        cat_indices_new = np.unique(cat_indices_new).tolist()
        
        for num, col in list(zip(cat_indices_new,X_train.columns[cat_indices_new])):
            # Encode binary categorical features
            if X_train[col].nunique()==2:
                value_1 = X_train[col].dropna().unique()[0]
                X_train[col] = (X_train[col]==value_1).astype(float)
                X_val[col] = (X_val[col]==value_1).astype(float)
                X_test[col] = (X_test[col]==value_1).astype(float)
                cat_indices_new.remove(num)
            else:
                # Note: The category dtype needs to entail all train categories when applying .astype("category") on test data
                dtype = pd.CategoricalDtype(categories=list(X_train[col].astype(str).fillna("nan").unique()))
                X_train[col] = X_train[col].astype(str).fillna("nan").astype(dtype)
                X_val[col] = X_val[col].astype(str).fillna("nan").astype(dtype)       
                X_test[col] = X_test[col].astype(str).fillna("nan").astype(dtype)       
                
        
        cont_indices = np.array([i for i in range(X_train.shape[1]) if i not in cat_indices_new])
        cont_col_names = X_train.iloc[:,cont_indices].columns
        
        X_concat = pd.concat([X_train, X_val, X_test])
        
        # Fill missing values of continuous columns with mean 
        if X_train.isna().sum().sum()>0:
            if use_test:
                X_val[cont_col_names] = X_val[cont_col_names].fillna(X_concat[cont_col_names].mean())
                X_test[cont_col_names] = X_test[cont_col_names].fillna(X_concat[cont_col_names].mean())
                X_train[cont_col_names] = X_train[cont_col_names].fillna(X_concat[cont_col_names].mean())
            else:
                X_val[cont_col_names] = X_val[cont_col_names].fillna(X_train[cont_col_names].mean())
                X_test[cont_col_names] = X_test[cont_col_names].fillna(X_train[cont_col_names].mean())
                X_train[cont_col_names] = X_train[cont_col_names].fillna(X_train[cont_col_names].mean())
            
        # if scaler is not None:
        #     X_train[cont_col_names] = scaler_function.fit_transform(X_train[cont_col_names])
        #     X_test[cont_col_names] = scaler_function.transform(X_test[cont_col_names])

        # if one_hot_encode:
        #     ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        #     new_x1 = ohe.fit_transform(X_train[:, cat_indices_new])
        #     X_train = np.concatenate([new_x1, X_train[:, num_mask]], axis=1)
        #     new_x1_test = ohe.transform(X_test[:, cat_indices_new])
        #     X_test = np.concatenate([new_x1_test, X_test[:, num_mask]], axis=1)
            
        #     cat_indices_new = []
            
        # Drop constant columns
        # drop_cols = X_train.columns[X_train.nunique()==X_train.shape[0]].values.tolist()
        drop_cols = X_train.columns[X_train.nunique()==1].values.tolist()
        if len(drop_cols)>0:
            print(f"Drop {len(drop_cols)} constant/unique features")
            original_categorical_names =  X_train.columns[cat_indices_new]
            X_train.drop(drop_cols,axis=1,inplace=True)
            X_val.drop(drop_cols,axis=1,inplace=True)
            X_test.drop(drop_cols,axis=1,inplace=True)
            cat_indices_new = [np.where(X_train.columns==i)[0][0] for i in original_categorical_names if i in X_train.columns]
        
        if self.heavy_tailed: # Todo: Might move to minimalistic
            y_train = np.log1p(y_train)
            y_val = np.log1p(y_val)
            y_test = np.log1p(y_test)

        self.preprocess_states.append("minimalistic")     
        return X_train, X_val, X_test, y_train, y_val, y_test, cat_indices_new     

    def minimalistic_postprocessing(self, X, y, **kwargs):
        if self.task_type=="regression":
            if self.heavy_tailed:
                y = np.expm1(y)
        return y

    def neuralnet_preprocessing(self, 
                                X_train, X_val, X_test, 
                                y_train, y_val, y_test, cat_indices, 
                                use_test=True,
                                num_scaler = "quantile",   
                               ):
        if self.task_type=="regression":
            # if self.heavy_tailed: # Todo: Might move to minimalistic
            #     y_train = np.log1p(y_train)

            if self.dataset_name != "tb_num_reg_wine_quality":
                self.target_scaler = StandardScaler()
                y_train = pd.Series(self.target_scaler.fit_transform(y_train.values.reshape(-1,1)).ravel(),
                  name=self.y_col, index = X_train.index)
                y_val = pd.Series(self.target_scaler.transform(y_val.values.reshape(-1,1)).ravel(),
                  name=self.y_col, index = X_val.index)
                y_test = pd.Series(self.target_scaler.transform(y_test.values.reshape(-1,1)).ravel(),
                  name=self.y_col, index = X_test.index)

        cat_indices_new = pd.Series(cat_indices).copy().values.tolist()
        
        # Drop constant columns
        # drop_cols = X_train.columns[X_train.nunique()==X_train.shape[0]].values.tolist()
        drop_cols = X_train.columns[X_train.nunique()==1].values.tolist()
        if len(drop_cols)>0:
            print(f"Drop {len(drop_cols)} constant/unique features")
            original_categorical_names =  X_train.columns[cat_indices_new]
            X_train.drop(drop_cols,axis=1,inplace=True)
            X_val.drop(drop_cols,axis=1,inplace=True)
            X_test.drop(drop_cols,axis=1,inplace=True)
            cat_indices_new = [np.where(X_train.columns==i)[0][0] for i in original_categorical_names if i in X_train.columns]

        # Drop constant nan cols
        if len(cat_indices_new)==0: # 
            nan_cols = X_train.columns[X_train.isna().sum()==X_train.shape[0]].values.tolist()
            if len(nan_cols)>0:
                print(f"Drop {len(nan_cols)} all-nan features")
            #     original_categorical_names =  X_train.columns[cat_indices_new]
                X_train.drop(nan_cols,axis=1,inplace=True)
                X_val.drop(nan_cols,axis=1,inplace=True)
                X_test.drop(nan_cols,axis=1,inplace=True)
                # cat_indices_new = [np.where(X_train.columns==i)[0][0] for i in original_categorical_names if i in X_train.columns]

        X_concat = pd.concat([X_train, X_val, X_test])
        
        cont_indices = np.array([i for i in range(X_train.shape[1]) if i not in cat_indices_new])
        cont_col_names = X_train.iloc[:,cont_indices].columns

        X_concat[cont_col_names] = X_concat[cont_col_names].astype(np.float32)
        X_train[cont_col_names] = X_train[cont_col_names].astype(np.float32)
        X_val[cont_col_names] = X_val[cont_col_names].astype(np.float32)
        X_test[cont_col_names] = X_test[cont_col_names].astype(np.float32)
        
        # Apply ordinal encoding to all categorical features
        if len(cat_indices_new)>0:
            cat_col_names = X_train.iloc[:,cat_indices_new].columns
            for col in cat_col_names:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", 
                                     unknown_value=X_train[col].nunique(),
                                     encoded_missing_value=X_train[col].nunique()
                                    )
                X_train[col] = enc.fit_transform(X_train[col].values.reshape(-1,1)).astype(int)
                X_val[col] = enc.transform(X_val[col].values.reshape(-1,1)).astype(int)
                X_test[col] = enc.transform(X_test[col].values.reshape(-1,1)).astype(int)

        # Fill missing values of continuous columns with mean 
        if X_train.isna().sum().sum()>0 or X_val.isna().sum().sum()>0 or X_test.isna().sum().sum()>0:
            if use_test:
                X_val[cont_col_names] = X_val[cont_col_names].fillna(X_concat[cont_col_names].mean())
                X_test[cont_col_names] = X_test[cont_col_names].fillna(X_concat[cont_col_names].mean())
                X_train[cont_col_names] = X_train[cont_col_names].fillna(X_concat[cont_col_names].mean())
                X_concat[cont_col_names] = X_concat[cont_col_names].fillna(X_concat[cont_col_names].mean())
            else:
                X_val[cont_col_names] = X_val[cont_col_names].fillna(X_train[cont_col_names].mean())
                X_test[cont_col_names] = X_test[cont_col_names].fillna(X_train[cont_col_names].mean())
                X_train[cont_col_names] = X_train[cont_col_names].fillna(X_train[cont_col_names].mean())
        
        print(cat_indices_new, self.x_scaled)
        if X_train.shape[1]!=len(cat_indices_new):
            if not self.x_scaled:
                # self.x_scaler = QuantileTransformer(
                #             n_quantiles= 1000
                #         )        
                # X_train[cont_col_names] = self.x_scaler.fit_transform(X_train[cont_col_names])
                # X_test[cont_col_names] = self.x_scaler.transform(X_test[cont_col_names])
                if num_scaler in ["quantile", "quantile-minmax"]:
                    quantile_noise = 1e-4
                    if use_test:
                        quantile_use = np.copy(X_concat[cont_col_names].values).astype(np.float64)
                    else:
                        quantile_use = np.copy(X_train[cont_col_names].values).astype(np.float64)
                        
                    stds = np.std(quantile_use, axis=0, keepdims=True)
                    noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                    quantile_use += noise_std * np.random.randn(*quantile_use.shape)    
                    if use_test:
                        quantile_use = pd.DataFrame(quantile_use, columns=cont_col_names, index=X_concat.index)
                    else:
                        quantile_use = pd.DataFrame(quantile_use, columns=cont_col_names, index=X_train.index)
                    
                    self.x_scaler = QuantileTransformer(
                        n_quantiles=min(quantile_use.shape[0], 1000),
                        output_distribution='normal')
        
                    self.x_scaler.fit(quantile_use.values.astype(np.float64))
                    X_train[cont_col_names] = self.x_scaler.transform(X_train[cont_col_names].values.astype(np.float64))
                    X_val[cont_col_names] = self.x_scaler.transform(X_val[cont_col_names].values.astype(np.float64))
                    X_test[cont_col_names] = self.x_scaler.transform(X_test[cont_col_names].values.astype(np.float64))
                
                    # self.x_scaled = True
                    if num_scaler == "quantile-minmax":
                        print("Apply minmax")
                        self.x_scaler_2 = MinMaxScaler()

                        self.x_scaler_2.fit(X_train[cont_col_names].values.astype(np.float64))
                        X_train[cont_col_names] = self.x_scaler_2.transform(X_train[cont_col_names].values.astype(np.float64))
                        X_val[cont_col_names] = self.x_scaler_2.transform(X_val[cont_col_names].values.astype(np.float64))
                        X_test[cont_col_names] = self.x_scaler_2.transform(X_test[cont_col_names].values.astype(np.float64))  
                elif num_scaler == "minmax":
                    print("Apply minmax")
                    self.x_scaler = MinMaxScaler()

                    self.x_scaler.fit(X_train[cont_col_names].values.astype(np.float64))
                    X_train[cont_col_names] = self.x_scaler.transform(X_train[cont_col_names].values.astype(np.float64))
                    X_val[cont_col_names] = self.x_scaler.transform(X_val[cont_col_names].values.astype(np.float64))
                    X_test[cont_col_names] = self.x_scaler.transform(X_test[cont_col_names].values.astype(np.float64))
                elif num_scaler == "zstandard":
                    print("Apply minmax")
                    self.x_scaler = StandardScaler()

                    self.x_scaler.fit(X_train[cont_col_names].values.astype(np.float64))
                    X_train[cont_col_names] = self.x_scaler.transform(X_train[cont_col_names].values.astype(np.float64))
                    X_val[cont_col_names] = self.x_scaler.transform(X_val[cont_col_names].values.astype(np.float64))
                    X_test[cont_col_names] = self.x_scaler.transform(X_test[cont_col_names].values.astype(np.float64))


                elif num_scaler == "RobustScaleSmoothClipTransform":   
                    import sklearn  
                    class RobustScaleSmoothClipTransform(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
                        def fit(self, X, y=None):
                            # don't deal with dataframes for simplicity
                            assert isinstance(X, np.ndarray)
                            self._median = np.median(X, axis=-2)
                            quant_diff = np.quantile(X, 0.75, axis=-2) - np.quantile(X, 0.25, axis=-2)
                            max = np.max(X, axis=-2)
                            min = np.min(X, axis=-2)
                            idxs = quant_diff == 0.0
                            # on indexes where the quantile difference is zero, do min-max scaling instead
                            quant_diff[idxs] = 0.5 * (max[idxs] - min[idxs])
                            factors = 1.0 / (quant_diff + 1e-30)
                            # if feature is constant on the training data,
                            # set factor to zero so that it is also constant at prediction time
                            factors[quant_diff == 0.0] = 0.0
                            self._factors = factors
                            return self
                    
                        def transform(self, X, y=None):
                            x_scaled = self._factors[None, :] * (X - self._median[None, :])
                            return x_scaled / np.sqrt(1 + (x_scaled / 3) ** 2)
                    print("Apply RobustScaleSmoothClipTransform")
                    self.x_scaler = RobustScaleSmoothClipTransform()

                    self.x_scaler.fit(X_train[cont_col_names].values.astype(np.float64))
                    X_train[cont_col_names] = self.x_scaler.transform(X_train[cont_col_names].values.astype(np.float64))
                    X_val[cont_col_names] = self.x_scaler.transform(X_val[cont_col_names].values.astype(np.float64))
                    X_test[cont_col_names] = self.x_scaler.transform(X_test[cont_col_names].values.astype(np.float64))

                        
        
        self.preprocess_states.append("neuralnet")        
        return X_train, X_val, X_test, y_train, y_val, y_test, cat_indices_new   
    

    def expert_postprocessing(self, X_train, y, **kwargs):
        if self.dataset_name in ["tb_cat_reg_nyc-taxi-green-dec-2016_exp", 
                                 "tb_cat_reg_nyc-taxi-green-dec-2016_duration_exp",
                                 "tb_cat_reg_diamonds_exp", "tb_cat_reg_medical_charges_exp", 
                                 "tb_num_reg_houses_exp", "tb_num_reg_MiamiHousing2016_exp", 
                                 "tb_num_reg_house_16H_exp", "tb_cat_reg_Brazilian_houses_exp", 
                                 
                                 ]:
            y = np.exp(y)
        return y
    
    def neuralnet_postprocessing(self, X, y):
        if self.task_type=="regression":
            if self.dataset_name != "tb_num_reg_wine_quality":
                if isinstance(y, pd.Series):
                    y = y.values.reshape(-1,1)
                y = pd.Series(self.target_scaler.inverse_transform(y.reshape(-1,1)).ravel(),
                  name=self.y_col, index = X.index)
                # if self.heavy_tailed:
                #     y = np.expm1(y)
                
        return y    

    def get_splits(self, X, y, seed=42):

        train_idx = self.X.loc[X["set"]=="Train"].index.values.tolist()
        test_idx = self.X.loc[X["set"]=="Test"].index.values.tolist()

        indices = []
        if self.val_strategy == "5CV-grouped":
            split = GroupKFold(5)
            for tr,va in split.split(self.X.iloc[train_idx], groups=self.X.iloc[train_idx]["case_concept_name"]):
                indices.append([self.X.iloc[train_idx].iloc[tr].index.values.tolist(),
                                self.X.iloc[train_idx].iloc[va].index.values.tolist(),
                                test_idx]
                              )	
        elif self.val_strategy =="10CV-grouped":
            split = GroupKFold(5)
            for tr,va in split.split(self.X.iloc[train_idx], groups=self.X.iloc[train_idx]["case_concept_name"]):
                indices.append([self.X.iloc[train_idx].iloc[tr].index.values.tolist(),
                                self.X.iloc[train_idx].iloc[va].index.values.tolist(),
                                test_idx]
                              )	
        elif self.val_strategy =="5CV-random":
            split = KFold(5, shuffle=True, random_state=seed)
            for tr,va in split.split(self.X.iloc[train_idx]): 
                indices.append([self.X.iloc[train_idx].iloc[tr].index.values.tolist(),
                                self.X.iloc[train_idx].iloc[va].index.values.tolist(),
                                test_idx]
                              )	
        elif self.val_strategy =="10CV-random":
            split = KFold(10, shuffle=True, random_state=seed)
            for tr,va in split.split(self.X.iloc[train_idx]):
                indices.append([self.X.iloc[train_idx].iloc[tr].index.values.tolist(),
                                self.X.iloc[train_idx].iloc[va].index.values.tolist(),
                                test_idx]
                              )	
        else:
            raise ValueError(f"val_strategy '{self.val_strategy}' not implemented!")
        
        
        return indices
