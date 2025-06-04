# -*- coding: utf-8 -*-
"""
Created on Fri May 23 12:53:24 2025
@author: kamirel
"""

import os
import argparse
import logging
import pandas as pd
import tensorflow as tf
from utils import *
from helpers import *
from itertools import product
from sklearn.metrics import mean_absolute_error
import numpy as np
#import pickle
import random

from tensorflow.keras.layers import (Input, Embedding, Dense, TimeDistributed,
                                     PReLU, Flatten, BatchNormalization,
                                     concatenate, LSTM, Bidirectional)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from keras.callbacks import (Callback, EarlyStopping, ModelCheckpoint, 
                             ReduceLROnPlateau)
from tensorflow.keras.models import load_model



# Configure logger
logger = logging.getLogger('LS_ICE_logger') 
logger.setLevel(logging.INFO) 
# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_model(output_dim=25, dropout_rate=0.2, learning_rate=0.002, 
              latent_dim=40, helpers_dic=None, feat_dic=None):
    print('drop:', dropout_rate)    
    input_layers = []
    emb_layers = []    
    # Trace input (always present)
    inp_trace = Input(
        shape=(output_dim, helpers_dic['trace_helper']['vocab_size']), 
        name='Xtrace')
    input_layers.append(inp_trace)    
    # Time features (shared across datasets)
    for time_feat in feat_dic['time_feat']:
        inp = Input(shape=(output_dim, 1), name=f"X{time_feat}")
        input_layers.append(inp)
        if time_feat in helpers_dic['time_helpers']:  # needs embedding
            vocab_size = helpers_dic['time_helpers'][time_feat]['size']
            dim = round(1.6 * (vocab_size - 1) ** 0.56)
            emb = Embedding(
                output_dim=dim, input_dim=vocab_size, 
                name=f"Emb{time_feat.capitalize()}")(Flatten()(inp))
            emb_layers.append(emb)
        else:
            emb_layers.append(inp)
    # Categorical features
    for cat_feat in feat_dic['cat_feat']:
        inp = Input(shape=(output_dim, 1), name=f"X{cat_feat}")
        input_layers.append(inp)
        vocab_size = helpers_dic['cat_helpers'][cat_feat]['size']
        dim = round(1.6 * (vocab_size - 1) ** 0.56)
        emb = Embedding(
            output_dim=dim, input_dim=vocab_size, 
            name=f"Emb{cat_feat.capitalize()}")(Flatten()(inp))
        emb_layers.append(emb)
    # Numerical features (no embedding)
    for num_feat in feat_dic['num_feat']:
        inp = Input(shape=(output_dim, 1), name=f"X{num_feat}")
        input_layers.append(inp)
        emb_layers.append(inp)    
    # Merge all features
    merged = concatenate([inp_trace] + emb_layers, name='concat_input')
    # Shared LSTM
    shared_lstm = Bidirectional(
        LSTM(
            latent_dim, return_sequences=True, dropout=dropout_rate, 
            recurrent_dropout=dropout_rate), name='LSTMshared1')(merged)
    b1 = BatchNormalization()(shared_lstm)

    # Trace prediction branch
    trace_lstm = Bidirectional(
        LSTM(latent_dim, return_sequences=True, dropout=dropout_rate, 
             recurrent_dropout=dropout_rate), name='LSTMtrace1')(b1)
    b2_1a = BatchNormalization()(trace_lstm)
    output_trace = TimeDistributed(
        Dense(helpers_dic['trace_helper']['vocab_size'], activation='softmax'
              ), name='trace_out')(b2_1a)

    # Time prediction branch
    time_lstm = Bidirectional(
        LSTM(
            latent_dim, return_sequences=True, dropout=dropout_rate, 
            recurrent_dropout=dropout_rate), name='LSTMtime1')(b1)
    b2_2a = BatchNormalization()(time_lstm)
    output_time = TimeDistributed(
        Dense(1, kernel_initializer='he_uniform'), name='time_out_nolr')(b2_2a)
    output_time = TimeDistributed(PReLU(), name='time_out')(output_time)

    # Build model
    model = Model(inputs=input_layers, outputs=[output_time, output_trace])
    opt = Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, 
                epsilon=1e-08, clipvalue=3.0)
    model.compile(
        loss={'trace_out': 'categorical_crossentropy', 
              'time_out': 'mean_absolute_error'},
        loss_weights={'trace_out': 1.0, 'time_out': 10.0}, optimizer=opt)
    return model


def train_model(config, epochs=500, feat_dic=None, helpers_dic=None, 
                config_file_template=None, config_base_name=None, 
                data_train=None, data_valid=None, out_dir=None, seed=None, 
                logger=None):
    training_generator = BagDataGenerator(
        data_train, config['output_dim'], feat_dic, helpers_dic, seed=seed)
    validation_generator = BagDataGenerator(
        data_valid, config['output_dim'], feat_dic, helpers_dic, seed=seed)    
    """
    for batch in training_generator:
        for name, arr in batch[0].items():
            if np.isnan(arr).any():
                print(arr)
                raise ValueError(f"NaN detected in input array: {name}")
                break
        break
    """       
    best_model_name = f"seed_{str(seed)}_best_model_{config_base_name(config)}.h5"
    best_model_path = os.path.join(out_dir, best_model_name) 
    model_checkpoint = ModelCheckpoint(
        best_model_path, monitor='val_loss', verbose=0, 
        save_best_only=True, save_weights_only=False, mode='auto')      
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=16, verbose=0, mode='auto', 
        min_delta=0.0001, cooldown=0, min_lr=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=59)   
    output_dim = config['output_dim']
    dropout = config['dropout']
    lr = config['lr']
    latent_dim = config['HUnits']   
    print('*** Now trainining:\n', config_base_name(config))
    logger.info(f'*** Now trainining:{config_base_name(config)}')    
    model = get_model(output_dim=output_dim, dropout_rate=dropout, 
                      learning_rate=lr, latent_dim=latent_dim,
                      helpers_dic=helpers_dic, feat_dic=feat_dic)      
    print(model.summary())
    training = model.fit(training_generator, 
                         validation_data=validation_generator,
                         epochs=epochs, verbose=2,
                         callbacks=[model_checkpoint, early_stopping, lr_reducer]) 
    return training, best_model_path


def inference_and_evaluate(model, data_test, feat_dic, helpers_dic, config,
                           output_csv_path=None, logger=None):
    # Prepare the test generator
    test_generator = BagDataGenerator(data_test, config['output_dim'], 
                                      feat_dic, helpers_dic, shuffle=False)
    # Predict and extract the regression output
    preds = model.predict(test_generator)
    y_pred = preds[0][:, -1, 0] 
    #y_pred = preds[0].flatten()  # first element = time_out predictions
    
    # Ensure y_pred length matches data_test
    assert len(y_pred) == len(data_test), "Prediction length mismatch"
    
    # Add predictions to dataframe
    result_df = data_test[['case_id', 'prefix_length', 'remtime_std']].copy()
    result_df['predicted_remtime'] = y_pred
    result_df['remtime_std'] = result_df['remtime_std'].apply(
        lambda x: float(x.strip().split(',')[-1]))
    result_df['predicted_remtime'] = result_df['predicted_remtime'].clip(lower=0)
    #result_df['remtime_std'] = result_df['remtime_std'] / 86400
    #result_df['predicted_remtime'] = result_df['predicted_remtime'] / 86400
    # Save to CSV
    result_df.to_csv(output_csv_path, index=False)   
    result_df=result_df[result_df['prefix_length']>1]    
    # Calculate and print MAE
    mae = mean_absolute_error(
        result_df['remtime_std'], result_df['predicted_remtime'])
    print(f"MAE on test set: {mae:.4f}")  
    logger.info(f"MAE on test set: {mae:.4f}")
    return result_df, mae


def main():
    seed_list = [42, 1337, 2023, 31415, 65537]
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs Available:", gpus)
    device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    parser = argparse.ArgumentParser(description='train_model')
    parser.add_argument('--dataset')
    args = parser.parse_args()
    out_dir = os.path.join(os.getcwd(), 'processed', args.dataset)    
    log_path = os.path.join(os.path.dirname(os.getcwd()),
                        'GraphGPS', 'PGTNet', 'raw_dataset',
                        args.dataset+'.xes')
    inter_case_df = pd.read_csv(
        os.path.join(out_dir, args.dataset+'_extended_.csv'))
    inter_case_df['duration'] = inter_case_df['duration'] / 86400
    inter_case_df['since_last'] = inter_case_df['since_last'] / 86400
    inter_case_df['remtime'] = inter_case_df['remtime'] / 86400
    if args.dataset in {'BPIC15_1', 'BPIC15_2', 'BPIC15_3', 'BPIC15_4',
                        'BPIC15_5'}:
        inter_case_df['case_id'] = inter_case_df['case_id'].astype(str)
    int_df, feat_dic, feat_lst = get_feat_dict(args.dataset, inter_case_df)     
    (data, data_train, data_valid, data_test) = read_data(
        log_path, out_dir, int_df, feat_lst, feat_dic)    
    # set the logger
    logger_path = os.path.join(out_dir, "LS_ICE_train.log")
    file_handler = logging.FileHandler(logger_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)     
    logger.info(f'final columns: {list(data.columns)}')    
    for key in feat_dic:
        feat_dic[key] = [col.replace('case:', 'case_') for col in feat_dic[key]]
    for key in feat_dic:
        feat_dic[key] = [col.replace('org:', 'org_') for col in feat_dic[key]]    
    helpers = get_helpers(data, feat_dic)
    config_base_name = lambda config: ' '.join(
        ["{}-{}".format(k,v) for k, v in config.items() if not k.startswith('_')])
    config_file_template = lambda config: 'inter_latent50' + config_base_name(config) + ' {epoch:02d} {loss:.5f} {val_loss:.5f}.h5'
 
    model_configurations = [
        {'output_dim': int(pd.concat([data_train, data_valid])['prefix_length'].max()), 
         'HUnits': units, 'dropout': do, 'lr': lr} 
        for units, do, lr in product([50], [0], [0.002])] 
    mae_list = []
    for seed in seed_list:
        logger.info(f'Training with seed {seed}')
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)   
        for config in model_configurations:
            training, best_model_path = train_model(
                config, feat_dic=feat_dic, helpers_dic=helpers,
                config_file_template=config_file_template,
                config_base_name=config_base_name,
                data_train=data_train, data_valid=data_valid,
                out_dir=out_dir, seed=seed, logger=logger)        
        output_csv_path = os.path.join(out_dir, args.dataset+'_seed_'+str(seed)+'_predictions.csv')
        best_model = load_model(best_model_path)
        for config in model_configurations:
            results_df, mae = inference_and_evaluate(
                best_model, data_test, feat_dic, helpers,config, 
                output_csv_path=output_csv_path, logger=logger)
            mae_list.append(mae)            
    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)
    logger.info(f'Average MAE for 5 randome seeds: {mae_mean} days.')
    logger.info(f'Standard deviation of MAE for 5 randome seeds: {mae_std} days.')

if __name__ == '__main__':
    main()    
    
    

