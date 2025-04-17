# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 11:38:47 2025
@author: Keyvan Amiri Elyasi
"""
import numpy as np
import pandas as pd
import pm4py
from collections import defaultdict
from catboost import Pool
#from pm4py.objects.log.importer.xes import importer as xes_importer
#import os
#import networkx as nx
#import matplotlib.pyplot as plt
#import torch
#import random
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
#from tqdm import tqdm
#from networkx.algorithms.simple_paths import shortest_simple_paths
#from catboost import CatBoostRegressor
#from sklearn.metrics import make_scorer, mean_absolute_error
#from sklearn.model_selection import train_test_split


def validate_and_fix_start_end(df):
    df = df.copy()
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    for case_id, group in df.groupby('case:concept:name'):
        # Get min and max timestamps in this case
        min_ts = group['time:timestamp'].min()
        max_ts = group['time:timestamp'].max()

        # Fix Start
        start_mask = (group['concept:name'] == 'Start')
        if start_mask.any():
            idxs = group[start_mask].index
            df.loc[idxs, 'time:timestamp'] = min_ts

        # Fix End
        end_mask = (group['concept:name'] == 'End')
        if end_mask.any():
            idxs = group[end_mask].index
            df.loc[idxs, 'time:timestamp'] = max_ts
    return df

def add_start_end_events(df):
    df = df.copy()
    df['sort_order'] = 1  # normal rows    
    new_rows = []
    for case_id, group in df.groupby('case:concept:name'):
        min_time = group['time:timestamp'].min()
        max_time = group['time:timestamp'].max()
        # Start rows
        new_rows.extend([
            {
                'case:concept:name': case_id,
                'concept:name': 'Start',
                'time:timestamp': min_time,
                'lifecycle:transition': 'Start',
                'org:resource': 'Start',
                'sort_order': 0
            },
            {
                'case:concept:name': case_id,
                'concept:name': 'Start',
                'time:timestamp': min_time,
                'lifecycle:transition': 'complete',
                'org:resource': 'Start',
                'sort_order': 0
            }
        ])
        # End rows
        new_rows.extend([
            {
                'case:concept:name': case_id,
                'concept:name': 'End',
                'time:timestamp': max_time,
                'lifecycle:transition': 'start',
                'org:resource': 'End',
                'sort_order': 2
            },
            {
                'case:concept:name': case_id,
                'concept:name': 'End',
                'time:timestamp': max_time,
                'lifecycle:transition': 'complete',
                'org:resource': 'End',
                'sort_order': 2
            }
        ])
    # Add artificial rows
    start_end_df = pd.DataFrame(new_rows)
    full_df = pd.concat([df, start_end_df], ignore_index=True)
    # Sort by case, timestamp, and sort_order
    full_df = full_df.sort_values(
        by=['case:concept:name', 'time:timestamp', 'sort_order']
    ).reset_index(drop=True)
    # Drop helper column
    full_df = full_df.drop(columns='sort_order')
    return full_df

def get_handover_two_ts(dataset_path, inp_dict, add_start_end):
    # define important columns
    case_col = inp_dict['case_col']
    act_col = inp_dict['act_col']
    time_col = inp_dict['time_col']
    trans_col = inp_dict['trans_col']
    res_col = inp_dict['res_col']
    graph_cols = [case_col]+[act_col]+[time_col]+[trans_col]+[res_col]
    # import the log
    log = pm4py.read_xes(dataset_path)
    log = log[graph_cols]
    if add_start_end:
        df = add_start_end_events(log)
    else:
        df = validate_and_fix_start_end(log)
    # get case ids
    case_lst = df[case_col].unique().tolist()
    
    def extract_handover_info(df: pd.DataFrame, case_id: str):
        df_case = df[df[case_col] == case_id].copy() 
        #df_case[time_col] = pd.to_datetime(df_case[time_col])
        df_case[trans_col] = df_case[trans_col].str.lower()
        df_case['__trans_order'] = df_case[trans_col].map(
            {'start': 0, 'complete': 1})
        df_case = df_case.sort_values(
            by=[time_col, '__trans_order']).drop(columns='__trans_order')
        #df_case = df_case.sort_values(time_col)
        # Temporary structure to pair start and complete
        pending_starts = defaultdict(list)
        executed_tasks = []
        for _, row in df_case.iterrows():
            activity = row[act_col]
            transition = row[trans_col].lower()
            resource = row[res_col]
            timestamp = row[time_col]
            key = (activity, resource)
            if transition == 'start':
                pending_starts[key].append(timestamp)              
            elif transition == 'complete' and pending_starts[key]:
                # Find the oldest valid start time (≤ complete time)
                valid_starts = [t for t in pending_starts[key] if t <= timestamp]
                if valid_starts:
                    start_time = min(valid_starts)  # get the oldest one
                    pending_starts[key].remove(start_time)
                    executed_tasks.append({
                        'resource': resource,
                        'activity': activity,
                        'start_time': start_time,
                        'end_time': timestamp
                    }) 
                else:
                    print('no start is found! check pre-processing')
        # Now extract handovers
        handovers = []
        for i in range(1, len(executed_tasks)):
            prev = executed_tasks[i-1]
            curr = executed_tasks[i]
            if prev['activity'] == 'End' and curr['activity'] == 'End':
                continue
            if prev['end_time'] <= curr['start_time']:
                # we focus on waiting time
                handovers.append({
                    'type': 'sequential',
                    'from_resource': prev['resource'],
                    'to_resource': curr['resource'],
                    'from_end_time': prev['end_time'],
                    'to_start_time': curr['start_time']
                    })
            else:
                # we focus on synchorinization time
                handovers.append({
                    'type': 'parallel',
                    'from_resource': prev['resource'],
                    'to_resource': curr['resource'],
                    'from_end_time': prev['end_time'], 
                    'to_end_time': curr['end_time']
                    })
        return {'executions': executed_tasks, 'handovers': handovers}
    
    exe_lst = []
    hand_lst = []
    for case_id in case_lst:
        result = extract_handover_info(df, case_id=case_id)
        exe_lst.extend(result['executions'])
        hand_lst.extend(result['handovers'])
    
    exe_lst.sort(key=lambda x: x['start_time'])
    hand_lst.sort(key=lambda x: x['from_end_time'])
    return exe_lst, hand_lst    

def temporal_train_test_split(log, train_ratio=0.8):
    # Get min timestamp per case
    case_start_times = log.groupby('case:concept:name')['time:timestamp'].min()
    # Sort cases by start time
    sorted_case_ids = case_start_times.sort_values().index.tolist()
    # Compute split index
    split_index = int(len(sorted_case_ids) * train_ratio)
    # Split case IDs
    train_case_ids = sorted_case_ids[:split_index]
    test_case_ids = sorted_case_ids[split_index:]
    # Split log
    train_df = log[log['case:concept:name'].isin(train_case_ids)]
    test_df = log[log['case:concept:name'].isin(test_case_ids)]
    return train_df, test_df, train_case_ids, test_case_ids

def generate_examples(log, two_ts=True, mode='rem_time'):
    examples = []
    example_keys = []
    for case_id, case_df in log.groupby('case:concept:name'):
        if len(case_df) < 4:
            # consist of only 2 Start and 2 End activities
            print(f'check preprocessing: {case_id} has less than 4 events.')
            continue
        max_timestamp = case_df['time:timestamp'].iloc[-1]
        min_timestamp = case_df['time:timestamp'].iloc[0]
        for idx in range(1, len(case_df) - 2, 2):
            # get prefix length
            prefix_length = int((idx+1)/2)
            # for ts='complete': get activity, resource and timestamp. 
            act_name = case_df['concept:name'].iloc[idx]
            resource = case_df['org:resource'].iloc[idx]
            end_time = case_df['time:timestamp'].iloc[idx]
            since_start = (end_time - min_timestamp
                           ).total_seconds() / 3600 / 24            
            day_of_week = end_time.strftime('%A')
            hour = end_time.hour
            # define target attribute
            if mode == 'rem_time':
                rem_time = (max_timestamp - end_time).total_seconds() / 3600 / 24
                target = rem_time
            else:
                next_w_time = (case_df['time:timestamp'].iloc[idx + 1] - end_time
                              ).total_seconds() / 3600 / 24
                next_p_time = (case_df['time:timestamp'].iloc[idx + 2] - case_df['time:timestamp'].iloc[idx + 1]
                              ).total_seconds() / 3600 / 24
                target = (next_w_time, next_p_time)
            example_keys.append((case_id, prefix_length))
            examples.append({
                'act_name': act_name,
                'resource': resource,
                'end_time': end_time,
                'since_start': since_start,
                'day_of_week': day_of_week,
                'hour': hour,
                'length': prefix_length, 
                'target': target
            })
    return example_keys, examples


def prepare_catboost_data(prefix_dict, max_nodes):
    rows_task1 = []
    rows_task2 = []

    for key, (resource, act_name, day_of_week, 
              since_start, hour, length, x_n, x_e, y
             ) in prefix_dict.items():
        case_id, prefix_length = key  # unpack the tuple
        # Pad x_e and x_n to shape (max_nodes, 9)
        pad_x_e = np.zeros((max_nodes, 9))
        pad_x_e[:x_e.shape[0], :] = x_e

        pad_x_n = np.zeros((max_nodes, 9))
        pad_x_n[:x_n.shape[0], :] = x_n

        # Flatten padded arrays
        flat_x_e = pad_x_e.flatten().tolist()
        flat_x_n = pad_x_n.flatten().tolist()

        base_features = [case_id, prefix_length, resource, act_name,
                         day_of_week, since_start, hour, length]

        # Task 1: use x_e
        rows_task1.append(base_features + flat_x_e + [y[0]])
        # Task 2: use x_n
        rows_task2.append(base_features + flat_x_n + [y[1]])

    # Define column names
    base_columns = ['case_id', 'prefix_length', 'resource', 'act_name',
                    'day_of_week', 'since_start', 'hour', 'length']
    x_e_columns = [f'x_e_{i}' for i in range(max_nodes * 9)]
    x_n_columns = [f'x_n_{i}' for i in range(max_nodes * 9)]

    columns_task1 = base_columns + x_e_columns + ['target']
    columns_task2 = base_columns + x_n_columns + ['target']

    df_task1 = pd.DataFrame(rows_task1, columns=columns_task1)
    df_task2 = pd.DataFrame(rows_task2, columns=columns_task2)

    return df_task1, df_task2


def predict_next_time(model1, model2, wait_df_test, process_df_test):
    label_cols = ['case_id', 'prefix_length', 'target']
    categorical_cols = ['resource', 'act_name', 'day_of_week']

    # Task 1 (edge model for next waiting time prediction)
    X1 = wait_df_test.drop(columns=label_cols)
    #y1 = wait_df_test['target']
    pool1 = Pool(X1, cat_features=categorical_cols)
    preds1 = np.maximum(model1.predict(pool1), 0)
    result_df1 = wait_df_test[['case_id', 'prefix_length', 'target']].copy()
    result_df1['prediction'] = preds1

    # Task 2 (node model for next processing time prediction)
    X2 = process_df_test.drop(columns=label_cols)
    #y2 = process_df_test['target']
    pool2 = Pool(X2, cat_features=categorical_cols)
    preds2 = np.maximum(model2.predict(pool2), 0)
    result_df2 = process_df_test[['case_id', 'prefix_length', 'target']].copy()
    result_df2['prediction'] = preds2

    return result_df1, result_df2
