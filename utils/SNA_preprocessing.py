# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 13:18:59 2025
@author: Keyvan Amiri Elyasi
"""
import os
import pandas as pd
from utils.SNA_role_discovery import ResourcePoolAnalyser
from collections import defaultdict, deque


def augment_args(args, cfg):
    args.abstraction = cfg['graph']['abstraction']
    args.rp_sim = cfg['graph']['rp_sim']
    #args.one_hup = cfg['graph']['one_hup']
    args.neighbors = cfg['graph']['neighbors']
    args.case_col = cfg['features']['case_col']
    args.act_col = cfg['features']['act_col']
    args.time_col = cfg['features']['time_col']
    if args.dataset in ['BPIC17']:
        args.time_format_inference =True
    else:
        args.time_format_inference =False
    args.trans_col = cfg['features']['trans_col']
    args.res_col = cfg['features']['res_col'] 
    case_features = cfg['features']['case_feat'] 
    full_feature_names = []
    for feature in case_features:
        full_feature_names.append('case:'+feature)
    args.case_feat = full_feature_names
    args.event_feat = cfg['features']['event_feat'] 
    args.act_window = cfg['features']['act_window'] 
    args.train_ratio = cfg['split']['train_ratio']     
    dataset_file = args.dataset + '.xes'
    args.dataset_path = os.path.join(args.root_path, 'data', dataset_file)
    args.processed_data_dir = os.path.join(
        args.root_path, 'data', 'processed', args.dataset)    
    if not os.path.exists(args.processed_data_dir):
        os.makedirs(args.processed_data_dir)        
    output_name = args.dataset + '_two_TS.csv'
    if args.abstraction:
        graph_name = args.dataset + '_queuing_graph_abstract.gpickle'
    else:
        graph_name = args.dataset + '_queuing_graph.gpickle'
    role_map_name = args.dataset + '_roles.pkl'
    train_id_name = args.dataset + '_train_ids.pkl'
    test_id_name = args.dataset + '_test_ids.pkl'   
    args.out_path = os.path.join(args.processed_data_dir, output_name)
    args.graph_path = os.path.join(args.processed_data_dir, graph_name)
    args.role_map = os.path.join(args.processed_data_dir, role_map_name)
    args.train_id = os.path.join(args.processed_data_dir, train_id_name)
    args.test_id = os.path.join(args.processed_data_dir, test_id_name)
    args.columns = ([args.case_col]+[args.act_col]+[args.time_col]+
                    [args.trans_col]+[args.res_col]+
                    args.case_feat+args.event_feat)    
    return args

def basic_process(df_inp, args):
    df = df_inp.copy()
    # remove Start and End activities (if any)
    df = df[df[args.act_col] != 'Start']
    df = df[df[args.act_col] != 'End']
    if args.dataset in ['BPIC20DD', 'BPIC20ID', 'BPIC20PTC']:
        df['org:role'].replace('MISSING', 'UNDEFINED', inplace=True)
        if args.dataset == 'BPIC20PTC':
            df[args.trans_col] = 'complete'
    elif args.dataset in ['BPIC17']:
        allowed_values = ['start', 'complete']
        df = df[df[args.trans_col].isin(allowed_values)]
    if (args.dataset in ['BPIC15_1', 'BPIC15_2', 'BPIC15_3', 'BPIC15_4',
                         'BPIC15_5', 'HelpDesk'] and not args.abstraction):
        if args.dataset != 'HelpDesk':
            threshold = 1.0 # percent
        else:
            threshold = 2.0 # percent
        # replace infrequent resources with OTHERS
        rel_freq = df[args.res_col].value_counts(normalize=True).mul(100)        
        df[args.res_col] = df[args.res_col].apply(
            lambda x: x if rel_freq[x] >= threshold else 'OTHERS')
    # remove unnecessary columns
    df = df[args.columns]
    return df


def match_start_complete(
        df, case_col='case:concept:name', act_col='concept:name', 
        res_col='org:resource', time_col='time:timestamp',
        trans_col='lifecycle:transition', case_features=None,
        event_features=None):    
    
    # Define a custom transition sort key
    def transition_sort_key(transition):
        if transition.lower() == 'start':
            return 0
        elif transition.lower() == 'complete':
            return 1
        
    df = df.copy()
    df['_transition_order'] = df[trans_col].apply(transition_sort_key)    
    # Full sort order
    df = df.sort_values(
        by=[case_col, time_col, '_transition_order']
    ).reset_index(drop=True)    
    df = df.drop(columns=['_transition_order'])
    result_rows = []    
    # Group by case
    for case_id, case_df in df.groupby(case_col):
        pending_starts = {}        
        for idx, row in case_df.iterrows():
            key = (row[act_col], row[res_col])            
            if row[trans_col].lower() == 'start':
                if key not in pending_starts:
                    pending_starts[key] = deque()
                pending_starts[key].append(row)            
            elif row[trans_col].lower() == 'complete':
                if key in pending_starts and pending_starts[key]:
                    start_row = pending_starts[key].popleft()                    
                    combined_row = {
                        case_col: case_id,
                        act_col: row[act_col],
                        res_col: row[res_col],
                        'start': start_row[time_col],
                        'end': row[time_col],
                    }
                    # add case features
                    for feature in case_features:
                        combined_row[feature] = row[feature]
                    # add event features for the last feature
                    for feature in event_features:
                        combined_row[feature] = row[feature]
                    result_rows.append(combined_row)
                else:
                    # No matching start found
                    pass
    result_df = pd.DataFrame(result_rows)    
    return result_df


def add_start_end_events(
        df_inp, num_trans, case_col='case:concept:name', act_col='concept:name',
        start_time_col='start', end_time_col='end', res_col='org:resource'):    
    # Define a custom activity sort key
    def activity_sort_key(activity):
        if activity.lower() == 'start':
            return -1  # Start first
        elif activity.lower() == 'end':
            return 1   # End last
        else:
            return 0   # Normal activities in between         
    df = df_inp.copy() 
    df['enabling_activity'] = df['enabling_activity'].fillna('Start')
    df['enabling_resource'] = df['enabling_resource'].fillna('Start')
    df['enabled_time'] = df['enabled_time'].fillna(df['start'])
    new_rows = []
    for case_id, group in df.groupby(case_col):
        min_time = group[start_time_col].min()
        max_time = group[end_time_col].max()
        last_act = group.loc[group[end_time_col].idxmax(), act_col]  
        last_res = group.loc[group[end_time_col].idxmax(), res_col]
        new_rows.extend([{case_col: case_id, act_col: 'Start', res_col: 'Start',
                          start_time_col: min_time, end_time_col: min_time,
                          'enabling_activity': pd.NA, 'enabling_resource': pd.NA,
                          'enabled_time': min_time}])
        new_rows.extend([{case_col: case_id, act_col: 'End', res_col: 'End',
                          start_time_col: max_time, end_time_col: max_time,
                          'enabling_activity': last_act, 'enabling_resource': last_res,
                          'enabled_time': max_time}])       
    # Add artificial rows
    start_end_df = pd.DataFrame(new_rows)
    full_df = pd.concat([df, start_end_df], ignore_index=True)
    full_df['_activity_order'] = full_df[act_col].apply(activity_sort_key)
    # Full sort order
    full_df = full_df.sort_values(
        by=[case_col, '_activity_order', end_time_col]
        ).reset_index(drop=True)
    full_df = full_df.drop(columns=['_activity_order'])
    return full_df


def role_discovery(df, rp_sim=None,
                   act_col='concept:name', res_col='org:resource'):
    
    log_copy = df.copy()
    log_copy.rename(columns={act_col: 'task', res_col: 'user'}, inplace=True)   
    res_analyzer = ResourcePoolAnalyser(log_copy, sim_threshold=rp_sim)
    resources = pd.DataFrame.from_records(res_analyzer.resource_table)
    role_map = dict(resources[['resource', 'role']].values)
    role_map.update({'Start': 'Start', 'End': 'End'})
    #print(role_map)
    log_copy['Role'] = log_copy['user'].map(role_map)
    log_copy.rename(columns={'task': act_col, 'user': res_col}, inplace=True) 
    return log_copy, role_map


def split_cases(df, case_col='case:concept:name', time_col='start',
                train_ratio=0.8):
    log = df.copy()
    # Get min timestamp per case
    case_start_times = log.groupby(case_col)[time_col].min()
    # Sort cases by start time
    sorted_case_ids = case_start_times.sort_values().index.tolist()
    # Compute split index
    split_index = int(len(sorted_case_ids) * train_ratio)
    # Split case IDs
    train_case_ids = sorted_case_ids[:split_index]
    test_case_ids = sorted_case_ids[split_index:]
    return sorted_case_ids, train_case_ids, test_case_ids


def get_intra_case_features(df, case_col='case:concept:name',
                            act_col='concept:name', act_window=5):
    df = df[df[act_col]!='Start']
    #time_cols = ['start', 'end', 'enabled_time']
    #df[time_cols] = df[time_cols].apply(pd.to_datetime)
    # Sort to ensure proper order within groups
    df = df.sort_values([case_col, 'prefix_length']).reset_index(drop=True)
    # Initialize new columns to get all target attributes
    df['rem_time'] = None
    df['next_proc'] = None
    df['next_wait'] = None
    # Group by case
    for case_id, group in df.groupby(case_col):
        max_end = group['end'].max()
        # For quick lookup
        group_idx = group.index
        for idx in group_idx:
            row = df.loc[idx]
            prefix_now = row['prefix_length']        
            # rem_time: max_end - current end
            df.at[idx, 'rem_time'] = (max_end - row['end']).total_seconds()
            # Find next row (prefix + 1)
            next_row = group[(group['prefix_length'] == prefix_now + 1)]
            if not next_row.empty:
                next_row = next_row.iloc[0]
                df.at[idx, 'next_proc'] = (
                    next_row['end'] - next_row['start']).total_seconds()
                df.at[idx, 'next_wait'] = (
                    next_row['start'] - next_row['enabled_time']).total_seconds()
    df[['rem_time', 'next_proc', 'next_wait'
        ]] = df[['rem_time', 'next_proc', 'next_wait']].astype(float)
    df = df[df[act_col]!='End']
    # For each timestamp column
    for col in ['start', 'end', 'enabled_time']:
        # Day name (e.g., Monday, Tuesday)
        df[f'{col}_day'] = df[col].dt.day_name()    
        # Hour shifted so that 0 = 8:00 AM
        df[f'{col}_hour'] = (df[col].dt.hour - 8) % 24
    df['agg_proc'] = df.groupby(case_col)['proc_time'].cumsum()
    df['agg_wait'] = df.groupby(case_col)['wait_time'].cumsum()
    df['since_start'] = df['agg_proc']+ df['agg_wait']
    df['proc_wait_ratio'] = df['agg_proc']/df['since_start']
    # get last act_window activities as separate features
    for i in range(1, act_window):
        df[f'act_{i}'] = df.groupby(case_col)[act_col].shift(i) 
    return df


def get_case_lengths(df, case_col='case:concept:name'):
    event_log = df.copy()
    case_lengths = event_log.groupby(case_col).size().to_dict()
    return case_lengths