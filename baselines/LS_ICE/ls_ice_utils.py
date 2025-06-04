import joblib
import logging
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences



def split_cases(df, case_col='case:concept:name', time_col='time:timestamp',
                train_ratio=0.8, val_ratio=0.2):
    log = df.copy()
    # Get min timestamp per case
    case_start_times = log.groupby(case_col)[time_col].min()
    # Sort cases by start time
    sorted_case_ids = case_start_times.sort_values().index.tolist()
    # Compute split index
    split_index2 = int(len(sorted_case_ids) * train_ratio)
    split_index1 = int(len(sorted_case_ids) * train_ratio*(1-val_ratio))
    # Split case IDs
    train_case_ids = sorted_case_ids[:split_index1]
    val_case_ids = sorted_case_ids[split_index1:split_index2]
    test_case_ids = sorted_case_ids[split_index2:]
    
    return train_case_ids, val_case_ids, test_case_ids 



def add_bos_eos_rows(df_inp, case_col='case:concept:name',
                     act_col='concept:name', time_col='time:timestamp'):
    df = df_inp.copy()
    df['_order'] = 1  # original rows get default sort priority
    groups = df.groupby(case_col)
    bos_rows = groups[time_col].min().reset_index()
    bos_rows[act_col] = '<BOS>'
    bos_rows['_order'] = 0
    eos_rows = groups[time_col].max().reset_index()
    eos_rows[act_col] = '<EOS>'
    eos_rows['_order'] = 2
    # Combine all rows
    df_all = pd.concat([df, bos_rows, eos_rows], ignore_index=True)    
    # Sort by case, time, and _order to enforce BOS before other events and EOS after
    df_all = df_all.sort_values(by=[case_col, time_col, '_order']
                                ).drop(columns='_order').reset_index(drop=True)
    df_all['duration'] = df_all.groupby(case_col)[time_col].transform(
        lambda x: (x - x.min()).dt.total_seconds())
    df_all['remtime'] = df_all.groupby(case_col)[time_col].transform(
        lambda x: (x.max() - x).dt.total_seconds())
    df_all.rename(columns={case_col: 'case_id', act_col: 'activity',
                           time_col: 'ts'}, inplace=True)
    df_all = df_all.reset_index(drop=True)
    df_all['event_id'] = df_all.index
    df_all = df_all[[
        'event_id', 'case_id', 'activity', 'ts', 'duration', 'remtime']]
    return df_all

def get_lead_ts(log, mode='candidate'):
    temp_log = log.copy()
    temp_log = temp_log.sort_values(['case_id', 'event_id'])
    temp_log['ts_next'] = temp_log.ts.shift(-1)
    temp_log.loc[temp_log['activity'] == '<EOS>', 'ts_next'] = np.nan
    if mode=='candidate':
        return log.merge(temp_log[['event_id', 'ts_next']], on='event_id')
    elif mode=='location':
        return log.merge(temp_log[['event_id', 'ts_next']], 
                         left_on='event_id', right_on='event_id')
  
def get_proc_time(merge):    
    return merge.apply(lambda x: (x.ts_next - x.ts).seconds, axis=1)

def get_throughp_act(log, act, time_unit = 'minute'):
    proc_time = get_proc_time(log)
    if time_unit == 'second':
        print(f'{act} - average proctime {round(proc_time.mean(),2)} and median proctime {round(proc_time.median(),2)} seconds')
    elif time_unit == 'minute':
        proc_time = proc_time/60
        print(f'{act} - average proctime {round(proc_time.mean(),2)} and median proctime {round(proc_time.median(),2)} minutes')
    elif time_unit == 'hour':
        proc_time = proc_time/3600
        print(f'{act} - average proctime {round(proc_time.mean(),2)} and median proctime {round(proc_time.median(),2)} hours')    
    else:
        print('time unit not available, choose second, minute, hour')    
    return proc_time

def get_throughp(log, act, time_unit = 'second'):
    proc_time = get_proc_time(log[log.activity == act]) 
    if time_unit == 'second':
        print(f'{act} - average proctime {round(proc_time.mean(),2)} and median proctime {round(proc_time.median(),2)} seconds')
    elif time_unit == 'minute':
        proc_time = proc_time/60
        print(f'{act} - average proctime {round(proc_time.mean(),2)} and median proctime {round(proc_time.median(),2)} minutes') 
    elif time_unit == 'hour':
        proc_time = proc_time/3600
        print(f'{act} - average proctime {round(proc_time.mean(),2)} and median proctime {round(proc_time.median(),2)} hours')    
    else:
        print('time unit not available, choose second, minute, hour')    
    return proc_time

def get_locations(log, thresh_time = 5, thresh_freq = 0.01, time_unit = 'minute'):    
    log = get_lead_ts(log, mode='location')
    load_locations = []
    total = len(log)    
    for load, freq in zip(log.activity.value_counts().index, log.activity.value_counts()):        
        if (load == '<EOS>') | (load == '<BOS>'):
            continue    
        dur = get_throughp(log, load, time_unit)        
        if np.mean(dur) < thresh_time:
            print('auto')        
        else:
            if freq/total < thresh_freq:
                print('unfreq')                
            else:
                print(load, f'{round(freq/total, 2)} added')
                load_locations.append(load)                
    return load_locations

def myround(x, base=5):
    return base * round(x/base)

def get_dur_range(log, act='W_Completeren aanvraag', time_unit='minutes', 
                  base=15, step=30, quantile=0.95): 
    proc_times = get_throughp_act(log[log.activity == act], act, time_unit='minute')
    print(len(proc_times))
    round_median = myround(np.median(proc_times), base)
    if round_median == 0:
        round_median = base        
    round_quantile = myround(np.quantile(proc_times, q = 0.95), base)
    if round_quantile == 0:
        round_quantile = base    
    duration_range = range(round_median, round_quantile+step, step)
    print('duration search range:', duration_range, 'length:', len(duration_range))    
    return duration_range


def get_duration_range_dic(log):
    dur_range_dic = {}
    for load in [col for col in log.activity.value_counts().index 
                 if ('<EOS>' not in col) & ('<BOS>' not in col)]:
        dur_range = get_dur_range(log, load, 'minutes', 15, 30, 0.95) 
        if dur_range != range(15, 45, 30):
            dur_range_dic[load] = dur_range
        else:
            dur_range_dic[load] = range(15, 300, 30)
    return dur_range_dic

def get_rf_relation(target_log, load, depth = 3,  threshold = 50):
    remtimes, loads = target_log.align(load, join='inner', copy=False)    
    if len(loads) > threshold:
        lr = RandomForestRegressor(max_depth=depth).fit(np.array(loads).reshape(-1,1), remtimes)
        r2 = lr.score(np.array(loads).reshape(-1,1), remtimes)    
    else:
        r2 = np.nan
    return r2

def get_location_config(log, location, threshold=50, dur_path=None):    
    all_activities = log.activity.value_counts().index
    dur_range_dic = joblib.load(dur_path)       
    duration_range = dur_range_dic[location]    
    config = {}        
    for target_activity in tqdm(all_activities):            
        if (target_activity == '<EOS>'):
            config[target_activity] = (np.nan, 0, np.nan)
            continue        
        target_log = log.loc[log.activity == target_activity, ].remtime        
        if len(target_log) <= threshold:
            config[target_activity] = (np.nan, 0, duration_range[0])    
        else:
            offset = 0
            best = None
            for diff in duration_range:
                load = log.loc[log.activity == location, 
                               ].groupby(['ts']).count().asfreq('1S').shift(
                                   1, freq=pd.DateOffset(hours=offset)
                                   ).rolling(f"{diff}min").count()['ts_col']
                relation = get_rf_relation(target_log, load, depth = 3)
                if best is None or relation > best[0]:
                    best = (relation, offset, diff)
            config[target_activity] = best
    return config


def get_configurations(log, dur_path=None, loc_path=None):    
    locations = joblib.load(loc_path)    
    configurations = {}
    for location in tqdm(locations):
        print(location)        
        configurations[location] = get_location_config(
            log, location, dur_path=dur_path)            
    return configurations

def fill_na_config(configs, na_val = 60*24):    
    for load_activity in list(configs.keys()):
        for key, val in zip(configs['{}'.format(load_activity)].keys(), configs['{}'.format(load_activity)].values()):
            if pd.isna(val[0]):
                configs['{}'.format(load_activity)][key] = (configs['{}'.format(load_activity)][key][0],
                                                            configs['{}'.format(load_activity)][key][1], (na_val))                
    return configs

def load_state_optdur(x, previous, load, configurations):
    offset = pd.DateOffset(minutes=0)
    diff = pd.DateOffset(minutes=configurations[load][x['activity']][2])
    return previous.loc[(previous >= x['ts']-diff-offset) & (previous < x['ts']-offset)].count()

def load_state_activecases(x, previous):    
    return previous.loc[(previous.ts <= x.ts) & (previous.ts_next >= x.ts)].ts.count()  


def get_load_log(log, load_log=None, configurations=None, load_state='actcase',
                 loc_path=None):
    """
    Function computes MLS-ICE features for all events in load log.
    If load_log=None functions compute MLS-ICE features for full log (log).
    Load_state determines which approach for computing the load at a single location is used, i.e. either active number of cases (actcase)
    or number of events in optimal duration (optdur). 
    """    
    if load_state == 'actcase':
        log = get_lead_ts(log, mode='location')    
    elif load_state == 'optdur':
        configurations = fill_na_config(configurations)        
    else:
        print(f'load state: {load_state}, not supported')
        return None    
    locations = joblib.load(loc_path)    
    if load_log is None:
        load_log = log.copy()
        print('computing load for full log')        
    for location in tqdm(locations):        
        if load_state =='actcase':
            previous = log.loc[log.activity == location][['ts', 'ts_next']]
            load_comp = pd.DataFrame(load_log.apply(lambda x: load_state_activecases(x, previous), axis=1))        
        elif load_state == 'optdur':
            previous = log.loc[log.activity == location, 'ts']
            load_comp = pd.DataFrame(load_log.apply(lambda x: load_state_optdur(x, previous, location, configurations), axis=1)) 
        load_log['load_{}'.format(location)] = load_comp        
    return load_log

# Custom sort function for activity
def activity_sort_key(activity):
    if activity == "<BOS>":
        return -1
    elif activity == "<EOS>":
        return 1
    else:
        return 0
    

def sort_and_add_features(group):
    # Sort by timestamp and custom activity priority
    group = group.sort_values(by=["ts", "activity"],
                              key=lambda col: col.map(activity_sort_key) if col.name == "activity" else col)
    group = group.reset_index(drop=True)    
    # Add prefix length
    group["prefix_length"] = range(len(group))    
    # Extract day and hour from timestamp
    group["day"] = group["ts"].dt.day
    group["hour"] = group["ts"].dt.hour    
    # Compute time since last event (in seconds)
    group["since_last"] = group["ts"].diff().dt.total_seconds().fillna(0)    
    return group

def min_max_normalize(df, train_case_ids, val_case_ids, test_case_ids,
                      numerical_columns):
    df['case_id'] = df['case_id'].astype(str)
    df_normalized = df.copy()    
    train_val_mask = df['case_id'].isin(train_case_ids) | df['case_id'].isin(val_case_ids)
    for col in numerical_columns:
        col_min = df.loc[train_val_mask, col].min()
        col_max = df.loc[train_val_mask, col].max()
        if col_max != col_min:
            df_normalized.loc[:, col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df_normalized.loc[:, col] = 0.0  # Avoid division by zero
    return df_normalized


def prepare_lstm_data(df_inp, train_case_ids, val_case_ids, test_case_ids,
                      enc_type, numerical_columns, logger=None):
    """
    Prepares sequential data for LSTM from event log dataframe.

    Parameters:
    - df: pandas DataFrame with columns: case_id, prefix_length, remtime, activity, and numerical features.
    - train_case_ids, val_case_ids, test_case_ids: Lists of case IDs for train, validation, and test.
    - enc_type: Encoding type for activity column: 'integer' or 'one_hot'.
    - numerical_columns: List of numerical column names to include in the input.

    Returns:
    - X_train, y_train: Padded input sequences and targets for training.
    - X_val, y_val: Padded input sequences and targets for validation.
    - X_test, y_test: Padded input sequences and targets for test.
    - activity_encoder: Fitted encoder for the activity column.
    - test_example_keys: List of tuples (case_id, prefix_length) for each test example.
    """
    df = df_inp.copy()    
    # Encode 'activity' column
    if enc_type == 'integer':
        activity_encoder = LabelEncoder()
        df['activity_enc'] = activity_encoder.fit_transform(df['activity'])
    elif enc_type == 'one_hot':
        activity_encoder = OneHotEncoder(sparse_output=False,
                                         handle_unknown='ignore')
        activity_encoder.fit(df[['activity']])
        df['activity_enc'] = list(activity_encoder.transform(df[['activity']]))
    else:
        raise ValueError("Unsupported enc_type. Use 'integer' or 'one_hot'.")
    # Determine maximum prefix length from training and validation (for padding)
    max_prefix_len = int(
        df[df['case_id'].isin(
            train_case_ids + val_case_ids)]['prefix_length'].max())
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    test_example_keys = []
    for case_id, group in df.groupby('case_id'):
        group = group.sort_values('prefix_length')
        group = group.reset_index(drop=True)      
        # For each prefix of length 1 to n-1 (exclude last event)
        for i in range(1, len(group)-1):
            prefix = group.iloc[:i] #get the prefix
            target = group.iloc[i]['remtime'] # target attribute
            # numerical  and categorical (integer/one-hot) features
            num_seq = prefix[numerical_columns].values             
            if enc_type == 'integer':
                act_seq = prefix['activity_enc'].values.reshape(-1, 1)
                if len(num_seq.shape) == 1: # make sure of correct shape (i=1)
                    num_seq = num_seq.reshape(-1, 1)
            else:
                act_seq = np.vstack(prefix['activity_enc'].values) 
            full_seq = np.hstack([act_seq, num_seq])
            if case_id in train_case_ids:
                X_train.append(full_seq)
                y_train.append(target)
            elif case_id in val_case_ids:
                X_val.append(full_seq)
                y_val.append(target)
            elif case_id in test_case_ids:
                X_test.append(full_seq)
                y_test.append(target)
                test_example_keys.append((case_id, i))  # i is prefix_length
                
    # Pad all sequences to the same length
    def pad(X):
        return pad_sequences(X, maxlen=max_prefix_len, dtype='float32', padding='pre')

    return (
        pad(X_train), np.array(y_train),
        pad(X_val), np.array(y_val),
        pad(X_test), np.array(y_test),
        activity_encoder,
        test_example_keys
    )
