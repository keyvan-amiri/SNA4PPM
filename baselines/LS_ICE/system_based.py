import os
import argparse
import pm4py
from pm4py.objects.log.util import dataframe_utils
import joblib
import numpy as np

from ls_ice_utils import (
    split_cases, add_bos_eos_rows, get_lead_ts, get_duration_range_dic,
    get_locations, get_configurations, get_load_log, sort_and_add_features)



def main():
    parser = argparse.ArgumentParser(description='LS_ICE')
    parser.add_argument('--dataset')
    parser.add_argument('--load_state', default='actcase')   
    args = parser.parse_args()
    log_path = os.path.join(os.path.dirname(os.getcwd()),
                            'GraphGPS', 'PGTNet', 'raw_dataset',
                            args.dataset+'.xes')
    log = pm4py.read_xes(log_path)
    log = log[['case:concept:name', 'concept:name', 'time:timestamp']]    
    # train-val-test case ids
    train_case_ids, val_case_ids, test_case_ids  = split_cases(log)
    df = add_bos_eos_rows(log)
    out_dir = os.path.join(os.getcwd(), 'processed', args.dataset)
    out_path = os.path.join(out_dir, args.dataset+'.csv')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)         
    df.to_csv(out_path, index=False)
    #need to add this for some reason - otherwise search doesnt work
    df['ts'] = df['ts'].dt.floor('s')
    #df.ts = df.ts.apply(lambda x: x[:-4]) 
    df = dataframe_utils.convert_timestamp_columns_in_df(df)
    df_copy = df.copy()
    df_copy = get_lead_ts(df_copy)
    duration_dic = get_duration_range_dic(df_copy)
    duration_dic['<EOS>'] = range(15, 300, 30)
    duration_dic['<BOS>'] = range(15, 300, 30)
    dur_dict_path = os.path.join(out_dir, "dur_range_dic.pickle")
    joblib.dump(duration_dic, dur_dict_path) 
    df_copy = df.copy()
    load_locations = get_locations(df_copy)
    load_location_path = os.path.join(out_dir, "load_locations.pickle")
    joblib.dump(load_locations, load_location_path)
    if args.load_state == 'actcase':
        configurations = {}
    else:
        df_copy = df.copy()
        df_copy = df_copy.sort_values('ts')
        df_copy.set_index(df_copy.ts, inplace=True)
        df_copy = df_copy.rename(columns={'ts':'ts_col'})
        configurations = get_configurations(df_copy, dur_path=dur_dict_path,
                                            loc_path=load_location_path)
    df_copy = df.copy()
    df_copy = df_copy.sort_values('ts')
    df_copy.set_index(df_copy.ts, inplace=True)
    #load_log = df_copy[df_copy.case_id.isin(list(np.random.choice(df_copy.case_id.unique(), size=1, replace=False )))]
    load_log = None
    load_log = get_load_log(
        log=df_copy, load_log=load_log, configurations=configurations,
        load_state=args.load_state, loc_path=load_location_path)
    load_log = load_log.groupby("case_id", group_keys=False).apply(sort_and_add_features)
    load_log.drop(columns=['event_id', 'ts', 'ts_next'], inplace=True)
    load_log = load_log[load_log['activity'] != "<BOS>"]
    load_log = load_log[load_log['activity'] != "<EOS>"]  
    
    cols = ['case_id', 'prefix_length', 'activity'] + [
        col for col in load_log.columns if col not in 
        ['case_id', 'prefix_length', 'activity', 'remtime']] + ['remtime']
    load_log = load_log[cols]    
    extended_df_path = os.path.join(out_dir, args.dataset+'_extended_.csv')
    load_log.to_csv(extended_df_path, index=False)
    train_id_path = os.path.join(out_dir, "train_ids.pickle")
    val_id_path = os.path.join(out_dir, "val_ids.pickle")
    test_id_path = os.path.join(out_dir, "test_ids.pickle")
    joblib.dump(train_case_ids, train_id_path)
    joblib.dump(val_case_ids, val_id_path)
    joblib.dump(test_case_ids, test_id_path)  

if __name__ == '__main__':
    main()  
    