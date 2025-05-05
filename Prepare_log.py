# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 13:16:13 2025

@author: kamirel
"""
import os
import argparse
import yaml
import pickle
import pm4py
import pandas as pd
#import networkx as nx
from utils.SNA_preprocessing import (
    augment_args, basic_process, add_start_end_events, match_start_complete,
    role_discovery, split_cases, get_intra_case_features)
from utils.SNA_graph_operations import (
    create_queuing_graph, get_inter_case_features)
from enable_estimates.config import Configuration
from enable_estimates.concurrency_oracle import HeuristicsConcurrencyOracle
from enable_estimates.estimator import StartTimeEstimator
from enable_estimates.utils import EventLogIDs, read_csv_log



def main():
    # Define and get parameters and arguements
    parser = argparse.ArgumentParser(description='Pre-Processing')
    parser.add_argument('--dataset')
    parser.add_argument('--cfg', default=None)
    args = parser.parse_args()
    args.root_path = os.getcwd()
    cfg_file = args.cfg if args.cfg is not None else args.dataset + '.yaml'       
    with open(os.path.join(args.root_path, 'cfg', cfg_file) , 'r') as f:
        cfg = yaml.safe_load(f)
    args = augment_args(args, cfg)   
    # load event log  
    log = pm4py.read_xes(args.dataset_path)
    log = basic_process(log, args)  
    num_trans = len(log[args.trans_col].unique())
    if num_trans == 2:
        event_log = match_start_complete(
            log, case_col=args.case_col, act_col=args.act_col,
            res_col=args.res_col, time_col=args.time_col,
            trans_col=args.trans_col, case_features=args.case_feat,
            event_features=args.event_feat)
        event_log.to_csv(args.out_path, index=False)   
        # Set up custom configuration
        configuration = Configuration(
            log_ids=EventLogIDs(
            case=args.case_col,
            activity=args.act_col,
            start_time="start",
            end_time="end",
            resource=args.res_col
            ),
            consider_start_times=True)   
        # Read event log
        event_log = read_csv_log(
            log_path=args.out_path,
            log_ids=configuration.log_ids,
            sort=True  # Sort log by end time
            )
        concurrency_oracle = HeuristicsConcurrencyOracle(event_log, configuration)
        extended_event_log = concurrency_oracle.add_enabled_times(
            event_log,
            set_nat_to_first_event=False)
    else:
        log = log.rename(columns={args.time_col: 'end'})
        log['start'] = pd.NA
        log.to_csv(args.out_path, index=False)
        configuration = Configuration(
            log_ids=EventLogIDs(
            case=args.case_col,
            activity=args.act_col,
            start_time="start",
            end_time="end",
            resource=args.res_col
            ),
            consider_start_times=False)   
        # Read event log
        event_log = read_csv_log(
            log_path=args.out_path,
            log_ids=configuration.log_ids,
            sort=True  # Sort log by end time
            )
        # Estimate start times
        extended_event_log = StartTimeEstimator(
            event_log, configuration).estimate(replace_recorded_start_times=True)
        extended_event_log = extended_event_log.drop(
            columns=[args.trans_col, 'available_time'])
        basic_features = [
            args.case_col, args.act_col, args.res_col, 'start', 'end',
            'enabling_activity', 'enabling_resource', 'enabled_time']
        all_columns = basic_features + args.case_feat + args.event_feat        
        extended_event_log = extended_event_log[all_columns] 
    # handle start and end
    processed_log = add_start_end_events(
        extended_event_log, num_trans, case_col=args.case_col, 
        act_col=args.act_col, res_col=args.res_col)
    # get training and test ids
    _, train_case_ids, test_case_ids = split_cases(
        processed_log, case_col=args.case_col, time_col='start',
        train_ratio=args.train_ratio) 
    with open(args.train_id, 'wb') as f:
        pickle.dump(train_case_ids, f)
    with open(args.test_id, 'wb') as f:
        pickle.dump(test_case_ids, f)        
    # add processing and waiting times
    processed_log['proc_time'] = (
        processed_log['end'] - processed_log['start']).dt.total_seconds()
    processed_log['wait_time'] = (
        processed_log['start'] - processed_log['enabled_time']).dt.total_seconds()
    # add prefix length, and sort the columns
    processed_log['prefix_length'] = processed_log.groupby(args.case_col).cumcount()
    # add roles to the dataframe
    final_log, role_map = role_discovery(
        processed_log, act_col=args.act_col, res_col=args.res_col,
        rp_sim=args.rp_sim) 
    final_log['enabling_role'] = final_log['enabling_resource'].map(role_map)
    with open(args.role_map, 'wb') as f:
        pickle.dump(role_map, f)
    queuing_graph = create_queuing_graph(
        final_log, role_map, abstraction=args.abstraction,
        case_col=args.case_col, act_col=args.act_col, res_col=args.res_col,
        neighbors=args.neighbors)
    with open(args.graph_path, 'wb') as f:
        pickle.dump(queuing_graph, f)  
    #nx.write_gpickle(queuing_graph, args.graph_path)
    result_log = get_intra_case_features(
        final_log, case_col=args.case_col, act_col=args.act_col,
        act_window=args.act_window)    
    tabular_df = get_inter_case_features(result_log, queuing_graph,
                                         neighbors=args.neighbors,
                                         abstraction=args.abstraction,
                                         case_col=args.case_col,
                                         res_col=args.res_col)     
    # remove unuseful columns
    tabular_df = tabular_df.loc[:, tabular_df.nunique() > 1]
    # add Train-test label
    tabular_df = tabular_df.copy()
    tabular_df.loc[:, 'set'] = tabular_df[args.case_col].apply(
        lambda x: 'Train' if x in train_case_ids else 
        ('Test' if x in test_case_ids else None))
    cols = [args.case_col, 'prefix_length', 'set'
            ] + [col for col in tabular_df.columns if col not in 
                 [args.case_col, 'prefix_length', 'set', 
                  'rem_time', 'next_proc', 'next_wait']] + [
                      'rem_time', 'next_proc', 'next_wait']
    tabular_df = tabular_df[cols]
    tabular_df.to_csv(args.out_path, index=False) 

if __name__ == '__main__':
    main()    