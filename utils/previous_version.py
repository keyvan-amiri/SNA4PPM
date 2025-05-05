# -*- coding: utf-8 -*-
"""
Created on Thu May  1 20:54:31 2025
@author: kamirel
"""
import networkx as nx
import numpy as np
import bisect
from tqdm import tqdm
import pandas as pd

def bulid_queuing_graph(df_inp, role_map, abstraction=False,
                        case_col='case:concept:name', act_col='concept:name',
                        start_col='start', end_col= 'end',
                        res_col='org:resource', role_col='Role'):
    # Define a custom activity sort key
    def activity_sort_key(activity):
        if activity.lower() == 'start':
            return -1  # Start first
        elif activity.lower() == 'end':
            return 1   # End last
        else:
            return 0   # Normal activities in between                
    df = df_inp.copy()    
    df['_activity_order'] = df[act_col].apply(activity_sort_key)
    df_sorted = df.sort_values(by=['end', '_activity_order'
                                   ]).drop(columns=['_activity_order'])
    work_graph = nx.DiGraph()
    
    for _, row in df_sorted.iterrows():
        case_id = row[case_col]
        pl = row['prefix_length']
        activity = row[act_col]
        resource = row[res_col]
        role = row[role_col]
        start_time = row['start']
        end_time = row['end']
        proc_time = row['proc_time']
        enabled_time = row['enabled_time'] if activity != 'Start' else None
        enb_act = row['enabling_activity'] if activity != 'Start' else None
        enb_res = row['enabling_resource'] if activity != 'Start' else None
        enb_rol = role_map[enb_res] if activity != 'Start' else None
        wait_time = row['wait_time'] if activity != 'Start' else 0
        if abstraction:
            trg_node = role
            src_node = enb_rol
        else:
            trg_node = resource
            src_node = enb_res
        if not trg_node in work_graph.nodes:
            # add the node
            work_graph.add_node(
                trg_node,
                data={'case_id': [case_id],'prefix_length': [pl],
                      'in_time': [start_time], 'out_time': [end_time],
                      'proc_time': [proc_time],
                      'src_act': [enb_act], 'trg_act': [activity],
                      'src_res': [enb_res], 'trg_res': [resource],
                      'src_role': [enb_rol], 'trg_role': [role]})
        else:
            # update the node
            data = work_graph.nodes[trg_node]['data']
            data['case_id'].append(case_id)
            data['prefix_length'].append(pl)
            data['in_time'].append(start_time)
            data['out_time'].append(end_time)
            data['proc_time'].append(proc_time)
            data['src_act'].append(enb_act)
            data['trg_act'].append(activity)
            data['src_res'].append(enb_res)
            data['trg_res'].append(resource)
            data['src_role'].append(enb_rol)
            data['trg_role'].append(role)
            work_graph.nodes[trg_node]['data'] = data
        if activity != 'Start':                            
            if not (src_node, trg_node) in work_graph.edges:
                # add the edge
                work_graph.add_edge(
                    src_node, trg_node,
                    data={'case_id': [case_id], 'in_time':[enabled_time],
                          'out_time': [start_time], 'wait_time': [wait_time],
                          'src_act': [enb_act], 'trg_act': [activity], 
                          'src_res': [enb_res], 'trg_res': [resource],
                          'src_role': [enb_rol], 'trg_role': [role]})
            else:
                # update the edge
                data = work_graph.edges[(src_node, trg_node)]['data']
                data['case_id'].append(case_id) 
                data['in_time'].append(enabled_time)
                data['out_time'].append(start_time)
                data['wait_time'].append(wait_time)
                data['src_act'].append(enb_act)
                data['trg_act'].append(activity)
                data['src_res'].append(enb_res)
                data['trg_res'].append(resource)
                data['src_role'].append(enb_rol)
                data['trg_role'].append(role)
                work_graph.edges[(src_node, trg_node)]['data'] = data
    return work_graph


def get_load(lst_in, lst_out, timestamp):
    # sort: end (complete lifecyle) is used for graph construction
    sorted_lst_in = sorted(lst_in)
    sorted_lst_out = sorted(lst_out)
    entered = bisect.bisect_right(sorted_lst_in, timestamp)
    exited = bisect.bisect_right(sorted_lst_out, timestamp)
    return entered - exited


def network_features (graph, node, timestamp, activity, resource,
                      queues=None, servers = None, one_hup=False,
                      window=True, window_size=10):
    timestamp = pd.to_datetime(timestamp, errors='coerce')
    if one_hup:
        queues = graph.out_edges(node)
        servers = list(graph.successors(node))        
    # handle edges (queue and waiting times)
    queue_length = {}
    wait_mean, wait_std, wait_min, wait_max = {},{},{},{} 
    for queue in queues:
        data = graph.edges[queue].get('data')
        in_time_lst = data.get('in_time')
        out_time_lst = data.get('out_time')
        in_time_lst = pd.to_datetime(in_time_lst, errors='coerce')
        out_time_lst = pd.to_datetime(out_time_lst, errors='coerce')    
        legth = get_load(
            in_time_lst, out_time_lst, timestamp)
        queue_length[queue] = legth
        index = max((i for i, t in enumerate(out_time_lst) if t <= timestamp),
                    default=-1)
        if index >= 0:
            wait_lst = data.get('wait_time')
            wait_lst = wait_lst[:index + 1]
            if one_hup:
                act_lst = data.get('src_act')
                res_lst = data.get('src_res')                
                act_lst = act_lst[:index + 1]
                res_lst = res_lst[:index + 1]                
                indices = [
                    i for i, (a, r) in enumerate(zip(act_lst, res_lst)) 
                    if a == activity and r == resource]
                if len(indices) == 0:
                    indices = [i for i, a in enumerate(act_lst) if a == activity]
                if len(indices)>0:
                    if window:
                        indices = indices[-window_size:] if len(indices) >= window_size else indices                
                    waiting_times = [wait_lst[i] for i in indices]
                else:
                    waiting_times = wait_lst[-window_size:] if len(wait_lst) >= window_size else wait_lst 
            else:
                waiting_times = wait_lst[-window_size:] if len(wait_lst) >= window_size else wait_lst 
            mean = np.mean(waiting_times)
            std = np.std(waiting_times, ddof=0)
            min_val = np.min(waiting_times)
            max_val = np.max(waiting_times)       
            wait_mean[queue] = mean
            wait_std[queue] = std
            wait_min[queue] = min_val
            wait_max[queue] = max_val            
                
    # handle nodes (workload and processing times)
    server_load = {}
    proc_mean, proc_std, proc_min, proc_max = {},{},{},{}
    for server in servers:
        data = graph.nodes[server].get('data')
        in_time_lst = data.get('in_time')
        out_time_lst = data.get('out_time')
        in_time_lst = pd.to_datetime(in_time_lst, errors='coerce')
        out_time_lst = pd.to_datetime(out_time_lst, errors='coerce')    
        load = get_load(
            in_time_lst, out_time_lst, timestamp)
        server_load[server] = load
        index = max((i for i, t in enumerate(out_time_lst) if t <= timestamp),
                    default=-1)
        if index >= 0:
            proc_lst = data.get('proc_time')
            proc_lst = proc_lst[:index + 1]
            if one_hup:        
                act_lst = data.get('src_act')
                res_lst = data.get('src_res')
                act_lst = act_lst[:index + 1]
                res_lst = res_lst[:index + 1]
                indices = [i for i, (a, r) in enumerate(zip(act_lst, res_lst)) 
                           if a == activity and r == resource]      
                if len(indices) == 0:
                    indices = [i for i, a in enumerate(act_lst) if a == activity]
                if len(indices)>0:
                    if window:
                        indices = indices[-window_size:] if len(indices) >= window_size else indices  
                    processing_times = [proc_lst[i] for i in indices]
                else:
                    processing_times = proc_lst[-window_size:] if len(proc_lst) >= window_size else proc_lst
            else:
                processing_times = proc_lst[-window_size:] if len(proc_lst) >= window_size else proc_lst
            mean = np.mean(processing_times)
            std = np.std(processing_times, ddof=0)
            min_val = np.min(processing_times)
            max_val = np.max(processing_times)       
            proc_mean[server] = mean
            proc_std[server] = std
            proc_min[server] = min_val
            proc_max[server] = max_val
            
    return (queue_length, server_load,
            wait_mean, wait_std, wait_min, wait_max,
           proc_mean, proc_std, proc_min, proc_max)


def add_inter_case_features(df, G, abstraction=True, one_hup=False,
                            act_col='concept:name', res_col='org:resource',
                            window=True, window_size=10):
    all_nodes = list(G.nodes())
    all_nodes.remove('Start')
    all_nodes.remove('End')
    all_edges = list(G.edges())
    all_edges = [t for t in all_edges if t[0] != 'Start']
    combined_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Feature Extraction"):
        time = row['end']
        activity = row[act_col]
        resource = row[res_col]
        if abstraction:
            node = row['Role']
        else:
            node = resource
        if one_hup:
            (q_length, s_load,
             w_mean, w_std, w_min, w_max,
             p_mean, p_std, p_min, p_max) = network_features(
                 G, node, time, activity, resource, one_hup=one_hup, 
                 window=window, window_size=window_size)
        else:
            (q_length, s_load,
             w_mean, w_std, w_min, w_max,
             p_mean, p_std, p_min, p_max) = network_features(
                 G, node, time, activity, resource, 
                 one_hup=one_hup, queues=all_edges, servers = all_nodes,
                 window=window, window_size=window_size)                 
        row_dict = {}
        # Node-level columns
        for node in all_nodes:
            row_dict[f"{node}_load"] = s_load.get(node, np.nan)
            row_dict[f"{node}_p_mean"] = p_mean.get(node, np.nan)
            row_dict[f"{node}_p_std"] = p_std.get(node, np.nan)
            row_dict[f"{node}_p_min"] = p_min.get(node, np.nan)
            row_dict[f"{node}_p_max"] = p_max.get(node, np.nan)
        # Edge-level columns
        for edge in all_edges:
            edge_name = f"{edge[0]}_to_{edge[1]}"
            row_dict[f"{edge_name}_queue"] = q_length.get(edge, np.nan)
            row_dict[f"{edge_name}_w_mean"] = w_mean.get(edge, np.nan)
            row_dict[f"{edge_name}_w_std"] = w_std.get(edge, np.nan)
            row_dict[f"{edge_name}_w_min"] = w_min.get(edge, np.nan)
            row_dict[f"{edge_name}_w_max"] = w_max.get(edge, np.nan)
        combined_data.append(row_dict)
    # Build and merge
    combined_df = pd.DataFrame(combined_data)
    df = pd.concat([df.reset_index(drop=True), combined_df], axis=1)
    return df
