# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 18:44:19 2025
@author: Keyvan Amiri Elyasi
"""
import networkx as nx
import numpy as np
import bisect
from tqdm import tqdm
import pandas as pd


def get_k_nearest_distances(time_lst, case_id_lst, pl_lst, k):
    distance_dict = dict()
    n = len(time_lst)
    for i in range(n):
        current_time = time_lst[i]
        key = (case_id_lst[i], pl_lst[i])
        distances = []
        for j in range(1, k + 1):
            idx = i - j
            if idx >= 0:
                delta = (current_time - time_lst[idx]).total_seconds()
                distances.append(delta)
            else:
                distances.append(-1)  # padding if not enough neighbors
        distance_dict[key] = distances
    return distance_dict


def create_queuing_graph(df_inp, role_map, abstraction=False,
                        case_col='case:concept:name', act_col='concept:name',
                        start_col='start', end_col= 'end',
                        res_col='org:resource', role_col='Role',
                        neighbors=None):
    
    # Define a custom activity sort key
    def activity_sort_key(activity):
        if activity.lower() == 'start':
            return -1  # Start first
        elif activity.lower() == 'end':
            return 1   # End last
        else:
            return 0   # Normal activities in between
    
    if abstraction:
        node_column = role_col
        src_column = 'enabling_role'
    else:
        node_column = res_col 
        src_column = 'enabling_resource'
    
    # create an empty directed graph          
    work_graph = nx.DiGraph()
    # processing nodes
    df = df_inp.copy() 
    df['_activity_order'] = df[act_col].apply(activity_sort_key)
    df_sorted = df.sort_values(by=['start', '_activity_order'
                                   ]).drop(columns=['_activity_order'])    
    node_names = df_sorted[node_column].unique()
    case_id_dict = df_sorted.groupby(node_column
                                     )[case_col].apply(list).to_dict()
    pl_dict = df_sorted.groupby(node_column
                                )['prefix_length'].apply(list).to_dict()
    in_time_dict = df_sorted.groupby(node_column)['start'].apply(list).to_dict()
    out_time_dict = df_sorted.groupby(node_column)['end'].apply(list).to_dict()
    for node in node_names:
        case_id_lst = case_id_dict[node]
        pl_lst = pl_dict[node]
        in_time_lst = in_time_dict[node]
        out_time_lst = out_time_dict[node]
        timestamps = sorted(set(in_time_lst + out_time_lst))
        in_arr = np.searchsorted(in_time_lst, timestamps, side='right')
        out_arr = np.searchsorted(out_time_lst, timestamps, side='right')
        workload = in_arr - out_arr
        workload_dict = dict(zip(timestamps, workload))
        inp_dist_dict = get_k_nearest_distances(
            in_time_lst, case_id_lst, pl_lst, neighbors)
        out_dist_dict = get_k_nearest_distances(
            out_time_lst, case_id_lst, pl_lst, neighbors)
        work_graph.add_node(
            node,
            data={'case_id': case_id_lst,'prefix_length': pl_lst,
                  'in_time': in_time_lst, 'out_time': out_time_lst,
                  'workload': workload_dict, 'inp_dist':inp_dist_dict,
                  'out_dist':out_dist_dict})
    # get sorted list of timestamps for workload_dict in each node
    for node in work_graph.nodes:
        work_graph.nodes[node]['data']['sorted_timestamps'] = sorted(
            work_graph.nodes[node]['data']['workload'].keys())
    # processing edges    
    df = df_inp.copy()  
    df = df[df[act_col]!= 'Start'] 
    df['_activity_order'] = df[act_col].apply(activity_sort_key)
    df_sorted = df.sort_values(by=['enabled_time', '_activity_order'
                                   ]).drop(columns=['_activity_order'])     
    edge_names = list(map(
        tuple, df_sorted[[src_column, node_column]].drop_duplicates().values))
    case_id_dict = df_sorted.groupby([src_column, node_column]
                                     )[case_col].apply(list).to_dict()
    pl_dict = df_sorted.groupby([src_column, node_column]
                                )['prefix_length'].apply(list).to_dict()
    in_time_dict = df_sorted.groupby([src_column, node_column]
                                     )['enabled_time'].apply(list).to_dict()
    out_time_dict = df_sorted.groupby([src_column, node_column]
                                      )['start'].apply(list).to_dict()
    for edge in edge_names:
        src_node = edge[0]
        trg_node = edge[1]
        case_id_lst = case_id_dict[edge]
        pl_lst = pl_dict[edge]
        in_time_lst = in_time_dict[edge]
        out_time_lst = out_time_dict[edge]
        timestamps = sorted(set(in_time_lst + out_time_lst))
        in_arr = np.searchsorted(in_time_lst, timestamps, side='right')
        out_arr = np.searchsorted(out_time_lst, timestamps, side='right')
        queue = in_arr - out_arr
        queue_dict = dict(zip(timestamps, queue)) 
        inp_dist_dict = get_k_nearest_distances(
            in_time_lst, case_id_lst, pl_lst, neighbors)
        out_dist_dict = get_k_nearest_distances(
            out_time_lst, case_id_lst, pl_lst, neighbors)
        work_graph.add_edge(
            src_node, trg_node,
            data={'case_id': case_id_lst, 'prefix_length':pl_lst,
                  'in_time': in_time_lst, 'out_time': out_time_lst,
                  'queue': queue_dict, 'inp_dist':inp_dist_dict,
                  'out_dist':out_dist_dict})
    # Although workload dictionary is already sorted, we make sure!
    for edge in work_graph.edges:
        work_graph.edges[edge]['data']['sorted_timestamps'] = sorted(
                work_graph.edges[edge]['data']['queue'].keys())
    return work_graph      
    

def get_workload(G, node, ref_time):
    workload_dict = G.nodes[node]['data']['workload']
    timestamps = G.nodes[node]['data']['sorted_timestamps']
    idx = bisect.bisect_right(timestamps, ref_time) - 1
    workload = workload_dict[timestamps[idx]] if idx >= 0 else None
    return workload

def get_queue(G, edge, ref_time):
    queue_dict = G.edges[edge]['data']['queue']
    timestamps = G.edges[edge]['data']['sorted_timestamps']
    idx = bisect.bisect_right(timestamps, ref_time) - 1
    queue = queue_dict[timestamps[idx]] if idx >= 0 else None
    return queue

def get_distances(G, step, case_id, pl, node_query=True, key='inp_dist'):
    if node_query:
        distances = G.nodes[step]['data'][key][(case_id, pl)]
    else:
        distances = G.edges[step]['data'][key][(case_id, pl)]
    return distances

    
def add_distance_columns(df, G, k, abstraction=False,
                         case_col='case:concept:name', pl_col='prefix_length',
                         res_col='org:resource', role_col='Role',
                         key='inp_dist', node_query=True):
    if abstraction:
        node_column = role_col
        src_column = 'enabling_role'
    else:
        node_column = res_col 
        src_column = 'enabling_resource'
    if node_query:
        key_str = 'node_'+str(key)
    else:
        key_str = 'edge_'+str(key)       
    def extract_distances(row):
        case_id = row[case_col]
        pl = row[pl_col]
        if node_query:
            step = row[node_column]
        else:
            step = (row[src_column], row[node_column])
        distances = get_distances(
            G, step, case_id, pl, node_query=node_query, key=key)
        return distances[:k]  # ensure only k elements

    # Apply the function row-wise and convert result to a DataFrame
    distance_df = df.apply(extract_distances, axis=1, result_type='expand')
    # Rename columns to dist_1, dist_2, ..., dist_k
    distance_df.columns = [f"{key_str}_{i+1}" for i in range(k)]
    # Concatenate original DataFrame and new distance columns
    return pd.concat([df, distance_df], axis=1)


def get_inter_case_features(df, G, neighbors=None, abstraction=False,
                            case_col='case:concept:name', 
                            res_col='org:resource', role_col='Role'):
    all_nodes = list(G.nodes())
    all_nodes.remove('Start')
    all_nodes.remove('End')
    all_edges = list(G.edges())
    all_edges = [t for t in all_edges if t[0] != 'Start']
    # Apply workload computation for each node
    for node in tqdm(all_nodes, desc="Get workload of nodes"):
        df[f"{node}_load"] = df['end'].apply(lambda t: get_workload(G, node, t))
    # Apply queue computation for each edge
    for edge in tqdm(all_edges, desc="Get queue of edges"):
        src_node = edge[0]
        trg_node = edge[1]
        df[f"{src_node}_to_{trg_node}_queue"] = df['end'].apply(
            lambda t: get_queue(G, edge, t))
    df = add_distance_columns(df, G, neighbors, abstraction=abstraction, 
                              case_col=case_col, pl_col='prefix_length', 
                              res_col=res_col, role_col=role_col, 
                              key='inp_dist', node_query=True)
    df = add_distance_columns(df, G, neighbors, abstraction=abstraction, 
                              case_col=case_col, pl_col='prefix_length', 
                              res_col=res_col, role_col=role_col, 
                              key='out_dist', node_query=True)
    df = add_distance_columns(df, G, neighbors, abstraction=abstraction, 
                              case_col=case_col, pl_col='prefix_length', 
                              res_col=res_col, role_col=role_col, 
                              key='inp_dist', node_query=False)
    df = add_distance_columns(df, G, neighbors, abstraction=abstraction, 
                              case_col=case_col, pl_col='prefix_length', 
                              res_col=res_col, role_col=role_col, 
                              key='out_dist', node_query=False)
    
    return df