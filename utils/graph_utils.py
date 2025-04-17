# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 11:45:26 2025
@author: Keyvan Amiri Elyasi
"""
import networkx as nx
import numpy as np
from tqdm import tqdm
from collections import deque
#import os
#import pandas as pd
#from pm4py.objects.log.importer.xes import importer as xes_importer
#import pm4py
#import matplotlib.pyplot as plt
#import torch
#import random
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
#from networkx.algorithms.simple_paths import shortest_simple_paths
#from catboost import CatBoostRegressor, Pool
#from sklearn.metrics import make_scorer, mean_absolute_error
#from sklearn.model_selection import train_test_split

def creat_work_graph(exe_lst, hand_lst):
    work_graph = nx.DiGraph()
    # add nodes to the graph
    for exe in exe_lst:
        res = exe['resource']
        in_time = exe['start_time']
        out_time = exe['end_time']
        if not res in work_graph.nodes:
            # add the node
            work_graph.add_node(res, data={'in_time':[in_time], 'out_time': [out_time]})
        else:
            # update the node
            data = work_graph.nodes[res]['data']
            data['in_time'].append(in_time) 
            data['out_time'].append(out_time)
            work_graph.nodes[res]['data'] = data
    # add edges to the graph
    for hand in hand_lst:
        if hand['type'] == 'sequential':
            src_node = hand['from_resource']
            trg_node = hand['to_resource']
            in_time = hand['from_end_time']
            out_time = hand['to_start_time']
        else:
            time1 = hand['from_end_time']
            time2 = hand['to_end_time']
            if time1 < time2:
                in_time = time1
                out_time = time2
                src_node = hand['from_resource']
                trg_node = hand['to_resource']
            elif time1 > time2:
                in_time = time2
                out_time = time1
                src_node = hand['to_resource']
                trg_node = hand['from_resource']
            else:
                in_time = time1
                out_time = time2
                if hand['from_resource'] == 'Start' or hand['to_resource'] == 'End':
                    src_node = hand['from_resource']
                    trg_node = hand['to_resource']
                elif hand['from_resource'] == 'End' or hand['to_resource'] == 'Start':
                    src_node = hand['to_resource']
                    trg_node = hand['from_resource']
                else:
                    src_node = hand['from_resource']
                    trg_node = hand['to_resource']
        if src_node == 'End':
            print(hand)
        if not (src_node, trg_node) in work_graph.edges:
            # add the edge
            work_graph.add_edge(src_node, trg_node, data={'in_time':[in_time], 'out_time': [out_time]})
        else:
            # update the edge
            data = work_graph.edges[(src_node, trg_node)]['data']
            data['in_time'].append(in_time) 
            data['out_time'].append(out_time)
            work_graph.edges[(src_node, trg_node)]['data'] = data 
    return work_graph


# window and window size are given in the configuration.
def sync_timestamp_lists(list1, list2, timestamp, window=False, window_size=None):
    # Filter both lists (only information available at the timestamp: using all previous event)
    filtered1 = [t for t in list1 if t <= timestamp]
    filtered2 = [t for t in list2 if t <= timestamp]
    min_len = min(len(filtered1), len(filtered2))    
    if window and min_len > window_size:
        lst1, lst2 = filtered1[-window_size:], filtered2[-window_size:]
    else:
        lst1, lst2 = filtered1[:min_len], filtered2[:min_len]   
    return lst1, lst2


def paths_up_to_length_s(G, source, target, s):
    result_paths = []
    queue = deque([(source, [source])])

    while queue:
        node, path = queue.popleft()

        if len(path) > s + 1:
            continue
        if node == target:
            result_paths.append(path)
            continue

        for succ in G.successors(node):
            if succ not in path:  # avoid cycles
                queue.append((succ, path + [succ]))
    return result_paths


def subgraph_from_paths(G, paths):
    # extract a subgraph based on secified paths.
    nodes_in_paths = set()
    edges_in_paths = set()
    for path in paths:
        nodes_in_paths.update(path)
        edges_in_paths.update((path[i], path[i+1]) for i in range(len(path) - 1))
    H = nx.DiGraph()  # since you're using a directed graph
    # Copy node attributes
    for node in nodes_in_paths:
        H.add_node(node, **G.nodes[node])
    # Copy edge attributes
    for u, v in edges_in_paths:
        H.add_edge(u, v, **G.edges[u, v])
    return H


def get_local_graph(G, node, mode="rem_time", length=3):
    """
    Extracts a local subgraph for a given node in two modes:    
    - 'next_time': includes the node and its direct successors.
    - 'rem_time': includes all nodes within any path of length euqal or 
                less than length from the node to 'End'.    
    """
    if mode == "next_time":
        successors = list(G.successors(node))
        subgraph = nx.DiGraph()        
        # Add the central node with attributes
        subgraph.add_node(node, **G.nodes[node])        
        for succ in successors:
            subgraph.add_node(succ, **G.nodes[succ])
            if G.has_edge(node, succ):
                subgraph.add_edge(node, succ, **G.edges[node, succ])
    elif mode == "rem_time":        
        all_paths = paths_up_to_length_s(G, node, 'End', length)
        subgraph = subgraph_from_paths(G, all_paths)
    else:
        raise ValueError("Mode must be next_time or rem_time")    
    return subgraph

def single_workload(G, container, timestamp, container_type='node'):
    if container_type=='node':
        in_time_lst = G.nodes[container]['data']['in_time']
        out_time_lst = G.nodes[container]['data']['out_time']
    elif container_type=='edge':
        in_time_lst = G.edges[container]['data']['in_time']
        out_time_lst = G.edges[container]['data']['out_time']        
    in_time = np.array(in_time_lst)
    out_time = np.array(out_time_lst)
    # A case is in the container if it entered at or before timestamp and left after timestamp
    active = (in_time <= timestamp) & (out_time > timestamp)
    return np.sum(active)

def remove_isolated_nodes(G):
    # Find nodes with both in-degree and out-degree equal to zero
    isolated_nodes = [node for node in G.nodes if G.in_degree(node) == 0 and G.out_degree(node) == 0]
    # Remove isolated nodes
    G.remove_nodes_from(isolated_nodes)    
    return G


def get_sub_graph (work_graph, resource, end_time, mode= "rem_time", length=3,
                   window=False, window_size=None):
    sub_graph = get_local_graph(work_graph, resource, mode=mode, length=length)
    nodes_to_remove = []
    for node in sub_graph.nodes:
        all_in_times = sub_graph.nodes[node]['data']['in_time']
        all_out_times = sub_graph.nodes[node]['data']['out_time']
        in_time_lst, out_time_lst = sync_timestamp_lists(
            all_in_times, all_out_times, end_time, 
            window=window, window_size=window_size)
        durations = [
            (t2 - t1).total_seconds() 
            for t1, t2 in zip(in_time_lst, out_time_lst)]
        durations = np.array(durations)
        if len(durations) == 0:
            nodes_to_remove.append(node)
            continue  # skip to next node
        sub_graph.nodes[node]['dur_count'] = len(durations)
        sub_graph.nodes[node]['dur_mean'] = durations.mean()/3600/24
        sub_graph.nodes[node]['dur_std'] = durations.std()/3600/24
        sub_graph.nodes[node]['dur_min'] = durations.min()/3600/24
        sub_graph.nodes[node]['dur_max'] = durations.max()/3600/24
        sub_graph.nodes[node]['dur_med'] = np.median(durations)/3600/24
        sub_graph.nodes[node]['dur_q1'] = np.percentile(durations, 25)/3600/24
        sub_graph.nodes[node]['dur_q3'] = np.percentile(durations, 75)/3600/24
        sub_graph.nodes[node]['workload'] = single_workload(
            sub_graph, node, end_time, container_type='node')  
        del sub_graph.nodes[node]['data']
    sub_graph.remove_nodes_from(nodes_to_remove)
    

    edges_to_remove = []
    for edge in sub_graph.edges:
        all_in_times = sub_graph.edges[edge]['data']['in_time']
        all_out_times = sub_graph.edges[edge]['data']['out_time']
        in_time_lst, out_time_lst = sync_timestamp_lists(
            all_in_times, all_out_times, end_time,
            window=window, window_size=window_size)
        durations = [
            (t2 - t1).total_seconds() 
            for t1, t2 in zip(in_time_lst, out_time_lst)]
        durations = np.array(durations)

        if len(durations) == 0:
            edges_to_remove.append(edge)
            continue  # skip to next edge

        sub_graph.edges[edge]['dur_count'] = len(durations)
        sub_graph.edges[edge]['dur_mean'] = durations.mean()/3600/24
        sub_graph.edges[edge]['dur_std'] = durations.std()/3600/24
        sub_graph.edges[edge]['dur_min'] = durations.min()/3600/24
        sub_graph.edges[edge]['dur_max'] = durations.max()/3600/24
        sub_graph.edges[edge]['dur_med'] = np.median(durations)/3600/24
        sub_graph.edges[edge]['dur_q1'] = np.percentile(durations, 25)/3600/24
        sub_graph.edges[edge]['dur_q3'] = np.percentile(durations, 75)/3600/24
        sub_graph.edges[edge]['workload'] = single_workload(
            sub_graph, edge, end_time, container_type='edge')
        del sub_graph.edges[edge]['data']
    sub_graph.remove_edges_from(edges_to_remove)
    
    sub_graph = remove_isolated_nodes(sub_graph)
    return sub_graph

def extract_star_features(G, reference_node):
    node_features_list = []
    edge_features_list = []
    edge_counts = []
    for succ in G.successors(reference_node):        
        # Node features
        node_data = G.nodes[succ]
        node_feat = [
            node_data.get("dur_mean", 0),
            node_data.get("dur_std", 0),
            node_data.get("dur_min", 0),
            node_data.get("dur_max", 0),
            node_data.get("dur_med", 0),
            node_data.get("dur_q1", 0),
            node_data.get("dur_q3", 0),
            node_data.get("workload", 0),
        ]           
        node_features_list.append(node_feat)
        # Edge features
        edge_data = G[reference_node][succ]
        edge_feat = [
            edge_data.get("dur_mean", 0),
            edge_data.get("dur_std", 0),
            edge_data.get("dur_min", 0),
            edge_data.get("dur_max", 0),
            edge_data.get("dur_med", 0),
            edge_data.get("dur_q1", 0),
            edge_data.get("dur_q3", 0),
            edge_data.get("workload", 0),
        ]
        edge_features_list.append(edge_feat)        
        edge_counts.append(edge_data.get("dur_count", 1))

    # Convert to arrays
    node_features = np.array(node_features_list)
    edge_features = np.array(edge_features_list)
    # Compute relative weights
    edge_weights = np.array(edge_counts) / sum(edge_counts)
    # Append weights as last column
    node_features = np.hstack([node_features, edge_weights[:, None]])
    edge_features = np.hstack([edge_features, edge_weights[:, None]])
    num_nodes = len(node_features_list)
    return node_features, edge_features, num_nodes


def process_next_time(keys, prefixes, work_graph):
    prefix_dict = {}
    node_count = []
    for idx in tqdm(range(len(prefixes)), desc="SNA Encoding Progress"):        
        key = keys[idx]
        example = prefixes[idx]
        resource = example['resource']
        act_name = example['act_name']
        end_time = example['end_time']
        since_start = example['since_start']
        day_of_week = example['day_of_week']
        hour = example['hour']
        length = example['length']
        y = np.array(example['target'])
        sub_graph = get_sub_graph (work_graph, resource, end_time, mode= "next_time", length=3)
        if (sub_graph.number_of_nodes() >= 1 and 
            sub_graph.number_of_edges() >= 2):
            x_n, x_e, num_nodes = extract_star_features(sub_graph, resource)
            prefix_dict[key] = (resource, act_name, day_of_week, since_start, hour, length, x_n, x_e, y)
            node_count.append(num_nodes)
    max_nodes = max(node_count)                  
    return prefix_dict, max_nodes