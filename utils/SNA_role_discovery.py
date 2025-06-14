﻿# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import pearsonr
import networkx as nx
from operator import itemgetter
import pandas as pd
from sys import stdout

def print_progress(percentage, text):
    stdout.write("\r%s" % text + str(percentage)[0:5] + chr(37) + "...      ")
    stdout.flush()

   
class ResourcePoolAnalyser():
    """
        This class evaluates the tasks durations and associates resources to it
     """

    def __init__(self, log, drawing=False, sim_threshold=0.7):
        """constructor"""
        self.data = self.read_resource_pool(log)
        self.drawing = drawing
        self.sim_threshold = sim_threshold
        
        self.tasks = {val: i for i, val in enumerate(self.data.task.unique())}
        self.users = {val: i for i, val in enumerate(self.data.user.unique())}
        
        self.roles, self.resource_table = self.discover_roles()

    def read_resource_pool(self, log):
        if isinstance(log, pd.DataFrame):
            filtered_list = log[['task', 'user']]
        else:
            filtered_list = pd.DataFrame(log.data)[['task', 'user']]
        filtered_list = filtered_list[~filtered_list.task.isin(['Start', 'End'])]
        filtered_list = filtered_list[filtered_list.user != 'AUTO']
        return filtered_list


    def discover_roles(self):
        associations = lambda x: (self.tasks[x['task']], self.users[x['user']])
        self.data['ac_rl'] = self.data.apply(associations, axis=1)
    
        freq_matrix = (self.data.groupby(by='ac_rl')['task']
                       .count()
                       .reset_index()
                       .rename(columns={'task': 'freq'}))
        freq_matrix = {x['ac_rl']: x['freq'] for x in freq_matrix.to_dict('records')}
        
        profiles = self.build_profile(freq_matrix)
    
        print_progress(((20 / 100)* 100),'Analysing resource pool ')
        # building of a correl matrix between resouces profiles
        correl_matrix = self.det_correl_matrix(profiles)
        print_progress(((40 / 100)* 100),'Analysing resource pool ')
        # creation of a rel network between resouces
        g = nx.Graph()
        for user in self.users.values():
            g.add_node(user)
        for rel in correl_matrix:
            # creation of edges between nodes excluding the same elements
            # and those below the similarity threshold 
            if rel['distance'] > self.sim_threshold and rel['x'] != rel['y']:
                g.add_edge(rel['x'],
                           rel['y'],
                           weight=rel['distance'])
        print_progress(((60 / 100) * 100),'Analysing resource pool ')
        # extraction of fully conected subgraphs as roles
        sub_graphs = list(nx.connected_components(g))
        print_progress(((80 / 100) * 100),'Analysing resource pool ')
        # role definition from graph
        roles = self.role_definition(sub_graphs)
        # plot creation (optional)
        # if drawing == True:
        #     graph_network(g, sub_graphs)
        print_progress(((100 / 100)* 100),'Analysing resource pool ')
        return roles
    
    def build_profile(self, freq_matrix):
        profiles=list()
        for user, idx in self.users.items():
            profile = [0,] * len(self.tasks)
            for ac_rl, freq in freq_matrix.items():
                if idx == ac_rl[1]:
                    profile[ac_rl[0]] = freq
            profiles.append({'user': idx, 'profile': profile})
        return profiles

    def det_correl_matrix(self, profiles):
        correl_matrix = list()
        for profile_x in profiles:
            for profile_y in profiles:
                #x = scipy.array(profile_x['profile'])
                #y = scipy.array(profile_y['profile'])
                x = np.array(profile_x['profile'])
                y = np.array(profile_y['profile'])
                r_row, p_value = pearsonr(x, y)
                correl_matrix.append(({'x': profile_x['user'],
                                            'y': profile_y['user'],
                                            'distance': r_row}))
        return correl_matrix

    def role_definition(self, sub_graphs):
        user_index = {v: k for k, v in self.users.items()}
        records= list()
        for i in range(0, len(sub_graphs)):
            users_names = [user_index[x] for x in sub_graphs[i]]
            records.append({'role': 'Role '+ str(i + 1),
                            'quantity': len(sub_graphs[i]),
                            'members': users_names})
        #Sort roles by number of resources
        records = sorted(records, key=itemgetter('quantity'), reverse=True)
        for i in range(0,len(records)):
            records[i]['role']='Role '+ str(i + 1)
        resource_table = list()
        for record in records:
            for member in record['members']:
                resource_table.append({'role': record['role'],
                                       'resource': member})
        return records, resource_table