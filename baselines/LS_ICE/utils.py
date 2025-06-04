import os
import pm4py
import pickle
import pandas as pd
import numpy as np
from keras.utils import Sequence
from itertools import accumulate
 

def impute_missing_values(data, feat_dic):
    # Impute numerical features with mean
    for col in feat_dic.get('num_feat', []):
        if col in data.columns:
            data[col] = data[col].fillna(data[col].mean())    
    # Impute categorical features with "UNKNOWN"
    for col in feat_dic.get('cat_feat', []):
        if col in data.columns:
            data[col] = data[col].fillna("UNKNOWN")    
    return data
    
def read_data(log_path, out_dir, int_df, feat_lst, feat_dic):
    train_id_path = os.path.join(out_dir, 'train_ids.pickle') 
    val_ids_path = os.path.join(out_dir, 'val_ids.pickle') 
    test_ids_path = os.path.join(out_dir, 'test_ids.pickle') 
    with open(train_id_path, 'rb') as f:
        train_ids = pickle.load(f)
    with open(val_ids_path, 'rb') as f:
        val_ids = pickle.load(f)
    with open(test_ids_path, 'rb') as f:
        test_ids = pickle.load(f)           
    data = pm4py.read_xes(log_path)
    data = impute_missing_values(data, feat_dic)
    # Sort by case and timestamp
    data = data.sort_values(['case:concept:name', 'time:timestamp'])
    # Group by 'case:concept:name'
    grouped = data.groupby('case:concept:name')
    # Add prefix_length as cumulative count starting at 1
    data['prefix_length'] = grouped.cumcount() + 1
    # Create 'trace' by cumulatively joining 'concept:name' values with commas
    data['trace'] = grouped['concept:name'].transform(lambda x: list(accumulate(x, lambda a, b: a + ', ' + b)))
    #data['trace'] = grouped['concept:name'].transform(lambda x: x.expanding().apply(lambda y: ','.join(y), raw=False))
    data.rename(columns={'case:concept:name': 'case_id'}, inplace=True)
    add_cols = [col for col in int_df.columns if col not in {
        'case_id', 'prefix_length', 'activity'}]
    assert data.set_index(['case_id', 'prefix_length']).index.isin(int_df.set_index(['case_id', 'prefix_length']).index).all(), "Unmatched rows found!"
    data = data.merge(int_df[['case_id', 'prefix_length'] + add_cols],
                      on=['case_id', 'prefix_length'], how='left') 
    keys_list = ['case_id', 'prefix_length', 'trace']+feat_lst+['remtime_std']
    data = data[keys_list]  
    grouped = data.groupby('case_id')
    for col in feat_lst:
        data[col] = grouped[col].transform(lambda x: list(accumulate(x.astype(str), lambda a, b: a + ', ' + b)))  
    data['remtime_std'] = grouped['remtime_std'].transform(lambda x: list(accumulate(x.astype(str), lambda a, b: a + ', ' + b)))     
    data.rename(columns=lambda x: x.replace('case:', 'case_'), inplace=True)
    data.rename(columns=lambda x: x.replace('org:', 'org_'), inplace=True)
    data_train = data[data['case_id'].isin(train_ids)]
    data_valid = data[data['case_id'].isin(val_ids)]
    data_test = data[data['case_id'].isin(test_ids)]    
    return (data, data_train, data_valid, data_test)





def to_integer_vector(vector, output_dim, cut):
    
    cut_integer_vector = vector[max(0, cut-output_dim):cut]
    cut_integer_vector = [0.0] * (output_dim - len(cut_integer_vector)) + cut_integer_vector
    
    return cut_integer_vector

def get_x(activity_vector, 
          time_dic,
          cat_dic,
          num_dic,
          helpers_dic,
          output_dim, cut):
    
    integer_vector = activity_vector if len(activity_vector) and type(activity_vector[0]) == int else [helpers_dic['trace_helper']['activity_to_integer'][act] for act in activity_vector] 
    act_cut_integer_vector = integer_vector[max(0, cut-output_dim):cut]; act_cut_integer_vector = [0.0] * (output_dim - len(act_cut_integer_vector)) + act_cut_integer_vector
    
    return_dic = {}
    for string in ['trace'] + list(time_dic.keys()) + list(cat_dic.keys()) + list(num_dic.keys()):
        if string == 'trace':
            return_dic['{}_seq'.format(string)] = np.zeros((output_dim, helpers_dic['trace_helper']['vocab_size']))
        else:
            return_dic['{}_seq'.format(string)] = np.zeros((output_dim, 1))
    
    time_vector_dic = {}
    for time_str, time_vec in time_dic.items():
        if ('day' in time_str) | ('hour' in time_str) :
            time_vector_dic['{}_integer_vector'.format(time_str)] = to_integer_vector([helpers_dic['time_helpers'][time_str]['{}_to_integer'.format(time_str)][val] for val in time_dic[time_str]], output_dim, cut)
        else:
            time_vector_dic['{}_integer_vector'.format(time_str)] = to_integer_vector(time_vec, output_dim, cut)
    
    cat_vector_dic = {}
    for cat_str, cat_vec in cat_dic.items():
        cat_vector_dic['{}_integer_vector'.format(cat_str)] = to_integer_vector([helpers_dic['cat_helpers'][cat_str]['{}_to_integer'.format(cat_str)][val] for val in cat_dic[cat_str]], output_dim, cut)
    
    num_vector_dic = {}
    for num_str, num_vec in num_dic.items():
        num_vector_dic['{}_integer_vector'.format(num_str)] = to_integer_vector(num_vec, output_dim, cut)        
    
    for p in range(output_dim):
        for v in range(helpers_dic['trace_helper']['vocab_size']):
            return_dic['trace_seq'][p, v] = 1 if act_cut_integer_vector[p] == v else 0
    
        for time_str in list(time_dic.keys()):
            return_dic['{}_seq'.format(time_str)][p] = time_vector_dic['{}_integer_vector'.format(time_str)][p]
        
        for cat_str in list(cat_dic.keys()):
            return_dic['{}_seq'.format(cat_str)][p] = cat_vector_dic['{}_integer_vector'.format(cat_str)][p]
        
        for num_str in list(num_dic.keys()):
            return_dic['{}_seq'.format(num_str)][p] = num_vector_dic['{}_integer_vector'.format(num_str)][p]    
            
    return return_dic


def get_y(activity_vector, time_vector, helpers_dic, output_dim, cut):
    integer_vector = activity_vector if len(activity_vector) and type(activity_vector[0]) == int else [helpers_dic['trace_helper']['activity_to_integer'][act] for act in activity_vector]
       
    remtrace_seq = np.zeros((output_dim, helpers_dic['trace_helper']['vocab_size']))
    time_out_seq = np.zeros((output_dim, 1))
    
    remact_cut_integer_vector = integer_vector[cut:output_dim+cut]
    remact_cut_integer_vector = remact_cut_integer_vector + [0.0] * (output_dim-len(remact_cut_integer_vector))
    
    time_out_cut_integer_vector = time_vector[cut-1:output_dim+cut]
    time_out_cut_integer_vector = time_out_cut_integer_vector + [0.0] * (output_dim-len(time_out_cut_integer_vector))
    
    for p in range(output_dim):
        time_out_seq[p] = time_out_cut_integer_vector[p]
    
    for p in range(output_dim):
        for v in range(helpers_dic['trace_helper']['vocab_size']):
            remtrace_seq[p, v] = 1 if remact_cut_integer_vector[p] == v else 0
    
    return remtrace_seq, time_out_seq


def get_batch(data_frame, selected_ids, output_dim, feat_dic, helpers_dic, selected_cuts_or_strategy='random', return_indexes=False):
    
    time_feat = feat_dic['time_feat']
    cat_feat = feat_dic['cat_feat']
    num_feat = feat_dic['num_feat']
    
    X_dic = {}
    for col in ['trace']+time_feat+cat_feat+num_feat:
        X_dic['X{}'.format(col)] = []  
    
    YTime = []; YTrace = []; idxs=[]
    
    for i, idx in enumerate(selected_ids):
        activity_vector = data_frame.loc[idx, 'trace'].split(', ')     
        
        time_dic = {'{}'.format(col): data_frame.loc[idx, col].split(', ')  for col in time_feat}
        cat_dic = {'{}'.format(col): data_frame.loc[idx, col].split(', ') for col in cat_feat}
        num_dic = {'{}'.format(col): data_frame.loc[idx, col].split(', ') for col in num_feat}
        
        
        remtime_std_vector = data_frame.loc[idx, 'remtime_std'].split(', ') 
        
        if selected_cuts_or_strategy == 'random':
            cut_list = [np.random.randint(1, len(activity_vector) + 1)] 
        elif selected_cuts_or_strategy == 'all':
            cut_list = range(1, len(activity_vector) + 1)
        else:
            cut_list = selected_cuts_or_strategy[i]
        
        for cut in cut_list:
            
            return_dic = get_x(activity_vector, 
                               time_dic, 
                               cat_dic, 
                               num_dic, 
                               helpers_dic,
                               output_dim, cut)
            
    
            for key in list(X_dic.keys()):
                X_dic[key].append(return_dic['{}_seq'.format(key.replace('X', ''))])
            
            [Trace_out, Time_out] = get_y(activity_vector, remtime_std_vector, helpers_dic, output_dim, cut)
            YTime.append(Time_out); YTrace.append(Trace_out)
            
            idxs.append(idx)
        
        
        X = {'{}'.format(key): np.array(X_dic[key]) for key in X_dic.keys()}
        
        Y = {
            'time_out': np.array(YTime), 
            'trace_out': np.array(YTrace)
        }
        
        IDs = {
            'Bag_ids':np.array(idxs)
        }
        
              
    if return_indexes:
        return X, Y, IDs
    else:
        return X, Y,
   
     

class BagDataGenerator(Sequence):
    def __init__(self, data_frame, output_dim, feat_dic, helpers_dic, 
                 batch_size=128, shuffle=True, override_indexes=None, 
                 seed=None, **batch_kwargs): 
        self.data_frame = data_frame
        self.indexes = np.array(self.data_frame.index) if override_indexes is None else override_indexes
        self.output_dim = output_dim
        self.feat_dic = feat_dic
        self.helpers_dic = helpers_dic
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.on_epoch_end()
        self.batch_kwargs = batch_kwargs
    def __len__(self): 
        return int(np.ceil(self.indexes.shape[0] / self.batch_size))
    def __getitem__(self, index): 
        selected_ids = self.indexes[index*self.batch_size : (index+1)*self.batch_size] 
        return self.__data_generation(selected_ids)
    def on_epoch_end(self):
        if self.shuffle == True:
            self.indexes = self.rng.permutation(self.indexes)
            #np.random.shuffle(self.indexes)
    def __data_generation(self, selected_ids): 
        return get_batch(self.data_frame, selected_ids, self.output_dim, self.feat_dic, self.helpers_dic, **self.batch_kwargs) 
    

def get_feat_dict(dataset, int_df):
    int_df.rename(columns={'duration': 'duration_std'}, inplace=True)
    int_df.rename(columns={'since_last': 'time_betw_std'}, inplace=True)   
    int_df.rename(columns={'remtime': 'remtime_std'}, inplace=True)   
    cols = [col for col in int_df.columns if col not in {
        'case_id', 'prefix_length', 'activity', 'day', 'hour', 'remtime_std',
        'duration_std', 'time_betw_std'}]
    int_df.rename(columns={col: col + '_std' for col in cols}, inplace=True)
    cols = [col + '_std' for col in cols]
    time_feat = ['day', 'hour',  'duration_std', 'time_betw_std']
    if dataset == 'HelpDesk':
        cat_feat = ['case:responsible_section', 'case:support_section',
                    'case:product', 'workgroup', 'seriousness_2',
                    'service_level' , 'service_type', 'org:resource']
        num_feat = [] + cols
    elif dataset in {'BPIC15_1', 'BPIC15_2', 'BPIC15_3', 'BPIC15_4', 'BPIC15_5'}:
        cat_feat = ['case:caseStatus', 'case:last_phase', 'case:Responsible_actor',
                    'case:termName', 'case:requestComplete',
                    'case:caseProcedure', 'case:Includes_subCases',
                    'monitoringResource', 'org:resource']
        num_feat = ['case:SUMleges'] + cols
    elif dataset == 'BPIC20DD':
        cat_feat = ['org:role', 'org:resource']
        num_feat = ['case:Amount'] + cols
    elif dataset == 'BPIC20ID':
        cat_feat = ['org:role', 'org:resource']
        num_feat = ['case:Amount', 'case:RequestedAmount',
                    'case:AdjustedAmount', 'case:OriginalAmount'] + cols
    elif dataset == 'BPIC20PTC':
        cat_feat = ['case:OrganizationalEntity', 'org:role', 'org:resource']
        num_feat = ['case:RequestedAmount'] + cols
    elif dataset == 'BPIC17':
        cat_feat = ['case:LoanGoal', 'case:ApplicationType', 'org:resource', 'Action', 'EventOrigin', 'Accepted', 'Selected']
        num_feat = ['case:RequestedAmount', 'FirstWithdrawalAmount',
                    'NumberOfTerms', 'MonthlyCost', 'CreditScore',
                    'OfferedAmount'] + cols
    feat_dic = {}
    feat_dic['time_feat'] = time_feat
    feat_dic['cat_feat'] = cat_feat
    feat_dic['num_feat'] = num_feat
    feat_lst = time_feat+cat_feat+num_feat
    return int_df, feat_dic, feat_lst 


