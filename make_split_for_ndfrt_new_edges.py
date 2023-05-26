# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:19:02 2023

@author: storm
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd 
import os
from collections import Counter
from config.definitions import ROOT_DIR
from split_and_entity_functions import check_edge_overlap, check_entity_overlap

def get_diff_between_entity_lists(df1,df2):
    df1_head = set(df1["head"])
    df1_tail = set(df1["tail"])
    entities_df1 = df1_head.union(df1_tail)
    df2_head = set(df2["head"])
    df2_tail = set(df2["tail"])
    entities_df2 = df2_head.union(df2_tail)
    return entities_df1-entities_df2    

def get_split_sizes(train,test,valid, dataset = ""):
    train_size = len(train)
    test_size = len(test)
    valid_size = len(valid)
    full_size = train_size+test_size+valid_size
    print(f'Size of new splits {dataset}:\nTrain:{round(train_size/full_size*100,2)}%\nTest:{round(test_size/full_size*100,2)}%\nValid:{round(valid_size/full_size*100,2)}%')

def save_set(train,test,valid,path):
    train.to_csv(path+'/new_train.tsv', sep="\t", index = False, header=False)

    test.to_csv(path+'/new_test.tsv', sep="\t", index = False, header=False)

    valid.to_csv(path+'/new_valid.tsv', sep="\t", index = False, header=False)

#Load old NDFRT split
path_load_N = os.path.join(ROOT_DIR, "BioNev", "NDFRT_DDA", 'transductive_split')
old_train = pd.read_table(path_load_N+'/new_train.tsv', names = ["head", "relation", "tail"])
old_test = pd.read_table(path_load_N+'/new_test.tsv', names = ["head", "relation", "tail"])
old_valid = pd.read_table(path_load_N+'/new_valid.tsv', names = ["head", "relation", "tail"])
full_old_ndfrt = pd.concat([old_train,old_test,old_valid])

#Load new edges for NDFRT
path_load = os.path.join(ROOT_DIR, "BioNev", "NDFRT_DDA")
new_edges = pd.read_table(path_load+("/NDFRT_new_edges.tsv"))

#Get entities which are not in old NDFRT split but are in new
entities_in_new_not_in_old = get_diff_between_entity_lists(new_edges, full_old_ndfrt)


#Entities which are in old NDFRT train
entities_in_old_train = set(old_train['head'].append(old_train['tail']))

#Get indexes for the ones which has to be in train, and those that can be in either test or valid
has_to_be_in_train_index = []
can_be_in_test_valid = []
dont_know_index = []
for ind in new_edges.index:
    if new_edges.iat[ind,0] in entities_in_new_not_in_old or new_edges.iat[ind,2] in entities_in_new_not_in_old:
        has_to_be_in_train_index.append(ind)
    elif new_edges.iat[ind,0] in entities_in_old_train and new_edges.iat[ind,2] in entities_in_old_train:
        can_be_in_test_valid.append(ind)
    else:
        dont_know_index.append(ind)


#Get split sizes
size_train = int(len(new_edges)*0.8)
size_test = int(len(new_edges)*0.1)

#Make the temporary train dataframe for the new edges
temp_new_train = pd.DataFrame(columns = ['head','relation','tail'])

temp_new_train = temp_new_train.append(new_edges.loc[has_to_be_in_train_index, :])

#Get the edges which has no place yet, and therefore can be sampled from, since the rest could be anywhere
temp_edges_no_train = pd.concat([new_edges, temp_new_train]).drop_duplicates(keep=False)

#Sample the splits for the new edges:
new_edges_train = temp_edges_no_train.sample(size_train-len(temp_new_train), random_state = 42)
new_edges_train = pd.concat([new_edges_train, temp_new_train])
new_edges_no_train = pd.concat([new_edges, new_edges_train]).drop_duplicates(keep=False)
new_edges_test = new_edges_no_train.sample(size_test, random_state = 42)
new_edges_valid = pd.concat([new_edges_no_train, new_edges_test]).drop_duplicates(keep=False)

#Give the new edges to old NDFRT split
imp_train = pd.concat([new_edges_train, old_train])
imp_test = pd.concat([new_edges_test, old_test])
imp_valid = pd.concat([new_edges_valid, old_valid])

#Load the Full Split
path_load_A = os.path.join(ROOT_DIR, "full_splits", "transductive_without_data_leakage")
train_A = pd.read_table(path_load_A+'/new_train.tsv', names = ["head", "relation", "tail"])
test_A = pd.read_table(path_load_A+'/new_test.tsv', names = ["head", "relation", "tail"])
valid_A = pd.read_table(path_load_A+'/new_valid.tsv', names = ["head", "relation", "tail"])

#Give the new edges to the full split
imp_train_A = pd.concat([train_A, new_edges_train])
imp_test_A = pd.concat([test_A, new_edges_test])
imp_valid_A = pd.concat([valid_A, new_edges_valid])

#Verify if the transductive and no data leakage hold true, and the split sizes still are good
print('Verifying metrics for improved NDFRT')
check_entity_overlap(imp_train, imp_test, imp_valid)
check_edge_overlap(imp_train, imp_test, imp_valid)
get_split_sizes(imp_train, imp_test, imp_valid, 'Improved NDFRT')

print('Verifying metrics for improved full split')
check_entity_overlap(imp_train_A, imp_test_A, imp_valid_A)
check_edge_overlap(imp_train_A, imp_test_A, imp_valid_A)
get_split_sizes(imp_train_A, imp_test_A, imp_valid_A, 'Improved Full split')

#Save the files
path_save_A = os.path.join(ROOT_DIR, "full_splits", 'transductive_without_data_leakage_with_new_edges')
path_save_N = os.path.join(ROOT_DIR, "BioNev", "NDFRT_DDA", 'tranductive_split_with_new_edges')


save_set(imp_train_A, imp_test_A, imp_valid_A, path_save_A)
save_set(imp_train,imp_test,imp_valid,path_save_N)


