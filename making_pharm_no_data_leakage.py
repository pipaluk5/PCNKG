# -*- coding: utf-8 -*-


import pandas as pd
from split_and_entity_functions import *

import os
from config.definitions import ROOT_DIR
path_load = os.path.join(ROOT_DIR, 'transductive_with_data_leakage')


train = pd.read_table(path_load+'/new_train.tsv', names=["head", "relation", "tail"])
test = pd.read_table(path_load+'/new_test.tsv', names=["head", "relation", "tail"])
valid = pd.read_table(path_load+'/new_valid.tsv', names=["head", "relation", "tail"])

dataset = train.append(test).append(valid).drop_duplicates()

#Setting size of each split
train_size = int(len(dataset)*0.8)
test_size = int(len(dataset)*0.1)
valid_size = int(len(dataset)*0.1)

#Making the train set
new_train = get_split_for_dataframe(dataset, train_size)

#Making the test set
dataset_no_train = dataset.append(new_train).drop_duplicates(keep=False).reset_index(drop=True)

new_test = get_split_for_dataframe(dataset_no_train, test_size)

#Making the valid set
new_valid = dataset_no_train.append(new_test).drop_duplicates(keep=False)


#Verifying the results
check_entity_overlap(new_train, new_test, new_valid)
check_edge_overlap(new_train, new_test, new_valid)

path_save = os.path.join(ROOT_DIR, 'PharmKG8k', 'inductive_without_data_leakage')

new_train.to_csv(path_save+'/new_train.tsv', sep="\t", index = False, header=False)

new_test.to_csv(path_save+'/new_test.tsv', sep="\t", index = False, header=False)

new_valid.to_csv(path_save+'/new_valid.tsv', sep="\t", index = False, header=False)