# -*- coding: utf-8 -*-
"""
Created on Wed May 24 13:38:05 2023

@author: Pip
"""
import os
from config.definitions import ROOT_DIR

import pandas as pd 
from split_and_entity_functions import check_edge_overlap, check_entity_overlap

#load
path_load_CTD = os.path.join(ROOT_DIR, "BioNev", "CTD_DDA", "CTD_original_set_UMLS")
path_load_NDFRT = os.path.join(ROOT_DIR, "BioNev", "NDFRT_DDA", "NDFRT_original_set_UMLS")
#save
path_save_CTD = os.path.join(ROOT_DIR, "BioNev", "CTD_DDA", "inductive_split")
path_save_NDFRT = os.path.join(ROOT_DIR, "BioNev", "NDFRT_DDA", "inductive_split")

train_CTD = pd.read_table(path_load_CTD+("/train_edges_umls.tsv"), names = ["head", "relation", "tail"])
test_CTD_must_be_split = pd.read_table(path_load_CTD+("/test_edges_umls.tsv"), names = ["head", "relation", "tail"])

train_NDFRT = pd.read_table(path_load_NDFRT+("/train_edges_umls.tsv"), names = ["head", "relation", "tail"])
test_NDFRT_must_be_split = pd.read_table(path_load_NDFRT+("/test_edges_umls.tsv"), names = ["head", "relation", "tail"])

def split_test(test_set):
    valid = test_set.sample(frac = 0.5, random_state = 42)
    test = pd.concat([test_set, valid], join = "outer").drop_duplicates(keep = False)
    return valid, test

valid_CTD, test_CTD = split_test(test_CTD_must_be_split)
valid_NDFRT, test_NDFRT = split_test(test_NDFRT_must_be_split)

#CTD save
train_CTD.to_csv(path_save_CTD+"/new_train.tsv", sep = "\t", index = False, header = False)
valid_CTD.to_csv(path_save_CTD+"/new_valid.tsv", sep = "\t", index = False, header = False)
test_CTD.to_csv(path_save_CTD+"/new_test.tsv", sep = "\t", index = False, header = False)

#NDFRT save
train_NDFRT.to_csv(path_save_NDFRT+"/new_train.tsv", sep = "\t", index = False, header = False)
valid_NDFRT.to_csv(path_save_NDFRT+"/new_valid.tsv", sep = "\t", index = False, header = False)
test_NDFRT.to_csv(path_save_NDFRT+"/new_test.tsv", sep = "\t", index = False, header = False)

print('Check NDFRT')
check_entity_overlap(train_NDFRT, test_NDFRT, valid_NDFRT)
check_edge_overlap(train_NDFRT, test_NDFRT, valid_NDFRT)

print('Check CTD')
check_entity_overlap(train_CTD, test_CTD, valid_CTD)
check_edge_overlap(train_CTD, test_CTD, valid_CTD)

