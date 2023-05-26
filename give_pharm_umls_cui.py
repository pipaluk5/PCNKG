# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:21:25 2023

@author: storm
"""

import os
from config.definitions import ROOT_DIR
path_save = os.path.join(ROOT_DIR, "PharmKG8k", "transductive_with_data_leakage")
path_load = os.path.join(ROOT_DIR, "PharmKG8k")
import pandas as pd

base_url = 'https://raw.githubusercontent.com/biomed-AI/PharmKG/master/data/PharmKG-8k/'

train = pd.read_table(base_url+'train.tsv', names=["head", "relation", "tail"])
test = pd.read_table(base_url+'test.tsv', names=["head", "relation", "tail"])
valid = pd.read_table(base_url+'valid.tsv', names=["head", "relation", "tail"])

pharmXumls = pd.read_csv(path_load+'/pharmkg8kXumls.csv').reset_index(drop = True)

pharmXumls.drop_duplicates(inplace=True)

train.drop_duplicates(inplace=True)
test.drop_duplicates(inplace=True)
valid.drop_duplicates(inplace=True)


def give_umls_cui(original_df, umls_trans):
    head_replaced = pd.merge(original_df, umls_trans, left_on = ("head"),  right_on  = ("name"), how = "inner")
    head_replaced = head_replaced[['UMLS_ID','relation','tail']]
    head_replaced.rename(columns=({'UMLS_ID': 'head'}), inplace = True)
    
    tail_replaced = pd.merge(head_replaced , umls_trans, left_on = ("tail"),  right_on  = ("name"), how = "inner")
    tail_replaced = tail_replaced[['head','relation','UMLS_ID']]
    tail_replaced.rename(columns=({'UMLS_ID':'tail'}), inplace = True)
    lost_edges = len(original_df)-len(tail_replaced)
    print(f'Lost {lost_edges} edges in the process')
    return tail_replaced

new_train = give_umls_cui(train, pharmXumls)
new_test = give_umls_cui(test, pharmXumls)
new_valid = give_umls_cui(valid, pharmXumls)

new_train.to_csv(path_save+'/new_train.tsv', sep="\t", index = False, header=False)
new_test.to_csv(path_save+'/new_test.tsv', sep="\t", index = False, header=False)
new_valid.to_csv(path_save+'/new_valid.tsv', sep="\t", index = False, header=False)


def get_entity_count(df1,df2,df3):
    df1_head = set(df1["head"])
    df1_tail = set(df1["tail"])
    entities_df1 = df1_head.union(df1_tail)
    df2_head = set(df2["head"])
    df2_tail = set(df2["tail"])
    entities_df2 = df2_head.union(df2_tail)
    df3_head = set(df3["head"])
    df3_tail = set(df3["tail"])
    entities_df3 = df3_head.union(df3_tail)
    all_entities = entities_df1.union(entities_df2).union(entities_df3)
    amount_entities = len(all_entities)
    return amount_entities

old_entity_count = get_entity_count(train, test, valid)

new_entity_count = get_entity_count(new_train, new_test, new_valid)

print(old_entity_count-new_entity_count)
