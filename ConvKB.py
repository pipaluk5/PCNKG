# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:09:43 2023

@author: storm
"""

import os
from config.definitions import ROOT_DIR
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

path_load_PharmKG8k_trans_with_data_leakage = os.path.join(ROOT_DIR, "PharmKG8k", "transductive_with_data_leakage")
path_load_PharmKG8k_inductive_without_data_leakage = os.path.join(ROOT_DIR, "PharmKG8k", "inductive_without_data_leakage")
path_load_PharmKG8k_trans_without_data_leakage = os.path.join(ROOT_DIR, "PharmKG8k", "transductive_without_data_leakage")
path_load_full_split_trans_without_data_leakage = os.path.join(ROOT_DIR, "full_splits", "transductive_without_data_leakage")
CTD_trans = os.path.join(ROOT_DIR, "BioNev", "CTD_DDA", "transductive_split")
NDFRT_trans = os.path.join(ROOT_DIR, "BioNev", "NDFRT_DDA", "transductive_split")
CTD_inductive = os.path.join(ROOT_DIR, "BioNev", "CTD_DDA", "inductive_split")
NDFRT_inductive = os.path.join(ROOT_DIR, "BioNev", "NDFRT_DDA", "inductive_split")
full_with_new_edges = os.path.join(ROOT_DIR, "full_splits", "transductive_without_data_leakage_with_new_edges")
NDFRT_with_new_edges = os.path.join(ROOT_DIR, "BioNev", "NDFRT_DDA","tranductive_split_with_new_edges")

#PHARM
# train = TriplesFactory.from_path(path_load_PharmKG8k_trans_with_data_leakage+'/new_train.tsv')
# test = TriplesFactory.from_path(path_load_PharmKG8k_trans_with_data_leakage+'/new_test.tsv',
#                                 entity_to_id=train.entity_to_id,
#                                 relation_to_id=train.relation_to_id,)
# valid = TriplesFactory.from_path(path_load_PharmKG8k_trans_with_data_leakage+'/new_valid.tsv',
#                                   entity_to_id=train.entity_to_id,
#                                   relation_to_id=train.relation_to_id,)

# train = TriplesFactory.from_path(path_load_PharmKG8k_inductive_without_data_leakage+'/new_train.tsv')
# test = TriplesFactory.from_path(path_load_PharmKG8k_inductive_without_data_leakage+'/new_test.tsv',
#                                 entity_to_id=train.entity_to_id,
#                                 relation_to_id=train.relation_to_id,)
# valid = TriplesFactory.from_path(path_load_PharmKG8k_inductive_without_data_leakage+'/new_valid.tsv',
#                                   entity_to_id=train.entity_to_id,
#                                   relation_to_id=train.relation_to_id,)
 
# train = TriplesFactory.from_path(path_load_PharmKG8k_trans_without_data_leakage+'/new_train.tsv')
# test = TriplesFactory.from_path(path_load_PharmKG8k_trans_without_data_leakage+'/new_test.tsv',
#                                 entity_to_id=train.entity_to_id,
#                                 relation_to_id=train.relation_to_id,)
# valid = TriplesFactory.from_path(path_load_PharmKG8k_trans_without_data_leakage+'/new_valid.tsv',
#                                   entity_to_id=train.entity_to_id,
#                                   relation_to_id=train.relation_to_id,)
#FULL
# train = TriplesFactory.from_path(path_load_full_split_trans_without_data_leakage+'/new_train.tsv')
# test = TriplesFactory.from_path(path_load_full_split_trans_without_data_leakage+'/new_test.tsv',
#                                 entity_to_id=train.entity_to_id,
#                                 relation_to_id=train.relation_to_id,)
# valid = TriplesFactory.from_path(path_load_full_split_trans_without_data_leakage+'/new_valid.tsv',
#                                   entity_to_id=train.entity_to_id,
#                                   relation_to_id=train.relation_to_id,)
#CTD
# train = TriplesFactory.from_path(CTD_trans+'/new_train.tsv')
# test = TriplesFactory.from_path(CTD_trans+'/new_test.tsv',
#                                 entity_to_id=train.entity_to_id,
#                                 relation_to_id=train.relation_to_id,)
# valid = TriplesFactory.from_path(CTD_trans+'/new_valid.tsv',
#                                   entity_to_id=train.entity_to_id,
#                                   relation_to_id=train.relation_to_id,)

# train = TriplesFactory.from_path(CTD_inductive+'/new_train.tsv')
# test = TriplesFactory.from_path(CTD_inductive+'/new_test.tsv',
#                                 entity_to_id=train.entity_to_id,
#                                 relation_to_id=train.relation_to_id,)
# valid = TriplesFactory.from_path(CTD_inductive+'/new_valid.tsv',
#                                   entity_to_id=train.entity_to_id,
#                                   relation_to_id=train.relation_to_id,)

#NDFRT
# train = TriplesFactory.from_path(NDFRT_trans+'/new_train.tsv')
# test = TriplesFactory.from_path(NDFRT_trans+'/new_test.tsv',
#                                 entity_to_id=train.entity_to_id,
#                                 relation_to_id=train.relation_to_id,)
# valid = TriplesFactory.from_path(NDFRT_trans+'/new_valid.tsv',
#                                   entity_to_id=train.entity_to_id,
#                                   relation_to_id=train.relation_to_id,)
 
# train = TriplesFactory.from_path(NDFRT_inductive+'/new_train.tsv')
# test = TriplesFactory.from_path(NDFRT_inductive+'/new_test.tsv',
#                                 entity_to_id=train.entity_to_id,
#                                 relation_to_id=train.relation_to_id,)
# valid = TriplesFactory.from_path(NDFRT_inductive+'/new_valid.tsv',
#                                   entity_to_id=train.entity_to_id,
#                                   relation_to_id=train.relation_to_id,)

#With new edges
# train = TriplesFactory.from_path(full_with_new_edges+'/new_train.tsv')
# test = TriplesFactory.from_path(full_with_new_edges+'/new_test.tsv',
#                                 entity_to_id=train.entity_to_id,
#                                 relation_to_id=train.relation_to_id,)
# valid = TriplesFactory.from_path(full_with_new_edges+'/new_valid.tsv',
#                                   entity_to_id=train.entity_to_id,
#                                   relation_to_id=train.relation_to_id,)

train = TriplesFactory.from_path(NDFRT_with_new_edges+'/new_train.tsv')
test = TriplesFactory.from_path(NDFRT_with_new_edges+'/new_test.tsv',
                                entity_to_id=train.entity_to_id,
                                relation_to_id=train.relation_to_id,)
valid = TriplesFactory.from_path(NDFRT_with_new_edges+'/new_valid.tsv',
                                  entity_to_id=train.entity_to_id,
                                  relation_to_id=train.relation_to_id,)

# path_save = os.path.join(ROOT_DIR, "models", "ConvKB", "PharmKG8k", "transductive_split_with_data_leakage")
# path_save = os.path.join(ROOT_DIR, "models", "ConvKB", "PharmKG8k", "inductive_without_data_leakage")
# path_save = os.path.join(ROOT_DIR, "models", "ConvKB", "PharmKG8k", "transductive_split_without_data_leakage")
# path_save = os.path.join(ROOT_DIR, "models", "ConvKB", "merge_data", "transductive_split_without_data_leakage")
# path_save = os.path.join(ROOT_DIR, "models", "ConvKB", "CTD_DDA", "transductive_split")
# path_save = os.path.join(ROOT_DIR, "models", "ConvKB", "CTD_DDA", "inductive_split")
# path_save = os.path.join(ROOT_DIR, "models", "ConvKB", "NDFRT_DDA", "transductive_split")
# path_save = os.path.join(ROOT_DIR, "models", "ConvKB", "NDFRT_DDA", "inductive_split")
# path_save = os.path.join(ROOT_DIR, "models", "ConvKB", "merge_data", "new_edges_transductive_without_data_leakage")
path_save = os.path.join(ROOT_DIR, "models", "ConvKB", "NDFRT_DDA", "new_edges_transductive_split")



epoch_num = 200

result = pipeline(
    training = train,
    testing = test,
    validation = valid,
    model="ConvKB",
    model_kwargs=dict(
        embedding_dim=100,
        num_filters=50,
        hidden_dropout_rate=0.0,
        entity_initializer="xavier_uniform",
        relation_initializer="xavier_uniform",
    ),
    regularizer="PowerSum",
    regularizer_kwargs=dict(
        apply_only_once=True,
        weight=0.0005,
        p=2.0,
        normalize=False,
    ),
    optimizer="Adam",
    optimizer_kwargs=dict(
        lr=5e-06,
    ),
    loss="SoftplusLoss",
    loss_kwargs=dict(
        reduction="mean",
    ),
    training_kwargs=dict(
        num_epochs=epoch_num,
        checkpoint_name=f"{epoch_num}_epochs.pt",
        checkpoint_directory=path_save,
        checkpoint_frequency=5,
        batch_size=256,
    ),
    training_loop="SLCWA",
    negative_sampler="bernoulli",
    negative_sampler_kwargs=dict(
        num_negs_per_pos=1,
    ),
    evaluator_kwargs=dict(
        filtered=True,
    ),
    random_seed=42,
)

result.save_to_directory(path_save)
hits1 = result.get_metric('hits@1')
hits3 = result.get_metric('hits@3')
hits10 = result.get_metric('hits@10')
mrr = result.get_metric('both.realistic.mean_reciprocal_rank')
with open(path_save+'/results_test.txt', 'w') as f:
    f.write(f'Hits at 1,3,10\n{hits1}\n{hits3}\n{hits10}\nMean reciprocal rank:\n{mrr}')