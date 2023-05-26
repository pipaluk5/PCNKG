# -*- coding: utf-8 -*-
import numpy as np
#from pykeen.hpo import optimize_hpo
from pykeen.pipeline import pipeline
import os
from config.definitions import ROOT_DIR
from pykeen.triples import TriplesFactory

NDFRT_trans = os.path.join(ROOT_DIR, "BioNev", "NDFRT_DDA", "transductive_split")
train = TriplesFactory.from_path(NDFRT_trans+'/new_train.tsv')
test = TriplesFactory.from_path(NDFRT_trans+'/new_test.tsv',
                                entity_to_id=train.entity_to_id,
                                relation_to_id=train.relation_to_id,)
valid = TriplesFactory.from_path(NDFRT_trans+'/new_valid.tsv',
                                  entity_to_id=train.entity_to_id,
                                  relation_to_id=train.relation_to_id,)

from pykeen.hpo import hpo_pipeline


from optuna.samplers import GridSampler

hpo_pipeline_result = hpo_pipeline(
    n_trials=30,
    sampler=GridSampler,
    sampler_kwargs=dict(
        search_space={
            "model.embedding_dim": [50, 100, 200],
            "model.scoring_fct_norm": [1, 2],
            "loss.margin": [0.5, 1.0, 1.5],
            "optimizer.lr": [0.0001, 0.001, 0.01],
            "negative_sampler.num_negs_per_pos": [10, 50, 100],
            "training.num_epochs": [50, 100, 150],
            "training.batch_size": [32, 64, 128],
            "regularizer.weight": [0.0001, 0.001, 0.01],
            "regularizer.p": [1, 2],
            "regularizer.normalize": [True, False],
        },
    ),
    training=train,
    testing=test,
    validation=valid,
    model='ComplEx',
)

path_save = os.path.join(ROOT_DIR, "models", "ComplEx","NDFRT_DDA", "hyper_parameter_optimization", "second_search")
hpo_pipeline_result.save_to_directory(path_save)
