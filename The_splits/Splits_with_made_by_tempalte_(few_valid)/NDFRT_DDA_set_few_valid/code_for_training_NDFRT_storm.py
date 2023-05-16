# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:14:21 2023

@author: storm
"""
path = 'D:/Documents - HDD/GitHub/Drug-Repurposing-Maja-Storm/Drug-Repurposing-Maja-Storm/Our_split/new_our_split/CTD_set_9may/'

from pykeen.triples import TriplesFactory

from pykeen.pipeline import pipeline

ndfrt_train = TriplesFactory.from_path(path+'train_CTD.tsv')
ndfrt_test = TriplesFactory.from_path(path+'test_CTD.tsv')
ndfrt_valid = TriplesFactory.from_path(path+'valid_CTD.tsv')


epoch_num = 1000

result = pipeline(
    training = ndfrt_train,
    testing = ndfrt_test,
    validation = ndfrt_valid,
    model = "TransE",
    #stopper='early',
    model_kwargs= dict(
      embedding_dim = 100,
      scoring_fct_norm = 1,
      entity_initializer = "xavier_uniform",
      relation_initializer = "xavier_uniform",
      entity_constrainer = "normalize",
    ),
    optimizer= "SGD",
    optimizer_kwargs = dict(
        lr = 0.0005,
        ),
    loss= "MarginRankingLoss",
    loss_kwargs = dict(
        reduction = "mean",
        margin = 1,
        ),
    training_kwargs=dict(
        num_epochs=epoch_num,

        checkpoint_name=f'{epoch_num}_epochs.pt',

        checkpoint_directory=path+f'CTD_models/TransE/train_model_batch256lr00005/{epoch_num}_epochs',
        
        checkpoint_frequency=5,
        
        batch_size=256,
        ),
    training_loop= "SLCWA",
    negative_sampler= "bernoulli",
    negative_sampler_kwargs= dict(
        num_negs_per_pos= 1,
        ),
    evaluator_kwargs= dict(
        filtered = True,
        ),
    random_seed=42,
)

result.save_to_directory(path+f'CTD_models/TransE/train_model_batch256lr00005/{epoch_num}_epochs')
print('hits at 1,3,10,100')
print(result.get_metric('hits@1'))
print(result.get_metric('hits@3'))
print(result.get_metric('hits@10'))
#print(result.get_metric('hits@100'))
print('Mean reciprocal rank:')
print(result.get_metric('both.realistic.mean_reciprocal_rank'))


