# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd 
import os
from config.definitions import ROOT_DIR
from split_and_entity_functions import check_edge_overlap, check_entity_overlap

#used to avoid data leakage
def can_be_moved_pharm(train_P):
    #getting train without relation
    train_P_without_relation = train_P[["head", "tail"]]
    #swapping around the head and tial of train without relation
    train_tail_head_without_relation = train_P_without_relation.rename(columns = {"head":"tail", "tail":"head"})
    train_tail_head_without_relation = train_tail_head_without_relation[["head", "tail"]]
    #deleting all that train and train head tial swaped have in commen, so that what is left only is the entity combinationes that have one occurrences 
    inner_train_P_x_train_tail_head = pd.concat([train_tail_head_without_relation, train_P_without_relation], join="outer").drop_duplicates(keep = False)
    #finding the intercestion between all the edges that are unique combination and the original train. 
    #what is left is all unique edge, that can be moved without causing overlaps between train, test and valid, since they are unique. 
    can_be_moved = pd.merge(inner_train_P_x_train_tail_head, train_P, on=("head","tail"), how = "inner")
    can_be_moved = can_be_moved[["head","relation","tail"]]
    return can_be_moved

def inner_join(new_dataframe, used_to_exclude):
    all_inner_head_head = pd.merge(new_dataframe, used_to_exclude, on=["head", "tail"], how="inner").drop_duplicates()

    used_to_exclude = used_to_exclude.rename(columns={"head": "tail", "tail": "head"})

    used_to_exclude_head_tail = used_to_exclude[["head", "tail"]]

    all_inner_head_tail = pd.merge(used_to_exclude_head_tail, new_dataframe,  how='inner', on=["head", "tail"])
    
    # new_dataframe_with_relation_xy = all_inner_head_head.append(all_inner_head_tail).drop_duplicates()
    new_dataframe_with_relation_xy = pd.concat([all_inner_head_head, all_inner_head_tail], join = "outer").drop_duplicates()
    
    with_relation_x = new_dataframe_with_relation_xy[["head", "relation_x", "tail"]].dropna()
    with_relation_x.rename(columns={"relation_x":"relation"}, inplace=True)
    with_relation = new_dataframe_with_relation_xy[["head", "relation", "tail"]].dropna()
    
    # new_dataframe_final = with_relation_x.append(with_relation).drop_duplicates()
    new_dataframe_final = pd.concat([with_relation_x, with_relation], join = "outer").drop_duplicates()
    
    return new_dataframe_final

def new_splits_all_overlaps_in_train(has_to_be_in_train, train, test, valid, total_number_of_all_sets):
    #MAKING TEST SET
    #delete all that has to be in train from train, so it can be sampled from. 
    train_and_have_to_be_in_train_inner = pd.merge(has_to_be_in_train, train, on = ("head", "relation", "tail"), how = "inner")
    # train_almost  = train_and_have_to_be_in_train_inner.append(train).drop_duplicates(keep = False)
    train_almost = pd.concat([train_and_have_to_be_in_train_inner, train], join = "outer").drop_duplicates(keep = False)


    #delete all that test and train have incommon from test
    test_inner_with_has_to_be_in_train = pd.merge(has_to_be_in_train, test, on = ("head", "relation", "tail"), how = "inner")
    # test_almost = test_inner_with_has_to_be_in_train.append(test).drop_duplicates(keep = False).reset_index(drop = True)
    test_almost = pd.concat([test_inner_with_has_to_be_in_train, test], join = "outer").drop_duplicates(keep = False).reset_index(drop = True)
    #Findes that sample size that has to be moved from train to test, to get a 8-1-1 split
    sampel_size_test = int((total_number_of_all_sets*0.1)-len(test_almost))
    
    #Sampling from almost train, that have to get into test
    sampel_test = train_almost.sample(sampel_size_test, random_state = 42)
    # test_new = test_almost.append(sampel_test).reset_index(drop = True).drop_duplicates()
    test_new = pd.concat([test_almost, sampel_test], join = "outer").reset_index(drop = True).drop_duplicates()
    
    #Deleting the sample the was sampled for train and put into test, from almost train
    # train_without_test_overlaps = sampel_test.append(train_almost).drop_duplicates(keep = False)
    train_without_test_overlaps = pd.concat([sampel_test, train_almost], join ="outer").drop_duplicates(keep = False)
    
    #MANING VALID SET    
    ##delete all that valid and train almost have incommon from valid
    valid_almost = pd.merge(has_to_be_in_train, valid, on = ("head", "relation", "tail"), how = "inner")
    # valid_almost = valid_almost.append(valid).drop_duplicates(keep = False).reset_index(drop = True)
    valid_almost = pd.concat([valid_almost, valid], join="outer").drop_duplicates(keep = False).reset_index(drop = True)

    #Findes that sample size that has to be moved from train to valid, to get a 8-1-1 split
    sampel_size_valid = int((total_number_of_all_sets*0.1)-len(valid_almost))
    
    #Sampling from train_without_test_overlaps, that have to get into valid
    sampel_valid = train_without_test_overlaps.sample(sampel_size_valid, random_state = 42)
    # valid_new = valid_almost.append(sampel_valid).reset_index(drop = True).drop_duplicates()
    valid_new = pd.concat([valid_almost, sampel_valid], join = "outer").reset_index(drop = True).drop_duplicates()
    
    # train_without_test_and_valid_overlaps = sampel_valid.append(train_without_test_overlaps).drop_duplicates(keep = False)
    train_without_test_and_valid_overlaps = pd.concat([sampel_valid, train_without_test_overlaps], join = "outer").drop_duplicates(keep = False)
    
    # train_new = has_to_be_in_train.append(train_without_test_and_valid_overlaps).drop_duplicates()
    train_new = pd.concat([has_to_be_in_train, train_without_test_and_valid_overlaps], join = "outer").drop_duplicates()
    
    return train_new, test_new, valid_new

def find_can_be_moved(train, overlaps_with_train):
    train_head_tail = pd.merge(train, train, left_on=("head"), right_on = ("tail"), how ="inner").drop_duplicates()
    
    train_x = train_head_tail[["head_x","relation_x","tail_x"]].drop_duplicates()
    train_y = train_head_tail[["head_y","relation_y","tail_y"]].drop_duplicates()
    
    train_x_y_outer = pd.merge(train_x, train_y, left_on = ("head_x","relation_x","tail_x"), right_on = ("head_y","relation_y","tail_y"), how ="outer").drop_duplicates(keep=False)
    more_than_one_comp_head_head = train_x_y_outer[train_x_y_outer["head_y"].isna()]
    more_than_one_comp_head_head = more_than_one_comp_head_head.rename(columns = {"head_x":"head","relation_x":"relation","tail_x":"tail"})
    more_than_one_comp_head_head = more_than_one_comp_head_head[["head","relation","tail"]]
    
    train_x_y_inner = pd.merge(train_x, train_y, left_on = ("tail_x"), right_on = ("head_y"), how ="inner").drop_duplicates()
    
    train_x_x = train_x_y_inner[["head_x","relation_x","tail_x"]].drop_duplicates()

    train_y_y = train_x_y_inner[["head_y","relation_y","tail_y"]].drop_duplicates()

    
    train_x_y_outer_tail = pd.merge(train_x_x, train_y_y, left_on = ("head_x","relation_x","tail_x"), right_on = ("head_y","relation_y","tail_y"), how ="outer").drop_duplicates(keep=False)
    more_than_one_comp_tail_tail = train_x_y_outer_tail[train_x_y_outer_tail["head_y"].isna()]
    
    more_than_one_comp_tail_tail = more_than_one_comp_tail_tail.rename(columns = {"head_x":"head","relation_x":"relation","tail_x":"tail"})
    more_than_one_comp_tail_tail = more_than_one_comp_tail_tail[["head","relation","tail"]]
    
    can_be_moved = pd.concat([more_than_one_comp_head_head, more_than_one_comp_tail_tail], join = "outer").drop_duplicates().reset_index(drop = True)
    
    can_be_moved_overlaps_inner = pd.merge(can_be_moved, overlaps_with_train, on = ("head", "relation", "tail"), how = "inner")
    can_be_moved = can_be_moved.append(can_be_moved_overlaps_inner).drop_duplicates(keep = False)
    can_be_moved = pd.concat([can_be_moved, can_be_moved_overlaps_inner], join = "outer").drop_duplicates(keep = False)
    return can_be_moved


def get_diff_between_entity_lists(df1,df2):
    df1_head = set(df1["head"])
    df1_tail = set(df1["tail"])
    entities_df1 = df1_head.union(df1_tail)
    df2_head = set(df2["head"])
    df2_tail = set(df2["tail"])
    entities_df2 = df2_head.union(df2_tail)
    return entities_df1-entities_df2    

def find_rows_to_move(from_df,to_df,entities):
    head = from_df['head']
    tail = from_df['tail']
    index_list = []
    i=-1
    for value in head:
        i = i+1
        if value in entities:
            index_list.append(i)
            #entities.remove(value)
    i = -1
    for value in tail:
        i = i+1
        if value in entities:
            index_list.append(i)
            #entities.remove(value)
    index_list=list(dict.fromkeys(index_list))
    return index_list

def move_rows(from_df, to_df, index_to_move):
    import pandas as pd
    for index in index_to_move:
        if index in from_df.index:
            # Get the row from from_df
            row_to_move = from_df.loc[index]

            # Remove the row from from_df
            from_df = from_df.drop(index)

            # Add the row to to_df
            to_df = pd.concat([to_df, row_to_move.to_frame().T])

    # Reset the index of to_df
    to_df = to_df.reset_index(drop=True)
    return from_df, to_df


def count_values(lst):
    from collections import Counter
    count_dict = Counter(lst)
    count_dict = dict(count_dict)
    return count_dict

def finding_can_be_moved(train, number_to_be_moved, can_be_moved_edges_train):
    print(f'Moving {number_to_be_moved} edges out of {len(can_be_moved_edges_train)} possible edges')
    train_entites = list(train['head'].append(train['tail']))
    entity_Count = count_values(train_entites)
    
    entity_count_df = pd.DataFrame(entity_Count.items(), columns=['entity', 'count'])
    
    entity_count_df = entity_count_df.sort_values(by = 'count', ascending = False).reset_index(drop = True)
    
    entity_count_df = entity_count_df[entity_count_df['count'] > 1]
    
    can_be_moved = pd.DataFrame(columns = ['head','relation','tail'])
    for index in can_be_moved_edges_train.index: 
        if index% 100 == 0: 
            print(f'Done with {index} out of {number_to_be_moved}')
        if can_be_moved_edges_train.iat[index,0] in list(entity_count_df['entity']) and can_be_moved_edges_train.iat[index,2] in list(entity_count_df['entity']):
            # Retrieve the row index where the first value and second value is found
            row_index_1 = entity_count_df[entity_count_df['entity'] == can_be_moved_edges_train.iat[index, 0]].index[0]
            row_index_2 = entity_count_df[entity_count_df['entity'] == can_be_moved_edges_train.iat[index, 2]].index[0]
            # Decrement the count for the first entity and second entity
            entity_count_df.at[row_index_1, 'count'] -= 1
            entity_count_df.at[row_index_2, 'count'] -= 1
            entity_count_df = entity_count_df[entity_count_df['count'] > 1]
            can_be_moved = can_be_moved.append(can_be_moved_edges_train.iloc[index])
            if len(can_be_moved) >= number_to_be_moved:
                break
    return can_be_moved

def making_inductive_set_with_overlaps(must_be_in_train, must_be_in_test, must_be_in_valid, full_dataset):
    #finding those the can be sampled from, be by removeing those the have to be in train, valid and test from the full dataset. 
    can_be_sampled_from = pd.concat([must_be_in_train, must_be_in_test, must_be_in_valid, full_dataset], join = "outer").drop_duplicates(keep = False)

    #Making the train dataset, by first finding the sample size. The sample size should be 80% of the full dataset - the number of edges in must_be_in_train 
    sampel_size_train = int((len(full_dataset)*0.8)-len(must_be_in_train))
    new_train_no_overlaps = can_be_sampled_from.sample(sampel_size_train, random_state = 42)

    #removing the edges that have be put into train, from can_be_sampled_from, so ensure no data leakage.
    can_be_sampled_from = pd.concat([new_train_no_overlaps, can_be_sampled_from], join = "outer").drop_duplicates(keep = False)

    sampel_size_test = int((len(full_dataset)*0.1)-len(must_be_in_test))
    new_test_no_overlaps = can_be_sampled_from.sample(sampel_size_test, random_state = 42)

    new_valid_no_overlaps = pd.concat([new_test_no_overlaps, can_be_sampled_from], join = "outer").drop_duplicates(keep = False)
    #Reset indexes:
    new_train_no_overlaps.reset_index(drop=True, inplace = True)
    new_test_no_overlaps.reset_index(drop=True, inplace = True)
    new_valid_no_overlaps.reset_index(drop=True, inplace = True)
    return new_train_no_overlaps, new_test_no_overlaps, new_valid_no_overlaps

def make_transductive(train,test,valid):
    #Move from test to train
    diff_entities = get_diff_between_entity_lists(test,train)
    rows_to_move = find_rows_to_move(test,train,diff_entities)
    almost_test, almost_train = move_rows(test, train, rows_to_move)
    amount_to_move = len(rows_to_move)
    #Find rows in train which can be moved back (Only return edges which can't cause data leakage)
    can_be_moved = can_be_moved_pharm(train)
    #Move from train to test
    to_move = finding_can_be_moved(almost_train, amount_to_move, can_be_moved)
    new_test = pd.concat([almost_test, to_move], join="outer").reset_index(drop = True)
    new_train = pd.concat([almost_train, to_move], join="outer").drop_duplicates(keep=False).reset_index(drop = True)
    #Move from valid to train
    diff_entities = get_diff_between_entity_lists(valid,new_train)
    rows_to_move = find_rows_to_move(valid,new_train,diff_entities)
    almost_valid, almost_train = move_rows(valid, new_train, rows_to_move)
    amount_to_move = len(rows_to_move)
    #Move from train to valid
    can_be_moved = pd.concat([can_be_moved, to_move], join = "outer").drop_duplicates(keep=False).reset_index(drop = True)
    to_move = finding_can_be_moved(almost_train, amount_to_move, can_be_moved)
    new_valid = pd.concat([almost_valid, to_move], join="outer").reset_index(drop = True)
    new_train = pd.concat([almost_train, to_move], join="outer").drop_duplicates(keep=False).reset_index(drop = True)
    return new_train,new_test,new_valid

def save_set(train,test,valid,path):
    train.to_csv(path+'/new_train.tsv', sep="\t", index = False, header=False)

    test.to_csv(path+'/new_test.tsv', sep="\t", index = False, header=False)

    valid.to_csv(path+'/new_valid.tsv', sep="\t", index = False, header=False)
    

def find_edges_with_data_leakage(train, test, valid):
    print('Making the train edges:')
    train_edges = set()
    train_size = len(train)
    for ind in train.index:
        if ind%10000 == 0:
            print(f'{ind/train_size*100}% done')
        head_tail = (train.iat[ind,0],train.iat[ind,2])
        tail_head = (train.iat[ind,2],train.iat[ind,0])
        train_edges.add(head_tail)
        train_edges.add(tail_head)
    print('Checking test edges against train edges')
    test_edges_to_move = set()
    for ind in test.index:
        if (test.iat[ind,0],test.iat[ind,2]) in train_edges:
            test_edges_to_move.add((test.iat[ind,0],test.iat[ind,2]))
    print('Checking valid edges against train edges')
    valid_edges_to_move = set()
    for ind in valid.index:
        if (valid.iat[ind,0],valid.iat[ind,2]) in train_edges:
            valid_edges_to_move.add((valid.iat[ind,0],valid.iat[ind,2]))
    return test_edges_to_move, valid_edges_to_move
    
def get_split_sizes(train,test,valid, dataset = ""):
    train_size = len(train)
    test_size = len(test)
    valid_size = len(valid)
    full_size = train_size+test_size+valid_size
    print(f'Size of new splits {dataset}:\nTrain:{round(train_size/full_size*100,2)}%\nTest:{round(test_size/full_size*100,2)}%\nValid:{round(valid_size/full_size*100,2)}%')

def move_data_leakage_edges(from_df, to_df, edge_set_to_move):
    indexes_to_move = []
    for ind in from_df.index:
        if (from_df.iat[ind,0],from_df.iat[ind,2]) in edge_set_to_move:
            indexes_to_move.append(ind)

    from_df, to_df = move_rows(from_df, to_df, indexes_to_move)

    return from_df, to_df

    
#MAKING PHARMKG8k
print('Starting on PharmKG8k split')
path_load_P = os.path.join(ROOT_DIR, "PharmKG8k", "inductive_without_data_leakage")
train_P = pd.read_table(path_load_P+("/new_train.tsv"), names = ["head", "relation", "tail"])
test_P = pd.read_table(path_load_P+("/new_test.tsv"), names = ["head", "relation", "tail"])
valid_P = pd.read_table(path_load_P+("/new_valid.tsv"), names = ["head", "relation", "tail"])

new_train_P, new_test_P, new_valid_P = make_transductive(train_P, test_P, valid_P)

#MAKING NDFRT
print('Starting on NDFRT_DDA split')
path_load_N = os.path.join(ROOT_DIR, "BioNev", "NDFRT_DDA")

ndfrt_full = pd.read_table(path_load_N+("/all_NDFRT_DDA_edges_with_UMLS.tsv"), names = ["head", "relation", "tail"])

overlap_train_P_full_N = inner_join(ndfrt_full, new_train_P)
overlap_test_P_full_N = inner_join(ndfrt_full, new_test_P)
overlap_valid_P_full_N = inner_join(ndfrt_full, new_valid_P)

train_N, test_N, valid_N = making_inductive_set_with_overlaps(overlap_train_P_full_N, overlap_test_P_full_N, overlap_valid_P_full_N, ndfrt_full)

new_train_N = pd.concat([train_N, overlap_train_P_full_N], join = 'outer').reset_index(drop = True)
new_test_N = pd.concat([test_N, overlap_test_P_full_N], join = 'outer').reset_index(drop = True)
new_valid_N = pd.concat([valid_N, overlap_valid_P_full_N], join = 'outer').reset_index(drop = True)

new_train_N, new_test_N, new_valid_N = make_transductive(new_train_N, new_test_N, new_valid_N)

#Making CTD sets
print('Starting on CTD_DDA split')
path_load_N = os.path.join(ROOT_DIR, "BioNev", "CTD_DDA")

ctd_full = pd.read_table(path_load_N+("/all_CTD_DDA_edges_with_UMLS.tsv"), names = ["head", "relation", "tail"])

overlap_train_P_full_C = inner_join(ctd_full, new_train_P)
overlap_test_P_full_C = inner_join(ctd_full, new_test_P)
overlap_valid_P_full_C = inner_join(ctd_full, new_valid_P)

overlap_train_N_full_C = inner_join(ctd_full, new_train_N)
overlap_test_N_full_C  = inner_join(ctd_full, new_test_N)
overlap_valid_N_full_C = inner_join(ctd_full, new_valid_N)

all_overlaps_train_C = pd.concat([overlap_train_P_full_C, overlap_train_N_full_C], join = 'outer').reset_index(drop = True)
all_overlaps_test_C = pd.concat([overlap_test_P_full_C, overlap_test_N_full_C], join = 'outer').reset_index(drop = True)
all_overlaps_valid_C = pd.concat([overlap_valid_P_full_C, overlap_valid_N_full_C], join = 'outer').reset_index(drop = True)

train_C, test_C, valid_C = making_inductive_set_with_overlaps(all_overlaps_train_C, all_overlaps_test_C, all_overlaps_valid_C, ctd_full)

new_train_C = pd.concat([train_C, all_overlaps_train_C], join = 'outer').reset_index(drop = True)
new_test_C = pd.concat([test_C, all_overlaps_test_C], join = 'outer').reset_index(drop = True)
new_valid_C = pd.concat([valid_C, all_overlaps_valid_C], join = 'outer').reset_index(drop = True)

new_train_C, new_test_C, new_valid_C = make_transductive(new_train_C, new_test_C, new_valid_C)


#Making the full dataset:
new_train_A = pd.concat([new_train_P, new_train_N, new_train_C], join = 'outer').reset_index(drop = True)
new_test_A = pd.concat([new_test_P, new_test_N, new_test_C], join = 'outer').reset_index(drop = True)
new_valid_A = pd.concat([new_valid_P, new_valid_N, new_valid_C], join = 'outer').reset_index(drop = True)


#Find edges with data_leakage in full split:
test_edge_set, valid_edge_set = find_edges_with_data_leakage(new_train_A, new_test_A, new_valid_A)

#Moving data leakage edges from test and valid to train:
#Pharm
new_test_P, new_train_P = move_data_leakage_edges(new_test_P, new_train_P, test_edge_set)
new_valid_P, new_train_P = move_data_leakage_edges(new_valid_P, new_train_P, valid_edge_set)

#NDFRT
new_test_N, new_train_N = move_data_leakage_edges(new_test_N, new_train_N, test_edge_set) 
new_valid_N, new_train_N = move_data_leakage_edges(new_valid_N, new_train_N, valid_edge_set)

#CTD
new_test_C, new_train_C = move_data_leakage_edges(new_test_C, new_train_C, test_edge_set) 
new_valid_C, new_train_C = move_data_leakage_edges(new_valid_C, new_train_C, valid_edge_set)    

#Making the new full split:
new_train_A = pd.concat([new_train_P, new_train_N, new_train_C], join = 'outer').reset_index(drop = True)
new_test_A = pd.concat([new_test_P, new_test_N, new_test_C], join = 'outer').reset_index(drop = True)
new_valid_A = pd.concat([new_valid_P, new_valid_N, new_valid_C], join = 'outer').reset_index(drop = True)


#Checking for transductive
print('Checking if PharmKG8k is transductive:')
check_entity_overlap(new_train_P, new_test_P, new_valid_P)
print('Checking if NDFRT_DDA is transductive:')
check_entity_overlap(new_train_N, new_test_N, new_valid_N)
print('Checking if CTD_DDA is transductive:')
check_entity_overlap(new_train_C, new_test_C, new_valid_C)
print('Checking if Full_Split is transductive:')
check_entity_overlap(new_train_A, new_test_A, new_valid_A)
print("Checking if Full_Split has data leakage (Note if this set doesn't have it, none of them have)")
check_edge_overlap(new_train_A, new_test_A, new_valid_A)

get_split_sizes(new_train_P,new_test_P,new_valid_P, "PharmKG8k")
get_split_sizes(new_train_N, new_test_N, new_valid_N, "NDFRT")
get_split_sizes(new_train_C, new_test_C, new_valid_C, "CTD")
get_split_sizes(new_train_A, new_test_A, new_valid_A, "Full_Split")


path_save_P = os.path.join(ROOT_DIR, "PharmKG8k", "transductive_without_data_leakage")
path_save_N = os.path.join(ROOT_DIR, "BioNev", "NDFRT_DDA", 'transductive_split')
path_save_C = os.path.join(ROOT_DIR, "BioNev", "CTD_DDA", 'transductive_split')
path_save_A = os.path.join(ROOT_DIR, "full_splits", "transductive_without_data_leakage")

save_set(new_train_P,new_test_P,new_valid_P,path_save_P)

save_set(new_train_N, new_test_N, new_valid_N, path_save_N)

save_set(new_train_C, new_test_C, new_valid_C, path_save_C)

save_set(new_train_A,new_test_A,new_valid_A,path_save_A)