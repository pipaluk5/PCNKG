Guide to running the code to create split:
1. give_pharm_umls_cui.py
2. making_pharm_no_data_leakage.py
3. split_BioNev_test_up_to_get_valid.py
4. making_the_transductive_split_with_no_data_leakage_for_all_the_datasets.py
5. make_split_for_ndfrt_new_edges.py

Files to run models:
(Choose the right path, all the paths are in there, some of them commented out)

ComplEx:
ComplEx_maja.py

ConvKB:
ConvKB.py

TransE:
TransE.py

Run hyperparameter_optimization for complex NDFRT without new edges:
ndfrt_ComplEx_hyper_parameter_optimization.py
