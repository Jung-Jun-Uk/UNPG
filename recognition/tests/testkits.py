import os
import torch
from .common_metrics import computing_sim_from_df, computing_sim_from_data, \
    verification, deepfeatures_extraction
from .ijb_metircs import ijb_feature_extraction_and_saving, computing_sim_from_ijb_df


def ijb1v1verification(model, dataloader, df_path, media_txt, pair_label_txt, 
                       device, embedding_size=512, metric='roc', re_eval=False):
    print(df_path)
    if not os.path.isfile(df_path) or re_eval:            
        ijb_feature_extraction_and_saving(model, dataloader, df_path, device=device)            
    similarities, labels = computing_sim_from_ijb_df(media_txt, pair_label_txt, df_path, embedding_size)
    acc = verification(similarities, labels, metric=metric)
    return acc


def face1v1verification(model, dataloader, device, metric):    
    similarities, labels = computing_sim_from_data(model, dataloader, device)
    acc = verification(similarities, labels, metric=metric)
    return acc


def kface1v1verification(test_pairs_txt, df_path, model, dataloader, device, metric, train=False, re_eval=False):        
    if not os.path.isfile(df_path) or train or re_eval:
        extracted_df_dict = deepfeatures_extraction(model, dataloader, device)
        torch.save(extracted_df_dict, df_path)    
    extracted_df_dict = torch.load(df_path)
        
    if isinstance(test_pairs_txt, str):
        print("Test txt file: ", test_pairs_txt)        
        similarities, labels = computing_sim_from_df(extracted_df_dict, test_pairs_txt, device)
        acc = verification(similarities, labels, metric=metric)
    elif isinstance(test_pairs_txt, list):
        for t_p_txt in test_pairs_txt:
            print("Test txt file: ", t_p_txt)
            similarities, labels = computing_sim_from_df(extracted_df_dict, t_p_txt, device)
            acc = verification(similarities, labels, device, metric=metric)            
    return acc

    
    