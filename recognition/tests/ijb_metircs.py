import numpy as np
import torch
import torch.nn as nn
from prettytable import PrettyTable

from sklearn.metrics import roc_curve


def ijb_feature_extraction_and_saving(model, dataloader, df_path, device, freq=25):
    print('=> starting inference engine ...', 'green')
    print('=> embedding features will be saved into {}'.format(df_path))

    model.eval()
    fio = open(df_path, 'w')
    with torch.no_grad():
        for i, (input, img_paths) in enumerate(dataloader):                    
            # compute output            
            embedding_feat = model(input[0].to(device))                        
            _feat = embedding_feat.data.cpu().numpy()
            # write feat into files
            for feat, path in zip(_feat, img_paths):
                fio.write('{} '.format(path))
                for e in feat:
                    fio.write('{} '.format(e))
                fio.write('\n')

            if (i+1) % freq == 0:
                print("Deep feature extracting ... {}/{}".format(i+1, len(dataloader)))                
    # close
    fio.close()


def read_template_media_list(path):
    ijb_meta, templates, medias = [], [], []
    with open(path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split(' ') # ex) ['1.jpg', '1', '69544']
        ijb_meta.append(parts[0])
        templates.append(int(parts[1]))
        medias.append(int(parts[2]))
    return np.array(templates), np.array(medias)    
    

def read_template_pair_list(path):
    t1, t2, label = [], [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        data = line.strip().split(' ')
        t1.append(int(data[0]))
        t2.append(int(data[1]))
        label.append(int(data[2]))
    return np.array(t1), np.array(t2), np.array(label)


def read_feats(feat_list, embedding_size):        
    with open(feat_list, 'r') as f:
        lines = f.readlines()        
    img_feats = []
    for line in lines:
        data = line.strip().split(' ')                  
        img_feats.append([float(ele) for ele in data[1:1+embedding_size]])
    img_feats = np.array(img_feats).astype(np.float32)
    return img_feats


def image2template_feature(img_feats=None,
                           templates=None,
                           medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
    
    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
                
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(
            face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)

            media_norm_feats += [np.mean(face_norm_feats[ind_m],
                                         0, keepdims=True)]
        
        media_norm_feats = np.array(media_norm_feats)        
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)        

        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(count_template))

    template_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    return template_feats, unique_templates


def computing_similarity_score(template_feats, unique_templates, p1=None, p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates)+1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    scores = np.zeros((len(p1),))   # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    # small batchsize instead of all pairs in one batch due to the memory limiation
    batchsize = 100000
    sublists = [total_pairs[i:i + batchsize]
                for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_feats[template2id[p1[s]]]
        feat2 = template_feats[template2id[p2[s]]]
        # similarity_score = distance_(feat1, feat2)
        similarity_score = np.sum(feat1 * feat2, -1)
        #scores[s] = 1-similarity_score
        scores[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return scores


def distance_(embeddings0, embeddings1):    
    cos = nn.CosineSimilarity(dim=1, eps=0)
    simi = torch.clamp(cos(embeddings0, embeddings1), min=-1, max=1)
    return simi.cpu().numpy()


def computing_sim_from_ijb_df(media_txt, pair_label_txt, df_path, embedding_size=512):
    # load the data
    templates, medias = read_template_media_list(media_txt)    
    p1, p2, labels = read_template_pair_list(pair_label_txt)        
    img_feats = read_feats(df_path, embedding_size)
    
    # calculate scores
    template_feats, unique_templates = image2template_feature(img_feats, templates, medias)                                                                      
    similarities = computing_similarity_score(template_feats, unique_templates, p1, p2)    
    return similarities, labels
