import numpy as np
import torch
import torch.nn.functional as F
from prettytable import PrettyTable

from sklearn.metrics import roc_curve


def verification(similarities, labels, device='cpu', metric='roc'):
    assert metric in ['best_th', 'roc']
    
    if metric == 'best_th':
        similarities = torch.Tensor(similarities).to(device)
        labels = torch.Tensor(labels).to(device)    
        acc, th = torch_cal_accuracy(similarities, labels)   
        acc, th = float(acc.data.cpu()), float(th.data.cpu())   
        print('cosine verification accuracy: ', acc, 'threshold: ', th, '\n\n')    

    elif metric == 'roc':
        similarities = np.asarray(similarities)
        labels = np.asarray(labels)
        fpr, tpr, _ = roc_curve(labels, similarities)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)  # select largest tpr at same fpr
        x_labels = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
        tar_far_score = []
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(
                list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))            
            tar_far_score.append('{:0.4f}'.format(tpr[min_index] * 100))
        tar_far_table = PrettyTable()
        tar_far_table.field_names = x_labels
        tar_far_table.add_row(tar_far_score)
        print(tar_far_table, '\n\n')
        
        acc = float(tar_far_score[3]) # TAR@(FAR=1e-3)        

    return acc


def deepfeatures_extraction(model, dataloader, device, freq=50):
    model.eval()
    extracted_df_dict = dict()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            data, hfdata, labels, img_path = batch
            data, hfdata = data.to(device), hfdata.to(device)

            df = model(data)
            hfdf = model(hfdata)
            f  = torch.cat((df, hfdf), 1).data.cpu()

            for idx in range(len(labels)):
                extracted_df_dict[img_path[idx]] = {'deepfeatures' : f[idx], 'label' : labels[idx]}
            
            if (i+1) % freq == 0:
                print("Deep feature extracting ... {}/{}".format(i+1, len(dataloader)))
            
    return extracted_df_dict


def read_test_pair_dataset(test_pair_path):
        pair_data = list()
        with open(test_pair_path) as f:
            data = f.readlines()
        for d in data:
            info = d[:-1].split(' ')
            pair_data.append(info)
                                           
        return pair_data


def computing_sim_from_df(extracted_df_dict, test_pairs_txt, device, split_batch_size=1024):          
    test_pairs_lst = read_test_pair_dataset(test_pairs_txt)
    id1_deepfeatures = []
    id2_deepfeatures = []
    labels = []
        
    for id1, id2, label in test_pairs_lst:
        df1 = extracted_df_dict[id1]['deepfeatures']
        df2 = extracted_df_dict[id2]['deepfeatures']
        
        id1_deepfeatures.append(df1)
        id2_deepfeatures.append(df2)
        labels.append(int(label))
            
    id1_deepfeatures = torch.stack(id1_deepfeatures, dim=0)
    id2_deepfeatures = torch.stack(id2_deepfeatures, dim=0)
    
    split_df1 = torch.split(id1_deepfeatures, split_batch_size)
    split_df2 = torch.split(id2_deepfeatures, split_batch_size)
    
    similarities = []
    for i, (df1, df2) in enumerate(zip(split_df1, split_df2)):
        df1 = df1.to(device)
        df2 = df2.to(device)
        sim = batch_cosine_similarity(df1,df2)
        similarities.extend(sim.data.cpu().tolist())
    
    return similarities, labels
    

def computing_sim_from_data(model, dataloader, device):
    model.eval()    
    similarities = []
    labels = []
    with torch.no_grad():
        for i, data in enumerate(dataloader): 
            data1, hfdata1, data2, hfdata2, label = data
            data1, hfdata1, data2, hfdata2 = \
                data1.to(device), hfdata1.to(device), \
                data2.to(device), hfdata2.to(device)
            
            df1    = model(data1)
            hfdf1   = model(hfdata1)
            df2    = model(data2)
            hfdf2   = model(hfdata2)
            
            f1  = torch.cat((df1, hfdf1), 1)
            f2  = torch.cat((df2, hfdf2), 1)

            similarities.extend(batch_cosine_similarity(f1,f2).data.cpu().tolist())
            labels.extend(label.data.tolist())

    return similarities, labels


def batch_cosine_similarity(x1, x2):
    """
    ex) x1 size [256, 512], x2 size [256, 512]
    similarity size = [256, 1]
    """    
    x1 = F.normalize(x1).unsqueeze(1)
    x2 = F.normalize(x2).unsqueeze(1)    
    x2t = torch.transpose(x2, 1, 2)
    similarity = torch.bmm(x1, x2t).squeeze()
    return similarity    


def torch_cal_accuracy(y_score, y_true, freq=10000):    
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th).long()
        acc = torch.mean((y_test == y_true).float())
        if acc > best_acc:
            best_acc = acc
            best_th = th
        if (i+1) % freq == 0 or (i+1) == len(y_score):
            print('Progress {}/{}'.format((i+1),len(y_score)))
    return best_acc, best_th    


