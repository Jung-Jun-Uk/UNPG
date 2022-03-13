from os import O_TRUNC
from shutil import disk_usage
from numpy import deprecate
import yaml
import torch
import random
from torch import embedding, nn
import torch.nn.functional as F
from typing import Tuple

from .iresnet import iresnet
from .head import ArcFace, CosFace, UNPG
from .magface import MagFace
from pytorch_metric_learning import losses


#from .unpg_fw import UNPG_FW
#from .unpg_ff_fw import UNPG_FF_FW


def build_models(model_name, drop_lastfc=True):
    model_args = model_name.split('-')    
    
    if model_args[0] == 'iresnet':
        _, net_depth = model_args
        model = iresnet(num_layers=int(net_depth), drop_lastfc=drop_lastfc)    
    return model


def box_and_whisker_algorithm(similarities, wisk):        
    l = similarities.size(0)
    sorted_x = torch.sort(input=similarities, descending=False)[0]
    
    lower_quartile = sorted_x[int(0.25 * l)]
    upper_quartile = sorted_x[int(0.75 * l)]
            
    IQR = (upper_quartile - lower_quartile)        
    minimum = lower_quartile - wisk * IQR        
    maximum = upper_quartile + wisk * IQR
    mask = torch.logical_and(sorted_x <= maximum, sorted_x >= minimum)
    sn_prime = sorted_x[mask]
    return sn_prime


def convert_label_to_similarity(normed_feature, label):
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return (similarity_matrix[positive_matrix], similarity_matrix[negative_matrix])


class HeadAndLoss(nn.Module):
    def __init__(self, in_feature, num_classes, criterion, head_name, aux_name, head_zoo='data/head.zoo.yaml'):
        super(HeadAndLoss, self).__init__()
        
        self.criterion = criterion
        self.head_name = head_name
        self.aux_name = aux_name
        
        # Head config file        
        with open(head_zoo) as f:
            head_zoo = yaml.load(f, Loader=yaml.FullLoader)        
        opt = head_zoo.get(head_name)
        # assert opt != None                            
        self.head_cfg = {head_name : opt}        
        print(self.head_cfg)

        print("Head config file", self.head_name)
        print(self.head_cfg)    
        print(aux_name)        
        aux_opt = head_zoo.get('unpg') if 'unpg' in aux_name else None        
        self.aux_cfg = {aux_name : aux_opt}        
        print("Aux config file", self.aux_name)
        print(self.aux_cfg)    
        
        if head_name == 'arcface':
            self.head = ArcFace(in_feature=in_feature, out_feature=num_classes, s=opt['s'], m=opt['m'])        
        elif head_name == 'cosface':
            self.head = CosFace(in_feature=in_feature, out_feature=num_classes, s=opt['s'], m=opt['m'])                
        elif head_name == 'magface':
            self.head = MagFace(in_feature=in_feature, out_feature=num_classes, s=opt['s'],
                                l_margin=opt['l_margin'], u_margin=opt['u_margin'], l_a=opt['l_a'],
                                u_a=opt['u_a'], lambda_g=opt['lambda_g'])        
                                 
        if 'unpg' in aux_name:
            self.aux = UNPG(s=opt['s'], wisk=aux_opt['wisk'])                   
            self.wisk = aux_opt['wisk']
        elif 'triplet' in aux_name:            
            self.aux = losses.TripletMarginLoss(margin=0.5)
        elif 'contrastive' in aux_name:
            self.aux = losses.ContrastiveLoss()
                
        self.num_gpu = torch.cuda.device_count()
                           
    def forward(self, deep_features, labels, rank=-1): 
        
        #if rank != -1: # We found performance degradation during work distributed training for metric losses. We will fix it later.
        #    deep_features, labels = gather(deep_features, labels)

        loss, opt_loss, regular_g = 0, 0, 0
        if self.head_name in ['arcface', 'cosface']:            
            cosine = self.head(deep_features, labels) 
                    
        norm_x = F.normalize(deep_features)
        sp, sn = convert_label_to_similarity(norm_x, labels)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        if 'unpg' in self.aux_name:
            aux_sn = []
            sn_prime = box_and_whisker_algorithm(sn, wisk=self.wisk)                
            aux_sn.append(sn_prime)            
            aux_sn = torch.cat(aux_sn, dim=0)           
            aux_sn = torch.stack([aux_sn] * self.num_gpu, dim=0)                        
            loss += self.aux(cosine, aux_sn, labels)                        
        else:
            if self.num_gpu > 1:
                loss += self.criterion(self.head.module.s * cosine, labels)
            else:
                loss += self.criterion(self.head.s * cosine, labels)            
                        
        if 'triplet' in self.aux_name:            
            loss += self.aux(deep_features, labels)            
        elif 'contrastive' in self.aux_name:                  
            loss += self.aux(deep_features, labels)    
        elif 'circle' in self.aux_name:
            loss += self.aux(deep_features, labels)
                        
        return loss
        

def gather(embeddings, labels):    
    return None, None







