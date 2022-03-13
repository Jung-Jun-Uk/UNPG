import os
import io

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from PIL import Image
import pickle

from .transforms import base_transform

"""
    ###################################################################
    Bin datasets list: LFW, CFP-FP, AgeDB-30, CALFW, and CPLFW  
    Bin dataasets is contained within the MS1M-ArcFace dataset.
    MS1M-ArcFace can be downloaded in below url.
    url: https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_
    ###################################################################
"""

class BinDatasets(data.Dataset):
    def __init__(self, bin_path, img_size):        
        bins, issame_list = pickle.load(open(bin_path, 'rb'), encoding='bytes')
        self.information = list()
        p_label_count = 0
        n_label_count = 0
        for i in range(0, len(issame_list)*2, 2):
            data1, data2, label = bins[i], bins[i+1], issame_list[int(i/2)]
            self.information.append({'data1' : data1, 'data2' : data2, 'label' : label}) 
            if label == 1:
                p_label_count += 1 
            else:
                n_label_count += 1
        print(bin_path, len(self.information), p_label_count, n_label_count)        
        self.transform = base_transform(img_size=img_size, mode='test')

    def __getitem__(self, index):
        info = self.information[index]
        data1 = info['data1']
        data2 = info['data2']
        label = info['label']
        
        data1 = Image.open(io.BytesIO(data1))
        data2 = Image.open(io.BytesIO(data2))

        data1, hfdata1 = self.transform(data1), self.transform(TF.hflip(data1))
        data2, hfdata2 = self.transform(data2), self.transform(TF.hflip(data2))

        return data1, hfdata1, data2, hfdata2, label
    
    def __len__(self):
        return len(self.information)


class BIN(object):
    def __init__(self, data_path, img_size, batch_size, cuda, workers):
        print(" BIN processing .. ")

        pin_memory = True if cuda else False        
        
        self.data_path = data_path
        self.dataset = BinDatasets(data_path, img_size)                    

        loader = torch.utils.data.DataLoader(
                            self.dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=workers, 
                            pin_memory=pin_memory)

        self.loader = loader        
        self.num_training_images = len(self.dataset.information)
        
        print("len binloader", len(self.loader))