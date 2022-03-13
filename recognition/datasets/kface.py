import os

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from PIL import Image

from .transforms import base_transform


"""
    ###################################################################

    K-Face : Korean Facial Image AI Training Dataset
    url    : http://www.aihub.or.kr/aidata/73

    Directory structure : High-ID-Accessories-Lux-Emotion
    ID example          : '19062421' ... '19101513' len 400
    Accessories example : 'S001', 'S002' .. 'S006'  len 6
    Lux example         : 'L1', 'L2' .. 'L30'       len 30
    Emotion example     : 'E01', 'E02', 'E03'       len 3
    S001 - L1, every emotion folder contaions a information txt file
    (ex. bbox, facial landmark) 
    
    ###################################################################
"""


class KFaceDatasets(data.Dataset):
    def __init__(self, data_path, test_idx_txt, img_size, acs, lux, eps, pose, mode='train', double=False):
        assert mode in ['train', 'test']        
        self.mode = mode
        self.data_path = data_path
        self.information, self.num_classes = create_kface_dataset(
                                                data_path=self.data_path,                                     
                                                test_idx_txt=test_idx_txt,
                                                accessories=acs, 
                                                luces=lux, 
                                                expressions=eps, 
                                                poses=pose, 
                                                mode=mode)
        
        print("\nCreat kface {} dataset".format(mode))
        print("Number of images        : ", len(self.information))
        print("Number of classes       : ", self.num_classes)
        print('Aceessories : ',acs)
        print('Lux         : ',lux)
        print('Expression  : ',eps)
        print('Pose        : ',pose)

        self.transform = base_transform(img_size=img_size, mode=mode)
        
    def __getitem__(self, index):
        info = self.information[index]
        img_path = os.path.join(self.data_path, info['img_path']) 
        label    = info['label']        
        img = Image.open(img_path)
        if self.mode == 'train':        
            data = self.transform(img)                        
            return data, label
        else:
            img = Image.open(img_path)
            data, hfdata = self.transform(img), self.transform(TF.hflip(img))
            return data, hfdata, label, info['img_path']
        
    def __len__(self):
        return len(self.information)


class KFace(object):    
    def __init__(self, data_path, test_idx_txt, acs, lux, eps, pose, img_size,
                 batch_size, cuda, workers, mode='train'):        
        print("KFace processing .. ")    
        
        pin_memory = True if cuda else False        
        acs = kface_accessories_converter(acs)
        lux = kface_luces_converter(lux)
        eps = kface_expressions_converter(eps)
        pose = kface_pose_converter(pose)

        dataset = KFaceDatasets(data_path, test_idx_txt, img_size, 
                                acs, lux, eps, pose, mode=mode)
        
        loader = torch.utils.data.DataLoader(
                            dataset, 
                            batch_size=batch_size, 
                            shuffle=(mode == 'train'),
                            num_workers=workers, 
                            pin_memory=pin_memory)

        self.loader = loader
        self.num_classes = dataset.num_classes
        self.num_training_images = len(dataset.information)
                                
        print("len {} loader".format(mode), len(self.loader))
        
    
def read_test_pair_dataset(test_pair_path):
        pair_data = list()
        with open(test_pair_path) as f:
            data = f.readlines()
        for d in data:
            info = d[:-1].split(' ')
            pair_data.append(info)
                                           
        return pair_data


def create_kface_dataset(data_path, test_idx_txt, 
                         accessories, luces, expressions, poses, mode='train'):
        
    assert isinstance(accessories, list)
    assert isinstance(luces, list)
    assert isinstance(expressions, list)
    assert isinstance(poses, list)
    
    with open(test_idx_txt) as f:
        except_lst = f.read().split('\n')[:-1]        
    identity_lst = sorted(os.listdir(data_path))
    information = list()
    
    if mode == 'train':
        identity_lst = list(set(identity_lst) - set(except_lst))
    elif mode == 'test':
        identity_lst = except_lst
    else:
        assert ValueError
    
    for i, idx in enumerate(identity_lst):
        num_labels = i+1
        # print(num_labels, "preprocessing..")
        for a in accessories:
            for l in luces:
                for e in expressions:
                    for p in poses:
                        image_path = os.path.join(idx, a, l, e, p) + '.jpg'
                        label = i
                        information.append({'img_path' : image_path, 'label' : label})
    return information, num_labels


def kface_accessories_converter(accessories):
    """
        ex) 
        parameters  : S1~3,5
        return      : [S001,S002,S003,S005]
    """

    accessories = accessories.lower()
    assert 's' == accessories[0]
    
    alst = []
    accessries = accessories[1:].split(',')
    for acs in accessries:
        acs = acs.split('~')
        if len(acs) == 1:
            acs = ['S' + acs[0].zfill(3)]
        else:
            acs = ['S' + str(a).zfill(3) for a in range(int(acs[0]), int(acs[1])+1)]
        alst.extend(acs)
    return alst

def kface_luces_converter(luces):
    """
        ex) 
        parameters  : L1~7,10~15,20~30
        return      : [L1, ... , L7, L10, ... L15, L20, ... , L30]
    """
    luces = luces.lower()
    assert 'l' == luces[0]

    llst = []
    luces = luces[1:].split(',')
    for lux in luces:
        lux = lux.split('~')
        if len(lux) == 1:
            lux = ['L' + lux[0]]
        else:
            lux = ['L' + str(l) for l in range(int(lux[0]), int(lux[1])+1)]
        llst.extend(lux)
    return llst


def kface_expressions_converter(expressions):
    """
        ex) 
        parameters  : E1~3
        return      : [E01, E02, E03]
    """
    expressions = expressions.lower()
    assert 'e' == expressions[0]    
    elst = []
    expressions = expressions[1:].split(',')
    for eps in expressions:
        eps = eps.split('~')
        if len(eps) == 1:
            eps = ['E' + eps[0].zfill(2)]
        else:
            eps = ['E' + str(e).zfill(2) for e in range(int(eps[0]), int(eps[1])+1)]
        elst.extend(eps)
    return elst


def kface_pose_converter(poses):
    """
        ex) 
        parameters  : C1,3,5,10~20
        return      : [C1,C3,C5,C10, ..., C20]
    """
    poses = poses.lower()
    assert 'c' == poses[0]    

    plst = []
    poses = poses[1:].split(',')
    for pose in poses:
        pose = pose.split('~')
        if len(pose) == 1:
            pose = ['C' + pose[0]]
        else:
            pose = ['C' + str(p) for p in range(int(pose[0]), int(pose[1])+1)]
        plst.extend(pose)
    return plst