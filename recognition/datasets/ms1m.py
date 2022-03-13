import os
import io

import torch
import random
from torch.nn.functional import dropout
from torch.utils import data
import torchvision.transforms.functional as TF
from PIL import Image
import pickle

from .transforms import base_transform


class ClassRandomSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, batch_size, samples_per_class, image_dict, image_list):
        self.image_dict = image_dict
        self.image_list = image_list

        self.classes = list(self.image_dict.keys())
        
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.sampler_length = len(image_list) // batch_size
        assert self.batch_size % self.samples_per_class == 0, '#Samples per class must divide batchsize!'

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            draws = self.batch_size // self.samples_per_class

            for _ in range(draws):
                class_key = random.choice(self.classes)
                class_idx_lst = []
                for _ in range(self.samples_per_class):
                    class_idx_lst.append(random.choice(self.image_dict[class_key])[-1])        
                subset.extend(class_idx_lst)                            
            yield subset

    def __len__(self):
        return self.sampler_length


class FaceDatasets(data.Dataset):
    def __init__(self, data_path, preprocessed_file, img_size, min_img, mode='train'):
        assert mode in ['train', 'test']               
        self.mode = mode
        self.data_path = data_path
        
        if not os.path.isfile(preprocessed_file):
            info_dict = create_face_dataset(self.data_path)
            with open(preprocessed_file, 'wb') as f:
                pickle.dump(info_dict, f)
        with open(preprocessed_file, 'rb') as f:
            self.info_dict = pickle.load(f)
        
        info = []
        new_info_dict = {}
        label = 0
        count = 0
        for img_path_lst in self.info_dict.values():
            temp_img_path_lst = []
            if len(img_path_lst) < min_img:
                continue            
            for img_path in img_path_lst:
                info.append({'img_path' : img_path, 'label' : label})            
                temp_img_path_lst.append([img_path, count])
                count += 1
            # new_info_dict[label] = img_path_lst
            new_info_dict[label] = temp_img_path_lst
            label += 1
                        
        self.information = info
        self.info_dict = new_info_dict
        self.num_classes = label

        print("\nCreat kface {} dataset".format(mode))
        print("Number of images        : ", len(self.information))
        print("Number of classes       : ", self.num_classes)

        self.transform = base_transform(img_size=img_size, mode=mode)
            
    def __getitem__(self, index):
        info = self.information[index]
        img_path = info['img_path']
        label = info['label']
        img = Image.open(img_path)
        if self.mode == 'train':
            data = self.transform(img)            
            return data, label                     
        else:
            data, hfdata = self.transform(img), self.transform(TF.hflip(img))
            return data, hfdata, label, info['img_path']

    def __len__(self):
        return len(self.information)


class MS1M(object):
    def __init__(self, data_path, preprocessed_file, img_size, min_img, 
                 batch_size, cuda, workers, mode='train', rank=-1):
        if rank in [-1, 0]:            
            print("MS1M processing .. ")

        pin_memory = True if cuda else False        

        dataset = FaceDatasets(data_path, preprocessed_file, img_size, min_img, mode)            
        #batch_sampler = ClassRandomSampler(batch_size=batch_size,
        #                                   samples_per_class=2,
        #                                   image_dict=dataset.info_dict,
        #                                   image_list=dataset.information)        
        
        # print("ClassRandomSampler")
        # print(batch_sampler)
        if mode == 'train':
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None     
        loader = torch.utils.data.DataLoader(
                            dataset, 
                            batch_size=batch_size, 
                            shuffle=(mode == 'train') and (train_sampler == None),
                            num_workers=workers, 
                            pin_memory=pin_memory,
                            sampler=train_sampler,
                            # batch_sampler=batch_sampler,
                            drop_last=True
                            )

        self.dataset = dataset
        self.loader = loader        
        self.num_classes = dataset.num_classes
        self.num_training_images = len(dataset.information)
        if rank in [-1, 0]:            
            print("len {} loader".format(mode), len(self.loader))


def create_face_dataset(data_path):
    """
    -data
        -id1
            -.jpg
        -id2
            -.jpg                      
    """
    information = dict()
    identities = sorted(os.listdir(data_path))
    for idx, identity in enumerate(identities):
        if idx % 1000 == 0:
            print(idx, "preprocessing...")              
        identity_path = os.path.join(data_path, identity)
        try:
            images = os.listdir(identity_path)
        except:
            continue
        if len(images) == 0:
            continue
        img_path_list = list()
        for img in images:
            img_path = os.path.join(identity_path, img)
            img_path_list.append(img_path)
        information[idx] = img_path_list
    
    return information