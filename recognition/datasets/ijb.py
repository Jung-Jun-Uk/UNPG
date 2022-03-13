import os
import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from PIL import Image
from .transforms import base_transform


class ImgInfLoader(data.Dataset):
    def __init__(self, data_dir, ann_file, img_size):
        self.data_dir = data_dir
        self.ann_file = os.path.join(data_dir, ann_file)
        self.transform = base_transform(img_size=img_size, mode='test')
        print('=> preparing dataset for inference ...')
        self.init()
        
    def init(self):
        with open(self.ann_file) as f:
            self.imgs = f.readlines()
            
    def __getitem__(self, index):
        ls = self.imgs[index].strip().split()
        # change here
        img_path = ls[0]        
        img_path = img_path.split('/')        
        img_path = os.path.join(self.data_dir, img_path[2], img_path[3])
        #img_path = img_path.replace("data/IJB", self.data_dir)
        if not os.path.isfile(img_path):
            raise Exception('{} does not exist'.format(img_path))
            exit(1)
        img = Image.open(img_path) 
        if img is None:
            raise Exception('{} is empty'.format(img_path))
            exit(1)
        _img = TF.hflip(img)
        return [self.transform(img), self.transform(_img)], img_path

    def __len__(self):
        return len(self.imgs)


class IJB(object):
    def __init__(self, data_dir, ann_file, img_size, batch_size, cuda, workers):
        print(" IJB processing .. ")

        pin_memory = True if cuda else False        
        self.data_dir = data_dir
        
        self.dataset = ImgInfLoader(data_dir, ann_file, img_size)                    

        loader = torch.utils.data.DataLoader(
                            self.dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=workers, 
                            pin_memory=pin_memory)

        self.loader = loader                        
        print("len IJBloader", len(self.loader))        