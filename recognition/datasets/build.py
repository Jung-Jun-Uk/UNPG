import os

from .kface import KFace
from .ms1m import MS1M
from .bin_datasets import BIN
from .ijb import IJB


def build_datasets(data_cfg, batch_size, cuda, workers, mode, rank=-1):
    assert mode in ['train', 'test']    
    cfg = data_cfg[mode]
    if cfg['dataset'] == 'kface':                   
        dataset = KFace(cfg['data_path'], cfg['test_idx_txt'], cfg['acs'], cfg['lux'], cfg['eps'], cfg['pose'], 
                        cfg['img_size'], batch_size, cuda, workers, mode=mode)
    elif cfg['dataset'] == 'ms1m':
        dataset = MS1M(cfg['data_path'], cfg['preprocessed_file'], cfg['img_size'], cfg['min_img'],
                       batch_size, cuda, workers, mode=mode, rank=rank)
    elif cfg['dataset'] == 'bin':                        
        root, file_names = cfg['root'], cfg['file_names']        
        if isinstance(file_names, str):                         
            data_path = os.path.join(root, file_names)
            dataset = BIN(data_path, cfg['img_size'], batch_size, cuda, workers)
        elif isinstance(file_names, list):
            data_path = [os.path.join(root, f) for f in file_names]
            dataset = [BIN(dp, cfg['img_size'], batch_size, cuda, workers) for dp in data_path]
                                
    elif cfg['dataset'] in ['ijbb', 'ijbc']:        
        dataset = IJB(cfg['root'], cfg['inf_list'], cfg['img_size'], batch_size, cuda, workers)

    return dataset
