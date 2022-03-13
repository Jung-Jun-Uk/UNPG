import yaml
import os
import sys
from pathlib import Path
import sys
import argparse
import torch

from datasets.build import build_datasets
from utils.general import select_device, Logger
from tests.testkits import kface1v1verification, face1v1verification, ijb1v1verification


def evals(data_cfg, save_dir, model, datasets, device, wname='last', train=False, re_eval=False):    
    cfg = data_cfg['test']
    if cfg['dataset'] == 'kface':
        save_feat_dir = save_dir / 'kface'
        save_feat_dir.mkdir(parents=True, exist_ok=True)  # make dir
        df_path = save_feat_dir / (wname + '_deepfeatures.pth')                   
        acc = kface1v1verification(cfg['test_pairs_txt'], df_path, model, datasets.loader, device, metric=cfg['metric'], train=train)

    elif cfg['dataset'] == 'bin':                
        if isinstance(datasets, list):
            for dataset in datasets:         
                print("Test data_path: ", dataset.data_path)       
                acc = face1v1verification(model, dataset.loader, device, cfg['metric'])
        else:
            acc = face1v1verification(model, datasets.loader, device, cfg['metric'])
        
    elif cfg['dataset'] in ['ijbb', 'ijbc']:
        save_feat_dir = save_dir / cfg['dataset'] 
        save_feat_dir.mkdir(parents=True, exist_ok=True)  # make dir
        df_path = save_feat_dir / (wname + '.feat.list')        
        media_txt = Path(cfg['root']) / cfg['media_txt']
        pair_label_txt = Path(cfg['root']) / cfg['pair_label_txt']
        acc = ijb1v1verification(model, datasets.loader, df_path, media_txt,
                                 pair_label_txt, device=device, metric=cfg['metric'])

    elif cfg['dataset'] == 'megface':
        acc = 0 # we will update later
    
    return acc

    
def inference(opt, device):
    save_dir = Path(opt.save_dir)
    with open(opt.data) as f:        
        data_cfg = yaml.load(f, Loader=yaml.FullLoader)
    data_name = opt.data.split('/')[-1]    
    sys.stdout = Logger(save_dir / 'test_log_{}.{}.txt'.format(opt.wname, data_name))

    #Configure
    cuda = device.type != 'cpu'
    ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
    model = ckpt['backbone'].to(device)
    
    if cuda and opt.global_rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    datasets = build_datasets(data_cfg, opt.batch_size, cuda, opt.workers, mode='test')
    evals(data_cfg, save_dir, model, datasets, device, opt.wname, train=False)
    
def parser():    
    parser = argparse.ArgumentParser(description='Face Test')
    parser.add_argument('--weights', type=str , default='', help='pretrained weights path')
    parser.add_argument('--wname'  , type=str , default='last', help='pretrained weights name: best or last')        
    parser.add_argument('--re_eval'  , action='store_true', help='re evaluation')        
    parser.add_argument('--data', type=str, default='data/bins.yaml', help='data yaml path')    
    parser.add_argument('--workers'           , type=int, default=4)
    parser.add_argument('--batch_size'        , type=int, default=512)

    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs_eccv_hope', help='save to project/name')
    parser.add_argument('--name', default='exp', help='run test dir name')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = parser()
    opt.global_rank = -1            
    opt.save_dir = Path(opt.project) / opt.name 
    # assert os.path.isdir(opt.save_dir), 'ERROR: --project_directory does not exist'        
    if opt.weights[-3:] != '.pt':
        opt.weights = opt.save_dir / 'weights' / (opt.wname + '.pt')     
        print(opt.weights)
    else:
        opt.wname = opt.weights[-7:-3]
    assert os.path.isfile(opt.weights), 'ERROR: --weight path does not exist'
        
    device = select_device(opt.device, batch_size=opt.batch_size, rank=opt.global_rank)
    inference(opt, device)