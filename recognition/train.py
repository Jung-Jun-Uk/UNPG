import yaml
import os
import sys
import datetime
import time

import argparse
from pathlib import Path
from copy import deepcopy

from torchtoolbox.optimizer import CosineWarmupLr

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.general import select_device, increment_path, Logger, AverageMeter, \
    print_argument_options, init_torch_seeds, is_parallel

from datasets.build import build_datasets
from models.build import build_models, HeadAndLoss
from evaluation import evals

def main(opt, device):

    save_dir = Path(opt.save_dir)   
    
    # Hyperparameters    
    with open(opt.data) as f:
        data_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    
    sys.stdout = Logger(save_dir / 'log_.txt', opt.resume)
    
    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
            
    # Save run settings    
    with open(save_dir / 'data_cfg.yaml', 'w') as f:
        yaml.dump(data_cfg, f, sort_keys=False)
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    
    # Print the config file(opt and hyp)        
    if opt.global_rank in [-1, 0]:
        print_argument_options(data_cfg, 'Data Config File')
        print_argument_options(hyp, 'Learning hyparperameters')
        print_argument_options(opt, 'Config File')
                     
    #Configure
    cuda = device.type != 'cpu'
    init_torch_seeds(opt.seed + 1 + opt.global_rank)
    
    train_dataset = build_datasets(data_cfg, opt.batch_size, cuda, opt.workers, mode='train', rank=opt.global_rank)
    test_dataset = build_datasets(data_cfg, opt.batch_size, cuda, opt.workers, mode='test', rank=opt.global_rank)
    
    model = build_models(opt.model, drop_lastfc=opt.drop_lastfc).to(device)
    criterion = nn.CrossEntropyLoss()    
    headandloss = HeadAndLoss(512, train_dataset.num_classes, criterion, opt.head, opt.aux, head_zoo=opt.head_zoo).to(device)

    # Save the head settings
    with open(save_dir / 'head_cfg.yaml', 'w') as f:
        yaml.dump(headandloss.head_cfg, f, sort_keys=False)    
    if opt.aux is not None:
        with open(save_dir / 'aux_cfg.yaml', 'w') as f:
            yaml.dump(headandloss.aux_cfg, f, sort_keys=False)    
    
    pretrained = opt.weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
        model_state_dict = ckpt['backbone'].float().state_dict()
        model.load_state_dict(model_state_dict, strict=False)
        if ckpt.get('headandloss') is not None:
            head_state_dict = ckpt['headandloss'].float().state_dict()            
            headandloss.load_state_dict(head_state_dict, strict=False)
            del head_state_dict
            
    if cuda and opt.global_rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        # headandloss = torch.nn.DataParallel(headandloss)
        headandloss.head = torch.nn.DataParallel(headandloss.head)        
        if opt.aux != '':
            headandloss.aux = torch.nn.DataParallel(headandloss.aux) 
        #headandloss.criterion = torch.nn.DataParallel(headandloss.criterion)
        
    if cuda and opt.global_rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)
        headandloss = DDP(headandloss, device_ids=[opt.local_rank], output_device=opt.local_rank)
    
    if opt.global_rank in [-1, 0]:
        print("Creat model  : {}".format(opt.model))
        print("Creat head   : {}".format(opt.head))

    optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr'], weight_decay=hyp['weight_decay'], momentum=hyp['momentum'])        
    optimizer.add_param_group({'params': headandloss.parameters()})

    batches_per_epoch = train_dataset.num_training_images // opt.batch_size
    scheduler = CosineWarmupLr(optimizer, batches_per_epoch, opt.max_epoch, base_lr=hyp['lr'], warmup_epochs=hyp['warmup_epochs'])    
    
    opt.scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Resume    
    best_acc, acc = 0.0, 0.0
    if pretrained and opt.resume:
        # Optimizer
        if ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scheduler') is not None:
            scheduler.load_state_dict(ckpt['scheduler'])        
        best_acc = ckpt['best_acc']        
        opt.start_epoch = ckpt['epoch'] + 1
        del ckpt, model_state_dict

    if opt.local_test:
        opt.data_cfg, opt.save_dir, opt.test_dataset = data_cfg, save_dir, test_dataset              
    for epoch in range(opt.start_epoch, opt.max_epoch): 
        if opt.global_rank != -1:
            train_dataset.loader.sampler.set_epoch(epoch)       
        if opt.global_rank in [-1, 0]:
            print("==> Epoch {}/{}".format(epoch+1, opt.max_epoch))        
        train(opt, model, headandloss, optimizer, scheduler, train_dataset.loader, device, opt.global_rank)                
        
        if opt.global_rank in [-1, 0]:
            acc = evals(data_cfg, save_dir, model, test_dataset, device, train=True)                   
            if acc >= best_acc:
                best_acc = acc                                            
            # Save backbone, head
            ckpt = {'epoch' : epoch,                     
                    'best_acc': best_acc,                     
                    'backbone' : deepcopy(model.module if is_parallel(model) else model).eval(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}
            
            if len(headandloss.state_dict().keys()) > 0: # if the parameter exists
                ckpt['headandloss'] = deepcopy(headandloss.module if is_parallel(headandloss) else headandloss).eval()
            else:
                ckpt['headandloss'] = None

            torch.save(ckpt, last)
            if best_acc == acc:
                torch.save(ckpt, best)
            del ckpt
            
    
def train(opt, model, headandloss, optimizer, scheduler, trainloader, device, rank):
    model.train()
    losses = AverageMeter()        
    meta_dict = {'snmin' : AverageMeter(), 'snmax' : AverageMeter(), 
                 'spwmin' : AverageMeter(), 'spwmax' : AverageMeter(),
                 'snwmin' : AverageMeter(), 'snwmax' : AverageMeter()}

    start_time = time.time() 
    for i, (data, labels) in enumerate(trainloader):                        
        data, labels = data.to(device), labels.to(device)          
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            deep_features = model(data)                        
            loss, metas = headandloss(deep_features, labels, rank)
        
        if torch.cuda.device_count() > 1:
            loss = loss.mean()

        opt.scaler.scale(loss).backward()
        opt.scaler.step(optimizer)
        opt.scaler.update()
        scheduler.step() # only use CosineWarmupLr
        
        for key, val in metas.items():            
            meta_dict[key].update(val.item(), labels.size(0))
        losses.update(loss.item(), labels.size(0))        

        if (i+1) % opt.print_freq == 0 and rank in [-1, 0]:
            elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
            start_time = time.time()            
            s = ''
            for key, val in meta_dict.items():
                s += key + ' {:.6f} ({:.6f}) '.format(val.val, val.avg)
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}), {} elapsed time (h:m:s): {}" \
                .format(i+1, len(trainloader), losses.val, losses.avg, s, elapsed))
          
        if opt.local_test and (i+1) % opt.eval_freq == 0 and rank in [-1, 0]:
            acc = evals(opt.data_cfg, opt.save_dir, model, opt.test_dataset, device, train=True)            
            model.train()
                        
        
def parser():    
    parser = argparse.ArgumentParser(description='Face training')
    parser.add_argument('--head_zoo', default='data/head.zoo.yaml', help='All head config files are included')
    parser.add_argument('--hyp', type=str, default='data/hyp.yaml', help='hyperparameters path')    
    parser.add_argument('--data', type=str, default='data/face.yaml', help='data yaml path')    
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--model', default='iresnet-100', help='iresnet-34 or iresnet-100')    
    parser.add_argument('--head', default='arcface', help='e.g., arcface, cosface, magface')
    parser.add_argument('--aux', type=str, default='', help='unified negative pair generation (e.g., unpg, unpgfw)')        
    parser.add_argument('--drop_lastfc', action='store_true', help='Do not apply dropout to last fc layer')
    parser.add_argument('--seed', type=int, default=1000)        

    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--workers', type=int, default=4)    
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--local_test', action='store_true')
    parser.add_argument('--eval_freq', type=int, default=1000)
    parser.add_argument('--print_freq', type=int, default=50)
    
    parser.add_argument('--nlog', action='store_true', help='nlog = not print log.txt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs_eccv_hope', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    args = parser.parse_args()

    return args


# data parallel script: python train.py 
# distributed parallel script:  python -m torch.distributed.launch --nproc_per_node 4 train.py
# We found performance degradation during work distributed training for metric losses. We will fix it later.
if __name__ == "__main__":    
    opt = parser()    
    names = (opt.model + '.' + opt.head)
    names += (opt.aux + '.')        
    names += (opt.name)
    opt.save_dir = increment_path(Path(opt.project) / names, exist_ok=opt.exist_ok)  # increment run
    
    # Set DDP variables    
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else os.path.join(opt.save_dir,'weights/last.pt')
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.weights, opt.resume = ckpt, True
        opt.hyp = str(Path(ckpt).parent.parent / 'hyp.yaml')
        opt.head_cfg = str(Path(ckpt).parent.parent / 'head.cfg.yaml')
        print('Resuming training from %s' % ckpt)
    
    device = select_device(opt.device, batch_size=opt.batch_size, rank=opt.global_rank)    
    #DDP mode
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    main(opt, device)