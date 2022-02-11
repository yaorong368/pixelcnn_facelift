import json
import os
import torch
import cv2
import numpy as np
import os
import shutil
import nibabel as nib
import argparse

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, parameter
from torch.utils.data import Dataset,DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from catalyst.callbacks import SchedulerCallback, CheckpointCallback
from catalyst import dl, metrics, utils
from catalyst.data import BatchPrefetchLoaderWrapper
from catalyst.contrib.nn.criterion.dice import DiceLoss


from utils.init_models import initialize_model
from utils.get_data_list import get_data_list
from utils.Dataset import my_dataset




class CustomRunner(dl.Runner):
    def __init__(
        self, 
        logdir: str,
        image_dataset: list,
        mask_dataset: list,
        batch_size: int,
        epochs: int,
        resize: int,
        num_colors:int,
        weight_cond_logits_loss: int,
        weight_prior_logits_loss: int,
        ):
        super().__init__()
        self._logdir = logdir
        self.logdir = logdir
        self.image_dataset = image_dataset
        self.mask_dataset = mask_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.resize = resize
        self.num_colors=num_colors
        
        self.weight_cond_logits_loss = weight_cond_logits_loss
        self.weight_prior_logits_loss = weight_prior_logits_loss
   
    
    def get_loggers(self):
        return {
            "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
        }
 
    def get_loaders(self, stage: str):
        train_dataset = my_dataset(
            img_list = self.image_dataset[0:800], 
            msk_list = self.mask_dataset[0:800],
            resize = self.resize,
            num_colors = self.num_colors,
         
            )
    
        test_dataset = my_dataset(
            img_list = self.image_dataset[800:], 
            msk_list = self.mask_dataset[800:],
            resize = self.resize,
            num_colors = self.num_colors,
        
            )
        
        if self.engine.is_ddp:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.engine.world_size,
                rank=self.engine.rank,
                shuffle=False,
            )
            
            valid_sampler = DistributedSampler(
                test_dataset,
                num_replicas=self.engine.world_size,
                rank=self.engine.rank,
                shuffle=False,
            )
        else:
            train_sampler = valid_sampler = None
        
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size = self.batch_size, 
            sampler=train_sampler,
            pin_memory=True,
            num_workers=2,
            )
        
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size = self.batch_size, 
            sampler=valid_sampler,
            pin_memory=True,
            num_workers=2,
            )   
        
        # train_loader = BatchPrefetchLoaderWrapper(train_loader)#, num_prefetches=40)
        # test_loader = BatchPrefetchLoaderWrapper(test_loader)#, num_prefetches=2)
        
        loaders = {
            "train": train_loader,
            "valid": test_loader,
        }

        return loaders
        
    
    def get_scheduler(self, stage: str, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs)
    
    
    def handle_batch(self, batch):
        ipt = batch['image']
        msk = batch['mask']
        tgt = batch['target']
        logits, prior_logits, cond_logits= self.model(ipt, msk)
        
        # logits = F.softmax(logits, dim=1)
        # prior_logits = F.softmax(prior_logits, dim=1)
        # cond_logits = F.softmax(cond_logits, dim=1)
        # tgt = F.one_hot(tgt, self.num_colors).permute(0, 4, 1, 2, 3)
        
        
        # self.batch = {'logits': logits,
        #               'prior_logits': prior_logits,
        #               'cond_logits': cond_logits,
        #               'tgt': tgt,
        #               }
        
        
        logits_loss = F.cross_entropy(logits, tgt)
        prior_logits_loss = F.cross_entropy(prior_logits, tgt)
        cond_logits_loss = F.cross_entropy(cond_logits, tgt)
         
        loss = logits_loss + \
                self.weight_cond_logits_loss * cond_logits_loss + \
                self.weight_prior_logits_loss * prior_logits_loss
                
        self.batch_metrics.update(
            {"loss": loss, "logits_loss":logits_loss, "cond_loss":cond_logits_loss}
        )


                            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parameters of pixelcnn training')
    parser.add_argument('-log_path', type=str, required=True,
                        help='path of log')
    parser.add_argument('-resize', type=int, default=128,
                        help='size of input image')
    parser.add_argument('-batch_size', type=int, default=5,
                        help='batch size')
    parser.add_argument('-constrained', type=bool, default=True,
                        help='use constrained model')
    parser.add_argument('-num_colors', type=int, default=8,
                        help='make sure to select the middle slice spatial direction')
    parser.add_argument('-filter_size', type=int, default=5,
                        help='kernel size of prior network')
    parser.add_argument('-filter_size_rs', type=int, default=5,
                        help='kernel size of conditional network')
    parser.add_argument('-depth', type=int, default=19,
                        help='number of layers of prior network')
    parser.add_argument('-depth_rs', type=int, default=32,
                        help='number of layers of conditional network')
    parser.add_argument('-num_filters_cond', type=int, default=36,
                        help='width of prior network')
    parser.add_argument('-num_filters_prior', type=int, default=72,
                        help='width of conditional network')
    parser.add_argument('-lr', type=float, default=0.00005,
                        help='learning rate')
    parser.add_argument('-epochs', type=int, default=300,
                        help='number of epochs')
    parser.add_argument('-weight_cond_logits_loss', type=float, default=1.0,
                        help='weight for the conditional loss')
    parser.add_argument('-weight_prior_logits_loss', type=float, default=0.0,
                        help='weight for the prior loss')
    parser.add_argument('-d', '--datasets', nargs='+', required=True)
    
    args = parser.parse_args()
    
    print('training on:', args.datasets)
    image_dataset, mask_dataset = get_data_list(args.datasets)

    img_size = (1, args.resize, args.resize)
    model = initialize_model(img_size,
                            args.num_colors,
                            args.depth,
                            args.depth_rs,
                            args.filter_size,
                            args.filter_size_rs,
                            args.constrained,
                            args.num_filters_prior,
                            args.num_filters_cond,
                            )

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.0001,
        )

    # criterion = CrossEntropyLoss()
    # criterion = DiceLoss(1, mode='macro')


    if os.path.exists(args.log_path):
        shutil.rmtree(args.log_path)
    logdir = args.log_path
    
    runner = CustomRunner(
        logdir=logdir,
        image_dataset=image_dataset,
        mask_dataset=mask_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        resize=args.resize,
        num_colors=args.num_colors,
        weight_cond_logits_loss=args.weight_cond_logits_loss,
        weight_prior_logits_loss=args.weight_prior_logits_loss,
    )
 

    runner.train(
        model=model, 
        criterion=None, 
        optimizer=optimizer,  
        loaders=None, 
        callbacks=[
            CheckpointCallback(logdir=logdir),
            # dl.CriterionCallback(
            #     input_key="logits", 
            #     target_key="tgt", 
            #     metric_key="logits_loss",
            #     ),
            # dl.CriterionCallback(
            #     input_key="prior_logits", 
            #     target_key="tgt", 
            #     metric_key="prior_logits_loss",
            #     ),
            # dl.CriterionCallback(
            #     input_key="cond_logits", 
            #     target_key="tgt", 
            #     metric_key="cond_logits_loss",
            #     ),
            
            # dl.MetricAggregationCallback(
            # metric_key="loss",
            # metrics={
            #     "logits_loss": 1.0, 
            #     "prior_logits_loss": weight_prior_logits_loss,
            #     "cond_logits_loss": weight_cond_logits_loss,
            #     },
            # mode="weighted_sum",
            #     ),
            
            dl.OptimizerCallback(
                metric_key="loss",
                ),
            dl.SchedulerCallback(),
            ],
        logdir=logdir, 
        num_epochs=args.epochs, 
        verbose=True,
        ddp=True,
        amp=False,
        )