import torch
import json
import cv2
import argparse

from catalyst.utils.distributed import get_nn_from_ddp_module
from catalyst.utils.misc import maybe_recursive_call
# from catalyst import utils

from torch.utils.data import Dataset

import nibabel as nib

from utils.init_models import initialize_model
from utils.Dataset import my_dataset
from utils.get_data_list import get_data_list

import numpy as np
import os

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

print('generating on:', args.datasets)

# image_dataset, mask_dataset = get_data_list(args.datasets)
image_dataset, mask_dataset = get_data_list(args.datasets, num_items=5)
            
img_size = (1, args.resize, args.resize)
model = initialize_model(img_size,
                         args.num_colors,
                         args.depth,
                         args.depth_rs,
                         args.filter_size,
                         args.filter_size_rs,
                         args.constrained,
                         args.num_filters_prior,
                         args.num_filters_cond)

checkpoint_path = os.path.join(args.log_path, 'last.pth')
checkpoint = torch.load(checkpoint_path)

model = get_nn_from_ddp_module(model)
maybe_recursive_call(
    model,
    "load_state_dict",
     recursive_args=checkpoint["model_state_dict"],
     )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

train_dataset = my_dataset(
        img_list = image_dataset, 
        msk_list = mask_dataset,
        resize = args.resize,
        num_colors = args.num_colors,
        )

list_data = []
# for m in range(10):
#     list.append(m)
    

    
# for m in range(928, 950):
#     list_data.append(m)

for i in range(train_dataset.__len__()):
# for i in list_data:
    squares = train_dataset.__getitem__(i)
    np.save('./generated_data/target%s.npy'%(i), squares['target'].squeeze())
    np.save('./generated_data/input%s.npy'%(i),squares['mask'][0].squeeze())

    con_cube = torch.tensor(np.expand_dims(squares['mask'],0), dtype=torch.float32)
    con_cube = con_cube.to(device)
    
    print('start generating', i)
    samples = model.sample(con_cube, return_likelihood=False, temp=1)

    sample = samples.cpu()
    predict = sample.detach().numpy().squeeze()
    np.save('./generated_data/output%s.npy'%(i),predict)