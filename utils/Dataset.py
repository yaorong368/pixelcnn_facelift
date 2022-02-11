import torch
from torch.utils.data import Dataset

import nibabel as nib
import numpy as np
import cv2


class my_dataset(Dataset):
    def __init__(
        self, 
        img_list: list,
        msk_list: list,
        resize: int,
        num_colors: int,
    ):
          
        self.img_list = img_list
        self.msk_list = msk_list
        self.resize = resize
        self.num_colors = num_colors
       
        
    def __getitem__(self, item):
        
        
        
        mask = np.load(self.msk_list[item])
        mask = np.floor(cv2.resize(mask, dsize=(self.resize, self.resize)))
#         mask = mask.get_fdata()
#         mask = self.__crop__(mask, slices, 'mask')
#         mask = np.flip(mask, axis=-1)
        
        
        image = np.load(self.img_list[item])
        image = cv2.resize(image, dsize=(self.resize, self.resize))
#         image = image.get_fdata()
#         image = self.__crop__(image, slices, 'image')
#         image = np.rot90(image, k=2)
        image_target = np.around(image/image.max()*(self.num_colors-1))
        
        image_prior = (image_target - image_target.min()) / (image_target.max() - image_target.min())
        
        
        ipt = image_prior*mask
    
        
        ipt = (torch.from_numpy(np.expand_dims(ipt,0))).to(torch.float32)
        mask = (torch.from_numpy(np.expand_dims(mask,0).copy())).to(torch.float32)
        
        con_ipt = torch.cat([ipt, mask], dim=0)
        image_prior = (torch.from_numpy(np.expand_dims(image_prior, 0))).to(torch.float32)
        target = (torch.from_numpy(np.expand_dims(image_target, 0))).to(torch.long)
        
        
        return {'image':image_prior, 'mask':con_ipt, 'target':target}
    
    def __len__(self):
           
        return len(self.img_list)
    
    def __crop__(self, img, coords, label):
        if label == 'mask':
            tgt_cube = cv2.resize(img[coords,:,:], dsize=(self.resize, self.resize))
        if label == 'image':
            tgt_cube = cv2.resize(img[:,:,coords], dsize=(self.resize, self.resize))
        return tgt_cube
    
    
    
class my_3d_dataset(Dataset):
    def __init__(
        self, 
        img_list: list,
        msk_list: list,
        resize: int,
        num_colors: int,
        num_slices: int,
        start_slice: int,
    ):
          
        self.img_list = img_list
        self.msk_list = msk_list
        self.resize = resize
        self.num_colors = num_colors
        self.num_slices = num_slices
        self.start_slice = start_slice
       
        
    def __getitem__(self, item):
        items = item // self.num_slices
        slices = item % self.num_slices
        
        mask = nib.load(self.msk_list[items])
        mask = mask.get_fdata()
        mask = self.__crop__(mask, slices, 'mask')
        mask = np.floor(cv2.resize(mask, dsize=(self.resize, self.resize)))
        mask = np.flip(mask, axis=-1)
        
        
        image = nib.load(self.img_list[items])
        image = image.get_fdata()
        
        con_image = self.__crop__(image, slices-1, 'image')
        con_image = cv2.resize(con_image, dsize=(self.resize, self.resize))
        con_image = np.rot90(con_image, k=2) 
        con_image = np.around(con_image/con_image.max()*(self.num_colors-1))
        con_image = (con_image - con_image.min()) / (con_image.max() - con_image.min())
        
        image = self.__crop__(image, slices, 'image')
        image = cv2.resize(image, dsize=(self.resize, self.resize))
        image = np.rot90(image, k=2)
        
        image_target = np.around(image/image.max()*(self.num_colors-1))
        
        image_prior = (image_target - image_target.min()) / (image_target.max() - image_target.min())
        
        
        ipt = image_prior*mask
    
        
        ipt = (torch.from_numpy(np.expand_dims(ipt,0))).to(torch.float32)
        mask = (torch.from_numpy(np.expand_dims(mask,0).copy())).to(torch.float32)
        con_image = (torch.from_numpy(np.expand_dims(con_image,0).copy())).to(torch.float32)
        
        con_ipt = torch.cat([ipt, con_image], dim=0)
        image_prior = (torch.from_numpy(np.expand_dims(image_prior, 0))).to(torch.float32)
        target = (torch.from_numpy(np.expand_dims(image_target, 0))).to(torch.long)
        
        
        return {'image':image_prior, 'mask':con_ipt, 'target':target}
    
    def __len__(self):
           
        return len(self.img_list) * self.num_slices
    
    def __crop__(self, img, coords, label):
        if label == 'mask':
            tgt_cube = img[self.start_slice+coords,:,:]
        if label == 'image':
            tgt_cube = img[:,:,self.start_slice+coords]
        return tgt_cube