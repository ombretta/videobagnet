#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:58:04 2020

@author: ombretta
"""

import torch
import h5py
import os
import pickle as pkl

def check_video_availability(video_list, dataset_path):
    new_list = []
    with h5py.File(str(dataset_path), 'r') as f:
        if "video_ids" in list(f.keys()):
            available_videos = [v.decode('UTF-8') for v in f["video_ids"]]
        else:    
            available_videos = list(f.keys())
    for video in video_list:
        if video in available_videos: new_list.append(video)
    return new_list


def load_data(filename, with_torch = False):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = torch.load(f, map_location='cpu') if with_torch == True else pkl.load(f)
        return data
    else: print("File", filename, "does not exists.")       
    
    
def load_videos_sets(data_path, dataset_path):
    data_path = str(data_path)
    if os.path.exists(data_path+"/train_videos.dat"):
        valid_videos = load_data(data_path+"/valid_videos.dat")
        test_videos = load_data(data_path+"/test_videos.dat")
        train_videos = load_data(data_path+"/train_videos.dat") 
        
        train_videos = check_video_availability(train_videos, dataset_path)
        test_videos = check_video_availability(test_videos, dataset_path)
        valid_videos = check_video_availability(valid_videos, dataset_path)
        
        return train_videos, test_videos, valid_videos
    return [], [], []


def get_breakfast_dataset(videos, dataset_path, sample_t_stride): 
    classes_labels = load_data("/tudelft.net/staff-bulk/ewi/insy/VisionLab/ombrettastraff/instructional_videos/i3d_breakfast/data/processed/classes_labels.dat")
    dataset = DatasetBreakfast(videos, classes_labels, dataset_path, sample_t_stride)
    return dataset


def check_data_shape(X, dataset_path):
    if "raw" in str(dataset_path): 
        X = X.permute(3, 0, 1, 2)
    if len(X.shape) > 4: X = X.squeeze(0)
    return X


def get_label_from_id(video_id, classes_labels):
    # Extract label from the video id
    video_class = video_id.split("_")[-1]
    if video_class == "ch0" or video_class == "ch1":
        video_class = video_id.split("_")[-2]
    y = torch.Tensor([classes_labels[video_class]]).squeeze().to(torch.float)
    return y


class DatasetBreakfast(torch.utils.data.Dataset):
    
    'Characterizes a dataset for PyTorch'
    def __init__(self, videos_ids, classes_labels, dataset_path, sample_t_stride):
        self.classes_labels = classes_labels
        self.dataset_path = dataset_path
        self.videos_ids = videos_ids
        self.file = h5py.File(self.dataset_path, 'r')
        self.sample_t_stride = sample_t_stride
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.videos_ids)
        
    def __getitem__(self, index): 
        
        'Generates one sample of data'
        
        video_id = self.videos_ids[index]
        tmp = self.file[video_id][...] 
        
        X = torch.from_numpy(tmp).to(torch.float) 
        X = check_data_shape(X, self.dataset_path)
        X = X[:,::self.sample_t_stride,:,:]
                
        # Extract label from the video id
        y = get_label_from_id(video_id, self.classes_labels).long()
        
        return X, y
