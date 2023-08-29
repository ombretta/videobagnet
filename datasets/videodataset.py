import json
from pathlib import Path
import os

import numpy as np
import torch
import torch.utils.data as data

from .loader import VideoLoader
import cv2

def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        # print(class_label, index)¡
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path, video_path_formatter):
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])
            if 'video_path' in value:
                video_paths.append(Path(value['video_path']))
            else:
                label = value['annotations']['label']
                
                video_paths.append(video_path_formatter(root_path, label, key))
                # print(video_path_formatter(root_path, label, key), label, key,
                #       os.path.exists(video_path_formatter(root_path, label, key)))

    return video_ids, video_paths, annotations


class VideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label'):
        
        # print(root_path)
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        # print("VideoDataset temporal_transform", self.temporal_transform)
        
        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type

    def __make_dataset(self, root_path, annotation_path, subset,
                       video_path_formatter):
        with annotation_path.open('r') as f:
            data = json.load(f)
        video_ids, video_paths, annotations = get_database(
            data, subset, root_path, video_path_formatter)
        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            #print(name, label)
            idx_to_class[label] = name

        n_videos = len(video_ids)
        #print(len(video_ids), "videos")        
        #print(len(video_paths), "video paths")
        #print(class_to_idx)

        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))


            if 'label' in annotations[i]:
                label = annotations[i]['label']

                # multi-label classification
                if isinstance(label, list):
                    label_id = torch.zeros((len(class_to_idx)))
                    for category in label:
                        if category in class_to_idx:
                            label_id[class_to_idx[category]] = 1
                            
                # single-label classification
                else:
                    if label in class_to_idx:
                        label_id = class_to_idx[label]
                    else: 
                        label_id = int(label)
            else:
                label = 'test'
                label_id = -1

            video_path = video_paths[i]
            if not video_path.exists():
                continue
            
            if 'segment' in annotations[i]:
                segment = annotations[i]['segment']
                if len(segment)>1 and segment[1] == 1:
                    continue
                if len(segment) == 2:
                    frame_indices = list(range(segment[0], segment[1]))
                elif isinstance(segment[0], list):
                    frame_indices = [j for i in segment for j in np.arange(i[0], i[1], 1, dtype=int)]
                else:
                    frame_indices = segment[:-1]
            else:
                # Added to deal with incomplete json files
                segment = [0, 8]
                frame_indices = list(range(segment[0], segment[1]))
            
            #print(i, video_path, segment, frame_indices, video_ids[i], label_id)

            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id
            }
            
            dataset.append(sample)
        
        return dataset, idx_to_class

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        #print("clip", clip)
        #print(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # to delete
        #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        #video = cv2.VideoWriter("prova.mp4", fourcc, 1, (32, 32))
        #for image in clip:
        #    print(image)
        #    open_cv_image = np.array(image)[:, :, ::-1].copy()  # Convert RGB to BGR
        #    video.write(open_cv_image)
        # end

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        
        return clip

    def __getitem__(self, index):
        path = self.data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]
        
        frame_indices = self.data[index]['frame_indices']
        #print("About to load", path, frame_indices)
 
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        
        # print("loading ", path, frame_indices)
        clip = self.__loading(path, frame_indices)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # print("clip", clip.shape, torch.max(clip), torch.min(clip)) 
        return clip, target

    def __len__(self):
        return len(self.data)
