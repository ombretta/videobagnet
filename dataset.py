from torchvision import get_image_backend

from datasets.videodataset import VideoDataset
from datasets.videodataset_multiclips import (VideoDatasetMultiClips,
                                              collate_fn)
from datasets.breakfast_segments import SegmentsDataset
from datasets.loader import VideoLoader, VideoLoaderDecord

import os
from pathlib import Path

def usual_image_name_formatter(x):
    return f'image_{x:05d}.jpg'

def mnist_image_name_formatter(x):
    return f'{x:d}.jpg'

def something_image_name_formatter(x):
    return f'{x:05d}.jpg'

def avi_video_formatter(x):
    return f'{x:d}.avi'

def mp4_video_formatter(x):
    return f'{x:d}.mp4'

def get_training_data(video_path,
                      annotation_path,
                      dataset_name,
                      input_type,
                      file_type,
                      n_segments=None,
                      spatial_transform=None,
                      temporal_transform=None,
                      target_transform=None,
                      use_image_features=False,
                      timm_model="xception41",
                      video_path_formatter = (lambda root_path, label, 
                          video_id: root_path / label / f'{video_id}.mp4')):
    assert dataset_name in [
        'kinetics', 'mini_kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 
        'breakfast', 'mini_breakfast', 'movingmnist', 'movingmnist_blackframes',
        'movingmnist_longterm', 'movingmnist_motiondiff', 'movingmnist_motionsame', 
        'movingmnist_frequencies', 'movingmnist_frequencies_complex', 'something',
        'movingmnist_static', 'charades', 'multiTHUMOS', 'breakfast_subactions',
        'breakfast_segments', 'multiTHUMOS_extended', 'breakfast_MNIST', 'MultiTHUMOS_MNIST',
        'breakfast_MNIST_subactions'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5', 'None']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'
        
        if 'movingmnist' in dataset_name:
            image_name_formatter = mnist_image_name_formatter
        elif 'something' in dataset_name:
            image_name_formatter = something_image_name_formatter
        elif '_MNIST' in dataset_name:
            #image_name_formatter = avi_video_formatter
            image_name_formatter = mp4_video_formatter # To use with the noisy dataset 
        else: image_name_formatter = usual_image_name_formatter

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)
        
        if 'movingmnist' in dataset_name or 'something' in dataset_name:
            video_path_formatter = (
            lambda root_path, label, video_id: root_path / video_id)

    else:
        if input_type == 'rgb':
            loader = VideoLoaderDecord()
            
        if dataset_name in ['kinetics', 'mini_kinetics']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    'train_256' / label / f'{video_id}')
        elif dataset_name in ['breakfast', 'breakfast_segments']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    label / f'{video_id}.mp4')
        elif dataset_name in ['breakfast_subactions']:
            #video_path_formatter = (lambda root_path, label, video_id: root_path /
            #                        video_id.split("_")[2] / 
            #                        f'{"_".join(video_id.split("_")[:-1])}.mp4')
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    video_id.split("_")[2] /
                                    f'{"_".join(video_id.split("_")[:3])}.mp4')
        elif dataset_name in ['charades']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    f'{video_id}.mp4')
        elif dataset_name in ['multiTHUMOS']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    f'{"_".join(video_id.split("_"))}.mp4')
        elif dataset_name in ['multiTHUMOS_extended']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    f'{"_".join(video_id.split("_")[:-1])}.mp4')
        elif dataset_name in ['hmdb51', 'ucf101']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    label / f'{video_id}.avi')
        elif "_MNIST" in dataset_name and "subactions" not in dataset_name:
            video_path_formatter = (lambda root_path, label, video_id: root_path / f'{video_id}.mp4')#.avi') # changed for dataset with noise
        elif "_MNIST" in dataset_name and "subactions" in dataset_name:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    f'{"_".join([video_id.split("_")[1]]+[video_id.split("_")[0].replace("stereoch0", "stereo00").replace("stereoch1", "stereo01")]+video_id.split("_")[1:3])}.mp4')

    #print("video_path_formatter", video_path_formatter)
    print("Building VideoDataset for", dataset_name)
    
    if dataset_name == 'breakfast_segments':
        training_data = SegmentsDataset(video_path,
                                     annotation_path,
                                     'training',
                                     n_segments,
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter)
    else:
        training_data = VideoDataset(video_path,
                                     annotation_path,
                                     'training',
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter)
    return training_data


def get_validation_data(video_path,
                        annotation_path,
                        dataset_name,
                        input_type,
                        file_type,
                        n_segments=None,
                        spatial_transform=None,
                        temporal_transform=None,
                        target_transform=None,
                        use_image_features=False,
                        timm_model="xception41",
                        collate_fn=collate_fn,
                        video_path_formatter = (lambda root_path, label, 
                            video_id: root_path / label / f'{video_id}.mp4')):
    assert dataset_name in [
        'kinetics', 'mini_kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 
        'breakfast', 'mini_breakfast', 'movingmnist', 'movingmnist_blackframes',
        'movingmnist_longterm',	'movingmnist_motiondiff', 'movingmnist_motionsame', 
        'movingmnist_frequencies', 'movingmnist_frequencies_complex', 'something',
        'movingmnist_static', 'charades', 'multiTHUMOS', 'breakfast_subactions', 
        'breakfast_segments', 'multiTHUMOS_extended', 'breakfast_MNIST', 'MultiTHUMOS_MNIST',
        'breakfast_MNIST_subactions'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5', 'None']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if 'movingmnist' in dataset_name:
            image_name_formatter = mnist_image_name_formatter
        elif 'something' in dataset_name:
            image_name_formatter = something_image_name_formatter
        elif '_MNIST' in dataset_name:
            #image_name_formatter = avi_video_formatter
            image_name_formatter = mp4_video_formatter # To use	with the noisy dataset
        else: image_name_formatter = usual_image_name_formatter
        
        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)

        if 'movingmnist' in dataset_name or 'something' in dataset_name:
            video_path_formatter = (
            lambda root_path, label, video_id: root_path / video_id)
    else:
        if input_type == 'rgb':
            loader = VideoLoaderDecord()
            
        if dataset_name in ['kinetics', 'mini_kinetics']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    'val_256' / label / f'{video_id}')
        elif dataset_name in ['breakfast', 'breakfast_segments']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    label / f'{video_id}.mp4')
        elif dataset_name in ['breakfast_subactions']:
            #video_path_formatter = (lambda root_path, label, video_id: root_path /
            #                        video_id.split("_")[2] /
            #                        f'{"_".join(video_id.split("_")[:-1])}.mp4')
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    video_id.split("_")[2] /
                                    f'{"_".join(video_id.split("_")[:3])}.mp4')
        elif dataset_name in ['charades']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    f'{video_id}.mp4')
        elif dataset_name in ['multiTHUMOS']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    f'{"_".join(video_id.split("_"))}.mp4')
        elif dataset_name in ['multiTHUMOS_extended']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    f'{"_".join(video_id.split("_")[:-1])}.mp4')
        elif dataset_name in ['hmdb51', 'ucf101']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    label / f'{video_id}.avi')
        elif "_MNIST" in dataset_name and "subactions" not in dataset_name:
            video_path_formatter = (lambda root_path, label, video_id: root_path / f'{video_id}.mp4') #.avi') # changed for dataset with noise
        elif "_MNIST" in dataset_name and "subactions" in dataset_name:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    f'{"_".join([video_id.split("_")[1]]+[video_id.split("_")[0].replace("stereoch", "stereo0").replace("stereoch1", "stereo01")]+video_id.split("_")[1:3])}.mp4')

    if dataset_name == 'breakfast_segments':
        validation_data = SegmentsDataset(video_path,
                                     annotation_path,
                                     'validation',
                                     n_segments,
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter)
        collate_fn = None
    else:
        validation_data = VideoDatasetMultiClips(
            video_path,
            annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter)
    return validation_data, collate_fn


def get_inference_data(video_path,
                       annotation_path,
                       dataset_name,
                       input_type,
                       file_type,
                       inference_subset,
                       n_segments=None,
                       spatial_transform=None,
                       temporal_transform=None,
                       target_transform=None,
                       use_image_features=False,
                       timm_model="xception41",
                       collate_fn=collate_fn,
                       video_path_formatter = (lambda root_path, label, 
                           video_id: root_path / label / f'{video_id}.mp4')):
    assert dataset_name in [
        'kinetics', 'mini_kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 
        'breakfast', 'mini_breakfast', 'movingmnist', 'movingmnist_blackframes',
        'movingmnist_longterm',	'movingmnist_motiondiff', 'movingmnist_motionsame', 
        'movingmnist_frequencies', 'movingmnist_frequencies_complex', 'something',
        'movingmnist_static', 'charades', 'multiTHUMOS', 'breakfast_subactions', 
        'breakfast_segments', 'multiTHUMOS_extended', 'breakfast_MNIST', 'MultiTHUMOS_MNIST',
        'breakfast_MNIST_subactions'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5', 'None']
    assert inference_subset in ['train', 'val', 'test']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if 'movingmnist' in dataset_name:
            image_name_formatter = mnist_image_name_formatter
        elif 'something' in dataset_name:
            image_name_formatter = something_image_name_formatter
        elif '_MNIST' in dataset_name:
            #image_name_formatter = avi_video_formatter
            image_name_formatter = mp4_video_formatter # To use	with the noisy dataset 
        else: image_name_formatter = usual_image_name_formatter
        
        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)

        if dataset_name in ['movingmnist', 'movingmnist_blackframes',
        'movingmnist_longterm', 'something']:
            video_path_formatter = (
            lambda root_path, label, video_id: root_path / video_id)
    else:
        if input_type == 'rgb':
            loader = VideoLoaderDecord()
            
        if dataset_name in ['kinetics', 'mini_kinetics']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    label / f'{video_id}')
        elif dataset_name in ['breakfast', 'breakfast_segments']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    label / f'{video_id}.mp4')
        elif dataset_name in ['breakfast_subactions']:
            #video_path_formatter = (lambda root_path, label, video_id: root_path /
            #                        video_id.split("_")[2] /
            #                        f'{"_".join(video_id.split("_")[:-1])}.mp4')
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    video_id.split("_")[2] /
                                    f'{"_".join(video_id.split("_")[:3])}.mp4')
        elif dataset_name in ['charades']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    f'{video_id}.mp4')
        elif dataset_name in ['multiTHUMOS']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    f'{"_".join(video_id.split("_"))}.mp4')
        elif dataset_name in ['multiTHUMOS_extended']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    f'{"_".join(video_id.split("_")[:-1])}.mp4')
        elif dataset_name in ['hmdb51', 'ucf101']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    label / f'{video_id}.avi')
        elif "_MNIST" in dataset_name and "subactions" not in dataset_name:
            video_path_formatter = (lambda root_path, label, video_id: root_path / f'{video_id}.mp4') #.avi') # changed for dataset with noise
        elif "_MNIST" in dataset_name and "subactions" in dataset_name:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    f'{"_".join([video_id.split("_")[1]]+[video_id.split("_")[0].replace("stereoch", "stereo0").replace("stereoch1", "stereo01")]+video_id.split("_")[1:3])}.mp4')

    if inference_subset == 'train':
        subset = 'training'
    elif inference_subset == 'val':
        subset = 'validation'
    elif inference_subset == 'test':
        if dataset_name in ['breakfast_MNIST']:
            subset = 'test'
        else:
            subset = 'testing'
    if dataset_name == 'breakfast_segments':
        inference_data = SegmentsDataset(video_path,
                                     annotation_path,
                                     'testing',
                                     n_segments,
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter)
        collate_fn = None
    else:
        inference_data = VideoDatasetMultiClips(
            video_path,
            annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter,
            target_type=['label', 'video_id', 'segment'])

    return inference_data, collate_fn
