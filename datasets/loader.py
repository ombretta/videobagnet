import io

import h5py

import decord
from decord import VideoReader
from decord import cpu, gpu

from PIL import Image
from torchvision import transforms

import torch
import numpy as np


class ImageLoaderPIL(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with path.open('rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class ImageLoaderAccImage(object):

    def __call__(self, path):
        import accimage
        return accimage.Image(str(path))


class VideoLoader(object):

    def __init__(self, image_name_formatter, image_loader=None):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader

    def __call__(self, video_path, frame_indices):
        print(video_path, frame_indices)
        video = []
        for i in frame_indices:
            image_path = video_path / self.image_name_formatter(i)
            #print(image_path)
            if image_path.exists():
                video.append(self.image_loader(image_path))
        return video


class VideoLoaderDecord_old(object):

    def __init__(self):
        decord.bridge.set_bridge('torch')

    def __call__(self, video_path, frame_indices):
        #print(video_path, frame_indices)
        with open(video_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
        # video = vr.get_batch(frame_indices)
        video_length = len(vr)
        video = []
        #print(video_path, len(vr), frame_indices)
        for i in frame_indices:
            if i >= video_length:
                #print("Frame index greater than video length!", i, video_length)
                index = video_length-1
            else:
                index = i
            frame = torch.Tensor.float(vr[index].permute(2, 0, 1))
            # print(video_path, frame.shape)
            # print("mean, std", torch.mean(frame), torch.std(frame))
            video.append(transforms.ToPILImage()(frame).convert("RGB")) 
            #print(video_path, frame.shape)
        return video


class VideoLoaderDecord(object):

    def __init__(self):
        decord.bridge.set_bridge('torch')

    def __call__(self, video_path, frame_indices):

        with open(video_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
        video_length = len(vr)
        video = []
        #print(video_path, len(vr), frame_indices)

        for i in frame_indices:
            if i >= video_length:
                index = video_length - 1
            else:
                index = i

            frame = Image.fromarray(np.array(vr[index]))
            video.append(frame)

        return video
    

class VideoLoaderHDF5(object):

    def __call__(self, video_path, frame_indices):
        with h5py.File(video_path, 'r') as f:
            video_data = f['video']
            
            video = []
            for i in frame_indices:
                if i < len(video_data):
                    # video.append(Image.open(io.BytesIO(video_data[i])))
                    frame_bgr = torch.tensor(video_data[i])
                    frame_rgb = frame_bgr[:, :, [2, 1, 0]] # Swap bgr to rgb
                    video.append(transforms.ToPILImage()(frame_rgb).convert("RGB")) 
                else:
                    return video

        return video


class VideoLoaderFlowHDF5(object):

    def __init__(self):
        self.flows = ['u', 'v']

    def __call__(self, video_path, frame_indices):
        with h5py.File(video_path, 'r') as f:

            flow_data = []
            for flow in self.flows:
                flow_data.append(f[f'video_{flow}'])

            video = []
            for i in frame_indices:
                if i < len(flow_data[0]):
                    frame = [
                        Image.open(io.BytesIO(video_data[i]))
                        for video_data in flow_data
                    ]
                    frame.append(frame[-1])  # add dummy data into third channel
                    video.append(Image.merge('RGB', frame))

        return video
    
    
