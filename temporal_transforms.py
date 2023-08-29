import random
import math


class Compose(object):

    def __init__(self, transforms, num_clips=1, train=False):
        self.transforms = transforms
        self.num_clips = num_clips
        self.train = train

    def __call__(self, frame_indices):

        out = []

        for n in range(self.num_clips):
            trans_frame_indices = frame_indices
            for i, t in enumerate(self.transforms):
                if isinstance(trans_frame_indices[0], list):
                    next_transforms = Compose(self.transforms[i:])
                    dst_frame_indices = [
                        next_transforms(clip_frame_indices)
                        for clip_frame_indices in trans_frame_indices
                    ]
                    return dst_frame_indices
                else:
                    trans_frame_indices = t(trans_frame_indices)
            out.append(trans_frame_indices)
        
        if self.train: out = out[0]
        if len(out) == 1: out = out[0]
        #print("trans_frame_indices", trans_frame_indices)
        return out


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out

class LastFramePadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(frame_indices[-1])

        return out


class TemporalBeginCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):

    def __init__(self, size):
        self.size = size
        # self.loop = LoopPadding(size)
        self.loop = LastFramePadding(size)

    def __call__(self, frame_indices):

        if self.size == -1: 
            size = max(len(frame_indices), 30)
            self.loop = LastFramePadding(size)
        else: size = self.size
        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (size // 2))
        end_index = min(begin_index + size, len(frame_indices))

        out = frame_indices[begin_index:end_index]
        #print(begin_index, end_index)

        if len(out) < size:
            out = self.loop(out)

        #print("video_length", len(frame_indices), "sampled", size, "len(out)", len(out))
        
        # print(out) 
        return out


class TemporalRandomCrop(object):

    def __init__(self, size):
        self.size = size
        # self.loop = LoopPadding(size)
        self.loop = LastFramePadding(size)

    def __call__(self, frame_indices):

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        if len(out) < self.size:
            out = self.loop(out)
        
        #print(out)    
        # print(len(out))
        return out


class TemporalEvenCrop(object):

    def __init__(self, size, n_samples=1):
        self.size = size
        self.n_samples = n_samples
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        # print("frame_indices", frame_indices)

        n_frames = len(frame_indices)
        stride = max(
            1, math.ceil((n_frames - 1 - self.size) / (self.n_samples - 1)))

        out = []
        for begin_index in frame_indices[::stride]:
            if len(out) >= self.n_samples:
                break
            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
            # The following line cancels the effect of the strided sampling! I re-implemented it.
            # sample = list(range(begin_index, end_index))
            sample = list(frame_indices[begin_index:end_index])

            if len(sample) < self.size:
                out.append(self.loop(sample))
                break
            else:
                out.append(sample)
        #print("out", out)
        return out


class SlidingWindow(object):

    def __init__(self, size, stride=0):
        self.size = size
        if stride == 0:
            self.stride = self.size
        else:
            self.stride = stride
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        out = []
        for begin_index in frame_indices[::self.stride]:
            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
            sample = list(range(begin_index, end_index))

            if len(sample) < self.size:
                out.append(self.loop(sample))
                break
            else:
                out.append(sample)
        return out


class TemporalSubsampling(object):

    def __init__(self, stride):
        self.stride = stride

    def __call__(self, frame_indices):
        return frame_indices[::self.stride]
    
    
#  Augmentations for CVPR: duration, order, composition
class AugmentDuration(object):
    '''Frames are downsampled with random stride (1, 2, 4, 8, 16), so that the 
    sub-action duration is altered.'''
    def __init__(self, strides=[1, 2, 4, 8, 16]):
        print("Augment duration with sampling strides", strides)
        self.strides = strides

    def __call__(self, frame_indices):
        stride = random.sample(self.strides,1)[0]
        return frame_indices[::stride]
    
class AugmentOrder(object):
    '''Frames are divided in segments which are randomly shuffled.
    relative_segment_size: percentage of the video length that is used as segment length.'''
    def __init__(self, relative_segment_size=0.1):
        print("Augment order with", relative_segment_size*100, "% segment length.")
        self.relative_segment_size = relative_segment_size

    def __call__(self, frame_indices):
        segment_size = max(8, round(self.relative_segment_size*len(frame_indices)))
        print("Augment order segment_size", segment_size)
        frame_indices = [
            frame_indices[i:(i + segment_size)]
            for i in range(0, len(frame_indices), segment_size)
        ]
        random.shuffle(frame_indices)
        #print("frame_indices", frame_indices)
        frame_indices = [t for segment in frame_indices for t in segment]
        return frame_indices
    
class AugmentComposition(object):
    '''50% of frames segments are omitted in order to remove sub-actions and 
    break frequent (co-)occurrences.'''
    def __init__(self, relative_segment_size=0.1, prob_of_omission=0.5):
        print("Augment composition with", relative_segment_size*100, "% segment length", prob_of_omission, "prob. of omission")
        self.relative_segment_size = relative_segment_size
        self.prob_of_omission = prob_of_omission

    def __call__(self, frame_indices):
        segment_size = max(8, round(self.relative_segment_size*len(frame_indices)))
        print("Augment composition segment_size", segment_size)
        frame_indices = [
            frame_indices[i:(i + segment_size)]
            for i in range(0, len(frame_indices), segment_size)
        ]
        random.shuffle(frame_indices)
        frame_indices = [t for segment in frame_indices[:max(1,int(len(frame_indices)*(1-self.prob_of_omission)))]
                         for t in segment]
        frame_indices = sorted(frame_indices)
        return frame_indices
    
    
class Shuffle(object):

    def __init__(self, block_size):
        self.block_size = block_size

    def __call__(self, frame_indices):
        frame_indices = [
            frame_indices[i:(i + self.block_size)]
            for i in range(0, len(frame_indices), self.block_size)
        ]
        random.shuffle(frame_indices)
        frame_indices = [t for block in frame_indices for t in block]
        return frame_indices


# Added as training sampling strategy 
class EvenCropsSampling(object):

    def __init__(self, size, sampling_stride=8, crop_size=8, augment_duration=False,
                 augment_order=False, augment_composition=False):
        self.size = size
        self.sampling_stride = sampling_stride
        self.crop_size = crop_size
        # self.padding = LoopPadding(size)
        self.padding = LastFramePadding(size)
        self.augment_duration = augment_duration
        self.augment_order = augment_order
        self.augment_composition = augment_composition

    def __call__(self, frame_indices):
        n_frames = len(frame_indices)

        crop_size = self.crop_size 
        if self.augment_duration:
            augmentation_factor = random.sample([1, 2, 4], 1)[0]
            crop_size = self.crop_size*augmentation_factor

        random_offset = random.randint(0, min(crop_size-1, int(n_frames/3)))

        stride = max(
            1, math.ceil((n_frames - self.size - random_offset) / (self.size/crop_size)))
        print("(n_frames - self.size - random_offset)", (n_frames - self.size - random_offset))
        print("stride", stride)

        out = []
        #print([i for i in range(0, n_frames, crop_size+stride)])
        begin_indices = list(range(random_offset, len(frame_indices), crop_size+stride))

        if self.augment_order:
            # 20% prob. of being shuffled
            # print(begin_indices)
            n_shuffled = int((len(begin_indices)-1)*0.3)
            n_not_shuffled = (len(begin_indices)-1) - n_shuffled
            shuffle_prob = random.sample([0]*n_not_shuffled + [1]*n_shuffled, len(begin_indices)-1)+[0]
            for idx, i in enumerate(shuffle_prob):
                if i==1:
                    next = begin_indices[idx+1]
                    begin_indices[idx+1] = begin_indices[idx]
                    begin_indices[idx] = next
            # print(shuffle_prob)
            # print(begin_indices)

        for begin_index in begin_indices:
            # print("begin_index", begin_index)
            if len(out) >= self.size:
                break
            end_index = min(len(frame_indices)-1, begin_index + crop_size)
            # sample = list(range(begin_index, end_index))
            sample = frame_indices[begin_index:end_index]
            out = out + sample
        if len(out) < self.size:
                out = self.padding(out)
        print(out)
        return out
