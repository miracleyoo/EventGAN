import torch
from torch.utils.data import Dataset
import numpy as np
import random
import pickle
from utils.event_utils import gen_discretized_event_volume, gen_discretized_event_volume_from_struct
import pytorch_utils


class MVSECSequence(Dataset):
    def __init__(self, args, train=True, path=None, start_time=-1):
        super(MVSECSequence, self).__init__()

        self.train = train
        self.args = args

        # store for center crop
        self.top_left = self.args.top_left
        self.image_size = self.args.image_size
        
        if start_time == -1:
            self.start_time = self.args.start_time
        else:
            self.start_time = start_time
            
        self.max_skip_frames = 0 #self.args.max_skip_frames

        self.flip_x = self.args.flip_x
        self.flip_y = self.args.flip_y

        if path is None:
            if self.train:
                self.path = args.train_file
            else:
                self.path = args.validation_file
                self.flip_x = 0
                self.flip_y = 0
        else:
            self.path = path


    def __len__(self):
        """ Return the first frame that has number_events before it """
        # length = self.num_images - self.start_frame - self.max_skip_frames - 1
        return 16

    def get_box(self):
        top_left = self.top_left
        if self.train:
            top =  int(np.random.rand()*(self.raw_image_size[0]-1-self.image_size[0]))
            left = int(np.random.rand()*(self.raw_image_size[1]-1-self.image_size[1]))
            top_left = [top, left]
        bottom_right = [top_left[0]+self.image_size[0],
                        top_left[1]+self.image_size[1]]

        return top_left, bottom_right

    def get_image(self, image, bbox):
        top_left, bottom_right = bbox

        image = image[top_left[0]:bottom_right[0],
                                 top_left[1]:bottom_right[1],None]

        image = image.transpose((2,0,1)).astype(np.float32)/255.
        image -= 0.5
        image *= 2.

        return image
        
    def get_events(self, events, bbox):
        top_left, bottom_right = bbox
        mask = np.logical_and(np.logical_and(events['y']>=top_left[0],
                                             events['y']<bottom_right[0]),
                              np.logical_and(events['x']>=top_left[1],
                                             events['x']<bottom_right[1]))

        events_masked = events[mask]
        events_shifted = events_masked
        events_shifted['x'] = events_masked['x'] - top_left[1]
        events_shifted['y'] = events_masked['y'] - top_left[0]

        # subtract out min to get delta time instead of absolute
        events_shifted['timestamp'] -= np.min(events_shifted['timestamp'])

        # convolution expects 4xN
        # events_shifted = events_shifted.astype(np.float32)
        return events_shifted

    def normalize_event_volume(self, event_volume):
        event_volume_flat = event_volume.view(-1)
        nonzero = torch.nonzero(event_volume_flat)
        nonzero_values = event_volume_flat[nonzero]
        if nonzero_values.shape[0]:
            lower = torch.kthvalue(nonzero_values,
                                   max(int(0.02 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            upper = torch.kthvalue(nonzero_values,
                                   max(int(0.98 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            max_val = max(abs(lower), upper)
            event_volume = torch.clamp(event_volume, -max_val, max_val)
            event_volume /= max_val
        return event_volume

    def apply_illum_augmentation(self, prev_image, next_image,
                                 gain_min=0.8, gain_max=1.2, gamma_min=0.8, gamma_max=1.2):
        random_gamma = gamma_min + random.random() * (gamma_max - gamma_min)
        random_gain = gain_min + random.random() * (gain_max - gain_min);
        prev_image = self.transform_gamma_gain_np(prev_image, random_gamma, random_gain)
        next_image = self.transform_gamma_gain_np(next_image, random_gamma, random_gain)
        return prev_image, next_image

    def transform_gamma_gain_np(self, image, gamma, gain):
        # apply gamma change and image gain.
        image = (1. + image) / 2.
        image = gain * np.power(image, gamma) 
        image = (image - 0.5) * 2.
        return np.clip(image, -1., 1.) 
         
    def get_single_item(self, ind):
        # Load and parse data
        with open(self.path,'rb') as f:
            data_packet = pickle.load(f)
        
        events = data_packet['events'][ind]
        prev_image = data_packet['images'][ind]
        next_image = data_packet['images'][ind+1]
        prev_image_ts = data_packet['timestamps'][ind]/10 # Compensate for the process bug
        next_image_ts = data_packet['timestamps'][ind+1]/10
        self.raw_image_size = prev_image.shape
        # print("raw_image_size", self.raw_image_size)
        
        # Pre-process
        bbox = self.get_box()
        next_image = self.get_image(next_image, bbox)
        prev_image = self.get_image(prev_image, bbox)
        events = self.get_events(events, bbox)
        event_volume = gen_discretized_event_volume_from_struct(events,
                                                    [self.args.n_time_bins*2,
                                                     self.image_size[0],
                                                     self.image_size[1]])
        
        # Transform
        if self.args.normalize_events:
            event_volume = self.normalize_event_volume(event_volume)

        prev_image_gt, next_image_gt = prev_image, next_image        
        if self.train:
            if np.random.rand() < self.flip_x:
                event_volume = torch.flip(event_volume, dims=[2])
                prev_image = np.flip(prev_image, axis=2)
                next_image = np.flip(next_image, axis=2)
            if np.random.rand() < self.flip_y:
                event_volume = torch.flip(event_volume, dims=[1])
                prev_image = np.flip(prev_image, axis=1)
                next_image = np.flip(next_image, axis=1)
            prev_image_gt, next_image_gt = prev_image, next_image
            if self.args.appearance_augmentation:
                prev_image, next_image = self.apply_illum_augmentation(prev_image, next_image)

        # Event volume is t-y-x
        output = { "prev_image" : prev_image.copy(),
                   "prev_image_gt" : prev_image_gt.copy(),
                   "prev_image_ts" : prev_image_ts,
                   "next_image" : next_image.copy(),
                   "next_image_gt" : next_image_gt.copy(),
                   "next_image_ts" : next_image_ts,
                   "event_volume" : event_volume }
        return output
                
    def __getitem__(self, ind_in):
        return self.get_single_item(ind_in)

class WeightedRandomSampler(pytorch_utils.data_loader.CheckpointSampler):
    """
    Samples from a data_source with weighted probabilities for each element.
    Weights do not need to sum to 1. 
    Typical use case is when you have multiple datasets, the weights for each dataset are
    set to 1/len(ds). This ensures even sampling amongst datasets with different lengths.
    weights - tensor with numel=len(data_source)
    
    """
    def __init__(self, data_source, weights):
        super(WeightedRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.weights = weights

    def next_dataset_perm(self):
        return torch.multinomial(self.weights, len(self.data_source), replacement=True).tolist()

def get_and_concat_datasets(path_file, options, train=True):
    ds_list = []
    ds_len_list = []
    with open(path_file) as f:
        paths = f.read().splitlines()
    for path_start in paths:
        if not path_start:
            break
        path, _ = path_start.split(' ')
        ds_list.append(MVSECSequence(options,
                                     train=train,
                                     path=path))
        # print(f"Loaded {path}")
        weight = np.sqrt(len(ds_list[-1]))
        if "indoor" in path:
            weight *= 2
        ds_len_list += [weight] * len(ds_list[-1])
    weights = 1. / torch.Tensor(ds_len_list)
    ds = torch.utils.data.ConcatDataset(ds_list)
    # # from torch.utils.data.sampler import SequentialSampler
    # sampler=pytorch_utils.data_loader.SequentialSampler(ds)
    sampler = WeightedRandomSampler(ds,
                                    weights)
    
    return ds, sampler
