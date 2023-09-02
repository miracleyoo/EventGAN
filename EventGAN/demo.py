import os.path as op
# from utils.viz_utils import gen_event_images
from pytorch_utils import BaseOptions
from models.eventgan_base import EventGANBase
import configs
import cv2
import numpy as np
import torch
import pickle
from tqdm import tqdm
from pathlib2 import Path
from utils.event_utils import gen_discretized_event_volume
# Read in images.
# prev_image = cv2.imread('EventGAN/example_figs/007203_01.png')
# next_image = cv2.imread('EventGAN/example_figs/007203.png')

# prev_image = cv2.resize(prev_image, (861, 260))
# next_image = cv2.resize(next_image, (861, 260))

# prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
# next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)

# images = np.stack((prev_image, next_image)).astype(np.float32)
# images = (images / 255. - 0.5) * 2.
# images = torch.from_numpy(images).cuda()

# Build network.

options = BaseOptions()
options.parser = configs.get_args(options.parser)
args = options.parse_args()

EventGAN = EventGANBase(args)

def dataset_metrics(path, metrics):
    file=open(path,'rb')
    stem = Path(path).stem
    data=pickle.load(file)
    images=torch.from_numpy(data['images']).float().cuda()

    images = images.float()/255.
    images -= 0.5
    images *= 2.

    # print(images.shape)
    # print(images.dtype)
    result=[]
    for index in range (images.shape[0]-1):
        current_image=images[index:index+2]
        #print(current_image.shape)
        event_volume_est = EventGAN.forward(current_image, is_train=False)[0][0]
        #print(event_volume1[0].shape)

        events_truth = data['events'][index]


        x = torch.tensor(events_truth['x'].copy())
        y = torch.tensor(events_truth['y'].copy())
        t = torch.tensor(events_truth['timestamp'].copy())
        t=(t-t.min()).float()
        p = torch.tensor(events_truth['polarity'].copy())

        events_truth=torch.stack((x,y,t,p),axis=1)

        event_volume_truth = gen_discretized_event_volume(events_truth,
                                                        [args.n_time_bins*2,
                                                        260,
                                                        346]).cuda()
        
        # Save the GT and predicted event volumes.
        out_info = {
            'gt_event_volume': event_volume_truth.cpu().numpy(),
            'gen_event_volume': event_volume_est.cpu().numpy()
        }
        results_folder = op.join('.', 'results')
        Path(results_folder).mkdir(exist_ok=True)
        out_path = op.join(results_folder, stem + '_{}.pkl'.format(index))
        with open(out_path, 'wb') as f:
            pickle.dump(out_info, f)
        
        result.append([metric.forward(event_volume_est, event_volume_truth) for metric in metrics])
    return result


from metrics import BinaryMatchF1,BinaryMatch

f1=BinaryMatchF1()
bm=BinaryMatch()

#print(dataset_metrics(r"/tsukimi/datasets/MVSEC/event_chunks_processed/indoor_flying3_data_left-8.pkl",[f1,bm]))

info =pickle.load(open(r"/tsukimi/datasets/MVSEC/data_paths.pkl",'rb'))
result = []
for file in tqdm(info['val'][100:]):
    path = r"/tsukimi/datasets/MVSEC/event_chunks_processed/"+file
    # print(dataset_metrics(path,[f1,bm]))
    result += dataset_metrics(path,[f1,bm])
    print(torch.Tensor(result).mean(0))
    # break
print(torch.Tensor(result).mean(0))
# event_images = gen_event_images(event_volume[-1], 'gen')

# event_image = event_images['gen_event_time_image'][0].cpu().numpy().sum(0)

# event_image *= 255. / event_image.max()
# event_image = event_image.astype(np.uint8)

# cv2.imwrite('simulated_event.png', event_image)
