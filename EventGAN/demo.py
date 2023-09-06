import os.path as op
from pytorch_utils import BaseOptions
from models.eventgan_base import EventGANBase
import configs
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib2 import Path
from utils.event_utils import gen_discretized_event_volume
from metrics import BinaryMatchF1, BinaryMatch, PoolMSE

options = BaseOptions()
options.parser = configs.get_args(options.parser)
args = options.parse_args()

EventGAN = EventGANBase(args)

def dataset_metrics(path, results_folder, metrics_dict, metrics_results):
    file=open(path,'rb')
    stem = Path(path).stem
    data=pickle.load(file)
    
    images=torch.from_numpy(data['images']).float().cuda()
    images = images.float()/255.
    images -= 0.5
    images *= 2.

    batch_event_voxels_pred = np.zeros((16, args.n_time_bins*2, 260, 346))
    batch_event_voxels_gt = np.zeros((16, args.n_time_bins*2, 260, 346))

    result=[]
    for index in range (images.shape[0]-1):
        current_image=images[index:index+2]
        event_volume_est = EventGAN.forward(current_image, is_train=False)[0][0]
        events_truth = data['events'][index]

        x = torch.tensor(events_truth['x'].copy())
        y = torch.tensor(events_truth['y'].copy())
        t = torch.tensor(events_truth['timestamp'].copy())
        t = (t-t.min()).float()
        p = torch.tensor(events_truth['polarity'].copy())

        events_truth=torch.stack((x,y,t,p),axis=1)
        event_volume_truth = gen_discretized_event_volume(events_truth,
                                                        [args.n_time_bins*2,
                                                        260,
                                                        346]).cuda()
        
        
        batch_event_voxels_gt[index] = event_volume_truth.cpu().numpy()
        batch_event_voxels_pred[index] = event_volume_est.cpu().numpy()
        
    # result.append([metric.forward(event_volume_est, event_volume_truth) for metric in metrics])
    # print(event_volume_est.shape, event_volume_truth.shape)
    for metric_name, metric in metrics_dict.items():
        metrics_results[metric_name].append(metric.forward(
            torch.Tensor(batch_event_voxels_pred).unsqueeze(0), 
            torch.Tensor(batch_event_voxels_gt).unsqueeze(0)).item()
        )
    
    # Save the GT and predicted event volumes.
    out_info = {
        'gt_event_volume': batch_event_voxels_gt,
        'gen_event_volume': batch_event_voxels_pred,
        'metrics': {k:v[-1] for k,v in metrics_results.items() if len(v) > 0},
    }
    
    out_path = op.join(results_folder, f'{stem}.pkl')
    
    with open(out_path, 'wb') as f:
        pickle.dump(out_info, f)
    
    # return result

metrics_dict = {
    'BinaryMatchF1_sum_c': BinaryMatchF1(op_type='sum_c'),
    'BinaryMatchF1_sum_cp': BinaryMatchF1(op_type='sum_cp'),
    'BinaryMatchF1_raw': BinaryMatchF1(op_type='raw'),
    'BinaryMatch_sum_c': BinaryMatch(op_type='sum_c'),
    'BinaryMatch_sum_cp': BinaryMatch(op_type='sum_cp'),
    'BinaryMatch_raw': BinaryMatch(op_type='raw'),
    'PoolMSE_2': PoolMSE(kernel_size=2),
    'PoolMSE_4': PoolMSE(kernel_size=4),
}

metrics_results = {
    'BinaryMatchF1_sum_c': [],
    'BinaryMatchF1_sum_cp': [],
    'BinaryMatchF1_raw': [],
    'BinaryMatch_sum_c': [],
    'BinaryMatch_sum_cp': [],
    'BinaryMatch_raw': [],
    'PoolMSE_2': [],
    'PoolMSE_4': [],
}

results_folder = op.join('/tsukimi/backup', 'EventGAN-pretrained-model-test-results-new')
Path(results_folder).mkdir(exist_ok=True)
info =pickle.load(open(r"/tsukimi/datasets/MVSEC/data_paths.pkl",'rb'))

for file in tqdm(info['test']): #[100:]):
    path = r"/tsukimi/datasets/MVSEC/event_chunks_processed/"+file
    dataset_metrics(path, results_folder, metrics_dict, metrics_results) #[f1,bm])
    for metric_name, metric in metrics_dict.items():
        print(metric_name, metrics_results[metric_name][-1])
    # break

# save the metrics results
with open(op.join(results_folder, 'metrics_results.pkl'), 'wb') as f:
    pickle.dump(metrics_results, f)

for metric_name, metric in metrics_dict.items():
    print(metric_name, np.mean(metrics_results[metric_name]))
    