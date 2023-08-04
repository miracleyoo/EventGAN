"""
Metrics implementation for 3D human pose comparisons
"""

import torch
import importlib
from torch import nn


__all__ = ['Accuracy', 'BinaryMatch', 'BinaryMatchF1', 'MeanRatio']


class BaseMetric(nn.Module):
    def forward(self, pred, gt, gt_mask=None):
        """
        Base forward method for metric evaluation
        Args:
            pred: predicted voxel
            gt: ground truth voxel
            gt_mask: ground truth mask

        Returns:
            Metric as single value
        """
        pass


class Accuracy(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, pred, y):
        y_label = torch.argmax(y, dim=-1)
        pred_label = torch.argmax(pred, dim=-1)

        acc = (y_label == pred_label).double().mean()
        return acc


class BinaryMatch(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, pred, y):
        # for all pixels in pred, if pred > 0.01, pred_binary = 1, else 0
        pred_binary = torch.where(pred > 0.01, torch.ones_like(pred), torch.zeros_like(pred))
        label_binary = torch.where(y > 0.01, torch.ones_like(y), torch.zeros_like(y))

        acc = (pred_binary == label_binary).double().mean()
        return acc


"""
class BinaryMatchF1(BaseMetric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
    def forward(self, pred, y):
        pred_binary = torch.where(pred > 0.01, torch.ones_like(pred), torch.zeros_like(pred))
        label_binary = torch.where(y > 0.01, torch.ones_like(y), torch.zeros_like(y))

        diff_index = (pred_binary != label_binary)
        TP = torch.logical_and(label_binary == 1, pred_binary == 1).double().sum()
        FP = torch.logical_and(label_binary[diff_index] == 0, pred_binary[diff_index] == 1).double().sum()
        FN = torch.logical_and(label_binary[diff_index] == 1, pred_binary[diff_index] == 0).double().sum()
        
        return 2 * TP / (2 * TP + FP + FN)
"""

class BinaryMatchF1(BaseMetric):
    def __init__(self, threshold=0.01, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        # from torchmetrics.classification import BinaryF1Score
        try:
            self.binary_f1_score = importlib.import_module('torchmetrics.functional.classification').binary_f1_score
        except:
            # install torchmetrics if not installed
            subprocess = importlib.import_module('subprocess')
            subprocess.run(['pip', 'install', 'torchmetrics'])
            self.binary_f1_score = importlib.import_module('torchmetrics.functional.classification').binary_f1_score
        
    def forward(self, pred, y):
        '''
        y = torch.sum(torch.abs(y),dim=(1,2))
        pred = torch.sum(torch.abs(pred),dim=(1,2))
        label_binary = torch.where(y > self.threshold, torch.ones_like(y), torch.zeros_like(y)).to(y.device)
        
        f1 = self.binary_f1_score(pred, label_binary, threshold=self.threshold)
        '''
        y=torch.sum(y,dim=0)
        pred=torch.sum(pred,dim=0)
        y = torch.where(y > self.threshold, torch.ones_like(y), torch.zeros_like(y)).to(y.device)
        pred = torch.where(pred > self.threshold, torch.ones_like(pred), torch.zeros_like(pred)).to(pred.device)
        # 计算 True Positive (TP)：预测为正类且目标为正类的数量
        TP = (pred * y).sum()

        # 计算 False Positive (FP)：预测为正类但目标为负类的数量
        FP = (pred * (1 - y)).sum()

        # 计算 False Negative (FN)：预测为负类但目标为正类的数量
        FN = ((1 -pred) * y).sum()

        TN=((1-pred)*(1-y)).sum()

        # 计算 Precision 和 Recall
        precision = TP / (TP + FP + 1e-8)  # 加上 1e-8 避免除数为 0
        recall = TP / (TP + FN + 1e-8)


        # 计算 F1 分数
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1
        


class MeanRatio(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, pred, y):
        ratio = (pred + 0.01) / (y + 0.01)
        # if ratio < 1, ratio = 1/ratio
        ratio = torch.where(ratio < 1, 1/ratio, ratio)
        ratio = ratio.mean()
        return ratio