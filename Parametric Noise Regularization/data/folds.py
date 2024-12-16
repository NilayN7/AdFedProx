import numpy as np
import torch

from data import dro_dataset

# import bisect
# import warnings

# from torch._utils import _accumulate
from torch import randperm, default_generator


class Subset(torch.utils.data.Dataset):
    """
    Subsets a dataset while preserving original indexing.

    NOTE: torch.utils.dataset.Subset loses original indexing.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        self.group_array = self.get_group_array(re_evaluate=True)
        self.label_array = self.get_label_array(re_evaluate=True)
        

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def get_group_array(self, re_evaluate=True):
        """Return an array [g_x1, g_x2, ...]"""
        # setting re_evaluate=False helps us over-write the group array if necessary (2-group DRO)
        if re_evaluate:
            group_array = self.dataset.get_group_array()[self.indices]        
            assert len(group_array) == len(self)
            return group_array
        else:
            return self.group_array

    def get_label_array(self, re_evaluate=True):
        if re_evaluate:
            label_array = self.dataset.get_label_array()[self.indices]
            assert len(label_array) == len(self)
            return label_array
        else:
            return self.label_array


class ConcatDataset(torch.utils.data.ConcatDataset):
    """
    Concate datasets

    Extends the default torch class to support group and label arrays.
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)

    def get_group_array(self):
        group_array = []
        for dataset in self.datasets:
            group_array += list(np.squeeze(dataset.get_group_array()))
        return group_array

    def get_label_array(self):
        label_array = []
        for dataset in self.datasets:
            label_array += list(np.squeeze(dataset.get_label_array()))
        return label_array

