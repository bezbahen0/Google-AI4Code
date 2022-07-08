import json
import torch
import pickle
import numpy as np

from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset


class XGBrankerDataSet:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        with open(self.data_path, "rb") as input_file:
            X_train, y_train, groups = pickle.load(input_file)
        return X_train, y_train, groups


class TransformersDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

    def load_data(self):
        with open(self.data_path, "rb") as f:
            self.ids = np.load(f)
            self.masks = np.load(f)
            self.fts = np.load(f)
            self.ranks = np.load(f)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.ids[index]),
            torch.from_numpy(self.masks[index]),
            torch.FloatTensor([self.fts[index]]),
            torch.FloatTensor([self.ranks[index]]),
        )

    def __len__(self):
        return self.ids.shape[0]
