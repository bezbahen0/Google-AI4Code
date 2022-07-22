import json
import torch
import pickle
import pandas as pd

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
    def __init__(self, data_id_path, data_source_path, model_name_or_path, max_len):
        super().__init__()
        self.data_id_path = data_id_path
        self.data_source_path = data_source_path
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def load_data(self):
        self.data_id = pd.read_parquet(self.data_id_path)
        self.data_source = pd.read_parquet(self.data_source_path)
        self.data_source = dict(
            zip(self.data_source.cell.values, self.data_source.source.values)
        )

    def __getitem__(self, index):
        row = self.data_id.iloc[index]

        text = self.data_source[row.cell_id] + " [SEP] " + self.data_source[row.cid]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt",
        )

        return (
            inputs["input_ids"].flatten(),
            inputs["attention_mask"].flatten(),
            torch.FloatTensor([row.label]),
        )

    def __len__(self):
        return self.data_id.shape[0]
