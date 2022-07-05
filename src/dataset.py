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
    def __init__(self, data_path, model_name_or_path, total_max_len, md_max_len, data_fts_path):
        super().__init__()
        self.data_path = data_path
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.data_fts_path = data_fts_path
    
    def load_data(self):
        self.data = pd.read_parquet(self.data_path)
        self.data_fts = json.load(open(self.data_fts_path))

    def __getitem__(self, index):
        row = self.data.iloc[index]
        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt",
        )

        code_inputs = self.tokenizer.batch_encode_plus(
            self.data_fts[row.id]["codes"],
            add_special_tokens=True,
            max_length=23,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        total_markdown = self.data_fts[row.id]["total_md"]
        total_code = self.data_fts[row.id]["total_code"]
        if total_markdown + total_code == 0:
            fts = torch.FloatTensor([0])
        else:
            fts = torch.FloatTensor(
                [total_markdown / float(total_markdown + total_code)]
            )

        ids = torch.cat(
            [inputs["input_ids"].flatten(), code_inputs["input_ids"].flatten()]
        )

        ids = ids[: self.total_max_len]
        ids = torch.nn.functional.pad(
            ids,
            (0, self.total_max_len - len(ids)),
            "constant",
            self.tokenizer.pad_token_id,
        )

        assert len(ids) == self.total_max_len

        mask = torch.cat(
            [
                inputs["attention_mask"].flatten(),
                code_inputs["attention_mask"].flatten(),
            ]
        )

        mask = mask[: self.total_max_len]
        mask = torch.nn.functional.pad(
            mask,
            (0, self.total_max_len - len(mask)),
            "constant",
            self.tokenizer.pad_token_id,
        )

        return (ids, mask, fts), torch.FloatTensor([row.pct_rank])
    
    def __len__(self):
        return self.data.shape[0]
