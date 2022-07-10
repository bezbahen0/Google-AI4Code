import os
import pickle
import json
import torch
import logging
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import AutoModel, AutoTokenizer


class Featurizer:
    def __init__(self, data_path, featurized_path, logger, max_len=128):
        self.data_path = data_path
        self.featurized_path = featurized_path
        self.logger = logger

    def featurize(self, mode="test"):
        if mode == "train":
            self._featurize_train()
        if mode == "test":
            self._featurize_test()

        self.logger.info(f"Save featurized data to {self.featurized_path}")

    def _featurize_train(self):
        pass

    def _featurize_test(self):
        pass

    def _load_data(self):
        self.logger.info(f"Loading data from {self.data_path}")
        return pd.read_parquet(self.data_path)


class XGBrankerFeaturizer(Featurizer):
    def __init__(
        self,
        data_path,
        featurized_path,
        tfidf_idf_path,
        tfidf_voc_path,
        logger,
        max_len=128,
    ):
        super().__init__(data_path, featurized_path, logger, max_len)
        self.tfidf_idf_path = tfidf_idf_path
        self.tfidf_voc_path = tfidf_voc_path

    def _featurize_train(self):
        data = self._load_data()

        self.logger.info("Fit tfidf vectorizer")
        tfidf = TfidfVectorizer(min_df=0.01)
        X_train = tfidf.fit_transform(data["source"].astype(str))

        data = (
            data[["id", "cell", "cell_type", "order", "source"]]
            .set_index("id", append=True)
            .swaplevel()
            .sort_index(level="id", sort_remaining=False)
        )

        df_order = data[["cell", "order"]]

        y_train = df_order.to_numpy()
        groups = df_order.groupby("id").size().to_numpy()

        # Add code cell ordering
        X_train = sparse.hstack(
            (
                X_train,
                np.where(
                    data["cell_type"] == "code",
                    data.groupby(["id", "cell_type"]).cumcount().to_numpy() + 1,
                    0,
                ).reshape(-1, 1),
            )
        )

        with open(self.tfidf_voc_path, "wb") as file:
            pickle.dump(tfidf.vocabulary_, file, 4)

        with open(self.tfidf_idf_path, "wb") as file:
            pickle.dump(tfidf.idf_, file, 4)

        with open(self.featurized_path, "wb") as featurized:
            pickle.dump([X_train, y_train, groups], featurized, pickle.HIGHEST_PROTOCOL)

    def _featurize_test(self):
        data = self._load_data()

        tfidf = TfidfVectorizer(min_df=0.01)
        tfidf.idf_ = pickle.load(open(self.tfidf_idf_path, "rb"))
        tfidf.vocabulary_ = pickle.load(open(self.tfidf_voc_path, "rb"))

        X_test = tfidf.transform(data["source"].astype(str))
        X_test = sparse.hstack(
            (
                X_test,
                np.where(
                    data["cell_type"] == "code",
                    data.groupby(["id", "cell_type"]).cumcount().to_numpy() + 1,
                    0,
                ).reshape(-1, 1),
            )
        )
        with open(self.featurized_path, "wb") as featurized:
            pickle.dump(X_test, featurized, pickle.HIGHEST_PROTOCOL)


class TransformersFeaturizer(Featurizer):
    def __init__(
        self,
        data_path,
        featurized_path,
        fts_out_path,
        model_name_or_path,
        num_selected_code_cells,
        md_max_len,
        total_max_len,
        logger,
    ):
        super().__init__(data_path, featurized_path, logger)
        self.fts_out_path = fts_out_path
        self.num_selected_code_cells = num_selected_code_cells
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def _featurize_train(self):
        train_data = self._load_data()
        self._preprocess(train_data)

    def _featurize_test(self):
        test_data = self._load_data()
        test_data["order"] = test_data.groupby(["id", "cell_type"]).cumcount()
        test_data["pred"] = test_data.groupby(["id", "cell_type"])["order"].rank(
            pct=True
        )
        self._preprocess(test_data)
    
    def _zeros(self, shape, dtype):
        '''alocation of desired memory during initialization, alternative to np.zeros'''
        array = np.empty(shape, dtype=dtype)
        array.fill(0)
        return array

    def _featurize(self, data, data_fts):
        shape = (len(data), self.total_max_len)

        inputs_ids = self._zeros(shape, dtype='int32')
        masks = self._zeros(shape, dtype='int32')
        inputs_features = self._zeros((len(data)), dtype='float32')
        ranks = self._zeros((len(data)), dtype='float32')
        for idx, row in tqdm(
            enumerate(data.itertuples()),
            desc="Feturise transformers data",
            total=len(data),
        ):
            inputs = self.tokenizer.encode_plus(
                row.source,
                None,
                add_special_tokens=True,
                max_length=self.md_max_len,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True,
                return_tensors="np",
            )

            code_inputs = self.tokenizer.batch_encode_plus(
                data_fts[row.id]["codes"],
                add_special_tokens=True,
                max_length=23,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )

            total_markdown = data_fts[row.id]["total_md"]
            total_code = data_fts[row.id]["total_code"]
            if total_markdown + total_code == 0:
                fts = 0
            else:
                fts = [total_markdown / float(total_markdown + total_code)]

            ids = np.concatenate(
                [
                    inputs["input_ids"].flatten(),
                    code_inputs["input_ids"].flatten(),
                ]
            )

            ids = ids[: self.total_max_len]
            ids = np.pad(
                ids,
                (0, self.total_max_len - len(ids)),
                "constant",
                constant_values=(self.tokenizer.pad_token_id),
            )

            assert len(ids) == self.total_max_len

            mask = np.concatenate(
                [
                    inputs["attention_mask"].flatten(),
                    code_inputs["attention_mask"].flatten(),
                ]
            )

            mask = mask[: self.total_max_len]
            mask = np.pad(
                mask,
                (0, self.total_max_len - len(mask)),
                "constant",
                constant_values=(self.tokenizer.pad_token_id),
            )

            inputs_ids[idx, :] = ids
            masks[idx, :] = mask
            inputs_features[idx] = np.asarray([fts])
            ranks[idx] = np.asarray([row.pct_rank])

        return inputs_ids, masks, inputs_features, ranks

    def _preprocess(self, data):
        data["pct_rank"] = data.groupby(["id", "cell_type"])["order"].rank(pct=True)

        data_fts = self._get_features(data)
        json.dump(data_fts, open(self.fts_out_path, "wt"))

        self.logger.info(f"Save data_fts to {self.fts_out_path}")
        data_markdowns = data[data["cell_type"] == "markdown"].reset_index(drop=True)
        ids, masks, fts, ranks = self._featurize(data_markdowns, data_fts)
        with open(self.featurized_path, "wb") as f:
            np.save(f, ids)
            np.save(f, masks)
            np.save(f, fts)
            np.save(f, ranks)

    def _sample_cells(self, cells, n):
        if n >= len(cells):
            return [cell[:200] for cell in cells]

        results = cells[:: len(cells) // n]

        if cells[-1] not in results:
            results[-1] = cells[-1]

        return results

    def _get_features(self, df):
        features = dict()
        df = df.sort_values("order").reset_index(drop=True)
        for idx, sub_df in tqdm(df.groupby("id"), desc="Get features"):
            features[idx] = dict()

            total_md = sub_df[sub_df.cell_type == "markdown"].shape[0]
            code_sub_df = sub_df[sub_df.cell_type == "code"]
            total_code = code_sub_df.shape[0]

            codes = self._sample_cells(
                code_sub_df.source.tolist(), self.num_selected_code_cells
            )

            features[idx]["total_code"] = total_code
            features[idx]["total_md"] = total_md
            features[idx]["codes"] = codes
        return features


def main():
    """Runs data processing scripts add features to clean data (saved in data/clean),
    splits the featurized data into training and test datasets and saves them as new
    dataset (in data/featurized)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--mode", type=str)

    parser.add_argument("--tfidf_idf_path", type=str)
    parser.add_argument("--tfidf_voc_path", type=str)
    parser.add_argument("--features_out_path", type=str)
    parser.add_argument("--num_selected_code_cells", type=int)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--md_max_len", type=int)
    parser.add_argument("--total_max_len", type=int)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--FEATURIZE--")

    if args.task == "xgbranker":
        dataset = XGBrankerFeaturizer(
            args.data,
            args.output,
            args.tfidf_idf_path,
            args.tfidf_voc_path,
            logger=logger,
        )
    if args.task == "transformer":
        dataset = TransformersFeaturizer(
            args.data,
            args.output,
            args.features_out_path,
            args.model_name_or_path,
            args.num_selected_code_cells,
            args.md_max_len,
            args.total_max_len,
            logger=logger,
        )

    dataset.featurize(mode=args.mode)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
