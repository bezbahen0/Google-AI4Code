import os
import pickle
import json
import logging
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


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
        processed_out_path,
        num_selected_code_cells,
        logger,
    ):
        super().__init__(data_path, featurized_path, logger)
        self.fts_out_path = fts_out_path
        self.processed_out_path = processed_out_path
        self.num_selected_code_cells = num_selected_code_cells

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

    def _preprocess(self, data):
        data["pct_rank"] = data.groupby(["id", "cell_type"])["order"].rank(pct=True)

        data_fts = self._get_features(data)
        json.dump(data_fts, open(self.fts_out_path, "wt"))

        self.logger.info(f"Save data_fts to {self.fts_out_path}")

        data_markdowns = data[data["cell_type"] == "markdown"].reset_index(drop=True)
        data_markdowns.to_parquet(self.featurized_path)
        data.to_parquet(self.processed_out_path)

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
    
    # XGBRanker needed
    parser.add_argument("--tfidf_idf_path", type=str)
    parser.add_argument("--tfidf_voc_path", type=str)

    # Transformers needed
    parser.add_argument("--processed_out_path", type=str)
    parser.add_argument("--features_out_path", type=str)
    parser.add_argument("--num_selected_code_cells", type=int)
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
            args.processed_out_path,
            args.num_selected_code_cells,
            logger=logger,
        )

    dataset.featurize(mode=args.mode)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
