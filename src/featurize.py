import os
import pickle
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
        pass

    def _featureize_train(self):
        pass

    def _featureize_test(self):
        pass

    def _load_data(self):
        self.logger.info(f"Loading clean data from {self.data_path}")
        return pd.read_parquet(self.data_path)


class XGBrankerFeaturizer(Featurizer):
    def __init__(self, data_path, featurized_path, logger, max_len=128):
        super().__init__(data_path, featurized_path, logger, max_len)
        self.tfidf_idf_path = os.path.join(
            os.path.dirname(self.featurized_path), "tfidf_idf.pkl"
        )
        self.tfidf_voc_path = os.path.join(
            os.path.dirname(self.featurized_path), "tfidf_voc.pkl"
        )

    def featurize(self, mode="test"):
        if mode == "train":
            self._featureize_train()
        if mode == "test":
            self._featureize_test()

        self.logger.info(f"Save featurized train data to {self.featurized_path}")

    def _featureize_train(self):
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

        df_ranks = data[["cell", "order"]]

        y_train = df_ranks.to_numpy()
        groups = df_ranks.groupby("id").size().to_numpy()

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

    def _featureize_test(self):
        tfidf = TfidfVectorizer(min_df=0.01)
        tfidf.idf_ = pickle.load(open(self.tfidf_idf_path, "rb"))
        tfidf.vocabulary_ = pickle.load(open(self.tfidf_voc_path, "rb"))

        X_test = tfidf.transform(df_test["source"].astype(str))
        X_test = sparse.hstack(
            (
                X_test,
                np.where(
                    df_test["cell_type"] == "code",
                    df_test.groupby(["id", "cell_type"]).cumcount().to_numpy() + 1,
                    0,
                ).reshape(-1, 1),
            )
        )
        with open(self.featurized_path, "wb") as featurized:
            pickle.dump(X_test, featurized, pickle.HIGHEST_PROTOCOL)


class TransformersFeaturizer(Featurizer):
    def __init__(self, data_path, featurized_path, logger, max_len=128):
        super().__init__(data_path, featurized_path, logger, max_len)

    def featurize(self, mode="test"):
        pass

    def _featureize_train(self):
        pass

    def _featureize_test(self):
        pass


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
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--FEATURIZE--")

    if args.task == "xgbranker":
        dataset = XGBrankerFeaturizer(args.data, args.output, logger)
    if args.task == "transformers":
        dataset = TransformersFeaturizer(args.data, args.output, logger)

    dataset.featurize(mode=args.mode)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
