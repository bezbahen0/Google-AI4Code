import os
import pickle
import logging
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class Dataset:
    def __init__(self, data_path, featurized_path, task, logger, max_len=128):
        self.data_path = data_path
        self.featurized_path = featurized_path
        self.task = task
        self.logger = logger
        if task == "transformers":
            self.max_len = max_len
            self.tokenizer_vectorizer = lambda text: models(
                text,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True,
            )

    def prepare_for_training(self):
        """
        Prepares the dataset for training
        """

        if self.task == "transformers":
            self._featureize_transformers()
        if self.task == "xgbranker":
            self._featurize_xgbranker()

        self.logger.info(f"Saving featurized data to {self.featurized_path}")

    def _load_data(self):
        self.logger.info(f"Loading clean data from {self.data_path}")
        return pd.read_parquet(self.data_path)

    def _featureize_transformers(self):
        data = self._load_data()
        pass

    def _featurize_xgbranker(self):
        data = self._load_data()
        tfidf = TfidfVectorizer(min_df=0.01)
        self.logger.info("Fit tfidf vectorizer")
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

        with open(os.path.join(os.path.dirname(self.featurized_path), 'tfidf.pkl'), "wb") as vectorizer:
            pickle.dump(tfidf, vectorizer, pickle.HIGHEST_PROTOCOL)

        with open(self.featurized_path, "wb") as featurized:
            pickle.dump([X_train, y_train, groups], featurized, pickle.HIGHEST_PROTOCOL)


def main():
    """Runs data processing scripts add features to clean data (saved in data/clean),
    splits the featurized data into training and test datasets and saves them as new
    dataset (in data/featurized)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--FEATURIZE--")

    dataset = Dataset(args.data, args.output, args.task, logger)
    dataset.prepare_for_training()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
