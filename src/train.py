import logging
import argparse

from .models import XGBrankerModel, TransformersModel
from .dataset import XGBrankerDataSet, TransformersDataset


def train_xgbranker(data_path, output_model_path):
    dataset = XGBrankerDataSet(data_path)
    X_train, y_train, groups = dataset.load_data()

    model = XGBrankerModel()
    model.model.fit(X_train, y_train[:, 1], group=groups)
    model.model.save_model(output_model_path)


def train_transformer(data_path, data_fts_path, model_name_or_path, md_max_len, total_max_len, batch_size, epochs):
    train_data = TransformersDataset(
        data_path,
        model_name_or_path=args.model_name_or_path,
        md_max_len=args.md_max_len,
        total_max_len=args.total_max_len,
        data_fts_path=data_fts_path,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.n_workers,
                          pin_memory=False, drop_last=True)


def main():
    """Runs data processing scripts add features to clean data (saved in data/clean),
    splits the featurized data into training and test datasets and saves them as new
    dataset (in data/featurized)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--task", type=str)

    parser.add_argument("--features_data_path", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--md_max_len", type=int)
    parser.add_argument("--total_max_len", type=int)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_workers", type=int, default=6)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--TRAIN MODEL--")

    if args.task == "xgbranker":
        train_xgbranker(args.data, args.output)
    if args.task == "transformer":
        train_transformer(
            args.data,
            args.feautures_data_path,
            args.model_name_or_path,
            args.md_max_len,
            args.total_max_len,
            args.batch_size,
            args.epochs,
        )

    logger.info(f"Save trained {args.task} model to {args.output}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
