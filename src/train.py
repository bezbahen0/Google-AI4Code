import pickle
import logging
import argparse


from .models import XGBrankerModel

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
    logger.info("--TRAIN MODEL--")

    with open(args.data, "rb") as input_file:
        X_train, y_train, groups = pickle.load(input_file)

    model = XGBrankerModel()
    model.model.fit(X_train, y_train, group=groups, verbose=True)
    model.model.save_model(args.output)

    

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()