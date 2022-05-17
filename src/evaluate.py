import logging
import os

import yaml
from dotenv import find_dotenv, load_dotenv

seed = yaml.safe_load(open("params.yaml"))["general"]["seed"]
model_params = yaml.safe_load(open("params.yaml"))["model"]


def main():
    """ Trains and evaluates model. Saves predictions for exploration (data/predictions)
        a Kaggle-ready submission file
    """

    logger = logging.getLogger(__name__)
    logger.info("--EVALUATE--")

    featurized_data_path = os.path.join("data", "featurized")
    scores_data_path = 
    predictions_data_path = os.path.join("data", "predictions")
    submissions_data_path = os.path.join("data", "submissions")

    logger.info(f"Loading featurized data from {featurized_data_path}")
    # loading steps go here

    logger.info("Calculating scores and creating evaluation plots")
    # scoring steps go here

    logger.info(f"Creating prediction datasets to {predictions_data_path}")
    # prediction saving steps go here

    logger.info(f"Creating a submission file to {submissions_data_path}")
    # submission creation steps go here


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
