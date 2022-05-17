import logging
import os

import yaml
from dotenv import find_dotenv, load_dotenv

seed = yaml.safe_load(open("params.yaml"))["general"]["seed"]


def main():
    """ Runs data processing scripts add features to clean data (saved in data/clean),
        splits the featurized data into training and test datasets and saves them as new
        dataset (in data/featurized)
    """

    logger = logging.getLogger(__name__)
    logger.info("--FEATURIZE--")

    clean_data_path = os.path.join("data", "clean")
    featurized_data_path = os.path.join("data", "featurized")

    logger.info(f"Loading clean data from {clean_data_path}")
    # loading steps go here

    logger.info("Featurizing data")
    # featurizing steps go here

    logger.info(f"Saving featurized data to {featurized_data_path}")
    # saving steps go here


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
