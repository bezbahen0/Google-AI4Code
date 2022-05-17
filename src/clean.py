import logging
import os

import yaml
from dotenv import find_dotenv, load_dotenv

seed = yaml.safe_load(open("params.yaml"))["general"]["seed"]


def main():
    """ Runs data processing scripts to turn raw data from (data/raw) into
        cleaned data ready to be analyzed (saved in data/clean).
    """

    logger = logging.getLogger(__name__)
    logger.info("--CLEAN--")

    raw_data_path = os.path.join("data", "raw")
    clean_data_path = os.path.join("data", "clean")

    logger.info(f"Loading raw data from {raw_data_path}")
    # loading steps go here

    logger.info("Cleaning data")
    # cleaning steps go here

    logger.info(f"Saving cleaned data to {clean_data_path}")
    # saving steps go here


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
