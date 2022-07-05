import logging
import os
import re

import yaml
import argparse
import pandas as pd

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


def sub_html_tags(text):
    cleared_text = re.sub(r"<.*?>", "", text)
    return cleared_text


def sub_latex_math(text):
    regex = r"(\$+)(?:(?!\1)[\s\S])*\1"
    regex1 = r"\\begin.*?\\end{.*?}"
    regex2 = r"\\[a-zA-Z]+"

    # Some text have \b who ident as \x08 this need change
    cleared_text = text.replace("\b", "\\b")
    cleared_text = cleared_text.replace("\n", "")

    cleared_text = re.sub(regex, "", cleared_text)
    cleared_text = re.sub(regex1, "", cleared_text)
    cleared_text = re.sub(regex2, "", cleared_text)
    return cleared_text


def sub_links(text):
    cleared_text = re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "", text)
    return cleared_text


def sub_email(text):
    cleared_text = re.sub(r"\S*@\S*\s?", "", text)
    return cleared_text


def preprocess_text(text):
    # Remove all the special characters
    text = re.sub(r"\W", " ", str(text))

    # remove all single characters
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)

    # Remove single characters from the start
    text = re.sub(r"\^[a-zA-Z]\s+", " ", text)

    # Substituting multiple spaces with single space
    text = re.sub(r"\s+", " ", text, flags=re.I)

    # Removing prefixed 'b'
    text = re.sub(r"^b\s+", "", text)

    # Converting to Lowercase
    text = text.lower()  # warning: the translation model is case-sensitive

    # remove digits
    text = re.sub(r"[0-9]+", "", text)
    return text


def markdown_sub_all(text):
    text = sub_html_tags(text)
    text = sub_latex_math(text)
    text = sub_links(text)
    text = sub_email(text)
    text = preprocess_text(text)
    return text


def code_sub_all(text):
    text = text.replace("\\n", "\n")
    return text


def clear_markdown(data, logger):
    logger.info("Cleaning data markdowns cells source")
    data.loc[data.cell_type == "markdown", "source"] = data.loc[
        data.cell_type == "markdown", "source"
    ].parallel_apply(markdown_sub_all)
    return data


def clear_code(data, logger):
    logger.info("Cleaning data code cells source")
    data.loc[data.cell_type == "code", "source"] = data.loc[
        data.cell_type == "code", "source"
    ].parallel_apply(code_sub_all)
    return data


def main():
    """Runs data processing scripts to turn raw data from (data/raw) into
    cleaned data ready to be analyzed (saved in data/clean).
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--clear", type=str)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--CLEAN--")

    logger.info(f"Loading raw data from {args.data}")

    data = pd.read_parquet(args.data)

    if args.clear == "all":
        data = clear_markdown(data)
        data = clear_code(data)

    if args.clear == "markdown":
        data = clear_markdown(data)

    if args.clear == "code":
        data = clear_code(data)

    logger.info(f"Save cleaned data to {args.output}")
    data.to_parquet(args.output)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
