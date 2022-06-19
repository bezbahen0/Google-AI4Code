import os
import re

import logging
import argparse
import pandas as pd

from tqdm import tqdm
from pandarallel import pandarallel
from .models import TranslationModel, LanguageIdentification

pandarallel.initialize(progress_bar=True)

NUM_PREDICTION_MARKDOWNS = 5  # median of count markdown cells // 2


def get_notebooks_lang(data, model_path, save_path):
    language_ident = LanguageIdentification(model_path)

    data = data.groupby("id")['source'].apply(lambda x: " ".join(x))
    languages = language_ident.predict_lang(data.to_list())[0]
    
    languages = pd.Series(languages, index=data.index)
    languages = languages.apply(lambda x: "".join(x).replace("__label__", ""))
    
    languages.to_csv(save_path, index=False)
    return languages


def get_not_target_index(df, target_lang="en"):
    return df[df != target_lang].index


def main():
    """Runs data processing scripts to turn cleaned data from (data/clean) into
    translated cleaned data ready to be analyzed (saved in data/translated).
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--translation_out", type=str)
    parser.add_argument("--lang_ident_out", type=str)
    parser.add_argument("--target_lang", type=str)
    parser.add_argument("--fasttext_ident_path", type=str)
    parser.add_argument("--translation_model_path", type=str)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--TRANSLATION--")

    logger.info(f"Loading data from {args.data}")

    # loading steps go here
    data = pd.read_parquet(args.data)

    markdowns_data = data[data.cell_type == "markdown"][['id', 'cell', 'source']]

    logger.info(
        f"Start predict languages for {len(markdowns_data.id.unique())} notebooks"
    )
    notebooks_lang_df = get_notebooks_lang(
        markdowns_data,
        args.fasttext_ident_path,
        args.lang_ident_out,
    )

    need_translate_ids = get_not_target_index(
        notebooks_lang_df, target_lang=args.target_lang
    )

    logger.info(
        f"Translate {len(need_translate_ids)} markdowns cells to {args.target_lang}"
    )

    translated = data.loc[df["id"].isin(need_translate_ids)]
    
    # data.loc[data.cell_type == "markdown", 'source'] = data.loc[data.cell_type == "markdown", 'source'].parallel_apply(sub_all)

    logger.info(f"Saving translated data to {args.output}")

    # saving steps go here
    translated.to_parquet(args.output)
    # data.to_parquet(args.output)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
