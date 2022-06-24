import os
import re

import logging
import argparse
import pandas as pd

from tqdm import tqdm
from .models import HelsinkiTranslationModel, LanguageIdentification
from .language_groups import GROUPS

tqdm.pandas()


def get_notebooks_lang(data, model_path, save_path):
    language_ident = LanguageIdentification(model_path)

    data = data.groupby("id")["source"].apply(lambda x: " ".join(x))
    languages = language_ident.predict_lang(data.to_list())[0]

    languages = pd.Series(languages, index=data.index)
    languages = languages.apply(lambda x: "".join(x).replace("__label__", ""))
    languages.to_csv(save_path)
    return languages


def get_not_target_indexs(df, supported_langs, target_lang="en"):
    return df[df.isin(supported_langs)][df != target_lang].index


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
    parser.add_argument("--helsinki_model_path", type=str)
    parser.add_argument("--helsinki_model_group", type=str)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--TRANSLATION--")

    logger.info(f"Loading data from {args.data}")

    # loading steps go here
    data = pd.read_parquet(args.data)

    markdowns_data = data[data.cell_type == "markdown"][["id", "cell", "source"]]

    logger.info(
        f"Start predict languages for {len(markdowns_data.id.unique())} notebooks"
    )
    languages_df = get_notebooks_lang(
        markdowns_data,
        args.fasttext_ident_path,
        args.lang_ident_out,
    )

    if args.helsinki_model_path is not None:
        translation_model = HelsinkiTranslationModel(args.helsinki_model_path)
    else:
        model_name = (
            f"Helsinki-NLP/opus-mt-{args.helsinki_model_group}-{args.target_lang}"
        )
        translation_model = HelsinkiTranslationModel(model_name)

    need_translate_ids = get_not_target_indexs(
        languages_df, GROUPS[args.helsinki_model_group], target_lang=args.target_lang
    )

    logger.info(
        f"Try translate {len(need_translate_ids)} notebooks to {args.target_lang}"
    )

    to_translate = data.loc[data["id"].isin(need_translate_ids)]   

    to_translate.loc[data.cell_type == "markdown", "source"] = to_translate.loc[
        data.cell_type == "markdown", "source"
    ].progress_apply(translation_model.predict_large_text)

    # data.loc[data.cell_type == "markdown", 'source'] = data.loc[data.cell_type == "markdown", 'source'].parallel_apply(sub_all)

    logger.info(f"Saving translated data to {args.translation_out}")

    # saving steps go here
    to_translate.to_parquet(args.translation_out)
    # data.to_parquet(args.output)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
