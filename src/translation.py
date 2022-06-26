import os
import re

import logging
import argparse
import pandas as pd
import numpy as np

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
    parser.add_argument("--output", type=str)
    parser.add_argument("--lang_ident_out", type=str)
    parser.add_argument("--target_lang", type=str)
    parser.add_argument("--fasttext_ident_path", type=str)
    parser.add_argument("--helsinki_model_path", type=str)
    parser.add_argument("--helsinki_model_group", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
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

    # to_translate.loc[to_translate.cell_type == "markdown", "source"] = to_translate.loc[
    #    to_translate.cell_type == "markdown", "source"
    # ].progress_apply(translation_model.predict_large_text)

    # Slice dataframe to batch and predict as batch
    translated = []
    num_slices = len(to_translate) // args.batch_size
    for df in tqdm(np.array_split(to_translate, num_slices), desc="Translate"):
        df.loc[df.cell_type == "markdown", "source"] = translation_model.predict_batch(
            df.loc[df.cell_type == "markdown", "source"].to_list()
        )
        translated.append(df)

    translated = pd.concat(translated)
    print(translated)
    print(to_translate)
    data = data.drop(translated.index)
    data = pd.concat([data, translated])

    logger.info(f"Saving translated data to {args.output}")

    # saving steps go here
    data.to_parquet(args.output)
    # data.to_parquet(args.output)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
