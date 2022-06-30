import os
import re
import glob

import logging
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from .models import MarianMTModel, LanguageIdentification

tqdm.pandas()


def predict_notebooks_lang(data, model_path, save_path):
    language_ident = LanguageIdentification(model_path)

    data = data.groupby("id")["source"].apply(lambda x: " ".join(x))
    languages = language_ident.predict_lang(data.to_list())[0]

    languages = pd.Series(languages, index=data.index)
    languages = languages.apply(lambda x: "".join(x).replace("__label__", ""))
    languages.to_csv(save_path)
    return languages


def target_indexs(df, target_lang):
    return df[df == target_lang]


def get_supported_languages(models_names):
    """model name have format: opus-mt-{source_lang}-{target_lang}"""
    languages = [model_name.replace("opus-mt-", "")[:2] for model_name in models_names]
    return languages, models_names


def main():
    """Runs data processing scripts to turn cleaned data from (data/clean) into
    translated cleaned data  (saved in data/translated).
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--lang_ident_out", type=str)
    parser.add_argument("--target_lang", type=str)
    parser.add_argument("--fasttext_ident_path", type=str)
    parser.add_argument("--marianmt_models_dir_path", type=str)
    parser.add_argument("--tokenizers_dir_path", type=str)
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

    languages_df = predict_notebooks_lang(
        markdowns_data,
        args.fasttext_ident_path,
        args.lang_ident_out,
    )

    languages, models_names = get_supported_languages(
        os.listdir(args.marianmt_models_dir_path)
    )
    if len(languages_df[languages_df != args.target_lang].index) == 0:
        logger.info(f"No notebooks found != {args.target_lang}, abort")
        return

    for language, model_name in zip(languages, models_names):
        model_path = os.path.join(args.marianmt_models_dir_path, model_name)
        tokenizer_path = os.path.join(args.tokenizers_dir_path, model_name)
        translation_model = MarianMTModel(
            model_path=model_path, tokenizer_path=tokenizer_path
        )

        need_translate_ids = target_indexs(languages_df, language).index
        to_translate = data.loc[data["id"].isin(need_translate_ids)]

        description = f"Translate {len(need_translate_ids)} from {language} notebooks to {args.target_lang}"

        translated = []
        num_slices = len(to_translate) // args.batch_size
        num_slices = 1 if num_slices == 0 else num_slices
        for df in tqdm(np.array_split(to_translate, num_slices), desc=description):
            matching = df.cell_type == "markdown", "source"

            df.loc[matching] = translation_model.predict_batch(
                df.loc[matching].to_list()
            )
            translated.append(df)

        translated = pd.concat(translated)
        data = data.drop(translated.index)
        data = pd.concat([data, translated])

    logger.info(f"Saving translated data to {args.output}")

    # saving steps go here
    data.to_parquet(args.output)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
