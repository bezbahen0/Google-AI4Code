import os

import fasttext
from xgboost import XGBRanker

from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)


class TransformersModel:
    def __init__(self, model_path):
        self.model = AutoModel.from_pretrained(model_path, num_labels=1)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.Huber(),
        )

        # self.top = nn.Linear(769, 1)
        self.tokenzier = AutoTokenizer.from_pretrained(model_path)


class TranslationModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    def translate(self, text):
        encoded_zh = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded_zh)
        result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return result


class LanguageIdentification:
    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path)

    def predict_lang(self, text, only_language=True):
        predictions = self.model.predict(text, k=1)  # returns top 1 matching languages
        return predictions


class XGBrankerModel:
    def __init__(self):
        self.model = XGBRanker(
            min_child_weight=10,
            subsample=0.5,
            tree_method="hist",
            verbosity=2,
            n_jobs=os.cpu_count(),
        )


if __name__ == "__main__":
    """
    Helsinki-NLP/opus-mt-ine-en support source languages:
        ['ca', 'es', 'os', 'ro', 'fy', 'cy', 'sc', 'is', 'yi', 'lb', 'an',
         'sq', 'fr', 'ht', 'rm', 'ps', 'af', 'uk', 'sl', 'lt', 'bg', 'be',
         'gd', 'si', 'en', 'br', 'mk', 'or', 'mr', 'ru', 'fo', 'co', 'oc',
         'pl', 'gl', 'nb', 'bn', 'id', 'hy', 'da', 'gv', 'nl', 'pt', 'hi',
         'as', 'kw', 'ga', 'sv', 'gu', 'wa', 'lv', 'el', 'it', 'hr', 'ur',
         'nn', 'de', 'cs', 'ine']

    haved languages
    ['en', 'pt', 'ko', 'ja', 'ru', 'tr', 'es', 'zh', 'fr', 'id', 'vi', 'de',
       'it', 'th', 'pl', 'ca', 'uk', 'nl', 'fa', 'mn', 'hu', 'ar', 'da', 'cs',
       'ms', 'ro', 'el', 'hr', 'no', 'sr', 'fi', 'kn', 'la', 'sv', 'si', 'sk',
       'az', 'hi', 'sl', 'bn', 'ce', 'eo', 'uz', 'gl', 'he', 'sh', 'oc', 'eu',
       'ml', 'ht', 'tl', 'bg', 'nd', 'gd', 'my', 'lt', 'ie', 'mr', 'pa', 'te',
       'bs', 'ur', 'ne', 'qu', 'be', 'fy', 'jb', 'ta', 'sq', 'ky', 'af', 'mk',
       'ia']
    """
    model = TranslationModel("Helsinki-NLP/opus-mt-zh-en")
    ru_text = [
        # "Ne vous mêlez pas des affaires des sorciers, car ils sont insidieux et prompts à la colère."
        "不要插手巫師的事務, 因為他們是微妙的, 很快就會發怒."
    ]
    result = model.translate(ru_text)
    print(result)
