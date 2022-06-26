import os

import fasttext
import torch
from xgboost import XGBRanker

from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MarianTokenizer,
    MarianMTModel,
    AdamW,
    get_linear_schedule_with_warmup,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class TransformersModel:
    def __init__(self, model_path):
        self.model = AutoModel.from_pretrained(model_path, num_labels=1)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.Huber(),
        )

        # self.top = nn.Linear(769, 1)
        self.tokenzier = AutoTokenizer.from_pretrained(model_path)


class HelsinkiTranslationModel:
    def __init__(self, model_path, max_length=512):
        self.tokenizer = MarianTokenizer.from_pretrained(model_path)
        self.model = MarianMTModel.from_pretrained(model_path).to(device)
        self.max_length = max_length

    def predict(self, text):
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        encoded = encoded.to(device)

        generated_tokens = self.model.generate(**encoded)
        result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return result[0]

    def predict_batch(self, list_text):
        if not list_text:
            return list_text

        encoded_batch = self.tokenizer.batch_encode_plus(
            list_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        )
        encoded_batch = encoded_batch.to(device)

        generated_tokens = self.model.generate(**encoded_batch)
        result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return result

    def predict_large_text(self, text):
        if len(text.split()) > self.max_length:
            return self.predict(text)

        chunks = []
        for text_chunk in self._split(text):
            chunks.append(text_chunk)

        translated_chunks = self.predict_batch(chunks)
        return " ".join(translated_chunks)

    def translate(self, list_text):
        translation_list = []
        for tr in list_text:
            translation_list.append(self.predict_large_text(tr))
        return translation_list

    def supported_lang(self):
        return self.tokenizer.source_lang

    def _split(self, text):
        splited_text = text.split()
        for i in range(0, len(splited_text), self.max_length):
            yield " ".join(splited_text[i : i + self.max_length])


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
    model = HelsinkiTranslationModel("Helsinki-NLP/opus-mt-ine-en")
    text = [
        "Ne vous mêlez pas des affaires des sorciers, car ils sont insidieux et prompts à la colère.",
        "不要插手巫師的事務, 因為他們是微妙的, 很快就會發怒.",
    ]
    result = model.translate(text)
    print(result)
    print(model.supported_lang())
