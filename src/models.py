import os

import fasttext

from xgboost import XGBRanker

import ctranslate2

import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

class TransformersModel(nn.Module):
    def __init__(self, model_path):
        super(TransformersModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.top = nn.Linear(769, 1)
        
    def forward(self, ids, mask, fts):
        x = self.model(ids, mask)[0]
        x = self.top(torch.cat((x[:, 0, :], fts),1))
        x = torch.sigmoid(x)
        return x


class MarianMTModel:
    def __init__(self, model_path, tokenizer_path, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = ctranslate2.Translator(model_path, device=device)
        self.max_length = max_length

    def predict(self, text):
        text = text[:max_length]
        source = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        results = self.model.translate_batch([source])
        results = results[0].hypotheses[0]

        results = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(results))
        return results

    def predict_batch(self, list_text):
        if not list_text:
            return list_text

        truncation_text = [text[: self.max_length] for text in list_text]
        tokenized_batch = self.tokenizer.batch_encode_plus(truncation_text)
        source = list(
            map(self.tokenizer.convert_ids_to_tokens, tokenized_batch.input_ids)
        )
        results = self.model.translate_batch(source)
        results = self.tokenizer.batch_decode(
            [
                self.tokenizer.convert_tokens_to_ids(res.hypotheses[0])
                for res in results
            ],
            skip_special_tokens=True,
        )

        return results


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
    haved languages
    ['en', 'pt', 'ko', 'ja', 'ru', 'tr', 'es', 'zh', 'fr', 'id', 'vi', 'de',
       'it', 'th', 'pl', 'ca', 'uk', 'nl', 'fa', 'mn', 'hu', 'ar', 'da', 'cs',
       'ms', 'ro', 'el', 'hr', 'no', 'sr', 'fi', 'kn', 'la', 'sv', 'si', 'sk',
       'az', 'hi', 'sl', 'bn', 'ce', 'eo', 'uz', 'gl', 'he', 'sh', 'oc', 'eu',
       'ml', 'ht', 'tl', 'bg', 'nd', 'gd', 'my', 'lt', 'ie', 'mr', 'pa', 'te',
       'bs', 'ur', 'ne', 'qu', 'be', 'fy', 'jb', 'ta', 'sq', 'ky', 'af', 'mk',
       'ia']
    """
    translator = ctranslate2.Translator(
        "data/pretrained_models/converted/opus-mt-ine-en", device="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ine-en")

    text = [
        " Задаем функцию для подсчета метрик",
        "Описание датасета Id идентификационный номер квартиры Helsinki_ Helsinki_",
        "Загрузка данных ",
    ]
    processed = []
    for t in text:
        processed.append(t[:512])
    tokenized_batch = tokenizer.batch_encode_plus(processed)
    source = list(map(tokenizer.convert_ids_to_tokens, tokenized_batch.input_ids))
    results = translator.translate_batch(source)
    print(
        tokenizer.batch_decode(
            [tokenizer.convert_tokens_to_ids(res.hypotheses[0]) for res in results],
            # list(map(tokenizer.convert_tokens_to_ids, results.hypotheses[0])),
        )
    )
