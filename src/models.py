import os

import fasttext
import torch
from xgboost import XGBRanker

import ctranslate2
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
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


class MarianMTModel:
    def __init__(self, model_path, tokenizer_path, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = ctranslate2.Translator(model_path, device=device)
        self.max_length = max_length

    def predict(self, text):
        source = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(
                text, padding=True, truncation=True, max_length=self.max_length
            )
        )
        results = self.model.translate_batch([source])
        results = results[0].hypotheses[0]

        results = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(results))
        return results

    def predict_batch(self, list_text):
        if not list_text:
            return list_text

        tokenized_batch = self.tokenizer.batch_encode_plus(
            list_text, padding=True, truncation=True, max_length=self.max_length
        )
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
        "data/pretrained_models/opus-mt-zh-en-converted", device="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

    text = [
        "不要插手巫師的事務, 因為他們是微妙的, 很快就會發怒 很快就會發怒.",
        "不要插手巫師的事務, 因為他們是微妙的, 很快就會發怒.",
        "不要插手巫師的事務, 因為他們是微妙的, 很快就會發怒.",
    ]
    # text = "Ne vous mêlez pas des affaires des sorciers, car ils sont insidieux et prompts à la colère."
    tokenized_batch = tokenizer.batch_encode_plus(text)
    source = list(map(tokenizer.convert_ids_to_tokens, tokenized_batch.input_ids))
    results = translator.translate_batch(source)
    print(
        tokenizer.batch_decode(
            [tokenizer.convert_tokens_to_ids(res.hypotheses[0]) for res in results],
            # list(map(tokenizer.convert_tokens_to_ids, results.hypotheses[0])),
            skip_special_tokens=True,
        )
    )
