import os
from xgboost import XGBRanker

from transformers import (
    AutoModel,
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

class XGBrankerModel:
    def __init__(self):
        self.model = XGBRanker(
            min_child_weight=10,
            subsample=0.5,
            tree_method="hist",
            verbosity=2,
            n_jobs=os.cpu_count()
        )