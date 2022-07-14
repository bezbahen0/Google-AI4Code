import tensorflow as tf
from transformers import TFAutoModel, AutoModel
import torch
import torch.nn as nn

def get_model() -> tf.keras.Model:
    backbone = TFAutoModel.from_pretrained('microsoft/codebert-base')
    input_ids = tf.keras.layers.Input(
        shape=(512,),
        dtype=tf.int32,
        name="input_ids",
    )
    attention_mask = tf.keras.layers.Input(
        shape=(512,),
        dtype=tf.int32,
        name="attention_mask",
    )
    feature = tf.keras.layers.Input(
        shape=(1,),
        dtype=tf.float32,
        name="feature",
    )
    x = backbone({"input_ids": input_ids, "attention_mask": attention_mask})[0]
    x = tf.concat([x[:, 0, :], feature], axis=1)
    outputs = tf.keras.layers.Dense(1, activation="linear", dtype="float32")(x)
    return tf.keras.Model(
        inputs=[input_ids, attention_mask, feature],
        outputs=outputs,
    )

class TransformersModel(nn.Module):
    def __init__(self):
        super(TransformersModel, self).__init__()
        self.model = AutoModel.from_pretrained('microsoft/codebert-base')
        self.top = nn.Linear(769, 1)
        
    def forward(self, ids, mask, fts):
        x = self.model(ids, mask)[0]
        x = self.top(torch.cat((x[:, 0, :], fts),1))
        return x

model = get_model()
model.load_weights("data/pretrained_models/model_0.h5")
print(model.summary())
print(model.layers[0].get_weights()[0])

model = TransformersModel()
print(model.layer[0].weight)
torch.save(model.state_dict(), "data/pretrained_models/test_pt_model_0.pt")
