import os
import torch
from transformers import pipeline, TrainingArguments, Trainer, DefaultDataCollator
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
from PIL import Image
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

MODEL_NAME = ""  # huggingface上の読み込みたいモデル名
feature_extractor = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# ハイパラ設定(必要であれば)

# データセットの読み込み
# https://note.com/npaka/n/n17ecbd890cd6
# https://huggingface.co/docs/transformers/tasks/image_classification
DATASET_PATH = "./data/experiment/"
data_dict = {"image": [], "label": []}
for flg in ["true", "false"]:
    for f in os.listdir(
        path=os.path.join(
            DATASET_PATH,
            flg,
        )
    ):
        if f.startswith("."):
            continue
        im = Image.open(os.path.join(DATASET_PATH, flg, f))
        data_dict["image"].append(im)
        data_dict["label"].append(flg == "true")
dd = DatasetDict({"train": data_dict})
# 前処理

data_collator = DefaultDataCollator()
# train
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dd["train"],
    # eval_dataset=food["test"],
    tokenizer=feature_extractor,
)
# test

# モデルの保存
