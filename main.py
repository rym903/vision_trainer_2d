import torch
from transformers import pipeline
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
import datasets 

MODEL_NAME = "" # huggingface上の読み込みたいモデル名
feature_extracter = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# データセットの読み込み

# train

# test

# モデルの保存