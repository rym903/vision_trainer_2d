import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

from vit_pytorch.efficient import ViT
from pathlib import Path
import seaborn as sns
import timm
from pprint import pprint

from utils import ImageFolderForPrediction, fine_tune, seed_everything, get_img_name
from model_environment import get_model_env

# from google.colab import drive
# drive.mount('/content/drive')

# Training settings
model_env = get_model_env()
epochs = model_env["epochs"]
lr = model_env["lr"]
gamma = model_env["gamma"]
device = model_env["device"]
train_dataset_dir = model_env["train_dataset_dir"]
val_dataset_dir = model_env["val_dataset_dir"]
test_dataset_dir = model_env["test_dataset_dir"]
seed = 42
seed_everything(seed)
train_transforms = model_env["train_transforms"]
val_transforms = model_env["val_transforms"]

train_data = datasets.ImageFolder(train_dataset_dir, train_transforms)
valid_data = datasets.ImageFolder(val_dataset_dir, val_transforms)
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=16, shuffle=True)

model = timm.create_model(model_env["model_base_name"], pretrained=True, num_classes=2)
# model.to("cuda:0")

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

fine_tune(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    device,
    epochs,
    model_name="test",
)


# 推論
# Dataset を作成する。
test_dataset = ImageFolderForPrediction(test_dataset_dir, val_transforms)
# DataLoader を作成する。
test_dataloader = DataLoader(test_dataset, batch_size=8)

predict_df = pd.DataFrame({"path": [], "label": []}, dtype=(str, int))
for batch in test_dataloader:
    inputs = batch["image"].to(device)
    outputs = model(inputs)

    labels = outputs.argmax(dim=1)

    result = {"path": batch["path"], "label": labels.to(torch.int64).to("cpu")}
    predict_df = pd.concat([predict_df, pd.DataFrame(result)])

predict_df["img_name"] = predict_df["path"].apply(get_img_name)
