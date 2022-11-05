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

from utils import ImageFolder, fine_tune, seed_everything, get_img_name

# from google.colab import drive
# drive.mount('/content/drive')

# Training settings
epochs = 50
lr = 3e-5
gamma = 0.7
seed = 42
device = "cuda"
train_dataset_dir = Path("./data/train")
val_dataset_dir = Path("./data/valid")
test_dataset_dir = Path("./data/test")

seed_everything(seed)

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


train_data = datasets.ImageFolder(train_dataset_dir, train_transforms)
valid_data = datasets.ImageFolder(val_dataset_dir, val_transforms)
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=16, shuffle=True)

model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=2)
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
test_dataset = ImageFolder(test_dataset_dir, val_transforms)
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
