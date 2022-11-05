import json
import os
import random
import re
from pathlib import Path

import datetime
import numpy as np
import torch
import torchvision
from tqdm.notebook import tqdm
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_url


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_img_name(path):
    m = re.search(r"([^/]+\.[^/]+)$", path)
    return m.group()


def fine_tune(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    device,
    epochs=50,
    model_name="default",
):
    if model_name == "default":
        dt = str(datetime.date.today())
        model_name = "model" + dt

    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            # data = data.to(device)
            # label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
        train_acc_list.append(epoch_accuracy)
        val_acc_list.append(epoch_val_accuracy)
        train_loss_list.append(epoch_loss)
        val_loss_list.append(epoch_val_loss)

    # モデルの保存
    model_cpu = model.to("cpu")
    model_save_path = f"model/{model_name}.pth"
    torch.save(model_cpu.state_dict(), model_save_path)
    print(f"fine-tuned model has been saved at {model_save_path}!")


def _get_img_paths(img_dir):
    img_dir = Path(img_dir)
    img_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    img_paths = [str(p) for p in img_dir.iterdir() if p.suffix in img_extensions]
    img_paths.sort()

    return img_paths


class ImageFolder(Dataset):
    def __init__(self, img_dir, transform):
        # 画像ファイルのパス一覧を取得する。
        self.img_paths = _get_img_paths(img_dir)
        self.transform = transform

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path)
        inputs = self.transform(img)

        return {"image": inputs, "path": path}

    def __len__(self):
        return len(self.img_paths)
