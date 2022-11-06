import pandas as pd
import torch
import timm
from torch.utils.data import DataLoader
from utils import ImageFolderForPrediction, get_img_name
from model_environment import get_model_env


def predict(img_dir, model_path, device):
    """_summary_

    Args:
        img_dir (_type_): _description_
        model_name (_type_): _description_
        device (_type_): _description_
    Returns:
        df_true: 好きフラグがついた画像のdf
        df_false: 好きでないフラグがついた画像のdf
    dfのカラムはimg_name, path, label
    """
    # モデルの読み込み
    model = timm.create_model(
        "vit_base_patch16_224_in21k", pretrained=True, num_classes=2
    )
    model.load_state_dict(torch.load(model_path))
    model_env = get_model_env()
    test_transforms = model_env["test_transforms"]
    # Dataset を作成する。
    test_dataset = ImageFolderForPrediction(img_dir, test_transforms)
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

    df_true = predict_df[predict_df["label"] == 1]
    df_false = predict_df[predict_df["label"] == 0]

    return df_true, df_false
