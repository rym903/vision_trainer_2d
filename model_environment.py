from torchvision import transforms
from pathlib import Path


def get_model_env():
    """モデルの環境などをまとめたdictを返す

    Returns:
        _type_: _description_
    """
    model_base_name = "vit_base_patch16_224_in21k"
    epochs = 50
    lr = 3e-5
    gamma = 0.7
    device = "cuda"
    train_dataset_dir = Path("./data/train")
    val_dataset_dir = Path("./data/valid")
    test_dataset_dir = Path("./data/test")
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

    model_env = {
        "model_base_name": model_base_name,
        "epochs": epochs,
        "lr": lr,
        "gamma": gamma,
        "device": device,
        "train_dataset_dir": train_dataset_dir,
        "val_dataset_dir": val_dataset_dir,
        "test_dataset_dir": test_dataset_dir,
        "train_transforms": train_transforms,
        "val_transforms": val_transforms,
        "test_transforms": val_transforms,
    }

    return model_env
