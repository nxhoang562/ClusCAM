#!/usr/bin/env python3
import argparse
import os
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchvision import models

# Mapping numeric labels to class names
LABEL_MAPPING = {
    0: "Mild_Demented",
    1: "Moderate_Demented",
    2: "Non_Demented",
    3: "Very_Mild_Demented"
}


def dict_to_image(image_dict: dict) -> np.ndarray:
    """
    Convert a dictionary with JPEG bytes to a grayscale image array.
    """
    byte_string = image_dict.get('bytes')
    if byte_string is None:
        raise TypeError(f"Expected dict with 'bytes', got {type(image_dict)}")
    arr = np.frombuffer(byte_string, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


class MRIAlzheimerDataset(Dataset):
    """
    PyTorch Dataset for Alzheimer MRI images stored in a DataFrame.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[callable] = None
    ):
        self.images = df['img_arr'].values
        self.labels = df['label'].values
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess training and test DataFrames from Parquet files.
    """
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    # Convert raw bytes -> image arrays
    df_train['img_arr'] = df_train['image'].apply(dict_to_image)
    df_test['img_arr'] = df_test['image'].apply(dict_to_image)
    df_train.drop(columns=['image'], inplace=True)
    df_test.drop(columns=['image'], inplace=True)

    # Map labels to human-readable class names
    df_train['class_name'] = df_train['label'].map(LABEL_MAPPING)
    df_test['class_name'] = df_test['label'].map(LABEL_MAPPING)

    return df_train, df_test


def save_processed_images(df: pd.DataFrame, output_dir: str) -> None:
    """
    Lưu lại tất cả ảnh trong df['img_arr'] thành file PNG trong thư mục output_dir.
    File sẽ đặt tên theo chỉ số và class_name, ví dụ '000123_Non_Demented.png'.
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, row in df.iterrows():
        img = row['img_arr']  # numpy.ndarray uint8 (grayscale)
        class_name = row['class_name']
        filename = f"{idx:06d}_{class_name}.png"
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, img)


def prepare_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32,
    val_split: float = 0.2,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Split DataFrame into train/val sets and return PyTorch DataLoaders.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df['class_name'],
        random_state=random_state
    )

    # Normalize images to [0,1] and convert to torch.Tensor
    def to_tensor(x):
        x = x.astype(np.float32) / 255.0
        tensor = torch.tensor(x).unsqueeze(0)  # (1, H, W)
        return tensor

    train_ds = MRIAlzheimerDataset(train_df, transform=to_tensor)
    val_ds = MRIAlzheimerDataset(val_df, transform=to_tensor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def build_model(num_classes: int = 4) -> nn.Module:
    """
    Create a ResNet18-based model adapted for single-channel input.
    """
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 50,
    patience: int = 7,
    checkpoint_path: Optional[str] = None
) -> nn.Module:
    """
    Train the model with early stopping on validation loss,
    and optionally save the best model state to `checkpoint_path`.
    """
    best_loss = float('inf')
    best_state = None
    counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        avg_val = val_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch}/{num_epochs} - Train: {avg_train:.4f}, Val: {avg_val:.4f}, Acc: {acc:.4f}")

        # Check for improvement
        if avg_val < best_loss:
            best_loss = avg_val
            best_state = model.state_dict()
            counter = 0
            if checkpoint_path:
                torch.save(best_state, checkpoint_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping... loading best model state.")
                model.load_state_dict(best_state)
                break
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    """
    Compute and display accuracy on the given DataLoader.
    """
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds.extend(outputs.argmax(dim=1).cpu().tolist())
            labels.extend(labs.tolist())
    acc = accuracy_score(labels, preds)
    print(f"Evaluation Accuracy: {acc:.4f}")


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load a ResNet18-based model state from file.
    """
    model = build_model(num_classes=len(LABEL_MAPPING))
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Alzheimer MRI Classification Training")
    parser.add_argument("--train_path", type=str, required=True, help="Path to train parquet file")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test parquet file")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--checkpoint", type=str, default="alzheimer_resnet18.pth",
                        help="File to save best model")
    parser.add_argument("--save_test_dir", type=str, default="processed_test_images",
                        help="Thư mục để lưu các ảnh test đã xử lý")
    args = parser.parse_args()

    # Load and preprocess data
    df_train, df_test = load_data(args.train_path, args.test_path)
    # Lưu ảnh đã tiền xử lý của tập test
    save_processed_images(df_test, args.save_test_dir)

    # Chuẩn bị DataLoader cho train/val
    train_loader, val_loader = prepare_dataloaders(df_train, batch_size=args.batch_size)

    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Xây dựng model, loss, optimizer
    model = build_model(num_classes=len(LABEL_MAPPING)).to(device)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df_train['label']),
        y=df_train['label']
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Huấn luyện và lưu checkpoint
    trained_model = train_loop(
        model, train_loader, val_loader,
        criterion, optimizer, device,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=args.checkpoint
    )
    print("Final evaluation on validation set:")
    evaluate(trained_model, val_loader, device)

    # Ví dụ load model để dùng cho XAI
    print(f"Loading model for XAI from {args.checkpoint}...")
    


if __name__ == "__main__":
    main()
