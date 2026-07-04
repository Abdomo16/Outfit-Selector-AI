"""
training/train_classifier.py

Training script for the clothing type classifier (ResNet18).
Dataset: Clothing Dataset Full — run datasets/clothing_dataset_full.py first.

Usage:
    python -m training.train_classifier [--epochs 20] [--lr 1e-4] [--batch 32]

Output:
    weights/classifier.pt  — saved at best validation accuracy
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATASET_DIR = Path("datasets/clothing_dataset_full")
WEIGHTS_DIR = Path("weights")
WEIGHTS_PATH = WEIGHTS_DIR / "classifier.pt"

CLASS_NAMES = [
    "Blazer", "Blouse", "Body", "Dress", "Hat",
    "Hoodie", "Longsleeve", "Outwear", "Pants", "Polo",
    "Pyjama", "Shirt", "Shoes", "Shorts", "Skip",
    "Skirt", "Socks", "Suit", "T-Shirt", "Undershirt"
]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(n_classes: int, device: str) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    WEIGHTS_DIR.mkdir(exist_ok=True)

    train_ds = datasets.ImageFolder(DATASET_DIR / "train", transform=TRAIN_TRANSFORM)
    val_ds   = datasets.ImageFolder(DATASET_DIR / "val",   transform=VAL_TRANSFORM)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4)

    model = build_model(len(train_ds.classes), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        # --- Validate ---
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_dl:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_acc = correct / len(val_ds)
        avg_loss = running_loss / len(train_ds)
        print(f"Epoch {epoch:02d}/{args.epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  ✓ Saved best model (acc={best_acc:.4f}) → {WEIGHTS_PATH}")

        scheduler.step()

    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")
    print(f"Weights saved to: {WEIGHTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train clothing classifier")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr",     type=float, default=1e-4)
    parser.add_argument("--batch",  type=int, default=32)
    train(parser.parse_args())
