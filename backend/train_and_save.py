import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import urllib.request

# ─────────────────────────────────────────────
# 1. Download PneumoniaMNIST 224×224 if missing
# ─────────────────────────────────────────────
dataset_file = 'pneumoniamnist_224.npz'
if not os.path.exists(dataset_file):
    print("Downloading pneumoniamnist_224.npz (224x224 version — ~35 MB)...")
    url = 'https://zenodo.org/records/10519652/files/pneumoniamnist_224.npz?download=1'
    urllib.request.urlretrieve(url, dataset_file)
    print("Download complete.\n")

data = np.load(dataset_file)
train_labels_np = data['train_labels'].squeeze()

# ─────────────────────────────────────────────
# 2. Class Imbalance Weights
# ─────────────────────────────────────────────
count_normal    = int(np.sum(train_labels_np == 0))
count_pneumonia = int(np.sum(train_labels_np == 1))
total           = len(train_labels_np)
weight_normal    = total / (2.0 * count_normal)
weight_pneumonia = total / (2.0 * count_pneumonia)
class_weights = torch.tensor([weight_normal, weight_pneumonia], dtype=torch.float32)

print(f"Train → Normal: {count_normal}, Pneumonia: {count_pneumonia}")
print(f"Class Weights  → Normal: {weight_normal:.3f}, Pneumonia: {weight_pneumonia:.3f}\n")

# ─────────────────────────────────────────────
# 3. PyTorch Dataset (224×224 with Augmentation)
# ─────────────────────────────────────────────
class PneumoniaMNISTDataset(Dataset):
    def __init__(self, images, labels, is_train=True):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long).squeeze()

        # ImageNet normalization (ResNet18 requirement)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_np = self.images[idx]
        # Dataset may be (H,W) or (H,W,1) — convert to PIL grayscale → RGB
        if img_np.ndim == 3:
            img_np = img_np.squeeze(-1)
        pil_img = Image.fromarray(img_np.astype(np.uint8), mode='L').convert('RGB')
        return self.transform(pil_img), self.labels[idx]

train_dataset = PneumoniaMNISTDataset(data['train_images'], data['train_labels'], is_train=True)
val_dataset   = PneumoniaMNISTDataset(data['val_images'],   data['val_labels'],   is_train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
print(f"DataLoaders: {len(train_loader)} train batches, {len(val_loader)} val batches\n")

# ─────────────────────────────────────────────
# 4. ResNet18 Transfer Learning Setup
# ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training device: {device}\n")

# Load pretrained ResNet18 (ImageNet weights)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers except the final fully-connected layer
for param in model.parameters():
    param.requires_grad = False

# Replace the final classification layer for 2-class problem
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

print("ResNet18 Transfer Learning: All conv layers FROZEN, only FC layer trainable.\n")

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ─────────────────────────────────────────────
# 5. Phase 1 — Train FC Layer Only (5 Epochs)
# ─────────────────────────────────────────────
EPOCHS_PHASE1 = 5
best_val_acc = 0.0

print("=" * 60)
print("Phase 1: Training FC layer only (Frozen Backbone)")
print("=" * 60)

for epoch in range(1, EPOCHS_PHASE1 + 1):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        train_correct += (out.argmax(1) == lbls).sum().item()
        train_total += lbls.size(0)

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            val_correct += (out.argmax(1) == lbls).sum().item()
            val_total += lbls.size(0)

    val_acc = (val_correct / val_total) * 100.0
    train_acc = (train_correct / train_total) * 100.0
    scheduler.step()

    print(f"Epoch [{epoch:02d}/{EPOCHS_PHASE1}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc

# ─────────────────────────────────────────────
# 6. Phase 2 — Unfreeze layer4 for fine-tuning
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 2: Unfreezing ResNet18 layer4 for fine-tuning")
print("=" * 60)

for param in model.layer4.parameters():
    param.requires_grad = True

optimizer2 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

EPOCHS_PHASE2 = 5
for epoch in range(1, EPOCHS_PHASE2 + 1):
    model.train()
    train_correct, train_total = 0, 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer2.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer2.step()
        train_correct += (out.argmax(1) == lbls).sum().item()
        train_total += lbls.size(0)

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            val_correct += (out.argmax(1) == lbls).sum().item()
            val_total += lbls.size(0)

    val_acc = (val_correct / val_total) * 100.0
    train_acc = (train_correct / train_total) * 100.0
    print(f"Epoch [{epoch:02d}/{EPOCHS_PHASE2}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(models_dir, exist_ok=True)
        save_path = os.path.join(models_dir, 'medvision_pneumonia_cnn.pth')
        torch.save(model.state_dict(), save_path)

print(f"\n{'=' * 60}")
print(f"Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Model saved to backend/models/medvision_pneumonia_cnn.pth")
print(f"{'=' * 60}")
