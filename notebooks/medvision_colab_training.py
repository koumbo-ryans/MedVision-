# ============================================================
# MedVision-CNN — Google Colab Training Notebook
# ResNet18 Transfer Learning on PneumoniaMNIST 224x224
# ============================================================
# INSTRUCTIONS:
# 1. Upload this file to Google Colab (File → Upload notebook)
# 2. Set Runtime → Change runtime type → GPU (T4)
# 3. Run all cells in order (Runtime → Run all)
# 4. Download medvision_pneumonia_cnn.pth at the end
# 5. Place the .pth file in: MedVision-CNN/backend/models/
# ============================================================

# ─── Cell 1: Install Dependencies ───────────────────────────
# (torch and torchvision come pre-installed on Colab)

# ─── Cell 2: Download Dataset ───────────────────────────────
import urllib.request
import os

dataset_file = 'pneumoniamnist_224.npz'
if not os.path.exists(dataset_file):
    print("Downloading PneumoniaMNIST 224x224 (~35 MB)...")
    url = 'https://zenodo.org/records/10519652/files/pneumoniamnist_224.npz?download=1'
    urllib.request.urlretrieve(url, dataset_file)
    print("Download complete!")
else:
    print(f"Dataset already present: {dataset_file}")

import numpy as np
data = np.load(dataset_file)
print("Keys:", list(data.keys()))
print(f"Train images shape: {data['train_images'].shape}")
print(f"Val images shape:   {data['val_images'].shape}")
print(f"Test images shape:  {data['test_images'].shape}")

# ─── Cell 3: Exploratory Data Analysis ──────────────────────
import matplotlib.pyplot as plt

class_names = {0: "Normal", 1: "Pneumonia"}
train_labels = data['train_labels'].squeeze()
normal_count    = int(np.sum(train_labels == 0))
pneumonia_count = int(np.sum(train_labels == 1))

print(f"\nClass Distribution (Training Set):")
print(f"  Normal:    {normal_count}  ({normal_count / len(train_labels):.1%})")
print(f"  Pneumonia: {pneumonia_count} ({pneumonia_count / len(train_labels):.1%})")

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("PneumoniaMNIST 224x224 — Sample Images", fontsize=14, fontweight='bold')

for cls_idx, cls_name in class_names.items():
    indices = np.where(train_labels == cls_idx)[0][:5]
    for col, sample_idx in enumerate(indices):
        ax = axes[cls_idx, col]
        ax.imshow(data['train_images'][sample_idx].squeeze(), cmap='gray')
        ax.set_title(cls_name, fontsize=9)
        ax.axis('off')

plt.tight_layout()
plt.show()

# ─── Cell 4: Build Dataset & DataLoaders ────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

class PneumoniaMNISTDataset(Dataset):
    def __init__(self, images, labels, is_train=True):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long).squeeze()
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
        if img_np.ndim == 3:
            img_np = img_np.squeeze(-1)
        pil_img = Image.fromarray(img_np.astype(np.uint8), mode='L').convert('RGB')
        return self.transform(pil_img), self.labels[idx]

train_dataset = PneumoniaMNISTDataset(data['train_images'], data['train_labels'], is_train=True)
val_dataset   = PneumoniaMNISTDataset(data['val_images'],   data['val_labels'],   is_train=False)
test_dataset  = PneumoniaMNISTDataset(data['test_images'],  data['test_labels'],  is_train=False)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
print(f"Batches → Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

# ─── Cell 5: Build ResNet18 Transfer Learning Model ─────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Compute class weights
total = len(train_dataset)
weight_normal    = total / (2.0 * normal_count)
weight_pneumonia = total / (2.0 * pneumonia_count)
class_weights = torch.tensor([weight_normal, weight_pneumonia], dtype=torch.float32).to(device)
print(f"Class weights → Normal: {weight_normal:.3f}, Pneumonia: {weight_pneumonia:.3f}")

# Load pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all backbone layers
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer for binary classification
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
print("ResNet18 loaded: backbone FROZEN, FC layer trainable.")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# ─── Cell 6: Phase 1 Training (FC Only, 5 Epochs) ───────────
train_accs, val_accs = [], []
best_val_acc = 0.0
EPOCHS_P1 = 5

print("=" * 55)
print("Phase 1: Training FC Layer only")
print("=" * 55)

for epoch in range(1, EPOCHS_P1 + 1):
    model.train()
    correct, total_b = 0, 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        correct += (out.argmax(1) == lbls).sum().item()
        total_b += lbls.size(0)
    train_acc = correct / total_b * 100

    model.eval()
    vc, vt = 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            vc += (out.argmax(1) == lbls).sum().item()
            vt += lbls.size(0)
    val_acc = vc / vt * 100
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    print(f"Epoch [{epoch}/{EPOCHS_P1}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'medvision_pneumonia_cnn.pth')

# ─── Cell 7: Phase 2 Fine-tuning (Unfreeze layer4, 5 Epochs) ──
print("\n" + "=" * 55)
print("Phase 2: Unfreezing layer4 for fine-tuning")
print("=" * 55)

for param in model.layer4.parameters():
    param.requires_grad = True

optimizer2 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
EPOCHS_P2 = 5

for epoch in range(1, EPOCHS_P2 + 1):
    model.train()
    correct, total_b = 0, 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer2.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer2.step()
        correct += (out.argmax(1) == lbls).sum().item()
        total_b += lbls.size(0)
    train_acc = correct / total_b * 100

    model.eval()
    vc, vt = 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            vc += (out.argmax(1) == lbls).sum().item()
            vt += lbls.size(0)
    val_acc = vc / vt * 100
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    print(f"Epoch [{epoch}/{EPOCHS_P2}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'medvision_pneumonia_cnn.pth')

print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")

# ─── Cell 8: Plot Training Curves ───────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(train_accs, label='Train Accuracy', color='#4facfe', linewidth=2)
plt.plot(val_accs,   label='Val Accuracy',   color='#00f2fe', linewidth=2, linestyle='--')
plt.axvline(x=4.5, color='gray', linestyle=':', label='Phase 2 starts')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('MedVision-CNN ResNet18 Training Curves')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ─── Cell 9: Full Test Evaluation ───────────────────────────
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    classification_report
)

model.load_state_dict(torch.load('medvision_pneumonia_cnn.pth', map_location=device))
model.eval()

all_labels, all_preds, all_probs = [], [], []
softmax = nn.Softmax(dim=1)

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out = model(imgs)
        probs = softmax(out)
        preds = out.argmax(1)
        all_labels.extend(lbls.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)

tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
sensitivity = tp / (tp + fn) * 100
specificity = tn / (tn + fp) * 100
precision   = tp / (tp + fp) * 100
accuracy    = (tp + tn) / (tp + tn + fp + fn) * 100
auc         = roc_auc_score(all_labels, all_probs)

print("=" * 55)
print("  MedVision-CNN Clinical Evaluation (Test Set)")
print("=" * 55)
print(f"  Accuracy:           {accuracy:.2f}%")
print(f"  Sensitivity/Recall: {sensitivity:.2f}%  ← Pneumonia Detection Rate")
print(f"  Specificity:        {specificity:.2f}%  ← Normal Detection Rate")
print(f"  Precision:          {precision:.2f}%")
print(f"  ROC-AUC:            {auc:.4f}")
print(f"  TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")
print("=" * 55)

# ─── Cell 10: Download Trained Weights ──────────────────────
from google.colab import files
print("Downloading medvision_pneumonia_cnn.pth...")
files.download('medvision_pneumonia_cnn.pth')
print("Done! Place this file in: MedVision-CNN/backend/models/")
