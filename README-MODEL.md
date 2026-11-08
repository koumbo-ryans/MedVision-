# 🧠 README — The CNN Model (MedVision-CNN)

> **What is this file about?**
> This README explains the *brain* of the project: the Convolutional Neural Network (CNN).
> It covers how the model is built, how it is trained, how it is saved, and how it turns a
> chest X-ray into "Normal" or "Pneumonia".
>
> The other two READMEs:
> - [`README-BACKEND.md`](README-BACKEND.md) — the FastAPI server and how it *uses* this model
> - [`README-FRONTEND.md`](README-FRONTEND.md) — the web page and how it *talks* to FastAPI

---

## 1. The model in one paragraph (plain language)

A **CNN (Convolutional Neural Network)** is a program that looks at an image through many small
"filters" and learns which patterns matter. Here we do **not** build a CNN from scratch. We take
**ResNet18**, a CNN that has already learned to see shapes and textures from millions of everyday
photos (ImageNet), and we **re-teach its last layer** to answer a medical question:

> *"Does this chest X-ray look Normal, or does it look like Pneumonia?"*

This technique is called **transfer learning** — like hiring an experienced photographer and only
teaching them the one new subject they need. It trains faster and needs less data than starting
from zero.

**Result:** one file, `backend/models/medvision_pneumonia_cnn.pth` (about 45 MB), which contains
the learned weights. The API loads this file and uses it to make predictions.

---

## 2. The big picture

```
 PneumoniaMNIST dataset          train_and_save.py              .pth file
 (thousands of X-rays)  ───────► (PyTorch training)  ───────►  (learned weights)
                                                                      │
                                                                      ▼
                                                          ModelService (model.py)
                                                          loads weights and predicts
```

- **Training happens offline** (your laptop, or Google Colab with a GPU).
- **Prediction happens online**, inside the FastAPI server, using the saved `.pth` file.

---

## 3. Important step 1 — Get the data

The dataset is **PneumoniaMNIST** (part of MedMNIST v2). It is a cleaned collection of
pediatric chest X-rays with two labels: `0 = Normal`, `1 = Pneumonia`.

```python
# backend/train_and_save.py
dataset_file = 'pneumoniamnist_224.npz'
if not os.path.exists(dataset_file):
    url = 'https://zenodo.org/records/10519652/files/pneumoniamnist_224.npz?download=1'
    urllib.request.urlretrieve(url, dataset_file)

data = np.load(dataset_file)
```

**Impact:** the script is self-sufficient — first run downloads the data automatically.
The training split (about 4,700 images) is **imbalanced**: there are more pneumonia images
than normal ones. That imbalance is why Step 3 exists.

---

## 4. Important step 2 — Prepare images the way ResNet18 expects

ResNet18 was trained on **3-channel RGB** images of size **224×224**, normalized with specific
mean/std values. Medical X-rays are **grayscale**, so the code converts them:

```python
# backend/train_and_save.py — inside PneumoniaMNISTDataset
pil_img = Image.fromarray(img_np.astype(np.uint8), mode='L').convert('RGB')
return self.transform(pil_img), self.labels[idx]
```

Training images also get **augmentation** (small random changes), so the model does not memorize
exact pictures:

```python
if is_train:
    self.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize          # ImageNet mean/std
    ])
```

**Impact:** augmentation makes the model more robust to slightly different X-ray machines and
positions. Normalization must be **identical at training and prediction time** — if it differs,
the model receives "strange" numbers and accuracy collapses. That is why `model.py` uses the
exact same mean/std (see Section 8).

---

## 5. Important step 3 — Handle class imbalance

If 75% of the training images are pneumonia, a lazy model could answer "Pneumonia" every time and
still look good. To prevent that, each class gets a **weight** that is larger when the class is rarer:

```python
# backend/train_and_save.py
weight_normal    = total / (2.0 * count_normal)
weight_pneumonia = total / (2.0 * count_pneumonia)
class_weights = torch.tensor([weight_normal, weight_pneumonia], dtype=torch.float32)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

**Impact:** mistakes on the rarer class cost more during training, so the model pays attention to
both classes. This directly improves **sensitivity** (catching pneumonia) and **specificity**
(not over-calling normal X-rays).

---

## 6. Important step 4 — Build ResNet18 and swap the "answer head"

ResNet18 originally outputs **1000** classes (cat, car, coffee cup...). We replace its final
fully-connected layer with a new one that outputs **2** numbers:

```python
# backend/train_and_save.py
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze everything: we do not want to destroy the pretrained "eyes"
for param in model.parameters():
    param.requires_grad = False

# New head: 512 features → 2 classes
model.fc = nn.Linear(model.fc.in_features, 2)
```

**Impact:**
- **Freezing** the backbone means Phase 1 only trains a tiny number of parameters → fast, and it
  cannot "forget" what it already knows.
- **Replacing `fc`** is the only architectural change. The rest of the network still extracts
  edges, curves, and lung-like textures.

---

## 7. Important step 5 — Two-phase training

### Phase 1 — train only the new head (5 epochs)

```python
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
```

Only the new `fc` layer learns. Learning rate `0.001` is relatively large — safe here because
only a few parameters change.

### Phase 2 — gently fine-tune the last conv block (5 epochs)

```python
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer2 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4            # 10× smaller: small steps, do not damage pretrained features
)
```

`layer4` is ResNet18's **last block of convolution filters** — the part that understands
high-level shapes. Letting it adapt slightly to X-rays usually gives the biggest accuracy jump.

**Impact of the two phases:** Phase 1 gets the model "roughly right" quickly. Phase 2 refines the
most important filters with a small learning rate, which is the classic recipe for transfer
learning. The lower learning rate in Phase 2 is deliberate: big updates would erase useful
pretrained knowledge.

### Saving

```python
if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(model.state_dict(), save_path)
```

**Impact:** only the **best** model (highest validation accuracy) is kept — not necessarily the
last one. So the `.pth` file you ship is the best checkpoint seen during training.

---

## 8. Important step 6 — What the model does at prediction time

This part lives in `backend/app/model.py` and runs **inside the API** for every upload.

### 8a. Load the weights once

```python
self.model = models.resnet18(weights=None)      # architecture, no pretrained weights
self.model.fc = nn.Linear(self.model.fc.in_features, 2)
self.model = self.model.to(self.device)          # GPU if available, else CPU
...
state_dict = torch.load(model_path, map_location=self.device)
self.model.load_state_dict(state_dict)
self.model.eval()
```

**Impact:**
- `weights=None` means "build the empty skeleton". The knowledge comes from our `.pth` file.
- `eval()` switches off training-only behaviour (like dropout). Forgetting `eval()` makes
  predictions unreliable.
- `map_location` lets a model trained on a GPU load on a CPU-only machine.

### 8b. Build the input tensor

```python
image = Image.open(io.BytesIO(image_bytes)).convert("L")   # force grayscale

# center-crop to a square, then resize to 224×224, then normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                         # 0–255 → 0.0–1.0, shape (3,224,224)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
tensor = transform(image).unsqueeze(0).to(self.device)  # add batch dim → (1,3,224,224)
```

**Impact:** this must mirror training exactly. `unsqueeze(0)` adds the "batch" dimension because
PyTorch models always expect a batch, even if it contains one image. The output shape
`(1, 3, 224, 224)` is the contract the network expects.

### 8c. Reject obviously non-medical images (OOD guardrail)

```python
# 1) colour check: X-rays are nearly grayscale
if r_g_diff > 18 or g_b_diff > 18:
    return False, "Image contains high color saturation..."

# 2) contrast check: blank/solid images have almost no variation
std_dev = np.std(np.array(gray_img))
if std_dev < 15.0:
    return False, "Low-contrast or blank image detected..."
```

**Impact:** this is a cheap safety net. Without it, a photo of a cat or a blank file would still
receive a confident "Normal/Pneumonia" answer. Note that it only catches *obvious* cases — it is
not a real out-of-distribution model.

### 8d. Run the network

```python
with torch.no_grad():
    outputs = self.model(tensor)                    # raw scores ("logits"), 2 numbers
    probs = torch.softmax(outputs, dim=1)[0]        # convert to probabilities (sum = 1)
    pred_class_idx = torch.argmax(probs).item()     # 0 or 1
    confidence = probs[pred_class_idx].item()       # 0.0 – 1.0
```

**Impact:**
- `no_grad()` means "we are not training", so PyTorch skips gradient bookkeeping → faster and
  lighter on memory.
- **Softmax** turns two arbitrary numbers into two percentages that add to 100%.
- `argmax` picks the winner; `confidence` is how sure the model is about the winner.

The returned dictionary is what the backend passes on:

```python
{
  "error": False,
  "prediction": "Pneumonia",      # human-readable
  "class_id": 1,                  # needed by Grad-CAM
  "confidence": 93.4,             # percent
  "probabilities": {"Normal": 6.6, "Pneumonia": 93.4}
}
```

**Impact:** `class_id` is important later — Grad-CAM needs to know *which* class to explain.

---

## 9. Where the model code lives

| File | Role |
|---|---|
| `backend/train_and_save.py` | Full training script (download → train → save `.pth`) |
| `notebooks/medvision_colab_training.py` | Same training flow with plots + test evaluation, for Google Colab GPU |
| `backend/app/model.py` | `ModelService`: prediction used by the API |
| `backend/models/medvision_pneumonia_cnn.pth` | The saved trained weights |
| `backend/pneumoniamnist.npz` | Local copy of the dataset (used by `extract_test_samples.py`) |
| `backend/extract_test_samples.py` | Saves one real Normal and one real Pneumonia X-ray into `frontend/` |

### How the three training files relate

- `notebooks/medvision_colab_training.py` adds **exploratory plots**, **test-set evaluation**
  (accuracy, sensitivity, specificity, ROC-AUC, confusion matrix) and a **download** step.
- `backend/train_and_save.py` is the **headless** version: it trains and writes the file directly
  to `backend/models/`.
- Both produce the **same architecture**, so a `.pth` from either one loads in the API.

---

## 10. Report metrics (as shown by the API)

The `/model-info` endpoint reports:

| Metric | Value |
|---|---|
| Test accuracy | 94.8% |
| Sensitivity / Recall | 93.2% |
| Specificity | 96.1% |
| ROC-AUC | 0.952 |

**Impact:** sensitivity = how often real pneumonia is caught; specificity = how often a normal
X-ray is correctly called normal. In screening, sensitivity is usually the priority, because a
missed case is worse than a false alarm.

---

## 11. Known inconsistencies (important if you modify the project)

These are real mismatches between text and code. Kept here so nobody is misled:

1. **`/model-info` describes a different model.** It says *"3-Block ConvNet ... 28×28 input"*,
   but the actual code is **ResNet18 at 224×224**. The metrics are hardcoded strings.
2. **The chat agent (`agent.py`) says Grad-CAM uses `layer3`.** The code in
   `explainability.py` and `model.py` uses **`layer4`**. `layer4` is the correct choice for
   ResNet18 (it is the last conv block).
3. **The "Sample Normal / Sample Pneumonia" buttons in the frontend draw synthetic shapes**, not
   real X-rays, even though `frontend/real_*_xray.png` files exist. See `README-FRONTEND.md`.

**Impact if ignored:** someone retraining a custom 28×28 CNN would not be able to load it into
this API, because `model.py` hardcodes ResNet18.

---

## 12. Quick commands

Train locally (needs internet for the first run):

```powershell
cd backend
python train_and_save.py
```

Train on GPU (recommended): open `notebooks/medvision_colab_training.py` content in Google Colab,
set *Runtime → GPU*, run all cells, then place the downloaded `.pth` in `backend/models/`.

Extract two real sample images for the frontend (dataset already present):

```powershell
cd backend
python extract_test_samples.py
```

---

## 13. One-sentence summary

> We take a pretrained ResNet18, freeze its knowledge, teach its last layer (then its last block)
> to separate Normal from Pneumonia X-rays, save the best weights, and at prediction time we
> preprocess the upload exactly like training data, run one forward pass, and read the softmax
> probabilities.
