import numpy as np
from PIL import Image
import os

# Load dataset
data = np.load('pneumoniamnist.npz')
test_images = data['test_images']
test_labels = data['test_labels'].squeeze()

# Find first Normal (0) and first Pneumonia (1) index
idx_normal = np.where(test_labels == 0)[0][0]
idx_pneumonia = np.where(test_labels == 1)[0][0]

# Save real dataset images to frontend directory
out_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
os.makedirs(out_dir, exist_ok=True)

img_norm = Image.fromarray(test_images[idx_normal], mode='L').resize((224, 224))
img_pneu = Image.fromarray(test_images[idx_pneumonia], mode='L').resize((224, 224))

img_norm.save(os.path.join(out_dir, 'real_normal_xray.png'))
img_pneu.save(os.path.join(out_dir, 'real_pneumonia_xray.png'))

print("Extracted real test dataset X-rays:")
print(f" - Saved 'real_normal_xray.png' (Index {idx_normal})")
print(f" - Saved 'real_pneumonia_xray.png' (Index {idx_pneumonia})")
