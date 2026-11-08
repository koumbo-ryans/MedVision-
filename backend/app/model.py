import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageStat
import io
import os
import numpy as np


class ModelService:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = {0: "Normal", 1: "Pneumonia"}
        self.model_loaded = False

        # Build ResNet18 — replace final FC layer for 2-class classification
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        # ResNet18 expects 3-channel RGB input — we handle grayscale → RGB conversion in transform
        self.model = self.model.to(self.device)

        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)

    def load_weights(self, model_path: str):
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model_loaded = True
        print(f"[MedVision] ResNet18 weights loaded from {model_path}")

    def get_target_layer(self):
        """Return the final conv layer for Grad-CAM targeting."""
        return self.model.layer4

    def validate_radiograph_input(self, image_bytes: bytes) -> tuple:
        """
        Out-of-Distribution (OOD) Guardrail:
        Rejects obviously non-medical images (colored photos, blank images).
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))

            # Color Saturation Check — X-rays are near-grayscale
            if img.mode in ("RGB", "RGBA"):
                rgb_img = img.convert("RGB")
                stat = ImageStat.Stat(rgb_img)
                r_g_diff = abs(stat.mean[0] - stat.mean[1])
                g_b_diff = abs(stat.mean[1] - stat.mean[2])
                if r_g_diff > 18 or g_b_diff > 18:
                    return False, "Image contains high color saturation. Please upload a grayscale chest radiograph."

            # Texture & Contrast Check — rejects blank / solid images
            gray_img = img.convert("L")
            np_arr = np.array(gray_img)
            std_dev = np.std(np_arr)
            if std_dev < 15.0:
                return False, "Low-contrast or blank image detected. Please upload a valid chest X-ray radiograph."

            return True, "Valid radiograph image"
        except Exception as e:
            return False, f"Invalid image format: {str(e)}"

    def transform_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        Converts uploaded image bytes to a normalized (1, 3, 224, 224) tensor
        compatible with ResNet18 (ImageNet normalization).
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("L")

        # Center crop to square before resizing
        w, h = image.size
        min_dim = min(w, h)
        left = (w - min_dim) / 2
        top = (h - min_dim) / 2
        image = image.crop((left, top, left + min_dim, top + min_dim))

        # Convert grayscale to RGB (ResNet18 requires 3 channels)
        image = image.convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet mean
                std=[0.229, 0.224, 0.225]      # ImageNet std
            )
        ])
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image_bytes: bytes) -> dict:
        # 1. OOD Guardrail
        is_valid, reason = self.validate_radiograph_input(image_bytes)
        if not is_valid:
            return {"error": True, "message": reason}

        # 2. Inference
        tensor = self.transform_image(image_bytes)
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_class_idx = torch.argmax(probs).item()
            confidence = probs[pred_class_idx].item()

        return {
            "error": False,
            "prediction": self.class_names[pred_class_idx],
            "class_id": pred_class_idx,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "Normal": round(probs[0].item() * 100, 2),
                "Pneumonia": round(probs[1].item() * 100, 2)
            }
        }
