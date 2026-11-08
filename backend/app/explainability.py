import torch
import numpy as np
import io
import base64
from PIL import Image
import matplotlib.cm as cm


class GradCAMEngine:
    def __init__(self, model):
        self.model = model
        # ResNet18 target layer: final conv block (layer4)
        self.target_layer = model.layer4
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, target_class: int = None):
        self.model.eval()
        tensor = tensor.clone().requires_grad_(True)
        output = self.model(tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        if self.gradients is None or self.activations is None:
            return None

        weights = self.gradients[0].cpu().numpy().mean(axis=(1, 2))
        activations = self.activations[0].cpu().numpy()

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        return cam


def create_gradcam_overlay_base64(raw_image_bytes: bytes, cam_matrix: np.ndarray) -> str:
    raw_img = Image.open(io.BytesIO(raw_image_bytes)).convert("RGB").resize((224, 224))
    raw_np = np.array(raw_img) / 255.0

    cam_tensor = torch.tensor(cam_matrix).unsqueeze(0).unsqueeze(0)
    cam_resized = torch.nn.functional.interpolate(
        cam_tensor, size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze().numpy()

    cmap = cm.get_cmap('jet')
    heatmap_rgb = cmap(cam_resized)[:, :, :3]

    overlay = np.clip(0.6 * raw_np + 0.4 * heatmap_rgb, 0, 1)
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

    buffered = io.BytesIO()
    overlay_img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
