import re

class MedVisionAgent:
    """
    Zero-cost AI Knowledge Assistant for MedVision-CNN.
    Provides structured technical & dataset answers while maintaining medical safety boundaries.
    """
    def __init__(self):
        self.disclaimer = "⚠️ **Medical Disclaimer**: MedVision-CNN is an educational research prototype. I cannot provide personal medical diagnosis or treatment advice."

    def answer(self, question: str) -> dict:
        q = question.lower().strip()

        # Medical Safety Guardrail
        medical_advice_keywords = ["diagnose me", "my chest hurts", "treatment", "doctor", "medicine", "should i take", "symptoms of pneumonia"]
        if any(kw in q for kw in medical_advice_keywords):
            return {
                "response": f"{self.disclaimer}\n\nIf you or someone you know is experiencing medical symptoms, please consult a qualified physician or healthcare provider immediately.",
                "category": "safety_guardrail"
            }

        # Query 1: Datasets & Links
        if any(kw in q for kw in ["dataset", "data", "download", "link", "source", "mendeley", "zenodo"]):
            return {
                "response": (
                    "📊 **MedVision-CNN Dataset Information**:\n\n"
                    "* **Primary Dataset**: PneumoniaMNIST (MedMNIST v2 benchmark).\n"
                    "* **Source Data**: Kermany et al., *Cell* 2018 (Guangzhou Women & Children's Medical Center).\n"
                    "* **Classes**: 2 categories (`Normal`, `Pneumonia`).\n"
                    "* **Image Count**: 5,856 total radiographs (4,708 Train / 524 Val / 624 Test).\n"
                    "* **Direct Download Links**:\n"
                    "  - 🟢 Zenodo Direct: https://zenodo.org/records/10519652/files/pneumoniamnist.npz\n"
                    "  - 🟢 Mendeley Data: https://data.mendeley.com/datasets/rscbjbr9sj/3"
                ),
                "category": "dataset_info"
            }

        # Query 2: Model Architecture & Training
        if any(kw in q for kw in ["train", "architecture", "cnn", "pytorch", "model", "epochs", "loss"]):
            return {
                "response": (
                    "🧠 **CNN Architecture & Training Specifications**:\n\n"
                    "* **Architecture**: Custom 3-Block ConvNet with BatchNorm & Adaptive Pooling.\n"
                    "* **Framework**: PyTorch 2.x.\n"
                    "* **Optimizer**: Adam (Learning Rate = 0.001, Weight Decay = 1e-4).\n"
                    "* **Loss Function**: CrossEntropyLoss.\n"
                    "* **Target Metrics**: ~94.8% Test Accuracy, 93.2% Sensitivity, 0.952 ROC-AUC."
                ),
                "category": "model_info"
            }

        # Query 3: Grad-CAM / Confidence
        if any(kw in q for kw in ["confidence", "heatmap", "grad-cam", "gradcam", "explain", "meaning"]):
            return {
                "response": (
                    "🔥 **Understanding Confidence & Grad-CAM Heatmaps**:\n\n"
                    "* **Confidence Score**: Represents the output probability of the Softmax layer (0% to 100%).\n"
                    "* **Grad-CAM**: Computes gradients of the target class with respect to the final conv layer (`layer3`).\n"
                    "* **Heatmap Overlay**: Red/yellow areas indicate high filter activations that influenced the model's prediction."
                ),
                "category": "explainability_info"
            }

        # Query 4: Local Execution & Deployment
        if any(kw in q for kw in ["run", "local", "docker", "deploy", "install", "start"]):
            return {
                "response": (
                    "🚀 **How to Run MedVision-CNN**:\n\n"
                    "1. **FastAPI Backend**: `cd backend && uvicorn app.main:app --reload`\n"
                    "2. **Docker Container**: `docker build -t medvision-cnn . && docker run -p 8000:8000 medvision-cnn`\n"
                    "3. **Frontend**: Open `frontend/index.html` in any modern web browser."
                ),
                "category": "deployment_info"
            }

        # Query 5: Limitations
        if any(kw in q for kw in ["limitation", "scope", "accuracy", "clinical"]):
            return {
                "response": (
                    "⚠️ **Model Limitations**:\n\n"
                    "1. **Pediatric Focus**: Trained primarily on pediatric chest X-rays (aged 1–5 years).\n"
                    "2. **Screening Prototype**: Designed solely as an educational demonstration, not a validated medical device."
                ),
                "category": "limitations_info"
            }

        # General Fallback Response
        return {
            "response": (
                "👋 Hello! I am the **MedVision-CNN AI Assistant**.\n\n"
                "You can ask me questions such as:\n"
                "• *'What dataset was used?'*\n"
                "• *'Give me the dataset download link.'*\n"
                "• *'How was the PyTorch CNN trained?'*\n"
                "• *'What does Grad-CAM heatmap mean?'*\n"
                "• *'How do I run this project in Docker?'*"
            ),
            "category": "general_help"
        }
