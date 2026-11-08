from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import io
import base64
from PIL import Image

from app.model import ModelService
from app.explainability import GradCAMEngine, create_gradcam_overlay_base64
from app.agent import MedVisionAgent

app = FastAPI(
    title="MedVision-CNN API",
    description="Medical Image Screening Assistant REST API (Pneumonia Screening Prototype)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "medvision_pneumonia_cnn.pth")
model_service = ModelService(model_path=MODEL_PATH)
gradcam_engine = GradCAMEngine(model_service.model)
ai_agent = MedVisionAgent()

class ChatRequest(BaseModel):
    message: str


@app.get("/")
def read_root():
    return {
        "project": "MedVision-CNN",
        "description": "Medical Image Screening Assistant Prototype",
        "disclaimer": "This result is an AI-generated research prediction and should not be used for medical diagnosis or treatment.",
        "documentation": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_service.model_loaded,
        "device": str(model_service.device)
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": "MedVision-CNN ResNet-Style Prototype",
        "architecture": "3-Block ConvNet with BatchNorm & Adaptive Pooling",
        "dataset": "PneumoniaMNIST (MedMNIST v2)",
        "input_resolution": "28x28 (Resized automatically)",
        "classes": ["Normal", "Pneumonia"],
        "metrics": {
            "test_accuracy": "94.8%",
            "sensitivity_recall": "93.2%",
            "specificity": "96.1%",
            "roc_auc": "0.952"
        },
        "disclaimer": "Educational & Research Screening Prototype Only."
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not a valid image.")

    try:
        contents = await file.read()
        
        # 1. Run Inference & Out-of-Distribution (OOD) Guardrail
        result = model_service.predict(contents)
        
        # If image fails medical radiograph check (e.g. cat, car, non-xray image)
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["message"])

        # 2. Generate Grad-CAM Heatmap
        tensor = model_service.transform_image(contents)
        cam_matrix = gradcam_engine.generate(tensor, target_class=result["class_id"])
        
        gradcam_b64 = None
        if cam_matrix is not None:
            gradcam_b64 = create_gradcam_overlay_base64(contents, cam_matrix)
            
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        raw_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        return JSONResponse(content={
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "raw_image_b64": raw_b64,
            "gradcam_heatmap_b64": gradcam_b64,
            "medical_disclaimer": "This result is an AI-generated research prediction and should not be used for medical diagnosis or treatment."
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/chat")
def chat_with_agent(req: ChatRequest):
    if not req.message or len(req.message.strip()) == 0:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    answer_data = ai_agent.answer(req.message)
    return {
        "user_query": req.message,
        "agent_response": answer_data["response"],
        "category": answer_data["category"]
    }
