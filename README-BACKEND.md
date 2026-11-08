# 🚀 README — The FastAPI Backend & How It Connects to the Model

> **What is this file about?**
> This README explains the server: what FastAPI does, what each endpoint is for, and — most
> importantly — **exactly how FastAPI talks to the CNN model**.
>
> Related READMEs:
> - [`README-MODEL.md`](README-MODEL.md) — how the CNN itself works
> - [`README-FRONTEND.md`](README-FRONTEND.md) — how the web page talks to this server

---

## 1. What is FastAPI and why is it here?

A trained CNN is just a file of numbers. It cannot receive uploads, validate file types, or send
JSON over the internet. **FastAPI is the "waiter"** between the outside world and the model:

```
Browser ──► FastAPI (receive image, validate, call model) ──► CNN ──► FastAPI ──► Browser
```

It is also why the frontend never touches PyTorch. The browser sends an image; the server does
the heavy lifting and returns a small JSON answer.

**FastAPI was chosen because** it is fast, has almost no boilerplate, and automatically creates
interactive documentation at `/docs`.

---

## 2. Backend folder map

```
backend/
├─ app/
│  ├─ main.py             ← the API itself (routes + startup)
│  ├─ model.py            ← ModelService: loads .pth and predicts
│  ├─ explainability.py   ← Grad-CAM heatmap generator
│  └─ agent.py            ← rule-based chat assistant (/chat)
├─ models/
│  └─ medvision_pneumonia_cnn.pth   ← trained CNN weights (~45 MB)
├─ train_and_save.py      ← offline training (not used by the server)
├─ extract_test_samples.py← offline utility for demo images
├─ pneumoniamnist.npz     ← dataset copy (offline utility only)
└─ requirements.txt       ← Python dependencies
```

`requirements.txt`:

```text
fastapi>=0.100.0
uvicorn>=0.22.0
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.5.0
numpy>=1.24.0
matplotlib>=3.7.0
python-multipart>=0.0.6      ← REQUIRED for file uploads
```

**Impact:** `python-multipart` is easy to forget. Without it, `UploadFile` endpoints fail at
startup with an error about form data.

---

## 3. Important step 1 — Startup: create the server and load the model ONCE

```python
# backend/app/main.py
app = FastAPI(
    title="MedVision-CNN API",
    description="Medical Image Screening Assistant REST API",
    version="1.0.0"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "medvision_pneumonia_cnn.pth")
model_service = ModelService(model_path=MODEL_PATH)   # ← loads the .pth here
gradcam_engine = GradCAMEngine(model_service.model)   # ← same model object
ai_agent = MedVisionAgent()
```

**Impact (very important):**

- These lines run **once, when the server starts** (when Python imports `main.py`).
- The 45 MB `.pth` is therefore loaded **a single time**, not on every request. Loading it per
  request would make each prediction take seconds longer.
- All requests share the **same** `model_service` and the **same** model object. That is why the
  server must be started in a way where the import succeeds (see Section 10).
- `GradCAMEngine` receives **the same model instance** (`model_service.model`). This is the key
  link between the API and the model — one object, used for both prediction and explanation.

The path `../models/...` is relative to `backend/app/`, so the final file is
`backend/models/medvision_pneumonia_cnn.pth`.

If the file is missing, `ModelService` still starts, but `model_loaded` stays `False` and
predictions are meaningless. `/health` exposes that state.

---

## 4. Important step 2 — Allow the browser to call the API (CORS)

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Plain-language meaning:** browsers block a web page from calling a *different* address unless
the server explicitly allows it. The frontend is opened as a local file (or on another port), so
without this middleware every `fetch` would fail with a CORS error — even though the API works
fine in `/docs`.

**Impact:** this is what makes `frontend/script.js` able to POST to `http://localhost:8000`.
`allow_origins=["*"]` is fine for a local prototype but must be restricted in production.

---

## 5. Endpoints at a glance

| Method | Path | Purpose | Returns |
|---|---|---|---|
| GET | `/` | Project info + disclaimer | JSON |
| GET | `/health` | Is the server alive? Is the model loaded? | JSON |
| GET | `/model-info` | Static description + reported metrics | JSON |
| POST | `/predict` | Upload an X-ray, get prediction + Grad-CAM | JSON |
| POST | `/chat` | Ask the rule-based assistant a question | JSON |
| GET | `/docs` | Auto-generated interactive API docs | HTML |

**Impact:** `/docs` (Swagger UI) lets you test uploads from the browser without the frontend.
`/health` is the fastest way to diagnose "is the model actually loaded?".

```python
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_service.model_loaded,
        "device": str(model_service.device)      # "cpu" or "cuda"
    }
```

---

## 6. ⭐ The main endpoint: `POST /predict` — step by step

This is where FastAPI and the CNN meet. Read it as a pipeline.

```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
```

`UploadFile = File(...)` tells FastAPI: *"this endpoint expects a multipart file upload named
`file`."* FastAPI parses the request body for you.

### Step 6.1 — Cheap validation before touching the model

```python
if not file.content_type.startswith("image/"):
    raise HTTPException(status_code=400, detail="File uploaded is not a valid image.")
```

**Impact:** rejects a PDF/zip immediately, before spending CPU/GPU time. The browser receives
HTTP 400 and shows an alert.

### Step 6.2 — Read the bytes (the hand-off point to the model)

```python
contents = await file.read()

# 1. Run inference & out-of-distribution guardrail
result = model_service.predict(contents)
```

**This single line is the main connection between FastAPI and the CNN.** `contents` is a raw
`bytes` object. `ModelService.predict()` (in `model.py`) does the rest:
validate → preprocess → forward pass → softmax → return a dict.

> Why bytes and not a PIL image? Because `ModelService` is then independent of FastAPI. It could
> be used by a script, a test, or another framework without changes. That separation is a good
> design choice.

### Step 6.3 — Model errors become HTTP errors

```python
if result.get("error"):
    raise HTTPException(status_code=400, detail=result["message"])
```

**Impact:** the model signals *"this is not a usable X-ray"* by returning
`{"error": True, "message": ...}`. FastAPI turns that into a clean `400` response with a
human-readable `detail`, which the frontend displays in an alert.

### Step 6.4 — Generate the explanation (second model call)

```python
tensor = model_service.transform_image(contents)          # rebuild input tensor
cam_matrix = gradcam_engine.generate(tensor, target_class=result["class_id"])

gradcam_b64 = None
if cam_matrix is not None:
    gradcam_b64 = create_gradcam_overlay_base64(contents, cam_matrix)
```

**Impact:**
- The image is transformed **twice** (once inside `predict()`, once here). It is simple and
  correct, just slightly wasteful. A small refactor could reuse one tensor.
- `target_class` is the predicted class from step 6.2 — that is why the model returns `class_id`.
- Grad-CAM produces a small 2D "importance map" (heatmap) showing which regions pushed the
  decision. It is optional: if it fails, the response still works (`gradcam_b64 = None`).

### Step 6.5 — Encode images as base64 (so JSON can carry them)

```python
pil_img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
buf = io.BytesIO()
pil_img.save(buf, format="PNG")
raw_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
```

**Impact:** JSON cannot hold binary image data. Base64 turns bytes into text.
The `data:image/png;base64,...` prefix is a **data URL**, which can be dropped straight into an
`<img src="...">`. The frontend stores both the original and the Grad-CAM overlay this way, and
switches between them with a button — no second download needed.

### Step 6.6 — Return one JSON object

```python
return JSONResponse(content={
    "prediction": result["prediction"],
    "confidence": result["confidence"],
    "probabilities": result["probabilities"],
    "raw_image_b64": raw_b64,
    "gradcam_heatmap_b64": gradcam_b64,
    "medical_disclaimer": "This result is an AI-generated research prediction ..."
})
```

**The response contract** (fix this and the frontend breaks):

| Field | Type | Example |
|---|---|---|
| `prediction` | string | `"Pneumonia"` / `"Normal"` |
| `confidence` | number (%) | `93.4` |
| `probabilities.Normal` | number (%) | `6.6` |
| `probabilities.Pneumonia` | number (%) | `93.4` |
| `raw_image_b64` | data URL string | `data:image/png;base64,...` |
| `gradcam_heatmap_b64` | data URL or `null` | `data:image/png;base64,...` |
| `medical_disclaimer` | string | ... |

### Step 6.7 — Safety net

```python
except HTTPException as he:
    raise he
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
```

**Impact:** `HTTPException` (our own 400s) is re-raised untouched; anything else — corrupt file,
out-of-memory, model bug — becomes a clean `500` instead of crashing the worker. The frontend
then knows the difference between "bad image" (400) and "server problem" (500).

---

## 7. How FastAPI is connected to the model — the full picture

```
                                    ┌──────────────────────────────────────────┐
POST /predict                       │            backend/app/main.py           │
(multipart file)                    │                                          │
     │                              │  1. check content_type                   │
     ▼                              │  2. contents = await file.read()         │
┌───────────────┐                   │  3. model_service.predict(contents) ─────┼──┐
│  main.py      │ ─────────────────►│  4. transform_image(contents)            │  │
│  /predict     │                   │  5. gradcam_engine.generate(tensor, id)  │  │
└───────────────┘                   │  6. create_gradcam_overlay_base64(...)   │  │
     ▲                              │  7. base64 encode raw image              │  │
     │                              │  8. JSONResponse(...)                    │  │
     │                              └──────────────────────────────────────────┘  │
     │                                                                            │
     │                              ┌──────────────────────────────────────────┐  │
     └──────────────────────────────┤            backend/app/model.py          │◄─┘
                                    │  ModelService                            │
                                    │   .model  ← ResNet18 + loaded weights    │
                                    │   .predict()                             │
                                    │     validate_radiograph_input()          │
                                    │     transform_image()                    │
                                    │     model(tensor) → softmax → argmax     │
                                    └──────────────────────────────────────────┘
```

**The connection is three lines of code:**

1. `model_service = ModelService(model_path=...)` → **build + load** at startup.
2. `result = model_service.predict(contents)` → **use** it in `/predict`.
3. `gradcam_engine = GradCAMEngine(model_service.model)` → **inspect the same object** for the
   heatmap.

Everything else is plumbing: validation, byte handling, base64, JSON.

Also note: the `model_service.predict()` return value's keys are **exactly** the keys the route
forwards. There is no second model and no database. The model is state that lives in memory for
the lifetime of the server.

---

## 8. Grad-CAM in plain language (`explainability.py`)

The user asked *"how do the heatmaps work?"*. Short version:

1. **Watch the last conv layer.** Hooks are attached once, at startup:
   ```python
   self.target_layer = model.layer4
   self.target_layer.register_forward_hook(self.save_activation)          # saves feature maps
   self.target_layer.register_full_backward_hook(self.save_gradient)      # saves gradients
   ```
   **Impact:** every forward/backward pass through `layer4` now leaves a copy of its activations
   and gradients in memory. This is how the engine "sees" what the model sees.

2. **Do a forward pass, then a backward pass for the chosen class.**
   ```python
   output = self.model(tensor)
   self.model.zero_grad()
   output[0, target_class].backward()
   ```
   **Impact:** the gradients tell us which feature maps changed the score for *that* class the
   most. We are not training — we only borrow the gradients, then discard them.

3. **Weight each feature map by its average gradient and sum.**
   ```python
   weights = self.gradients[0].cpu().numpy().mean(axis=(1, 2))
   cam = np.zeros(activations.shape[1:], dtype=np.float32)
   for i, w in enumerate(weights):
       cam += w * activations[i]
   cam = np.maximum(cam, 0)          # keep only positive evidence
   cam = cam / np.max(cam)           # normalize to 0..1
   ```
   **Impact:** the result is a small grid (e.g. 7×7) of "importance" values.

4. **Turn the grid into a colored overlay.**
   ```python
   cam_resized = torch.nn.functional.interpolate(cam_tensor, size=(224, 224), ...)
   heatmap_rgb = cm.get_cmap('jet')(cam_resized)[:, :, :3]
   overlay = np.clip(0.6 * raw_np + 0.4 * heatmap_rgb, 0, 1)
   ```
   Red/yellow = regions that pushed the prediction; blue = little influence. The heatmap is
   blended with the original so anatomy stays visible.

**Impact for the user:** the heatmap is a *sanity check*, not proof. A good X-ray model should
light up the lungs, not the corners or the text labels. Treat it as "where the model looked".

---

## 9. The chat endpoint (`/chat` + `agent.py`)

```python
class ChatRequest(BaseModel):
    message: str

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
```

**Impact:** Pydantic (`BaseModel`) automatically validates the JSON body — a missing `message`
field returns a `422` before your code runs. Note this endpoint is **synchronous** because the
agent does no I/O: it is just keyword matching.

The "AI" here is **not** an LLM. `agent.py` lowercases the question and checks keyword lists:

```python
if any(kw in q for kw in ["diagnose me", "my chest hurts", "treatment", ...]):
    return { "response": self.disclaimer + "...", "category": "safety_guardrail" }
```

**Impact:** it is deterministic, free, and offline. One important guardrail: questions asking for
personal medical advice are refused and redirected to a real doctor. Other keywords route to
pre-written answers about the dataset, the model, Grad-CAM, deployment, and limitations.

---

## 10. Running the backend

```powershell
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- `app.main:app` means *"file `app/main.py`, variable `app`"* — this is why imports must be
  `from app.model import ...` and not `from model import ...` when run from `backend/`.
- `--reload` restarts on file changes. Note that **every restart reloads the 45 MB model**, so
  startup takes a moment.
- Open `http://localhost:8000/docs` to test endpoints interactively.
- If port 8000 is busy: `uvicorn app.main:app --reload --port 8001` (then update the frontend
  URLs in `frontend/script.js`).

### Quick manual test

```powershell
curl.exe -X POST http://localhost:8000/predict -F "file=@C:\path\to\xray.png"
```

---

## 11. Tests (`tests/test_api.py`)

The file uses FastAPI's `TestClient`, which calls the app **without a real network**:

```python
from app.main import app
client = TestClient(app)

def test_predict_endpoint():
    img = Image.new("L", (28, 28), color=128)   # solid gray image
    ...
    response = client.post("/predict", files={"file": ("test_xray.png", buf, "image/png")})
    assert response.status_code == 200
```

**Known caveat:** a solid gray image has **zero standard deviation**, so the OOD guardrail rejects
it and the API correctly returns `400`. The test still asserts `200`, so it will fail against the
current code. The fix is to either use a real X-ray sample in the test or assert `400` with the
"Low-contrast" detail.

Run the tests **from `backend/`** so that `app` is importable:

```powershell
cd backend
pytest ..\tests -q
```

---

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: app` | Started from wrong folder | Run `uvicorn app.main:app` from `backend/` |
| `Form data requires "python-multipart"` | Missing dependency | `pip install python-multipart` |
| `/health` shows `model_loaded: false` | `.pth` missing or wrong path | Put the file at `backend/models/medvision_pneumonia_cnn.pth` |
| Every upload returns 400 | Guardrail rejecting the image | Use a real grayscale X-ray |
| Frontend says "not reachable" | Server not running / wrong port | Start uvicorn on 8000 |
| `500 Inference error: ...` | Corrupt image or runtime bug | Check the uvicorn terminal traceback |

---

## 13. One-sentence summary

> FastAPI loads the CNN once at startup, exposes a `/predict` endpoint that reads an uploaded
> image into bytes, hands those bytes to `ModelService.predict()`, turns the resulting
> probabilities into JSON, and also asks Grad-CAM to explain the same model object — then returns
> both the prediction and the heatmap as base64 images.
