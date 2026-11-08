# 🔌 README — How FastAPI Is Connected to the Code (Frontend ↔ Backend)

> **What is this file about?**
> This README explains the **frontend code** (`frontend/index.html`, `styles.css`, `script.js`)
> and, step by step, **how that code is connected to the FastAPI backend** — the URLs, the
> requests, the JSON contracts, the images, and the chat.
>
> Related READMEs:
> - [`README-MODEL.md`](README-MODEL.md) — the CNN itself
> - [`README-BACKEND.md`](README-BACKEND.md) — the FastAPI server and how it calls the model

---

## 1. The setup in plain language

There are **two separate programs**:

| Part | Where it runs | Job |
|---|---|---|
| Frontend | Your browser, from a local file or a static server | Show the page, take an image, display results |
| Backend | `uvicorn` on `http://localhost:8000` | Run the CNN, build the heatmap, answer questions |

They are connected **only by HTTP**. The frontend knows nothing about PyTorch. It just sends a
file and receives JSON.

```
┌──────────────────────────┐          HTTP (fetch)          ┌──────────────────────────┐
│  Browser                 │  POST /predict (image) ──────► │  FastAPI  :8000          │
│  index.html + script.js  │  ◄────── JSON + base64 images  │  app/main.py             │
│                          │                                │        │                 │
│                          │  POST /chat (text) ──────────► │        ▼                 │
│                          │  ◄────── JSON answer           │  ModelService → CNN     │
└──────────────────────────┘                                └──────────────────────────┘
```

**Key idea:** the browser and the server are different "origins". That is why the backend needs
the CORS middleware you saw in `README-BACKEND.md`. Without it, the calls below would be blocked.

---

## 2. The connection points: two constants

Everything starts at the top of `frontend/script.js`:

```javascript
const API_URL = "http://localhost:8000/predict";
const CHAT_API_URL = "http://localhost:8000/chat";
```

**Impact:** these two lines are the *entire* address book. If you change the backend port, these
must change too — otherwise the page says "backend not reachable". There are no proxies, no
environment files, no build step.

The project is **plain static HTML/CSS/JS**. There is no React, no bundler, no `npm install`.
Because of that, opening the file directly works (`file://`), and CORS is *especially* important.

---

## 3. Important step 1 — The user picks an image

Three ways to choose a file, all ending in the same function:

```javascript
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files.length > 0) handleFileSelect(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
});

function handleFileSelect(file) {
  selectedFile = file;
  dropzone.querySelector("p").innerText = `Selected: ${file.name}`;
}
```

**Impact:** click-to-browse and drag-and-drop behave the same. The `File` object is stored in
`selectedFile` — a module-level variable — so the Analyze button can use it later. **No upload
happens yet**; the file stays in the browser until the button is pressed.

---

## 4. ⭐ Important step 2 — Sending the image to FastAPI

```javascript
async function submitAnalysis() {
  if (!selectedFile) { alert("Please select or drop a Chest X-Ray image first!"); return; }

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running PyTorch Inference...`;

  const formData = new FormData();
  formData.append("file", selectedFile);      // field name MUST be "file"

  const response = await fetch(API_URL, {
    method: "POST",
    body: formData
  });
  const data = await response.json();
  ...
}
```

**Impact, line by line:**

- `FormData` + `formData.append("file", ...)` creates a **multipart/form-data** body. The field
  name **must be `"file"`** because the backend declares `file: UploadFile = File(...)`.
  Rename it and FastAPI answers `422 Unprocessable Entity`.
- The `fetch(API_URL, {method:"POST", body:formData})` is the actual connection. No
  `Content-Type` header is set on purpose — the browser adds
  `multipart/form-data; boundary=...` automatically. Setting it by hand breaks the upload.
- `await response.json()` parses the JSON the backend returned.
- Disabling the button prevents double submissions while the CNN is thinking.

### Error handling is split in two

```javascript
if (!response.ok) {
  alert(`⚠️ Image Validation Error:\n${data.detail || "Invalid image file."}`);
  return;
}
```

**Impact:** this is the "**bad image**" branch (HTTP 400 from the OOD guardrail, or 500 from the
server). `data.detail` comes straight from FastAPI. Because the backend includes a helpful
`detail` string, the user learns *why* — e.g. *"Low-contrast or blank image detected."*

```javascript
} catch (error) {
  alert("Backend API at http://localhost:8000/predict is not reachable. ...");
} finally {
  analyzeBtn.disabled = false;
  analyzeBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Analyze Radiograph`;
}
```

**Impact:** the `catch` covers "**network/server down**" (server not started, wrong port, CORS
blocked, JSON parse failure). The `finally` always restores the button, so the UI cannot get stuck
in a loading state.

**Three failure modes, three clear messages:**
| What happened | HTTP | Where it is caught |
|---|---|---|
| Not an image / blank / color photo | 400 | `!response.ok` → `data.detail` |
| Model crash | 500 | `!response.ok` → `data.detail` |
| Server not running | — | `catch` |

---

## 5. Important step 3 — Using the JSON answer

The backend returns the contract described in `README-BACKEND.md`. The frontend consumes it in
`displayResults(data)`:

```javascript
function displayResults(data) {
  placeholderState.style.display = "none";
  resultsContainer.style.display = "flex";

  const isPneumonia = data.prediction === "Pneumonia";

  predictionBadge.innerText = isPneumonia ? "PNEUMONIA DETECTED" : "NORMAL RADIOGRAPH";
  predictionBadge.className = `prediction-badge ${isPneumonia ? 'badge-pneumonia' : 'badge-normal'}`;

  confidenceText.innerText = `${data.confidence}%`;
  confidenceFill.style.width = `${data.confidence}%`;     // 0–100 → bar width

  probPneumonia.innerText = `${data.probabilities.Pneumonia}%`;
  probNormal.innerText = `${data.probabilities.Normal}%`;

  currentRawB64 = data.raw_image_b64;
  currentGradcamB64 = data.gradcam_heatmap_b64 || data.raw_image_b64;

  toggleView("gradcam");
}
```

**Impact — this is where the model's numbers become UI:**

| Backend field | UI effect |
|---|---|
| `prediction` | Green "NORMAL" or red "PNEUMONIA" badge |
| `confidence` | Number + width of the progress bar |
| `probabilities.*` | The two percentage rows |
| `raw_image_b64` | Stored for the "Original X-Ray" button |
| `gradcam_heatmap_b64` | Stored for the heatmap button |

`gradcam_heatmap_b64 || data.raw_image_b64` is a **fallback**: if Grad-CAM failed (backend sent
`null`), the heatmap button simply shows the original image instead of a broken icon.

`toggleView("gradcam")` makes the **heatmap the default view** — the most interesting output is
shown first.

---

## 6. Important step 4 — Switching between original and heatmap

The backend sends **two images already encoded as base64 data URLs**, so switching is instant and
needs no second request:

```javascript
function toggleView(viewType) {
  if (viewType === "raw") {
    visualizerImg.src = currentRawB64;
    btnRaw.classList.add("active");
    btnGradcam.classList.remove("active");
  } else {
    visualizerImg.src = currentGradcamB64;
    btnGradcam.classList.add("active");
    btnRaw.classList.remove("active");
  }
}
```

**Impact:** both images arrive once with the prediction, are cached in two variables, and the
`<img src>` is swapped on click. This is why `/predict` returns `raw_image_b64` as well as
`gradcam_heatmap_b64` — one round trip, two views.

The HTML binds the buttons:

```html
<button class="btn-toggle active" id="btnRaw" onclick="toggleView('raw')">Original X-Ray</button>
<button class="btn-toggle" id="btnGradcam" onclick="toggleView('gradcam')">Grad-CAM Heatmap</button>
```

**Impact:** the functions in `script.js` are global, so inline `onclick` can call them. That works
because there is no module system.

---

## 7. Important step 5 — The chat connection

Same idea, different content type.

```javascript
const res = await fetch(CHAT_API_URL, {
  method: "POST",
  headers: { "Content-Type": "application/json" },   // required: JSON, not a file
  body: JSON.stringify({ message: query })
});

const data = await res.json();
appendChatMessage("agent", data.agent_response);
```

**Impact:** here the body is **JSON**, so the `Content-Type: application/json` header is required
— the opposite of the upload case. The backend validates it with Pydantic (`ChatRequest`), so
`{ "message": "..." }` is the exact shape it accepts.

The reply is inserted as **plain text**:

```javascript
function appendChatMessage(sender, text) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `chat-msg ${sender}`;
  msgDiv.innerText = text;                  // textContent, not innerHTML
  chatMessages.appendChild(msgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;   // auto-scroll to newest
}
```

**Impact:** using `innerText` (instead of `innerHTML`) prevents injected HTML/scripts from running
— a small but real security good practice. The trade-off is that the agent's `**markdown**` and
emojis appear as literal characters rather than formatted text.

`chatInput` + Enter is wired via `onkeypress="handleChatKeyPress(event)"`, which simply calls
`sendChatMessage()` when the key is `Enter`.

---

## 8. The end-to-end lifecycle (one image)

```
1. User drops xray.png
        → handleFileSelect() stores the File in `selectedFile`          (no network)

2. User clicks "Analyze Radiograph"
        → submitAnalysis() builds FormData("file" → xray.png)
        → POST http://localhost:8000/predict                            (network)

3. FastAPI /predict
        → validates content type
        → ModelService.predict(bytes) → guardrail → tensor → ResNet18 → softmax
        → GradCAMEngine.generate(...) → heatmap overlay
        → JSONResponse(prediction, confidence, probabilities, 2 base64 images)

4. Back in the browser
        → displayResults() updates badge, bar, percentages
        → toggleView("gradcam") shows the heatmap
        → user can flip to the original with one click                  (no network)

5. User asks the chat a question
        → POST http://localhost:8000/chat {"message": "..."}
        → MedVisionAgent.answer() returns text + category
        → appendChatMessage() renders it                                (network)
```

**Impact:** one image = one `/predict` request. Everything the UI needs (both images, both
probabilities, confidence) travels in that single response. This keeps the frontend simple and
makes the page feel instant after the first result.

---

## 9. What the "Sample" buttons actually send

```javascript
async function loadSample(sampleType) {
  const canvas = document.createElement("canvas");
  canvas.width = 224; canvas.height = 224;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#0a0d14";
  ctx.fillRect(0, 0, 224, 224);
  ctx.fillStyle = "#253248";
  ctx.beginPath();
  ctx.ellipse(70, 112, 35, 70, 0, 0, 2 * Math.PI);    // left "lung"
  ctx.ellipse(154, 112, 35, 70, 0, 0, 2 * Math.PI);   // right "lung"
  ctx.fill();
  ...
  canvas.toBlob(async (blob) => { ... await submitAnalysis(); }, "image/png");
}
```

**Impact / caveat:** these buttons generate a **synthetic drawing** on a canvas, then upload it
like a real file. They are for demonstrating the UI, **not** for meaningful predictions — the CNN
was trained on real X-rays. Note the repository already contains two **real** dataset images
(`frontend/real_normal_xray.png`, `frontend/real_pneumonia_xray.png`, produced by
`backend/extract_test_samples.py`), but the buttons do **not** use them. If you want a realistic
demo, fetch those files and build a `File`/`Blob` from them instead.

Also, the synthetic drawing is dark and low-contrast; depending on the exact pixels it may be
rejected by the backend's OOD guardrail (`std_dev < 15`, or a color-saturation difference above
18). That would show the red "Image Validation Error" alert — expected behaviour, not a bug.

---

## 10. Running the whole project

### Option A — two steps (simplest)

```powershell
# Terminal 1 — backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `frontend/index.html` directly in a browser (double-click works).

### Option B — serve the frontend too (avoids some `file://` quirks)

```powershell
# Terminal 2 — frontend
cd frontend
python -m http.server 5500
```

Then visit `http://localhost:5500`.

### Full local URL map

| What | URL |
|---|---|
| API root | `http://localhost:8000/` |
| Health | `http://localhost:8000/health` |
| Interactive docs | `http://localhost:8000/docs` |
| Predict | `http://localhost:8000/predict` (POST) |
| Chat | `http://localhost:8000/chat` (POST) |
| Frontend | `frontend/index.html` or `http://localhost:5500` |

**Impact:** order matters — start the backend **before** pressing Analyze. If the `.pth` is
missing, the page still loads, but predictions will be wrong or the server will fail to start.

---

## 11. Troubleshooting the connection

| Symptom on the page | Meaning | Fix |
|---|---|---|
| "Backend API ... is not reachable" | `fetch` failed | Start `uvicorn` on port 8000 |
| Browser console: CORS error | Middleware missing/restricted | Keep `CORSMiddleware` in `main.py` |
| `422 Unprocessable Entity` | Wrong field name or content type | Send field `file` as `FormData`, chat as JSON |
| "Image Validation Error: Low-contrast..." | OOD guardrail rejected the image | Use a real grayscale X-ray |
| "Inference error: ..." (500) | Model/runtime bug | Read the uvicorn traceback |
| Heatmap button shows the raw image | `gradcam_heatmap_b64` was `null` | Check the server log for Grad-CAM |
| Chat always shows the fallback intro | Keyword miss or server down | Try "dataset" / "grad-cam" / "docker" |
| Everything blank after upload | JS error before `displayResults` | Open DevTools → Console |

**Impact:** almost all "broken" reports are one of three things: server not running, wrong port,
or an image the guardrail rejects. Check `/health` first, then the browser console.

---

## 12. Files involved in the connection

| File | Connection role |
|---|---|
| `frontend/index.html` | Buttons + `<img>` + chat drawer; calls global JS functions |
| `frontend/script.js` | **The entire HTTP client**: `fetch` to `/predict` and `/chat` |
| `frontend/styles.css` | Visual only — no networking |
| `backend/app/main.py` | Receives requests, returns JSON, adds CORS headers |
| `backend/app/model.py` | Turns uploaded bytes into a prediction |
| `backend/app/explainability.py` | Builds the heatmap image returned in the same JSON |
| `backend/app/agent.py` | Answers `/chat` questions |

---

## 13. One-sentence summary

> The frontend is a static page whose only job is to `fetch` a file to `/predict` (multipart) and
> text to `/chat` (JSON); FastAPI does all the work and sends back one JSON object containing the
> label, confidence, both probabilities, and two base64 images that the page displays and toggles —
> so the "connection" is just two constants, two `fetch` calls, and an agreed JSON shape.
