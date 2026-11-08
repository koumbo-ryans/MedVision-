const API_URL = "http://localhost:8000/predict";
const CHAT_API_URL = "http://localhost:8000/chat";

let currentRawB64 = "";
let currentGradcamB64 = "";
let selectedFile = null;

// DOM Elements
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const placeholderState = document.getElementById("placeholderState");
const resultsContainer = document.getElementById("resultsContainer");

const predictionBadge = document.getElementById("predictionBadge");
const confidenceText = document.getElementById("confidenceText");
const confidenceFill = document.getElementById("confidenceFill");
const probPneumonia = document.getElementById("probPneumonia");
const probNormal = document.getElementById("probNormal");
const visualizerImg = document.getElementById("visualizerImg");

const btnRaw = document.getElementById("btnRaw");
const btnGradcam = document.getElementById("btnGradcam");

const chatDrawer = document.getElementById("chatDrawer");
const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");

// 1. Setup Dropzone Event Listeners
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    handleFileSelect(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    handleFileSelect(e.target.files[0]);
  }
});

function handleFileSelect(file) {
  selectedFile = file;
  dropzone.querySelector("p").innerText = `Selected: ${file.name}`;
}

// 2. Submit Image for Inference to FastAPI Backend
async function submitAnalysis() {
  if (!selectedFile) {
    alert("Please select or drop a Chest X-Ray image first!");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running PyTorch Inference...`;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      alert(`⚠️ Image Validation Error:\n${data.detail || "Invalid image file."}`);
      return;
    }

    displayResults(data);

  } catch (error) {
    alert("Backend API at http://localhost:8000/predict is not reachable. Make sure uvicorn server is running!");
    console.error("API Error:", error);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Analyze Radiograph`;
  }
}

// 3. Display Real API Results
function displayResults(data) {
  placeholderState.style.display = "none";
  resultsContainer.style.display = "flex";

  const isPneumonia = data.prediction === "Pneumonia";

  predictionBadge.innerText = isPneumonia ? "PNEUMONIA DETECTED" : "NORMAL RADIOGRAPH";
  predictionBadge.className = `prediction-badge ${isPneumonia ? 'badge-pneumonia' : 'badge-normal'}`;

  confidenceText.innerText = `${data.confidence}%`;
  confidenceFill.style.width = `${data.confidence}%`;

  probPneumonia.innerText = `${data.probabilities.Pneumonia}%`;
  probNormal.innerText = `${data.probabilities.Normal}%`;

  currentRawB64 = data.raw_image_b64;
  currentGradcamB64 = data.gradcam_heatmap_b64 || data.raw_image_b64;

  toggleView("gradcam");
}

// 4. Toggle Raw vs Grad-CAM View
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

// 5. Send Real Sample Image to Backend API
async function loadSample(sampleType) {
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#0a0d14";
  ctx.fillRect(0, 0, 224, 224);

  ctx.fillStyle = "#253248";
  ctx.beginPath();
  ctx.ellipse(70, 112, 35, 70, 0, 0, 2 * Math.PI);
  ctx.ellipse(154, 112, 35, 70, 0, 0, 2 * Math.PI);
  ctx.fill();

  if (sampleType === "pneumonia") {
    ctx.fillStyle = "rgba(240, 240, 255, 0.75)";
    ctx.beginPath();
    ctx.arc(154, 120, 24, 0, 2 * Math.PI);
    ctx.fill();
  }

  canvas.toBlob(async (blob) => {
    const file = new File([blob], `sample_${sampleType}.png`, { type: "image/png" });
    selectedFile = file;
    dropzone.querySelector("p").innerText = `Selected Sample: sample_${sampleType}.png`;
    await submitAnalysis();
  }, "image/png");
}

// 6. Floating AI Chatbot Handlers
function toggleChatDrawer() {
  const isVisible = chatDrawer.style.display === "flex";
  chatDrawer.style.display = isVisible ? "none" : "flex";
}

function handleChatKeyPress(event) {
  if (event.key === "Enter") {
    sendChatMessage();
  }
}

async function sendChatMessage() {
  const query = chatInput.value.trim();
  if (!query) return;

  appendChatMessage("user", query);
  chatInput.value = "";

  try {
    const res = await fetch(CHAT_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: query })
    });

    if (!res.ok) throw new Error("Agent unreachable");
    const data = await res.json();
    appendChatMessage("agent", data.agent_response);

  } catch (err) {
    appendChatMessage("agent", "👋 I am the MedVision-CNN Assistant. Ensure the FastAPI backend server is running on port 8000!");
  }
}

function appendChatMessage(sender, text) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `chat-msg ${sender}`;
  msgDiv.innerText = text;
  chatMessages.appendChild(msgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
