const uploadBox = document.getElementById("uploadBox");
const fileInput = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");
const dropText = document.getElementById("dropText");
const predictBtn = document.getElementById("predictBtn");
const loader = document.getElementById("loader");
const resultBox = document.getElementById("resultBox");
const errorBox = document.getElementById("errorBox");
const predClass = document.getElementById("predClass");
const predConf = document.getElementById("predConf");
const probList = document.getElementById("probList");

let selectedFile = null;

// ক্লিক করে ফাইল সিলেক্ট
uploadBox.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    handleFile(e.target.files[0]);
  }
});

// ড্র্যাগ এন্ড ড্রপ
uploadBox.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadBox.classList.add("dragover");
});

uploadBox.addEventListener("dragleave", () => {
  uploadBox.classList.remove("dragover");
});

uploadBox.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadBox.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    handleFile(e.dataTransfer.files[0]);
  }
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("শুধুমাত্র ইমেজ ফাইল আপলোড করুন।");
    return;
  }

  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    dropText.hidden = true;
  };
  reader.readAsDataURL(file);

  predictBtn.disabled = false;
  hideError();
  resultBox.hidden = true;
}

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  resultBox.hidden = true;
  loader.hidden = false;
  predictBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "একটি সমস্যা হয়েছে।");
    }

    showResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    loader.hidden = true;
    predictBtn.disabled = false;
  }
});

function showResult(data) {
  predClass.textContent = data.predicted_class;
  predConf.textContent = (data.confidence * 100).toFixed(2) + "%";

  probList.innerHTML = "";
  for (const [label, prob] of Object.entries(data.probabilities)) {
    const pct = (prob * 100).toFixed(2);
    const row = document.createElement("div");
    row.innerHTML = `
      <div class="prob-row">
        <span>${label}</span>
        <span>${pct}%</span>
      </div>
      <div class="bar-bg">
        <div class="bar-fill" style="width:${pct}%"></div>
      </div>
    `;
    probList.appendChild(row);
  }

  resultBox.hidden = false;
}

function showError(msg) {
  errorBox.textContent = "⚠️ " + msg;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}
