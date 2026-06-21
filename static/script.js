/* =====================================================
   VisionAI - script.js
   Handles: drag & drop, file preview, webcam capture,
   AJAX prediction, dark mode toggle.
   ===================================================== */

document.addEventListener("DOMContentLoaded", () => {

  /* ---------- Dark mode ---------- */
  const themeToggle = document.getElementById("themeToggle");
  const root = document.documentElement;
  const savedTheme = localStorageGet("visionai-theme") || "light";
  root.setAttribute("data-bs-theme", savedTheme);
  updateThemeIcon(savedTheme);

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const current = root.getAttribute("data-bs-theme");
      const next = current === "dark" ? "light" : "dark";
      root.setAttribute("data-bs-theme", next);
      localStorageSet("visionai-theme", next);
      updateThemeIcon(next);
    });
  }

  function updateThemeIcon(theme) {
    if (!themeToggle) return;
    themeToggle.innerHTML = theme === "dark"
      ? '<i class="fa-solid fa-sun"></i>'
      : '<i class="fa-solid fa-moon"></i>';
  }

  // Safe localStorage wrappers (works in normal browser context)
  function localStorageSet(key, val) {
    try { localStorage.setItem(key, val); } catch (e) { /* ignore */ }
  }
  function localStorageGet(key) {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  }

  /* ---------- Elements ---------- */
  const dropZone = document.getElementById("dropZone");
  const dropZoneContent = document.getElementById("dropZoneContent");
  const fileInput = document.getElementById("fileInput");
  const previewImg = document.getElementById("previewImg");
  const submitBtn = document.getElementById("submitBtn");
  const clearBtn = document.getElementById("clearBtn");
  const uploadForm = document.getElementById("uploadForm");
  const loadingIndicator = document.getElementById("loadingIndicator");

  const webcamBtn = document.getElementById("webcamBtn");
  const webcamPanel = document.getElementById("webcamPanel");
  const webcamVideo = document.getElementById("webcamVideo");
  const webcamCanvas = document.getElementById("webcamCanvas");
  const captureBtn = document.getElementById("captureBtn");
  const closeWebcamBtn = document.getElementById("closeWebcamBtn");

  let webcamStream = null;

  if (!dropZone) return; // Not on the recognize page

  /* ---------- Drag & drop / browse ---------- */
  dropZone.addEventListener("click", () => fileInput.click());

  ["dragenter", "dragover"].forEach(evt =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    })
  );

  ["dragleave", "drop"].forEach(evt =>
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    })
  );

  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length) {
      fileInput.files = files;
      handleFileSelect(files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFileSelect(fileInput.files[0]);
  });

  function handleFileSelect(file) {
    const allowed = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
    if (!allowed.includes(file.type)) {
      alert("Unsupported file type. Please upload JPG, JPEG, PNG, or WEBP.");
      resetUpload();
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewImg.classList.remove("d-none");
      dropZoneContent.classList.add("d-none");
      submitBtn.disabled = false;
      clearBtn.classList.remove("d-none");
    };
    reader.readAsDataURL(file);
  }

  clearBtn.addEventListener("click", resetUpload);

  function resetUpload() {
    fileInput.value = "";
    previewImg.src = "";
    previewImg.classList.add("d-none");
    dropZoneContent.classList.remove("d-none");
    submitBtn.disabled = true;
    clearBtn.classList.add("d-none");
  }

  uploadForm.addEventListener("submit", () => {
    loadingIndicator.classList.remove("d-none");
    submitBtn.disabled = true;
  });

  /* ---------- Webcam ---------- */
  webcamBtn.addEventListener("click", async () => {
    webcamPanel.classList.remove("d-none");
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
      webcamVideo.srcObject = webcamStream;
    } catch (err) {
      alert("Could not access webcam. Please check permissions.");
      webcamPanel.classList.add("d-none");
    }
  });

  closeWebcamBtn.addEventListener("click", closeWebcam);

  function closeWebcam() {
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      webcamStream = null;
    }
    webcamPanel.classList.add("d-none");
  }

  captureBtn.addEventListener("click", () => {
    const context = webcamCanvas.getContext("2d");
    webcamCanvas.width = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;
    context.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

    webcamCanvas.toBlob((blob) => {
      const formData = new FormData();
      formData.append("image", blob, "webcam_capture.jpg");
      submitViaAjax(formData);
      closeWebcam();
    }, "image/jpeg", 0.92);
  });

  /* ---------- AJAX prediction (used by webcam) ---------- */
  function submitViaAjax(formData) {
    loadingIndicator.classList.remove("d-none");

    fetch(window.PREDICT_API_URL, {
      method: "POST",
      body: formData,
    })
      .then(res => res.json())
      .then(data => {
        loadingIndicator.classList.add("d-none");
        if (data.error) {
          alert(data.error);
          return;
        }
        renderResult(data);
      })
      .catch(() => {
        loadingIndicator.classList.add("d-none");
        alert("Something went wrong while predicting. Please try again.");
      });
  }

  function renderResult(data) {
    const resultCardBody = document.querySelector("#resultCard .card-body");
    const top5Html = data.top5.map(item => `
      <li>
        <span class="top5-label">${item.label}</span>
        <div class="top5-bar-wrap"><div class="top5-bar" style="width: ${item.confidence}%"></div></div>
        <span class="top5-pct">${item.confidence}%</span>
      </li>
    `).join("");

    resultCardBody.innerHTML = `
      <h5 class="card-title-custom"><i class="fa-solid fa-list-check"></i> Prediction Result</h5>
      <div id="resultContent">
        <img src="${data.image_url}" class="result-thumb" alt="Captured image">
        <h3 class="result-label mt-3">${data.prediction}</h3>
        <div class="confidence-wrap">
          <div class="d-flex justify-content-between">
            <span>Confidence</span>
            <span class="fw-bold">${data.confidence}%</span>
          </div>
          <div class="progress confidence-bar">
            <div class="progress-bar confidence-fill" style="width: ${data.confidence}%"></div>
          </div>
        </div>
        <h6 class="top5-heading mt-4">Top 5 Predictions</h6>
        <ul class="top5-list">${top5Html}</ul>
        <a href="/history/report/${data.id}" class="btn btn-outline-app mt-2">
          <i class="fa-solid fa-file-pdf"></i> Download PDF Report
        </a>
      </div>
    `;
  }
});
