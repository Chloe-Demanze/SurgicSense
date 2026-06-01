document.addEventListener("DOMContentLoaded", () => {
  // const API_BASE = "http://localhost:8000";

  const previewBox = document.getElementById("preview");
  const resultBox = document.getElementById("result");

  const currentModelText = document.getElementById("currentModelText");

  const segmentBtn = document.getElementById("segmentBtn");
  const changeBtn = document.getElementById("changeBtn");

  //  loader
  const btnText = document.getElementById("btnText");
  const btnLoader = document.getElementById("btnLoader");

  // =========================
  // GET MODEL FROM STORAGE
  // =========================
  const model = localStorage.getItem("lastModelName");

  if (!model) {
    currentModelText.textContent = "No model selected";
    segmentBtn.disabled = true;
    return;
  }

  currentModelText.textContent = model;

  // =========================
  // LOAD PREVIEW
  // =========================
  loadPreview(model);

  async function loadPreview(model) {
    try {
      const formData = new FormData();
      formData.append("model_name", model);

      const res = await fetch(`${API_BASE}/api/preview`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Preview failed");

      const data = await res.json();

      previewBox.innerHTML = "";

      const img = document.createElement("img");
      img.src = `${API_BASE}${data.texture_url}?t=${Date.now()}`;

      previewBox.appendChild(img);
    } catch (err) {
      previewBox.innerHTML = "<p>Preview failed</p>";
    }
  }

  // =========================
  // SEGMENT
  // =========================
  segmentBtn.onclick = async () => {
    btnText.textContent = "Processing...";
    btnLoader.classList.remove("hidden");
    segmentBtn.disabled = true;

    try {
      const formData = new FormData();
      formData.append("model_name", model);
      formData.append("method", "kmeans Jay");

      const res = await fetch(`${API_BASE}/api/segment`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Segmentation failed");
      }

      const data = await res.json();

      resultBox.innerHTML = "";

      const img = document.createElement("img");
      img.src = `${API_BASE}${data.texture_url}?t=${Date.now()}`;

      resultBox.appendChild(img);

      localStorage.setItem("lastResult", `${API_BASE}${data.texture_url}`);
      localStorage.setItem("currentCase", JSON.stringify({
        modelName: model,
        resultUrl: `${API_BASE}${data.texture_url}`,
        assessment: null
      }));
    } catch (err) {
      alert(err.message);
    } finally {
      btnText.textContent = "Run Segmentation";
      btnLoader.classList.add("hidden");
      segmentBtn.disabled = false;
    }
  };

  // =========================
  // CHANGE BUTTON
  // =========================
  changeBtn.onclick = () => {
    window.location.href = "upload-scan.html";
  };
});
