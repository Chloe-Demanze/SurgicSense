document.addEventListener("DOMContentLoaded", () => {
  const image = document.getElementById("printImage");
  const message = document.getElementById("noImage");

  const result = localStorage.getItem("lastResult");
  const caseData = JSON.parse(localStorage.getItem("currentCase") || "null");

  // =========================
  // SHOW SCAN IMAGE
  // =========================
  if (result) {
    image.src = result;
    image.classList.add("show");
    message.style.display = "none";
  } else {
    message.textContent = "No scan available.";
  }

  // =========================
  // SHOW ASSESSMENT DATA
  // =========================
  const assessment = caseData?.assessment;
  if (assessment) {
    const section = document.getElementById("assessmentSection");
    const table = document.getElementById("assessmentTable");
    const labels = {
      size: "Size",
      depth: "Depth",
      necroticType: "Necrotic Tissue Type",
      necroticAmount: "Total Necrotic Tissue",
      granulationType: "Granulation Tissue Type",
      granulationAmount: "Total Granulation Tissue",
      edges: "Edges",
      skin: "Periulcer Skin Viability",
    };
    table.innerHTML = Object.entries(labels)
      .map(([key, label]) => `<tr><td><strong>${label}</strong></td><td>${assessment[key] ?? "—"}</td></tr>`)
      .join("");
    section.style.display = "block";
  }

  // =========================
  // PRINT
  // =========================
  document.getElementById("printBtn").onclick = () => {
    if (!result && !assessment) {
      alert("No scan available to print.");
      return;
    }
    window.print();
  };
});
