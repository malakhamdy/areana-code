"use strict";

const state = { files: [], response: null, activeDocument: 0, activeTab: "overview" };
const el = (id) => document.getElementById(id);
const dropZone = el("dropZone");
const fileInput = el("fileInput");
const analyzeButton = el("analyzeButton");

const fieldLabels = {
  national_id: "National ID", name: "Name", address: "Address",
  date_of_birth: "Date of birth", gender: "Gender",
  birth_governorate: "Birth governorate", barcode: "Barcode"
};
const arabicFields = new Set(["name", "address", "gender", "birth_governorate"]);

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
}
function formatBytes(bytes) { return bytes < 1048576 ? `${(bytes/1024).toFixed(0)} KB` : `${(bytes/1048576).toFixed(1)} MB`; }
function pct(value) { return `${Math.round(Number(value || 0) * 100)}%`; }
function titleCase(value) { return String(value || "").replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase()); }
function statusClass(value) { return String(value || "").toLowerCase().replaceAll("_", "-"); }

function openPicker() { fileInput.click(); }
dropZone.addEventListener("click", openPicker);
dropZone.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") openPicker(); });
["dragenter", "dragover"].forEach(name => dropZone.addEventListener(name, event => { event.preventDefault(); dropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach(name => dropZone.addEventListener(name, event => { event.preventDefault(); dropZone.classList.remove("dragging"); }));
dropZone.addEventListener("drop", event => addFiles([...event.dataTransfer.files]));
fileInput.addEventListener("change", () => { addFiles([...fileInput.files]); fileInput.value = ""; });

function addFiles(incoming) {
  const accepted = incoming.filter(file => ["image/jpeg","image/png","image/webp","image/bmp"].includes(file.type) && file.size <= 20*1024*1024);
  state.files = [...state.files, ...accepted].slice(0, 2);
  renderFiles();
}
function renderFiles() {
  el("fileList").innerHTML = state.files.map((file, index) => {
    const url = URL.createObjectURL(file);
    return `<div class="file-item"><img src="${url}" alt="Card image ${index+1}"><div class="file-meta"><strong>${escapeHTML(file.name)}</strong><small>${formatBytes(file.size)} · image ${index+1}</small></div><button data-remove="${index}" aria-label="Remove image">×</button></div>`;
  }).join("");
  el("fileList").querySelectorAll("button[data-remove]").forEach(button => button.addEventListener("click", () => {
    state.files.splice(Number(button.dataset.remove), 1); renderFiles();
  }));
  analyzeButton.disabled = state.files.length === 0;
}

analyzeButton.addEventListener("click", analyze);
el("dismissError").addEventListener("click", () => el("errorPanel").classList.add("hidden"));

async function analyze() {
  if (!state.files.length) return;
  showLoading();
  const form = new FormData();
  state.files.forEach(file => form.append("files", file));
  form.append("detector", el("detector").value);
  form.append("variants", el("variants").value);
  form.append("device", el("device").value);
  try {
    const response = await fetch("/api/analyze", { method: "POST", body: form });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Server returned ${response.status}`);
    state.response = payload; state.activeDocument = 0;
    renderResponse();
    el("resultsSection").classList.remove("hidden");
    el("resultsSection").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    el("errorText").textContent = error.message || "Unknown processing error.";
    el("errorPanel").classList.remove("hidden");
    el("errorPanel").scrollIntoView({ behavior: "smooth" });
  } finally {
    el("loadingPanel").classList.add("hidden");
    el("uploadSection").classList.remove("hidden");
  }
}
function showLoading() {
  el("errorPanel").classList.add("hidden");
  el("resultsSection").classList.add("hidden");
  el("uploadSection").classList.add("hidden");
  el("loadingPanel").classList.remove("hidden");
  const messages = [
    ["Detecting the physical card…", "Finding corners without assuming upload dimensions."],
    ["Building canonical geometry…", "Correcting perspective and stabilizing field positions."],
    ["Reading Arabic fields…", "Generating independent preprocessing and OCR candidates."],
    ["Validating evidence…", "Checking NID structure and cross-field consistency."]
  ];
  let index = 0;
  el("loadingTitle").textContent = messages[0][0]; el("loadingText").textContent = messages[0][1];
  const timer = setInterval(() => {
    if (el("loadingPanel").classList.contains("hidden")) return clearInterval(timer);
    index = (index + 1) % messages.length;
    el("loadingTitle").textContent = messages[index][0]; el("loadingText").textContent = messages[index][1];
  }, 1800);
}

function currentDocument() { return state.response.documents[state.activeDocument]; }
function renderResponse() {
  renderDocumentSwitcher();
  const doc = currentDocument(), result = doc.result, engine = state.response.engine;
  el("engineBanner").className = `engine-banner ${engine.provider.includes("fallback") ? "fallback" : ""}`;
  el("engineBanner").innerHTML = `<span>${engine.provider.includes("fallback") ? "◇" : "●"}</span><div><strong>${escapeHTML(engine.name)}</strong>${escapeHTML(engine.detail)}</div>`;
  renderMetrics(result); renderWarnings(result.warnings); renderFields(result); renderConsistency(result.cross_field_validation);
  renderGeometry(doc); renderCrops(doc); renderEvidence(result); renderJson(doc);
}
function renderDocumentSwitcher() {
  const host = el("documentSwitcher");
  host.innerHTML = state.response.documents.map((doc,index) => `<button class="${index===state.activeDocument?"active":""}" data-document="${index}">Image ${index+1}</button>`).join("");
  host.querySelectorAll("button").forEach(button => button.addEventListener("click", () => { state.activeDocument = Number(button.dataset.document); renderResponse(); }));
}
function renderMetrics(result) {
  const quality = result.image_quality || {};
  const items = [
    ["Document", result.is_egyptian_id ? "ID detected" : "Uncertain", result.is_egyptian_id ? "good" : "uncertain"],
    ["Side", String(result.side || "unknown").toUpperCase(), ""],
    ["Card confidence", pct(result.card_detection_confidence), ""],
    ["Side confidence", pct(result.side_confidence), ""],
    ["Image quality", pct(quality.overall_score), quality.sufficient ? "good" : "uncertain"]
  ];
  el("metrics").innerHTML = items.map(item => `<div class="metric"><small>${item[0]}</small><strong class="${item[2]}">${escapeHTML(item[1])}</strong></div>`).join("");
}
function renderWarnings(warnings = []) {
  const host = el("warnings");
  host.classList.toggle("hidden", !warnings.length);
  host.innerHTML = warnings.length ? `<strong>Review required</strong>${warnings.map(w => `<p>• ${escapeHTML(w)}</p>`).join("")}` : "";
}
function combinedFields(result) { return {...(result.fields || {}), ...(result.derived || {})}; }
function renderFields(result) {
  const fields = combinedFields(result), entries = Object.entries(fields);
  if (!entries.length) { el("fieldsTable").innerHTML = `<div class="warnings"><strong>No semantic fields were forced for this uncertain side.</strong></div>`; return; }
  const rows = entries.map(([name, field]) => {
    const value = field.normalized || "Not extracted", confidence = Number(field.final_confidence || 0);
    return `<div class="field-row"><div><span class="field-name">${escapeHTML(fieldLabels[name] || titleCase(name))}</span><small class="field-source">${escapeHTML(titleCase(field.source))}</small></div><div class="field-value ${arabicFields.has(name)?"rtl":""}" ${arabicFields.has(name)?'dir="rtl" lang="ar"':""}>${escapeHTML(value)}</div><div>${pct(confidence)}<progress class="confidence-bar" max="100" value="${Math.round(confidence*100)}"></progress></div><div>${pct(field.localization_confidence)}</div><span class="status-badge ${statusClass(field.status)}">${escapeHTML(field.status)}</span></div>`;
  }).join("");
  el("fieldsTable").innerHTML = `<div class="field-table"><div class="field-row header"><span>Field</span><span>Canonical value</span><span>Confidence</span><span>Localization</span><span>Status</span></div>${rows}</div>`;
}
function renderConsistency(cross = {}) {
  const matches = cross.matches || [], mismatches = cross.mismatches || [], warnings = cross.warnings || [];
  el("consistency").innerHTML = `<h4>Cross-field consistency · ${escapeHTML(titleCase(cross.overall_consistency || "insufficient evidence"))}</h4><div class="consistency-grid">${matches.map(x=>`<span class="check-chip">✓ ${escapeHTML(titleCase(x))}</span>`).join("")}${mismatches.map(x=>`<span class="check-chip bad">× ${escapeHTML(titleCase(x))}</span>`).join("")}${warnings.map(x=>`<span class="check-chip warn">! ${escapeHTML(x)}</span>`).join("") || '<span class="check-chip warn">No independent second source available</span>'}</div>`;
}
function renderGeometry(doc) {
  const a = doc.artifacts;
  const stages = [
    ["Original image", "Preserved upload", a.original], ["Processing canvas", "Aspect ratio preserved + letterbox", a.processing_canvas],
    ["Physical card detection", "Ordered corners on processing canvas", a.card_detection], ["Canonical card", "Perspective-corrected 1280 × 808", a.canonical_card],
    ["Semantic localization", "Dynamic canonical-card field boxes", a.field_localization]
  ];
  el("geometryGallery").innerHTML = stages.map((stage,index)=>`<figure class="stage-card ${index===4?"wide":""}"><img src="${stage[2]}" alt="${escapeHTML(stage[0])}"><figcaption class="stage-caption"><b>${index+1}</b><span>${escapeHTML(stage[0])}<small>${escapeHTML(stage[1])}</small></span></figcaption></figure>`).join("");
  el("transformJson").textContent = JSON.stringify(doc.result.debug?.transformations || {}, null, 2);
}
function renderCrops(doc) {
  const crops = doc.artifacts.crops || {}, processed = doc.artifacts.preprocessed || {};
  const fields = new Set([...Object.keys(crops), ...Object.keys(processed)]);
  el("cropGallery").innerHTML = fields.size ? [...fields].map(name => {
    const variants = Object.entries(processed[name] || {});
    return `<section class="crop-section"><div class="crop-title"><h4>${escapeHTML(fieldLabels[name] || titleCase(name))}</h4><span>${variants.length} preprocessing variant${variants.length===1?"":"s"}</span></div><div class="crop-grid">${crops[name]?`<figure class="crop-card original"><img src="${crops[name]}" alt="${escapeHTML(name)} canonical crop"><p>Canonical crop · source of truth</p></figure>`:""}${variants.map(([variant,url])=>`<figure class="crop-card"><img src="${url}" alt="${escapeHTML(variant)}"><p>${escapeHTML(titleCase(variant))}</p></figure>`).join("")}</div></section>`;
  }).join("") : `<div class="warnings">No field crops are available for this side.</div>`;
}
function renderEvidence(result) {
  const fields = combinedFields(result);
  el("evidenceList").innerHTML = Object.entries(fields).map(([name,field]) => {
    const candidates = field.candidates || [];
    const candidateTable = candidates.length ? `<table class="candidate-table"><thead><tr><th>Variant</th><th>Raw OCR</th><th>Normalized</th><th>OCR</th><th>Rank</th><th>Validation</th></tr></thead><tbody>${candidates.map(c=>`<tr><td>${escapeHTML(titleCase(c.preprocessing_variant))}</td><td class="candidate-value">${escapeHTML(c.raw||"—")}</td><td class="candidate-value">${escapeHTML(c.normalized||"—")}</td><td>${pct(c.confidence)}</td><td>${pct(c.score)}</td><td>${escapeHTML(c.validation?.status||"—")}</td></tr>`).join("")}</tbody></table>` : `<p class="evidence-reason">No OCR candidates; this value may be derived from NID structure.</p>`;
    return `<details class="evidence-card" ${name==="national_id"?"open":""}><summary><span>${escapeHTML(fieldLabels[name]||titleCase(name))}</span><span class="status-badge ${statusClass(field.status)}">${escapeHTML(field.status)}</span></summary><div class="evidence-body"><p class="evidence-reason">${escapeHTML(field.verification?.reason || "No independent verification evidence.")}</p>${candidateTable}<pre class="evidence-json">${escapeHTML(JSON.stringify({validation:field.validation,verification:field.verification,warnings:field.warnings},null,2))}</pre></div></details>`;
  }).join("") || `<div class="warnings">No field evidence is available.</div>`;
}
function renderJson(doc) {
  const sensitive = el("sensitiveToggle").checked;
  const payload = sensitive ? doc.result : doc.safe_result;
  el("jsonOutput").textContent = JSON.stringify(payload, null, 2);
  el("jsonMode").textContent = sensitive ? "Complete user-controlled export" : "Sensitive identifiers masked";
}
el("sensitiveToggle").addEventListener("change", () => { if(state.response) renderJson(currentDocument()); });
el("downloadJson").addEventListener("click", () => {
  if (!state.response) return;
  const sensitive = el("sensitiveToggle").checked, doc = currentDocument();
  const payload = sensitive ? doc.result : doc.safe_result;
  const blob = new Blob([JSON.stringify(payload,null,2)], {type:"application/json"});
  const url = URL.createObjectURL(blob), link = document.createElement("a");
  link.href=url; link.download=`egyptian-id-result-${doc.index}${sensitive?"-complete":"-masked"}.json`; link.click(); URL.revokeObjectURL(url);
});

document.querySelectorAll(".result-tabs button").forEach(button => button.addEventListener("click", () => {
  state.activeTab = button.dataset.tab;
  document.querySelectorAll(".result-tabs button").forEach(x=>x.classList.toggle("active",x===button));
  document.querySelectorAll(".tab-panel").forEach(panel=>panel.classList.toggle("active",panel.id===`tab-${state.activeTab}`));
}));
