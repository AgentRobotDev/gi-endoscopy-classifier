const CLASS_INFO = {
  "dyed-lifted-polyps":
    "A polyp that has been injected with dye and lifted from the mucosal layer prior to endoscopic removal.",
  "dyed-resection-margins":
    "Tissue margins after polyp resection, dyed to verify complete removal with clear borders.",
  "esophagitis":
    "Inflammation of the esophageal lining, commonly caused by acid reflux — visible as redness and erosions.",
  "normal-cecum":
    "Healthy appearance of the cecum, the junction of the small and large intestine.",
  "normal-pylorus":
    "Healthy pyloric valve — the muscular gateway between the stomach and the small intestine.",
  "normal-z-line":
    "Normal Z-line — the sharp boundary where the pale esophagus meets the pink stomach lining.",
  "polyps":
    "Abnormal tissue growths projecting from the mucosal surface; potential precancerous lesions.",
  "ulcerative-colitis":
    "Chronic inflammatory bowel disease causing continuous ulceration and bleeding of the colon.",
};

let allSamples = {};
let activeFilter = "all";
let selectedGalleryUrl = null;

// ── Init ───────────────────────────────────────────────────────────────────

async function init() {
  document.getElementById("predictBtn").addEventListener("click", runInference);

  const galleryToggle = document.getElementById("galleryToggle");
  const galleryBody = document.getElementById("galleryBody");

  function toggleGallery() {
    const expanded = galleryToggle.getAttribute("aria-expanded") === "true";
    galleryToggle.setAttribute("aria-expanded", String(!expanded));
    galleryBody.hidden = expanded;
  }

  galleryToggle.addEventListener("click", toggleGallery);
  galleryToggle.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleGallery(); }
  });

  try {
    const res = await fetch("/samples");
    allSamples = await res.json();
  } catch {
    allSamples = {};
  }

  if (Object.keys(allSamples).length === 0) {
    document.getElementById("noSamples").style.display = "block";
  } else {
    renderFilterBar();
    renderGallery();
  }
}

// ── Filter bar ─────────────────────────────────────────────────────────────

function renderFilterBar() {
  const bar = document.getElementById("filterBar");
  const classes = ["all", ...Object.keys(allSamples).sort()];

  bar.innerHTML = classes
    .map(
      (cls) =>
        `<button class="filter-btn${cls === activeFilter ? " active" : ""}"
                 data-class="${cls}" role="tab"
                 aria-selected="${cls === activeFilter}">
           ${cls === "all" ? "All Classes" : formatClass(cls)}
         </button>`
    )
    .join("");

  bar.querySelectorAll(".filter-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeFilter = btn.dataset.class;
      bar.querySelectorAll(".filter-btn").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      renderGallery();
    });
  });
}

// ── Image row ──────────────────────────────────────────────────────────────

function renderGallery() {
  const grid = document.getElementById("imageGrid");

  let images;
  if (activeFilter === "all") {
    const all = Object.entries(allSamples).flatMap(([cls, srcs]) =>
      srcs.map((src) => ({ src, cls }))
    );
    images = shuffle(all).slice(0, 6);
  } else {
    images = shuffle(
      (allSamples[activeFilter] ?? []).map((src) => ({ src, cls: activeFilter }))
    ).slice(0, 6);
  }

  grid.innerHTML = `<div class="single-row">
    ${images
      .map(
        ({ src, cls }) => `
      <div class="img-card${src === selectedGalleryUrl ? " selected" : ""}"
           data-src="${src}" data-class="${cls}" role="button" tabindex="0"
           aria-label="${formatClass(cls)} sample image">
        <img src="${src}" alt="${formatClass(cls)}" loading="lazy">
      </div>`
      )
      .join("")}
  </div>`;

  grid.querySelectorAll(".img-card").forEach((card) => {
    card.addEventListener("click", () => selectCard(card));
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") selectCard(card);
    });
  });
}

function selectCard(card) {
  clearGallerySelection();
  card.classList.add("selected");
  selectedGalleryUrl = card.dataset.src;
  showPreview(card.dataset.src);
}

function clearGallerySelection() {
  document.querySelectorAll(".img-card.selected").forEach((c) =>
    c.classList.remove("selected")
  );
}

// ── Preview panel ──────────────────────────────────────────────────────────

function showPreview(src) {
  const panel = document.getElementById("predictPanel");
  const img = document.getElementById("selectedImg");
  img.src = src;
  panel.style.display = "flex";
  document.getElementById("results").innerHTML = "";
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Inference ──────────────────────────────────────────────────────────────

async function runInference() {
  const btn = document.getElementById("predictBtn");
  const results = document.getElementById("results");

  btn.disabled = true;
  btn.textContent = "Analyzing…";
  results.innerHTML =
    '<div class="msg"><span class="spinner"></span>Running inference…</div>';

  try {
    const blob = await fetch(selectedGalleryUrl).then((r) => r.blob());
    const file = new File([blob], "sample.jpg", { type: blob.type || "image/jpeg" });

    const form = new FormData();
    form.append("file", file);

    const res = await fetch("/predict", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail ?? `HTTP ${res.status}`);
    }

    const { predictions } = await res.json();
    renderResults(predictions);
  } catch (e) {
    results.innerHTML = `<div class="msg error">Error: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Inference";
  }
}

// ── Results rendering ──────────────────────────────────────────────────────

function renderResults(predictions) {
  const top = predictions[0];
  const results = document.getElementById("results");

  results.innerHTML = `
    <div class="result-top">
      <div class="result-prediction-label">Prediction</div>
      <div class="result-class-name">${formatClass(top.class)}</div>
      <div class="result-confidence">${(top.probability * 100).toFixed(1)}% confidence</div>
      <div class="result-desc">${CLASS_INFO[top.class] ?? ""}</div>
    </div>
    <div class="result-bars">
      ${predictions
        .map((p, i) => {
          const pct = (p.probability * 100).toFixed(1);
          const isTop = i === 0;
          return `
          <div class="bar-row">
            <div class="bar-label${isTop ? " top" : ""}">${formatClass(p.class)}</div>
            <div class="bar-track">
              <div class="bar-fill${isTop ? " top" : ""}" data-width="${pct}%"></div>
            </div>
            <div class="bar-pct${isTop ? " top" : ""}">${pct}%</div>
          </div>`;
        })
        .join("")}
    </div>`;

  // Double rAF to trigger CSS transition after DOM paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      results.querySelectorAll(".bar-fill").forEach((bar) => {
        bar.style.width = bar.dataset.width;
      });
    });
  });
}

// ── Metrics ────────────────────────────────────────────────────────────────

async function initMetrics() {
  const container = document.getElementById("metricsContent");

  let metrics;
  try {
    const res = await fetch("/static/metrics.json");
    if (!res.ok) throw new Error("not found");
    metrics = await res.json();
  } catch {
    return; // placeholder message already in HTML
  }

  const fmt = (v) => (v * 100).toFixed(1) + "%";

  // ── Summary cards ────────────────────────────────
  const summary = `
    <div class="metrics-summary">
      <div class="stat-card">
        <div class="stat-label">Accuracy</div>
        <div class="stat-value">${fmt(metrics.accuracy)}</div>
        <div class="stat-desc">Top-1 on validation set</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Macro F1</div>
        <div class="stat-value">${fmt(metrics.macro_f1)}</div>
        <div class="stat-desc">Equal weight per class</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Weighted F1</div>
        <div class="stat-value">${fmt(metrics.weighted_f1)}</div>
        <div class="stat-desc">Weighted by class size</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Val Images</div>
        <div class="stat-value">${Object.values(metrics.per_class).reduce((s, c) => s + c.support, 0).toLocaleString()}</div>
        <div class="stat-desc">20% holdout split</div>
      </div>
    </div>`;

  // ── Per-class table ──────────────────────────────
  const tableRows = metrics.classes.map((cls) => {
    const c = metrics.per_class[cls];
    return `
      <tr>
        <td class="td-class">${formatClass(cls)}</td>
        <td class="right">
          <div class="metric-bar-cell">
            <div class="metric-inline-bar">
              <div class="metric-inline-fill" style="width:${(c.precision*100).toFixed(1)}%"></div>
            </div>
            ${fmt(c.precision)}
          </div>
        </td>
        <td class="right">
          <div class="metric-bar-cell">
            <div class="metric-inline-bar">
              <div class="metric-inline-fill" style="width:${(c.recall*100).toFixed(1)}%"></div>
            </div>
            ${fmt(c.recall)}
          </div>
        </td>
        <td class="right">
          <div class="metric-bar-cell">
            <div class="metric-inline-bar">
              <div class="metric-inline-fill" style="width:${(c.f1*100).toFixed(1)}%"></div>
            </div>
            ${fmt(c.f1)}
          </div>
        </td>
        <td class="right">
          <div class="metric-bar-cell">
            <div class="metric-inline-bar">
              <div class="metric-inline-fill" style="width:${(c.auc*100).toFixed(1)}%"></div>
            </div>
            ${fmt(c.auc)}
          </div>
        </td>
        <td class="right">${c.support}</td>
      </tr>`;
  }).join("");

  const table = `
    <div class="metrics-table-wrap">
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Class</th>
            <th class="right">Precision</th>
            <th class="right">Recall</th>
            <th class="right">F1</th>
            <th class="right">AUC-ROC</th>
            <th class="right">Images</th>
          </tr>
        </thead>
        <tbody>${tableRows}</tbody>
      </table>
    </div>`;

  // ── Confusion matrix ─────────────────────────────
  const cm = metrics.confusion_matrix;
  const n = metrics.classes.length;
  const rowMax = cm.map((row) => Math.max(...row));

  const cols = 1 + n; // label col + n data cols
  const rows = 1 + n; // header row + n data rows

  let cells = "";

  // corner
  cells += `<div class="cm-cell corner" style="grid-column:1;grid-row:1"></div>`;

  // column headers (predicted class) — row 1
  metrics.classes.forEach((cls, j) => {
    cells += `<div class="cm-cell header-row" style="grid-column:${j+2};grid-row:1"
                   title="${formatClass(cls)}">${formatClass(cls)}</div>`;
  });

  // data rows
  metrics.classes.forEach((cls, i) => {
    // row label (true class)
    cells += `<div class="cm-cell header-col" style="grid-column:1;grid-row:${i+2}">${formatClass(cls)}</div>`;

    cm[i].forEach((val, j) => {
      const isDiag = i === j;
      const intensity = rowMax[i] > 0 ? val / rowMax[i] : 0;
      let bg;
      if (isDiag) {
        bg = `rgba(20,184,166,${0.12 + intensity * 0.75})`;
      } else {
        bg = intensity > 0 ? `rgba(248,113,113,${0.08 + intensity * 0.65})` : "var(--surface-2)";
      }
      const textColor = isDiag && intensity > 0.5 ? "#0a0e1a" : "var(--text)";
      cells += `<div class="cm-cell" style="grid-column:${j+2};grid-row:${i+2};background:${bg};color:${textColor}"
                     title="True: ${formatClass(cls)} → Pred: ${formatClass(metrics.classes[j])} (${val})">${val}</div>`;
    });
  });

  const matrix = `
    <div style="margin-top:8px">
      <div class="cm-heading">Confusion Matrix</div>
      <p class="cm-subheading">Rows = true class &nbsp;·&nbsp; Columns = predicted class &nbsp;·&nbsp; Diagonal = correct predictions</p>
      <div class="cm-wrap">
        <div class="cm-grid" style="grid-template-columns: 80px repeat(${n}, 58px); grid-template-rows: 64px repeat(${n}, 44px);">
          ${cells}
        </div>
      </div>
    </div>`;

  container.innerHTML = summary + table + matrix;
}

// ── Architecture ───────────────────────────────────────────────────────────

const ARCHITECTURE = [
  {
    name: "Input Preprocessing",
    tag: "torchvision.transforms",
    highlight: false,
    code: `transforms.Resize(256)\ntransforms.CenterCrop(224)\ntransforms.ToTensor()\ntransforms.Normalize(\n  mean=[0.485, 0.456, 0.406],\n  std=[0.229, 0.224, 0.225]\n)`,
    desc: `Every image is resized to 256px, center-cropped to <strong>224×224</strong>, then normalized using the mean and standard deviation from ImageNet. Normalization centers the pixel distribution so gradient descent converges faster and the pretrained weights remain valid.`,
  },
  {
    name: "Stem Convolution",
    tag: "backbone — frozen in Phase 1",
    highlight: false,
    code: `Conv2d(3, 32, kernel_size=3,\n       stride=2, padding=1, bias=False)\nBatchNorm2d(32)\nSiLU()`,
    desc: `The first layer applies <strong>32 convolutional filters</strong> of size 3×3 across the image, halving the spatial resolution from 224×224 to 112×112 (stride=2). This detects basic visual primitives — edges, corners, color gradients. Batch normalization stabilizes activations. SiLU (sigmoid × input) is the activation function.`,
  },
  {
    name: "MBConv Blocks × 16",
    tag: "backbone — 7 stages, pretrained on ImageNet",
    highlight: false,
    code: `# Each block:\nConv2d(in, in*6, kernel_size=1)   # expand\nDepthwiseConv2d(kernel=3 or 5)    # spatial filter\nSqueezeExcitation()               # channel attention\nConv2d(in*6, out, kernel_size=1)  # project\n# Spatial: 112×112 → 7×7\n# Channels: 32 → 320`,
    desc: `The backbone. Each <strong>Mobile Inverted Bottleneck</strong> block expands channels, applies a depthwise convolution (one filter per channel — very efficient), then re-weights channels using Squeeze-and-Excitation attention, then compresses back down. Across 7 stages the spatial resolution shrinks from 112×112 to 7×7 while channels grow from 32 to 320. These weights are <strong>pretrained on ImageNet</strong> and frozen during Phase 1 training.`,
  },
  {
    name: "Head Convolution",
    tag: "backbone — top of feature extractor",
    highlight: false,
    code: `Conv2d(320, 1280, kernel_size=1, bias=False)\nBatchNorm2d(1280)\nSiLU()`,
    desc: `A <strong>1×1 convolution</strong> expands the 320-channel feature maps to 1280 channels before pooling. Unlike spatial convolutions, a 1×1 conv operates on each pixel independently — it learns a weighted combination of the 320 features, creating a richer representation without changing the 7×7 spatial dimensions.`,
  },
  {
    name: "Global Average Pooling",
    tag: "spatial collapse",
    highlight: false,
    code: `AdaptiveAvgPool2d(output_size=1)\n# 1280 × 7 × 7  →  1280 × 1 × 1\n# (flattened to a 1280-dim vector)`,
    desc: `Each of the 1280 feature maps (each 7×7) is averaged down to a <strong>single number</strong>, producing a 1280-dimensional vector. This is where all spatial information is discarded — the model no longer knows where in the image features were located. This is why the classifier can tell you <em>what</em> is in the image but not <em>where</em>.`,
  },
  {
    name: "Classifier Head",
    tag: "fine-tuned — our custom replacement",
    highlight: true,
    code: `Dropout(p=0.2)\nLinear(in_features=1280, out_features=8)`,
    desc: `Our <strong>custom replacement</strong> for EfficientNet's original 1000-class ImageNet head. Dropout randomly zeros 20% of the 1280 activations during training, forcing the network to learn redundant representations and reducing overfitting. The linear layer is a matrix multiply: <strong>1280 inputs → 8 raw scores</strong>, one per Kvasir class.`,
  },
  {
    name: "Output",
    tag: "inference only",
    highlight: false,
    code: `Softmax(dim=1)\n# 8 raw scores → 8 probabilities\n# all values in [0, 1], sum to 1.0\n\n# Prediction = argmax(probabilities)`,
    desc: `Softmax converts the 8 raw scores (logits) into <strong>probabilities</strong> that sum to 1.0. The predicted class is whichever has the highest probability. During training we use cross-entropy loss directly on the logits (numerically more stable than applying softmax first); softmax is only applied at inference time for display.`,
  },
];

function initArchitecture() {
  const layers = document.getElementById("archLayers");

  layers.innerHTML = ARCHITECTURE.map((layer, i) => `
    <div class="arch-layer${layer.highlight ? " highlight" : ""}">
      <div class="arch-step">
        <div class="arch-step-num">${i + 1}</div>
        <div class="arch-connector"></div>
      </div>
      <div class="arch-name-col">
        <div class="arch-layer-name">${layer.name}</div>
        <div class="arch-layer-tag">${layer.tag}</div>
      </div>
      <div class="arch-code-col">
        <pre class="arch-code">${layer.code}</pre>
      </div>
      <div class="arch-desc-col">
        <p class="arch-desc">${layer.desc}</p>
      </div>
    </div>
  `).join("");
}

// ── Helpers ────────────────────────────────────────────────────────────────

function formatClass(cls) {
  return cls.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

document.addEventListener("DOMContentLoaded", () => { init(); initMetrics(); initArchitecture(); });
