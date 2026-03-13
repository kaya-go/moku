/**
 * Moku Corner Annotator — annotation logic
 *
 * Annotations format (per image):
 *   { boxes: [{id, x, y, w, h, category}] }
 *
 * category codes: 0=black_stone, 1=white_stone, 2=board_corner
 *
 * Keyboard shortcuts:
 *   Click empty space  → add board_corner bbox (max 4)
 *   Click on bbox      → select
 *   Drag selected      → move
 *   Right-click bbox   → delete
 *   Del / Backspace    → delete selected
 *   S / Ctrl+S         → save
 *   N / ArrowRight     → next image
 *   P / ArrowLeft      → prev image
 *   Escape             → deselect
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CORNER_CAT   = 2;
const MAX_CORNERS  = 4;
const CORNER_SIZE  = 20;   // default new-bbox side length in image pixels
const MAG_SIZE     = 180;  // magnifier canvas width/height
const MAG_ZOOM     = 4;    // magnifier zoom factor
const DRAG_THRESH  = 4;    // pixels before a mousedown is treated as a drag

const CAT_COLORS = {
  0: '#e7298a',  // black_stone — magenta
  1: '#1b9e77',  // white_stone — teal
  2: '#d95f02',  // board_corner — orange
};
const CAT_NAMES = { 0: 'black_stone', 1: 'white_stone', 2: 'board_corner' };

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let images = [];          // [{id, filename, source, flagged, flag_reason, ...}]
let currentIdx  = 0;
let currentAnns = [];     // [{id, x, y, w, h, category}] — image-pixel coords
let selectedId  = null;

let isDragging   = false;
let hasDragged   = false;
let dragStartX   = 0, dragStartY = 0;
let dragOrigX    = 0, dragOrigY  = 0;

let scale   = 1.0;        // canvas px → image px factor
let mouseX  = 0, mouseY  = 0;  // canvas coords

let canvas, ctx, magCanvas, magCtx, currentImg;

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

window.onload = () => {
  canvas    = document.getElementById('main-canvas');
  ctx       = canvas.getContext('2d');
  magCanvas = document.getElementById('mag-canvas');
  magCtx    = magCanvas.getContext('2d');

  canvas.addEventListener('mousedown',    onMouseDown);
  canvas.addEventListener('mousemove',    onMouseMove);
  canvas.addEventListener('mouseup',      onMouseUp);
  canvas.addEventListener('contextmenu',  onRightClick);
  document.addEventListener('keydown',    onKeyDown);

  loadImageList();
};

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

async function loadImageList() {
  const resp = await fetch('/api/images');
  const data = await resp.json();
  images = data.images || [];
  renderSidebar();
  if (images.length > 0) loadImage(0);
}

async function loadImage(idx) {
  if (idx < 0 || idx >= images.length) return;
  currentIdx = idx;
  selectedId = null;

  const img = images[idx];
  document.getElementById('image-name').textContent = img.filename || '';

  // Load annotations (corrected if available, else original)
  const annResp = await fetch(`/api/annotations/${img.id}`);
  const annData = await annResp.json();
  currentAnns = (annData.boxes || []).map(b => ({ ...b }));

  // Load image
  const imgEl = new Image();
  imgEl.onload = () => {
    currentImg = imgEl;
    fitCanvas(imgEl.width, imgEl.height);
    render();
    renderAnnPanel();
  };
  imgEl.src = `/api/image/${img.filename}`;

  renderSidebar();
  setStatus();
}

// ---------------------------------------------------------------------------
// Canvas / rendering
// ---------------------------------------------------------------------------

function fitCanvas(w, h) {
  const wrap  = document.getElementById('canvas-wrap');
  const maxW  = wrap.clientWidth  - 20;
  const maxH  = wrap.clientHeight - 20;
  scale       = Math.min(1, maxW / w, maxH / h);
  canvas.width  = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
}

function render() {
  if (!currentImg) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(currentImg, 0, 0, canvas.width, canvas.height);

  // Draw bboxes
  currentAnns.forEach(box => {
    const sel   = box.id === selectedId;
    const color = CAT_COLORS[box.category] ?? '#ffffff';
    const sx = box.x * scale, sy = box.y * scale;
    const sw = box.w * scale, sh = box.h * scale;

    ctx.strokeStyle = color;
    ctx.lineWidth   = sel ? 3 : 1.5;
    ctx.strokeRect(sx, sy, sw, sh);

    if (sel) {
      ctx.fillStyle = color + '30';
      ctx.fillRect(sx, sy, sw, sh);
    }

    // Label
    const lbl = boxLabel(box);
    ctx.fillStyle = color;
    ctx.font      = 'bold 11px monospace';
    ctx.fillText(lbl, sx + 2, sy - 3);
  });

  // Crosshair
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx.lineWidth   = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(mouseX, 0); ctx.lineTo(mouseX, canvas.height);
  ctx.moveTo(0, mouseY); ctx.lineTo(canvas.width, mouseY);
  ctx.stroke();
  ctx.setLineDash([]);
}

function renderMagnifier() {
  if (!currentImg) return;
  const half   = (MAG_SIZE / MAG_ZOOM) / 2;
  const srcX   = mouseX / scale - half;
  const srcY   = mouseY / scale - half;
  const srcW   = MAG_SIZE / MAG_ZOOM;
  const srcH   = MAG_SIZE / MAG_ZOOM;

  magCtx.clearRect(0, 0, MAG_SIZE, MAG_SIZE);
  magCtx.drawImage(currentImg, srcX, srcY, srcW, srcH, 0, 0, MAG_SIZE, MAG_SIZE);

  // Crosshair in magnifier
  magCtx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
  magCtx.lineWidth   = 1;
  magCtx.beginPath();
  magCtx.moveTo(MAG_SIZE / 2, 0); magCtx.lineTo(MAG_SIZE / 2, MAG_SIZE);
  magCtx.moveTo(0, MAG_SIZE / 2); magCtx.lineTo(MAG_SIZE, MAG_SIZE / 2);
  magCtx.stroke();

  // Bboxes in magnifier
  currentAnns.forEach(box => {
    const color = CAT_COLORS[box.category] ?? '#ffffff';
    const bx = (box.x - srcX) * MAG_ZOOM;
    const by = (box.y - srcY) * MAG_ZOOM;
    const bw = box.w * MAG_ZOOM;
    const bh = box.h * MAG_ZOOM;
    magCtx.strokeStyle = color;
    magCtx.lineWidth   = 1.5;
    magCtx.strokeRect(bx, by, bw, bh);
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function boxLabel(box) {
  if (box.category !== CORNER_CAT) return CAT_NAMES[box.category] ?? `cls${box.category}`;
  // TL / TR / BR / BL based on position relative to centroid of all corners
  const corners = currentAnns.filter(b => b.category === CORNER_CAT);
  if (corners.length < 2) return 'corner';
  const cx = corners.reduce((s, b) => s + b.x + b.w / 2, 0) / corners.length;
  const cy = corners.reduce((s, b) => s + b.y + b.h / 2, 0) / corners.length;
  const bcx = box.x + box.w / 2, bcy = box.y + box.h / 2;
  return (bcy < cy ? 'T' : 'B') + (bcx < cx ? 'L' : 'R');
}

function getCanvasCoords(e) {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function boxAt(cx, cy) {
  // Return the topmost box whose scaled rect contains (cx, cy)
  for (let i = currentAnns.length - 1; i >= 0; i--) {
    const b  = currentAnns[i];
    const sx = b.x * scale, sy = b.y * scale;
    const sw = b.w * scale, sh = b.h * scale;
    if (cx >= sx && cx <= sx + sw && cy >= sy && cy <= sy + sh) return b;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Mouse events
// ---------------------------------------------------------------------------

function onMouseDown(e) {
  if (e.button !== 0) return;
  const { x, y } = getCanvasCoords(e);
  dragStartX = x; dragStartY = y;
  hasDragged  = false;

  const hit = boxAt(x, y);
  if (hit) {
    selectedId  = hit.id;
    isDragging  = true;
    dragOrigX   = hit.x;
    dragOrigY   = hit.y;
  } else {
    isDragging = false;
  }
  render();
  renderAnnPanel();
}

function onMouseMove(e) {
  const { x, y } = getCanvasCoords(e);
  mouseX = x; mouseY = y;

  if (isDragging) {
    const dist = Math.hypot(x - dragStartX, y - dragStartY);
    if (dist > DRAG_THRESH) {
      hasDragged = true;
      const box = currentAnns.find(b => b.id === selectedId);
      if (box) {
        box.x = dragOrigX + (x - dragStartX) / scale;
        box.y = dragOrigY + (y - dragStartY) / scale;
      }
    }
  }

  render();
  renderMagnifier();
  setStatus();
}

function onMouseUp(e) {
  if (e.button !== 0) return;
  const { x, y } = getCanvasCoords(e);

  if (!isDragging && !hasDragged) {
    // Plain click on empty space → add corner
    const nCorners = currentAnns.filter(b => b.category === CORNER_CAT).length;
    if (nCorners >= MAX_CORNERS) {
      setStatus('⚠ Max 4 corners reached — right-click or Del to remove one first.');
    } else {
      const imgX = x / scale - CORNER_SIZE / 2;
      const imgY = y / scale - CORNER_SIZE / 2;
      const newId = Date.now();
      currentAnns.push({ id: newId, x: imgX, y: imgY, w: CORNER_SIZE, h: CORNER_SIZE, category: CORNER_CAT });
      selectedId = newId;
      renderAnnPanel();
    }
  }

  isDragging = false;
  hasDragged = false;
  render();
  renderAnnPanel();
}

function onRightClick(e) {
  e.preventDefault();
  const { x, y } = getCanvasCoords(e);
  const hit = boxAt(x, y);
  if (hit) removeBox(hit.id);
}

function onKeyDown(e) {
  if (e.target.matches('input, textarea')) return;
  switch (e.key) {
    case 'Delete': case 'Backspace': deleteSelected(); break;
    case 's': case 'S': if (!e.ctrlKey && !e.metaKey) saveAnnotations(); break;
    case 'n': case 'ArrowRight': nextImage(); break;
    case 'p': case 'ArrowLeft':  prevImage(); break;
    case 'Escape': selectedId = null; render(); renderAnnPanel(); break;
  }
  if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); saveAnnotations(); }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

function deleteSelected() {
  if (selectedId === null) return;
  removeBox(selectedId);
}

function removeBox(id) {
  currentAnns = currentAnns.filter(b => b.id !== id);
  if (selectedId === id) selectedId = null;
  render();
  renderAnnPanel();
  setStatus();
}

function prevImage() { if (currentIdx > 0) loadImage(currentIdx - 1); }
function nextImage() { if (currentIdx < images.length - 1) loadImage(currentIdx + 1); }

async function saveAnnotations() {
  if (images.length === 0) return;
  const img = images[currentIdx];
  await fetch(`/api/annotations/${img.id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ boxes: currentAnns, image_id: img.id, filename: img.filename }),
  });
  images[currentIdx].annotated = true;
  renderSidebar();
  setStatus('✓ Saved!', 2000);
}

// ---------------------------------------------------------------------------
// UI rendering
// ---------------------------------------------------------------------------

function renderSidebar() {
  const list = document.getElementById('image-list');
  list.innerHTML = '';
  images.forEach((img, i) => {
    const el = document.createElement('div');
    el.className = 'img-item'
      + (i === currentIdx ? ' active'    : '')
      + (img.flagged       ? ' flagged'   : '')
      + (img.annotated     ? ' annotated' : '');
    el.title = img.flag_reason || img.filename || '';
    el.innerHTML = `<span class="img-num">${i + 1}</span><span class="img-src">${img.source || ''}</span>`;
    el.onclick = () => loadImage(i);
    list.appendChild(el);
  });
  document.getElementById('image-counter').textContent = `${currentIdx + 1} / ${images.length}`;
}

function renderAnnPanel() {
  const corners = currentAnns.filter(b => b.category === CORNER_CAT);
  const others  = currentAnns.filter(b => b.category !== CORNER_CAT);

  let html = `<div class="grp-title">Board Corners (${corners.length}/${MAX_CORNERS})</div>`;

  if (corners.length === 0) {
    html += `<div style="color:#555;font-size:10px;">Click image to add corners</div>`;
  }
  corners.forEach(box => {
    const lbl = boxLabel(box);
    const sel = box.id === selectedId;
    html += `<div class="ann-item${sel ? ' sel' : ''}" onclick="selectBox(${box.id})">
      <span class="corner-lbl">${lbl}</span>
      <span class="ann-coords">(${Math.round(box.x)}, ${Math.round(box.y)})</span>
      <button class="del-btn" onclick="event.stopPropagation(); removeBox(${box.id})">✕</button>
    </div>`;
  });

  if (others.length > 0) {
    html += `<div class="grp-title" style="margin-top:10px;">Other (${others.length})</div>`;
    others.forEach(box => {
      const name = CAT_NAMES[box.category] ?? `cls${box.category}`;
      html += `<div class="ann-item"><span style="color:#aaa">${name}</span>
        <span class="ann-coords">(${Math.round(box.x)}, ${Math.round(box.y)})</span></div>`;
    });
  }

  html += `<div class="hint">
    <strong>Click</strong> empty → add corner<br>
    <strong>Drag</strong> box → move<br>
    <strong>Right-click</strong> → delete<br>
    <strong>Del</strong> → delete selected<br>
    <strong>S</strong> → save<br>
    <strong>N / P</strong> → next / prev<br>
    <strong>Esc</strong> → deselect
  </div>`;

  document.getElementById('ann-list').innerHTML = html;
}

function selectBox(id) {
  selectedId = id;
  render();
  renderAnnPanel();
}

let _statusTimer = null;
function setStatus(msg, clearAfterMs) {
  const bar = document.getElementById('status-bar');
  if (_statusTimer) { clearTimeout(_statusTimer); _statusTimer = null; }

  if (msg) {
    bar.textContent = msg;
    if (clearAfterMs) _statusTimer = setTimeout(() => setStatus(), clearAfterMs);
    return;
  }

  const img = images[currentIdx];
  if (!img) { bar.textContent = 'No images loaded'; return; }
  const nCorners = currentAnns.filter(b => b.category === CORNER_CAT).length;
  const cursor   = `cursor (${Math.round(mouseX / scale)}, ${Math.round(mouseY / scale)})`;
  bar.textContent = `[${currentIdx + 1}/${images.length}] ${img.filename}  |  corners: ${nCorners}/${MAX_CORNERS}  |  ${cursor}`;
}
