/**
 * Moku Annotator - full annotation tool for corners, black & white stones.
 *
 * Features:
 *   - Pan & zoom (scroll wheel, Space+drag)
 *   - Magnifier that follows the mouse
 *   - Drag & drop any annotation
 *   - Tool modes: corner, black stone, white stone, move
 *   - Sidebar filtering: all / flagged / corrected / not corrected
 *   - Keyboard shortcuts
 *
 * Annotations format (per image):
 *   { boxes: [{id, x, y, w, h, category}] }
 *
 * Categories: 0=black_stone, 1=white_stone, 2=board_corner
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

var CAT_BLACK    = 0;
var CAT_WHITE    = 1;
var CAT_CORNER   = 2;
var MAX_CORNERS  = 4;
var CORNER_SIZE  = 20;
var STONE_SIZE   = 20;
var MAG_SIZE     = 200;
var MAG_ZOOM     = 5;
var DRAG_THRESH  = 3;
var HIT_PADDING  = 6;

var CAT_COLORS = {
  0: '#ec4899',
  1: '#14b8a6',
  2: '#f97316',
};
var CAT_NAMES = { 0: 'black', 1: 'white', 2: 'corner' };

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

var allImages   = [];
var filteredIdx = [];
var currentFilteredPos = -1;
var currentAnns = [];
var selectedId  = null;
var activeTool  = 'corner';

var viewX = 0, viewY = 0;
var zoom  = 1.0;
var baseScale = 1.0;

var isDragging   = false;
var hasDragged   = false;
var dragStartX   = 0, dragStartY = 0;
var dragOrigX    = 0, dragOrigY  = 0;
var isPanning    = false;
var panStartX    = 0, panStartY = 0;
var panOrigViewX = 0, panOrigViewY = 0;
var spaceDown    = false;
var lockedId     = null;  // corner locked to mouse cursor

var mouseScreenX = 0, mouseScreenY = 0;
var magEnabled   = true;

var canvas, ctx, magCanvas, magCtx, currentImg;
var canvasWrap;

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

window.onload = function() {
  canvas     = document.getElementById('main-canvas');
  ctx        = canvas.getContext('2d');
  magCanvas  = document.getElementById('mag-canvas');
  magCtx     = magCanvas.getContext('2d');
  canvasWrap = document.getElementById('canvas-wrap');

  canvas.addEventListener('mousedown',   onMouseDown);
  canvas.addEventListener('mousemove',   onMouseMove);
  canvas.addEventListener('mouseup',     onMouseUp);
  canvas.addEventListener('mouseleave',  onMouseLeave);
  canvas.addEventListener('contextmenu', onRightClick);
  canvasWrap.addEventListener('wheel',   onWheel, { passive: false });

  document.addEventListener('keydown', onKeyDown);
  document.addEventListener('keyup',   onKeyUp);

  canvasWrap.addEventListener('contextmenu', function(e) { e.preventDefault(); });

  loadImageList();
};

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

async function loadImageList() {
  var resp = await fetch('/api/images');
  var data = await resp.json();
  allImages = data.images || [];
  applyFilter();
  if (filteredIdx.length > 0) loadImage(0);
}

function applyFilter() {
  var filterVal = document.getElementById('filter-select').value;
  var searchVal = document.getElementById('sidebar-search').value.toLowerCase().trim();

  filteredIdx = [];
  for (var i = 0; i < allImages.length; i++) {
    var img = allImages[i];
    if (filterVal === 'flagged' && !img.flagged) continue;
    if (filterVal === 'corrected' && !img.annotated) continue;
    if (filterVal === 'not-corrected' && img.annotated) continue;
    if (filterVal === 'flagged-not-corrected' && (!img.flagged || img.annotated)) continue;
    if (searchVal) {
      var haystack = (img.filename || '') + ' ' + (img.source || '') + ' ' + (img.flag_reason || '');
      if (haystack.toLowerCase().indexOf(searchVal) === -1) continue;
    }
    filteredIdx.push(i);
  }

  var statsEl = document.getElementById('sidebar-stats');
  var nFlagged = allImages.filter(function(i) { return i.flagged; }).length;
  var nCorrected = allImages.filter(function(i) { return i.annotated; }).length;
  statsEl.textContent = filteredIdx.length + ' shown \u00b7 ' + nFlagged + ' flagged \u00b7 ' + nCorrected + ' corrected';

  renderSidebar();

  if (currentFilteredPos >= 0 && filteredIdx.length > 0) {
    var curGlobalIdx = filteredIdx[currentFilteredPos];
    if (curGlobalIdx === undefined) {
      currentFilteredPos = Math.min(currentFilteredPos, filteredIdx.length - 1);
    }
  }
}

async function loadImage(filteredPos) {
  if (filteredPos < 0 || filteredPos >= filteredIdx.length) return;
  currentFilteredPos = filteredPos;
  var globalIdx = filteredIdx[filteredPos];
  var img = allImages[globalIdx];
  selectedId = null;
  _dirty = false;

  document.getElementById('image-name').textContent = img.filename || '';

  var annResp = await fetch('/api/annotations/' + encodeURIComponent(img.filename));
  var annData = await annResp.json();
  currentAnns = (annData.boxes || []).map(function(b) { return Object.assign({}, b); });

  var imgEl = new Image();
  imgEl.onload = function() {
    currentImg = imgEl;
    resetZoom();
    render();
    renderAnnPanel();
  };
  imgEl.src = '/api/image/' + img.filename;

  renderSidebar();
  setStatus();
}

// ---------------------------------------------------------------------------
// Pan & Zoom
// ---------------------------------------------------------------------------

function resetZoom() {
  if (!currentImg) return;
  var maxW = canvasWrap.clientWidth;
  var maxH = canvasWrap.clientHeight;
  baseScale = Math.min(maxW / currentImg.width, maxH / currentImg.height);
  zoom = 1.0;
  var scaledW = currentImg.width * baseScale;
  var scaledH = currentImg.height * baseScale;
  viewX = (maxW - scaledW) / 2;
  viewY = (maxH - scaledH) / 2;
  updateCanvasTransform();
  updateZoomIndicator();
}

function updateCanvasTransform() {
  if (!currentImg) return;
  var s = baseScale * zoom;
  canvas.width  = currentImg.width;
  canvas.height = currentImg.height;
  canvas.style.transform = 'translate(' + viewX + 'px, ' + viewY + 'px) scale(' + s + ')';
}

function updateZoomIndicator() {
  document.getElementById('zoom-indicator').textContent = Math.round(baseScale * zoom * 100) + '%';
}

function screenToImage(sx, sy) {
  var s = baseScale * zoom;
  return { x: (sx - viewX) / s, y: (sy - viewY) / s };
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function render() {
  if (!currentImg) return;
  var w = currentImg.width;
  var h = currentImg.height;

  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(currentImg, 0, 0);

  currentAnns.forEach(function(box) {
    var sel   = box.id === selectedId;
    var color = CAT_COLORS[box.category] || '#ffffff';
    var cx = box.x + box.w / 2;
    var cy = box.y + box.h / 2;

    ctx.strokeStyle = color;
    ctx.lineWidth   = sel ? 3 : 1.5;

    if (box.category === CAT_CORNER) {
      ctx.strokeRect(box.x, box.y, box.w, box.h);
      var arm = 8;
      ctx.beginPath();
      ctx.moveTo(cx - box.w/2 - arm, cy);
      ctx.lineTo(cx + box.w/2 + arm, cy);
      ctx.moveTo(cx, cy - box.h/2 - arm);
      ctx.lineTo(cx, cy + box.h/2 + arm);
      ctx.stroke();
    } else {
      var r = Math.max(box.w, box.h) / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cx, cy, 2, 0, Math.PI * 2);
      ctx.fill();
    }

    if (sel) {
      ctx.fillStyle = color + '25';
      ctx.fillRect(box.x - 2, box.y - 2, box.w + 4, box.h + 4);
    }

    var lbl = boxLabel(box);
    ctx.fillStyle = color;
    ctx.font = (sel ? 'bold 13' : 'bold 11') + 'px -apple-system, sans-serif';
    ctx.fillText(lbl, box.x + box.w + 4, box.y + box.h / 2 + 4);
  });

  if (!isPanning) {
    var im = screenToImage(mouseScreenX, mouseScreenY);
    if (im.x >= 0 && im.x <= w && im.y >= 0 && im.y <= h) {
      ctx.strokeStyle = 'rgba(255,255,255,0.2)';
      ctx.lineWidth = 0.5;
      ctx.setLineDash([6, 6]);
      ctx.beginPath();
      ctx.moveTo(im.x, 0); ctx.lineTo(im.x, h);
      ctx.moveTo(0, im.y); ctx.lineTo(w, im.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}

function renderMagnifier() {
  if (!currentImg || !magEnabled) {
    document.getElementById('mag-wrap').style.display = 'none';
    return;
  }
  var magWrap = document.getElementById('mag-wrap');
  var wrapRect = canvasWrap.getBoundingClientRect();

  var offset = 20;
  var mx = mouseScreenX + wrapRect.left + offset;
  var my = mouseScreenY + wrapRect.top - MAG_SIZE - offset;
  if (my < 0) my = mouseScreenY + wrapRect.top + offset;
  if (mx + MAG_SIZE > window.innerWidth) mx = mouseScreenX + wrapRect.left - MAG_SIZE - offset;

  magWrap.style.left = mx + 'px';
  magWrap.style.top = my + 'px';
  magWrap.style.display = 'block';

  var im = screenToImage(mouseScreenX, mouseScreenY);
  var half = (MAG_SIZE / MAG_ZOOM) / 2;
  var srcX = im.x - half;
  var srcY = im.y - half;
  var srcW = MAG_SIZE / MAG_ZOOM;

  magCtx.clearRect(0, 0, MAG_SIZE, MAG_SIZE);
  magCtx.drawImage(currentImg, srcX, srcY, srcW, srcW, 0, 0, MAG_SIZE, MAG_SIZE);

  magCtx.strokeStyle = 'rgba(255, 60, 60, 0.8)';
  magCtx.lineWidth = 1;
  magCtx.beginPath();
  magCtx.moveTo(MAG_SIZE / 2, 0); magCtx.lineTo(MAG_SIZE / 2, MAG_SIZE);
  magCtx.moveTo(0, MAG_SIZE / 2); magCtx.lineTo(MAG_SIZE, MAG_SIZE / 2);
  magCtx.stroke();

  currentAnns.forEach(function(box) {
    var color = CAT_COLORS[box.category] || '#ffffff';
    var bx = (box.x - srcX) * MAG_ZOOM;
    var by = (box.y - srcY) * MAG_ZOOM;
    var bw = box.w * MAG_ZOOM;
    var bh = box.h * MAG_ZOOM;
    magCtx.strokeStyle = color;
    magCtx.lineWidth = 1.5;
    if (box.category === CAT_CORNER) {
      magCtx.strokeRect(bx, by, bw, bh);
    } else {
      magCtx.beginPath();
      magCtx.arc(bx + bw/2, by + bh/2, Math.max(bw, bh)/2, 0, Math.PI * 2);
      magCtx.stroke();
    }
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function boxLabel(box) {
  if (box.category === CAT_BLACK) return 'B';
  if (box.category === CAT_WHITE) return 'W';
  var corners = currentAnns.filter(function(b) { return b.category === CAT_CORNER; });
  if (corners.length < 2) return 'C';
  var cx2 = corners.reduce(function(s, b) { return s + b.x + b.w / 2; }, 0) / corners.length;
  var cy2 = corners.reduce(function(s, b) { return s + b.y + b.h / 2; }, 0) / corners.length;
  var bcx = box.x + box.w / 2, bcy = box.y + box.h / 2;
  return (bcy < cy2 ? 'T' : 'B') + (bcx < cx2 ? 'L' : 'R');
}

function getCanvasCoords(e) {
  var r = canvasWrap.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function boxAtImage(imgX, imgY) {
  var s = baseScale * zoom;
  var padImg = HIT_PADDING / s;
  for (var i = currentAnns.length - 1; i >= 0; i--) {
    var b = currentAnns[i];
    var cx = b.x + b.w / 2;
    var cy = b.y + b.h / 2;
    if (b.category !== CAT_CORNER) {
      var r = Math.max(b.w, b.h) / 2 + padImg;
      if (Math.hypot(imgX - cx, imgY - cy) <= r) return b;
    } else {
      if (imgX >= b.x - padImg && imgX <= b.x + b.w + padImg &&
          imgY >= b.y - padImg && imgY <= b.y + b.h + padImg) return b;
    }
  }
  return null;
}

function getCurrentImage() {
  if (currentFilteredPos < 0 || currentFilteredPos >= filteredIdx.length) return null;
  return allImages[filteredIdx[currentFilteredPos]];
}

// ---------------------------------------------------------------------------
// Mouse events
// ---------------------------------------------------------------------------

function onMouseDown(e) {
  var coords = getCanvasCoords(e);
  var sx = coords.x, sy = coords.y;

  if (e.button === 1 || (e.button === 0 && spaceDown)) {
    e.preventDefault();
    isPanning = true;
    panStartX = e.clientX;
    panStartY = e.clientY;
    panOrigViewX = viewX;
    panOrigViewY = viewY;
    canvasWrap.style.cursor = 'grabbing';
    return;
  }

  if (e.button !== 0) return;

  var im = screenToImage(sx, sy);

  // If a corner is locked, click releases it
  if (lockedId !== null) {
    lockedId = null;
    markDirty();
    render();
    renderMagnifier();
    renderAnnPanel();
    return;
  }

  dragStartX = sx;
  dragStartY = sy;
  hasDragged = false;

  var hit = boxAtImage(im.x, im.y);
  if (hit) {
    // Click on a corner -> lock it to the mouse
    if (hit.category === CAT_CORNER) {
      selectedId = hit.id;
      lockedId = hit.id;
      // Snap center to mouse immediately
      hit.x = im.x - hit.w / 2;
      hit.y = im.y - hit.h / 2;
      markDirty();
      render();
      renderMagnifier();
      renderAnnPanel();
      return;
    }
    selectedId = hit.id;
    isDragging = true;
    dragOrigX  = hit.x;
    dragOrigY  = hit.y;
    render();
    renderAnnPanel();
  } else if (activeTool === 'move') {
    isPanning = true;
    panStartX = e.clientX;
    panStartY = e.clientY;
    panOrigViewX = viewX;
    panOrigViewY = viewY;
    canvasWrap.style.cursor = 'grabbing';
  } else {
    isDragging = false;
    selectedId = null;
    render();
    renderAnnPanel();
  }
}

function onMouseMove(e) {
  var coords = getCanvasCoords(e);
  mouseScreenX = coords.x;
  mouseScreenY = coords.y;

  if (isPanning) {
    viewX = panOrigViewX + (e.clientX - panStartX);
    viewY = panOrigViewY + (e.clientY - panStartY);
    updateCanvasTransform();
    render();
    renderMagnifier();
    return;
  }

  // Locked corner follows mouse exactly
  if (lockedId !== null) {
    var lockedBox = currentAnns.find(function(b) { return b.id === lockedId; });
    if (lockedBox) {
      var imLock = screenToImage(coords.x, coords.y);
      lockedBox.x = imLock.x - lockedBox.w / 2;
      lockedBox.y = imLock.y - lockedBox.h / 2;
    }
  }

  if (isDragging && selectedId !== null) {
    var dist = Math.hypot(coords.x - dragStartX, coords.y - dragStartY);
    if (dist > DRAG_THRESH) {
      hasDragged = true;
      var box = currentAnns.find(function(b) { return b.id === selectedId; });
      if (box) {
        var s = baseScale * zoom;
        box.x = dragOrigX + (coords.x - dragStartX) / s;
        box.y = dragOrigY + (coords.y - dragStartY) / s;
        markDirty();
      }
    }
  }

  if (spaceDown) {
    canvasWrap.style.cursor = 'grab';
  } else if (activeTool === 'move') {
    var im2 = screenToImage(coords.x, coords.y);
    canvasWrap.style.cursor = boxAtImage(im2.x, im2.y) ? 'move' : 'grab';
  } else {
    var im3 = screenToImage(coords.x, coords.y);
    canvasWrap.style.cursor = boxAtImage(im3.x, im3.y) ? 'move' : 'crosshair';
  }

  render();
  renderMagnifier();
  setStatus();
}

function onMouseUp(e) {
  if (isPanning) {
    isPanning = false;
    canvasWrap.style.cursor = spaceDown ? 'grab' : (activeTool === 'move' ? 'grab' : 'crosshair');
    return;
  }

  if (e.button !== 0) return;
  var coords = getCanvasCoords(e);

  if (!isDragging && !hasDragged && activeTool !== 'move') {
    var im = screenToImage(coords.x, coords.y);
    var hit = boxAtImage(im.x, im.y);
    if (hit) {
      selectedId = hit.id;
    } else {
      addAnnotation(im.x, im.y);
    }
  }

  isDragging = false;
  hasDragged = false;
  render();
  renderMagnifier();
  renderAnnPanel();
}

function onMouseLeave() {
  document.getElementById('mag-wrap').style.display = 'none';
  isPanning = false;
  isDragging = false;
  // Don't clear lockedId on leave — keep corner locked
}

function onRightClick(e) {
  e.preventDefault();
  var coords = getCanvasCoords(e);
  var im = screenToImage(coords.x, coords.y);
  var hit = boxAtImage(im.x, im.y);
  if (hit) removeBox(hit.id);
}

function onWheel(e) {
  e.preventDefault();
  var coords = getCanvasCoords(e);
  var imBefore = screenToImage(coords.x, coords.y);

  var delta = -e.deltaY * 0.001;
  zoom = Math.max(0.1, Math.min(50, zoom * (1 + delta)));

  var s = baseScale * zoom;
  viewX = coords.x - imBefore.x * s;
  viewY = coords.y - imBefore.y * s;

  updateCanvasTransform();
  updateZoomIndicator();
  render();
  renderMagnifier();
}

// ---------------------------------------------------------------------------
// Keyboard
// ---------------------------------------------------------------------------

function onKeyDown(e) {
  if (e.target.matches('input, textarea, select')) return;

  if (e.key === ' ') {
    e.preventDefault();
    spaceDown = true;
    if (!isPanning) canvasWrap.style.cursor = 'grab';
    return;
  }

  switch (e.key) {
    case 'Delete': case 'Backspace': deleteSelected(); break;
    case 'ArrowRight': if (e.shiftKey) nextFiltered(); else nextImage(); break;
    case 'ArrowLeft':  if (e.shiftKey) prevFiltered(); else prevImage(); break;
    case 'Escape':
      if (lockedId !== null) { lockedId = null; }
      selectedId = null; render(); renderAnnPanel(); break;
    case '1': setTool('corner'); break;
    case '2': setTool('black'); break;
    case '3': setTool('white'); break;
    case 'v': case 'V': setTool('move'); break;
    case '0': resetZoom(); render(); break;
    case 'm': case 'M':
      magEnabled = !magEnabled;
      if (!magEnabled) document.getElementById('mag-wrap').style.display = 'none';
      break;
    case 's': case 'S':
      if (e.ctrlKey || e.metaKey) { e.preventDefault(); saveAnnotations(); }
      break;
  }
}

function onKeyUp(e) {
  if (e.key === ' ') {
    spaceDown = false;
    if (!isPanning) canvasWrap.style.cursor = activeTool === 'move' ? 'grab' : 'crosshair';
  }
}

// ---------------------------------------------------------------------------
// Tool mode
// ---------------------------------------------------------------------------

function setTool(tool) {
  activeTool = tool;
  document.querySelectorAll('.tool-btn').forEach(function(el) { el.classList.remove('active'); });
  var btn = document.getElementById('tool-' + tool);
  if (btn) btn.classList.add('active');
  canvasWrap.style.cursor = tool === 'move' ? 'grab' : 'crosshair';
}

// ---------------------------------------------------------------------------
// Annotation actions
// ---------------------------------------------------------------------------

function addAnnotation(imgX, imgY) {
  var cat, size;
  if (activeTool === 'corner') {
    if (currentAnns.filter(function(b) { return b.category === CAT_CORNER; }).length >= MAX_CORNERS) {
      showToast('Max 4 corners - delete one first', 'error');
      return;
    }
    cat = CAT_CORNER;
    size = CORNER_SIZE;
  } else if (activeTool === 'black') {
    cat = CAT_BLACK;
    size = STONE_SIZE;
  } else if (activeTool === 'white') {
    cat = CAT_WHITE;
    size = STONE_SIZE;
  } else {
    return;
  }

  var newId = Date.now() + Math.floor(Math.random() * 1000);
  currentAnns.push({
    id: newId,
    x: imgX - size / 2,
    y: imgY - size / 2,
    w: size,
    h: size,
    category: cat,
  });
  selectedId = newId;
  if (cat === CAT_CORNER) lockedId = newId;  // auto-lock new corners
  markDirty();
  render();
  renderMagnifier();
  renderAnnPanel();
}

function deleteSelected() {
  if (selectedId === null) return;
  removeBox(selectedId);
}

function removeBox(id) {
  currentAnns = currentAnns.filter(function(b) { return b.id !== id; });
  if (selectedId === id) selectedId = null;
  markDirty();
  render();
  renderMagnifier();
  renderAnnPanel();
  setStatus();
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

var _dirty = false;

function markDirty() { _dirty = true; }

async function autoSaveIfDirty() {
  if (_dirty) {
    await saveAnnotations();
    _dirty = false;
  }
}

async function prevImage() {
  if (currentFilteredPos > 0) {
    await autoSaveIfDirty();
    loadImage(currentFilteredPos - 1);
  }
}
async function nextImage() {
  if (currentFilteredPos < filteredIdx.length - 1) {
    await autoSaveIfDirty();
    loadImage(currentFilteredPos + 1);
  }
}

async function prevFiltered() {
  for (var i = currentFilteredPos - 1; i >= 0; i--) {
    if (allImages[filteredIdx[i]].flagged) { await autoSaveIfDirty(); loadImage(i); return; }
  }
  showToast('No more flagged images before', 'error');
}

async function nextFiltered() {
  for (var i = currentFilteredPos + 1; i < filteredIdx.length; i++) {
    if (allImages[filteredIdx[i]].flagged) { await autoSaveIfDirty(); loadImage(i); return; }
  }
  showToast('No more flagged images after', 'error');
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

async function saveAnnotations() {
  var img = getCurrentImage();
  if (!img) return;
  await fetch('/api/annotations/' + encodeURIComponent(img.filename), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ boxes: currentAnns, filename: img.filename }),
  });
  _dirty = false;
  var gIdx = filteredIdx[currentFilteredPos];
  allImages[gIdx].annotated = true;
  allImages[gIdx].corner_count = currentAnns.filter(function(b) { return b.category === CAT_CORNER; }).length;
  renderSidebar();
  showToast('Saved!');
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

function renderSidebar() {
  var list = document.getElementById('image-list');
  list.innerHTML = '';

  filteredIdx.forEach(function(gIdx, fPos) {
    var img = allImages[gIdx];
    var el = document.createElement('div');
    el.className = 'img-item'
      + (fPos === currentFilteredPos ? ' active' : '')
      + (img.flagged    ? ' flagged'   : '')
      + (img.annotated  ? ' annotated' : '');
    el.title = img.flag_reason || img.filename || '';

    var badges = '';
    if (img.flagged) badges += '<span class="img-badge flag"></span>';
    if (img.annotated) badges += '<span class="img-badge ok"></span>';

    var cc = img.corner_count || 0;
    var cornerBadge = img.annotated
      ? '<span class="corner-badge ' + (cc === 4 ? 'ok' : 'warn') + '">' + cc + '</span>'
      : '';

    el.innerHTML = '<span class="img-num">' + (gIdx + 1) + '</span>' + badges +
      '<span class="img-src">' + (img.source || img.filename || '') + '</span>' + cornerBadge;
    el.onclick = function() { autoSaveIfDirty().then(function() { loadImage(fPos); }); };
    list.appendChild(el);

    if (fPos === currentFilteredPos) {
      requestAnimationFrame(function() { el.scrollIntoView({ block: 'nearest' }); });
    }
  });

  document.getElementById('image-counter').textContent =
    (currentFilteredPos + 1) + ' / ' + filteredIdx.length;
}

// ---------------------------------------------------------------------------
// Annotation panel
// ---------------------------------------------------------------------------

function renderAnnPanel() {
  var corners = currentAnns.filter(function(b) { return b.category === CAT_CORNER; });
  var blacks  = currentAnns.filter(function(b) { return b.category === CAT_BLACK; });
  var whites  = currentAnns.filter(function(b) { return b.category === CAT_WHITE; });

  var html = '';

  html += '<div class="grp-title">Corners (' + corners.length + '/' + MAX_CORNERS + ')</div>';
  if (corners.length === 0) {
    html += '<div style="color:var(--text-muted);font-size:12px;padding:2px 6px;">Select Corner tool & click</div>';
  }
  corners.forEach(function(box) {
    var lbl = boxLabel(box);
    var sel = box.id === selectedId;
    var cx = Math.round(box.x + box.w / 2);
    var cy = Math.round(box.y + box.h / 2);
    html += '<div class="ann-item' + (sel ? ' sel' : '') + '" onclick="selectBox(' + box.id + ')">' +
      '<span class="corner-lbl">' + lbl + '</span>' +
      '<span class="ann-coords">(' + cx + ', ' + cy + ')</span>' +
      '<button class="del-btn" onclick="event.stopPropagation(); removeBox(' + box.id + ')" title="Delete">\u2715</button>' +
      '</div>';
  });

  if (blacks.length > 0 || activeTool === 'black') {
    html += '<div class="grp-title">Black Stones (' + blacks.length + ')</div>';
    blacks.forEach(function(box) {
      var sel = box.id === selectedId;
      var cx = Math.round(box.x + box.w / 2);
      var cy = Math.round(box.y + box.h / 2);
      html += '<div class="ann-item' + (sel ? ' sel' : '') + '" onclick="selectBox(' + box.id + ')">' +
        '<span class="stone-lbl black">B</span>' +
        '<span class="ann-coords">(' + cx + ', ' + cy + ')</span>' +
        '<button class="del-btn" onclick="event.stopPropagation(); removeBox(' + box.id + ')" title="Delete">\u2715</button>' +
        '</div>';
    });
  }

  if (whites.length > 0 || activeTool === 'white') {
    html += '<div class="grp-title">White Stones (' + whites.length + ')</div>';
    whites.forEach(function(box) {
      var sel = box.id === selectedId;
      var cx = Math.round(box.x + box.w / 2);
      var cy = Math.round(box.y + box.h / 2);
      html += '<div class="ann-item' + (sel ? ' sel' : '') + '" onclick="selectBox(' + box.id + ')">' +
        '<span class="stone-lbl white">W</span>' +
        '<span class="ann-coords">(' + cx + ', ' + cy + ')</span>' +
        '<button class="del-btn" onclick="event.stopPropagation(); removeBox(' + box.id + ')" title="Delete">\u2715</button>' +
        '</div>';
    });
  }

  var img = getCurrentImage();
  if (img) {
    html += '<div class="grp-title" style="margin-top:8px;">Image Info</div>';
    html += '<div style="font-size:12px;color:var(--text-muted);padding:2px 6px;line-height:1.6;">';
    html += 'Source: ' + (img.source || '-') + '<br>';
    if (img.flagged) html += '\u26a0 ' + (img.flag_reason || 'Flagged') + '<br>';
    if (img.width && img.height) html += 'Size: ' + img.width + '\u00d7' + img.height + '<br>';
    html += '</div>';
  }

  document.getElementById('ann-list').innerHTML = html;
}

function selectBox(id) {
  selectedId = id;
  render();
  renderAnnPanel();
}

// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------

var _toastTimer = null;
function showToast(msg, type) {
  var el = document.getElementById('toast');
  el.textContent = msg;
  el.className = type === 'error' ? 'show error' : 'show';
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(function() { el.className = ''; _toastTimer = null; }, 1800);
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

function setStatus() {
  var bar = document.querySelector('#status-bar .status-text');
  var img = getCurrentImage();
  if (!img) { bar.textContent = 'No images loaded'; return; }

  var im = screenToImage(mouseScreenX, mouseScreenY);
  var nCorners = currentAnns.filter(function(b) { return b.category === CAT_CORNER; }).length;
  var nBlack   = currentAnns.filter(function(b) { return b.category === CAT_BLACK; }).length;
  var nWhite   = currentAnns.filter(function(b) { return b.category === CAT_WHITE; }).length;
  var tool     = activeTool.charAt(0).toUpperCase() + activeTool.slice(1);

  bar.textContent = '[' + (currentFilteredPos + 1) + '/' + filteredIdx.length + '] ' + img.filename +
    '  \u00b7  Tool: ' + tool +
    '  \u00b7  Corners: ' + nCorners +
    '  \u00b7  B: ' + nBlack + '  W: ' + nWhite +
    '  \u00b7  (' + Math.round(im.x) + ', ' + Math.round(im.y) + ')';
}
