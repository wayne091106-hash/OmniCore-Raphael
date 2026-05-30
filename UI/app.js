/* ════════════════════════════════════════════════════════════════════════
   app.js — Raphael UI 邏輯
   ════════════════════════════════════════════════════════════════════════
   設計核心：所有畫面更新都經由 applyMessage(channel, payload)。
   channel 字串與後端 bridge.py 的 Channel enum 一一對應。

   接 WebSocket 時，只要：
       ws.onmessage = e => { const m = JSON.parse(e.data); applyMessage(m.channel, m.payload); }
   即可驅動整個畫面。目前用 mock 資料展示版面。
   ════════════════════════════════════════════════════════════════════════ */

const $ = id => document.getElementById(id);

// ── 內聯 SVG 圖示（不依賴外部 webfont，file:// 直接可用）──────
const ICONS = {
  'adjustments': '<path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h7M15 18h5"/><circle cx="16" cy="6" r="2"/><circle cx="8" cy="12" r="2"/><circle cx="13" cy="18" r="2"/>',
  'x': '<path d="M5 5l14 14M19 5L5 19"/>',
  'bulb': '<path d="M9 18h6M10 21h4"/><path d="M12 3a6 6 0 0 0-3 11c.5.4 1 1 1 2h4c0-1 .5-1.6 1-2a6 6 0 0 0-3-11z"/>',
  'microphone': '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M6 11a6 6 0 0 0 12 0M12 17v4"/>',
  'activity': '<path d="M3 12h4l3 8 4-16 3 8h4"/>',
  'video-off': '<path d="M3 3l18 18M10 6h6a2 2 0 0 1 2 2v3l3-2v8M14 14v2a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2"/>',
  'layout-sidebar-right-collapse': '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16M10 10l-2 2 2 2"/>',
  'layout-sidebar-right-expand': '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16M8 10l2 2-2 2"/>',
  'tool': '<path d="M7 10l-4 4 3 3 4-4M14 7l3-3 3 3-3 3M7 10l7 7M14 7L7 14"/>',
  'check': '<path d="M5 12l5 5L20 7"/>',
  'plug': '<path d="M9 2v6M15 2v6M7 8h10v3a5 5 0 0 1-10 0zM12 16v6"/>',
  'chevron-down': '<path d="M6 9l6 6 6-6"/>',
  'video': '<path d="M3 7a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1zM15 10l6-3v10l-6-3"/>',
  'adjustments-horizontal': '<path d="M4 6h8M16 6h4M4 12h4M12 12h8M4 18h12M18 18h2"/><circle cx="14" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="16" cy="18" r="2"/>',
  'paperclip': '<path d="M21.4 11.6l-8.5 8.5a6 6 0 0 1-8.5-8.5l9.2-9.2a4 4 0 0 1 5.7 5.7l-9.2 9.2a2 2 0 1 1-2.8-2.8l8.5-8.5"/>',
  'file': '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
  'download': '<path d="M12 3v12M7 10l5 5 5-5"/><path d="M5 21h14"/>',
};
function svgIcon(name) {
  const inner = ICONS[name] || '';
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${inner}</svg>`;
}
function renderIcons(root = document) {
  root.querySelectorAll('.ic[data-i]').forEach(el => {
    if (el.dataset.rendered) return;
    el.innerHTML = svgIcon(el.dataset.i);
    el.dataset.rendered = '1';
  });
}
function makeIcon(name) {
  const s = document.createElement('span');
  s.className = 'ic'; s.dataset.i = name; s.innerHTML = svgIcon(name); s.dataset.rendered = '1';
  return s;
}
renderIcons();

// ── 狀態統計 ──────────────────────────────────────────────
const stat = { turns: 0, tokens: 0, events: 0, tools: 0 };
function bumpStat(k, n = 1) { stat[k] += n; $('st-' + k).textContent = stat[k]; }
function setTokens(n) { stat.tokens = n; $('st-tokens').textContent = n; $('tk-count').textContent = n + ' tk'; }

// ── 對話：串流轉錄需要「續寫同一顆泡泡」，用 ref 暫存 ──────
let curRaeBubble = null;   // Raphael 回應（output transcript）
let curUserBubble = null;  // 使用者 ASR（input transcript）
const sensorStats = { gate: 'WATCHING', cap: 0, sem: 0, drift: 0, objects: 0, flow: 0, roi: 0, triggered: false };
let latestVisionBoxes = [];
let latestIdentityBoxes = [];
let currentFrameRenderedFeedback = false;
const FEATURE_TOGGLE_IDS = [
  'tog-vision-gate',
  'tog-vision-overlay',
  'tog-vision-proactive',
  'tog-visual-identity',
  'tog-voice-identity',
  'tog-mouth-sync',
  'tog-advanced-voice-gate',
  'tog-full-duplex',
  'tog-computer-tools',
  'tog-voice-interrupt',
];

function convEl() { return $('conv'); }

function addLabel(kind, text) {
  const l = document.createElement('div');
  l.className = 'msg-label ' + kind;
  l.textContent = text;
  convEl().appendChild(l);
  return l;
}
function addBubble(kind, text) {
  const b = document.createElement('div');
  b.className = 'bubble ' + kind;
  b.textContent = text;
  convEl().appendChild(b);
  scrollConv();
  return b;
}
function scrollConv() { const c = convEl(); c.scrollTop = c.scrollHeight; }

function formatFps(n) {
  const v = Number(n || 0);
  return v >= 10 ? Math.round(v).toString() : v.toFixed(1);
}

function updateCameraImage(src) {
  const cam = $('cam');
  let img = cam.querySelector('img');
  if (!img) {
    img = document.createElement('img');
    cam.prepend(img);
    cam.querySelector('.ph')?.remove();
  }
  img.onload = () => {
    if (img.naturalWidth && img.naturalHeight) {
      cam.style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`;
    }
    drawVisionOverlay();
  };
  img.src = src;
}

function ensureVisionOverlay() {
  const cam = $('cam');
  let layer = cam.querySelector('.vision-layer');
  if (!layer) {
    layer = document.createElement('div');
    layer.className = 'vision-layer';
    cam.appendChild(layer);
  }
  return layer;
}

function drawVisionOverlay() {
  const cam = $('cam');
  const img = cam.querySelector('img');
  const layer = ensureVisionOverlay();
  const rect = cam.getBoundingClientRect();
  layer.replaceChildren();
  if (!featureEnabled('vision_overlay')) return;
  const boxes = [
    ...(currentFrameRenderedFeedback ? [] : latestVisionBoxes),
    ...latestIdentityBoxes,
  ];
  if (!img || !img.naturalWidth || !img.naturalHeight || !boxes.length) return;

  const imgRatio = img.naturalWidth / img.naturalHeight;
  const boxRatio = rect.width / rect.height;
  let drawW = rect.width;
  let drawH = rect.height;
  let offX = 0;
  let offY = 0;
  if (boxRatio > imgRatio) {
    drawH = rect.height;
    drawW = drawH * imgRatio;
    offX = (rect.width - drawW) / 2;
  } else {
    drawW = rect.width;
    drawH = drawW / imgRatio;
    offY = (rect.height - drawH) / 2;
  }
  for (const box of boxes) {
    const x = offX + Number(box.x1 || 0) * drawW;
    const y = offY + Number(box.y1 || 0) * drawH;
    const w = Math.max(1, (Number(box.x2 || 0) - Number(box.x1 || 0)) * drawW);
    const h = Math.max(1, (Number(box.y2 || 0) - Number(box.y1 || 0)) * drawH);
    if (!Number.isFinite(x + y + w + h) || w < 3 || h < 3) continue;
    const kind = String(box.kind || 'semantic').replace('_', '-');
    const label = box.label || (kind === 'fast' ? 'FAST' : 'GLOBAL');
    const el = document.createElement('div');
    el.className = 'vision-box ' + kind;
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
    el.style.width = `${w}px`;
    el.style.height = `${h}px`;
    const tag = document.createElement('div');
    tag.className = 'vision-tag';
    tag.textContent = label;
    if (y < 24) tag.style.top = '-1px';
    el.appendChild(tag);
    layer.appendChild(el);
  }
}

window.addEventListener('resize', drawVisionOverlay);

// ════════════════════════════════════════════════════════════
//  applyMessage —— 唯一的畫面更新入口（對齊 bridge Channel）
// ════════════════════════════════════════════════════════════
function applyMessage(channel, payload) {
  switch (channel) {

    // ── 感知資料顯示 ──────────────────────────────────────
    case 'sensor_view': {
      // payload: {jpeg(b64)|null, vad_speaking, gate_state, drift, fps_capture, fps_semantic}
      if (payload.rendered_feedback != null) {
        currentFrameRenderedFeedback = !!payload.rendered_feedback;
      }
      if (payload.jpeg) {
        updateCameraImage('data:image/jpeg;base64,' + payload.jpeg);
      }
      if (Array.isArray(payload.feedback_boxes)) {
        latestVisionBoxes = payload.feedback_boxes;
        sensorStats.objects = latestVisionBoxes.length;
        drawVisionOverlay();
      }
      if (Array.isArray(payload.identity_boxes)) {
        latestIdentityBoxes = payload.identity_boxes;
        drawVisionOverlay();
      }
      if (payload.triggered != null) {
        $('cam')?.classList.toggle('triggered', !!payload.triggered);
        sensorStats.triggered = !!payload.triggered;
      }
      if (payload.gate_state != null) sensorStats.gate = payload.gate_state;
      if (payload.fps_capture != null) sensorStats.cap = Number(payload.fps_capture || 0);
      if (payload.fps_semantic != null) sensorStats.sem = Number(payload.fps_semantic || 0);
      if (payload.drift != null) sensorStats.drift = Number(payload.drift || 0);
      if (payload.flow_pixels != null) sensorStats.flow = Number(payload.flow_pixels || 0);
      if (payload.roi_drift != null) sensorStats.roi = Number(payload.roi_drift || 0);
      if (payload.objects != null && !Array.isArray(payload.feedback_boxes)) sensorStats.objects = Number(payload.objects || 0);
      const gate = sensorStats.gate;
      const cap = sensorStats.cap;
      const sem = sensorStats.sem;
      const driftN = sensorStats.drift;
      const drift = driftN.toFixed(2);
      $('gate-state').textContent = gate;
      $('fps-cap').textContent = '擷取 ' + formatFps(cap);
      $('fps-sem').textContent = '語義 ' + formatFps(sem);
      $('drift-text').textContent = '漂移 ' + drift;
      if (payload.vad_speaking != null) {
        $('vad-text').textContent = payload.vad_speaking ? '偵測到人聲' : '待機';
      }
      // 數值橫條圖表
      const setBar=(fill,val,txt,pct)=>{ const f=$(fill); if(f) f.style.width=Math.max(0,Math.min(100,pct))+'%'; const t=$(txt); if(t) t.textContent=val; };
      setBar('sb-cap', formatFps(cap), 'sb-cap-v', (cap/30)*100);
      setBar('sb-sem', formatFps(sem), 'sb-sem-v', (sem/10)*100);
      setBar('sb-drift', drift, 'sb-drift-v', driftN*200);
      setBar('sb-flow', Math.round(sensorStats.flow), 'sb-flow-v', Math.min(100, sensorStats.flow * 3));
      setBar('sb-roi', sensorStats.roi.toFixed(2), 'sb-roi-v', sensorStats.roi * 200);
      const objN = Number(sensorStats.objects ?? 0);
      setBar('sb-obj', objN, 'sb-obj-v', objN*10);
      if ($('rs-gate')) $('rs-gate').textContent = gate;
      break;
    }

    // ── VAD 音量（即時，給底部 meter）─────────────────────
    case 'vad_event': {
      // payload: {speaking, probability}
      const pct = Math.round((payload.probability ?? 0) * 100);
      const fill = $('vad-fill');
      fill.style.width = Math.max(4, pct) + '%';
      fill.classList.toggle('speaking', !!payload.speaking);
      $('mic-state')?.classList.toggle('speaking', !!payload.speaking);
      const speaker = payload.speaker_label || '';
      const reason = payload.gate_reason || '';
      if (payload.speaking && payload.listening === false) {
        $('vad-text').textContent = '背景聲';
      } else if (payload.speaking && payload.listening === true) {
        if (speaker) {
          $('vad-text').textContent = `聆聽 ${speaker}`;
        } else if (reason === 'mouth_sync') {
          $('vad-text').textContent = '視聽同步';
        } else {
          $('vad-text').textContent = '聆聽中';
        }
      } else {
        $('vad-text').textContent = payload.speaking ? '偵測到人聲' : '待機';
      }
      break;
    }

    // ── 感知事件日誌（左欄下方捲動區）─────────────────────
    case 'vision_event': {
      // payload: {jpeg, reason, detail}  → 這裡只用 reason/detail 做日誌
      addEventLog(payload.reason, payload.detail);
      bumpStat('events');
      break;
    }

    // ── 使用者語音 ASR（你要顯示的轉錄）───────────────────
    case 'transcript_in': {
      if (payload.done && !payload.text && !curUserBubble) {
        break; // 忽略沒有文字且沒有現存泡泡的空結束事件（避免打字模式出現空白泡泡打斷對話）
      }
      if (!curUserBubble) { addLabel('user', '你'); curUserBubble = addBubble('user', ''); }
      curUserBubble.textContent += payload.text;
      if (payload.done) curUserBubble = null;
      scrollConv();
      break;
    }

    // ── Raphael 回應字幕 ──────────────────────────────────
    case 'transcript_out': {
      if (payload.done) { curRaeBubble = null; bumpStat('turns'); break; }
      if (!curRaeBubble) { addLabel('rae', '◉ RAPHAEL'); curRaeBubble = addBubble('rae', ''); }
      curRaeBubble.textContent += payload.text;
      scrollConv();
      break;
    }

    // ── 工具呼叫（中欄內嵌 + 右欄獨立區，雙重呈現）────────
    case 'tool_call': {
      // payload: {name, args}
      addToolBubble(payload.name, payload.args, null);   // 對話流內嵌
      addToolCard(payload.name, payload.args, null);     // 右欄獨立
      bumpStat('tools');
      break;
    }
    case 'tool_result': {
      // payload: {name, result, _meta: {ok, tool, duration_ms}}
      addToolBubble(payload.name, null, payload.result); // 對話流內嵌
      updateLastToolCard(payload);                       // 右欄補上結果 + _meta
      break;
    }
    case 'task_voice': {
      speakTaskVoice(payload?.text || payload);
      break;
    }

    // ── 記憶寫入（log_observation → 右欄記憶區）───────────
    case 'memory_write': {
      // payload: {memory, category, importance}
      addMemCard(payload);
      break;
    }

    // ── 主動開口提示 ──────────────────────────────────────
    case 'proactive': {
      const banner = $('proactive');
      banner.style.display = 'flex';
      $('proactive-text').textContent = '主動開口：' + reasonText(payload);
      break;
    }

    // ── AI 音訊輸出（24kHz PCM16 base64）─────────────────
    case 'audio_out': {
      if (payload && payload.data) enqueueAudio(payload.data);
      break;
    }

    // ── 被打斷 ────────────────────────────────────────────
    case 'interrupted': {
      curRaeBubble = null;
      playQueue = [];
      isPlaying = false;
      break;
    }

    // ── 連線狀態 ──────────────────────────────────────────
    case 'status': {
      setConn(payload);
      break;
    }

    case 'usage': {
      const total = payload?.totalTokenCount ?? payload?.total_tokens ?? payload?.tokens;
      if (total != null) setTokens(total);
      break;
    }

    case 'memory_accounts': {
      updateMemoryAccounts(payload);
      break;
    }

    case 'pong': {
      break;
    }

    // ── 系統訊息 / 錯誤 ───────────────────────────────────
    case 'error': {
      addLabel('sys', 'SYSTEM');
      addBubble('sys', '⚠ ' + payload);
      break;
    }
  }
}

// ── 小工具函式 ────────────────────────────────────────────
function reasonText(r) {
  if (r && typeof r === 'object') r = r.type || r.reason || r.detail || JSON.stringify(r);
  if (typeof r !== 'string') return String(r ?? '');
  const map = { 'vision:semantic': '偵測到場景語義變化', 'vision:object': '偵測到物件變化',
                'vision:object_motion': '偵測到物體進入或移動',
                'vision:motion': '偵測到物體進入或移動',
                'vision:fast_motion': '偵測到快速移動', 'vision:fast_burst': '偵測到快速移動' };
  return map[r] ?? r;
}

const EVT_MAP = {
  object_appeared: { cls: 'appeared', label: '物件出現' },
  object_left:     { cls: 'leave',    label: '物件離開' },
  person_enter:    { cls: 'enter',    label: '人員進入' },
  person_leave:    { cls: 'leave',    label: '人員離開' },
  person_moved:    { cls: 'moved',    label: '人物移動' },
  semantic:        { cls: 'appeared', label: '語義變化' },
  fast_motion:     { cls: 'moved',    label: '快速移動' },
  identity_saved:  { cls: 'enter',    label: '已記住身份' },
  identity_seen:   { cls: 'appeared', label: '辨識身份' },
  voice_identity_saved: { cls: 'enter', label: '已記住聲音' },
  armed:           { cls: 'enter',    label: '準備觸發' },
  watching:        { cls: '',         label: '恢復監看' },
};
function formatVisionDetail(detail) {
  return String(detail ?? '')
    .replace(/^armed/, '已鎖定')
    .replace(/^cooldown end -> WATCHING \(anchor reset\)$/i, '冷卻結束，已重設參考畫面')
    .replace(/^FIRE/i, '觸發')
    .replace(/fast_burst/g, '快速移動')
    .replace(/semantic/g, '語義變化')
    .replace(/object/g, '物件')
    .replace(/drift=/g, '漂移=')
    .replace(/sharp=/g, '清晰度=');
}
function addEventLog(reason, detail) {
  const cfg = EVT_MAP[reason] ?? { cls: '', label: reason };
  const e = document.createElement('div');
  e.className = 'evt fresh ' + cfg.cls;
  const k = document.createElement('span');
  k.className = 'k';
  k.textContent = cfg.label;
  const d = document.createElement('span');
  d.className = 'd';
  d.textContent = formatVisionDetail(detail);
  e.append(k, d);
  const log = $('evt-log');
  log.prepend(e);
  setTimeout(() => e.classList.remove('fresh'), 400);
  while (log.children.length > 60) log.lastChild.remove();
}

function addToolBubble(name, args, result) {
  if (args !== null) {
    const b = addBubble('tool', '');
    b.innerHTML = `<span class="ic" style="font-size:14px">${svgIcon('tool')}</span><span>${friendlyToolName(name)}：<code>${formatToolArgs(name, args)}</code></span>`;
  } else {
    const b = addBubble('tool result', '');
    b.innerHTML = `<span class="ic" style="font-size:14px">${svgIcon('check')}</span><span>${friendlyToolName(name)} → <code>${formatToolResult(name, result)}</code>${fileLinksHtml(result)}</span>`;
  }
}

let pendingToolCards = [];
const TOOL_NAMES_ZH = {
  'delegate_tool_task': '委派工具任務',
  'search_memories': '搜尋記憶',
  'filtered_search_memories': '篩選記憶',
  'get_all_memories': '列出記憶',
  'get_memory_stats': '記憶統計',
  'store_memory': '寫入記憶',
  'store_visual_identity': '寫入圖像身份',
  'get_visual_identity_stats': '圖像身份統計',
  'delete_visual_identity': '刪除圖像身份',
  'store_voice_identity': '寫入聲紋身份',
  'get_voice_identity_stats': '聲紋身份統計',
  'delete_voice_identity': '刪除聲紋身份',
  'minimax::gmail_read': '讀取 Gmail',
  'minimax::gmail_list': '列出 Gmail',
  'minimax::gmail_send': '寄送 Gmail',
  'minimax::web_search': '網路搜尋',
  'minimax::web_image_search': '圖片搜尋',
  'minimax::download_image': '下載圖片',
  'minimax::weather_get': '查詢天氣',
  'minimax::calculator': '計算',
  'minimax::site_memory_search': '搜尋網站記憶',
  'minimax::site_memory_remember': '記住網站入口',
  'minimax::site_memory_mark_failure': '記住失敗網址',
  'minimax::website_find': '尋找網站入口',
  'minimax::read_file': '讀取檔案',
  'minimax::write_file': '寫入檔案',
  'minimax::read_file_range': '讀取檔案片段',
  'minimax::list_directory': '列出資料夾',
  'minimax::list_directory_recursive': '遞迴列出資料夾',
  'minimax::search_files': '搜尋檔案',
  'minimax::search_text': '搜尋文字',
  'minimax::replace_in_file': '取代檔案文字',
  'minimax::path_exists': '檢查路徑',
  'minimax::make_directory': '建立資料夾',
  'minimax::copy_file': '複製檔案',
  'minimax::move_file': '移動檔案',
  'minimax::delete_file': '刪除檔案',
  'minimax::file_hash': '計算檔案雜湊',
  'minimax::detect_file_type': '辨識檔案類型',
  'minimax::zip_create': '建立壓縮檔',
  'minimax::zip_extract': '解壓縮',
  'minimax::download_file': '下載檔案',
  'minimax::http_request': 'HTTP 請求',
  'minimax::json_parse': '解析 JSON',
  'minimax::json_query': '查詢 JSON',
  'minimax::csv_read': '讀取 CSV',
  'minimax::csv_write': '寫入 CSV',
  'minimax::sqlite_query': 'SQLite 查詢',
  'minimax::sqlite_schema': 'SQLite 結構',
  'minimax::git_status': 'Git 狀態',
  'minimax::git_log': 'Git 紀錄',
  'minimax::git_diff': 'Git 差異',
  'minimax::git_show': 'Git 檢視',
  'minimax::git_branch': 'Git 分支',
  'minimax::system_info': '系統資訊',
  'minimax::disk_usage': '磁碟容量',
  'minimax::network_ping': 'Ping 網路',
  'minimax::dns_lookup': 'DNS 查詢',
  'minimax::port_check': '連接埠檢查',
  'minimax::python_run': '執行 Python',
  'minimax::regex_extract': '正規擷取',
  'minimax::text_summarize_basic': '文字摘要',
  'minimax::image_info': '圖片資訊',
  'minimax::image_resize': '調整圖片',
  'minimax::pdf_extract_text': '擷取 PDF 文字',
  'minimax::docx_extract_text': '擷取 Word 文字',
  'minimax::xlsx_read_sheet': '讀取 Excel',
  'minimax::xlsx_write_sheet': '寫入 Excel',
  'minimax::base64_encode_text': 'Base64 編碼',
  'minimax::base64_decode_text': 'Base64 解碼',
  'minimax::url_encode': 'URL 編碼',
  'minimax::url_decode': 'URL 解碼',
  'minimax::uuid_generate': '產生 UUID',
  'minimax::timestamp_convert': '時間轉換',
  'minimax::list_processes': '列出程序',
  'minimax::computer_active_window': '目前視窗',
  'minimax::computer_list_windows': '列出視窗',
  'minimax::computer_focus_window': '切換視窗',
  'minimax::computer_screenshot_window': '視窗截圖',
  'minimax::computer_screenshot': '螢幕截圖',
  'minimax::computer_screen_size': '螢幕尺寸',
  'minimax::computer_mouse_position': '滑鼠座標',
  'minimax::computer_click': '滑鼠點擊',
  'minimax::computer_double_click': '滑鼠雙擊',
  'minimax::computer_move_mouse': '移動滑鼠',
  'minimax::computer_drag_mouse': '拖曳滑鼠',
  'minimax::computer_scroll': '滾動滑鼠',
  'minimax::computer_type_text': '鍵盤輸入',
  'minimax::computer_press_key': '按鍵',
  'minimax::computer_hotkey': '快捷鍵',
  'minimax::computer_locate_image': '尋找畫面圖片',
  'minimax::computer_control': '電腦操作',
  'minimax::browser_open': '背景瀏覽器開啟',
  'minimax::browser_get_page': '讀取背景頁面',
  'minimax::browser_links': '列出背景頁面連結',
  'minimax::browser_follow_link': '背景瀏覽器前往連結',
  'minimax::browser_click': '背景瀏覽器點擊',
  'minimax::browser_fill': '背景瀏覽器輸入',
  'minimax::browser_press_key': '背景瀏覽器按鍵',
  'minimax::browser_wait': '背景瀏覽器等待',
  'minimax::browser_back': '背景瀏覽器上一頁',
  'minimax::browser_scroll': '背景瀏覽器滾動',
  'minimax::browser_screenshot': '背景瀏覽器截圖',
  'minimax::browser_login': '背景登入',
  'minimax::browser_close': '關閉背景瀏覽器',
};
function friendlyToolName(name) {
  return TOOL_NAMES_ZH[name] || TOOL_NAMES_ZH[name?.replace?.(/^minimax::/, '')] || name;
}
function jsonPreview(value, max = 160) {
  let text = '';
  try { text = JSON.stringify(redactSecrets(value ?? {})); }
  catch (e) { text = String(redactSecrets(value)); }
  return text.length > max ? text.slice(0, max) + '...' : text;
}
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function redactSecrets(value) {
  const re = /(密碼|password|passwd|pwd|passcode|token|secret|client_secret)\s*(?:[:=：]|為|为|是|is)?\s*([^\s,，;；。]+)/gi;
  if (typeof value === 'string') return value.replace(re, '$1=********');
  if (Array.isArray(value)) return value.map(redactSecrets);
  if (value && typeof value === 'object') {
    const out = {};
    Object.entries(value).forEach(([k, v]) => {
      out[k] = /(密碼|password|passwd|pwd|passcode|token|secret|client_secret)/i.test(k) ? '********' : redactSecrets(v);
    });
    return out;
  }
  return value;
}
function collectFileRefs(value, out = []) {
  if (!value || out.length > 8) return out;
  if (Array.isArray(value)) {
    value.forEach(v => collectFileRefs(v, out));
    return out;
  }
  if (typeof value === 'object') {
    const url = value.file_url || value.url;
    const path = value.path || value.dest || value.dest_path;
    const name = value.filename || value.name || (path ? String(path).split(/[\\/]/).pop() : '檔案');
    if (url && (String(url).startsWith('/files/') || String(url).startsWith('http'))) {
      out.push({ url: String(url), name: String(name || '檔案'), mime: value.mime || '' });
    }
    Object.entries(value).forEach(([k, v]) => {
      if (!['file_url','url','path','dest','dest_path','filename','name','mime'].includes(k)) collectFileRefs(v, out);
    });
  }
  return out;
}
function fileLinksHtml(result) {
  const refs = collectFileRefs(result);
  if (!refs.length) return '';
  return refs.map(ref => {
    const isImg = /^image\//.test(ref.mime || '') || /\.(png|jpe?g|webp|gif|bmp)$/i.test(ref.url);
    const link = `<a class="file-chip" href="${esc(ref.url)}" target="_blank" download><span class="ic">${svgIcon('download')}</span><span>${esc(ref.name)}</span></a>`;
    const img = isImg ? `<img class="file-preview-img" src="${esc(ref.url)}" alt="${esc(ref.name)}">` : '';
    return link + img;
  }).join('');
}
function formatToolArgs(name, args) {
  args = redactSecrets(args || {});
  const base = name?.replace?.(/^minimax::/, '') || name;
  if (base === 'delegate_tool_task') return (args.task || '工具任務').slice(0, 120);
  if (base === 'search_memories') return `查詢「${args.query || ''}」`;
  if (base === 'store_visual_identity') return `名稱「${args.label || '目前使用者'}」`;
  if (base === 'store_voice_identity') return `名稱「${args.label || '目前使用者'}」`;
  if (base === 'gmail_read') return `搜尋郵件「${args.query || ''}」`;
  if (base === 'gmail_send') return `寄給 ${args.to || '(未指定)'}，主旨「${args.subject || '(無主旨)'}」`;
  if (base === 'weather_get') return `地點「${args.city || ''}」`;
  if (base === 'web_search') return `搜尋「${args.query || ''}」`;
  if (base === 'web_image_search') return `搜尋圖片「${args.query || ''}」`;
  if (base === 'download_image') return args.url || '';
  if (base === 'calculator') return String(args.expression || '');
  if (base === 'site_memory_search') return `查詢「${args.query || ''}」`;
  if (base === 'site_memory_remember') return `${args.service || ''} → ${args.url || ''}`;
  if (base === 'site_memory_mark_failure') return `${args.service || ''} → ${args.url || ''}`;
  if (base === 'website_find') return `尋找「${args.query || ''}」`;
  if (base.startsWith('computer_')) {
    if (base === 'computer_active_window') return '確認目前前景視窗';
    if (base === 'computer_list_windows') return `最多列出 ${args.max_items || 40} 個視窗`;
    if (base === 'computer_focus_window') return `目標「${args.target || ''}」`;
    if (base === 'computer_screenshot_window') return args.target ? `截取「${args.target}」` : '截取目前視窗';
    if (base === 'computer_type_text') return `輸入 ${String(args.text || '').length} 個字`;
    if (base === 'computer_hotkey') return `快捷鍵 ${JSON.stringify(args.keys || [])}`;
    if (base === 'computer_control') return `執行 ${(args.steps || []).length} 個操作步驟`;
    if ('x' in args || 'y' in args) return `座標 (${args.x ?? 0}, ${args.y ?? 0})`;
    return jsonPreview(args, 120);
  }
  if (base.startsWith('browser_')) {
    if (base === 'browser_login') return `登入 ${args.url || ''}，帳號 ${args.username || ''}`;
    if (base === 'browser_open') return args.url || '';
    if (base === 'browser_links') return args.query ? `搜尋連結「${args.query}」` : `最多 ${args.limit || 80} 個連結`;
    if (base === 'browser_follow_link') return `連結 #${args.index}`;
    if (base === 'browser_fill') return `填入 ${args.target || ''}`;
    if (base === 'browser_click') return `點擊 ${args.target || ''}`;
    if (base === 'browser_wait') return `等待 ${args.milliseconds || 1000}ms`;
    if (base === 'browser_back') return '返回上一頁';
    if (base === 'browser_scroll') return `滾動 ${args.amount || 800}`;
    return jsonPreview(args, 120);
  }
  if (['read_file','read_file_range','path_exists','list_directory','list_directory_recursive','search_files','search_text','file_hash','detect_file_type','pdf_extract_text','docx_extract_text','xlsx_read_sheet','git_status','git_log','git_diff'].includes(base)) {
    return args.path || args.root || args.repo_path || args.db_path || jsonPreview(args, 120);
  }
  if (base === 'http_request') return `${args.method || 'GET'} ${args.url || ''}`;
  if (base === 'download_file') return `${args.url || ''} → ${args.dest_path || ''}`;
  if (base.startsWith('git_')) return args.repo_path || '.';
  return jsonPreview(args, 180);
}
function formatToolResult(name, result) {
  const base = name?.replace?.(/^minimax::/, '') || name;
  if (typeof result === 'string') return redactSecrets(result).slice(0, 220);
  result = redactSecrets(result || {});
  if (result.summary) return String(result.summary).slice(0, 260);
  if (result.progress_snapshot?.summary) return String(result.progress_snapshot.summary).slice(0, 260);
  if (result.error) return `失敗：${String(result.error).slice(0, 220)}`;
  if (base === 'search_memories' && Array.isArray(result.results)) {
    if (!result.results.length) return '沒有找到相關記憶';
    return `找到 ${result.results.length} 筆：${String(result.results[0].memory || '').slice(0, 120)}`;
  }
  if (base === 'gmail_read') {
    if (result.found === false) return '沒有找到符合條件的郵件';
    return `找到郵件：${result.subject || '(無主旨)'}；${String(result.snippet || '').slice(0, 140)}`;
  }
  if (base === 'gmail_list' && Array.isArray(result.messages)) {
    return `列出 ${result.messages.length} 封郵件`;
  }
  if (base === 'list_processes') {
    const top = Array.isArray(result.top_processes) ? result.top_processes.slice(0, 4).map(p => `${p.name} x${p.count}`).join('、') : '';
    return `列出 ${result.process_count || 0} 個程序${top ? '；常見：' + top : ''}`;
  }
  if (base === 'site_memory_search') {
    const first = Array.isArray(result.sites) && result.sites.length ? result.sites[0] : null;
    return `找到 ${(result.sites || []).length} 個入口、${(result.failures || []).length} 個失敗紀錄${first ? '；優先：' + first.url : ''}`;
  }
  if (base === 'site_memory_remember' && result.site) return `已記住：${result.site.service || ''} ${result.site.url || ''}`;
  if (base === 'site_memory_mark_failure' && result.failure) return `已記住失敗網址：${result.failure.url || ''}`;
  if (base === 'website_find') {
    const best = result.best || {};
    return best.url || best.final_url ? `找到入口：${best.title || ''} ${best.final_url || best.url}` : '沒有找到可驗證入口';
  }
  if (base === 'computer_active_window' && result.active_window) {
    const w = result.active_window;
    return `目前視窗：${w.process || ''} - ${w.title || ''}`;
  }
  if (base === 'computer_list_windows' && Array.isArray(result.windows)) {
    const active = result.active_window || {};
    return `找到 ${result.count ?? result.windows.length} 個視窗；目前：${active.title || ''}`;
  }
  if (base === 'computer_focus_window' && result.active_window) {
    const w = result.active_window;
    return `已切換到：${w.process || ''} - ${w.title || ''}`;
  }
  if (base === 'computer_screenshot_window' && result.file_url) {
    const w = result.window || {};
    return `已截取視窗：${w.title || result.filename || result.path || ''}`;
  }
  if (base === 'computer_screenshot' && result.file_url) {
    const w = result.active_window || {};
    return `已截圖：${result.filename || result.path || ''}${w.title ? '；前景：' + w.title : ''}`;
  }
  if (base.startsWith('computer_') && result.success === true) return '電腦操作完成';
  if (base === 'browser_login') return result.needs_user_action ? (result.message || '需要使用者確認') : (result.logged_in ? '背景登入流程已完成' : (result.message || '已送出登入流程'));
  if (base === 'browser_links') {
    const first = Array.isArray(result.links) && result.links.length ? result.links[0] : null;
    return `列出 ${result.count || 0} 個連結${first ? '；第一個：' + (first.text || first.href || '') : ''}`;
  }
  if (base === 'browser_follow_link' || base === 'browser_wait' || base === 'browser_back') {
    return result.message || `背景瀏覽器：${result.title || result.url || '完成'}`;
  }
  if (base.startsWith('browser_') && result.success === true) return result.message || `背景瀏覽器：${result.title || result.url || '完成'}`;
  if (Array.isArray(result.results)) return `取得 ${result.results.length} 筆結果`;
  if (Array.isArray(result.rows)) return `取得 ${result.rows.length} 列資料`;
  if (Array.isArray(result.items)) return `列出 ${result.items.length} 個項目`;
  if (Array.isArray(result.files)) return `列出 ${result.files.length} 個檔案`;
  if (result.success === true) return `完成${result.path ? '：' + result.path : ''}`;
  if (result.stdout) return String(result.stdout).slice(0, 220);
  if (result.stored) return `已寫入記憶：${result.id || ''}`;
  if (result.ok === true && result.label) return `已建立：${result.label}`;
  if (result.enabled === true && Array.isArray(result.identities)) return `共有 ${result.count ?? result.identities.length} 筆`;
  return jsonPreview(result, 220);
}
function addToolCard(name, args, result) {
  const c = document.createElement('div');
  c.className = 'tool-card pending';
  c.dataset.toolName = name;
  c.innerHTML = `<div class="tn"><span class="ic" style="font-size:13px">${svgIcon('tool')}</span>${friendlyToolName(name)}</div>
                 <div class="ta">${formatToolArgs(name, args ?? {})}</div>`;
  $('tools-body').prepend(c);
  pendingToolCards.push(c);
  while ($('tools-body').children.length > 30) $('tools-body').lastChild.remove();
}
function updateLastToolCard(payload) {
  const idx = pendingToolCards.map(c => c.dataset.toolName).lastIndexOf(payload.name);
  const card = idx >= 0 ? pendingToolCards.splice(idx, 1)[0] : pendingToolCards.pop();
  if (!card) return;
  card.classList.remove('pending');
  const result = payload.result ?? payload;
  const meta = payload._meta;
  const r = document.createElement('div');
  r.className = 'tr';
  const rs = formatToolResult(payload.name, result);
  let metaStr = '';
  if (meta) {
    const ok = meta.ok ? '✓' : '✗';
    const ms = meta.duration_ms != null ? ` ${meta.duration_ms}ms` : '';
    metaStr = ` <span style="opacity:.5;font-size:11px">[${ok}${ms}]</span>`;
  }
  r.innerHTML = '→ <code>' + rs.slice(0,180) + '</code>' + metaStr;
  const files = fileLinksHtml(result);
  if (files) r.innerHTML += files;
  if (meta && !meta.ok) card.classList.add('error');
  card.appendChild(r);
}

function addMemCard(p) {
  p = redactSecrets(p || {});
  const minImp = parseInt($('set-imp').value || '1', 10);
  if ((p.importance ?? 3) < minImp) return;
  const isUser = p.category === 'user';
  const c = document.createElement('div');
  c.className = 'mem-card' + (isUser ? ' user' : '');
  const catMap = { personal:'個人', preference:'偏好', technical:'技術', project:'專案', event:'事件', credential:'帳密', other:'其他', user:'使用者' };
  c.innerHTML = `<div class="mt">${catMap[p.category] || '記憶'}<span class="imp">重要度 ${p.importance ?? '-'}</span></div>
                 <div class="mc">${p.memory}</div>`;
  $('mem-body').prepend(c);
  while ($('mem-body').children.length > 40) $('mem-body').lastChild.remove();
}

function setConn(s) {
  const el = $('st-conn');
  const map = { connected: ['● 已連線', 'var(--gold)'], reconnecting: ['● 重連中…', 'var(--amber)'],
                disconnected: ['● 已斷線', 'var(--pink)'] };
  const [t, c] = map[s] ?? ['● ' + s, 'var(--txt2)'];
  el.textContent = t; el.style.color = c;
  if (s === 'connected') { $('cam').classList.add('active'); }
  if (s === 'disconnected') $('cam').classList.remove('active');
}

// ── 統一三欄寬度管理（避免左拖/右拖/收合互相覆蓋）──────
const layout = { left: 360, right: 320, rightSaved: 320, collapsed: false };
function applyGrid(){
  const r = layout.right;
  $('main').style.gridTemplateColumns = `${layout.left}px 1fr ${r}px`;
}
applyGrid();  // 初始套用，確保 JS 接管後三欄正確

// 左欄右邊界拖曳
(function(){
  const drag = $('col-drag');
  let active = false, startX = 0, startW = 0;
  drag.addEventListener('mousedown', e => {
    active = true; startX = e.clientX; startW = layout.left;
    drag.classList.add('dragging');
    $('main').classList.add('dragging');
    document.body.style.cursor = 'col-resize'; document.body.style.userSelect = 'none';
    e.preventDefault();
  });
  window.addEventListener('mousemove', e => {
    if (!active) return;
    layout.left = Math.max(220, Math.min(900, startW + e.clientX - startX));
    applyGrid();
  });
  window.addEventListener('mouseup', () => {
    if (!active) return;
    active = false; drag.classList.remove('dragging');
    $('main').classList.remove('dragging');
    document.body.style.cursor = ''; document.body.style.userSelect = '';
  });
})();

// 右欄左邊界拖曳（往左拉變寬）
(function(){
  const drag = $('col-drag-r');
  if (!drag) return;
  let active = false, startX = 0, startW = 0;
  drag.addEventListener('mousedown', e => {
    if (layout.collapsed) return;  // 收合狀態不可拖
    active = true; startX = e.clientX; startW = layout.right;
    drag.classList.add('dragging');
    $('main').classList.add('dragging');
    document.body.style.cursor = 'col-resize'; document.body.style.userSelect = 'none';
    e.preventDefault();
  });
  window.addEventListener('mousemove', e => {
    if (!active) return;
    // 往左拖（clientX 變小）→ 右欄變寬
    layout.right = Math.max(240, Math.min(620, startW - (e.clientX - startX)));
    applyGrid();
  });
  window.addEventListener('mouseup', () => {
    if (!active) return;
    active = false; drag.classList.remove('dragging');
    $('main').classList.remove('dragging');
    document.body.style.cursor = ''; document.body.style.userSelect = '';
  });
})();

// 右欄上下分界拖曳（工具/記憶比例）
(function(){
  const drag = $('row-drag-r'), top = $('tools-sec'), bot = $('mem-sec');
  if(!drag||!top||!bot) return;
  let active=false, startY=0, startTop=0, totalH=0;
  drag.addEventListener('mousedown', e=>{
    active=true; startY=e.clientY; startTop=top.offsetHeight;
    totalH=top.offsetHeight+bot.offsetHeight;
    drag.classList.add('dragging');
    document.body.style.cursor='row-resize'; document.body.style.userSelect='none';
    e.preventDefault();
  });
  window.addEventListener('mousemove', e=>{
    if(!active) return;
    let h=Math.max(60, Math.min(totalH-60, startTop+(e.clientY-startY)));
    const tp=(h/totalH)*100;
    top.style.flex=`0 0 ${tp}%`; bot.style.flex=`0 0 ${100-tp}%`;
  });
  window.addEventListener('mouseup', ()=>{
    if(!active) return; active=false; drag.classList.remove('dragging');
    document.body.style.cursor=''; document.body.style.userSelect='';
  });
})();

// ════════════════════════════════════════════════════════════
//  UI 互動（不依賴後端）
// ════════════════════════════════════════════════════════════

// 收合右欄（修 bug：統一走 applyGrid，不再直接寫 grid）
$('btn-collapse').onclick = () => {
  const m = $('main');
  layout.collapsed = !layout.collapsed;
  if (layout.collapsed) {
    layout.rightSaved = layout.right;
    layout.right = 44;
  } else {
    layout.right = layout.rightSaved;
  }
  m.classList.toggle('mem-collapsed', layout.collapsed);
  const ic = $('btn-collapse').querySelector('.ic');
  const name = layout.collapsed ? 'layout-sidebar-right-expand' : 'layout-sidebar-right-collapse';
  ic.dataset.i = name; ic.innerHTML = svgIcon(name);
  applyGrid();
};

// ── 設定面板：開關 + 分頁 ──────────────────────────────
const overlay = $('overlay');
function openSettings(){ overlay.classList.add('show'); }
function closeSettings(){ overlay.classList.remove('show'); }
$('btn-settings').onclick = openSettings;
$('btn-close-settings').onclick = $('btn-cancel-settings').onclick = closeSettings;
overlay.onclick = e => { if (e.target === overlay) closeSettings(); };

// ── 連線分割按鈕：下拉快速設定 ─────────────────────────
const splitBtn = $('split-connect');
function openQuickCfg(){ splitBtn.classList.add('open'); }
function closeQuickCfg(){ splitBtn.classList.remove('open'); }
$('btn-connect-menu').onclick = e => {
  e.stopPropagation();
  splitBtn.classList.toggle('open');
};
// 點外面關閉
document.addEventListener('click', e => {
  if (!splitBtn.contains(e.target)) closeQuickCfg();
});
// 快速設定 → 完整設定：開面板並跳到連線分頁
$('qc-open-settings').onclick = () => {
  closeQuickCfg();
  openSettings();
  document.querySelector('.nav-item[data-pane="connect"]')?.click();
};
// quick-cfg ⇄ 設定面板雙向同步（同一份值）
function bindSync(quickId, fullId){
  const q = $(quickId), f = $(fullId);
  if (!q || !f) return;
  const sync = (src, dest) => {
    if (dest.value !== src.value) {
      dest.value = src.value;
      dest.dispatchEvent(new Event('input'));
      dest.dispatchEvent(new Event('change'));
    }
  };
  q.addEventListener('input', () => sync(q, f));
  q.addEventListener('change', () => sync(q, f));
  f.addEventListener('input', () => sync(f, q));
  f.addEventListener('change', () => sync(f, q));
}
bindSync('q-voice', 'voice');
bindSync('q-thinking', 'thinking');
bindSync('q-respond', 'set-respond');
bindSync('q-drift', 'set-drift');
bindSync('q-fps', 'set-fps');

$('btn-save-settings').onclick = () => {
  console.log('settings saved', collectSettings());
  saveMinimaxSettings();
  syncFeatureControls();
  // 儲存後給一個微妙的視覺回饋（按鈕短暫變綠/打勾感）
  const b = $('btn-save-settings');
  const orig = b.textContent;
  b.textContent = '已儲存 ✓';
  b.style.color = 'var(--green)'; b.style.borderColor = 'rgba(127,220,179,.5)';
  setTimeout(() => { b.textContent = orig; b.style.color = ''; b.style.borderColor = ''; closeSettings(); }, 700);
};

// 分頁切換
const PANE_META = {
  connect:    { title: '連線 · Connection',  sub: '展前參數 · 連線後鎖定' },
  persona:    { title: '角色 · Persona',     sub: '人格與行為原則 · 連線前設定' },
  perception: { title: '感知 · Perception',  sub: '視覺與語音偵測閾值' },
  dialogue:   { title: '對話 · Dialogue',    sub: '互動與回應行為' },
  system:     { title: '系統 · System',      sub: '工具、Session、除錯' },
};
$('nav-list').addEventListener('click', e => {
  const item = e.target.closest('.nav-item'); if (!item) return;
  const key = item.dataset.pane;
  document.querySelectorAll('.nav-item').forEach(n => n.classList.toggle('active', n === item));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('active', p.dataset.pane === key));
  const meta = PANE_META[key];
  if (meta) { $('pane-title').textContent = meta.title; $('pane-sub').textContent = meta.sub; }
});

let memoryAccounts = { accounts: [], current: 'default', backend: '' };
function updateMemoryAccounts(payload = {}) {
  memoryAccounts = {
    accounts: payload.accounts || memoryAccounts.accounts || ['default'],
    current: payload.current || memoryAccounts.current || 'default',
    backend: payload.backend || memoryAccounts.backend || '',
  };
  const sel = $('memory-user');
  if (sel) {
    sel.innerHTML = '';
    memoryAccounts.accounts.forEach(id => {
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = id;
      sel.appendChild(opt);
    });
    sel.value = memoryAccounts.current;
  }
  const status = $('memory-user-status');
  if (status) {
    const backend = memoryAccounts.backend ? ` · ${memoryAccounts.backend}` : '';
    status.textContent = `目前帳號：${memoryAccounts.current}${backend}`;
  }
  try { localStorage.setItem('raphael.memoryUser', memoryAccounts.current); } catch(e) {}
}

function selectedMemoryUser() {
  return $('memory-user')?.value || memoryAccounts.current || localStorage.getItem('raphael.memoryUser') || 'default';
}

function requestMemoryAccounts(action = 'list', userId = '') {
  wsSend('memory_account', { action, user_id: userId });
}

function collectSettings() {
  const tog = id => $(id)?.classList.contains('on') ?? false;
  return {
    persona: $('set-persona').value,
    drift_threshold: parseFloat($('set-drift').value),
    semantic_fps: parseInt($('set-fps').value, 10),
    vad_threshold: parseFloat($('set-vad').value),
    min_importance: parseInt($('set-imp').value, 10),
    thinking_trace: tog('tog-thinking-trace'),
    bilingual:      tog('tog-bilingual'),
    proactive:      tog('tog-proactive'),
    interruptible:  tog('tog-interrupt'),
    hide_subtitle:  tog('tog-subtitle'),
    task_voice:     tog('tog-task-voice'),
    session_resumption: tog('tog-resume'),
    google_search:  tog('tog-search'),
    debug_mode:     tog('tog-debug'),
    features: collectFeatureFlags(),
    minimax_settings: collectMinimaxSettings(),
  };
}

function collectMinimaxSettings() {
  const num = (id, fallback) => {
    const value = parseFloat($(id)?.value ?? fallback);
    return Number.isFinite(value) ? value : fallback;
  };
  return {
    model: ($('minimax-model')?.value || 'minimaxai/minimax-m2.7').trim(),
    base_url: ($('minimax-base-url')?.value || 'https://integrate.api.nvidia.com/v1').trim(),
    max_tool_rounds: Math.round(num('set-mm-rounds', 16)),
    request_timeout: Math.round(num('set-mm-timeout', 180)),
    temperature: Number(num('set-mm-temp', 0.2).toFixed(2)),
  };
}

let minimaxSettingsFromLocal = false;
function applyMinimaxSettings(settings = {}, { fromRuntime = false } = {}) {
  if (fromRuntime && minimaxSettingsFromLocal) return;
  const set = (id, value) => {
    const el = $(id);
    if (el && value != null) {
      el.value = value;
      el.dispatchEvent(new Event('input'));
      el.dispatchEvent(new Event('change'));
    }
  };
  set('minimax-model', settings.model);
  set('minimax-base-url', settings.base_url);
  set('set-mm-rounds', settings.max_tool_rounds);
  set('set-mm-timeout', settings.request_timeout);
  set('set-mm-temp', settings.temperature);
}

function saveMinimaxSettings() {
  try { localStorage.setItem('raphael.minimaxSettings', JSON.stringify(collectMinimaxSettings())); } catch(e) {}
}

function restoreMinimaxSettings() {
  let settings = null;
  try { settings = JSON.parse(localStorage.getItem('raphael.minimaxSettings') || 'null'); } catch(e) {}
  if (!settings) return;
  minimaxSettingsFromLocal = true;
  applyMinimaxSettings(settings);
}

function featureEnabled(name) {
  const map = {
    vision_gate: 'tog-vision-gate',
    vision_overlay: 'tog-vision-overlay',
    vision_proactive: 'tog-vision-proactive',
    visual_identity: 'tog-visual-identity',
    voice_identity: 'tog-voice-identity',
    mouth_sync: 'tog-mouth-sync',
    advanced_voice_gate: 'tog-advanced-voice-gate',
    full_duplex: 'tog-full-duplex',
    computer_tools: 'tog-computer-tools',
    voice_interrupt: 'tog-voice-interrupt',
  };
  const el = $(map[name]);
  return el ? el.classList.contains('on') : true;
}

function collectFeatureFlags() {
  return {
    vision_gate: featureEnabled('vision_gate'),
    vision_overlay: featureEnabled('vision_overlay'),
    vision_proactive: featureEnabled('vision_proactive'),
    visual_identity: featureEnabled('visual_identity'),
    visual_identity_auto_enroll: featureEnabled('visual_identity'),
    voice_identity: featureEnabled('voice_identity'),
    voice_identity_auto_enroll: featureEnabled('voice_identity'),
    mouth_sync: featureEnabled('mouth_sync'),
    advanced_voice_gate: featureEnabled('advanced_voice_gate'),
    full_duplex: featureEnabled('full_duplex'),
    computer_tools: featureEnabled('computer_tools'),
    voice_interrupt: featureEnabled('voice_interrupt'),
  };
}

function syncFeatureControls() {
  try { localStorage.setItem('raphael.featureFlags.v2', JSON.stringify(collectFeatureFlags())); } catch(e) {}
  if (!featureEnabled('vision_gate')) {
    latestVisionBoxes = [];
    sensorStats.objects = 0;
    sensorStats.drift = 0;
    sensorStats.flow = 0;
    sensorStats.roi = 0;
    sensorStats.triggered = false;
    $('cam')?.classList.remove('triggered');
  }
  if (!featureEnabled('visual_identity')) {
    latestIdentityBoxes = [];
  }
  drawVisionOverlay();
  wsSend('feature_control', collectFeatureFlags());
  if (ws && ws.readyState === WebSocket.OPEN) {
    wsSend('session_config', collectSessionConfig());
  }
}

function restoreFeatureFlags() {
  // v2 key：一次性重置掉舊版把身份/對嘴/進階閘門記成 on 的狀態，套用正確展覽預設。
  let flags = null;
  try { flags = JSON.parse(localStorage.getItem('raphael.featureFlags.v2') || 'null'); } catch(e) {}
  if (!flags) return;
  const mapping = {
    vision_gate: 'tog-vision-gate',
    vision_overlay: 'tog-vision-overlay',
    vision_proactive: 'tog-vision-proactive',
    visual_identity: 'tog-visual-identity',
    voice_identity: 'tog-voice-identity',
    mouth_sync: 'tog-mouth-sync',
    advanced_voice_gate: 'tog-advanced-voice-gate',
    full_duplex: 'tog-full-duplex',
    computer_tools: 'tog-computer-tools',
    voice_interrupt: 'tog-voice-interrupt',
  };
  Object.entries(mapping).forEach(([key, id]) => {
    const el = $(id);
    if (el && flags[key] != null) el.classList.toggle('on', !!flags[key]);
  });
}

// ── slider：即時數值 + 軌道填充百分比 (--p) ─────────────
const SLIDERS = [
  ['set-drift','set-drift-v',v=>(+v).toFixed(2)],
  ['set-fps','set-fps-v',v=>v],
  ['set-vad','set-vad-v',v=>(+v).toFixed(3)],
  ['set-imp','set-imp-v',v=>v],
  ['set-mm-rounds','set-mm-rounds-v',v=>v],
  ['set-mm-timeout','set-mm-timeout-v',v=>v],
  ['set-mm-temp','set-mm-temp-v',v=>(+v).toFixed(2)],
];
function syncSlider(el, out, fmt){
  const min = +el.min, max = +el.max, v = +el.value;
  const p = ((v - min) / (max - min)) * 100;
  el.style.setProperty('--p', p + '%');
  $(out).textContent = fmt(el.value);
}
SLIDERS.forEach(([s,o,f])=>{
  const el=$(s);
  if(el){
    syncSlider(el,o,f);
    el.addEventListener('input', ()=>syncSlider(el,o,f));
    el.addEventListener('change', ()=>syncSlider(el,o,f));
  }
});

restoreMinimaxSettings();
restoreFeatureFlags();
restoreTaskVoiceSetting();

// ── toggle 卡片：全部使用 .on class 切換 ────────────────
['tog-thinking-trace','tog-bilingual','tog-proactive','tog-interrupt',
 'tog-subtitle','tog-task-voice','tog-resume','tog-search','tog-debug', ...FEATURE_TOGGLE_IDS]
  .forEach(id => {
    const el = $(id);
    if (!el) return;
    el.onclick = () => {
      el.classList.toggle('on');
      if (id === 'tog-task-voice') {
        try { localStorage.setItem('raphael.taskVoice', el.classList.contains('on') ? '1' : '0'); } catch(e) {}
      }
      if (id === 'tog-proactive' || FEATURE_TOGGLE_IDS.includes(id)) {
        syncFeatureControls();
        wsSend('source_control', collectSourceState());
      }
    };
  });

// q-respond ⇄ set-respond 同步已在上方 bindSync 中一同連動

// ── 修 dropdown 旋轉 bug：用 JS 管 .open class，不再依賴 :focus-within ──
document.querySelectorAll('.sel-wrap').forEach(wrap => {
  const sel = wrap.querySelector('select'); if (!sel) return;
  
  let ignoreNextClick = false;

  const setOpen = (val) => {
    wrap.classList.toggle('open', val);
  };

  sel.addEventListener('click', () => {
    if (ignoreNextClick) {
      ignoreNextClick = false;
      return;
    }
    const isOpen = wrap.classList.contains('open');
    setOpen(!isOpen);
  });

  sel.addEventListener('blur', () => {
    setOpen(false);
    ignoreNextClick = true;
    setTimeout(() => { ignoreNextClick = false; }, 200);
  });

  sel.addEventListener('change', () => {
    setOpen(false);
    sel.blur();
  });

  sel.addEventListener('keydown', e => {
    if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') {
      setOpen(false);
      if (e.key === 'Escape') sel.blur();
    }
  });
});
// ── 來源開關下拉（連線後即時斷來源）──────────────────────
const SRC_KEYS = ['vision','audio','tool','memory'];
const srcMenu = $('src-menu');
$('src-trigger').onclick = e => { e.stopPropagation(); srcMenu.classList.toggle('open'); };
document.addEventListener('click', e => { if(!srcMenu.contains(e.target)) srcMenu.classList.remove('open'); });
document.querySelectorAll('.src-item').forEach(item=>{
  item.onclick = async () => {
    const key = item.dataset.src;
    const sw = $('src-'+key);
    const on = !sw.classList.contains('on');
    sw.classList.toggle('on', on);
    if(key==='memory'){ try{ localStorage.setItem('raphael.memory', on?'1':'0'); }catch(e){} }
    wsSend('source_control', collectSourceState());
    if (key === 'audio') {
      if (on && shouldUseBrowserMic()) await startBrowserAudio();
      if (!on) stopBrowserAudio();
    }
    if (key === 'vision') {
      if (on && shouldUseBrowserVision()) await startBrowserVideo();
      if (!on) stopBrowserVideo();
      if (on && !shouldUseBrowserVision()) stopBrowserVideo();
    }
  };
});

// slider 即時數值（quick-cfg）
[['q-ctx','q-ctx-v',v=>v],['q-temp','q-temp-v',v=>(+v).toFixed(2)],
 ['q-drift','q-drift-v',v=>(+v).toFixed(2)],['q-fps','q-fps-v',v=>v+' fps']]
 .forEach(([s,o,f])=>{
   const el=$(s);
   if(el){
     const update = () => { const t=$(o); if(t)t.textContent=f(el.value); };
     el.addEventListener('input', update);
     el.addEventListener('change', update);
   }
 });

// 儲存為預設值打勾
const qcSave = $('qc-save-default');
if(qcSave){
  // 啟動時還原
  try{ if(localStorage.getItem('raphael.saveDefault')==='1') qcSave.classList.add('on'); }catch(e){}
  qcSave.onclick = () => {
    qcSave.classList.toggle('on');
    const on = qcSave.classList.contains('on');
    try{ localStorage.setItem('raphael.saveDefault', on?'1':'0'); }catch(e){}
    if(on) saveQuickDefaults();
  };
}
function saveQuickDefaults(){
  const cfg = {
    voice:$('q-voice')?.value, respond:$('q-respond')?.value, thinking:$('q-thinking')?.value,
    ctx:$('q-ctx')?.value, temp:$('q-temp')?.value, drift:$('q-drift')?.value, fps:$('q-fps')?.value,
    features: collectFeatureFlags(),
  };
  try{ localStorage.setItem('raphael.quickDefaults', JSON.stringify(cfg)); }catch(e){}
}
function restoreQuickDefaults(){
  let cfg=null;
  try{ cfg=JSON.parse(localStorage.getItem('raphael.quickDefaults')||'null'); }catch(e){}
  if(!cfg) return;
  const set=(id,v)=>{ const el=$(id); if(el&&v!=null){ el.value=v; el.dispatchEvent(new Event('input')); el.dispatchEvent(new Event('change')); }};
  set('q-voice',cfg.voice); set('q-respond',cfg.respond); set('q-thinking',cfg.thinking);
  set('q-ctx',cfg.ctx); set('q-temp',cfg.temp); set('q-drift',cfg.drift); set('q-fps',cfg.fps);
}
restoreQuickDefaults();
updateMemoryAccounts({
  accounts: [localStorage.getItem('raphael.memoryUser') || 'default'],
  current: localStorage.getItem('raphael.memoryUser') || 'default',
});
// 還原記憶來源開關
(function restoreMemorySrc(){
  let saved=null; try{ saved=localStorage.getItem('raphael.memory'); }catch(e){}
  if(saved === '0') $('src-memory')?.classList.remove('on');
  else $('src-memory')?.classList.add('on');
})();

// 發送文字
function sendText(){
  const inp=$('text-input'); const t=inp.value.trim(); if(!t) return;
  applyMessage('transcript_in',{text:t,done:true});
  wsSend('text_in', { text: t });
  inp.value='';
}
$('btn-send').onclick=sendText;
$('text-input').onkeydown=e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendText();} };

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(reader.error || new Error('讀取檔案失敗'));
    reader.readAsDataURL(file);
  });
}
async function uploadFiles(files) {
  if (!files || !files.length) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    applyMessage('error', '請先連線，再上傳檔案。');
    return;
  }
  for (const file of files) {
    if (file.size > 50 * 1024 * 1024) {
      applyMessage('error', `檔案太大，略過：${file.name}`);
      continue;
    }
    try {
      addLabel('user', '你');
      const b = addBubble('user', '');
      b.innerHTML = `<span class="ic" style="font-size:14px">${svgIcon('file')}</span> 已上傳：${esc(file.name)}`;
      const dataUrl = await fileToDataUrl(file);
      wsSend('file_upload', {
        name: file.name,
        mime: file.type || 'application/octet-stream',
        size: file.size,
        data_b64: dataUrl,
      });
    } catch (e) {
      applyMessage('error', `檔案上傳失敗：${file.name}：${e.message || e}`);
    }
  }
}
$('btn-file')?.addEventListener('click', () => $('file-input')?.click());
$('file-input')?.addEventListener('change', e => {
  uploadFiles(Array.from(e.target.files || []));
  e.target.value = '';
});
// 連線鈕
$('btn-connect').onclick=async ()=>{
  applySettingsToSession();
  closeQuickCfg();
  showCfgSummary();
  await refreshRuntime();
  if (shouldUseBrowserMic() && collectSourceState().audio) {
    startBrowserAudio().catch(e => applyMessage('error', '瀏覽器麥克風啟動失敗：' + (e.message || e)));
  }
  connectWS();
};
$('btn-disconnect').onclick=()=>{
  disconnectWS();
  srcMenu.classList.remove('open');
  $('cfg-summary').classList.remove('show');
};
$('btn-interrupt').onclick=()=>{
  wsSend('user_interrupt', {});
  playQueue = [];
  isPlaying = false;
  curRaeBubble = null;
};

if ($('memory-user')) {
  $('memory-user').onchange = () => requestMemoryAccounts('select', $('memory-user').value);
}
if ($('btn-memory-add')) {
  $('btn-memory-add').onclick = () => {
    const inp = $('memory-user-new');
    const userId = (inp?.value || '').trim();
    if (!userId) return;
    requestMemoryAccounts('create', userId);
    inp.value = '';
  };
}
if ($('btn-memory-delete')) {
  $('btn-memory-delete').onclick = () => {
    const userId = selectedMemoryUser();
    if (!userId) return;
    requestMemoryAccounts('delete', userId);
  };
}

// 連線時把展前設定鎖定並顯示摘要
function applySettingsToSession(){
  ['voice','thinking','mic','set-persona','set-respond'].forEach(id=>{
    const el=$(id); if(el) el.setAttribute('disabled','');
  });
}
function unlockSettings(){
  ['voice','thinking','mic','set-persona','set-respond'].forEach(id=>{
    const el=$(id); if(el) el.removeAttribute('disabled');
  });
}
function showCfgSummary(){
  const v=$('voice')?.value||'Puck';
  const t=$('thinking')?.value||'OFF';
  const r=$('q-respond')?.value||'語音';
  const sum=$('cfg-summary');
  sum.innerHTML=`<span>音色 <b>${v}</b></span><span class="sep"></span><span>思考 <b>${t}</b></span><span class="sep"></span><span>回應 <b>${r}</b></span>`;
  sum.classList.add('show');
}

// ════════════════════════════════════════════════════════════
//  WebSocket 接入點
// ════════════════════════════════════════════════════════════
let ws = null;
let audioCtx = null;
let browserStream = null;
let browserAudioCtx = null;
let browserMicSource = null;
let browserMicProcessor = null;
let browserMicMute = null;
let browserVideoEl = null;
let browserVideoTimer = null;
let browserVideoCanvas = null;
let browserVideoFrames = 0;
let browserVideoFpsAt = 0;
let browserVideoMeasuredFps = 0;
let browserVideoLastSendAt = 0;
let manualDisconnect = false;
let reconnectTimer = null;
let reconnectAttempts = 0;
let heartbeatTimer = null;
const runtime = { perception: false, localVision: false, localVad: false };

async function refreshRuntime() {
  try {
    const res = await fetch('/runtime', { cache: 'no-store' });
    if (!res.ok) return runtime;
    const data = await res.json();
    runtime.perception = !!data.perception;
    runtime.localVision = !!data.local_vision;
    runtime.localVad = !!data.local_vad;
    if (data.minimax_settings) applyMinimaxSettings(data.minimax_settings, { fromRuntime: true });
  } catch (e) {}
  return runtime;
}

refreshRuntime();

function shouldUseBrowserVision() {
  return collectSourceState().vision && !runtime.localVision;
}

function clearReconnectTimer() {
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectTimer = null;
}

function scheduleReconnect() {
  if (manualDisconnect || reconnectTimer) return;
  const delay = Math.min(10000, 1000 * Math.pow(1.8, reconnectAttempts++));
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    applySettingsToSession();
    connectWS();
  }, delay);
}

function startHeartbeat() {
  stopHeartbeat();
  heartbeatTimer = setInterval(() => {
    wsSend('ping', { t: Date.now() });
  }, 25000);
}

function stopHeartbeat() {
  if (heartbeatTimer) clearInterval(heartbeatTimer);
  heartbeatTimer = null;
}

function connectWS() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  manualDisconnect = false;
  clearReconnectTimer();
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = async () => {
    reconnectAttempts = 0;
    applyMessage('status','connected');
    document.querySelector('.ctrlbar').classList.add('live');
    document.body.classList.add('connected');
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    await refreshRuntime();
    wsSend('session_config', collectSessionConfig());
    requestMemoryAccounts('list');
    startHeartbeat();
    if (shouldUseBrowserMic() && collectSourceState().audio) await startBrowserAudio();
    if (shouldUseBrowserVision()) await startBrowserVideo();
  };
  ws.onclose = () => {
    applyMessage('status','disconnected');
    document.querySelector('.ctrlbar').classList.remove('live');
    document.body.classList.remove('connected');
    stopHeartbeat();
    stopBrowserMedia();
    ws = null;
    if (manualDisconnect) {
      $('cfg-summary').classList.remove('show');
      unlockSettings();
    } else {
      scheduleReconnect();
    }
  };
  ws.onerror = () => applyMessage('error','WebSocket 連線錯誤');
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    applyMessage(m.channel, m.payload);
  };
}

function disconnectWS() {
  manualDisconnect = true;
  clearReconnectTimer();
  stopHeartbeat();
  if (ws) ws.close();
}

function wsSend(channel, payload) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ channel, payload }));
  }
}

function collectSourceState() {
  const on = key => $('src-' + key)?.classList.contains('on') ?? true;
  return {
    vision: on('vision'),
    audio: on('audio'),
    tool: on('tool'),
    memory: on('memory'),
    proactive: $('tog-proactive')?.classList.contains('on') ?? true,
    features: collectFeatureFlags(),
  };
}

function collectSessionConfig() {
  return {
    voice: $('voice')?.value || $('q-voice')?.value || 'Puck',
    thinking: $('thinking')?.value || $('q-thinking')?.value || 'OFF',
    respond: $('set-respond')?.value || $('q-respond')?.value || '語音',
    persona: $('set-persona')?.value || '',
    memory_user: selectedMemoryUser(),
    sources: collectSourceState(),
    features: collectFeatureFlags(),
    minimax_settings: collectMinimaxSettings(),
    vision_settings: {
      drift_threshold: parseFloat($('set-drift')?.value || $('q-drift')?.value || '0.15'),
      semantic_fps: parseInt($('set-fps')?.value || $('q-fps')?.value || '10', 10),
      capture_fps: 24,
      enable_gate: featureEnabled('vision_gate'),
      render_feedback: featureEnabled('vision_overlay'),
      emit_proactive: featureEnabled('vision_proactive'),
    },
  };
}

function shouldUseBrowserMic() {
  const selected = $('mic')?.value || '';
  return selected.includes('瀏覽器') || !runtime.localVad;
}

function isLocalSecureContext() {
  return location.protocol === 'https:' || ['localhost', '127.0.0.1', '::1'].includes(location.hostname);
}

async function ensureBrowserStream(constraints) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('此瀏覽器不支援麥克風/鏡頭擷取');
  }
  if (!isLocalSecureContext()) {
    throw new Error('瀏覽器麥克風/鏡頭需要 https 或 localhost');
  }
  if (!browserStream) {
    browserStream = await navigator.mediaDevices.getUserMedia(constraints);
    return browserStream;
  }
  const needAudio = constraints.audio && !browserStream.getAudioTracks().length;
  const needVideo = constraints.video && !browserStream.getVideoTracks().length;
  if (needAudio || needVideo) {
    const keepAudio = constraints.audio || browserStream.getAudioTracks().some(t => t.readyState === 'live');
    const keepVideo = constraints.video || browserStream.getVideoTracks().some(t => t.readyState === 'live');
    browserStream.getTracks().forEach(t => t.stop());
    browserStream = await navigator.mediaDevices.getUserMedia({
      audio: keepAudio,
      video: keepVideo,
    });
  }
  return browserStream;
}

function bytesToBase64(bytes) {
  let bin = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

function resampleTo16k(input, inputRate) {
  if (inputRate === 16000) return input;
  const ratio = inputRate / 16000;
  const outLen = Math.max(1, Math.round(input.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = pos - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}

async function resumeBrowserAudioInput() {
  if (browserAudioCtx && browserAudioCtx.state === 'suspended') {
    try { await browserAudioCtx.resume(); } catch (e) {}
  }
}
['pointerdown','keydown','touchstart'].forEach(evt => {
  document.addEventListener(evt, resumeBrowserAudioInput, { passive: true });
});

async function startBrowserAudio() {
  if (browserMicProcessor) {
    await resumeBrowserAudioInput();
    return;
  }
  try {
    const stream = await ensureBrowserStream({ audio: true, video: collectSourceState().vision });
    if (browserVideoTimer && browserVideoEl && browserVideoEl.srcObject !== stream) {
      browserVideoEl.srcObject = stream;
      await browserVideoEl.play();
    }
    browserAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    await resumeBrowserAudioInput();
    browserMicSource = browserAudioCtx.createMediaStreamSource(stream);
    browserMicProcessor = browserAudioCtx.createScriptProcessor(4096, 1, 1);
    browserMicMute = browserAudioCtx.createGain();
    browserMicMute.gain.value = 0;
    browserMicProcessor.onaudioprocess = e => {
      if (!ws || ws.readyState !== WebSocket.OPEN || !collectSourceState().audio) return;
      const input = e.inputBuffer.getChannelData(0);
      const mono16 = resampleTo16k(input, browserAudioCtx.sampleRate);
      const pcm = new Int16Array(mono16.length);
      let peak = 0;
      for (let i = 0; i < mono16.length; i++) {
        const v = Math.max(-1, Math.min(1, mono16[i]));
        peak = Math.max(peak, Math.abs(v));
        pcm[i] = v < 0 ? v * 32768 : v * 32767;
      }
      const speaking = peak > 0.025;
      const probability = Math.min(1, peak * 12);
      applyMessage('vad_event', { speaking, probability });
      wsSend('audio_in', {
        pcm_b64: bytesToBase64(new Uint8Array(pcm.buffer)),
        speaking,
        probability
      });
    };
    browserMicSource.connect(browserMicProcessor);
    browserMicProcessor.connect(browserMicMute);
    browserMicMute.connect(browserAudioCtx.destination);
  } catch (e) {
    applyMessage('error', '瀏覽器麥克風啟動失敗：' + e.message);
    $('src-audio')?.classList.remove('on');
    wsSend('source_control', collectSourceState());
  }
}

function stopBrowserAudio() {
  if (browserMicProcessor) browserMicProcessor.disconnect();
  if (browserMicSource) browserMicSource.disconnect();
  if (browserMicMute) browserMicMute.disconnect();
  if (browserAudioCtx) browserAudioCtx.close().catch(()=>{});
  browserMicProcessor = null;
  browserMicSource = null;
  browserMicMute = null;
  browserAudioCtx = null;
}

async function startBrowserVideo() {
  if (!shouldUseBrowserVision()) return;
  if (browserVideoTimer || !ws || ws.readyState !== WebSocket.OPEN) return;
  try {
    const stream = await ensureBrowserStream({ audio: shouldUseBrowserMic() && collectSourceState().audio, video: true });
    browserVideoEl = browserVideoEl || document.createElement('video');
    browserVideoEl.muted = true;
    browserVideoEl.playsInline = true;
    browserVideoEl.srcObject = stream;
    await browserVideoEl.play();
    if (shouldUseBrowserMic() && collectSourceState().audio && browserMicProcessor) {
      stopBrowserAudio();
      await startBrowserAudio();
    }

    browserVideoCanvas = browserVideoCanvas || document.createElement('canvas');
    const ctx = browserVideoCanvas.getContext('2d');
    browserVideoFrames = 0;
    browserVideoMeasuredFps = 0;
    browserVideoLastSendAt = 0;
    browserVideoFpsAt = performance.now();
    const previewFps = () => 24;
    const backendFps = () => {
      const raw = parseInt($('q-fps')?.value || $('set-fps')?.value || '10', 10);
      return Math.max(1, Math.min(10, Number.isFinite(raw) ? raw : 4));
    };
    const scheduleNext = () => {
      browserVideoTimer = setTimeout(sendFrame, 1000 / previewFps());
    };
    const sendFrame = () => {
      browserVideoTimer = null;
      if (!ws || ws.readyState !== WebSocket.OPEN || !shouldUseBrowserVision()) return;
      if (!browserVideoEl.videoWidth || !browserVideoEl.videoHeight) {
        scheduleNext();
        return;
      }
      // 高質檔位：送 Gemini 的單張畫面拉到 ~1024px / JPEG 0.9，提升 VLM 理解力（讀字/辨物）。
      const maxW = 1024;
      const scale = Math.min(1, maxW / browserVideoEl.videoWidth);
      const w = Math.max(1, Math.round(browserVideoEl.videoWidth * scale));
      const h = Math.max(1, Math.round(browserVideoEl.videoHeight * scale));
      if (browserVideoCanvas.width !== w || browserVideoCanvas.height !== h) {
        browserVideoCanvas.width = w;
        browserVideoCanvas.height = h;
      }
      ctx.drawImage(browserVideoEl, 0, 0, w, h);
      const rawDataUrl = browserVideoCanvas.toDataURL('image/jpeg', 0.9);
      const jpeg = rawDataUrl.split(',')[1];
      updateCameraImage(rawDataUrl);
      browserVideoFrames += 1;
      const now = performance.now();
      const dt = now - browserVideoFpsAt;
      if (dt >= 1000) {
        browserVideoMeasuredFps = browserVideoFrames / (dt / 1000);
        browserVideoFrames = 0;
        browserVideoFpsAt = now;
      }
      applyMessage('sensor_view', {
        local_preview: true,
        gate_state: 'WATCHING',
        fps_capture: browserVideoMeasuredFps,
      });
      if (now - browserVideoLastSendAt >= 1000 / backendFps()) {
        browserVideoLastSendAt = now;
        wsSend('video_in', { jpeg_b64: jpeg });
      }
      scheduleNext();
    };
    sendFrame();
  } catch (e) {
    applyMessage('error', '瀏覽器鏡頭啟動失敗：' + e.message);
    $('src-vision')?.classList.remove('on');
    wsSend('source_control', collectSourceState());
  }
}

function stopBrowserVideo() {
  if (browserVideoTimer) clearTimeout(browserVideoTimer);
  browserVideoTimer = null;
  if (browserVideoEl) browserVideoEl.srcObject = null;
}

function stopBrowserMedia() {
  stopBrowserAudio();
  stopBrowserVideo();
  if (browserStream) browserStream.getTracks().forEach(t => t.stop());
  browserStream = null;
}

// ── Web Audio 播放佇列（24kHz PCM16）──────────────────────
let playQueue = [];
let isPlaying = false;
let taskVoiceLast = { text: '', at: 0 };

function taskVoiceEnabled() {
  return $('tog-task-voice')?.classList.contains('on') ?? true;
}

function restoreTaskVoiceSetting() {
  let saved = null;
  try { saved = localStorage.getItem('raphael.taskVoice'); } catch(e) {}
  if (saved == null) return;
  $('tog-task-voice')?.classList.toggle('on', saved !== '0');
}

function speakTaskVoice(text) {
  text = String(text || '').replace(/\s+/g, ' ').trim();
  if (!text || !taskVoiceEnabled() || !('speechSynthesis' in window)) return;
  const now = Date.now();
  if (taskVoiceLast.text === text && now - taskVoiceLast.at < 4000) return;
  taskVoiceLast = { text, at: now };
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'zh-TW';
  utterance.rate = 1.06;
  utterance.pitch = 1.0;
  utterance.volume = 0.86;
  window.speechSynthesis.speak(utterance);
}

function enqueueAudio(b64) {
  const raw = atob(b64);
  const buf = new Int16Array(raw.length / 2);
  for (let i = 0; i < buf.length; i++) buf[i] = raw.charCodeAt(i*2) | (raw.charCodeAt(i*2+1) << 8);
  const float = new Float32Array(buf.length);
  for (let i = 0; i < buf.length; i++) float[i] = buf[i] / 32768;
  playQueue.push(float);
  if (!isPlaying) drainAudio();
}
function drainAudio() {
  if (!playQueue.length || !audioCtx) { isPlaying = false; return; }
  isPlaying = true;
  const float = playQueue.shift();
  const ab = audioCtx.createBuffer(1, float.length, 24000);
  ab.getChannelData(0).set(float);
  const src = audioCtx.createBufferSource();
  src.buffer = ab;
  src.connect(audioCtx.destination);
  src.onended = drainAudio;
  src.start();
}

// ════════════════════════════════════════════════════════════
//  MOCK 展示資料（除錯用，connectWS 改為 startMock 可啟用）
// ════════════════════════════════════════════════════════════
let mockTimers = [];
function stopMock(){ mockTimers.forEach(t=>clearInterval(t)); mockTimers=[]; }
function startMock(){
  stopMock();  // 防止重複連線疊加多組 interval（動畫變快 bug）
  setTokens(0);
  // 感知事件流
  const reasons=['object_appeared','object_left','person_enter','person_leave','person_moved'];
  const details=['偵測到新物件於下右','物件 #118 已離開畫面','有人進入畫面（下右）','人物 #18 已離開畫面','人物 #1 移動至中中'];
  let ei=0;
  mockTimers.push(setInterval(()=>{ const i=ei++%reasons.length; applyMessage('vision_event',{reason:reasons[i],detail:details[i]}); },1400));

  // VAD 音量起伏
  mockTimers.push(setInterval(()=>{ applyMessage('vad_event',{speaking:Math.random()>0.6,probability:Math.random()}); },300));

  // 感知資料（fps/drift/gate）
  mockTimers.push(setInterval(()=>{ applyMessage('sensor_view',{jpeg:null,vad_speaking:Math.random()>0.7,
    gate_state:['WATCHING','ANALYZING','TRIGGERED'][Math.floor(Math.random()*3)],
    drift:Math.random()*0.4,fps_capture:26+Math.random()*4,fps_semantic:3.5+Math.random()}); },800));

  // 一段腳本化對話（展示工具 + 記憶 + 主動開口）
  const script=[
    [800,()=>{ applyMessage('proactive','vision:semantic'); }],
    [400,()=>{ applyMessage('tool_call',{name:'describe_environment',args:{}}); }],
    [900,()=>{ applyMessage('tool_result',{name:'describe_environment',
      result:{description:'畫面中有 2 人、0 個移動物件',persons_detail:['人物#1（位於中中）','人物#4（位於中左）']}}); }],
    [300,()=>{ applyMessage('memory_write',{memory:'畫面出現兩位人物，位於中央與中左',category:'observation',importance:2}); }],
    [600,()=>{ '我注意到有兩位朋友在現場，你們正在聊些什麼呢？有需要 Raphael 幫忙的地方嗎？'.split('').forEach((ch,i)=>setTimeout(()=>applyMessage('transcript_out',{text:ch}),i*40)); }],
    [3200,()=>applyMessage('transcript_out',{text:'',done:true})],
    [400,()=>setTokens(4677)],
    [800,()=>{ applyMessage('transcript_in',{text:'はい、いいです。',done:true}); }],
    [600,()=>{ applyMessage('memory_write',{memory:'使用者以日語回應，表示同意',category:'user',importance:3}); }],
  ];
  let t=0; script.forEach(([d,fn])=>{ t+=d; setTimeout(fn,t); });
}

