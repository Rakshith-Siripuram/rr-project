// ── GET HTML ELEMENTS ────────────────────────────────────────
const video       = document.getElementById('webcam');
const liveView    = document.getElementById('liveView');
const placeholder = document.getElementById('placeholder');
const startBtn    = document.getElementById('startBtn');
const stopBtn     = document.getElementById('stopBtn');
const switchBtn   = document.getElementById('switchBtn');
const statusDot   = document.getElementById('statusDot');
const statusText  = document.getElementById('status-text');
const logEl       = document.getElementById('log');
const overlay     = document.getElementById('loadingOverlay');
const loaderFill  = document.getElementById('loaderFill');
const loadMsg     = document.getElementById('loadMsg');
const loadPercent = document.getElementById('loadPercent');

// ── STATE VARIABLES ───────────────────────────────────────────
let model;           // the ML model (loaded from internet)
let stream;          // webcam video stream
let facingMode   = 'environment'; // 'environment' = back cam, 'user' = front cam
let isPredicting = false;         // is detection loop running?
let animationId  = null;          // stores setTimeout ID so we can cancel it
const children   = [];            // tracks all bounding box elements on screen

// ── SETTINGS ──────────────────────────────────────────────────
const DETECTION_INTERVAL = 300;   // run detection every 300ms (3 times/sec)
const WARNING_THRESHOLD  = 40000; // bbox area > 40000px means object is "near"
const REPEAT_INTERVAL    = 3000;  // don't repeat same voice alert for 3 sec

let lastSpoken     = '';
let lastSpokenTime = 0;

// ── FAKE PROGRESS BAR ─────────────────────────────────────────
let fakeProgress = 0;
const fakeTimer = setInterval(() => {
  fakeProgress += Math.random() * 6;
  if (fakeProgress > 88) fakeProgress = 88;
  loaderFill.style.width  = fakeProgress + '%';
  loadPercent.textContent = Math.round(fakeProgress) + '%';
}, 400);

// ── STATUS HELPER ─────────────────────────────────────────────
function setStatus(cls, text) {
  statusDot.className    = 'status-dot ' + cls;
  statusText.textContent = text;
}

// ── LOG HELPER ────────────────────────────────────────────────
function log(msg, type) {
  logEl.innerHTML = type === 'warn'
    ? `<span class="warn">⚠ ${msg}</span>`
    : `<span>›</span> ${msg}`;
}

// ── SPEAK ─────────────────────────────────────────────────────
function speak(text) {
  if (speechSynthesis.speaking) return;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.1;
  speechSynthesis.speak(u);
}

// ── VOICE RECOGNITION ─────────────────────────────────────────
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition;

if (SR) {
  recognition = new SR();
  recognition.continuous     = true;
  recognition.interimResults = false;
  recognition.lang           = 'en-US';
  recognition.onend = () => { try { recognition.start(); } catch(e){} };
  recognition.onerror = e => console.warn('Speech recognition error:', e.error);
  recognition.onresult = e => {
    const cmd = e.results[e.results.length - 1][0].transcript.toLowerCase().trim();
    if      (cmd.includes('start') || cmd.includes('turn on'))  { enableCam();    speak('Camera on');  }
    else if (cmd.includes('stop')  || cmd.includes('turn off')) { stopCam();      speak('Camera off'); }
    else if (cmd.includes('switch'))                            { switchCamera(); speak('Switching');  }
  };
}

// ── LOAD MODEL ────────────────────────────────────────────────
cocoSsd.load({ base: 'lite_mobilenet_v2' }).then(loaded => {
  model = loaded;
  clearInterval(fakeTimer);
  loaderFill.style.width  = '100%';
  loadPercent.textContent = '100%';
  loadMsg.textContent     = 'READY';

  setTimeout(() => {
    overlay.style.display  = 'none';
    startBtn.disabled  = false;
    switchBtn.disabled = false;
    setStatus('', 'READY');
    log('Model loaded. Press Start Camera.');
    if (recognition) try { recognition.start(); } catch(e){}
  }, 400);

}).catch(err => {
  clearInterval(fakeTimer);
  loadMsg.textContent     = 'FAILED — check internet connection';
  loadPercent.textContent = '';
  console.error('Model load error:', err);
});

// ── ENABLE CAMERA ─────────────────────────────────────────────
function enableCam() {
  if (!model || isPredicting) return;

  navigator.mediaDevices.getUserMedia({ video: { facingMode } })
    .then(s => {
      stream          = s;
      video.srcObject = stream;
      placeholder.style.display = 'none';

      video.onloadeddata = () => {
        isPredicting      = true;
        startBtn.disabled = true;
        stopBtn.disabled  = false;
        setStatus('active', 'DETECTING');
        log('Detection running.');
        predictLoop();
      };
    })
    .catch(err => {
      log('Camera error: ' + err.message);
      console.error(err);
    });
}

// ── PREDICT LOOP ──────────────────────────────────────────────
function predictLoop() {
  if (!isPredicting) return;

  model.detect(video).then(predictions => {
    if (!isPredicting) return; // stopCam() may have fired during detect()

    // remove old bounding boxes
    children.forEach(c => { if (liveView.contains(c)) liveView.removeChild(c); });
    children.length = 0;

    let proximityMsg = '';
    const seen = [];

    predictions.forEach(pred => {
      if (pred.score < 0.60) return;

      const [x, y, w, h] = pred.bbox;
      const isNear = (w * h) > WARNING_THRESHOLD;

      const box = document.createElement('div');
      box.className = 'highlighter' + (isNear ? ' warning-box' : '');
      box.style.cssText = `left:${x}px; top:${y}px; width:${w}px; height:${h}px;`;

      const lbl = document.createElement('div');
      lbl.className = 'label' + (isNear ? ' warning-label' : '');
      lbl.style.cssText = `left:${x}px; top:${Math.max(0, y - 22)}px;`;
      lbl.textContent = `${pred.class} ${Math.round(pred.score * 100)}%`;

      liveView.appendChild(box);
      liveView.appendChild(lbl);
      children.push(box, lbl);

      seen.push(pred.class);
      if (isNear) proximityMsg = `Warning, ${pred.class} is near`;
    });

    if (seen.length) log('Detected: ' + seen.join(', '));
    else             log('Scanning...');

    const now = Date.now();
    if (proximityMsg && (proximityMsg !== lastSpoken || now - lastSpokenTime > REPEAT_INTERVAL)) {
      speak(proximityMsg);
      lastSpoken     = proximityMsg;
      lastSpokenTime = now;
    }

    animationId = setTimeout(predictLoop, DETECTION_INTERVAL);

  }).catch(err => {
    console.error('Detection error:', err);
    if (isPredicting) animationId = setTimeout(predictLoop, DETECTION_INTERVAL);
  });
}

// ── SWITCH CAMERA ─────────────────────────────────────────────
async function switchCamera() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode } });
    video.srcObject = stream;
    log('Camera switched.');
  } catch(err) {
    log('Switch failed.');
    console.error(err);
  }
}

// ── STOP CAMERA ───────────────────────────────────────────────
function stopCam() {
  isPredicting = false;
  if (animationId) { clearTimeout(animationId); animationId = null; }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    video.srcObject = null;
    stream = null;
  }
  speechSynthesis.cancel();
  children.forEach(c => { if (liveView.contains(c)) liveView.removeChild(c); });
  children.length = 0;
  placeholder.style.display = 'flex';
  startBtn.disabled = false;
  stopBtn.disabled  = true;
  lastSpoken = '';
  setStatus('', 'READY');
  log('Camera stopped.');
}

// ── BUTTON EVENTS ─────────────────────────────────────────────
startBtn.addEventListener('click',  enableCam);
stopBtn.addEventListener('click',   stopCam);
switchBtn.addEventListener('click', switchCamera);
