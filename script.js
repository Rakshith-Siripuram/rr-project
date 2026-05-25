
const TM_URL =
  "https://teachablemachine.withgoogle.com/models/b9lFDElzn/";

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

let cocoModel;
let mobileNetModel;
let tmModel;

let stream;
let facingMode = 'environment';

let isPredicting = false;
let animationId = null;
const children = [];
─
const DETECTION_INTERVAL = 700;
const WARNING_THRESHOLD = 40000;
const REPEAT_INTERVAL = 3000;

const REQUIRED_STABILITY = 3;

let lastSpoken = '';
let lastSpokenTime = 0;

let stablePrediction = '';
let stableCount = 0;

let fakeProgress = 0;

const fakeTimer = setInterval(() => {

  fakeProgress += Math.random() * 6;

  if (fakeProgress > 88) fakeProgress = 88;

  loaderFill.style.width =
    fakeProgress + '%';

  loadPercent.textContent =
    Math.round(fakeProgress) + '%';

}, 400);

function setStatus(cls, text) {

  statusDot.className =
    'status-dot ' + cls;

  statusText.textContent = text;
}

function log(msg, type) {

  logEl.innerHTML =
    type === 'warn'
      ? `<span class="warn">⚠ ${msg}</span>`
      : `<span>›</span> ${msg}`;
}


// ── SPEAK ────────────────────────────────────────────────
function speak(text) {

  if (speechSynthesis.speaking) return;

  const u =
    new SpeechSynthesisUtterance(text);

  u.rate = 1.1;

  speechSynthesis.speak(u);
}



// ── LOAD MODELS ──────────────────────────────────────────
async function loadModels() {

  try {

    // COCO
    loadMsg.textContent =
      'Loading COCO SSD...';

    cocoModel =
      await cocoSsd.load({
        base: 'lite_mobilenet_v2'
      });


    // MobileNet
    loadMsg.textContent =
      'Loading MobileNet...';

    mobileNetModel =
      await mobilenet.load();


    // TM
    loadMsg.textContent =
      'Loading Teachable Machine...';

    tmModel =
      await tmImage.load(
        TM_URL + "model.json",
        TM_URL + "metadata.json"
      );


    // FINISH
    clearInterval(fakeTimer);

    loaderFill.style.width = '100%';

    loadPercent.textContent = '100%';

    loadMsg.textContent = 'READY';

    setTimeout(() => {

      overlay.style.display = 'none';

      startBtn.disabled = false;

      switchBtn.disabled = false;

      setStatus('', 'READY');

      log(
        'Hybrid AI + TM loaded.'
      );

    }, 400);

  }

  catch(err) {

    clearInterval(fakeTimer);

    loadMsg.textContent =
      'FAILED — Check internet';

    loadPercent.textContent = '';

    console.error(
      'MODEL LOAD ERROR:',
      err
    );
  }
}

loadModels();


// ── ENABLE CAMERA ────────────────────────────────────────
function enableCam() {

  if (
    !cocoModel ||
    !mobileNetModel ||
    !tmModel ||
    isPredicting
  ) return;

  navigator.mediaDevices
    .getUserMedia({
      video: { facingMode }
    })

    .then(s => {

      stream = s;

      video.srcObject = stream;

      placeholder.style.display = 'none';

      video.onloadeddata = () => {

        isPredicting = true;

        startBtn.disabled = true;

        stopBtn.disabled = false;

        setStatus(
          'active',
          'DETECTING'
        );

        log('Detection running.');

        predictLoop();
      };
    })

    .catch(err => {

      log(
        'Camera error: ' + err.message,
        'warn'
      );

      console.error(err);
    });
}

// ── STEP 1: COCO ──────────────────────────────

if (seen.length > 0) {

  const bestCoco =
    predictions.reduce((a, b) =>
      a.score > b.score ? a : b
    );

  // Strong COCO confidence
  if (bestCoco.score > 0.75) {

    const label =
      bestCoco.class;

    log(
      `COCO: ${label} (${Math.round(bestCoco.score * 100)}%)`
    );

    speakResult(label);

    animationId =
      setTimeout(
        predictLoop,
        DETECTION_INTERVAL
      );

    return;
  }
}


// ── STEP 2: MOBILENET ─────────────────────────

const classifications =
  await mobileNetModel.classify(video);

if (classifications.length > 0) {

  const bestMobile =
    classifications[0];

  const mobileLabel =
    bestMobile.className.split(',')[0];

  // Strong MobileNet confidence
  if (bestMobile.probability > 0.90) {

    log(
      `MobileNet: ${mobileLabel} (${Math.round(bestMobile.probability * 100)}%)`
    );

    speakResult(mobileLabel);

    animationId =
      setTimeout(
        predictLoop,
        DETECTION_INTERVAL
      );

    return;
  }
}


// ── STEP 3: TEACHABLE MACHINE ─────────────────

const tmPredictions =
  await tmModel.predict(video);

let bestTM =
  tmPredictions.reduce((a, b) =>
    a.probability > b.probability ? a : b
  );

if (bestTM.probability > 0.85) {

  const tmLabel =
    bestTM.className;

  log(
    `TM: ${tmLabel} (${Math.round(bestTM.probability * 100)}%)`
  );

  speakResult(tmLabel);
}


// ── UNKNOWN ───────────────────────────────────
else {

  log('Unknown Object');
}
function speakResult(label) {

  if (
    label === stablePrediction
  ) {

    stableCount++;

  }

  else {

    stablePrediction = label;

    stableCount = 1;
  }

  if (
    stableCount >= REQUIRED_STABILITY
  ) {

    const now = Date.now();

    if (
      label !== lastSpoken ||

      now - lastSpokenTime >
      REPEAT_INTERVAL
    ) {

      speak(label);

      lastSpoken = label;

      lastSpokenTime = now;
    }
  }
}

    // ── LOOP AGAIN ───────────────────────────────────
    animationId =
      setTimeout(
        predictLoop,
        800
      );

  }

  catch(err) {

    console.error(
      'Prediction error:',
      err
    );

    if (isPredicting) {

      animationId =
        setTimeout(
          predictLoop,
          800
        );
    }
  }
}

// ── SWITCH CAMERA ─────────────────────────────────────────
async function switchCamera() {

  if (stream) {

    stream
      .getTracks()
      .forEach(t => t.stop());
  }

  facingMode =
    facingMode === 'environment'
      ? 'user'
      : 'environment';

  try {

    stream =
      await navigator.mediaDevices
        .getUserMedia({
          video: { facingMode }
        });

    video.srcObject = stream;

    log('Camera switched.');

  }

  catch(err) {

    log('Switch failed.', 'warn');

    console.error(err);
  }
}


// ── STOP CAMERA ───────────────────────────────────────────
function stopCam() {

  isPredicting = false;

  if (animationId) {

    clearTimeout(animationId);

    animationId = null;
  }

  if (stream) {

    stream
      .getTracks()
      .forEach(t => t.stop());

    video.srcObject = null;

    stream = null;
  }

  speechSynthesis.cancel();

  children.forEach(c => {

    if (liveView.contains(c)) {

      liveView.removeChild(c);
    }
  });

  children.length = 0;

  placeholder.style.display = 'flex';

  startBtn.disabled = false;

  stopBtn.disabled = true;

  lastSpoken = '';

  stablePrediction = '';

  stableCount = 0;

  setStatus('', 'READY');

  log('Camera stopped.');
}


// ── BUTTON EVENTS ─────────────────────────────────────────
startBtn.addEventListener(
  'click',
  enableCam
);

stopBtn.addEventListener(
  'click',
  stopCam
);

switchBtn.addEventListener(
  'click',
  switchCamera
);
