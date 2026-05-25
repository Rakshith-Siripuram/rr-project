
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
const DETECTION_INTERVAL = 300;
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

// ── MAIN PREDICTION LOOP ─────────────────────────────────
async function predictLoop() {

  if (!isPredicting) return;

  try {

    // ── COCO DETECTION ────────────────────────────────
    const predictions =
      await cocoModel.detect(video);

    if (!isPredicting) return;

    // Clear old boxes
    children.forEach(c => {

      if (liveView.contains(c)) {

        liveView.removeChild(c);
      }
    });

    children.length = 0;

    let seen = [];

    // Draw boxes
    predictions.forEach(pred => {

      if (pred.score < 0.70) return;

      const [x, y, w, h] = pred.bbox;

      const box =
        document.createElement('div');

      box.className = 'highlighter';

      box.style.cssText = `
        left:${x}px;
        top:${y}px;
        width:${w}px;
        height:${h}px;
      `;

      const lbl =
        document.createElement('div');

      lbl.className = 'label';

      lbl.style.cssText = `
        left:${x}px;
        top:${Math.max(0, y - 22)}px;
      `;

      lbl.textContent =
        `${pred.class} ${Math.round(pred.score * 100)}%`;

      liveView.appendChild(box);

      liveView.appendChild(lbl);

      children.push(box, lbl);

      seen.push(pred.class);
    });


    // ── BEST COCO ─────────────────────────────────────
    let bestCoco = null;

    if (predictions.length > 0) {

      bestCoco =
        predictions.reduce((a, b) =>
          a.score > b.score ? a : b
        );
    }


    // ── TM PREDICTION ────────────────────────────────
    const tmPredictions =
      await tmModel.predict(video);

    let bestTM =
      tmPredictions.reduce((a, b) =>
        a.probability > b.probability ? a : b
      );


    // ── FINAL DECISION ───────────────────────────────
    let finalLabel = 'Unknown';

    let finalConfidence = 0;

    let source = '';


    // Prefer TM if highly confident
    if (
      bestTM.probability > 0.85
    ) {

      finalLabel =
        bestTM.className;

      finalConfidence =
        bestTM.probability;

      source = 'TM';
    }


    // Otherwise use COCO
    else if (
      bestCoco &&
      bestCoco.score > 0.70
    ) {

      finalLabel =
        bestCoco.class;

      finalConfidence =
        bestCoco.score;

      source = 'COCO';
    }


    // ── MOBILENET FALLBACK ───────────────────────────
    else {

      const classifications =
        await mobileNetModel.classify(video);

      if (classifications.length > 0) {

        const bestMobile =
          classifications[0];

        if (
          bestMobile.probability > 0.90
        ) {

          finalLabel =
            bestMobile.className.split(',')[0];

          finalConfidence =
            bestMobile.probability;

          source = 'MobileNet';
        }
      }
    }


    // ── STABILITY FILTER ─────────────────────────────
    if (
      finalLabel === stablePrediction
    ) {

      stableCount++;

    }

    else {

      stablePrediction =
        finalLabel;

      stableCount = 1;
    }


    // ── DISPLAY ──────────────────────────────────────
    if (
      finalConfidence < 0.60
    ) {

      log('Unknown Object');

    }

    else {

      log(
        `${source}: ${finalLabel} (${Math.round(finalConfidence * 100)}%)`
      );

      if (
        stableCount >= REQUIRED_STABILITY
      ) {

        const now = Date.now();

        if (
          finalLabel !== lastSpoken ||

          now - lastSpokenTime >
          REPEAT_INTERVAL
        ) {

          speak(finalLabel);

          lastSpoken = finalLabel;

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
