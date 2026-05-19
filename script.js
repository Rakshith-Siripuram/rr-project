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


// ── MODELS ───────────────────────────────────────────────────
let cocoModel;
let mobileNetModel;


// ── CAMERA / STATE ──────────────────────────────────────────
let stream;
let facingMode = 'environment';

let isPredicting = false;
let animationId = null;

const children = [];


// ── SETTINGS ────────────────────────────────────────────────
const DETECTION_INTERVAL = 300;

const WARNING_THRESHOLD = 40000;

const REPEAT_INTERVAL = 3000;

const REQUIRED_STABILITY = 3;


// ── SPEECH MEMORY ───────────────────────────────────────────
let lastSpoken = '';
let lastSpokenTime = 0;

let stablePrediction = '';
let stableCount = 0;


// ── FAKE LOADING BAR ────────────────────────────────────────
let fakeProgress = 0;

const fakeTimer = setInterval(() => {

  fakeProgress += Math.random() * 6;

  if (fakeProgress > 88) fakeProgress = 88;

  loaderFill.style.width =
    fakeProgress + '%';

  loadPercent.textContent =
    Math.round(fakeProgress) + '%';

}, 400);


// ── STATUS ──────────────────────────────────────────────────
function setStatus(cls, text) {

  statusDot.className =
    'status-dot ' + cls;

  statusText.textContent = text;
}


// ── LOG ─────────────────────────────────────────────────────
function log(msg, type) {

  logEl.innerHTML =
    type === 'warn'
      ? `<span class="warn">⚠ ${msg}</span>`
      : `<span>›</span> ${msg}`;
}


// ── SPEAK ───────────────────────────────────────────────────
function speak(text) {

  if (speechSynthesis.speaking) return;

  const u =
    new SpeechSynthesisUtterance(text);

  u.rate = 1.1;

  speechSynthesis.speak(u);
}


// ── VOICE RECOGNITION ───────────────────────────────────────
const SR =
  window.SpeechRecognition ||
  window.webkitSpeechRecognition;

let recognition;

if (SR) {

  recognition = new SR();

  recognition.continuous = true;

  recognition.interimResults = false;

  recognition.lang = 'en-US';

  recognition.onend = () => {

    try {

      recognition.start();

    } catch(e) {}
  };

  recognition.onerror = e => {

    console.warn(
      'Speech recognition error:',
      e.error
    );
  };

  recognition.onresult = e => {

    const cmd =
      e.results[e.results.length - 1][0]
        .transcript
        .toLowerCase()
        .trim();

    // START
    if (
      cmd.includes('start') ||
      cmd.includes('turn on')
    ) {

      enableCam();

      speak('Camera on');
    }

    // STOP
    else if (
      cmd.includes('stop') ||
      cmd.includes('turn off')
    ) {

      stopCam();

      speak('Camera off');
    }

    // SWITCH
    else if (
      cmd.includes('switch')
    ) {

      switchCamera();

      speak('Switching camera');
    }
  };
}


// ── LOAD MODELS ─────────────────────────────────────────────
Promise.all([

  cocoSsd.load({
    base: 'lite_mobilenet_v2'
  }),

  mobilenet.load()

])

.then(([loadedCoco, loadedMobileNet]) => {

  cocoModel = loadedCoco;

  mobileNetModel = loadedMobileNet;

  clearInterval(fakeTimer);

  loaderFill.style.width = '100%';

  loadPercent.textContent = '100%';

  loadMsg.textContent = 'READY';

  setTimeout(() => {

    overlay.style.display = 'none';

    startBtn.disabled = false;

    switchBtn.disabled = false;

    setStatus('', 'READY');

    log('Hybrid AI loaded successfully.');

    if (recognition) {

      try {

        recognition.start();

      } catch(e) {}
    }

  }, 400);

})

.catch(err => {

  clearInterval(fakeTimer);

  loadMsg.textContent =
    'FAILED — Check internet';

  loadPercent.textContent = '';

  console.error(err);
});


// ── ENABLE CAMERA ───────────────────────────────────────────
function enableCam() {

  if (
    !cocoModel ||
    !mobileNetModel ||
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


// ── MAIN PREDICTION LOOP ────────────────────────────────────
async function predictLoop() {

  if (!isPredicting) return;

  try {

    // ── OBJECT DETECTION ─────────────────────────────────
    const predictions =
      await cocoModel.detect(video);

    if (!isPredicting) return;


    // ── REMOVE OLD BOXES ────────────────────────────────
    children.forEach(c => {

      if (liveView.contains(c)) {

        liveView.removeChild(c);
      }
    });

    children.length = 0;


    let proximityMsg = '';

    const seen = [];


    // ── DRAW DETECTIONS ─────────────────────────────────
    predictions.forEach(pred => {

      if (pred.score < 0.60) return;

      const [x, y, w, h] = pred.bbox;

      const isNear =
        (w * h) > WARNING_THRESHOLD;

      // BOX
      const box =
        document.createElement('div');

      box.className =
        'highlighter' +
        (isNear ? ' warning-box' : '');

      box.style.cssText = `
        left:${x}px;
        top:${y}px;
        width:${w}px;
        height:${h}px;
      `;

      // LABEL
      const lbl =
        document.createElement('div');

      lbl.className =
        'label' +
        (isNear ? ' warning-label' : '');

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

      if (isNear) {

        proximityMsg =
          `Warning, ${pred.class} is near`;
      }
    });


    // ── COCO DETECTION SUCCESS ──────────────────────────
    if (seen.length > 0) {

      log(
        'Detected: ' + seen.join(', ')
      );

      stablePrediction = '';
      stableCount = 0;

      const now = Date.now();

      if (
        proximityMsg &&
        (
          proximityMsg !== lastSpoken ||
          now - lastSpokenTime >
          REPEAT_INTERVAL
        )
      ) {

        speak(proximityMsg);

        lastSpoken = proximityMsg;

        lastSpokenTime = now;
      }
    }


    // ── FALLBACK TO MOBILENET ───────────────────────────
    else {

      log('Trying classification...');

      const classifications =
        await mobileNetModel.classify(video);

      if (classifications.length > 0) {

        const best =
          classifications[0];

        // original full label
        const objectName =
          best.className;

        // clean label
        const cleanName =
          objectName.split(',')[0];

        const confidence =
          Math.round(
            best.probability * 100
          );

        log(
          `Classification: ${cleanName} (${confidence}%)`
        );


        // ── STABILITY FILTER ─────────────────────────
        if (
          cleanName === stablePrediction
        ) {

          stableCount++;

        } else {

          stablePrediction = cleanName;

          stableCount = 1;
        }


        // ── SPEAK ONLY IF STABLE ────────────────────
        if (
          stableCount >= REQUIRED_STABILITY
        ) {

          const now = Date.now();

          if (
            cleanName !== lastSpoken ||

            now - lastSpokenTime >
            REPEAT_INTERVAL
          ) {

            speak(cleanName);

            lastSpoken = cleanName;

            lastSpokenTime = now;
          }
        }

      } else {

        log('Scanning...');
      }
    }


    // ── LOOP AGAIN ─────────────────────────────────────
    animationId =
      setTimeout(
        predictLoop,
        DETECTION_INTERVAL
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
          DETECTION_INTERVAL
        );
    }
  }
}


// ── SWITCH CAMERA ───────────────────────────────────────────
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


// ── STOP CAMERA ─────────────────────────────────────────────
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


// ── BUTTON EVENTS ───────────────────────────────────────────
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
