// ─────────────────────────────────────────────────────────────
// HTML ELEMENTS
// ─────────────────────────────────────────────────────────────

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


// ─────────────────────────────────────────────────────────────
// STATE VARIABLES
// ─────────────────────────────────────────────────────────────

let model;

let stream;

let facingMode = 'environment';

let isPredicting = false;

let animationId = null;

const children = [];


// ─────────────────────────────────────────────────────────────
// SETTINGS
// ─────────────────────────────────────────────────────────────

const DETECTION_INTERVAL = 500;

const WARNING_THRESHOLD = 40000;

const REPEAT_INTERVAL = 3000;

const API_INTERVAL = 3000;

let lastApiCall = 0;

let lastSpoken = '';

let lastSpokenTime = 0;


// ─────────────────────────────────────────────────────────────
// FAKE LOADING BAR
// ─────────────────────────────────────────────────────────────

let fakeProgress = 0;

const fakeTimer = setInterval(() => {

  fakeProgress += Math.random() * 6;

  if (fakeProgress > 88) {
    fakeProgress = 88;
  }

  loaderFill.style.width =
    fakeProgress + '%';

  loadPercent.textContent =
    Math.round(fakeProgress) + '%';

}, 400);


// ─────────────────────────────────────────────────────────────
// STATUS HELPER
// ─────────────────────────────────────────────────────────────

function setStatus(cls, text) {

  statusDot.className =
    'status-dot ' + cls;

  statusText.textContent = text;
}


// ─────────────────────────────────────────────────────────────
// LOG HELPER
// ─────────────────────────────────────────────────────────────

function log(msg, type) {

  logEl.innerHTML =
    type === 'warn'
      ? `<span class="warn">⚠ ${msg}</span>`
      : `<span>›</span> ${msg}`;
}


// ─────────────────────────────────────────────────────────────
// SPEAK
// ─────────────────────────────────────────────────────────────

function speak(text) {

  const now = Date.now();

  if (
    text === lastSpoken &&
    now - lastSpokenTime < REPEAT_INTERVAL
  ) {
    return;
  }

  if (speechSynthesis.speaking) {
    speechSynthesis.cancel();
  }

  const u =
    new SpeechSynthesisUtterance(text);

  u.rate = 1.0;

  speechSynthesis.speak(u);

  lastSpoken = text;

  lastSpokenTime = now;
}


// ─────────────────────────────────────────────────────────────
// VOICE RECOGNITION
// ─────────────────────────────────────────────────────────────

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
    }
    catch(e){}
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

    console.log('VOICE:', cmd);

    if (
      cmd.includes('start') ||
      cmd.includes('turn on')
    ) {

      enableCam();

      speak('Camera on');
    }

    else if (
      cmd.includes('stop') ||
      cmd.includes('turn off')
    ) {

      stopCam();

      speak('Camera off');
    }

    else if (
      cmd.includes('switch')
    ) {

      switchCamera();

      speak('Switching camera');
    }
  };
}


// ─────────────────────────────────────────────────────────────
// LOAD MODEL
// ─────────────────────────────────────────────────────────────

cocoSsd.load({
  base: 'lite_mobilenet_v2'
})

.then(loaded => {

  model = loaded;

  clearInterval(fakeTimer);

  loaderFill.style.width = '100%';

  loadPercent.textContent = '100%';

  loadMsg.textContent = 'READY';

  setTimeout(() => {

    overlay.style.display = 'none';

    startBtn.disabled = false;

    switchBtn.disabled = false;

    setStatus('', 'READY');

    log('Model loaded.');

    if (recognition) {

      try {

        recognition.start();

      } catch(e){}
    }

  }, 400);

})

.catch(err => {

  clearInterval(fakeTimer);

  loadMsg.textContent =
    'FAILED';

  loadPercent.textContent = '';

  console.error(err);
});


// ─────────────────────────────────────────────────────────────
// ENABLE CAMERA
// ─────────────────────────────────────────────────────────────

function enableCam() {

  if (!model || isPredicting) return;

  navigator.mediaDevices
    .getUserMedia({
      video: {
        facingMode
      }
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

      console.error(err);

      log(
        'Camera error',
        'warn'
      );
    });
}


// ─────────────────────────────────────────────────────────────
// PREDICT LOOP
// ─────────────────────────────────────────────────────────────

async function predictLoop() {

  if (!isPredicting) return;

  try {

    // =========================================================
    // REMOVE OLD BOXES
    // =========================================================

    children.forEach(c => {

      if (liveView.contains(c)) {

        liveView.removeChild(c);
      }
    });

    children.length = 0;


    // =========================================================
    // LOCAL COCO SSD DETECTION
    // =========================================================

    const predictions =
      await model.detect(video);

    const seen = [];

    predictions.forEach(pred => {

      if (pred.score < 0.60) return;

      const [x, y, w, h] =
        pred.bbox;

      const isNear =
        (w * h) > WARNING_THRESHOLD;

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
        `${pred.class}
        ${Math.round(pred.score * 100)}%`;

      liveView.appendChild(box);

      liveView.appendChild(lbl);

      children.push(box, lbl);

      seen.push(pred.class);

      // SPEAK FOR NEAR OBJECTS

      if (isNear) {

        speak(
          `${pred.class} is near`
        );
      }
    });


    // =========================================================
    // ROBOFLOW API CALL
    // =========================================================

    const nowTime = Date.now();

    if (
      nowTime - lastApiCall >
      API_INTERVAL
    ) {

      lastApiCall = nowTime;

      try {

        const canvas =
          document.createElement('canvas');

        canvas.width =
          video.videoWidth;

        canvas.height =
          video.videoHeight;

        const ctx =
          canvas.getContext('2d');

        ctx.drawImage(
          video,
          0,
          0
        );

        const imageData =
          canvas.toDataURL('image/jpeg');


        fetch(
          'https://serverless.roboflow.com/rakshithsiri23-gmail-com/workflows/general-segmentation-api',
          {

            method: 'POST',

            headers: {
              'Content-Type':
                'application/json'
            },

            body: JSON.stringify({

              api_key:
                'VC8KGjxJ9Ezt5HVVzXaS',

              inputs: {

                image: {

                  type: 'base64',

                  value: imageData
                },

                classes:
                  'bottle, chair, table, laptop, phone, pen, fan, person, charger, penstand'
              }
            })
          }
        )

        .then(res => res.json())

        .then(result => {

          console.log(
            'ROBOFLOW:',
            result
          );

          if (
            result.outputs &&
            result.outputs.length > 0
          ) {

            result.outputs.forEach(item => {

              if (!item.predictions) return;

              item.predictions.forEach(pred => {

                const x =
                  pred.x -
                  pred.width / 2;

                const y =
                  pred.y -
                  pred.height / 2;

                const box =
                  document.createElement('div');

                box.className =
                  'highlighter';

                box.style.cssText = `
                  left:${x}px;
                  top:${y}px;
                  width:${pred.width}px;
                  height:${pred.height}px;
                  border:3px solid cyan;
                `;

                const lbl =
                  document.createElement('div');

                lbl.className =
                  'label';

                lbl.style.cssText = `
                  left:${x}px;
                  top:${Math.max(0, y - 22)}px;
                  background:cyan;
                  color:black;
                `;

                lbl.textContent =
                  `${pred.class}
                  ${Math.round(pred.confidence * 100)}%`;

                liveView.appendChild(box);

                liveView.appendChild(lbl);

                children.push(box, lbl);

                // SPEAK CUSTOM OBJECTS

                speak(
                  `${pred.class} detected`
                );
              });
            });
          }
        })

        .catch(err => {

          console.error(
            'Roboflow error:',
            err
          );
        });

      } catch(err) {

        console.error(err);
      }
    }


    // =========================================================
    // LOG
    // =========================================================

    if (seen.length) {

      log(
        'Detected: ' +
        [...new Set(seen)].join(', ')
      );
    }

    else {

      log('Scanning...');
    }


    // =========================================================
    // LOOP AGAIN
    // =========================================================

    animationId =
      setTimeout(
        predictLoop,
        DETECTION_INTERVAL
      );

  }

  catch(err) {

    console.error(
      'Detection error:',
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


// ─────────────────────────────────────────────────────────────
// SWITCH CAMERA
// ─────────────────────────────────────────────────────────────

async function switchCamera() {

  if (stream) {

    stream.getTracks()
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
        video: {
          facingMode
        }
      });

    video.srcObject = stream;

    log('Camera switched.');

  }

  catch(err) {

    console.error(err);

    log(
      'Switch failed',
      'warn'
    );
  }
}


// ─────────────────────────────────────────────────────────────
// STOP CAMERA
// ─────────────────────────────────────────────────────────────

function stopCam() {

  isPredicting = false;

  if (animationId) {

    clearTimeout(animationId);

    animationId = null;
  }

  if (stream) {

    stream.getTracks()
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

  placeholder.style.display =
    'flex';

  startBtn.disabled = false;

  stopBtn.disabled = true;

  setStatus('', 'READY');

  log('Camera stopped.');
}


// ─────────────────────────────────────────────────────────────
// BUTTON EVENTS
// ─────────────────────────────────────────────────────────────

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
