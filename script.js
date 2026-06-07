const TM_URL = "https://teachablemachine.withgoogle.com/models/b9lFDElzn/";

const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const placeholder = document.getElementById('placeholder');

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const switchBtn = document.getElementById('switchBtn');

const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('status-text');

const logEl = document.getElementById('log');

const overlay = document.getElementById('loadingOverlay');
const loaderFill = document.getElementById('loaderFill');
const loadMsg = document.getElementById('loadMsg');
const loadPercent = document.getElementById('loadPercent');

let cocoModel;
let mobileNetModel;
let tmModel;

let stream;
let facingMode = 'environment';

let isPredicting = false;
let animationId = null;

const children = [];

const DETECTION_INTERVAL = 300;
const REPEAT_INTERVAL = 5000;
const REQUIRED_STABILITY = 3;
const COCO_STABILITY = 3;

let recognition = null;

let lastSpoken = '';
let lastSpokenTime = 0;

let stablePrediction = '';
let stableCount = 0;
let cocoStablePrediction = '';
let cocoStableCount = 0;

let availableVoices = [];
speechSynthesis.onvoiceschanged = function () {
    availableVoices = speechSynthesis.getVoices();
};

let isSpeaking = false;

function setStatus(cls, text) {
    statusDot.className = 'status-dot ' + cls;
    statusText.textContent = text;
}

function log(msg, type) {
    logEl.innerHTML =
        type === 'warn'
        ? `<span class="warn">⚠ ${msg}</span>`
        : `<span>›</span> ${msg}`;
}

function speak(text) {

    if (isSpeaking) return;

    console.log("Speaking:", text);
    speechSynthesis.cancel();

    

    isSpeaking = true;

    let speech = new SpeechSynthesisUtterance(text);
    speech.rate = 1;
    speech.pitch = 1;
    speech.volume = 1;

    speech.onend = function () {
    isSpeaking = false;
};

    speech.onerror = function (e) {
        console.log("Speech error", e);
        isSpeaking = false;

        if (recognition) {
            try { recognition.start(); } catch (err) {}
        }
    };

    setTimeout(function () {
        speechSynthesis.speak(speech);
    }, 100);
}

let fakeProgress = 0;

const fakeTimer = setInterval(() => {
    fakeProgress += Math.random() * 6;

    if (fakeProgress > 88) fakeProgress = 88;

    loaderFill.style.width = fakeProgress + '%';
    loadPercent.textContent = Math.round(fakeProgress) + '%';
}, 400);

Promise.all([
    cocoSsd.load({ base: 'lite_mobilenet_v2' }),
    mobilenet.load(),
    tmImage.load(TM_URL + "model.json", TM_URL + "metadata.json")
])
.then(([loadedCoco, loadedMobileNet, loadedTM]) => {
    cocoModel = loadedCoco;
    mobileNetModel = loadedMobileNet;
    tmModel = loadedTM;

    clearInterval(fakeTimer);

    loaderFill.style.width = '100%';
    loadPercent.textContent = '100%';
    loadMsg.textContent = 'READY';

    setTimeout(() => {
        overlay.style.display = 'none';
        startBtn.disabled = false;
        switchBtn.disabled = false;
        setStatus('', 'READY');
        log('Models loaded successfully');
    }, 400);
})
.catch(err => {
    clearInterval(fakeTimer);
    loadMsg.textContent = 'FAILED';
    console.error(err);
});

async function enableCam() {
    if (!cocoModel || !mobileNetModel || !tmModel || isPredicting) return;

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: facingMode },
            audio: false
        });

        video.srcObject = stream;
        placeholder.style.display = 'none';

        await video.play();

        isPredicting = true;

        startBtn.disabled = true;
        stopBtn.disabled = false;

        setStatus('active', 'DETECTING');
        log('Camera started');

        speechSynthesis.cancel();
log("Detection started");

        predictLoop();

    } catch (err) {
        console.log(err);
        log('Camera access denied', 'warn');
    }
}

async function predictLoop() {
    if (!isPredicting) return;

    try {
        const predictions = await cocoModel.detect(video);

        if (!isPredicting) return;

        children.forEach(c => {
            if (liveView.contains(c)) liveView.removeChild(c);
        });

        children.length = 0;

        let seen = [];

        const scaleX = liveView.offsetWidth / video.videoWidth;
        const scaleY = liveView.offsetHeight / video.videoHeight;

        predictions.forEach(pred => {
            if (pred.score < 0.75) return;

            const [x, y, w, h] = pred.bbox;

            const box = document.createElement('div');
            box.className = 'highlighter';
            box.style.left = (x * scaleX) + 'px';
            box.style.top = (y * scaleY) + 'px';
            box.style.width = (w * scaleX) + 'px';
            box.style.height = (h * scaleY) + 'px';

            const lbl = document.createElement('div');
            lbl.className = 'label';
            lbl.style.left = (x * scaleX) + 'px';
            lbl.style.top = Math.max(0, (y * scaleY) - 22) + 'px';
            lbl.textContent = pred.class + " " + Math.round(pred.score * 100) + "%";

            liveView.appendChild(box);
            liveView.appendChild(lbl);

            children.push(box, lbl);

            seen.push(pred.class);
        });

      if (seen.length > 0) {

    log('COCO detected: ' + seen.join(', '));

    const msg = seen[0];

    if (msg === cocoStablePrediction) {
        cocoStableCount++;
    } else {
        cocoStablePrediction = msg;
        cocoStableCount = 1;
    }

    if (cocoStableCount >= COCO_STABILITY) {

        const now = Date.now();

        if (
            msg !== lastSpoken ||
            now - lastSpokenTime > REPEAT_INTERVAL
        ) {
            speak(msg);
            lastSpoken = msg;
            lastSpokenTime = now;
        }
    }
}
         else {
            const tmPredictions = await tmModel.predict(video);

            if (!isPredicting) return;

            let bestTM = tmPredictions.reduce((a, b) =>
                a.probability > b.probability ? a : b
            );

            const tmSorted = [...tmPredictions].sort((a, b) => b.probability - a.probability);

            const topProb = tmSorted[0].probability;
            const secondProb = tmSorted.length > 1 ? tmSorted[1].probability : 0;
            const tmMargin = topProb - secondProb;

            if (bestTM.probability > 0.92 && tmMargin > 0.30) {
                const tmName = bestTM.className;

                log('TM Detected: ' + tmName + ' (' + Math.round(topProb * 100) + '%)');

                const now = Date.now();

                if (tmName !== lastSpoken || now - lastSpokenTime > REPEAT_INTERVAL) {
                    speak(tmName);
                    lastSpoken = tmName;
                    lastSpokenTime = now;
                }
            } else {
                log('Trying MobileNet');

                const classifications = await mobileNetModel.classify(video);

                if (!isPredicting) return;

                if (
    classifications.length > 0 &&
    classifications[0].probability > 0.75
) {
                    const best = classifications[0];

                    const cleanName = best.className.split(',')[0];
                    const blockedClasses = [
    "pen",
    "pencil",
    "charger",
    "remote control"
];

if (blockedClasses.includes(cleanName.toLowerCase())) {
    return;
}
                    const confidence = Math.round(best.probability * 100);

                    log('MobileNet: ' + cleanName + ' (' + confidence + '%)');

                    if (cleanName === stablePrediction) {
                        stableCount++;
                    } else {
                        stablePrediction = cleanName;
                        stableCount = 1;
                    }

                    if (stableCount >= REQUIRED_STABILITY) {
                        const now = Date.now();

                        if (cleanName !== lastSpoken || now - lastSpokenTime > REPEAT_INTERVAL) {
                            speak(cleanName);
                            lastSpoken = cleanName;
                            lastSpokenTime = now;
                        }
                    }
                }
            }
        }

        if (isPredicting) {
            animationId = setTimeout(predictLoop, DETECTION_INTERVAL);
        }

    } catch (err) {
        console.error("Prediction error:", err);

        if (isPredicting) {
            animationId = setTimeout(predictLoop, DETECTION_INTERVAL);
        }
    }
}

async function switchCamera() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
    }

    facingMode = facingMode === 'environment' ? 'user' : 'environment';

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode }
        });

        video.srcObject = stream;

        log('Camera switched');

    } catch (err) {
        log('Switch failed', 'warn');
        console.error(err);
    }
}

function stopCam() {
    isPredicting = false;

    if (animationId) {
        clearTimeout(animationId);
        animationId = null;
    }

    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        video.srcObject = null;
        stream = null;
    }

    speechSynthesis.cancel();

    children.forEach(c => {
        if (liveView.contains(c)) liveView.removeChild(c);
    });

    children.length = 0;

    placeholder.style.display = 'flex';

    startBtn.disabled = false;
    stopBtn.disabled = true;

    lastSpoken = '';
    stablePrediction = '';
    stableCount = 0;
    cocoStablePrediction = '';
cocoStableCount = 0;

    setStatus('', 'READY');
    log('Camera stopped');
}

const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();

    recognition.continuous = true;
    recognition.lang = 'en-US';
    recognition.interimResults = false;

    recognition.onstart = function () {
    console.log("Voice recognition started");
};

    recognition.onresult = function (event) {
        if (isSpeaking) return;

        const transcript =
            event.results[event.results.length - 1][0].transcript
                .toLowerCase()
                .trim();

        console.log("Heard:", transcript);
        log("Voice: " + transcript);

        if (transcript.includes("start camera")) {
            if (!isPredicting) {
                speak("Starting camera");

                setTimeout(() => {
                    enableCam();
                }, 500);
            }
        }

        if (transcript.includes("stop camera")) {
            if (isPredicting) {
                speak("Stopping camera");

                setTimeout(() => {
                    stopCam();
                }, 500);
            }
        }

        if (transcript.includes("switch camera")) {
            speak("Switching camera");

            setTimeout(() => {
                switchCamera();
            }, 500);
        }
    };

    recognition.onerror = function (event) {
        console.log("Speech recognition error:", event.error);

        if (
            event.error === 'no-speech' ||
            event.error === 'audio-capture'
        ) return;

        log("Voice error: " + event.error, 'warn');
    };

recognition.onend = function () {
    setTimeout(() => {
        if (!isSpeaking) {
            try {
                recognition.start();
            } catch (e) {}
        }
    }, 800);
};

    try {
        recognition.start();
    } catch (e) {}

} else {
    console.log("Speech Recognition not supported");
    log("Speech recognition not supported", "warn");
}

startBtn.addEventListener('click', enableCam);
stopBtn.addEventListener('click', stopCam);
switchBtn.addEventListener('click', switchCamera);
