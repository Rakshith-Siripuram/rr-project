const TM_URL =
  "https://teachablemachine.withgoogle.com/models/b9lFDElzn/";

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
const REPEAT_INTERVAL = 3000;
const REQUIRED_STABILITY = 3;

let lastSpoken = '';
let lastSpokenTime = 0;

let stablePrediction = '';
let stableCount = 0;

speechSynthesis.onvoiceschanged = function () {
    speechSynthesis.getVoices();
};

function setStatus(cls, text){

    statusDot.className =
        'status-dot ' + cls;

    statusText.textContent = text;
}

function log(msg, type){

    logEl.innerHTML =
        type === 'warn'
        ? `<span class="warn">⚠ ${msg}</span>`
        : `<span>›</span> ${msg}`;
}

function speak(text){

    console.log("Speaking:", text);

    speechSynthesis.cancel();

    let speech =
        new SpeechSynthesisUtterance(text);

    speech.rate = 1;
    speech.pitch = 1;
    speech.volume = 1;

    speech.onerror = function(e){
        console.log("Speech error", e);
    };

    setTimeout(function(){

        speechSynthesis.speak(speech);

    },100);
}

let fakeProgress = 0;

const fakeTimer = setInterval(() => {

    fakeProgress += Math.random() * 6;

    if(fakeProgress > 88){
        fakeProgress = 88;
    }

    loaderFill.style.width =
        fakeProgress + '%';

    loadPercent.textContent =
        Math.round(fakeProgress) + '%';

},400);

Promise.all([

    cocoSsd.load({
        base: 'lite_mobilenet_v2'
    }),

    mobilenet.load(),

    tmImage.load(
        TM_URL + "model.json",
        TM_URL + "metadata.json"
    )

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

    },400);

})

.catch(err => {

    clearInterval(fakeTimer);

    loadMsg.textContent =
        'FAILED';

    console.error(err);
});

async function enableCam(){

    if(
        !cocoModel ||
        !mobileNetModel ||
        !tmModel ||
        isPredicting
    ){
        return;
    }

    try{

        stream =
            await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false
            });

        video.srcObject = stream;

        placeholder.style.display = 'none';

        await video.play();

        isPredicting = true;

        startBtn.disabled = true;
        stopBtn.disabled = false;

        setStatus(
            'active',
            'DETECTING'
        );

        log('Camera started');

        speechSynthesis.cancel();

        let startVoice =
            new SpeechSynthesisUtterance(
                "Detection started"
            );

        speechSynthesis.speak(startVoice);

        predictLoop();

    }

    catch(err){

        console.log(err);

        log(
            'Camera access denied',
            'warn'
        );
    }
}

async function predictLoop(){

    if(!isPredicting){
        return;
    }

    try{

        const predictions =
            await cocoModel.detect(video);

        if(!isPredicting){
            return;
        }

        children.forEach(c => {

            if(liveView.contains(c)){

                liveView.removeChild(c);
            }
        });

        children.length = 0;

        let seen = [];

        predictions.forEach(pred => {

            if(pred.score < 0.60){
                return;
            }

            const [x, y, w, h] =
                pred.bbox;

            const box =
                document.createElement('div');

            box.className = 'highlighter';

            box.style.left = x + 'px';
            box.style.top = y + 'px';
            box.style.width = w + 'px';
            box.style.height = h + 'px';

            const lbl =
                document.createElement('div');

            lbl.className = 'label';

            lbl.style.left = x + 'px';
            lbl.style.top =
                Math.max(0, y - 22) + 'px';

            lbl.textContent =
                pred.class +
                " " +
                Math.round(pred.score * 100) +
                "%";

            liveView.appendChild(box);
            liveView.appendChild(lbl);

            children.push(box);
            children.push(lbl);

            seen.push(pred.class);
        });

        if(seen.length > 0){

            log(
                'COCO detected: ' +
                seen.join(', ')
            );

            const msg = seen[0];

            const now = Date.now();

            if(
                msg !== lastSpoken ||
                now - lastSpokenTime >
                REPEAT_INTERVAL
            ){

                speak(msg);

                lastSpoken = msg;
                lastSpokenTime = now;
            }
        }

        else{

            const tmPredictions =
                await tmModel.predict(video);

            let bestTM =
                tmPredictions.reduce((a,b) =>
                    a.probability > b.probability
                    ? a : b
                );

            if(bestTM.probability > 0.85){

                const tmName =
                    bestTM.className;

                log(
                    'TM Detected: ' +
                    tmName
                );

                const now = Date.now();

                if(
                    tmName !== lastSpoken ||
                    now - lastSpokenTime >
                    REPEAT_INTERVAL
                ){

                    speak(tmName);

                    lastSpoken = tmName;
                    lastSpokenTime = now;
                }
            }

            else{

                log('Trying MobileNet');

                const classifications =
                    await mobileNetModel.classify(video);

                if(classifications.length > 0){

                    const best =
                        classifications[0];

                    const cleanName =
                        best.className
                        .split(',')[0];

                    const confidence =
                        Math.round(
                            best.probability * 100
                        );

                    log(
                        'MobileNet: ' +
                        cleanName +
                        ' (' +
                        confidence +
                        '%)'
                    );

                    if(
                        cleanName === stablePrediction
                    ){

                        stableCount++;

                    }else{

                        stablePrediction =
                            cleanName;

                        stableCount = 1;
                    }

                    if(
                        stableCount >=
                        REQUIRED_STABILITY
                    ){

                        const now = Date.now();

                        if(
                            cleanName !== lastSpoken ||
                            now - lastSpokenTime >
                            REPEAT_INTERVAL
                        ){

                            speak(cleanName);

                            lastSpoken = cleanName;

                            lastSpokenTime = now;
                        }
                    }
                }
            }
        }

        animationId =
            setTimeout(
                predictLoop,
                DETECTION_INTERVAL
            );
    }

    catch(err){

        console.error(
            "Prediction error:",
            err
        );

        if(isPredicting){

            animationId =
                setTimeout(
                    predictLoop,
                    DETECTION_INTERVAL
                );
        }
    }
}

async function switchCamera(){

    if(stream){

        stream
        .getTracks()
        .forEach(t => t.stop());
    }

    facingMode =
        facingMode === 'environment'
        ? 'user'
        : 'environment';

    try{

        stream =
            await navigator.mediaDevices
            .getUserMedia({
                video: { facingMode }
            });

        video.srcObject = stream;

        log('Camera switched');

    }

    catch(err){

        log(
            'Switch failed',
            'warn'
        );

        console.error(err);
    }
}

function stopCam(){

    isPredicting = false;

    if(animationId){

        clearTimeout(animationId);

        animationId = null;
    }

    if(stream){

        stream
        .getTracks()
        .forEach(t => t.stop());

        video.srcObject = null;

        stream = null;
    }

    speechSynthesis.cancel();

    children.forEach(c => {

        if(liveView.contains(c)){

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

    log('Camera stopped');
}
/* VOICE COMMAND SYSTEM */
const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;
if (SpeechRecognition) {
    const recognition =
        new SpeechRecognition();
    recognition.continuous = true;
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.onstart = function () {
        console.log("Voice recognition started");
        log("Voice commands enabled");
    };
    recognition.onresult = function (event) {
        const transcript =
            event.results[event.results.length - 1][0]
            .transcript
            .toLowerCase()
            .trim();
        console.log("Heard:", transcript);
        log("Voice: " + transcript);
        /* START CAMERA */
        if (
            transcript.includes("start camera") ||
            transcript.includes("start detection") ||
            transcript.includes("start")
        ) {
            if (!isPredicting) {
               enableCam();
            }
        }
        /* STOP CAMERA */
        if (
            transcript.includes("stop camera") ||
            transcript.includes("stop detection") ||
            transcript.includes("stop")
        ) {
            if (isPredicting) {
                stopCam();
            }
        }
        /* SWITCH CAMERA */
        if (
            transcript.includes("switch camera")
        ) {
            switchCamera();
        }
    };

    recognition.onerror = function (event) {
        console.log(
            "Speech recognition error:",
            event.error
        );
    };
    recognition.onend = function () {
        recognition.start();
    };
    recognition.start();
} else {
    console.log(
        "Speech Recognition not supported"
    );
    log(
        "Speech recognition not supported",
        "warn"
    );
}
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
