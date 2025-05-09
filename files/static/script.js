const videoElement = document.getElementById("webcam");
const canvasElement = document.getElementById("overlay");
const canvasCtx = canvasElement.getContext("2d");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const rebaScoreElement = document.getElementById("rebaScore");
const riskLevelElement = document.getElementById("riskLevel");
const scoreChartCanvas = document.getElementById("scoreChart");
const clipsListElement = document.getElementById("clipsList");
const manualInputsForm = document.getElementById("manualInputsForm");

let pose = null;
let camera = null;
let animationFrameId = null;
let scoreChart = null;
let lastSentTime = 0;
const sendInterval = 1000; // 1秒ごとに送信

let mediaRecorder = null;
let recordedBlobs = [];
let recordingStream = null;
const clipDuration = 5000; // 5秒間のクリップ
let highRiskTimer = null; // 高リスク状態が一定時間続いた場合に録画開始するためのタイマー
const highRiskDurationThreshold = 2000; // 2秒間高リスクが続いたら録画開始
let isCurrentlyRecording = false;

// Mediapipe Poseの初期化
function initializePose() {
    pose = new Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });
    pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        smoothSegmentation: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    pose.onResults(onPoseResults);
}

// カメラの初期化と映像取得開始
async function startCamera() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 }, 
            audio: false // 音声は不要
        });
        videoElement.srcObject = stream;
        videoElement.onloadedmetadata = () => {
            videoElement.play();
            // 録画用のストリームも準備
            recordingStream = videoElement.captureStream ? videoElement.captureStream() : videoElement.mozCaptureStream ? videoElement.mozCaptureStream() : null;
            if (!recordingStream) {
                console.warn("captureStream API is not supported by this browser. Video clip recording might not work.");
                //代替としてcanvasからストリームを取得することも検討できるが、パフォーマンス影響大
            }
        };

        camera = new Camera(videoElement, { // CameraクラスはMediapipeのデモで使われるものと仮定
            onFrame: async () => {
                if (videoElement.readyState >= 3) { 
                    await pose.send({ image: videoElement });
                }
            },
            width: 640,
            height: 480
        });
        await camera.start(); // camera.start()がストリーム設定を上書きしないか注意
    } else {
        throw new Error("getUserMedia not supported");
    }
}

// 姿勢推定結果の処理
function onPoseResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
    }

    if (results.poseLandmarks) {
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: "#00FF00", lineWidth: 4 });
        drawLandmarks(canvasCtx, results.poseLandmarks, { color: "#FF0000", lineWidth: 2 });

        const currentTime = Date.now();
        if (currentTime - lastSentTime >= sendInterval) {
            sendLandmarksToBackend(results.poseLandmarks);
            lastSentTime = currentTime;
        }
    }
    canvasCtx.restore();
}

// 手入力データを取得
function getManualInputs() {
    const formData = new FormData(manualInputsForm);
    const manualInputs = {};
    for (const [key, value] of formData.entries()) {
        const element = manualInputsForm.elements[key];
        if (element.type === "checkbox") {
            manualInputs[key] = element.checked;
        } else if (element.type === "select-one") {
            manualInputs[key] = isNaN(parseFloat(value)) ? value : parseFloat(value);
        } else {
            manualInputs[key] = value;
        }
    }
    return manualInputs;
}

// ランドマークデータをバックエンドに送信
async function sendLandmarksToBackend(landmarks) {
    const manualInputs = getManualInputs();
    const filmingSide = manualInputsForm.elements["filmingSide"].value;
    const processedLandmarks = landmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z, visibility: lm.visibility }));

    try {
        const response = await fetch("http://localhost:8001/calculate_reba", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ landmarks: processedLandmarks, manual_inputs: manualInputs, filming_side: filmingSide })
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`Backend error: ${response.status} ${errorData.detail || response.statusText}`);
        }
        const data = await response.json();
        updateScoreDisplay(data.reba_score, data.risk_level);
        updateScoreChart(data.reba_score);

        if (data.is_high_risk) {
            if (!highRiskTimer && !isCurrentlyRecording) {
                console.log("High risk detected, starting timer for recording...");
                highRiskTimer = setTimeout(() => {
                    if (data.is_high_risk) { // 再度確認（スコアが短時間で変動する可能性のため）
                       startRecordingClip();
                    }
                    highRiskTimer = null;
                }, highRiskDurationThreshold);
            }
        } else {
            if (highRiskTimer) {
                clearTimeout(highRiskTimer);
                highRiskTimer = null;
                console.log("Risk level lowered, recording timer cancelled.");
            }
        }
    } catch (error) {
        console.error("Error sending/receiving data:", error);
        updateScoreDisplay(null, `エラー: ${error.message}`);
    }
}

// スコア表示を更新
function updateScoreDisplay(score, riskLevel) {
    rebaScoreElement.textContent = score !== null && score !== undefined ? score.toFixed(1) : "-";
    riskLevelElement.textContent = riskLevel || "評価待ち";
}

// スコアグラフを初期化・更新
function initializeScoreChart() {
    if (scoreChart) scoreChart.destroy();
    scoreChart = new Chart(scoreChartCanvas, {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "REBAスコア",
                data: [],
                borderColor: "rgb(75, 192, 192)",
                tension: 0.1
            }]
        },
        options: {
            scales: { y: { beginAtZero: true, suggestedMax: 15 } },
            animation: false
        }
    });
}

function updateScoreChart(score) {
    if (!scoreChart || score === null || score === undefined) return;
    const label = new Date().toLocaleTimeString();
    scoreChart.data.labels.push(label);
    scoreChart.data.datasets[0].data.push(score);
    const maxDataPoints = 60; // グラフに表示する最大データポイント数 (例: 60秒分)
    if (scoreChart.data.labels.length > maxDataPoints) {
        scoreChart.data.labels.shift();
        scoreChart.data.datasets[0].data.shift();
    }
    scoreChart.update("quiet"); // アニメーションなしで更新
}

// 動画クリップ録画開始
function startRecordingClip() {
    if (isCurrentlyRecording || !recordingStream) {
        console.log("Already recording or stream not available.");
        return;
    }
    isCurrentlyRecording = true;
    recordedBlobs = [];
    try {
        mediaRecorder = new MediaRecorder(recordingStream, { mimeType: "video/webm; codecs=vp9" });
    } catch (e) {
        console.error("Exception while creating MediaRecorder:", e);
        try {
            mediaRecorder = new MediaRecorder(recordingStream, { mimeType: "video/webm; codecs=vp8" });
        } catch (e2) {
            console.error("Exception while creating MediaRecorder (fallback):", e2);
            alert("MediaRecorder is not supported with available codecs.");
            isCurrentlyRecording = false;
            return;
        }
    }

    console.log("Started recording clip");
    mediaRecorder.onstop = (event) => {
        console.log("Recorder stopped");
        const superBuffer = new Blob(recordedBlobs, { type: "video/webm" });
        addClipToList(superBuffer);
        isCurrentlyRecording = false;
    };
    mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
            recordedBlobs.push(event.data);
        }
    };
    mediaRecorder.start();
    setTimeout(() => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
        }
    }, clipDuration);
}

// 録画されたクリップをリストに追加
function addClipToList(videoBlob) {
    const clipURL = URL.createObjectURL(videoBlob);
    const li = document.createElement("li");
    const video = document.createElement("video");
    video.controls = true;
    video.src = clipURL;
    video.width = 320; // サムネイルサイズ
    const downloadLink = document.createElement("a");
    downloadLink.href = clipURL;
    downloadLink.download = `reba_clip_${new Date().toISOString()}.webm`;
    downloadLink.textContent = "Download Clip";
    downloadLink.style.display = "block";
    li.appendChild(video);
    li.appendChild(downloadLink);
    clipsListElement.appendChild(li);
}

// 開始ボタンの処理
startButton.addEventListener("click", async () => {
    try {
        if (!pose) initializePose();
        await startCamera();
        startButton.disabled = true;
        stopButton.disabled = false;
        initializeScoreChart();
        lastSentTime = Date.now();
        console.log("REBAスコア算出開始");
    } catch (error) {
        console.error("カメラの起動に失敗しました:", error);
        alert(`カメラの起動に失敗しました: ${error.message}`);
        startButton.disabled = false;
        stopButton.disabled = true;
    }
});

// 停止ボタンの処理
stopButton.addEventListener("click", () => {
    if (camera) {
        camera.stop();
        camera = null; // Cameraインスタンスをクリア
    }
    if (videoElement.srcObject) {
        videoElement.srcObject.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
    }
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
    }
    if (highRiskTimer) {
        clearTimeout(highRiskTimer);
        highRiskTimer = null;
    }
    isCurrentlyRecording = false;
    startButton.disabled = false;
    stopButton.disabled = true;
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    updateScoreDisplay(null, "評価待ち");
    console.log("REBAスコア算出停止");
});

// 初期状態設定
stopButton.disabled = true;
initializeScoreChart();
console.log("script.js loaded and initialized for recording.");

