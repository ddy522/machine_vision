const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const cameraSelect = document.getElementById('cameraSelect');
const modeTitle = document.getElementById('modeTitle');
const yoloBtn = document.getElementById('yoloBtn');
const opencvBtn = document.getElementById('opencvBtn');
const captureBtn = document.getElementById('captureBtn');
const resultArea = document.getElementById('result_area');

let stream = null;
let ws = null;
let yoloActive = false;
let yoloLoopActive = false;

// ✅ 카메라 리스트 가져오기
async function getCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videoDevices = devices.filter(device => device.kind === 'videoinput');

  cameraSelect.innerHTML = '';
  videoDevices.forEach((device, index) => {
    const option = document.createElement('option');
    option.value = device.deviceId;
    option.text = device.label || `Camera ${index + 1}`;
    cameraSelect.appendChild(option);
  });
}

// ✅ 선택된 카메라로 시작
async function startCamera(deviceId) {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }

  stream = await navigator.mediaDevices.getUserMedia({
    video: { deviceId: { exact: deviceId } }
  });
  video.srcObject = stream;
}

// ✅ 카메라 선택 변경 시
cameraSelect.addEventListener('change', async () => {
  await startCamera(cameraSelect.value);
});

// ✅ YOLO 모드 시작
function startYoloMode() {
  modeTitle.textContent = 'YOLO 실시간 객체 탐지';
  yoloActive = true;
  yoloLoopActive = true;

  captureBtn.style.display = 'none';
  canvas.style.display = 'inline-block';
  resultArea.style.display='none';


  if (!ws || ws.readyState !== 1) {
    ws = new WebSocket(`ws://${window.location.host}/ws`);

    ws.onopen = () => {
      console.log('✅ YOLO WebSocket 연결됨');
      startYoloLoop();
    };

    ws.onmessage = (event) => {
      const detections = JSON.parse(event.data);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      detections.forEach(det => {
        const [x1, y1, x2, y2] = [det.xmin, det.ymin, det.xmax, det.ymax];
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = 'red';
        ctx.fillText(`${det.name} ${det.conf.toFixed(2)}`, x1, y1 > 10 ? y1 - 5 : y1 + 15);
        if(det.state == "error"){
          // alert("error")
        }

      });
    };

    ws.onerror = (err) => console.error('WebSocket 에러:', err);
  } else {
    startYoloLoop();
  }
}

// ✅ YOLO 전송 루프
function startYoloLoop() {
  const loop = () => {
    if (!yoloLoopActive) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const image_data_url = canvas.toDataURL('image/jpeg');
    ws.send(image_data_url);

    setTimeout(loop, 3000);
  };
  loop();
}

// ✅ OpenCV 모드 시작
function startOpenCVMode() {
  modeTitle.textContent = 'OpenCV 검사 모드';
  yoloActive = false;
  yoloLoopActive = false;

  captureBtn.style.display = 'inline-block';
  canvas.style.display = 'none';
  resultArea.style.display='inline-block';
}

// ✅ 캡처 → 서버로 전송
captureBtn.addEventListener('click', () => {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const image_data_url = canvas.toDataURL('image/jpeg');
  
  
  fetch('/opencv-check', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: image_data_url })
  })
  .then(res => res.json())
  .then(data => {
    resultArea.innerHTML = `
      <table border="0" style="border-collapse: collapse; margin: 10px 0; width:100%; height:100%">
        <tr>
          <td style="text-align: center;">
            <p style='margin:0px;'>판정 결과: ${data.result}</p>
            <img src="${data.image}" alt="원본 이미지" style="width:250px; height:190px; border:1px solid #ccc;"/>
          </td>
          <td style="text-align: center;">
            <p style='margin:0px;'>결함 부위</p>
            ${data.img2 ? `<img src="${data.img2}" alt="이미지2" style="width:250px;height:190px; border:1px solid #ccc;"/>` : '<p>결함부위 없음</p>'}
          </td>
        </tr>
      </table>
    `;
  })
  .catch(err => console.error(err));
});



// ✅ 버튼 연결
yoloBtn.addEventListener('click', startYoloMode);
opencvBtn.addEventListener('click', startOpenCVMode);

// ✅ 초기화
async function init() {
  await getCameras();
  if (cameraSelect.options.length > 0) {
    await startCamera(cameraSelect.value);
    startYoloMode();
  }
}
init();
