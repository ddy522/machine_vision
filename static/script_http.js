// static/script.js

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// [1] 웹캠 켜기
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
});

// [2] 일정 간격으로 캡처 & 서버로 전송
video.addEventListener('play', () => {
  const sendFrame = () => {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const image_data_url = canvas.toDataURL('image/jpeg');

    fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: image_data_url })
    })
    .then(response => response.json())
    .then(detections => {
      // [3] 캔버스 초기화 & 다시 그림
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      detections.forEach(det => {
        const [x1, y1, x2, y2] = [det.xmin, det.ymin, det.xmax, det.ymax];
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = 'red';
        ctx.fillText(det.name + " " + det.conf.toFixed(2), x1, y1 > 10 ? y1 - 5 : y1 + 15);
      });
    });

    setTimeout(sendFrame, 1000); // 1초 간격
  };

  sendFrame();
});
