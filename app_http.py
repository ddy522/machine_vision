# app.py
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO  # ✅ ultralytics에서 YOLO 불러옴
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# [1] YOLO 모델 로드 (자기 모델로 교체)
model = YOLO('./bestbestbest.pt')  # ✅ torch.hub 대신 ultralytics 사용

@app.route('/')
def index():
    return render_template('index_http.html')

@app.route('/predict', methods=['POST'])
def predict():
    # [2] 클라이언트에서 base64 이미지 수신
    data = request.json['image']
    encoded_data = data.split(',')[1]
    img_bytes = base64.b64decode(encoded_data)
    img = Image.open(BytesIO(img_bytes)).convert('RGB')

    # [3] YOLO 추론
    results = model(img)  # ✅ ultralytics YOLO는 리스트 반환
    detections = []
    for box in results[0].boxes:
        xyxy = box.xyxy.tolist()[0]
        conf = box.conf.tolist()[0]
        cls = int(box.cls.tolist()[0])
        name = model.names[cls]
        detections.append({
            'xmin': xyxy[0],
            'ymin': xyxy[1],
            'xmax': xyxy[2],
            'ymax': xyxy[3],
            'conf': conf,
            'name': name
        })

    # [4] 박스 좌표만 JSON으로 반환
    return jsonify(detections)

if __name__ == '__main__':
    app.run(debug=True)
