# 식기세척기 오염물질 자동 검수 프로젝트

## 📌 개요
- 세척 완료된 그릇의 오염·파손 여부를 AI + 머신비전으로 자동 검수합니다.
- 기존 육안 검사 한계를 극복하여 불량품 재세척·제거를 자동화합니다.

---

## ⚙️ 사용 기술
- **백엔드:** FastAPI, OpenCV, YOLO (Ultralytics)
- **프론트엔드:** HTML, JavaScript (Vanilla)
- **카메라 제어:** MediaDevices API
- **모델:** SVM, Random Forest, CNN, YOLOv5/YOLOv8
- **기타:** Pillow, WebSocket, HTTP API

---

## 🚀 주요 기능
1. **실시간 그릇 이미지 캡처**
2. **OpenCV 기반 이미지 전처리**
3. **머신러닝/딥러닝 모델로 불량 판별**
4. **YOLO 실시간 탐지 (WebSocket)**
5. **불량 판단 후 액션 처리 흐름 시연**

---

## 🧩 프로젝트 구조
```plaintext
├── backend/               # FastAPI 서버 코드
│   ├── main.py            # 서버 진입점
│   ├── models/            # YOLO 모델 관련 코드
│   ├── opencv/            # OpenCV 전처리 모듈
│   ├── static/            # 정적 파일 (예: 샘플 이미지)
│   ├── templates/         # HTML 템플릿
├── frontend/              # 웹페이지 (HTML, JS)
│   ├── index.html
│   ├── script.js
├── data/                  # 샘플 데이터, 라벨링 데이터
├── README.md
```

📂 실행 방법
  ✅ 1) 의존성 설치
  pip install -r requirements.txt
  
  ✅ 2) YOLO 모델 다운로드
  ultralytics 프레임워크 사용
  train2_best.pt 등 가중치 파일 경로 명시
  
  ✅ 3) 서버 실행
  uvicorn main:app --reload
  
  ✅ 4) 웹페이지 접속
  http://localhost:8000 접속
  실시간 탐지 & 정지 이미지 판별 테스트

🔬 주요 알고리즘 설명
  📷 이미지 전처리
  OpenCV: 컵 영역 Crop, Blur, 이진화, Contour 추출
  깨짐 여부: 외곽선 원형도 계산
  오염 여부: HSV 색공간 마스크링

  🧠 모델
  SVM / RF: 특징변수 기반 분류
  CNN: 기본 분류 성능 향상 실험
  YOLO: 실시간 객체 탐지

✅ 추후 개선 계획
실제 상용 식기세척기 라인 연계 실증
카메라 각도/조도 변화 대응 데이터셋 확장
전이학습(ResNet) 최적화
Edge Device 연동 실험
