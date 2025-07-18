from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Body
from ultralytics import YOLO
from io import BytesIO
import base64
from PIL import Image
import uvicorn
import model_cv as cvm
import cv2
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

model = YOLO("./bestbestbest.pt")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        encoded_data = data.split(",")[1]
        img_bytes = base64.b64decode(encoded_data)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        results = model(img)
        detections = []
        for box in results[0].boxes:
            xyxy = box.xyxy.tolist()[0]
            conf = box.conf.tolist()[0]
            cls = int(box.cls.tolist()[0])
            name = model.names[cls]
            state = "success"
            if (name == "spoils" or name == "broken") :
                state = "error"
                
            detections.append({
                "xmin": xyxy[0],
                "ymin": xyxy[1],
                "xmax": xyxy[2],
                "ymax": xyxy[3],
                "conf": conf,
                "name": name,
                "state": state
            })
            
            
                
        await websocket.send_json(detections)

@app.post("/opencv-check")
async def opencv_check(data: dict = Body(...)):
    try:
        img1_b64 = img2_b64 = img3_b64 = None
        img_data = data['image'].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        # OpenCV는 바이트를 numpy 배열로 변환해서 처리
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

         # === 여기에 OpenCV 판정 로직 적용 ===

        # 1) 컵 이미지 크롭
        crop_img = cvm.crop_cup_from_image(img)  # crop_cup_from_image는 외부 함수임

        if crop_img is None:
            result = "판정 불가 (크롭 실패)"
        else:
            # 2) 원형성 분석
            circularity, img2_b64 = cvm.analyze_circularity(crop_img)
            threshold = 0.9
            if circularity is None:
                result = "판정 불가 (원형성 분석 실패)"
            elif circularity > threshold:
                # 3) 빨간 결함 체크
                defect, img1_b64, img2_b64, img3_b64 = cvm.check_red_defect(crop_img)
                if defect == True:
                    result = "NG (spoil)"
                else:
                    result = "OK"
            else:
                result = "NG (broken)"

            # 4) 판정과정 이미지에 시각적 표시 (예: 빨간 사각형 또는 텍스트)
            # 여기선 예시로 빨간 테두리 그리기
            height, width = crop_img.shape[:2]
            cv2.rectangle(crop_img, (0, 0), (width-1, height-1), (0, 0, 255), 3)

            # 판정 결과를 원본 이미지 위에 텍스트로 표시
            cv2.putText(img, f"Result: {result}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # 전체 이미지에 크롭된 컵 이미지를 덮어씌우기 (선택사항)
            # 예) 원본 이미지 좌측 상단에 크롭 이미지 붙이기
            # img[0:height, 0:width] = crop_img

        # 결과 이미지를 base64로 다시 인코딩해서 보내기
        _, buffer = cv2.imencode('.jpg', img)
        img_encoded = base64.b64encode(buffer).decode('utf-8')
        img_data_url = f"data:image/jpeg;base64,{img_encoded}"

        return {
            "result": result,
            "image": img_data_url,
            "img1": img1_b64,
            "img2": img2_b64,
            "img3": img3_b64
        }
    except Exception as e:
        print("OpenCV 처리 중 에러:", e)
        return {"result": "ERROR", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
