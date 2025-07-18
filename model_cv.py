import os
import cv2
import base64
import numpy as np
# import matplotlib.pyplot as plt  # 시각화는 안 씀


def check_red_defect(test_img_path: str) -> bool:
    # ======================================
    # === [1] 검사 이미지 로드 + 블러
    # ======================================
    test_img = test_img_path
    if test_img is None:
        raise FileNotFoundError("검사 이미지 경로 확인하세요")

    test_img = cv2.GaussianBlur(test_img, (3, 3), 0)

    # ======================================
    # === [2] 컵 영역만 추출 (Otsu)
    # ======================================
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    cup_only_img = cv2.bitwise_and(test_img, test_img, mask=mask)

    # ======================================
    # === [3] 컵 외곽선 연두색으로 그림
    # ======================================
    COLOR = (0, 200, 0)
    thickness = 5
    cv2.drawContours(cup_only_img, contours, -1, COLOR, thickness)

    # ======================================
    # === [4] 정상 컵 이미지 로드
    # ======================================
    ref_img_path = 'C:/Users/한국전파진흥협회/OneDrive/바탕 화면/Flask_server/test_img/ok_main.jpg'
    ref_img = cv2.imread(ref_img_path)
    if ref_img is None:
        raise FileNotFoundError("정상 이미지 경로 확인하세요")

    h, w = ref_img.shape[:2]
    cup_only_img = cv2.resize(cup_only_img, (w, h))
    ref_img = cv2.resize(ref_img, (w, h))

    # ======================================
    # === [5] 조명 평탄화 & 원형 마스크
    # ======================================
    gray_test = cv2.cvtColor(cup_only_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_test, (51, 51), 0)
    divided = cv2.divide(gray_test, blurred, scale=255)

    _, thresh = cv2.threshold(divided, 30, 255, cv2.THRESH_BINARY)
    contours_circle, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours_circle, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
    circle_mask = np.zeros(gray_test.shape, dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, -1)

    # ======================================
    # === [6] 정상 범위 마스크 & 결함 마스크 (연두 테두리 포함)
    # ======================================

    # 1) 정상 컵 HSV 범위
    ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = cv2.split(ref_hsv)
    mean_h, std_h = np.mean(h_), np.std(h_)
    mean_s, std_s = np.mean(s_), np.std(s_)
    mean_v, std_v = np.mean(v_), np.std(v_)

    lower = np.array([
        max(0, mean_h - 2*std_h),
        max(0, mean_s - 2*std_s),
        max(0, mean_v - 2*std_v)
    ], dtype=np.uint8)
    upper = np.array([
        min(179, mean_h + 2*std_h),
        min(255, mean_s + 2*std_s),
        min(255, mean_v + 2*std_v)
    ], dtype=np.uint8)

    test_hsv = cv2.cvtColor(cup_only_img, cv2.COLOR_BGR2HSV)
    normal_mask = cv2.inRange(test_hsv, lower, upper)

    # 2) 흰색 & 검정 포함 유지
    lower_white = np.array([0, 0, 120])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(test_hsv, lower_white, upper_white)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    black_mask = cv2.inRange(test_hsv, lower_black, upper_black)

    # 3) ✅ 연두 테두리 포함
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])
    green_mask = cv2.inRange(test_hsv, lower_green, upper_green)

    # 4) 정상 범위 + 흰색 + 검정 + 연두
    normal_combined = cv2.bitwise_or(normal_mask, white_mask)
    normal_combined = cv2.bitwise_or(normal_combined, black_mask)
    normal_combined = cv2.bitwise_or(normal_combined, green_mask)

    # 5) 결함 = 정상 제외
    defect_mask = cv2.bitwise_not(normal_combined)
    defect_in_circle = cv2.bitwise_and(defect_mask, defect_mask, mask=circle_mask)

    # ======================================
    # === [7] 결함 빨간색 표시 & 연두색 제거
    # ======================================
    result = cup_only_img.copy()

    green_lower = np.array([0, 180, 0])
    green_upper = np.array([0, 255, 0])
    green_mask = cv2.inRange(result, green_lower, green_upper)
    result[green_mask > 0] = [0, 0, 0]

    result[defect_in_circle > 0] = [0, 0, 255]

    # ======================================
    # === [8] 연두 컨투어와 빨간 결함 겹침 제거
    # ======================================
    green_hsv = cv2.cvtColor(cup_only_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])
    green_contour_mask = cv2.inRange(green_hsv, lower_green, upper_green)

    lower_red_bgr = np.array([0, 0, 150])
    upper_red_bgr = np.array([80, 80, 255])
    red_mask_final = cv2.inRange(result, lower_red_bgr, upper_red_bgr)

    overlap_mask = cv2.bitwise_and(green_contour_mask, red_mask_final)
    result[overlap_mask > 0] = [0, 0, 0]

    # ======================================
    # === ✅ 컵 내부 빨간 결함 여부 (컨투어 기반)
    # ======================================
    red_in_circle = cv2.bitwise_and(red_mask_final, circle_mask)

    contours, _ = cv2.findContours(red_in_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    has_large_defect = False
    min_defect_area = 100  # 픽셀 면적 임계치

    for cnt in contours:
        if cv2.contourArea(cnt) > min_defect_area:
            has_large_defect = True
            break
        
        
        
     # 웹 출력용 base64 인코딩 이미지 생성
    base64_cup_only_img = img_to_base64(cup_only_img)
    base64_result = img_to_base64(result)
    base64_red_in_circle = img_to_base64(red_in_circle)

    # 결함 여부, 그리고 시각화 이미지 3개 같이 반환
    return has_large_defect, base64_cup_only_img, base64_result, base64_red_in_circle    



def analyze_circularity(image_path, resize_to=(128, 128), show_contour=True, circularity_threshold=0.85):
    image = image_path
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return None

    # 1. 전처리
    image = cv2.resize(image, resize_to)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. 컨투어 검출
    contours, _ = cv2.findContours(bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("컨투어를 찾을 수 없습니다.")
        return None

    # 3. 가장 큰 컨투어 선택
    c = max(contours, key=cv2.contourArea)

    # 4. 둘레와 면적 계산
    peri = cv2.arcLength(c, True)
    area = cv2.contourArea(c)
    if peri == 0:
        circularity = 0
    else:
        circularity = (4 * np.pi * area) / (peri ** 2)
        
        
    # 5) 컨투어 그려진 이미지 준비
    contour_img = image.copy()
    cv2.drawContours(contour_img, [c], -1, (0, 255, 0), 2)

    # 6) base64로 변환
    contour_img_b64 = img_to_base64(contour_img)

    return circularity, contour_img_b64


def crop_cup_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None  # 검출 실패

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return image[y:y+h, x:x+w]


def img_to_base64(img):
    # OpenCV 이미지 → JPEG 인코딩 → base64 문자열 반환
    _, buffer = cv2.imencode('.jpg', img)
    img_bytes = buffer.tobytes()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{img_b64}"