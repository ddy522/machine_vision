from ultralytics import YOLO
import cv2
import numpy as np
import model_cv as cvm


img_ng = '/Users/hadoyi/Documents/현토에버_SW스쿨/ng(오염)_crop/ok_42.jpg'
img_ok = '/Users/hadoyi/Documents/현토에버_SW스쿨/ok_crop/ok_60.jpg'
img_j = '/Users/hadoyi/DEV/flask_pj/test_img/ng2.png'
# yolo test
# model = YOLO('./best.pt')
# results = model.predict('/Users/hadoyi/Documents/현토에버_SW스쿨/ng(오염)_crop/ok_42.jpg', show=True)


# openCV test

threshold = 0.9

image = cv2.imread(img_j)
crop_img = cvm.crop_cup_from_image(image)

result_ac = cvm.analyze_circularity(crop_img)

if result_ac > threshold:
    result_defect = cvm.check_red_defect(crop_img)
    if result_defect == True:
        print("오염불량")
    else:
        print("정상")
else:
    print("찌그러짐")

