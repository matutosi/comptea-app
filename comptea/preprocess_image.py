# pip install opencv-python pillow numpy

import cv2
import numpy as np

def correct_skew(image):
    """画像の傾き補正"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    angle = 0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            # 水平に近い線（±30°以内）のみ使用
            if abs(np.degrees(angle_rad)) < 30:
                angles.append(angle_rad)
        if len(angles) > 0:
            angle = np.degrees(np.mean(angles))
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    corrected = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return corrected

if __name__ == "__main__":
    # 画像の読み込みと傾き補正
    image = cv2.imread('input_skewed.png')
    corrected_image = correct_skew(image)
    cv2.imwrite('corrected_image.png', corrected_image)

    # 画像の前処理を実行
    processed_image = preprocess_image('input.png')
    #処理後の画像を保存
    cv2.imwrite('processed_image.png', processed_image)

    # OCRエンジンの設定も重要
