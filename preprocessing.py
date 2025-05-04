import os
import cv2
from mtcnn import MTCNN
import numpy as np

# 경로 설정
train_dir = 'train'
child_names = ['rahee', 'hwi', 'lia']

# 해상도 기준
MIN_WIDTH = 100
MIN_HEIGHT = 100
MAX_WIDTH = 800

# 얼굴 탐지기 초기화
detector = MTCNN()

def is_valid_image(img_path):
    try:
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"[스킵] {img_path} - 이미지 로드 실패")
            return None

        h, w = image_bgr.shape[:2]

        # 너무 작은 이미지 제거
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            print(f"[삭제] {img_path} - 해상도 너무 낮음")
            return False

        # 너무 큰 이미지 리사이즈
        if w > MAX_WIDTH:
            ratio = MAX_WIDTH / w
            image_bgr = cv2.resize(image_bgr, (MAX_WIDTH, int(h * ratio)))

        # RGB로 변환
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 얼굴 탐지
        results = detector.detect_faces(image_rgb)

        if len(results) != 1:
            print(f"[삭제] {img_path} - 얼굴 수: {len(results)}")
            return False

        return True

    except Exception as e:
        print(f"[스킵] {img_path} - 오류: {e}")
        return None

def preprocess():
    for name in child_names:
        folder_path = os.path.join(train_dir, name)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            result = is_valid_image(file_path)
            if result is False:
                os.remove(file_path)
            # None인 경우는 오류로 스킵

if __name__ == "__main__":
    preprocess()
    print("전처리 완료 ✅")
