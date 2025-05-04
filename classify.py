from tqdm import tqdm
import os
import shutil
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor
import threading

# GPU 메모리 자동 할당
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(physical_gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# 경로 설정
SAMPLE_DIR = "sample/sample"
RESULT_DIR = "result"
#MODEL_PATH = "sample/model/face_recognition_model.h5"
#LABEL_ENCODER_PATH = "sample/model/label_encoder.npy"
MODEL_PATH = "sample/model/face_recognition_model_retrained.h5"
LABEL_ENCODER_PATH = "sample/model/label_encoder_updated.npy"
CHILDREN_NAMES = ["rahee", "hwi", "lia"]
NUM_WORKERS = 128

# 디렉토리 준비
os.makedirs(RESULT_DIR, exist_ok=True)
for name in CHILDREN_NAMES + ["기타"]:
    os.makedirs(os.path.join(RESULT_DIR, name), exist_ok=True)

# 호환성 이슈 없는 모델 로딩
custom_objects = {
    'Functional': tf.keras.Model  # weight_decay 이슈 우회
}
model = load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
label_encoder = np.load(LABEL_ENCODER_PATH, allow_pickle=True)

# 얼굴 탐지기 초기화
detector = MTCNN()
lock = threading.Lock()  # 파일 복사를 위한 스레드 동기화

def align_face(img, landmarks):
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def extract_faces(image_path, required_size=(160, 160)):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[경고] 이미지 로드 실패: {image_path}")
            return []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb)

        faces = []
        for result in results:
            x1, y1, w, h = result['box']
            x2, y2 = x1 + w, y1 + h
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)

            face = rgb[y1:y2, x1:x2]
            padding_h, padding_w = int(h * 0.2), int(w * 0.2)
            ex1 = max(0, x1 - padding_w)
            ey1 = max(0, y1 - padding_h)
            ex2 = min(rgb.shape[1], x2 + padding_w)
            ey2 = min(rgb.shape[0], y2 + padding_h)
            extended_face = rgb[ey1:ey2, ex1:ex2]

            try:
                aligned = align_face(extended_face, result['keypoints'])
                face_img = aligned
            except:
                face_img = extended_face

            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                face_yuv = cv2.cvtColor(face_img, cv2.COLOR_RGB2YUV)
                face_yuv[:, :, 0] = cv2.equalizeHist(face_yuv[:, :, 0])
                face_img = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2RGB)

            face_img = cv2.resize(face_img, required_size)
            faces.append(face_img)
        return faces
    except Exception as e:
        print(f"[오류 건너뜀] MTCNN 오류 {os.path.basename(img_path)}: {e}")
        return []

def classify_image_group(img_files, progress):
    for img_file in img_files:
        img_path = os.path.join(SAMPLE_DIR, img_file)
        faces = extract_faces(img_path)

        predicted_names = []

        for face in faces:
            face_array = img_to_array(face)
            face_array = preprocess_input(face_array)
            face_array = tf.convert_to_tensor(np.expand_dims(face_array, axis=0))

            with tf.device('/GPU:0'):
                prediction = model.predict(face_array, verbose=0)[0]

            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]

            predicted_name = label_encoder[predicted_class] if confidence >= 0.6 else "기타"
            predicted_names.append(predicted_name)

        with lock:
            copied = False
            for name in set(predicted_names or ["기타"]):
                dest = os.path.join(RESULT_DIR, name, img_file)
                try:
                    shutil.copy2(img_path, dest)
                    print(f"[{name}] {img_file} 저장 완료")
                    copied = True
                except FileNotFoundError:
                    print(f"[경고] {img_file} 파일이 존재하지 않음")

            if copied and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except FileNotFoundError:
                    pass

        progress.update(1)

def main():
    all_images = [f for f in os.listdir(SAMPLE_DIR)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(SAMPLE_DIR, f))]

    image_groups = np.array_split(all_images, NUM_WORKERS)
    total = len(all_images)

    print(f"{total}장의 이미지를 {NUM_WORKERS}개의 스레드로 분할하여 처리합니다.")

    progress = tqdm(total=total, desc="진행률", unit="img")

    def wrapped_group(group):
        classify_image_group(group, progress)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(wrapped_group, group) for group in image_groups]
        for future in futures:
            future.result()

    progress.close()
    print("모든 이미지 분류 완료!")

if __name__ == "__main__":
    main()