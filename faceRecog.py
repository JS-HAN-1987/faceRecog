import os
import shutil
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from mtcnn import MTCNN

# 경로 설정
TRAIN_DIR = "train"
SAMPLE_DIR = "sample"
OUTPUT_DIR = os.path.join(SAMPLE_DIR)

# 얼굴 감지기 초기화
detector = MTCNN()

# 자녀 이름 정의
children_names = ["rahee", "hwi", "lia"]

# 출력 폴더 생성
for name in children_names + ["기타"]:
    output_folder = os.path.join(OUTPUT_DIR, name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"폴더 생성: {output_folder}")


def align_face(img, landmarks):
    """눈 위치를 기준으로 얼굴을 정렬합니다."""
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']

    # 양쪽 눈의 중심점 계산
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")

    # 두 눈 사이의 각도 계산
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # 이미지 중심 계산
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # 회전 행렬 계산
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 이미지 회전
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    return rotated


def extract_face(image_path, required_size=(160, 160)):
    """이미지에서 얼굴을 감지하고 추출합니다."""
    try:
        # 이미지 로드
        pixels = cv2.imread(image_path)
        if pixels is None:
            print(f"이미지 로드 실패: {image_path}")
            return []

        # BGR에서 RGB로 변환
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

        # 얼굴 감지
        results = detector.detect_faces(pixels)

        face_arrays = []
        for result in results:
            x1, y1, width, height = result['box']
            x2, y2 = x1 + width, y1 + height

            # 경계 확인
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(pixels.shape[1], x2)
            y2 = min(pixels.shape[0], y2)

            # 얼굴 추출
            face = pixels[y1:y2, x1:x2]

            # 확장된 얼굴 영역 (얼굴 주변 컨텍스트 포함)
            # 얼굴 크기의 20% 확장
            h, w = y2 - y1, x2 - x1
            padding_h, padding_w = int(h * 0.2), int(w * 0.2)

            # 확장된 경계 계산 (이미지 경계 체크)
            ex1 = max(0, x1 - padding_w)
            ey1 = max(0, y1 - padding_h)
            ex2 = min(pixels.shape[1], x2 + padding_w)
            ey2 = min(pixels.shape[0], y2 + padding_h)

            # 확장된 얼굴 추출
            extended_face = pixels[ey1:ey2, ex1:ex2]

            # 얼굴 정렬 (눈의 위치가 감지된 경우)
            if 'keypoints' in result:
                try:
                    aligned_face = align_face(extended_face, result['keypoints'])
                    face = aligned_face
                except:
                    face = extended_face  # 정렬 실패 시 확장된 얼굴 사용
            else:
                face = extended_face

            # 히스토그램 평활화 (명암 보정)
            # 컬러 이미지의 경우 루미넌스 채널만 평활화
            if len(face.shape) == 3 and face.shape[2] == 3:
                face_yuv = cv2.cvtColor(face, cv2.COLOR_RGB2YUV)
                face_yuv[:, :, 0] = cv2.equalizeHist(face_yuv[:, :, 0])
                face = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2RGB)

            # 크기 조정
            face = cv2.resize(face, required_size)

            face_arrays.append((face, (x1, y1, x2, y2)))

        return face_arrays
    except Exception as e:
        print(f"얼굴 추출 중 오류 발생: {e}")
        return []


def create_data_generator():
    """데이터 증강을 위한 생성기를 생성합니다."""
    return ImageDataGenerator(
        rotation_range=20,  # 이미지 회전 (±20도)
        width_shift_range=0.1,  # 가로 이동
        height_shift_range=0.1,  # 세로 이동
        shear_range=0.1,  # 전단 변환
        zoom_range=0.1,  # 확대/축소
        horizontal_flip=True,  # 좌우 반전
        brightness_range=[0.8, 1.2]  # 밝기 조절
    )


def load_dataset():
    """학습 데이터셋을 로드하고 전처리합니다."""
    faces = []
    labels = []

    # 각 자녀 폴더에서 이미지 처리
    for child_name in children_names:
        child_dir = os.path.join(TRAIN_DIR, child_name)
        if not os.path.exists(child_dir):
            print(f"경고: {child_dir} 폴더가 존재하지 않습니다.")
            continue

        print(f"{child_name}의 얼굴 데이터 로드 중...")

        # 이미지 파일 처리
        for img_file in os.listdir(child_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(child_dir, img_file)

            # 얼굴 추출
            extracted_faces = extract_face(img_path)

            for face, _ in extracted_faces:
                # 전처리
                face = img_to_array(face)
                face = preprocess_input(face)

                # 데이터셋에 추가
                faces.append(face)
                labels.append(child_name)

    return np.array(faces), np.array(labels)


def build_model(input_shape, num_classes):
    """얼굴 인식 모델을 구축합니다."""
    # MobileNetV2를 기본 모델로 사용
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # 커스텀 레이어 추가
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)  # 노드 증가
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # 모델 생성
    model = Model(inputs=base_model.input, outputs=outputs)

    # 기본 모델 레이어 고정 (이후 미세 조정 가능)
    for layer in base_model.layers:
        layer.trainable = False

    return model


def train_classifier():
    """얼굴 인식 분류기를 학습시킵니다."""
    # 데이터셋 로드
    faces, labels = load_dataset()

    if len(faces) == 0:
        print("학습 데이터가 없습니다. 학습을 진행할 수 없습니다.")
        return None, None

    print(f"로드된 얼굴 이미지: {len(faces)}")

    # 라벨 인코딩
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(
        faces, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    # 모델 구축
    num_classes = len(label_encoder.classes_)
    model = build_model((160, 160, 3), num_classes)

    # 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 데이터 증강 설정
    datagen = create_data_generator()
    datagen.fit(X_train)

    # 조기 종료 설정
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # 모델 체크포인트 설정
    checkpoint_path = os.path.join(OUTPUT_DIR, "model_checkpoint.h5")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # 모델 학습 (데이터 증강 사용)
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=30,  # 더 긴 훈련
        callbacks=[early_stopping, model_checkpoint],
        steps_per_epoch=len(X_train) // 32
    )

    # 모델 평가
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"테스트 정확도: {accuracy:.4f}")

    # 예측 및 분류 리포트 출력
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\n분류 리포트:")
    print(classification_report(
        y_test, y_pred_classes,
        target_names=label_encoder.classes_
    ))

    # 학습 과정 시각화
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='훈련 정확도')
    plt.plot(history.history['val_accuracy'], label='검증 정확도')
    plt.title('모델 정확도')
    plt.xlabel('에폭')
    plt.ylabel('정확도')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='훈련 손실')
    plt.plot(history.history['val_loss'], label='검증 손실')
    plt.title('모델 손실')
    plt.xlabel('에폭')
    plt.ylabel('손실')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))
    plt.close()

    # 모델 저장
    model.save(os.path.join(OUTPUT_DIR, "face_recognition_model.h5"))

    # 학습된 라벨 저장
    np.save(os.path.join(OUTPUT_DIR, "label_encoder.npy"), label_encoder.classes_)

    return model, label_encoder


def classify_faces(model, label_encoder):
    """샘플 이미지에서 얼굴을 인식하고 분류합니다."""
    if not os.path.exists(SAMPLE_DIR):
        print(f"샘플 디렉토리가 존재하지 않습니다: {SAMPLE_DIR}")
        return

    sample_files = [f for f in os.listdir(SAMPLE_DIR)
                    if os.path.isfile(os.path.join(SAMPLE_DIR, f))
                    and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not sample_files:
        print("샘플 이미지를 찾을 수 없습니다.")
        return

    print(f"샘플 이미지 {len(sample_files)}장을 분류합니다...")

    # 분류 결과 저장
    results = {}

    for img_file in sample_files:
        img_path = os.path.join(SAMPLE_DIR, img_file)

        # 얼굴 추출
        extracted_faces = extract_face(img_path)

        if not extracted_faces:
            print(f"이미지에서 얼굴을 찾을 수 없습니다: {img_file}")
            # 얼굴이 없는 경우 기타 폴더로 이동
            dest_path = os.path.join(OUTPUT_DIR, "기타", img_file)
            shutil.copy2(img_path, dest_path)
            continue

        # 이미지 내의 각 얼굴 분류
        for face, face_box in extracted_faces:
            # 전처리
            face_array = img_to_array(face)
            face_array = preprocess_input(face_array)
            face_array = np.expand_dims(face_array, axis=0)

            # 예측
            prediction = model.predict(face_array)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]

            # 예측 결과 가져오기
            predicted_name = label_encoder.classes_[predicted_class]

            # 기준 신뢰도 (0.6 미만은 기타로 분류) - 신뢰도 향상
            if confidence < 0.6:
                predicted_name = "기타"

            print(f"이미지 {img_file}의 얼굴: {predicted_name} (신뢰도: {confidence:.4f})")

            # 결과 저장
            if img_file not in results:
                results[img_file] = []
            results[img_file].append(predicted_name)

            # 얼굴 표시된 이미지 저장
            original_img = cv2.imread(img_path)
            x1, y1, x2, y2 = face_box
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_img, f"{predicted_name} ({confidence:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            debug_folder = os.path.join(OUTPUT_DIR, "debug")
            os.makedirs(debug_folder, exist_ok=True)
            cv2.imwrite(os.path.join(debug_folder, f"debug_{img_file}"), original_img)

    # 결과에 따라 이미지 파일 분류
    for img_file, predicted_names in results.items():
        img_path = os.path.join(SAMPLE_DIR, img_file)

        # 중복 제거
        unique_names = set(predicted_names)

        # 각 이름의 폴더로 복사
        for name in unique_names:
            dest_path = os.path.join(OUTPUT_DIR, name, img_file)
            shutil.copy2(img_path, dest_path)
            print(f"이미지 {img_file}를 {name} 폴더로 복사했습니다.")


def main():
    print("=== 자녀 얼굴 인식 및 사진 분류 프로그램 ===")

    # 모델 학습
    print("\n1. 얼굴 인식 모델 학습 중...")
    model, label_encoder = train_classifier()

    if model is None:
        print("모델 학습에 실패했습니다.")
        return

    # 얼굴 분류
    print("\n2. 샘플 이미지 분류 중...")
    classify_faces(model, label_encoder)

    print("\n분류 완료!")
    print(f"결과는 {OUTPUT_DIR} 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()