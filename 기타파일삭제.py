import os


def main():
    # 기본 경로 설정
    base_path = "."
    result_path = os.path.join(base_path, "result")

    # 검사할 폴더 경로 설정
    etc_folder = os.path.join(result_path, "기타")
    comparison_folders = [
        os.path.join(result_path, "hwi"),
        os.path.join(result_path, "lia"),
        os.path.join(result_path, "rahee")
    ]

    # 삭제된 파일과 유지된 파일 카운트
    deleted_count = 0
    kept_count = 0

    # 기타 폴더에 파일이 있는지 확인
    if not os.path.exists(etc_folder):
        print(f"'{etc_folder}' 폴더가 존재하지 않습니다.")
        return

    # 비교 폴더들이 모두 존재하는지 확인
    for folder in comparison_folders:
        if not os.path.exists(folder):
            print(f"'{folder}' 폴더가 존재하지 않습니다.")
            return

    # 기타 폴더의 모든 파일 검사
    for filename in os.listdir(etc_folder):
        etc_file_path = os.path.join(etc_folder, filename)

        # 디렉토리는 건너뛰기
        if os.path.isdir(etc_file_path):
            continue

        # 파일이 하나의 비교 폴더라도 있는지 확인
        found_in_any_folder = False

        for folder in comparison_folders:
            comparison_file_path = os.path.join(folder, filename)

            # 파일 이름이 같은 파일이 존재하면 플래그 설정
            if os.path.exists(comparison_file_path):
                found_in_any_folder = True
                break

        # 하나 이상의 폴더에 같은 이름의 파일이 있으면 삭제
        if found_in_any_folder:
            try:
                os.remove(etc_file_path)
                print(f"삭제됨: {filename} (폴더에 중복 발견)")
                deleted_count += 1
            except Exception as e:
                print(f"'{filename}' 삭제 중 오류 발생: {e}")
        else:
            kept_count += 1
            print(f"유지됨: {filename} (중복 없음)")

    # 결과 요약 출력
    print("\n===== 작업 완료 =====")
    print(f"삭제된 파일: {deleted_count}개")
    print(f"유지된 파일: {kept_count}개")


if __name__ == "__main__":
    main()