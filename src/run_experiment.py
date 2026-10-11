from face_recognition_app import RESULT_DIR, train_model


if __name__ == "__main__":
    train_model()
    print(f"实验完成，结果已保存到：{RESULT_DIR / 'experiment_result.txt'}")
