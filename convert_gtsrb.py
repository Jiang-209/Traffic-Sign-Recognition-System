import os
import cv2
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# 配置路径
TRAIN_DIR = "E:\Diploma Project\GTSRB\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images"
TEST_DIR = "E:\Diploma Project\GTSRB\GTSRB_Final_Test_Images\GTSRB_test\Final_Test\Images"
TEST_CSV = "E:\Diploma Project\GTSRB\GTSRB_Final_Test_Images\GTSRB_test\Final_Test\Images\GT-final_test.csv"
SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)
IMG_SIZE = 32  # 统一尺寸


# 处理训练集
def load_train_data():
    images = []
    labels = []

    print("加载训练集...")
    for class_id in os.listdir(TRAIN_DIR):
        class_path = os.path.join(TRAIN_DIR, class_id)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(int(class_id))

    X_train = np.array(images)
    y_train = np.array(labels)
    print("训练集 shape:", X_train.shape, y_train.shape)
    return X_train, y_train


# 处理测试集
def load_test_data():
    images = []
    labels = []

    print("加载测试集...")
    df = pd.read_csv(TEST_CSV,sep=';')
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(TEST_DIR, row["Filename"])
        label = row["ClassId"]
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(label)

    X_test = np.array(images)
    y_test = np.array(labels)
    print("测试集 shape:", X_test.shape, y_test.shape)
    return X_test, y_test


# 主函数
if __name__ == "__main__":
    X_train, y_train = load_train_data()
    X_test, y_test = load_test_data()

    # 保存为 .p 文件
    with open(os.path.join(SAVE_DIR, "train.p"), "wb") as f:
        pickle.dump({'features': X_train, 'labels': y_train}, f)

    with open(os.path.join(SAVE_DIR, "test.p"), "wb") as f:
        pickle.dump({'features': X_test, 'labels': y_test}, f)

    print("✅ 数据处理完成，train.p 和 test.p 已生成！")