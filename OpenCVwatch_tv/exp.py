import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelBinarizer

# 函数：读取图像和标签
def read_images_labels(data_folder):
    images = []
    labels = []
    for subdir in os.listdir(data_folder):
        label = subdir
        for filename in os.listdir(os.path.join(data_folder, subdir)):
            img_path = os.path.join(data_folder, subdir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)
    return images, labels

# 函数：划分训练集和验证集
def split_train_validation(images, labels, ratio=0.8):
    num_samples = len(images)
    num_train = int(num_samples * ratio)
    train_images = images[:num_train]
    train_labels = labels[:num_train]
    validation_images = images[num_train:]
    validation_labels = labels[num_train:]
    return train_images, train_labels, validation_images, validation_labels

# 函数：训练人脸识别模型
def train_face_recognition_model(algorithm, images, labels):
    if algorithm == "Eigenface":
        model = cv2.face.EigenFaceRecognizer_create()
    elif algorithm == "FisherFace":
        model = cv2.face.FisherFaceRecognizer_create()
    elif algorithm == "LBPH":
        model = cv2.face.LBPHFaceRecognizer_create()
    else:
        raise ValueError("Invalid algorithm specified!")

    label_binarizer = LabelBinarizer()
    binarized_labels = label_binarizer.fit_transform(labels)

    model.train(images, np.array(binarized_labels))
    return model

# 函数：计算置信度和预测
def compute_confidence_and_prediction(model, test_images):
    confidence_values = []
    predictions = []
    for img in test_images:
        label, confidence = model.predict(img)
        confidence_values.append(confidence)
        predictions.append(label)
    return np.array(confidence_values), np.array(predictions)

# 函数：计算精确度和召回率
def compute_precision_recall(labels, scores, positive_class):
    binary_labels = np.array([1 if label == positive_class else 0 for label in labels])
    precision, recall, _ = precision_recall_curve(binary_labels, scores)
    return precision, recall

# 函数：绘制P-R曲线
def plot_precision_recall_curve(labels, scores, algorithm):
    thresholds = np.linspace(0.7, 1, 10)
    precisions = []
    recalls = []
    for threshold in thresholds:
        precision, recall = compute_precision_recall(labels, scores, positive_class=algorithm)
        precisions.append(precision)
        recalls.append(recall)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    avg_precision = np.mean(precisions, axis=1)
    avg_recall = np.mean(recalls, axis=1)
    plt.plot(avg_recall, avg_precision, label=algorithm)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

# 主程序
if __name__ == "__main__":
    test_folder = "C:/Users/YC/Desktop/archive"  # 测试数据文件夹路径
    algorithms = ["Eigenface", "FisherFace", "LBPH"]  # 三种算法

    # 读取测试数据
    test_images, test_labels = read_images_labels(test_folder)

    # 划分训练集和验证集（4:1）
    train_images, train_labels, validation_images, validation_labels = split_train_validation(test_images, test_labels)
    print("训练集标签数量:", len(train_labels))
    print("验证集标签数量:", len(validation_labels))
    for algorithm in algorithms:
        # 训练模型
        model = train_face_recognition_model(algorithm, train_images, train_labels)

        # 计算验证集置信度和预测
        confidence_values, _ = compute_confidence_and_prediction(model, validation_images)
        print(len(validation_labels), len(confidence_values))

        # 绘制P-R曲线
        plot_precision_recall_curve(validation_labels, confidence_values, algorithm)
