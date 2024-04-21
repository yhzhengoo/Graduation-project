import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import time
# 1. 加载数据集
dataset_path = "C:/Users/YC/Desktop/archive"
#image_size = (100, 100)  # 设定图像尺寸

def load_dataset(dataset_path):
    X = []  # 存储图像数据
    y = []  # 存储标签
    for person_id in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_id)
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像
            #img = cv2.resize(img, image_size)  # 调整图像尺寸
            X.append(img.flatten())  # 将图像展平为一维数组
            y.append(int(person_id[1:]))  # 使用文件夹名称去除前缀"s"作为标签
    return np.array(X), np.array(y)

X, y = load_dataset(dataset_path)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 3. 训练三种算法并进行预测


eigenface=cv2.face.EigenFaceRecognizer_create()
eigenface.train(X_train, y_train)
confidence_thresholds = [3300]  # 设置置信度阈值范围
precisions = []
recalls = []

for threshold in confidence_thresholds:
    confidences = []
    correct = 0
    total = 0
    start_time=time.time()
    for i in range(len(X_test)):
        label, confidence = eigenface.predict(X_test[i])

        #confidences.append(confidence)
        if confidence < threshold:
            if label == y_test[i]:
                correct += 1
            total += 1
    #print(confidences)
    precision = correct / total if total != 0 else 1
    recall = correct / len(X_test)
    precisions.append(precision)
    recalls.append(recall)
    end_time=time.time()
print("et:",(end_time-start_time)/len(X_test))
print("ep:",precisions[-1])
print("er:",recalls[-1])

    # 4. 绘制P-R曲线
#plt.plot(recalls, precisions, label='eigenface',linestyle='dashed')

fisherface=cv2.face.FisherFaceRecognizer_create()
fisherface.train(X_train, y_train)
confidence_thresholds = [1200]  # 设置置信度阈值范围
precisions = []
recalls = []
for threshold in confidence_thresholds:
    confidences = []
    correct = 0
    total = 0
    start_time = time.time()
    for i in range(len(X_test)):
        label, confidence = fisherface.predict(X_test[i])

        #confidences.append(confidence)
        if confidence < threshold:
            if label == y_test[i]:
                correct += 1
            total += 1
    #print(confidences)
    precision = correct / total if total != 0 else 1
    recall = correct / len(X_test)
    precisions.append(precision)
    recalls.append(recall)
    end_time = time.time()
    # 4. 绘制P-R曲线
#plt.plot(recalls, precisions, label='fisherface',linestyle='dotted')
print("ft:",(end_time-start_time)/len(X_test))
print("fp:",precisions[-1])
print("fr:",recalls[-1])

#LBPH数据
def load_dataset1(dataset_path):
    X = []  # 存储图像数据
    y = []  # 存储标签
    for person_id in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_id)
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像
            #img = cv2.resize(img, image_size)  # 调整图像尺寸
            X.append(img)  # 将图像展平为一维数组
            y.append(int(person_id[1:]))  # 使用文件夹名称去除前缀"s"作为标签
    return np.array(X), np.array(y)

X, y = load_dataset1(dataset_path)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

lbphface=cv2.face.LBPHFaceRecognizer_create()
lbphface.train(X_train, y_train)
confidence_thresholds = [74]  # 设置置信度阈值范围
precisions = []
recalls = []
for threshold in confidence_thresholds:
    confidences = []
    correct = 0
    total = 0
    start_time = time.time()
    for i in range(len(X_test)):
        label, confidence = lbphface.predict(X_test[i])

        #confidences.append(confidence)
        if confidence < threshold:
            if label == y_test[i]:
                correct += 1
            total += 1
    #print(confidences)
    precision = correct / total if total != 0 else 1
    recall = correct / len(X_test)
    precisions.append(precision)
    recalls.append(recall)
    end_time = time.time()

print("lt:", (end_time - start_time) / len(X_test))
print("lp:", precisions[-1])
print("lr:", recalls[-1])
    # 4. 绘制P-R曲线
#plt.plot(recalls, precisions, label='lbphface')

'''
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()'''