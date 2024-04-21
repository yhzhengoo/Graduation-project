import cv2
import dlib
import matplotlib.pyplot as plt
# 读取图像
img = cv2.imread('myself.jpg')

# 初始化 Dlib 的人脸检测器和关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 将图像转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用人脸检测器检测人脸
faces = detector(gray)

for face in faces:
    # 检测关键点
    landmarks = predictor(gray, face)

    # 标注关键点
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 6, (0, 0, 139), -1)

# 将图像从 BGR 格式转换为 RGB 格式
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 创建 Matplotlib 图形
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.axis('off')  # 关闭坐标轴
plt.show()




