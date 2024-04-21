import dlib
import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号'-'显示为方块的问题
plt.rcParams.update({'font.size': 14})
# 加载人脸检测器和预训练的人脸检测模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 加载图像
image = cv2.imread("myself.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. 检测图像中的人脸
faces = detector(gray)

# 2. 提取人脸的LBP特征
def compute_lbp(image):
    lbp = feature.local_binary_pattern(image, 8, 1, method="uniform")
    return lbp.astype(np.uint8)

# 3. 计算整张脸的LBP直方图
def compute_histogram(lbp_image):
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 10), range=(0, 256))
    return hist.astype(np.float32)

# 4. 将LBP特征图进行分块，共分成64个区域
def divide_into_blocks(lbp_image, num_blocks=8):
    h, w = lbp_image.shape
    block_h, block_w = h // num_blocks, w // num_blocks
    blocks = []
    for i in range(num_blocks):
        for j in range(num_blocks):
            block = lbp_image[i * block_h: (i + 1) * block_h, j * block_w: (j + 1) * block_w]
            blocks.append(block)
    return blocks

# 5. 显示预处理后的图像、LBP特征图和整张脸LBP直方图
def show_results(image, lbp_face, lbp_hist, blocks, block1_hist, block6_hist):
    plt.figure(figsize=(15, 5))

    # 显示预处理后的图像
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('预处理后的图像')
    plt.axis('off')


    # 显示LBP特征图
    plt.subplot(1, 3, 2)
    # 在LBP特征图上显示网格
    # 定义网格行数和列数
    # 创建与原始图像相同大小的空白图像
    # 定义网格行数和列数
    num_blocks = 8

    # 绘制水平线
    for i in range(1, num_blocks):
        cv2.line(lbp_face, (0, i * lbp_face.shape[0] // num_blocks), (lbp_face.shape[1], i * lbp_face.shape[0] // num_blocks),
                 (0, 0, 255), 3)

    # 绘制垂直线
    for j in range(1, num_blocks):
        cv2.line(lbp_face, (j * lbp_face.shape[1] // num_blocks, 0), (j * lbp_face.shape[1] // num_blocks, lbp_face.shape[0]),
                 (0, 0, 255), 3)

    plt.imshow(lbp_face, cmap='gray')
    plt.title('LBP特征图')
    plt.axis('off')

    # 显示整张脸的LBP直方图
    plt.subplot(1, 3, 3)
    plt.bar(np.arange(len(lbp_hist)), lbp_hist)
    plt.title('LBP直方图')
    #plt.axis('off')

    plt.show()

    plt.bar(np.arange(len(block1_hist)), block1_hist)
    plt.title('块19直方图')
    plt.show()


    plt.bar(np.arange(len(block6_hist)), block6_hist)
    plt.title('块32直方图')

    plt.show()

# 处理每张检测到的人脸
for i, face in enumerate(faces):
    # 获取人脸区域
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    face_region = gray[y:y+h, x:x+w]

    # 2. 提取人脸的LBP特征
    lbp_face = compute_lbp(face_region)

    # 3. 计算整张脸的LBP直方图
    lbp_hist = compute_histogram(lbp_face)

    # 4. 将LBP特征图进行分块，共分成64个区域
    blocks = divide_into_blocks(lbp_face)

    # 5. 计算块1和块6的LBP直方图
    block1_hist = compute_histogram(blocks[19])
    block6_hist = compute_histogram(blocks[32])

    # 显示结果
    show_results(face_region, lbp_face, lbp_hist, blocks, block1_hist, block6_hist)