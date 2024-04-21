import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号'-'显示为方块的问题
plt.rcParams.update({'font.size': 18})

# 加载 real 和 fake 图片
real_image = cv2.imread('real.png', cv2.IMREAD_GRAYSCALE)
fake_image = cv2.imread('fake.png', cv2.IMREAD_GRAYSCALE)

# 修改 fake 图片大小为 250x250
real_resized = cv2.resize(real_image, (250, 250))
fake_resized = cv2.resize(fake_image, (250, 250))

# 计算 real 和 fake 图片的总梯度
real_gradient = cv2.Sobel(real_resized, cv2.CV_64F, 1, 1, ksize=5)
fake_gradient = cv2.Sobel(fake_resized, cv2.CV_64F, 1, 1, ksize=5)

# 提取 real 和 fake 图片的 HOG 特征
real_hog_feature, real_hog_image = hog(real_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16),visualize=True)
fake_hog_feature, fake_hog_image = hog(fake_resized,orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16),visualize=True)

# 创建子图布局
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# 绘制原图
axs[0, 0].imshow(real_resized,cmap='gray')
axs[0, 0].set_title('正样本')
axs[0, 0].axis('off')
axs[1, 0].imshow(fake_resized,cmap='gray')
axs[1, 0].set_title('负样本')
axs[1, 0].axis('off')
# 绘制总梯度图
axs[0, 1].imshow(real_gradient, cmap='gray')
axs[0, 1].set_title('正样本梯度')
axs[0, 1].axis('off')
axs[1, 1].imshow(fake_gradient, cmap='gray')
axs[1, 1].set_title('负样本梯度')
axs[1, 1].axis('off')
# 绘制 HOG 特征图
axs[0, 2].imshow(real_hog_image, cmap='gray')
axs[0, 2].set_title('正样本HOG')
axs[0, 2].axis('off')
axs[1, 2].imshow(fake_hog_image, cmap='gray')
axs[1, 2].set_title('负样本HOG')
axs[1, 2].axis('off')
# 调整子图间距和标签显示
plt.tight_layout()
plt.show()
