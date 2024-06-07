from math import sqrt
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
dirnow = os.path.dirname(os.path.abspath(__file__))
os.chdir(dirnow)

# 读取图像
img = Image.open('image.png').convert("L")

# 获取图像的大小
width, height = img.size
img1 = np.array(img).astype(np.float32)
img2 = img1 ** 2

# 创建一个与原图大小相同的数组用于存储方差
variance_map = np.zeros((height, width), dtype=np.float32)

# 遍历图像的每个像素,计算其邻域方差
edge_width = 10
for i in tqdm(range(edge_width, height-edge_width)):
    for j in range(edge_width, width-edge_width):
        # 提取当前像素及其8个邻域像素
        variance = sqrt(np.average(img2[i-1:i+2, j-1:j+2]) - np.average(img1[i-1:i+2, j-1:j+2]) ** 2)
        
        # 将方差存储到结果数组中
        variance_map[i, j] = variance

# 将结果保存为新的图像
variance_map = (variance_map - variance_map.min()) / (variance_map.max() - variance_map.min())
alpha = 0.1
variance_map[variance_map < alpha] = 0
variance_map = 255 * variance_map
variance_map = variance_map.astype(np.uint8)
variance_img = Image.fromarray(variance_map)
variance_img.save('variance_map.png')