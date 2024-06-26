from PIL import Image
import numpy as np
from get_red_arr import get_red_arr

# 读入图像
def get_out_img(var_img: Image) -> Image:
    img = var_img.convert('L')
    width, height = img.size

    gray_arr  = np.array(img).astype(np.double) / 255 # 将灰度图转换为 NumPy 数组
    gray_arr  = (gray_arr - gray_arr.min()) / (gray_arr.max() - gray_arr.min() + 1e-4) # 1, 0
    green_arr = gray_arr * (220/255)
    img = img.resize((width // 2, height // 2))

    gray_arr = np.array(img).astype(np.double) / 255
    gray_arr = (gray_arr - gray_arr.min()) / (gray_arr.max() - gray_arr.min() + 1e-4) # 1, 0

    red_arr = get_red_arr(gray_arr) # 将三个通道合并为一个RGB图像
    red_pic = np.stack([255 * red_arr, 255 * red_arr, 255 * red_arr], axis=-1)

    new_img   = Image.fromarray(red_pic.astype(np.uint8)).resize((width, height)).convert('L') # 创建PIL Image对象
    red_arr   = np.array(new_img)
    final_arr = np.stack([red_arr.astype(np.uint8), (255 * green_arr).astype(np.uint8), np.zeros((height, width)).astype(np.uint8)], axis=-1)
    final_img = Image.fromarray(final_arr, mode='RGB')
    return final_img

if __name__ == "__main__":
    import os
    dirnow = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dirnow)
    var_img = Image.open("var_img.png")
    get_out_img(var_img).save('out_img.png')
