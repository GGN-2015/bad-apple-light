from PIL import Image
import numpy as np
from get_var_arr import get_var_arr

def get_var_image(img: Image, erase_width=10) -> Image:
    img = img.convert("L") # 获取黑白图像
    img = np.array(img).astype(np.double)
    var = get_var_arr(img)
    var_map = (var - var.min()) / (var.max() - var.min() + (1e-4))
    var_map = 255 * var_map
    var_map = var_map.astype(np.uint8)
    var_map[:+erase_width, :] = 0 # erase border
    var_map[-erase_width:, :] = 0
    var_map[:, :+erase_width] = 0
    var_map[:, -erase_width:] = 0
    var_img = Image.fromarray(var_map)
    return var_img # 将结果存储到新的图像

if __name__ == "__main__":
    import os
    dirnow = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dirnow)
    img = Image.open("image.png")
    get_var_image(img).save("var_img.png")