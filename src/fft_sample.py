from math import sqrt
import numpy as np
from scipy.fft import fft2, ifft2
from tqdm import tqdm
from PIL import Image
import os
dirnow = os.path.dirname(os.path.abspath(__file__))
os.chdir(dirnow)

img = Image.open('image.png').convert('L')
width, height = img.size
img_array = np.array(img)
fft_img = fft2(img_array)

def getdis(x1, y1, x2, y2) -> float:
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

for radius in tqdm(range(200, 500, 20)):
    rows, cols = fft_img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.uint8)

    h0 = 5
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if getdis(i, j, crow, ccol) <= radius:
                mask[i, j] = 1
            else:
                mask[i, j] = h0 / (h0 + getdis(i, j, crow, ccol) - radius)

    fft_filtered = fft_img * mask
    result = np.abs(ifft2(fft_filtered))
    #result = np.log(np.abs(fft_filtered) + 1)
    result = (result - result.min()) / (result.max() - result.min())
    result = (result * 255).astype(np.uint8)
    final_img = Image.fromarray(result)
    final_img.save('tmp/fft_result_%04d.png' % radius)