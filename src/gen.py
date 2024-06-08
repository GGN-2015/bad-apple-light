from PIL import Image
from tqdm import tqdm
from var_img import get_var_image
from out_img import get_out_img

import os
dirnow = os.path.dirname(os.path.abspath(__file__))
os.chdir(dirnow)

files = list(os.listdir("frames"))
for file in tqdm(files):
    infile  = os.path.join(dirnow, "frames", file)
    outfile = os.path.join(dirnow, "new_frames", file)

    img = Image.open(infile) # readin
    var = get_var_image(img) # get var
    out = get_out_img(var)   # get final pic
    out.save(outfile)        # save