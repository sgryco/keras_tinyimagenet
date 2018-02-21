#!/usr/bin/python
# coding=utf-8

# Base Python File (make_png.py)
# Created: ven. 09 févr. 2018 23:16:49 GMT
# Version: 1.0
#
# This Python script was developped by Cory.
#
# (c) Cory <sgryco@gmail.com>
import os
from multiprocessing import Pool, cpu_count
from scipy.misc import imread, imsave, imresize


def convert(imgpath, size=64):
    img = imread(os.path.join(imagenet, imgpath))
    # if img not square, make it
    h, w = img.shape[0], img.shape[1]
    if h > w:
        h0 = (h - w) / 2
        img_resized = img[h0:h0+w, :]
    else:
        w0 = (w - h) / 2
        img_resized = img[:, w0:w0+h]
    img_resized = imresize(img_resized, (size, size))
    out_file = os.path.join(out_dir, imgpath.replace("JPEG", "png"))
    imsave(out_file, img_resized)
    return out_file


# list all folderso
imagenet = "imagenet"
dirs = os.listdir(imagenet)
out_dir = "tiny"
os.mkdir(out_dir)
limit = 1
pool = Pool(cpu_count())

with open("200_wnids.txt", "r") as sel:
    classes = [v.strip() for v in sel.readlines()]
for folder in dirs:
    if folder not in classes:
        continue
    localpath = os.path.join(imagenet, folder)
    imgs = [os.path.join(folder, img)
            for img in os.listdir(localpath)[:100000]]
    os.mkdir(os.path.join(out_dir, folder))
    ret = pool.map(convert, imgs)
    print(ret)




# for each folder,
# list all files
# batch convert to size, save to outdir
#loop

