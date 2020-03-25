#!/usr/bin/python
# -*-coding:utf-8-*-

# Read .pfm file from `Disparity` data of Scene Flow Dataset
# [Scene Flow Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

import numpy as np
import re
import sys
from PIL import Image


def load_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF' or header == b'PF':
        color = True
    elif header == 'Pf' or header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match((r'^(\d+)\s(\d+)\s$').encode('utf-8'), file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:  # big-endian
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flip(data,0)
    #image = Image.fromarray(data)
    #image = image.rotate(180)
    file.close()
    return np.array(data)


def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    str1 = 'PF\n' if color else 'Pf\n'
    file.write(str1.encode())
    str2 = '%d %d\n' % (image.shape[1], image.shape[0])
    file.write(str2.encode())

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    str3 = '%f\n' % scale
    file.write(str3.encode())

    image.tofile(file)


if __name__ == '__main__':
    f = '/data/jack/flyingthings3d/disparity/TRAIN/C/0000/right/0014.pfm'
    img = Image.fromarray(load_pfm(f))
    print(np.array(img))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save('test0001.png')