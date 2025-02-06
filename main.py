# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:50:48 2024

@author: shuaishuai
"""
from tool import *
from matplotlib import pyplot as plt
from skimage.transform import warp, rotate, rescale
from skimage import filters

#input two fits
file1 = 'aia.lev1_euv_12s.2024-01-31T062707Z.304.image_lev1.fits'
file2 = 'sdi_lev10_20240131_062713.991_v01.fits.gz'

im1, header1 = fitsread(file1)    #AIA 304
im2, header2 = fitsread(file2)    #SDI 1216

# 剔除hot pixels, 如果图像热点少，可以不用。
im2, ray = removeray(im2, T=0.5)

# 位移到图像中心
im1 = center_im(im1, header1, mode=1)
im2 = center_im(im2, header2, mode=2)

#掩膜
D = disk(4096, 4096, 1500)


#让两幅图像的大小一致，如果输入其他图像，在这里要调整。
R1 = 360 - header1['CROTA2']
im1 = rotate(im1, R1, order=0)  # 对304调整角度
R2 = 360 - header2['CROTA2']
im2 = rotate(im2, R2, order=0)  # 对sdi调整角度

im2 = rescale(im2, 5 / 6)  # sdi比例尺0.5角秒，304为0.60015角秒，这里做一个粗略缩放4608*5/6=3840
im2 = np.pad(im2, 128)  # 3840+128*2=4096，将缩放后的sdi扩展到跟304一样

sim1 = filters.gaussian(im1, 21)  # 对304图像高斯滤波，可以省略

im2_input = im2 / np.mean(im2[D]) * 10000
sim1_input = sim1 / np.mean(sim1[D]) * 10000

#计算304与sdi光流
d, model, flag, flow, err = align_opflow_2(im2_input, sim1_input, winsize=101, step=10, r_t=4, arrow=0, sample=20,
                                           mask=D)  # sample抽取样本行列数
print('scale rot(arcsec) dx(pixel) dy(pixel):')
tform = SimilarityTransform(scale=1 / d[0], rotation=-d[1], translation=[-d[2], -d[3]])
cim2 = warp(im2, tform, output_shape=(im2.shape[0], im2.shape[1]))  ####用tform则尺度、旋转、位移相反；用tform.inverse则不用变
im_three = np.dstack((im1, im2, cim2)).transpose(2, 0, 1)  # (3*4096*4096)

im = im_three[:, 2400:3000, 2600:3200]

region_show(im)

plt.show()