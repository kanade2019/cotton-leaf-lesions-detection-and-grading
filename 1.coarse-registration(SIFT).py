import cv2
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import os

image_path = './data/origin_image1'
outpath = './data/image(SIFT)'
filecount = 0

def getMat(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    good_match = []
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_match.append(m)
    pts_1 = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
    pts_2 = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)
    H, status = cv2.findHomography(pts_1, pts_2, cv2.RANSAC, 1)
    return H

def registerImage(dst, src, H):
    out = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return out

def threshold(img, thresh_num):
    if len(img.shape) == 2 or img.shape[2] == 1:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype("uint8")
    _, threshold_image = cv2.threshold(gray, thresh_num, 255, cv2.THRESH_BINARY)
    return threshold_image

def RemovalInterference(img):
    _, _, stats, _ = cv2.connectedComponentsWithStats(img)
    for istat in stats:
        if istat[4] < 120:
            cv2.rectangle(img, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), 0, thickness=-1)
    return img

for filename in os.listdir(image_path):
    if filename.find('GRE') != -1:
        print(filename)
        filepath = image_path + '/' + filename
        gre = nir = red = reg = cv2.imread(filepath, 0)
        file_num = filename[:filename.index('GRE')]
        nir = cv2.imread(image_path + '/' + file_num + "NIR_S1.TIF", 0)
        red = cv2.imread(image_path + '/' + file_num + "RED_S1.TIF", 0)
        reg = cv2.imread(image_path + '/' + file_num + "REG_S1.TIF", 0)
        rgb = cv2.imread(image_path + '/' + file_num + "RGB.JPG", 0)
#         H = [getMat(gre, nir), getMat(gre, red), getMat(gre, reg)]
#         print(filename)
#         nir = registerImage(gre, nir, H[0])
#         red = registerImage(gre, red, H[1])
#         reg = registerImage(gre, reg, H[2])
        BAND = [rgb, gre, red, nir, reg]
        sp = BAND[1].shape#以rgb图像为基准
        BAND[0] = cv2.resize(BAND[0], (1280, 960))
        #对准彩色和绿色
        #二值化图像
        binary_BAND = []
        binary_BAND.append(threshold(BAND[0], 150))
        binary_BAND.append(threshold(BAND[1], 100))
        #去除干扰点
        for img in binary_BAND:
            RemovalInterference(img)
        #透视变换矩阵
        H = [0]
        #获取绿色对准彩色的变换矩阵
        H.append(getMat(binary_BAND[0], binary_BAND[1]))
        #对准图像
        alignment_BAND = [BAND[0]]
        #获取绿色的对准图像
        alignment_BAND.append(registerImage(BAND[0], BAND[1], H[1]))

        #获取透视变换矩阵
        for img in BAND[2:]:
            H.append(getMat(alignment_BAND[1], img))
        #获取剩余对准图像
        nir = registerImage(gre, nir, H[3])
        red = registerImage(gre, red, H[2])
        reg = registerImage(gre, reg, H[4])
        cv2.imwrite(outpath + '/' + file_num + 'GRE.TIF', alignment_BAND[1])
        cv2.imwrite(outpath + '/' + file_num + 'NIR.TIF', nir)
        cv2.imwrite(outpath + '/' + file_num + 'RED.TIF', red)
        cv2.imwrite(outpath + '/' + file_num + 'REG.TIF', reg)