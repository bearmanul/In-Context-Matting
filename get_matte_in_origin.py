import cv2
import numpy as np
from PIL import Image

#image=cv2.imread('person/alpha/0b2890f68dce57c7d4a45b5e0ff5a019.jpg',flags=1)
# 图像文件路径
image_path_1 = 'person/alpha/20240615221715.png'
image_path_2 = 'person/fg/20240615221630.jpg'

# 使用OpenCV读取图像，IMREAD_UNCHANGED参数保证Alpha通道不会被忽略
#image_with_alpha = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
image_Alpha = cv2.imread(image_path_1, flags=1)
image_origin= cv2.imread(image_path_2, flags=1)

# 使用索引访问BGR通道
blue = image_Alpha[:, :, 0]
Alpha=np.round(blue / 255, decimals=2)

blue = image_origin[:, :, 0]
green = image_origin[:, :, 1]
red = image_origin[:, :, 2]

for i in range(image_origin.shape[0]):  # 外层循环遍历行
    for j in range(image_origin.shape[1]):  # 内层循环遍历当前行的列
        blue[i][j]=Alpha[i][j]*blue[i][j]
        green[i][j] = Alpha[i][j] * green[i][j]
        red[i][j] = Alpha[i][j] * red[i][j]

RGB_array = np.dstack((red, green, blue))
image = Image.fromarray(RGB_array.astype('uint8'), 'RGB')
image.save('person/result/003.jpg')

# #测试老师的alpha matte
# def check_alpha_channel(image_path):
#     with Image.open(image_path) as img:
#         return img.mode in ('RGBA', 'LA')  # 'RGBA'是带Alpha的RGB图像，'LA'是带Alpha的灰度图像
#
# image_path = 'person/alpha/backpack_02.png'
# if check_alpha_channel(image_path):
#     print("图片有Alpha通道")
# else:
#     print("图片没有Alpha通道")
#
# # # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
# # image = cv2.imread(image_path, flags=1)
# # print(image.shape)


