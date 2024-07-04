import cv2
import os
import numpy as np
from PIL import Image
from blurfusion_foreground_estimation import FB_blur_fusion_foreground_estimator_2

#image=cv2.imread('person/alpha/0b2890f68dce57c7d4a45b5e0ff5a019.jpg',flags=1)
# 图像文件路径
def get_matte_in_origin(directory_alpha,directory_origin):
    for i in range(len(os.listdir(directory_alpha))):
        alpha_name = os.listdir(directory_alpha)[i]
        origin_name = os.listdir(directory_origin)[i]
        alpha_path = os.path.join(directory_alpha, alpha_name)
        origin_path = os.path.join(directory_origin, origin_name)

        # 使用OpenCV读取图像，IMREAD_UNCHANGED参数保证Alpha通道不会被忽略
        # image_Alpha = cv2.imread(alpha_path, flags=1)
        # image_origin= cv2.imread(origin_path, flags=1)

        alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)/255.0
        image = cv2.imread(origin_path)/255.0

        # # 使用索引访问BGR通道
        # blue = image_Alpha[:, :, 0]
        # Alpha=np.round(blue / 255, decimals=2)
        #
        # blue = image_origin[:, :, 0]
        # green = image_origin[:, :, 1]
        # red = image_origin[:, :, 2]
        #
        # for i in range(image_origin.shape[0]):  # 外层循环遍历行
        #     for j in range(image_origin.shape[1]):  # 内层循环遍历当前行的列
        #         blue[i][j]=Alpha[i][j]*blue[i][j]
        #         green[i][j] = Alpha[i][j] * green[i][j]
        #         red[i][j] = Alpha[i][j] * red[i][j]
        #
        # RGB_array = np.dstack((red, green, blue))
        # image_save = Image.fromarray(RGB_array.astype('uint8'), 'RGB')
        # image_save.save(os.path.join('frame/cat_matte', origin_name))

        output = FB_blur_fusion_foreground_estimator_2(image, alpha)
        # composite onto a white background
        composite = output * alpha[:, :, np.newaxis] + (1 - alpha[:, :, np.newaxis])

        # Save using opencv
        cv2.imwrite('time-step/result/{}'.format(alpha_name), composite * 255)


path1 = 'time-step/image'#原图所在的地方
path2 = 'time-step/alpha'#模型跑出来的matte
get_matte_in_origin(path2,path1)

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


