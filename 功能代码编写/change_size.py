import cv2
import numpy as np
import os
from PIL import Image

#image=cv2.imread('person/alpha/0b2890f68dce57c7d4a45b5e0ff5a019.jpg',flags=1)
# 图像文件路径
def change_image_size(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # image_path = 'frame/cat/frame_0002.jpg'

        # 使用OpenCV读取图像，IMREAD_UNCHANGED参数保证Alpha通道不会被忽略
        #image_with_alpha = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.imread(filepath, flags=1)

        # print(image.shape)

        blue = image[:, :, 0]
        green = image[:, :, 1]
        red = image[:, :, 2]

        # blue_new = np.zeros((image.shape[1], image.shape[1]))
        # green_new = np.zeros((image.shape[1], image.shape[1]))
        # red_new = np.zeros((image.shape[1], image.shape[1]))

        # for i in range(image.shape[1]):  # 外层循环遍历行
        #     if i==image.shape[1]-image.shape[0]:
        #         t = 0
        #     if i > image.shape[1]-image.shape[0]:
        #         t = t + 1
        #     for j in range(image.shape[1]):  # 内层循环遍历当前行的列
        #         if i < image.shape[1]-image.shape[0] :
        #             blue_new[i][j] = blue[0][j]
        #             green_new[i][j] = green[0][j]
        #             red_new[i][j] = red[0][j]
        #         else:
        #             blue_new[i][j] = blue[t][j]
        #             green_new[i][j] = green[t][j]
        #             red_new[i][j] = red[t][j]
        blue_new = np.full((image.shape[1]-image.shape[0], image.shape[1]), 255, dtype=np.uint8)
        blue_new_list = blue_new.tolist()
        blue_list = blue.tolist()
        green_list = green.tolist()
        red_list = red.tolist()
        blue_combined_list = blue_new_list + blue_list
        green_combined_list = blue_new_list + green_list
        red_combined_list = blue_new_list + red_list
        blue_combined_matrix = np.array(blue_combined_list)
        green_combined_matrix = np.array(green_combined_list)
        red_combined_matrix = np.array(red_combined_list)




        RGB_array = np.dstack((red_combined_matrix, green_combined_matrix, blue_combined_matrix))
        image_save = Image.fromarray(RGB_array.astype('uint8'), 'RGB')
        image_save.save(os.path.join('frame/yw_change_size', filename))

path = 'frame/yw'
change_image_size(path)