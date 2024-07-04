import cv2
import os
import shutil
import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image


def video2frame(videos_path, frames_save_path, time_interval):
    '''
    :param videos_path: 视频的存放路径
    :param frames_save_path: 视频切分成帧之后图片的保存路径
    :param time_interval: 保存间隔
    :return:
    '''
    vidcap = cv2.VideoCapture(videos_path)
    ret, image = vidcap.read()
    count = 0
    while True:
        ret, image = vidcap.read()
        if not ret:
            break
        count += 1
        if count % time_interval == 0:
            #cv2.imencode('.jpg', image)[1].tofile(frames_save_path + "/frame%d.jpg" % count)
            output_img_path = os.path.join(frames_save_path, f'frame_{count:04d}.jpg')
            cv2.imwrite(output_img_path, image)
        # if count == 20:
        #   break
    print(f'Saved frame {count}')


def frame2video(directory_frame, video_path, fps):
    files = sorted(os.listdir(directory_frame))
    index = 0
    for filename in files:
        # 确保是图片文件（这里以常见的图片扩展名为例，你可以根据需要调整）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 构建新文件名
            new_filename = f'frame_{index}.jpg'
            old_file = os.path.join(directory_frame, filename)
            new_file = os.path.join(directory_frame, new_filename)

            # 重命名文件
            # 注意：这里假设所有图片都将被转换为jpg格式，如果原文件不是jpg，需要先转换格式
            shutil.move(old_file, new_file)  # 使用move直接改名并可处理格式转换逻辑

            index += 1

    image_files = ["frame/yw_origin_matte/frame_{}.jpg".format(i) for i in
                   range(len(os.listdir(directory_frame)))]  # 假设你有从frame_1.jpg到frame_100.jpg的图片

    # 使用ImageSequenceClip函数读取图片序列并设置帧率
    clip = ImageSequenceClip(image_files, fps=fps)

    clip.write_videofile('video/yw_matte.mp4', codec="libx264")

    print("视频已生成")



if __name__ == '__main__':
    directory_frame = 'frame/yw_origin_matte'  # 帧存放路径
    video_path = 'video'  # 合成视频存放的路径
    fps = 25  # 帧率，每秒钟帧数越多，所显示的动作就会越流畅
    frame2video(directory_frame, video_path, fps)

# if __name__ == '__main__':
#     videos_path = 'video/7356431-uhd_3840_2160_25fps.mp4'
#     frames_save_path = 'frame/yw'
#     time_interval = 1  # 隔一帧保存一次
#     video2frame(videos_path, frames_save_path, time_interval)