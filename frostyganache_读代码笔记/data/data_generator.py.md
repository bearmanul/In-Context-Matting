# ChatGPT的概括
这段代码定义了一系列用于数据预处理和增强的类和函数，用于处理图像数据。下面是对每个定义的函数和类的解释：

1. `maybe_random_interp(cv2_interp)`: 这是一个函数，用于根据 `RANDOM_INTERP` 变量的值随机选择插值方式。如果 `RANDOM_INTERP` 为真，则从指定的插值方式列表 `interp_list` 中随机选择一种插值方式，否则返回传入的插值方式 `cv2_interp`。

2. `ToTensor`: 这是一个类，用于将样本中的图像数据从 NumPy 数组转换为 PyTorch 张量，并进行归一化处理。在 `__call__` 方法中，首先将图像从 BGR 转换为 RGB 格式，然后将其转换为 PyTorch 张量，并根据指定的均值和标准差进行归一化处理。

3. `RandomAffine`: 这是一个类，用于执行随机仿射变换。在 `__call__` 方法中，根据指定的参数生成随机仿射变换的参数，然后将图像和 alpha 通道应用该变换。

4. `RandomJitter`: 这是一个类，用于随机改变图像的色调。在 `__call__` 方法中，首先将图像转换为 HSV 颜色空间，然后对色调、饱和度和亮度进行随机扰动，最后将图像转换回 BGR 颜色空间。

5. `RandomHorizontalFlip`: 这是一个类，用于执行随机水平翻转图像和 alpha 通道。在 `__call__` 方法中，根据指定的概率随机对图像和 alpha 通道进行水平翻转。

6. `RandomCrop`: 这是一个类，用于执行随机裁剪图像。在 `__call__` 方法中，根据指定的参数随机裁剪图像，并将结果返回。

7. `CropResize`: 这是一个类，用于将图像裁剪为正方形，并调整大小到目标尺寸。在 `__call__` 方法中，根据指定的目标尺寸裁剪图像，并将结果返回。

8. `OriginScale`: 这是一个类，用于将图像缩放到 32 的倍数。在 `__call__` 方法中，将图像填充到目标尺寸的最小 32 的倍数大小，并返回结果。

9. `GenMask`: 这是一个类，用于生成 mask。在 `__call__` 方法中，生成 trimap 和 mask，并返回结果。

10. **Composite类**：这个类实现了一个 `__call__` 方法，该方法接受一个样本作为输入，并对其进行处理，主要功能是合成图像。它首先从样本中获取前景、背景和alpha通道，然后根据alpha通道将前景和背景图像进行合成，生成合成图像。合成过程中，对alpha通道进行了修剪和处理，确保其取值范围在0到1之间，同时对前景和背景图像的像素值进行了修剪，确保其范围在0到255之间。
    
11. **CutMask类**：这个类实现了一个 `__call__` 方法，该方法也接受一个样本作为输入，并对其进行处理，主要功能是对遮罩进行切割。它首先根据给定的概率决定是否执行切割操作，然后随机选择一个区域并将其复制到另一个随机选择的区域，实现了对遮罩的随机扰动。
    
12. **DataGenerator类**：这个类继承自 `Dataset`，用于生成训练和验证数据集。在初始化时，根据所处阶段（训练或验证）加载相应的数据，包括前景、背景、alpha通道等。在 `__getitem__` 方法中，根据所处阶段加载对应的图像数据，并进行一系列数据增强操作，包括随机仿射变换、生成遮罩、切割遮罩、随机裁剪、随机抖动、合成图像等。最后将处理后的样本转换为张量并返回。
    
13. **MultiDataGeneratorDoubleSet类**：这个类也继承自 `Dataset`，用于生成包含训练和验证数据集的双数据集。与 `DataGenerator` 类相似，它也实现了 `__getitem__` 方法和 `__len__` 方法，其中 `__getitem__` 方法中加载了包含图像、alpha通道和trimap的样本，并进行了一系列数据增强操作。
    
14. **ContextDataset类**：这个类用于生成包含上下文信息的数据集。它根据所处阶段加载训练或验证数据集，并根据给定的类别和子类别信息选择相应的上下文图像。然后将原始图像和上下文图像进行合并，并返回合并后的样本。
    
15. **InContextDataset类**：这个类也用于生成包含上下文信息的数据集，与 `ContextDataset` 类类似。它加载训练或验证数据集，并根据类别和子类别信息选择相应的上下文图像。然后将原始图像和上下文图像进行合并，并返回合并后的样本。与 `ContextDataset` 类不同的是，它对返回的样本进行了一些额外的处理，例如将图像属性名称修改为符合上下文信息的名称。

# 随机仿射变换

[仿射变换及其变换矩阵的理解 - shine-lee - 博客园 (cnblogs.com)](https://www.cnblogs.com/shine-lee/p/10950963.html)

# 理解图像中“上下文信息”的概念
[图像处理中底层、高层特征、上下文信息理解_图像的上下文信息-CSDN博客](https://blog.csdn.net/PLANTTHESON/article/details/133962508)

# 代码注释

```python
import cv2
import os
import math
import numbers
import random
import logging
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision import transforms

from icm.util import instantiate_from_config
from icm.data.image_file import get_dir_ext


TRIMAP_CHANNEL = 1  # 三通道trimap

RANDOM_INTERP = True  # 是否随机插值

CUTMASK_PROB = 0  # 切割掩码的概率

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR,
               cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]  # 插值方式列表


def maybe_random_interp(cv2_interp):
    # 随机选择插值方式
    if RANDOM_INTERP:
        return np.random.choice(interp_list)
    else:
        return cv2_interp


class ToTensor(object):
    """
    将样本中的 ndarrays 转换为带有归一化的 Tensors。
    """

    def __init__(self, phase="test", norm_type='imagenet'):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # 均值
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)  # 标准差
        self.phase = phase  # 阶段
        self.norm_type = norm_type  # 归一化类型

    def __call__(self, sample):
        # 将 GBR 图像转换为 RGB
        image, alpha, trimap, mask = sample['image'][:, :, ::-1], sample['alpha'], sample['trimap'], sample['mask']

        alpha[alpha < 0] = 0  # 将 alpha 值小于 0 的设置为 0
        alpha[alpha > 1] = 1  # 将 alpha 值大于 1 的设置为 1

        # 交换颜色轴，因为
        # numpy 图像: H x W x C
        # torch 图像: C x H x W
        #其中C表示通道数（比如RGB图像的通道数为3）、H表示图像的高度、W表示图像的宽度。
        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        trimap[trimap < 85] = 0  # 将 trimap 值小于 85 的设置为 0
        trimap[trimap >= 170] = 1  # 将 trimap 值大于等于 170 的设置为 1
        trimap[trimap >= 85] = 0.5  # 将 trimap 值大于等于 85 小于 170 的设置为 0.5

        mask = np.expand_dims(mask.astype(np.float32), axis=0)

        if self.phase == "train":
            # 将 GBR 图像转换为 RGB
            fg = sample['fg'][:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['fg'] = torch.from_numpy(fg).sub_(self.mean).div_(self.std)  # 根据均值和标准差归一化
            bg = sample['bg'][:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['bg'] = torch.from_numpy(bg).sub_(self.mean).div_(self.std)  # 根据均值和标准差归一化

        sample['image'], sample['alpha'], sample['trimap'] = \
            torch.from_numpy(image), torch.from_numpy(
                alpha), torch.from_numpy(trimap)

        if self.norm_type == 'imagenet':
            # 归一化图像
            sample['image'] /= 255.
            sample['image'] = sample['image'].sub_(self.mean).div_(self.std)  # 根据均值和标准差归一化
        elif self.norm_type == 'sd':
            sample['image'] = sample['image'].to(dtype=torch.float32) / 127.5 - 1.0  # 标准化
        else:
            raise NotImplementedError(
                "norm_type {} is not implemented".format(self.norm_type))

        if TRIMAP_CHANNEL == 3:
            sample['trimap'] = F.one_hot(
                sample['trimap'], num_classes=3).permute(2, 0, 1).float()  # 转换为 one-hot 编码
        elif TRIMAP_CHANNEL == 1:
            sample['trimap'] = sample['trimap'][None, ...].float()  # 转换为 FloatTensor
        else:
            raise NotImplementedError("TRIMAP_CHANNEL can only be 3 or 1")

        sample['mask'] = torch.from_numpy(mask).float()  # 转换为 FloatTensor

        return sample


class RandomAffine(object):
    """
    随机仿射变换
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError(
                        "translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError(
                        "If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """获取仿射变换参数

        Returns:
            sequence: 要传递给仿射变换的参数
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params(
                (0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(
                self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        sample['fg'], sample['alpha'] = fg, alpha

        return sample

    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # 用于计算仿射变换的逆矩阵的辅助方法

        # 如 PIL.Image.rotate 中所述
        # 我们需要计算仿射变换矩阵的逆矩阵: M = T * C * RSS * C^-1
        # 其中 T 是平移矩阵: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C 是保持中心的平移矩阵: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS 是旋转加缩放和剪切矩阵
        # 它与 torchvision 中的原始函数不同
        # 顺序改为翻转 -> 缩放 -> 旋转 -> 剪切
        # x 和 y 有不同的比例因子
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # 因此，逆矩阵是 M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # 带缩放和剪切的反转旋转矩阵
        d = math.cos(angle + shear) * math.cos(angle) + \
            math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # 应用平移和中心平移的逆转和中心平移: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + \
            matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + \
            matrix[4] * (-center[1] - translate[1])

        # 应用中心平移: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix


class RandomJitter(object):
    """
    随机改变图像的色调
    """

    def __call__(self, sample):
        sample_ori = sample.copy()
        fg, alpha = sample['fg'], sample['alpha']
        # 如果 alpha 全为 0 则跳过
        if np.all(alpha == 0):
            return sample_ori
        # 转换为 HSV 空间，将图像转换为 float32 图像以保持空间转换的精度。
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # 色调噪声
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(
            fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # 饱和度噪声
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        if np.isnan(sat_bar):
            return sample_ori
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat > 1] = 2 - sat[sat > 1]
        fg[:, :, 1] = sat
        # 价值噪声
        val_bar = fg[:, :, 2][alpha > 0].mean()
        if np.isnan(val_bar):
            return sample_ori
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val > 1] = 2 - val[val > 1]
        fg[:, :, 2] = val
        # 转换回 BGR 空间
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample['fg'] = fg*255

        return sample


class RandomHorizontalFlip(object):
    """
    随机水平翻转图像和标签
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        if np.random.uniform(0, 1) < self.prob:
            fg = cv2.flip(fg, 1)
            alpha = cv2.flip(alpha, 1)
        sample['fg'], sample['alpha'] = fg, alpha

        return sample

class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        # 初始化随机裁剪类
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2  # 计算裁剪边界
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        # 在样本中随机裁剪图像
        fg, alpha, trimap, mask, name = sample['fg'],  sample[
            'alpha'], sample['trimap'], sample['mask'], sample['image_name']
        bg = sample['bg']
        h, w = trimap.shape
        bg = cv2.resize(
            bg, (w, h), interpolation=maybe_random_interp(cv2.INTER_CUBIC))  # 调整背景大小
        if w < self.output_size[0]+1 or h < self.output_size[1]+1:
            ratio = 1.1*self.output_size[0] / \
                h if h < w else 1.1*self.output_size[1]/w  # 计算调整比例
            while h < self.output_size[0]+1 or w < self.output_size[1]+1:
                # 循环调整大小直到满足要求
                fg = cv2.resize(fg, (int(w*ratio), int(h*ratio)),
                                interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha = cv2.resize(alpha, (int(w*ratio), int(h*ratio)),
                                   interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                trimap = cv2.resize(
                    trimap, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                bg = cv2.resize(bg, (int(w*ratio), int(h*ratio)),
                                interpolation=maybe_random_interp(cv2.INTER_CUBIC))
                mask = cv2.resize(
                    mask, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                h, w = trimap.shape
        small_trimap = cv2.resize(
            trimap, (w//4, h//4), interpolation=cv2.INTER_NEAREST)  # 调整 trimap 大小
        unknown_list = list(zip(*np.where(small_trimap[self.margin//4:(h-self.margin)//4,
                                                       self.margin//4:(w-self.margin)//4] == 128)))
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            # 如果未知区域过少，则随机选择裁剪位置
            left_top = (np.random.randint(
                0, h-self.output_size[0]+1), np.random.randint(0, w-self.output_size[1]+1))
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0]*4, unknown_list[idx][1]*4)  # 从未知区域随机选择裁剪位置

        fg_crop = fg[left_top[0]:left_top[0]+self.output_size[0],
                     left_top[1]:left_top[1]+self.output_size[1], :]  # 裁剪前景图像
        alpha_crop = alpha[left_top[0]:left_top[0]+self.output_size[0],
                           left_top[1]:left_top[1]+self.output_size[1]]  # 裁剪 alpha 通道
        bg_crop = bg[left_top[0]:left_top[0]+self.output_size[0],
                     left_top[1]:left_top[1]+self.output_size[1], :]  # 裁剪背景图像
        trimap_crop = trimap[left_top[0]:left_top[0]+self.output_size[0],
                             left_top[1]:left_top[1]+self.output_size[1]]  # 裁剪 trimap
        mask_crop = mask[left_top[0]:left_top[0]+self.output_size[0],
                         left_top[1]:left_top[1]+self.output_size[1]]  # 裁剪 mask

        if len(np.where(trimap == 128)[0]) == 0:
            # 如果不存在足够的未知区域，记录错误并调整大小到目标尺寸
            self.logger.error("{} does not have enough unknown area for crop. Resized to target size."
                              "left_top: {}".format(name, left_top))
            fg_crop = cv2.resize(
                fg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(
                alpha, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(
                trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            bg_crop = cv2.resize(
                bg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_CUBIC))
            mask_crop = cv2.resize(
                mask, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)

        sample.update({'fg': fg_crop, 'alpha': alpha_crop,
                      'trimap': trimap_crop, 'mask': mask_crop, 'bg': bg_crop})
        return sample


class CropResize(object):
    # 将图像裁剪为正方形，并调整大小到目标尺寸
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # 裁剪图像为正方形，并调整大小到目标尺寸
        img, alpha, trimap, mask = sample['image'], sample['alpha'], sample['trimap'], sample['mask']
        h, w = img.shape[:2]
        if h == w:
            # 如果图像已经是正方形，则直接调整大小
            img_crop = cv2.resize(
                img, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(
                alpha, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(
                trimap, self.output_size, interpolation=cv2.INTER_NEAREST)
            mask_crop = cv2.resize(
                mask, self.output_size, interpolation=cv2.INTER_NEAREST)
        elif h > w:
            # 如果图像高度大于宽度，则在上下裁剪
            margin = (h-w)//2
            img = img[margin:margin+w, :]
            alpha = alpha[margin:margin+w, :]
            trimap = trimap[margin:margin+w, :]
            mask = mask[margin:margin+w, :]
            img_crop = cv2.resize(
                img, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(
                alpha, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(
                trimap, self.output_size, interpolation=cv2.INTER_NEAREST)
            mask_crop = cv2.resize(
                mask, self.output_size, interpolation=cv2.INTER_NEAREST)
        else:
            # 如果图像宽度大于高度，则在左右裁剪
            margin = (w-h)//2
            img = img[:, margin:margin+h]
            alpha = alpha[:, margin:margin+h]
            trimap = trimap[:, margin:margin+h]
            mask = mask[:, margin:margin+h]
            img_crop = cv2.resize(
                img, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(
                alpha, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(
                trimap, self.output_size, interpolation=cv2.INTER_NEAREST)
            mask_crop = cv2.resize(
                mask, self.output_size, interpolation=cv2.INTER_NEAREST)
        sample.update({'image': img_crop, 'alpha': alpha_crop,
                      'trimap': trimap_crop, 'mask': mask_crop})
        return sample


class OriginScale(object):
    # 将图像缩放到32的倍数
    def __call__(self, sample):
        h, w = sample["alpha_shape"]

        if h % 32 == 0 and w % 32 == 0:
            return sample

        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w

        padded_image = np.pad(
            sample['image'], ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        padded_trimap = np.pad(
            sample['trimap'], ((0, pad_h), (0, pad_w)), mode="reflect")
        padded_mask = np.pad(
            sample['mask'], ((0, pad_h), (0, pad_w)), mode="reflect")

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap
        sample['mask'] = padded_mask

        return sample


class GenMask(object):
    # 生成 mask
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]

    def __call__(self, sample):
        alpha_ori = sample['alpha']
        h, w = alpha_ori.shape

        max_kernel_size = 30
        alpha = cv2.resize(alpha_ori, (640, 640),
                           interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        # generate trimap
        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv2.erode(
            fg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        bg_mask = cv2.erode(
            bg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])

        fg_width = np.random.randint(1, 30)
        bg_width = np.random.randint(1, 30)
        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        trimap = cv2.resize(trimap, (w, h), interpolation=cv2.INTER_NEAREST)
        sample['trimap'] = trimap

        # generate mask
        low = 0.01
        high = 1.0
        thres = random.random() * (high - low) + low
        seg_mask = (alpha >= thres).astype(np.int).astype(np.uint8)
        random_num = random.randint(0, 3)
        if random_num == 0:
            seg_mask = cv2.erode(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        elif random_num == 1:
            seg_mask = cv2.dilate(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        elif random_num == 2:
            seg_mask = cv2.erode(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
            seg_mask = cv2.dilate(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        elif random_num == 3:
            seg_mask = cv2.dilate(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
            seg_mask = cv2.erode(
                seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])

        seg_mask = cv2.resize(
            seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        sample['mask'] = seg_mask

        return sample

class Composite(object):
    # 合成前景和背景图像
    def __call__(self, sample):
        # 调整前景、背景和 alpha 通道的像素值范围
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        alpha[alpha < 0] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0] = 0
        fg[fg > 255] = 255
        bg[bg < 0] = 0
        bg[bg > 255] = 255

        # 使用 alpha 通道合成图像
        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        sample['image'] = image
        return sample


class CutMask(object):
    # 随机裁剪 mask
    def __init__(self, perturb_prob=0):
        self.perturb_prob = perturb_prob

    def __call__(self, sample):
        # 根据概率决定是否进行裁剪
        if np.random.rand() < self.perturb_prob:
            return sample

        mask = sample['mask']  # 获取 mask
        h, w = mask.shape
        # 随机选择裁剪尺寸
        perturb_size_h, perturb_size_w = random.randint(
            h // 4, h // 2), random.randint(w // 4, w // 2)
        x = random.randint(0, h - perturb_size_h)
        y = random.randint(0, w - perturb_size_w)
        x1 = random.randint(0, h - perturb_size_h)
        y1 = random.randint(0, w - perturb_size_w)

        # 将裁剪的区域替换为另一个随机位置的区域
        mask[x:x+perturb_size_h, y:y+perturb_size_w] = mask[x1:x1 +
                                                            perturb_size_h, y1:y1+perturb_size_w].copy()

        sample['mask'] = mask
        return sample


class DataGenerator(Dataset):
    # 数据集类，用于生成训练、验证或测试数据
    def __init__(self, data, crop_size=512, phase="train"):
        self.phase = phase
        self.crop_size = crop_size
        self.alpha = data.alpha

        # 根据不同阶段加载数据
        if self.phase == "train":
            self.fg = data.fg
            self.bg = data.bg
            self.merged = []
            self.trimap = []

        else:
            self.fg = []
            self.bg = []
            self.merged = data.merged
            self.trimap = data.trimap

        # 图像处理的一系列变换操作
        train_trans = [
            RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
            GenMask(),
            CutMask(perturb_prob=CUTMASK_PROB),
            RandomCrop((self.crop_size, self.crop_size)),
            RandomJitter(),
            Composite(),
            ToTensor(phase="train")]

        test_trans = [OriginScale(), ToTensor()]

        # 根据阶段选择相应的图像变换
        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'val':
                transforms.Compose([
                    OriginScale(),
                    ToTensor()
                ]),
            'test':
                transforms.Compose(test_trans)
        }[phase]

        self.fg_num = len(self.fg)

    def __getitem__(self, idx):
        # 根据阶段加载图像数据和相应的 alpha 通道数据
        if self.phase == "train":
            fg = cv2.imread(self.fg[idx % self.fg_num])
            alpha = cv2.imread(
                self.alpha[idx % self.fg_num], 0).astype(np.float32)/255
            bg = cv2.imread(self.bg[idx], 1)

            # 合成前景和 alpha 通道
            fg, alpha = self._composite_fg(fg, alpha, idx)

            # 获取图像名称
            image_name = os.path.split(self.fg[idx % self.fg_num])[-1]
            sample = {'fg': fg, 'alpha': alpha,
                      'bg': bg, 'image_name': image_name}

        else:
            image = cv2.imread(self.merged[idx])
            alpha = cv2.imread(self.alpha[idx], 0)/255.
            trimap = cv2.imread(self.trimap[idx], 0)
            mask = (trimap >= 170).astype(np.float32)
            image_name = os.path.split(self.merged[idx])[-1]

            sample = {'image': image, 'alpha': alpha, 'trimap': trimap,
                      'mask': mask, 'image_name': image_name, 'alpha_shape': alpha.shape}

        # 应用图像变换
        sample = self.transform(sample)

        return sample

    def _composite_fg(self, fg, alpha, idx):

        # 根据概率合成两张前景图像
        if np.random.rand() < 0.5:
            idx2 = np.random.randint(self.fg_num) + idx
            fg2 = cv2.imread(self.fg[idx2 % self.fg_num])
            alpha2 = cv2.imread(
                self.alpha[idx2 % self.fg_num], 0).astype(np.float32)/255.
            h, w = alpha.shape
            fg2 = cv2.resize(
                fg2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha2 = cv2.resize(
                alpha2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

            alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
            if np.any(alpha_tmp < 1):
                fg = fg.astype(
                    np.float32) * alpha[:, :, None] + fg2.astype(np.float32) * (1 - alpha[:, :, None])
                # 两个50%透明度的重叠应为25%
                alpha = alpha_tmp
                fg = fg.astype(np.uint8)

        # 根据概率对图像进行缩放
        if np.random.rand() < 0.25:
            fg = cv2.resize(
                fg, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha = cv2.resize(
                alpha, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        return fg, alpha

    def __len__(self):
        # 返回数据集长度
        if self.phase == "train":
            return len(self.bg)
        else:
            return len(self.alpha)


class MultiDataGeneratorDoubleSet(Dataset):
    # 将数据集划分为训练集和验证集
    def __init__(self, data, crop_size=1024, phase="train",norm_type='imagenet'):
        self.phase = phase
        self.crop_size = crop_size
        data = instantiate_from_config(data)

        # 根据阶段加载数据
        if self.phase == "train":
            self.alpha = data.alpha_train
            self.merged = data.merged_train
            self.trimap = data.trimap_train

        elif self.phase == "val":
            self.alpha = data.alpha_val
            self.merged = data.merged_val
            self.trimap = data.trimap_val

        train_trans = [
            # RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),

            # CutMask(perturb_prob=CUTMASK_PROB),
            CropResize((self.crop_size, self.crop_size)),
            # RandomJitter(),
            ToTensor(phase="val",norm_type=norm_type)]

        val_trans = [CropResize(
            (self.crop_size, self.crop_size)), ToTensor(phase="val",norm_type=norm_type)]

        self.transform = {
            'train':
                transforms.Compose(train_trans),

            'val':
                transforms.Compose(val_trans)
        }[phase]

        self.alpha_num = len(self.alpha)

    def __getitem__(self, idx):

        # 加载图像数据
        image = cv2.imread(self.merged[idx])
        alpha = cv2.imread(self.alpha[idx], 0)/255.
        trimap = cv2.imread(self.trimap[idx], 0).astype(np.float32)
        mask = (trimap >= 170).astype(np.float32)
        image_name = os.path.split(self.merged[idx])[-1]

        dataset_name = self.get_dataset_name(image_name)
        sample = {'image': image, 'alpha': alpha, 'trimap': trimap,
                  'mask': mask, 'image_name': image_name, 'alpha_shape': alpha.shape, 'dataset_name': dataset_name}

        sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.alpha)

    def get_dataset_name(self, image_name):
        # 获取数据集名称
        image_name = image_name.split('.')[0]
        if image_name.startswith('o_'):
            return 'AIM'
        elif image_name.endswith('_o') or image_name.endswith('_5k'):
            return 'PPM'
        elif image_name.startswith('m_'):
            return 'AM2k'
        elif image_name.endswith('_input'):
            return 'RWP636'
        elif image_name.startswith('p_'):
            return 'P3M'
        
        else:
            # raise ValueError('image_name {} not recognized'.format(image_name))
            return 'RM1k'
        
class ContextDataset(Dataset):
    # 上下文数据集
    def __init__(self, data, crop_size=1024, phase="train",norm_type='imagenet'):
        self.phase = phase
        self.crop_size = crop_size
        data = instantiate_from_config(data)

        # 根据阶段加载数据
        if self.phase == "train":
            self.dataset = data.dataset_train
            self.image_class_dict = data.image_class_dict_train

        elif self.phase == "val":
            self.dataset = data.dataset_val
            self.image_class_dict = data.image_class_dict_val

        # 将字典转换为列表
        for key, value in self.image_class_dict.items():
            self.image_class_dict[key] = list(value.items())
        self.dataset = list(self.dataset.items())
        
        train_trans = [
            # RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),

            # CutMask(perturb_prob=CUTMASK_PROB),
            CropResize((self.crop_size, self.crop_size)),
            # RandomJitter(),
            ToTensor(phase="val",norm_type=norm_type)]

        val_trans = [CropResize(
            (self.crop_size, self.crop_size)), ToTensor(phase="val",norm_type=norm_type)]

        self.transform = {
            'train':
                transforms.Compose(train_trans),

            'val':
                transforms.Compose(val_trans)
        }[phase]

    def __getitem__(self, idx):
        # 获取数据样本
        cv2.setNumThreads(0)
        
        image_name, image_info = self.dataset[idx]

        # 获取图像样本
        dataset_name = image_info['dataset_name']
        image_sample = self.get_sample(image_name, dataset_name)

        # 获取上下文图像
        class_name = str(
            image_info['class'])+'-'+str(image_info['sub_class'])+'-'+str(image_info['HalfOrFull'])
        (context_image_name, context_dataset_name) = self.image_class_dict[class_name][np.random.randint(
            len(self.image_class_dict[class_name]))]
        context_image_sample = self.get_sample(
            context_image_name, context_dataset_name)

        # 合并图像和上下文
        image_sample['context_image'] = context_image_sample['image']
        image_sample['context_guidance'] = context_image_sample['alpha']
        image_sample['context_image_name'] = context_image_sample['image_name']

        return image_sample

    def __len__(self):
        # 返回数据集长度
        return len(self.dataset)

    def get_sample(self, image_name, dataset_name):
        # 获取图像样本
        cv2.setNumThreads(0)
        image_dir, label_dir, trimap_dir, merged_ext, alpha_ext, trimap_ext = get_dir_ext(
            dataset_name)
        image_path = os.path.join(image_dir, image_name + merged_ext) if 'open-images' not in dataset_name else os.path.join(
            image_dir, image_name.split('_')[0] + merged_ext)
        label_path = os.path.join(label_dir, image_name + alpha_ext)
        trimap_path = os.path.join(trimap_dir, image_name + trimap_ext)

        image = cv2.imread(image_path)
        alpha = cv2.imread(label_path, 0)/255.
        trimap = cv2.imread(trimap_path, 0).astype(np.float32)
        mask = (trimap >= 170).astype(np.float32)
        image_name = os.path.split(image_path)[-1]

        sample = {'image': image, 'alpha': alpha, 'trimap': trimap,
                  'mask': mask, 'image_name': image_name, 'alpha_shape': alpha.shape, 'dataset_name': dataset_name}

        sample = self.transform(sample)
        return sample
    def get_sample_example(self, image_dir, mask_dir, img_list, mask_list, index):
        # 获取示例图像样本
        image = cv2.imread(os.path.join(image_dir, img_list[index]))
        alpha = cv2.imread(os.path.join(mask_dir, mask_list[index]), 0)/255.

        # 将 alpha 调整到与图像相同的大小
        alpha = cv2.resize(alpha, (image.shape[1], image.shape[0]))

        # 未使用
        trimap = cv2.imread(os.path.join(mask_dir, mask_list[index]))/255.
        mask = (trimap >= 170).astype(np.float32)
        image_name = ''
        dataset_name = ''
        
        sample = {'image': image, 'alpha': alpha, 'trimap': trimap,
                  'mask': mask, 'image_name': image_name, 'alpha_shape': alpha.shape, 'dataset_name': dataset_name}

        sample = self.transform(sample)
        return sample['image'], sample['alpha']
    
class InContextDataset(Dataset):
    # 上下文数据集
    def __init__(self, data, crop_size=1024, phase="train",norm_type='imagenet'):
        self.phase = phase
        self.crop_size = crop_size
        data = instantiate_from_config(data)

        # 根据阶段加载数据
        if self.phase == "train":
            self.dataset = data.dataset_train
            self.image_class_dict = data.image_class_dict_train

        elif self.phase == "val":
            self.dataset = data.dataset_val
            self.image_class_dict = data.image_class_dict_val

        # 将字典转换为列表
        for key, value in self.image_class_dict.items():
            self.image_class_dict[key] = list(value.items())
        self.dataset = list(self.dataset.items())

        train_trans = [
            # RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),

            # CutMask(perturb_prob=CUTMASK_PROB),
            CropResize((self.crop_size, self.crop_size)),
            # RandomJitter(),
            ToTensor(phase="val",norm_type=norm_type)]

        val_trans = [CropResize(
            (self.crop_size, self.crop_size)), ToTensor(phase="val",norm_type=norm_type)]

        self.transform = {
            'train':
                transforms.Compose(train_trans),

            'val':
                transforms.Compose(val_trans)
        }[phase]

    def __getitem__(self, idx):
        # 获取数据样本
        cv2.setNumThreads(0)

        image_name, image_info = self.dataset[idx]

        # 获取图像样本
        dataset_name = image_info['dataset_name']
        image_sample = self.get_sample(image_name, dataset_name)

        # 获取上下文图像
        class_name = str(
            image_info['class'])+'-'+str(image_info['sub_class'])+'-'+str(image_info['HalfOrFull'])
        
        context_set = self.image_class_dict[class_name]
        if len(context_set) > 2:
            # 从上下文集合中删除图像名称（字典）
            context_set = [x for x in context_set if x[0] != image_name]
            
        (reference_image_name, context_dataset_name) = context_set[np.random.randint(
            len(context_set))]
        reference_image_sample = self.get_sample(
            reference_image_name, context_dataset_name)

        # 合并图像和上下文
        image_sample['reference_image'] = reference_image_sample['source_image']
        image_sample['guidance_on_reference_image'] = reference_image_sample['alpha']
        image_sample['reference_image_name'] = reference_image_sample['image_name']

        return image_sample

    def __len__(self):
        # 返回数据集长度
        return len(self.dataset)

    def get_sample(self, image_name, dataset_name):
        # 获取图像样本
        cv2.setNumThreads(0)
        image_dir, label_dir, trimap_dir, merged_ext, alpha_ext, trimap_ext = get_dir_ext(
            dataset_name)
        image_path = os.path.join(image_dir, image_name + merged_ext) if 'open-images' not in dataset_name else os.path.join(
            image_dir, image_name.split('_')[0] + merged_ext)
        label_path = os.path.join(label_dir, image_name + alpha_ext)
        trimap_path = os.path.join(trimap_dir, image_name + trimap_ext)

        image = cv2.imread(image_path)
        alpha = cv2.imread(label_path, 0)/255.
        trimap = cv2.imread(trimap_path, 0).astype(np.float32)
        mask = (trimap >= 170).astype(np.float32)
        image_name = os.path.split(image_path)[-1]

        if 'open-images' in dataset_name:
            # 将 alpha 量化为 0 和 1
            alpha[alpha < 0.5] = 0
            alpha[alpha >= 0.5] = 1
            
        sample = {'image': image, 'alpha': alpha, 'trimap': trimap,
                  'mask': mask, 'image_name': image_name, 'alpha_shape': alpha.shape, 'dataset_name': dataset_name}

        sample = self.transform(sample)
        
        # 修改'image'为'source_image'
        sample['source_image'] = sample['image']
        del sample['image']
        
        return sample

```