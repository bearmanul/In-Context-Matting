```python
import json
import os
import glob  # 用于文件路径模式匹配
import functools  # 提供高阶函数
import numpy as np


class ImageFile(object):
    def __init__(self, phase='train'):
        # 初始化图像文件类，设置阶段（训练或测试）
        self.phase = phase
        # 初始化随机数生成器
        self.rng = np.random.RandomState(0)

    def _get_valid_names(self, *dirs, shuffle=True):
        # 获取有效文件名
        name_sets = [self._get_name_set(d) for d in dirs]

        # 使用 reduce 函数计算交集
        def _join_and(a, b):
            return a & b

        # 获取所有目录的交集
        valid_names = list(functools.reduce(_join_and, name_sets))
        # 打乱文件名顺序
        if shuffle:
            self.rng.shuffle(valid_names)

        return valid_names

    @staticmethod
    def _get_name_set(dir_name):
        # 获取目录下所有文件的名称集合
        path_list = glob.glob(os.path.join(dir_name, '*'))
        name_set = set()
        for path in path_list:
            name = os.path.basename(path)
            name = os.path.splitext(name)[0]
            name_set.add(name)
        return name_set

    @staticmethod
    def _list_abspath(data_dir, ext, data_list):
        # 根据文件名列表生成绝对路径
        return [os.path.join(data_dir, name + ext) for name in data_list]


class ImageFileTrain(ImageFile):
    def __init__(self,
                 alpha_dir="train_alpha",
                 fg_dir="train_fg",
                 bg_dir="train_bg",
                 alpha_ext=".jpg",
                 fg_ext=".jpg",
                 bg_ext=".jpg"):
        # 初始化训练图像文件类
        super(ImageFileTrain, self).__init__(phase="train")

        self.alpha_dir = alpha_dir
        self.fg_dir = fg_dir
        self.bg_dir = bg_dir
        self.alpha_ext = alpha_ext
        self.fg_ext = fg_ext
        self.bg_ext = bg_ext

        # 获取有效的前景文件列表
        self.valid_fg_list = self._get_valid_names(self.fg_dir, self.alpha_dir)
        # 获取背景文件列表
        self.valid_bg_list = [os.path.splitext(name)[0] for name in os.listdir(self.bg_dir)]

        # 生成文件的绝对路径
        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_fg_list)
        self.fg = self._list_abspath(self.fg_dir, self.fg_ext, self.valid_fg_list)
        self.bg = self._list_abspath(self.bg_dir, self.bg_ext, self.valid_bg_list)

    def __len__(self):
        # 返回训练文件的数量
        return len(self.alpha)


class ImageFileTest(ImageFile):
    def __init__(self,
                 alpha_dir="test_alpha",
                 merged_dir="test_merged",
                 trimap_dir="test_trimap",
                 alpha_ext=".png",
                 merged_ext=".png",
                 trimap_ext=".png"):
        # 初始化测试图像文件类
        super(ImageFileTest, self).__init__(phase="test")

        self.alpha_dir = alpha_dir
        self.merged_dir = merged_dir
        self.trimap_dir = trimap_dir
        self.alpha_ext = alpha_ext
        self.merged_ext = merged_ext
        self.trimap_ext = trimap_ext

        # 获取有效的图像文件列表
        self.valid_image_list = self._get_valid_names(
            self.alpha_dir, self.merged_dir, self.trimap_dir, shuffle=False)

        # 生成文件的绝对路径
        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_image_list)
        self.merged = self._list_abspath(self.merged_dir, self.merged_ext, self.valid_image_list)
        self.trimap = self._list_abspath(self.trimap_dir, self.trimap_ext, self.valid_image_list)

    def __len__(self):
        # 返回测试文件的数量
        return len(self.alpha)


dataset = {'AIM', 'PPM', 'AM2k_train', 'AM2k_val', 'RWP636', 'P3M_val_np', 'P3M_train', 'P3M_val_p'}
# 一组数据集的名称集合

def get_dir_ext(dataset):
    # 获取数据集的目录和文件扩展名
    image_dir = './datasets/
    image_dir = './datasets/ICM57/image'
    label_dir = './datasets/ICM57/alpha'
    trimap_dir = './datasets/ICM57/trimap'
    
    merged_ext = '.jpg'
    alpha_ext = '.png'
    trimap_ext = '.png'
    return image_dir, label_dir, trimap_dir, merged_ext, alpha_ext, trimap_ext


class MultiImageFile(object):
    def __init__(self):
        # 初始化多图像文件类
        self.rng = np.random.RandomState(1)

    def _get_valid_names(self, *dirs, shuffle=True):
        # 获取有效文件名
        name_sets = [self._get_name_set(d) for d in dirs]

        # 使用 reduce 函数计算交集
        def _join_and(a, b):
            return a & b

        valid_names = list(functools.reduce(_join_and, name_sets))

        # 确保训练和验证集顺序相同
        if shuffle:
            valid_names.sort()
            self.rng.shuffle(valid_names)

        return valid_names

    @staticmethod
    def _get_name_set(dir_name):
        # 获取目录下所有文件的名称集合
        path_list = glob.glob(os.path.join(dir_name, '*'))
        name_set = set()
        for path in path_list:
            name = os.path.basename(path)
            name = os.path.splitext(name)[0]
            name_set.add(name)
        return name_set

    @staticmethod
    def _list_abspath(data_dir, ext, data_list):
        # 根据文件名列表生成绝对路径
        return [os.path.join(data_dir, name + ext) for name in data_list]


class MultiImageFileDoubleSet(MultiImageFile):
    def __init__(self, ratio=0.9, dataset_name=['AIM', 'PPM', 'AM2k_train', 'AM2k_val', 'RWP636', 'P3M_val_np']):
        # 初始化多图像文件双集类
        super(MultiImageFileDoubleSet, self).__init__()

        self.alpha_train = []
        self.merged_train = []
        self.trimap_train = []
        self.alpha_val = []
        self.merged_val = []
        self.trimap_val = []

        # 为每个数据集加载和分割数据
        for dataset_name_ in dataset_name:
            merged_dir, alpha_dir, trimap_dir, merged_ext, alpha_ext, trimap_ext = get_dir_ext(dataset_name_)
            valid_image_list = self._get_valid_names(alpha_dir, merged_dir, trimap_dir)

            alpha = self._list_abspath(alpha_dir, alpha_ext, valid_image_list)
            merged = self._list_abspath(merged_dir, merged_ext, valid_image_list)
            trimap = self._list_abspath(trimap_dir, trimap_ext, valid_image_list)

            alpha_train, alpha_val = self._split(alpha, ratio)
            merged_train, merged_val = self._split(merged, ratio)
            trimap_train, trimap_val = self._split(trimap, ratio)

            self.alpha_train.extend(alpha_train)
            self.merged_train.extend(merged_train)
            self.trimap_train.extend(trimap_train)
            self.alpha_val.extend(alpha_val)
            self.merged_val.extend(merged_val)
            self.trimap_val.extend(trimap_val)

    def _split(self, data_list, ratio):
        # 按比例分割数据
        num = len(data_list)
        split = int(num * ratio)
        return data_list[:split], data_list[split:]


class ContextData():
    '''
    dataset_name: 对应于 /datasets 文件夹中的数据集文件

    返回:
    dataset: dict2list，key: image_name,
             value: {"dataset_name": "AIM", "class": "animal",
                     "sub_class": null, "HalfOrFull": "half", "TransparentOrOpaque": "SO"}

    image_class_dict: dict，key: class_name (class-sub_class)
                      value: dict，key: image_name，value: dataset_name
    '''

    def __init__(self, ratio=0.9, dataset_name=['PPM', 'AM2k', 'RWP636', 'P3M_val_np']):
        dataset = {}
        for dataset_name_ in dataset_name:
            json_dir = os.path.join('datasets', dataset_name_ + '.json')
            # 读取 json 文件并附加到数据集中
            with open(json_dir) as f:
                new_data = json.load(f)
                # 过滤出 "instance_area_ratio" 列表中每个元素大于0.1的项
                # 检查 "instance_area_ratio" 是否存在
                if 'instance_area_ratio' in new_data[list(new_data.keys())[0]].keys():
                    new_data_ = {}
                    for k, v in new_data.items():
                        if min(v['instance_area_ratio']) < 0.3:
                            x = min(v['instance_area_ratio'])
                            r = np.random.rand()
                            if 100 * (0.6 - x) * x / 9 > r ** 0.2:
                                new_data_[k] = v
                        else:
                            new_data_[k] = v
                    new_data = new_data_
                dataset.update(new_data)

        # 使用种子打乱数据集
        self.rng = np.random.RandomState(1)
        dataset_list = list(dataset.items())
        dataset_list.sort()
        self.rng.shuffle(dataset_list)

        # 将数据集拆分为训练集和验证集
        dataset_train, dataset_val = self._split(dataset_list, ratio)

        # 获取 image_class_dict
        image_class_dict_train = self.get_image_class_dict(dataset_train)
        image_class_dict_val = self.get_image_class_dict(dataset_val)

        self.image_class_dict_train = image_class_dict_train
        self.image_class_dict_val = image_class_dict_val
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val

    def _split(self, data_list, ratio):
        # 按比例拆分数据集
        num = len(data_list)
        split = int(num * ratio)

        dataset_train = dict(data_list[:split])
        dataset_val = dict(data_list[split:])
        return dataset_train, dataset_val

    def get_image_class_dict(self, dataset):
        # 获取 image_class_dict
        image_class_dict = {}
        for image_name, image_info in dataset.items():
            class_name = str(image_info['class']) + '-' + str(image_info['sub_class']) + '-' + str(image_info['HalfOrFull'])
            if class_name not in image_class_dict.keys():
                image_class_dict[class_name] = {}
                image_class_dict[class_name][image_name] = image_info['dataset_name']
            else:
                image_class_dict[class_name][image_name] = image_info['dataset_name']
        return image_class_dict


if __name__ == "__main__":
    # 测试 MultiImageFileDoubleSet 类
    test = MultiImageFileDoubleSet()
    print(0)

```