```python
import pytorch_lightning as pl
from icm.util import instantiate_from_config
from torch.utils.data import DataLoader
import numpy as np
import torch

def worker_init_fn(worker_id):
    # 为每个worker设置一个不同的随机种子，以确保数据加载的随机性
    np.random.seed(torch.randint(0, 2**32 - 1, size=(1,)).item())

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, train=None, validation=None, test=None, predict=None, num_workers=None,
                 batch_size=None, shuffle_train=False, batch_size_val=None):
        super().__init__()
        # 初始化数据模块的参数
        self.batch_size = batch_size  # 训练集的批量大小
        self.batch_size_val = batch_size_val  # 验证集的批量大小
        self.dataset_configs = dict()  # 数据集配置字典
        self.num_workers = num_workers if num_workers is not None else batch_size * 2  # 工作线程数
        self.shuffle_train = shuffle_train  # 是否对训练数据进行洗牌
        
        # 如果传入了训练集配置，添加到数据集配置字典中，并创建对应的数据加载方法
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        
        # 如果传入了验证集配置，添加到数据集配置字典中，并创建对应的数据加载方法
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        
        # 用于调试时调用setup方法
        # self.setup()

    def setup(self, stage=None):
        # 根据数据集配置实例化数据集
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        
    def _train_dataloader(self):
        # 创建并返回训练数据加载器
        return DataLoader(self.datasets["train"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=self.shuffle_train,
                          worker_init_fn=worker_init_fn)
        
    def _val_dataloader(self):
        # 创建并返回验证数据加载器
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size if self.batch_size_val is None else self.batch_size_val,
                          num_workers=self.num_workers,
                          shuffle=True,  # 验证集通常不需要洗牌，这里是为了调试
                          worker_init_fn=worker_init_fn)
        
    def prepare_data(self):
        # 调用父类的prepare_data方法
        return super().prepare_data()

```

