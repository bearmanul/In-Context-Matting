# 代码注释
```python
# 导入必要的模块和类型
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.nn.functional as nnf
import numpy as np
import abc

# 设置低资源标志位
LOW_RESOURCE = False 

# 定义一个抽象基类 AttentionControl
class AttentionControl(abc.ABC):
    
    # 定义一个步进回调函数，默认返回输入的张量
    def step_callback(self, x_t):
        return x_t
    
    # 定义一个在步骤之间调用的函数，默认不执行任何操作
    def between_steps(self):
        return
    
    # 定义一个属性，用于获取无条件注意力层的数量
    @property
    def num_uncond_att_layers(self):
        # 如果是低资源模式，返回注意力层数；否则返回0
        return self.num_att_layers if LOW_RESOURCE else 0
    
    # 定义一个抽象方法，子类必须实现
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    # 定义类的调用行为
    def __call__(self, attn, is_cross: bool, place_in_unet: str, ensemble_size=1, token_batch_size=1):
        # 如果当前注意力层大于等于无条件注意力层数量
        if self.cur_att_layer >= self.num_uncond_att_layers:
            # 如果是低资源模式，调用 forward 函数
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet, ensemble_size, token_batch_size)
            else:
                h = attn.shape[0]
                # 通过 forward 函数处理注意力张量
                attn = self.forward(attn, is_cross, place_in_unet, ensemble_size, token_batch_size)
        # 增加当前注意力层计数
        self.cur_att_layer += 1
        # 如果当前注意力层计数达到总层数，重置计数并增加步骤计数
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        # 返回处理后的注意力张量
        return attn
    
    # 重置函数，重置步骤和注意力层计数
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    # 初始化函数，初始化步骤和注意力层计数
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

# 定义一个继承自 AttentionControl 的空控制类
class EmptyControl(AttentionControl):
    
    # 实现抽象方法 forward，直接返回输入的注意力张量
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
# 定义一个继承自 AttentionControl 的注意力存储类
class AttentionStore(AttentionControl):

    # 定义一个静态方法，用于获取空的存储结构
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    # 实现抽象方法 forward
    def forward(self, attn, is_cross: bool, place_in_unet: str, ensemble_size=1, token_batch_size=1):
        # 获取注意力张量的头数量
        num_head = attn.shape[0]//token_batch_size
        # 根据位置和类型生成存储键
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # 如果有存储分辨率并且是自注意力
        if self.store_res is not None:
            if attn.shape[1] in self.store_res and (is_cross is False):
                # 对注意力张量进行维度变换和平均
                attn = attn.reshape(-1, ensemble_size, *attn.shape[1:])
                attn = attn.mean(dim=1)
                attn = attn.reshape(-1,num_head , *attn.shape[1:])
                attn = attn.mean(dim=1)
                # 将处理后的注意力张量存储
                self.step_store[key].append(attn)
        # 如果注意力张量大小不超过48*48且是自注意力
        elif attn.shape[1] <= 48 ** 2 and (is_cross is False):  # 避免内存开销
            # 对注意力张量进行维度变换和平均
            attn = attn.reshape(-1, ensemble_size, *attn.shape[1:])
            attn = attn.mean(dim=1)
            attn = attn.reshape(-1,num_head , *attn.shape[1:])
            attn = attn.mean(dim=1)
            # 将处理后的注意力张量存储
            self.step_store[key].append(attn)

        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

    # 在步骤之间调用的函数
    def between_steps(self):
        # 如果注意力存储为空，初始化存储
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            # 将步骤存储的内容累加到总存储中
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        # 重置步骤存储
        self.step_store = self.get_empty_store()

    # 获取平均注意力的函数
    def get_average_attention(self):
        # 计算每一步的平均注意力
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    # 重置函数
    def reset(self):
        super(AttentionStore, self).reset()
        # 删除步骤存储并清空 CUDA 缓存
        del self.step_store
        torch.cuda.empty_cache()
        # 重新初始化步骤存储和注意力存储
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    # 初始化函数
    def __init__(self,store_res = None):
        super(AttentionStore, self).__init__()
        # 初始化步骤存储和注意力存储
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        # 处理存储分辨率
        store_res = [store_res] if isinstance(store_res, int) else list(store_res) 
        self.store_res = []
        for res in store_res:
            self.store_res.append(res**2)

```