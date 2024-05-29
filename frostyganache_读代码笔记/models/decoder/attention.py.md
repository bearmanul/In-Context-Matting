
# 注意力机制
对我来说很难理解，看了好几篇文章：
1. 感性认识：人类视觉的选择性注意力机制。翻译时，注意力分配模型分配给不同英文单词的注意力大小不同。[注意力机制（Attention mechanism）基本原理详解及应用 - Jerry_Jin - 博客园 (cnblogs.com)](https://www.cnblogs.com/jins-note/p/13056604.html)
2. 认识Query, Key, Value（q, k, v）：[Attention：注意力机制 - killens - 博客园 (cnblogs.com)](https://www.cnblogs.com/killens/p/16303795.html)
3. 公式和代码实现：[人工智能 - Vision Transformers的注意力层概念解释和代码实现 - deephub - SegmentFault 思否](https://segmentfault.com/a/1190000044678798)
4. self attention和cross attention :[Self -Attention、Multi-Head Attention、Cross-Attention_cross attention-CSDN博客](https://blog.csdn.net/philosophyatmath/article/details/128013258)
# permute
[PyTorch 两大转置函数 transpose() 和 permute(), 以及RuntimeError: invalid argument 2: view size is not compati_transpose permute-CSDN博客](https://blog.csdn.net/xinjieyuan/article/details/105232802)

# GELU激活函数
[GELU激活函数介绍和笔记-CSDN博客](https://blog.csdn.net/kkxi123456/article/details/122694916)

# element-wise product
$\bigodot$ element-wise product
表示两个矩阵对应位置元素进行乘积 ($a_{ij}*b_{ij}$)

# 代码注释
```python
import torch  # 导入PyTorch库
from torch import Tensor, nn  # 从PyTorch中导入Tensor和神经网络模块
import math  # 导入数学模块
from typing import Tuple, Type  # 导入类型提示模块
import os  # 导入操作系统接口模块
import numpy as np  # 导入NumPy库
from PIL import Image  # 导入Python图像库

class Attention(nn.Module):
    """
    一个注意力层，可以在投影到查询、键和值之后对嵌入的大小进行下采样。
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.internal_dim = embedding_dim // downsample_rate  # 内部维度，可能会下采样
        self.num_heads = num_heads  # 注意力头的数量
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."  # 确保内部维度能被头数整除

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)  # 查询投影
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)  # 键投影
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)  # 值投影
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)  # 输出投影

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape  # 获取张量的形状
        x = x.reshape(b, n, num_heads, c // num_heads)  # 重塑张量
        return x.transpose(1, 2)  # 转置张量以分离头

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape  # 获取分离后的张量形状
        x = x.transpose(1, 2)  # 转置张量以组合头
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # 重塑张量为原始形状

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # 输入投影
        q = self.q_proj(q)  # 查询投影
        k = self.k_proj(k)  # 键投影
        v = self.v_proj(v)  # 值投影

        # 分离到多个头
        q = self._separate_heads(q, self.num_heads)  # 查询分离头
        k = self._separate_heads(k, self.num_heads)  # 键分离头
        v = self._separate_heads(v, self.num_heads)  # 值分离头

        # 注意力计算
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)  # 缩放
        attn = torch.softmax(attn, dim=-1)  # 应用softmax得到注意力权重

        # 获得输出
        out = attn @ v  # 注意力加权的值
        out = self._recombine_heads(out)  # 重新组合头
        out = self.out_proj(out)  # 输出投影

        return out  # 返回输出

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)  # 第一个线性层
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)  # 第二个线性层
        self.act = act()  # 激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))  # 前向传播

# 从https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
# 自身来自https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))  # 初始化权重参数
        self.bias = nn.Parameter(torch.zeros(num_channels))  # 初始化偏置参数
        self.eps = eps  # 防止除零的小数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)  # 计算均值
        s = (x - u).pow(2).mean(1, keepdim=True)  # 计算方差
        x = (x - u) / torch.sqrt(s + self.eps)  # 归一化
        x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 重新加权和偏置
        return x  # 返回归一化后的张量

```