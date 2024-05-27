
[bottleneck介绍](https://zhuanlan.zhihu.com/p/349717627)

```python
import torch
from torch import nn
from torch.nn import functional as F

def c2_msra_fill(module: nn.Module) -> None:
    """
    使用 Caffe2 中实现的 "MSRAFill" 来初始化 `module.weight`。
    同时将 `module.bias` 初始化为 0。

    Args:
        module (torch.nn.Module): 需要初始化的模块。
    """
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)

class Conv2d(torch.nn.Conv2d):
    """
    对 :class:`torch.nn.Conv2d` 进行包装，支持空输入和更多特性。
    """

    def __init__(self, *args, **kwargs):
        """
        除了 `torch.nn.Conv2d` 中的参数外，还支持以下额外的关键字参数：

        Args:
            norm (nn.Module, optional): 规范化层
            activation (callable(Tensor) -> Tensor): 可调用的激活函数

        这里假定规范化层在激活函数之前使用。
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm 不支持空输入!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): BN、SyncBN、FrozenBN、GN 之一；
            或者一个接受通道数并返回规范化层的可调用对象。

    Returns:
        nn.Module or None: 规范化层
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
        }[norm]
    return norm(out_channels)

class CNNBlockBase(nn.Module):
    """
    CNN 块假定具有输入通道数、输出通道数和步长。
    `forward()` 方法的输入和输出必须是 NCHW 张量。
    该方法可以执行任意计算，但必须符合给定的通道和步长规范。

    属性:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        stride (int): 步长
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        任何子类的 `__init__` 方法也应该包含这些参数。

        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            stride (int): 步长
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        """
        使该块不可训练。
        该方法将所有参数设置为 `requires_grad=False`，
        并将所有 BatchNorm 层转换为 FrozenBatchNorm。

        Returns:
            该块本身
        """
        for p in self.parameters():
            p.requires_grad = False
        return self
    
class BottleneckBlock(CNNBlockBase):
    """
    ResNet-50、101 和 152 中使用的标准瓶颈残差块。
    包含三个卷积层，内核大小分别为 1x1、3x3、1x1，并在需要时包含投影快捷连接。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="GN",
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Args:
            bottleneck_channels (int): 3x3 "瓶颈" 卷积层的输出通道数。
            num_groups (int): 3x3 卷积层的组数。
            norm (str or callable): 所有卷积层的规范化。
                有关支持的格式，请参见 :func:`layers.get_norm`。
            stride_in_1x1 (bool): 当 stride>1 时，是否将 stride 放在
                第一个 1x1 卷积中或瓶颈 3x3 卷积中。
            dilation (int): 3x3 卷积层的扩张率。
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                c2_msra_fill(layer)

        # 初始化每个残差分支中的最后一个规范化层为零，
        # 以便在开始时，残差分支以零开始，
        # 并且每个残差块的行为类似于一个恒等映射。
        # 参见 "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" 的第 5.1 节：
        # "对于 BN 层，可学习的缩放系数 γ 被初始化为 1，
        # 除了每个残差块的最后一个 BN，其中 γ 被初始化为 0。"
        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO：当我们需要使用这段代码从头开始训练 GN 模型时，这样做会降低性能。
        # 在需要时将其作为选项添加。

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out

```