# ChatGPT的概括

> 这个类定义了一组用于训练神经网络的损失函数，特别适用于图像分割和抠图任务。让我们逐个解释其组成部分：
> 
> 1. **LossFunction**：这个类包含了用于训练的各种损失函数。在初始化时，可通过 `losses` 参数指定可用的损失函数。每个损失函数都作为类内的单独方法实现。可用的损失函数包括：
> 
>     - `loss_gradient_penalty`：计算梯度惩罚损失，如果预测的梯度与真实值偏离，则对模型进行惩罚。
>     - `loss_pha_laplacian`：计算预测和真实 alpha 抠图之间的拉普拉斯损失。
>     - `unknown_l1_loss`：计算三分图中未知区域内的 L1 损失。
>     - `known_l1_loss`：计算三分图中已知区域内的 L1 损失。
>     - `smooth_l1_loss`：计算预测和真实 alpha 抠图之间的平滑 L1 损失。
>     - `known_smooth_l1_loss`：计算三分图中已知区域内的平滑 L1 损失。
>     - `cross_entropy_loss`：计算预测和真实 alpha 抠图之间的二元交叉熵损失。
>     - `focal_loss`：计算聚焦损失，是二元交叉熵损失的一种变体，专注于难例。
> 
>     `forward` 方法负责根据输入的预测、目标和样本图计算损失。它遍历指定的损失函数并将损失聚合到一个字典中。
> 
> 2. **LossFunction2**：这个类与 `LossFunction` 类似，但允许为分割和抠图任务使用不同的损失函数集。它具有单独的参数 `losses_seg` 和 `losses_matting`，分别表示分割和抠图任务的不同损失集。`forward_single_sample` 方法计算单个样本的损失，而 `forward` 方法在多个样本之间聚合损失。
> 
> 3. **辅助函数**：文件中还包括一些辅助函数：
> 
>     - `laplacian_loss`：计算预测和真实 alpha 抠图之间的拉普拉斯损失。
>     - `laplacian_pyramid`：构建图像的拉普拉斯金字塔。
>     - `gauss_kernel`：生成高斯核。
>     - `gauss_convolution`：执行高斯卷积。
>     - `downsample` 和 `upsample`：使用高斯卷积对图像进行下采样和上采样。
>     - `crop_to_even_size`：将图像裁剪为偶数尺寸。
> 
> 这些函数在内部用于损失函数的计算。总的来说，这些类和函数提供了训练图像分割和抠图任务的深度学习模型所需的基本组件。

# 拉普拉斯损失
[拉普拉斯损失](./拉普拉斯损失.md)     

# 代码注释

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import focal_loss

class LossFunction(nn.Module):
    '''
    Loss function set
    losses=['unknown_l1_loss', 'known_l1_loss',
            'loss_pha_laplacian', 'loss_gradient_penalty',
            'smooth_l1_loss', 'cross_entropy_loss', 'focal_loss']
    '''
    def __init__(self,
                 *,
                 losses,
                 ):
        super(LossFunction, self).__init__()
        self.losses = losses

    def loss_gradient_penalty(self, sample_map ,preds, targets):
        preds = preds['phas']
        targets = targets['phas']
        h,w = sample_map.shape[2:]
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            #sample_map for unknown area
            scale = sample_map.shape[0]*262144/torch.sum(sample_map)

        #gradient in x
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(dtype=preds.type())
        delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
        delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)

        #gradient in y 
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(dtype=preds.type())
        delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
        delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)

        #loss
        loss = (F.l1_loss(delta_pred_x*sample_map, delta_gt_x*sample_map)* scale + \
            F.l1_loss(delta_pred_y*sample_map, delta_gt_y*sample_map)* scale + \
            0.01 * torch.mean(torch.abs(delta_pred_x*sample_map))* scale +  \
            0.01 * torch.mean(torch.abs(delta_pred_y*sample_map))* scale)

        return dict(loss_gradient_penalty=loss)

    def loss_pha_laplacian(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = laplacian_loss(preds['phas'], targets['phas'])

        return dict(loss_pha_laplacian=loss)

    def unknown_l1_loss(self, sample_map, preds, targets):
        h,w = sample_map.shape[2:]
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            #sample_map for unknown area
            scale = sample_map.shape[0]*262144/torch.sum(sample_map)

        # scale = 1

        loss = F.l1_loss(preds['phas']*sample_map, targets['phas']*sample_map)*scale
        return dict(unknown_l1_loss=loss)

    def known_l1_loss(self, sample_map, preds, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1
        h,w = sample_map.shape[2:]
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0]*262144/torch.sum(new_sample_map)
        # scale = 1

        loss = F.l1_loss(preds['phas']*new_sample_map, targets['phas']*new_sample_map)*scale
        return dict(known_l1_loss=loss)

    def smooth_l1_loss(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = F.smooth_l1_loss(preds['phas'], targets['phas'])

        return dict(smooth_l1_loss=loss)

    def known_smooth_l1_loss(self, sample_map, preds, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1
        h,w = sample_map.shape[2:]
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0]*262144/torch.sum(new_sample_map)
        # scale = 1

        loss = F.smooth_l1_loss(preds['phas']*new_sample_map, targets['phas']*new_sample_map)*scale
        return dict(known_l1_loss=loss)
    
    def cross_entropy_loss(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = F.binary_cross_entropy_with_logits(preds['phas'], targets['phas'])

        return dict(cross_entropy_loss=loss)
    
    def focal_loss(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = focal_loss.sigmoid_focal_loss(preds['phas'], targets['phas'], reduction='mean')

        return dict(focal_loss=loss)
    def forward(self, sample_map, preds, targets):
        
        preds = {'phas': preds}
        targets = {'phas': targets}
        losses = dict()
        for k in self.losses:
            if k=='unknown_l1_loss' or k=='known_l1_loss' or k=='loss_gradient_penalty' or k=='known_smooth_l1_loss':
                losses.update(getattr(self, k)(sample_map, preds, targets))
            else:
                losses.update(getattr(self, k)(preds, targets))
        return losses

class LossFunction2(nn.Module):
    '''
    Loss function set
    losses=['unknown_l1_loss', 'known_l1_loss',
            'loss_pha_laplacian', 'loss_gradient_penalty',
            'smooth_l1_loss', 'cross_entropy_loss', 'focal_loss']
    '''
    def __init__(self,
                 *,
                 losses_seg = ['known_smooth_l1_loss'],
                 losses_matting = ['unknown_l1_loss', 'known_l1_loss','loss_pha_laplacian', 'loss_gradient_penalty',],
                 ):
        super(LossFunction2, self).__init__()
        self.losses_seg = losses_seg
        self.losses_matting = losses_matting

    def loss_gradient_penalty(self, sample_map ,preds, targets):
        preds = preds['phas']
        targets = targets['phas']
        h,w = sample_map.shape[2:]
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            #sample_map for unknown area
            scale = sample_map.shape[0]*262144/torch.sum(sample_map)
```
这个 `if-else` 语句用于计算一个变量 `scale`。在这里，它检查 `sample_map` 中所有元素的总和是否为零。如果总和为零，意味着 `sample_map` 中没有非零元素，即没有未知区域，因此 `scale` 被设置为零。否则，如果总和不为零，`scale` 的值被计算为 `sample_map.shape[0] * 262144 / torch.sum(sample_map)`。这个计算过程可能是为了对未知区域的损失进行归一化处理。

## sobel算子

```python
        #gradient in x
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(dtype=preds.type())
        delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
        delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)
```

这段代码执行以下操作：

1. 创建 Sobel 算子作为卷积核，用于在 x 方向计算图像的梯度。Sobel 算子是一种常用的边缘检测算子，用于计算图像中每个像素点的梯度值。这个 Sobel 算子的定义如下：
   
   [[-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]]
   
   这个算子将在 x 方向上对图像进行卷积操作，从而计算出图像在 x 方向上的梯度值。

2. 将输入的预测值 `preds` 和目标值 `targets` 分别与 Sobel 算子进行二维卷积操作，从而计算它们在 x 方向上的梯度。这里使用了 PyTorch 中的 `F.conv2d()` 函数来执行卷积操作，并且通过设置 `padding=1` 来保持输出的大小与输入相同。

这段代码的目的是计算预测值和目标值在 x 方向上的梯度，以便在后续计算损失时使用。  


```python
        #gradient in y 
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(dtype=preds.type())
        delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
        delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)

        #loss
        loss = (F.l1_loss(delta_pred_x*sample_map, delta_gt_x*sample_map)* scale + \
            F.l1_loss(delta_pred_y*sample_map, delta_gt_y*sample_map)* scale + \
            0.01 * torch.mean(torch.abs(delta_pred_x*sample_map))* scale +  \
            0.01 * torch.mean(torch.abs(delta_pred_y*sample_map))* scale)

        return dict(loss_gradient_penalty=loss)
```

这段代码计算了损失值 `loss`，具体计算步骤如下：

1. 计算 x 方向上的预测值梯度 `delta_pred_x` 和目标值梯度 `delta_gt_x`，以及 y 方向上的预测值梯度 `delta_pred_y` 和目标值梯度 `delta_gt_y`。

2. 计算四个部分的损失值：
   - x 方向上的预测值和目标值的 L1 损失乘以 `scale`。
   - y 方向上的预测值和目标值的 L1 损失乘以 `scale`。
   - x 方向上的预测值梯度的绝对值的均值乘以 `0.01` 再乘以 `scale`。
   - y 方向上的预测值梯度的绝对值的均值乘以 `0.01` 再乘以 `scale`。

3. 将以上四个部分的损失值相加，得到最终的损失值 `loss`。

这段代码的目的是计算梯度惩罚损失，其中损失函数包括预测值和目标值之间的 L1 损失，以及梯度的绝对值的均值。

```python
    def loss_pha_laplacian(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = laplacian_loss(preds['phas'], targets['phas'])

        return dict(loss_pha_laplacian=loss)

    def unknown_l1_loss(self, sample_map, preds, targets):
        h,w = sample_map.shape[2:]
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            #sample_map for unknown area
            scale = sample_map.shape[0]*262144/torch.sum(sample_map)

        # scale = 1

        loss = F.l1_loss(preds['phas']*sample_map, targets['phas']*sample_map)*scale
        return dict(unknown_l1_loss=loss)

    def known_l1_loss(self, sample_map, preds, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1
        h,w = sample_map.shape[2:]
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0]*262144/torch.sum(new_sample_map)
        # scale = 1

        loss = F.l1_loss(preds['phas']*new_sample_map, targets['phas']*new_sample_map)*scale
        return dict(known_l1_loss=loss)

    def smooth_l1_loss(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = F.smooth_l1_loss(preds['phas'], targets['phas'])

        return dict(smooth_l1_loss=loss)

    def known_smooth_l1_loss(self, sample_map, preds, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1
        h,w = sample_map.shape[2:]
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0]*262144/torch.sum(new_sample_map)
        # scale = 1

        loss = F.smooth_l1_loss(preds['phas']*new_sample_map, targets['phas']*new_sample_map)*scale
        return dict(known_l1_loss=loss)
    
    def cross_entropy_loss(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = F.binary_cross_entropy_with_logits(preds['phas'], targets['phas'])

        return dict(cross_entropy_loss=loss)
```

## focal loss

```python
    def focal_loss(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = focal_loss.sigmoid_focal_loss(preds['phas'], targets['phas'], reduction='mean')

        return dict(focal_loss=loss)
```

[10分钟理解Focal loss数学原理与Pytorch代码（翻译）-腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1669261)

```python
    def forward_single_sample(self, sample_map, preds, targets):
        # check if targets only have element 0 and 1
        if torch.all(targets == 0) or torch.all(targets == 1):
            
            preds = {'phas': preds}
            targets = {'phas': targets}
            losses = dict()
            for k in self.losses_seg:
                if k=='unknown_l1_loss' or k=='known_l1_loss' or k=='loss_gradient_penalty' or k=='known_smooth_l1_loss':
                    losses.update(getattr(self, k)(sample_map, preds, targets))
                else:
                    losses.update(getattr(self, k)(preds, targets))
            return losses
        else:
            preds = {'phas': preds}
            targets = {'phas': targets}
            losses = dict()
            for k in self.losses_matting:
                if k=='unknown_l1_loss' or k=='known_l1_loss' or k=='loss_gradient_penalty' or k=='known_smooth_l1_loss':
                    losses.update(getattr(self, k)(sample_map, preds, targets))
                else:
                    losses.update(getattr(self, k)(preds, targets))
            return losses
        
    def forward(self, sample_map, preds, targets):
        losses = dict()
        for i in range(preds.shape[0]):
            losses_ = self.forward_single_sample(sample_map[i].unsqueeze(0), preds[i].unsqueeze(0), targets[i].unsqueeze(0))
            for k in losses_:
                if k in losses:
                    losses[k] += losses_[k]
                else:
                    losses[k] = losses_[k]
        return losses
#-----------------Laplacian Loss-------------------------#
def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels

def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid
```

## 高斯核

```python
def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel
```

这个函数定义了一个二维高斯核，用于图像处理中的平滑操作，通常用于卷积操作以实现图像的模糊或平滑效果。

这个高斯核是一个 5x5 的矩阵，其中心点的值最大，其它点的值逐渐减小。这种分布使得中心的像素点在卷积操作中所起的作用更大，而边缘处的像素点所起的作用相对较小，从而实现了图像的平滑效果。

这个函数首先创建了一个张量表示高斯核的值，并将其设定为指定的设备和数据类型。然后通过除以 256 来进行归一化处理，以确保所有值的总和为 1。最后，将高斯核的维度从 2D 扩展到 3D，以符合 PyTorch 中卷积操作的要求。

```python
def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img

def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]
```

## 下采样上采样

## 下采样和上采样
下采样（Downsampling）和上采样（Upsampling）是图像处理和计算机视觉领域中常用的技术，用于改变图像的分辨率或大小。

1. **下采样（Downsampling）**：
   - 下采样是指将图像的分辨率降低，通常是通过减少图像中像素的数量来实现的。
   - 在下采样过程中，通常会使用一些滤波技术，如平均池化或最大池化，来减少像素的数量。这些滤波器会根据一定的规则在图像上滑动，将每个滑动窗口内的像素值进行池化操作，从而得到一个更小的输出。
   - 下采样可以减少图像的尺寸，降低计算复杂度，同时也有助于提取图像的重要特征。

2. **上采样（Upsampling）**：
   - 上采样是指将图像的分辨率增加，通常是通过增加图像中像素的数量来实现的。
   - 在上采样过程中，通常会使用插值技术，如双线性插值或最近邻插值，来估算缺失的像素值。这些插值方法会根据已知像素的值来推断新像素的值，从而得到一个更大的输出。
   - 上采样可以用于图像的放大操作，使图像变得更清晰，也可以用于将低分辨率图像恢复到原始分辨率。

这两种技术经常在神经网络中使用，例如在卷积神经网络（CNN）的池化层中进行下采样，或者在转置卷积层（反卷积层）中进行上采样。它们在图像处理和计算机视觉任务中都起着重要的作用，如图像分类、目标检测、语义分割等。
