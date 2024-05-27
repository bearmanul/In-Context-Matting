
# 概述类和函数

#### OneWayAttentionBlock
- **OneWayAttentionBlock**: 实现单向注意力机制的块。
  - **__init__**: 初始化注意力模块和归一化层。
  - **forward**: 执行前向传播，依次处理输入查询和上下文，并返回处理后的输出。

#### compute_correspondence_matrix
- **compute_correspondence_matrix**: 计算源特征和参考特征之间的对应矩阵，通过余弦相似度进行比较。

#### maskpooling
- **maskpooling**: 对掩码进行池化操作，以降低其分辨率。

#### dilate
- **dilate**: 实现图像的膨胀操作。

#### erode
- **erode**: 实现图像的腐蚀操作，通过反向膨胀实现。

#### generate_trimap
- **generate_trimap**: 生成三值掩码，通过先腐蚀再膨胀实现。

#### calculate_attention_score_
- **calculate_attention_score_**: 计算掩码区域和非掩码区域的注意力得分。

#### refine_mask_by_attention
- **refine_mask_by_attention**: 通过迭代计算和注意力图细化输入掩码。

#### InContextCorrespondence
- **InContextCorrespondence**: 实现上下文相关的融合。
  - **__init__**: 初始化模块参数。
  - **forward**: 执行前向传播，计算特征对应矩阵，生成掩码并细化。

#### TrainingFreeAttention
- **TrainingFreeAttention**: 实现无训练的注意力机制。
  - **__init__**: 初始化模块参数。
  - **forward**: 执行前向传播，包括特征调整和注意力计算。
  - **resize_input_to_res**: 调整输入特征分辨率。
  - **get_roi_features**: 通过掩码池化获取感兴趣区域的特征。
  - **maskpool**: 执行掩码池化操作。
  - **compute_attention**: 计算特征和参考特征之间的注意力。
  - **compute_attention_single**: 计算单个特征和参考特征之间的注意力。
  - **reshape_attn_output**: 重塑注意力输出为指定尺寸。

#### TrainingCrossAttention
- **TrainingCrossAttention**: 实现带有交叉注意力机制的训练模块。
  - **__init__**: 初始化模块参数和单向注意力模块。
  - **forward**: 执行前向传播，包括特征调整、注意力计算和输出重塑。
  - **resize_input_to_res**: 调整输入特征分辨率。
  - **get_roi_features**: 通过掩码池化获取感兴趣区域的特征。
  - **maskpool**: 执行掩码池化操作。
  - **compute_attention**: 计算特征和参考特征之间的注意力。

### TrainingFreeAttentionBlocks 类

`TrainingFreeAttentionBlocks` 类实现了一种上下文融合的方法，主要用于图像特征的注意力机制计算。

#### `__init__` 方法
- 参数包括 `res_ratio`, `pool_type`, `temp_softmax`, `use_scale`, `upsample_mode`, `bottle_neck_dim`, `use_norm`。
- 初始化一个 `TrainingFreeAttention` 模块。

#### `forward` 方法
- 输入：`feature_of_reference_image` (参考图像特征), `ft_attn_of_source_image` (源图像特征), `guidance_on_reference_image` (参考图像指导)。
- 步骤：
  1. 获取源图像的特征和参考图像的特征。
  2. 将参考图像指导二值化。
  3. 使用 `attn_module` 计算注意力输出。
  4. 对注意力输出进行求和和处理。
  5. 计算无训练的自注意力输出，并调整大小。
  6. 返回包含 `trimap` (三分图), `feature` (特征), `mask` (掩码) 的字典。

#### `training_free_self_attention` 方法
- 输入：`x` (输入张量), `self_attn_maps` (自注意力图)。
- 步骤：
  1. 调整输入张量 `x` 的大小以匹配注意力图的空间维度。
  2. 将注意力图和输入特征图进行矩阵乘法，计算自注意力输出。
  3. 返回调整回原始空间维度的输出张量。

### SemiTrainingAttentionBlocks 类

`SemiTrainingAttentionBlocks` 类实现了一种上下文融合的方法，主要用于半训练的注意力机制计算。

#### `__init__` 方法
- 参数包括 `res_ratio`, `pool_type`, `upsample_mode`, `bottle_neck_dim`, `use_norm`, `in_ft_dim`, `in_attn_dim`, `attn_out_dim`, `ft_out_dim`, `training_cross_attn`。
- 根据是否使用交叉注意力初始化不同的注意力模块 (`TrainingCrossAttention` 或 `TrainingFreeAttention`)。
- 初始化注意力模块和特征模块的列表。
- 初始化多尺度特征融合模块。

#### `forward` 方法
- 输入：`feature_of_reference_image` (参考图像特征), `ft_attn_of_source_image` (源图像特征), `guidance_on_reference_image` (参考图像指导)。
- 步骤：
  1. 获取源图像的特征和参考图像的特征。
  2. 将参考图像指导二值化。
  3. 使用 `attn_module` 计算注意力输出。
  4. 计算无训练的自注意力输出，并调整大小。
  5. 将注意力输出和源图像的特征进行拼接和处理。
  6. 在多尺度融合块中前向传播。
  7. 调整 `self_attn_output` 的大小，并计算平均值。
  8. 返回包含 `trimap` (三分图), `feature` (特征), `mask` (掩码) 的字典。

#### `training_free_self_attention` 方法
- 输入：`x` (输入张量), `self_attn_maps` (自注意力图)。
- 步骤：
  1. 调整输入张量 `x` 的大小以匹配注意力图的空间维度。
  2. 对每个注意力图和输入张量进行逐元素乘法。
  3. 通过注意力模块进行处理。
  4. 返回处理后的张量。

### MultiScaleFeatureFusion 类

`MultiScaleFeatureFusion` 类使用多个卷积层或瓶颈块来压缩和融合特征维度。

#### `__init__` 方法
- 参数包括 `in_feature_dim`, `out_feature_dim`, `use_bottleneck`。
- 初始化模块列表，每个模块为一个融合块 (`Fusion_Block`)。

#### `forward` 方法
- 输入：`features` (特征字典)。
- 步骤：
  1. 获取特征字典中的特征。
  2. 通过每个融合块前向传播特征。
  3. 返回最终融合的特征。

# 腐蚀、膨胀
[计算机视觉（一）——形态学操作：腐蚀、膨胀、开闭运算、形态学梯度、顶帽与黑帽_形态学 腐蚀和膨胀-CSDN博客](https://blog.csdn.net/qq_41433002/article/details/115266567)

# 代码注释
```python
# 引入必要的库
from einops import rearrange  # 用于重排张量
from torch import einsum  # 用于高效的爱因斯坦求和
import torch  # PyTorch库
import torch.nn as nn  # PyTorch中的神经网络模块
from torch.nn import functional as F  # PyTorch中的功能性操作

# 引入自定义模块
from icm.models.decoder.bottleneck_block import BottleneckBlock
from icm.models.decoder.detail_capture import Basic_Conv3x3, Basic_Conv3x3_attn, Fusion_Block
import math  # 数学库
from icm.models.decoder.attention import Attention, MLPBlock

# 定义一个单向注意力块类
class OneWayAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,  # 输入特征的维度
        n_heads,  # 注意力头的数量
        d_head,  # 每个头的维度
        mlp_dim_rate,  # MLP扩展率
    ):
        super().__init__()

        # 初始化注意力模块
        self.attn = Attention(dim, n_heads, downsample_rate=dim//d_head)
        self.norm1 = nn.LayerNorm(dim)  # 初始化第一个归一化层
        self.mlp = MLPBlock(dim, int(dim*mlp_dim_rate))  # 初始化MLP模块
        self.norm2 = nn.LayerNorm(dim)  # 初始化第二个归一化层

    # 前向传播函数
    def forward(self, q, context_all):
        output = []  # 用于存储输出的列表
        for i in range(len(q)):
            x = q[i].unsqueeze(0)  # 添加批次维度
            context = context_all[i].unsqueeze(0)  # 添加批次维度
            x = self.norm1(x)  # 对x进行归一化
            context = self.norm1(context)  # 对context进行归一化
            x = self.attn(q=x, k=context, v=context) + x  # 计算注意力并添加残差连接
            x = self.norm2(x)  # 对x进行第二次归一化
            x = self.mlp(x) + x  # 通过MLP并添加残差连接
            output.append(x.squeeze(0))  # 移除批次维度并添加到输出列表

        return output  # 返回输出
```
## compute_correspondence_matrix
```python
# 计算对应矩阵的函数
def compute_correspondence_matrix(source_feature, ref_feature):
    """
    计算源特征和参考特征之间的对应矩阵
    Args:
        source_feature: [B, C, H, W] 源特征
        ref_feature: [B, C, H, W] 参考特征
    Returns:
        correspondence_matrix: [B, H*W, H*W] 对应矩阵
    """
    # 将[B, C, H, W]转换为[B, H, W, C]
    source_feature = source_feature.permute(0, 2, 3, 1)
    ref_feature = ref_feature.permute(0, 2, 3, 1)

    # 将[B, H, W, C]转换为[B, H*W, C]
    source_feature = torch.reshape(source_feature, [source_feature.shape[0], -1, source_feature.shape[-1]])
    ref_feature = torch.reshape(ref_feature, [ref_feature.shape[0], -1, ref_feature.shape[-1]])

    # 对特征进行归一化
    source_feature = F.normalize(source_feature, p=2, dim=-1)
    ref_feature = F.normalize(ref_feature, p=2, dim=-1)

    # 计算余弦相似度
    cos_sim = torch.matmul(source_feature, ref_feature.transpose(1, 2))  # [B, H*W, H*W]

    return cos_sim  # 返回余弦相似度矩阵
```
### 掩码一正一反的作用
在计算注意力得分时，掩码（mask）被用于指定需要关注的区域和需要忽略的区域。具体来说：

1. **正掩码（mask_pos）**：
   - 掩码 `mask` 被重复以匹配注意力图的维度，从而创建 `mask_pos`。
   - `mask_pos` 用于选中注意力图中的某些区域，这些区域是我们感兴趣的（即掩码值为1的区域）。

2. **反掩码（mask_neg）**：
   - `mask_neg` 是掩码的取反，即 `1 - mask_pos`。
   - `mask_neg` 用于选中注意力图中的其他区域，这些区域是不感兴趣的（即掩码值为0的区域）。

通过分别计算正掩码和反掩码区域的注意力得分，我们可以比较两者，进而判断模型对不同区域的关注度。

### 注意力得分的计算
注意力得分的计算分为以下几步：

1. **获取掩码和注意力图的维度**：
   ```python
   B, H, W = mask.shape
   ```

2. **创建正掩码并计算正掩码区域的注意力得分**：
   ```python
   mask_pos = mask.repeat(1, attention_map.shape[1], 1, 1)
   score_pos = torch.sum(attention_map * mask_pos, dim=(2, 3))
   score_pos = score_pos / torch.sum(mask_pos, dim=(2, 3))
   ```
   - `mask_pos` 将 `mask` 的维度重复，使其与 `attention_map` 匹配。
   - `score_pos` 计算正掩码区域的注意力得分，通过逐元素相乘和求和实现。
   - 最后，`score_pos` 通过正掩码的和进行归一化。

3. **创建反掩码并计算反掩码区域的注意力得分**：
   ```python
   mask_neg = 1 - mask_pos
   score_neg = torch.sum(attention_map * mask_neg, dim=(2, 3))
   score_neg = score_neg / torch.sum(mask_neg, dim=(2, 3))
   ```
   - `mask_neg` 是 `mask_pos` 的取反。
   - `score_neg` 计算反掩码区域的注意力得分。
   - 最后，`score_neg` 通过反掩码的和进行归一化。

### 不同的得分类型
根据 `score_type` 的不同，计算注意力得分的方法也有所不同：

1. **Classification**:
   - 分类得分模式下，比较正掩码和反掩码区域的得分。
   - 如果正掩码区域的得分高于反掩码区域的得分，则该区域得分为1，否则为0。
   ```python
   if score_type == 'classification':
       score = torch.zeros_like(score_pos)
       score[score_pos > score_neg] = 1
   ```

2. **Softmax**:
   - 在 softmax 模式下，应该使用 softmax 函数对正掩码和反掩码区域的得分进行归一化处理。这段代码未实现 softmax 模式，但通常情况下，softmax 会应用于正掩码和反掩码的分数来计算最终的得分。

3. **Ratio**:
   - 在 ratio 模式下，通常会计算正掩码区域得分与反掩码区域得分的比率。这段代码同样未实现 ratio 模式，但通常情况下，会计算 `score_pos / score_neg` 来得到比率得分。

### 总结
- **掩码一正一反**：通过正掩码和反掩码，可以区分模型在感兴趣区域和不感兴趣区域的关注度。
- **得分类型**：不同的得分类型（classification, softmax, ratio）用于不同的场景，classification 实现了简单的区域比较，其他类型可以实现更复杂的关注度计算。
- **注意力得分**：通过计算和比较正掩码和反掩码区域的得分，可以评估模型的注意力集中情况。


```python
# 掩码池化函数
def maskpooling(mask, res):
    '''
    掩码池化以降低掩码的分辨率
    Input:
    mask: [B, 1, H, W] 输入掩码
    res: 分辨率
    Output: [B, 1, res, res] 池化后的掩码
    '''
    mask[mask > 0] = 1  # 将掩码中的所有正值设为1
    mask = -1 * mask  # 掩码取反
    kernel_size = mask.shape[2] // res  # 计算池化的核大小
    mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=kernel_size, padding=0)  # 最大池化
    mask = -1 * mask  # 取反回去
    return mask  # 返回池化后的掩码

# 膨胀操作函数
def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2  # 计算填充大小
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')  # 反射填充
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)  # 最大池化实现膨胀
    return out  # 返回膨胀后的图像

# 腐蚀操作函数
def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)  # 腐蚀通过反向膨胀实现
    return out  # 返回腐蚀后的图像

# 生成三值掩码函数
def generate_trimap(mask, erode_kernel_size=10, dilate_kernel_size=10):
    eroded = erode(mask, erode_kernel_size)  # 先腐蚀
    dilated = dilate(mask, dilate_kernel_size)  # 再膨胀
    trimap = torch.zeros_like(mask)  # 创建一个与掩码大小相同的全零张量
    trimap[dilated == 1] = 0.5  # 膨胀区域设为0.5
    trimap[eroded == 1] = 1  # 腐蚀区域设为1
    return trimap  # 返回三值掩码

# 计算注意力得分函数
def calculate_attention_score_(mask, attention_map, score_type):
    '''
    计算注意力图的注意力得分
    mask: [B, H, W] 掩码，值为0或1
    attention_map: [B, H*W, H, W] 注意力图
    '''
    B, H, W = mask.shape  # 获取掩码的维度
    mask_pos = mask.repeat(1, attention_map.shape[1], 1, 1)  # 重复掩码以匹配注意力图的维度 [B, H*W, H, W]
    score_pos = torch.sum(attention_map * mask_pos, dim=(2, 3))  # 计算掩码位置的注意力得分 [B, H*W]
    score_pos = score_pos / torch.sum(mask_pos, dim=(2, 3))  # 归一化注意力得分 [B, H*W]

    mask_neg = 1 - mask_pos  # 计算掩码取反后的区域
    score_neg = torch.sum(attention_map * mask_neg, dim=(2, 3))  # 计算掩码取反位置的注意力得分
    score_neg = score_neg / torch.sum(mask_neg, dim=(2, 3))

    assert score_type in ['classification', 'softmax', 'ratio']  # 确认得分类型

    if score_type == 'classification':
        score = torch.zeros_like(score_pos)  # 初始化得分张量
        score[score_pos > score_neg] = 1  # 分类得分

    return score.reshape(B, H, W)  # 返回重塑后的得分张量

# 通过注意力细化掩码函数
def refine_mask_by_attention(mask, attention_maps, iterations=10, score_type='classification'):
    '''
    通过注意力图细化掩码
    mask: [B, H, W] 输入掩码
    attention_maps: [B, H*W, H, W] 注意力图
    '''
    assert mask.shape[1] == attention_maps.shape[2]  # 确保掩码和注意力图尺寸匹配
    for i in range(iterations):
        score = calculate_attention_score_(mask, attention_maps, score_type=score_type)  # 计算得分

        if score.equal(mask):  # 如果得分和掩码相等，停止迭代
            break
        else:
            mask = score  # 否则更新掩码

    assert i != iterations - 1  # 确保没有达到最大迭代次数
    return mask  # 返回
class InContextCorrespondence(nn.Module):
    '''
    上下文相关融合的一种实现

    forward(feature_of_reference_image, feature_of_source_image, guidance_on_reference_image)
    '''

    def __init__(self,
                 use_bottle_neck=False,
                 in_dim=1280,
                 bottle_neck_dim=512,
                 refine_with_attention=False,
                 ):
        super().__init__()
        self.use_bottle_neck = use_bottle_neck
        self.refine_with_attention = refine_with_attention

    def forward(self, feature_of_reference_image, ft_attn_of_source_image, guidance_on_reference_image):
        '''
        feature_of_reference_image: [B, C, H, W] 参考图像的特征
        ft_attn_of_source_image: {"ft": [B, C, H, W], "attn": [B, H_1, W_1, H_1*W_1]} 源图像的特征及其注意力图
        guidance_on_reference_image: [B, 1, H_2, W_2] 参考图像的引导信息
        '''

        # 获取参考图像的高度和宽度
        h, w = guidance_on_reference_image.shape[-2:]
        h_attn, w_attn = ft_attn_of_source_image['attn'].shape[-3:-1]

        feature_of_source_image = ft_attn_of_source_image['ft']  # 源图像的特征
        attention_map_of_source_image = ft_attn_of_source_image['attn']  # 源图像的注意力图

        cos_sim = compute_correspondence_matrix(
            feature_of_source_image, feature_of_reference_image)  # 计算特征之间的对应矩阵

        index = torch.argmax(cos_sim, dim=-1)  # 获取对应矩阵每行的最大值的索引

        mask_ref = maskpooling(guidance_on_reference_image,
                               h_attn)  # 对引导信息进行掩码池化

        mask_ref = mask_ref.reshape(mask_ref.shape[0], -1)  # 重塑掩码

        new_index = torch.gather(mask_ref, 1, index)  # 根据掩码获取新的索引
        res = int(new_index.shape[-1]**0.5)
        new_index = new_index.reshape(
            new_index.shape[0], res, res).unsqueeze(1)  # 重塑新索引

        mask_result = new_index  # 获取最终掩码结果

        if self.refine_with_attention:  # 如果需要使用注意力进行细化
            mask_result = refine_mask_by_attention(
                mask_result, attention_map_of_source_image, iterations=10, score_type='classification')  # 使用注意力细化掩码

        mask_result = F.interpolate(
            mask_result.float(), size=(h, w), mode='bilinear')  # 双线性插值得到最终掩码

        pesudo_trimap = generate_trimap(
            mask_result, erode_kernel_size=self.kernel_size, dilate_kernel_size=self.kernel_size)  # 生成伪三值掩码

        output = {}  # 输出字典
        output['trimap'] = pesudo_trimap  # 伪三值掩码
        output['feature'] = feature_of_source_image  # 源图像的特征
        output['mask'] = mask_result  # 掩码结果

        return output  # 返回输出字典


class TrainingFreeAttention(nn.Module):
    def __init__(self, res_ratio=4, pool_type='average', temp_softmax=1, use_scale=False, upsample_mode='bilinear', use_norm=False) -> None:
        super().__init__()
        self.res_ratio = res_ratio
        self.pool_type = pool_type
        self.temp_softmax = temp_softmax
        self.use_scale = use_scale
        self.upsample_mode = upsample_mode
        if use_norm:
            self.norm = nn.LayerNorm(use_norm, elementwise_affine=True)
        else:
            self.idt = nn.Identity()

    def forward(self, features, features_ref, roi_mask,):
        # roi_mask: [B, 1, H, W]
        # features: [B, C, h, w]
        # features_ref: [B, C, h, w]
        B, _, H, W = roi_mask.shape
        if self.res_ratio == None:
            H_attn, W_attn = features.shape[2], features.shape[3]
        else:
            H_attn = H//self.res_ratio
            W_attn = W//self.res_ratio
            features, features_ref = self.resize_input_to_res(
                features, features_ref, (H, W))  # [H//res_ratio, W//res_ratio]

        # List, len = B, each element: [C_q, dim], dim = H//res_ratio * W//res_ratio
        features_ref = self.get_roi_features(features_ref, roi_mask)

        features = features.reshape(
            B, -1, features.shape[2] * features.shape[3]).permute(0, 2, 1)  # [B, C, dim]
        # List, len = B, each element: [C_q, C]
        attn_output = self.compute_attention(features, features_ref)

        # List, len = B, each element: [C_q, H_attn, W_attn]
        attn_output = self.reshape_attn_output(attn_output, (H_attn, W_attn))

        return attn_output

    def resize_input_to_res(self, features, features_ref, size):
        # features: [B, C, h, w]
        # features_ref: [B, C, h, w]
        H, W = size
        target_H,
        target_W = H // self.res_ratio, W // self.res_ratio
        features = F.interpolate(features, size=(target_H, target_W), mode=self.upsample_mode)
        features_ref = F.interpolate(features_ref, size=(target_H, target_W), mode=self.upsample_mode)
        return features, features_ref

    def get_roi_features(self, feature, mask):
        '''
        通过掩码池化获取特征令牌
        feature: [B, C, h, w]
        mask: [B, 1, H, W]  [0,1]
        返回值: 列表，长度为B，每个元素为：[token_num, C]
        '''

        # 确保掩码只有元素0和1
        assert torch.all(torch.logical_or(mask == 0, mask == 1))
        # assert mask.max() == 1 and mask.min() == 0

        B, _, H, W = mask.shape
        h, w = feature.shape[2:]

        output = []
        for i in range(B):
            mask_ = mask[i]
            feature_ = feature[i]
            feature_ = self.maskpool(feature_, mask_)
            output.append(feature_)
        return output

    def maskpool(self, feature, mask):
        '''
        通过掩码池化获取特征令牌
        feature: [C, h, w]
        mask: [1, H, W]  [0,1]
        返回值: [token_num, C]
        '''
        kernel_size = mask.shape[1] // feature.shape[1] if self.res_ratio == None else self.res_ratio
        if self.pool_type == 'max':
            mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=kernel_size, padding=0)
        elif self.pool_type == 'average':
            mask = F.avg_pool2d(mask, kernel_size=kernel_size, stride=kernel_size, padding=0)
        elif self.pool_type == 'min':
            mask = -1 * mask
            mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=kernel_size, padding=0)
            mask = -1 * mask
        else:
            raise NotImplementedError

        # 逐元素乘法mask和feature
        feature = feature * mask

        index = (mask > 0).reshape(1, -1).squeeze()
        feature = feature.reshape(feature.shape[0], -1).permute(1, 0)

        feature = feature[index]
        return feature

    def compute_attention(self, features, features_ref):
        '''
        features: [B, C, dim]
        features_ref: 列表，长度为B，每个元素为：[C_q, dim]
        返回值: 列表，长度为B，每个元素为：[C_q, C]
        '''
        output = []
        for i in range(len(features_ref)):
            feature_ref = features_ref[i]
            feature = features[i]
            feature = self.compute_attention_single(feature, feature_ref)
            output.append(feature)
        return output

    def compute_attention_single(self, feature, feature_ref):
        '''
        使用softmax计算注意力
        feature: [C, dim]
        feature_ref: [C_q, dim]
        返回值: [C_q, C]
        '''
        scale = feature.shape[-1] ** -0.5 if self.use_scale else 1.0
        feature = self.norm(feature) if hasattr(self, 'norm') else feature
        feature_ref = self.norm(feature_ref) if hasattr(self, 'norm') else feature_ref
        sim = einsum('i d, j d -> i j', feature_ref, feature) * scale
        sim = sim / self.temp_softmax
        sim = sim.softmax(dim=-1)
        return sim

    def reshape_attn_output(self, attn_output, attn_size):
        '''
        attn_output: 列表，长度为B，每个元素为：[C_q, C]
        返回值: 列表，长度为B，每个元素为：[C_q, H_attn, W_attn]
        '''
        # attn_output[0].shape[1] 求平方根得到 H_attn, W_attn
        H_attn, W_attn = attn_size

        output = []
        for i in range(len(attn_output)):
            attn_output_ = attn_output[i]
            attn_output_ = attn_output_.reshape(attn_output_.shape[0], H_attn, W_attn)
            output.append(attn_output_)
        return output
class TrainingCrossAttention(nn.Module):
    def __init__(self, res_ratio=4, pool_type='average', temp_softmax=1, use_scale=False, upsample_mode='bilinear', use_norm=False, dim=1280,
                 n_heads=4,
                 d_head=320,
                 mlp_dim_rate=0.5,) -> None:
        super().__init__()
        self.res_ratio = res_ratio  # 分辨率缩放比例
        self.pool_type = pool_type  # 池化类型
        self.temp_softmax = temp_softmax  # softmax温度参数
        self.use_scale = use_scale  # 是否使用缩放参数
        self.upsample_mode = upsample_mode  # 上采样模式
        if use_norm:
            self.norm = nn.LayerNorm(use_norm, elementwise_affine=True)
        else:
            self.idt = nn.Identity()

        # 定义一个单向注意力模块
        self.attn_module = OneWayAttentionBlock(
            dim,
            n_heads,
            d_head,
            mlp_dim_rate,
        )

    def forward(self, features, features_ref, roi_mask,):
        # roi_mask: [B, 1, H, W]
        # features: [B, C, h, w]
        # features_ref: [B, C, h, w]
        B, _, H, W = roi_mask.shape
        if self.res_ratio == None:
            H_attn, W_attn = features.shape[2], features.shape[3]
        else:
            H_attn = H // self.res_ratio
            W_attn = W // self.res_ratio
            features, features_ref = self.resize_input_to_res(
                features, features_ref, (H, W))  # [H//res_ratio, W//res_ratio]

        # 列表，长度为B，每个元素为：[C_q, dim]，dim = H//res_ratio * W//res_ratio
        features_ref = self.get_roi_features(features_ref, roi_mask)

        features = features.reshape(
            B, -1, features.shape[2] * features.shape[3]).permute(0, 2, 1)  # [B, C, dim]
        # 列表，长度为B，每个元素为：[C_q, C]
        features_ref = self.attn_module(features_ref, features)

        attn_output = self.compute_attention(features, features_ref)

        # 列表，长度为B，每个元素为：[C_q, H_attn, W_attn]
        attn_output = self.reshape_attn_output(attn_output, (H_attn, W_attn))

        return attn_output

    def resize_input_to_res(self, features, features_ref, size):
        # features: [B, C, h, w]
        # features_ref: [B, C, h, w]
        H, W = size
        target_H, target_W = H // self.res_ratio, W // self.res_ratio
        features = F.interpolate(features, size=(
            target_H, target_W), mode=self.upsample_mode)
        features_ref = F.interpolate(features_ref, size=(
            target_H, target_W), mode=self.upsample_mode)
        return features, features_ref

    def get_roi_features(self, feature, mask):
        '''
        通过掩码池化获取特征令牌
        feature: [B, C, h, w]
        mask: [B, 1, H, W]  [0,1]
        返回值: 列表，长度为B，每个元素为：[token_num, C]
        '''

        # 确保掩码只有元素0和1
        assert torch.all(torch.logical_or(mask == 0, mask == 1))
        # assert mask.max() == 1 and mask.min() == 0

        B, _, H, W = mask.shape
        h, w = feature.shape[2:]

        output = []
        for i in range(B):
            mask_ = mask[i]
            feature_ = feature[i]
            feature_ = self.maskpool(feature_, mask_)
            output.append(feature_)
        return output

    def maskpool(self, feature, mask):
        '''
        通过掩码池化获取特征令牌
        feature: [C, h, w]
        mask: [1, H, W]  [0,1]
        返回值: [token_num, C]
        '''
        kernel_size = mask.shape[1] // feature.shape[1] if self.res_ratio == None else self.res_ratio
        if self.pool_type == 'max':
            mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=kernel_size, padding=0)
        elif self.pool_type == 'average':
            mask = F.avg_pool2d(mask, kernel_size=kernel_size, stride=kernel_size, padding=0)
        elif self.pool_type == 'min':
            mask = -1 * mask
            mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=kernel_size, padding=0)
            mask = -1 * mask
        else:
            raise NotImplementedError

        # 逐元素乘法mask和feature
        feature = feature * mask

        index = (mask > 0).reshape(1, -1).squeeze()
        feature = feature.reshape(feature.shape[0], -1).permute(1, 0)

        feature = feature[index]
        return feature

    def compute_attention(self, features, features_ref):
        '''
        features: [B, C, dim]
        features_ref: 列表，长度为B，每个元素为：[C_q, dim]
        返回值: 列表，长度为B，每个元素为：[C_q, C]
        '''
        output = []
        for i in range(len(features_ref)):
            feature_ref = features_ref[i]
            feature = features[i]
            feature = self.compute_attention_single(feature, feature_ref)
            output.append(feature)
        return output
```

## einsum

```python
    def compute_attention_single(self, feature, feature_ref):
        '''
        使用softmax计算注意力
        feature: [C, dim]
        feature_ref: [C_q, dim]
        返回值: [C_q, C]
        '''
        scale = feature.shape[-1] ** -0.5 if self.use_scale else 1.0
        feature = self.norm(feature) if hasattr(self, 'norm') else feature
        feature_ref = self.norm(feature_ref) if hasattr(self, 'norm') else feature_ref
        sim = einsum('i d, j d -> i j', feature_ref, feature) * scale
        sim = sim / self.temp_softmax
        sim = sim.softmax(dim=-1)
        return sim
```

[einsum is all you needed！ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/650247173)

**计算相似度**：
`sim = einsum('i d, j d -> i j', feature_ref, feature) * scale`

- 使用 `einsum('i d, j d -> i j', feature_ref, feature)` 计算两个特征矩阵的点积相似度。
    - `'i d, j d -> i j'` 表示对两个矩阵进行逐元素乘积并对 `d` 维求和，结果为一个大小为 `[C_q, C]` 的矩阵。
    - 这个操作相当于对两个矩阵的每个特征向量计算内积，从而得到它们之间的相似度。
- 乘以缩放因子 `scale`

```python
    def reshape_attn_output(self, attn_output, attn_size):
        '''
        attn_output: 列表，长度为B，每个元素为：[C_q, C]
        返回值: 列表，
长度为B，每个元素为：[C_q, H_attn, W_attn]
        '''
        # attn_output[0].shape[1] sqrt to get H_attn, W_attn
        H_attn, W_attn = attn_size

        output = []
        for i in range(len(attn_output)):
            attn_output_ = attn_output[i]
            attn_output_ = attn_output_.reshape(
                attn_output_.shape[0], H_attn, W_attn)
            output.append(attn_output_)
        return output
class TrainingFreeAttentionBlocks(nn.Module):
    '''
    in_context_fusion的一种实现

    forward(参考图像特征, 源图像特征, 参考图像指导)
    '''

    def __init__(self,
                 res_ratio=8,
                 pool_type='min',
                 temp_softmax=1000,
                 use_scale=False,
                 upsample_mode='bicubic',
                 bottle_neck_dim=None,
                 use_norm=False,
                 ):
        super().__init__()

        self.attn_module = TrainingFreeAttention(res_ratio=res_ratio,
                                                 pool_type=pool_type,
                                                 temp_softmax=temp_softmax,
                                                 use_scale=use_scale,
                                                 upsample_mode=upsample_mode,
                                                 use_norm=use_norm,)

    def forward(self, feature_of_reference_image, ft_attn_of_source_image, guidance_on_reference_image):
        '''
        feature_of_reference_image: [B, C, H, W]
        ft_attn_of_source_image: {"ft_cor": [B, C, H, W], "attn": {'24':[B, H_1, W_1, H_1*W_1],} "ft_matting": [B, C, H, W]}
        guidance_on_reference_image: [B, 1, H_2, W_2]
        '''
        # assert feature_of_reference_image.shape[0] == 1
        # 获取源图像的高度和宽度
        h, w = guidance_on_reference_image.shape[-2:]

        features_cor = ft_attn_of_source_image['ft_cor']
        features_matting = ft_attn_of_source_image['ft_matting']
        features_ref = feature_of_reference_image

        # 将参考图像指导阈值化为0或1
        guidance_on_reference_image[guidance_on_reference_image > 0.5] = 1
        guidance_on_reference_image[guidance_on_reference_image <= 0.5] = 0
        attn_output = self.attn_module(
            features_cor, features_ref, guidance_on_reference_image)

        attn_output = [attn_output_.sum(dim=0).unsqueeze(
            0).unsqueeze(0) for attn_output_ in attn_output]
        attn_output = torch.cat(attn_output, dim=0)

        self_attn_output = self.training_free_self_attention(
            attn_output, ft_attn_of_source_image['attn'])

        # 调整大小
        self_attn_output = F.interpolate(
            self_attn_output, size=(h, w), mode='bilinear')

        output = {}
        output['trimap'] = self_attn_output
        output['feature'] = features_matting
        output['mask'] = attn_output

        return output

    def training_free_self_attention(self, x, self_attn_maps):
        '''
        使用注意力图计算自注意力。

        参数：
        x（torch.Tensor）：输入张量。形状：[B, 1, H, W]
        self_attn_maps（torch.Tensor）：注意力图。形状：{'24': [B, H1, W1, H1*W1]}

        返回：
        torch.Tensor：自注意力计算结果。
        '''

        # 原始 x 的尺寸
        # 基于你的注释，假设 x 的形状为 [B, 1, H, W]
        B, _, H, W = x.shape

        # 注意力图的维度
        assert len(self_attn_maps) == 1
        # 获取字典中的唯一值
        self_attn_maps = list(self_attn_maps.values())[0]
        _, H1, W1, _ = self_attn_maps.shape

        # 调整大小以匹配注意力图的空间维度
        # 根据你的 PyTorch 版本，你可能需要 align_corners
        x = F.interpolate(x, size=(H1, W1), mode='bilinear',
                          align_corners=True)

        # 为矩阵相乘重塑注意力图和 x
        # 从 [B, H1, W1, H1*W1] 重塑为 [B, H1*W1, H1*W1]
        self_attn_maps = self_attn_maps.view(B, H1 * W1, H1 * W1)
        # 从 [B, 1, H1, W1] 重塑为 [B, 1, H1*W1]
        x = x.view(B, 1, H1 * W1)

        # 应用自注意力机制
        # 注意力图和输入特征图的矩阵乘法
        # 这一步本质上计算了输入中特征向量的加权和，
        # 其中权重由注意力图定义。
        # 与转置相乘以得到形状 [B, 1, H1*W1]
        out = torch.matmul(x, self_attn_maps.transpose(1, 2))

        # 将输出张量重塑为原始空间维度
        out = out.view(B, 1, H1, W1)  # 重塑回空间维度

        return out



class SemiTrainingAttentionBlocks(nn.Module):
    '''
    in_context_fusion的一种实现

    forward(参考图像特征, 源图像特征, 参考图像指导)
    '''

    def __init__(self,
                 res_ratio=8,
                 pool_type='min',
                 upsample_mode='bicubic',
                 bottle_neck_dim=None,
                 use_norm=False,
                 in_ft_dim=[1280, 960],
                 in_attn_dim=[24**2, 48**2],
                 attn_out_dim=256,
                 ft_out_dim=512,
                 training_cross_attn=False,
                 ):
        super().__init__()
        if training_cross_attn:
            self.attn_module = TrainingCrossAttention(
                res_ratio=res_ratio,
                pool_type=pool_type,
                temp_softmax=1,
                use_scale=True,
                upsample_mode=upsample_mode,
                use_norm=use_norm,
            )
        else:
            self.attn_module = TrainingFreeAttention(res_ratio=res_ratio,
                                                     pool_type=pool_type,
                                                     temp_softmax=1,
                                                     use_scale=True,
                                                     upsample_mode=upsample_mode,
                                                     use_norm=use_norm)
        self.attn_module_list = nn.ModuleList()
        self.ft_attn_module_list = nn.ModuleList()
        # 初始化注意力模块列表和特征模块列表
        for i in range(len(in_attn_dim)):
            # 添加基本的 3*3 卷积注意力模块
            self.attn_module_list.append(Basic_Conv3x3_attn(
                in_attn_dim[i], attn_out_dim, int(math.sqrt(in_attn_dim[i]))))
            # 添加基本的 3*3 卷积模块
            self.ft_attn_module_list.append(Basic_Conv3x3(
                ft_out_dim[i] + attn_out_dim, ft_out_dim[i]))
        # 初始化特征模块列表
        self.ft_module_list = nn.ModuleList()
        for i in range(len(in_ft_dim)):
            # 添加基本的 3*3 卷积模块
            self.ft_module_list.append(
                Basic_Conv3x3(in_ft_dim[i], ft_out_dim[i]))

        ft_out_dim_ = [2*d for d in ft_out_dim]
        # 初始化特征融合模块
        self.fusion = MultiScaleFeatureFusion(ft_out_dim_, ft_out_dim)

    def forward(self, feature_of_reference_image, ft_attn_of_source_image, guidance_on_reference_image):
        '''
        feature_of_reference_image: [B, C, H, W]  # 参考图像特征
        ft_attn_of_source_image: {"ft_cor": [B, C, H, W], "attn": [B, H_1, W_1, H_1*W_1], "ft_matting": {'24':[B, C, H, W]} }  # 源图像特征
        guidance_on_reference_image: [B, 1, H_2, W_2]  # 参考图像指导
        '''
        # assert feature_of_reference_image.shape[0] == 1
        # 获取源图像的高度和宽度
        h, w = guidance_on_reference_image.shape[-2:]

        features_cor = ft_attn_of_source_image['ft_cor']
        features_matting = ft_attn_of_source_image['ft_matting']
        features_ref = feature_of_reference_image

        # 将参考图像指导阈值化为0或1
        guidance_on_reference_image[guidance_on_reference_image > 0.5] = 1
        guidance_on_reference_image[guidance_on_reference_image <= 0.5] = 0
        attn_output = self.attn_module(
            features_cor, features_ref, guidance_on_reference_image)

        attn_output = [attn_output_.sum(dim=0).unsqueeze(
            0).unsqueeze(0) for attn_output_ in attn_output]
        attn_output = torch.cat(attn_output, dim=0)

        self_attn_output = self.training_free_self_attention(
            attn_output, ft_attn_of_source_image['attn'])

        # concat attn and ft_matting
        attn_ft_matting = {}
        for i, key in enumerate(features_matting.keys()):
            if key in self_attn_output.keys():
                features_matting[key] = self.ft_module_list[i](
                    features_matting[key])
                attn_ft_matting[key] = torch.cat(
                    [features_matting[key], self_attn_output[key]], dim=1)

                attn_ft_matting[key] = self.ft_attn_module_list[i](
                    attn_ft_matting[key])

            else:
                attn_ft_matting[key] = self.ft_module_list[i](
                    features_matting[key])

        # 在多尺度融合块中前向传播
        attn_ft_matting = self.fusion(attn_ft_matting)

        att_look = []
        # 调整大小并平均 self_attn_output
        for i, key in enumerate(self_attn_output.keys()):
            att__ = F.interpolate(
                self_attn_output[key].mean(dim=1).unsqueeze(1), size=(h, w), mode='bilinear')
            att_look.append(att__)
        att_look = torch.cat(att_look, dim=1)
        att_look = att_look.mean(dim=1).unsqueeze(1)

        output = {}

        output['trimap'] = att_look
        output['feature'] = attn_ft_matting
        output['mask'] = attn_output

        return output

    def training_free_self_attention(self, x, self_attn_maps):
        '''
        计算使用注意力图的加权 attn maps。

        参数：
        x（torch.Tensor）：输入张量。形状：[B, 1, H, W]
        self_attn_maps（torch.Tensor）：注意力图。形状：{'24':[B, H1, W1, H1*W1], '48':[B, H2, W2, H2*W2]}

        返回：
        torch.Tensor：注意力计算结果。{'24':[B, 1, H1*W1, H1, W1], '48':[B, 1, H2*W2, H2, W2]}
        '''

        # 原始 x 的尺寸
        # 基于你的注释，假设 x 的形状为 [B, 1, H, W]
        B, _, H, W = x.shape
        out = {}
        for i, key in enumerate(self_attn_maps.keys()):
            # 注意力图的维度
            _, H1, W1, _ = self_attn_maps[key].shape

            # 调整大小以匹配注意力图的空间维度
            # 根据你的 PyTorch 版本，你可能需要 align_corners
            x_ = F.interpolate(x, size=(H1, W1), mode='bilinear',
                               align_corners=True)

            # 为矩阵相乘重塑注意力图和 x
            # 从 [B, H1, W1, H1*W1] 重塑为 [B, H1*W1, H1*W1]
            self_attn_map_ = self_attn_maps[key].view(
                B, H1 * W1, H1 * W1).transpose(1, 2)
            # 从 [B, 1, H1, W1] 重塑为 [B, 1, H1*W1]
            x_ = x_.reshape(B, H1 * W1, 1)

            # 传播，逐元素乘法 x_ 和 self_attn_maps
            x_ = x_ * self_attn_map_
            x_ = x_.reshape(B, H1 * W1, H1, W1)
            x_ = x_.permute(0, 2, 3, 1)
            # 通过 3*3 基本卷积注意力模块传播
            x_ = self.attn_module_list[i](x_)
            out[key] = x_

        return out


class MultiScaleFeatureFusion(nn.Module):
    '''
    使用 N 个卷积层或瓶颈块来压缩特征维度

    使用 M 个卷积层和上采样来融合特征

    '''

    def __init__(self,
                 in_feature_dim=[],  # 输入特征维度
                 out_feature_dim=[],  # 输出特征维度
                 use_bottleneck=False) -> None:  # 是否使用瓶颈块
        super().__init__()
        assert len(in_feature_dim) == len(out_feature_dim)
        # 初始化模块列表
        self.module_list = nn.ModuleList()
        for i in range(len(in_feature_dim)-1):
            # 添加融合块
            self.module_list.append(Fusion_Block(
                in_feature_dim[i], out_feature_dim[i]))

    def forward(self, features):
        # features: {'32': tensor, '16': tensor, '8': tensor}

        key_list = list(features.keys())
        ft = features[key_list[0]]
        for i in range(len(key_list)-1):
            # 通过融合块前向传播
            ft = self.module_list[i](ft, features[key_list[i+1]])

        return ft

```