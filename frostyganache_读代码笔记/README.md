我在读代码过程中做的笔记，笔记文件是按照原来代码的结构组织的。我把标题当书签用了，可以用右边的Outline快速跳转。

很多是用GPT辅助概括、写注释的内容，加上理解代码过程中看的一些资料。我看的资料链接也放在. py. md里，很多是因为我知识欠缺看的。

代码是看完了一遍，脑子里的概念还是比较零碎，不成体系，也有很多没弄清楚的。

# 和论文对应的代码（大致按论文顺序）


## 3.4. In-Context Similarity

![[inter.png|400]]

![[intra.png|400]]

![[inter-and-intra.png]]
>图是附录中的

### 参考资料 
对我来说注意力机制很难理解，也是对理解论文很重要的一个概念，看了好几篇文章（还是看不太懂）：
1. 感性认识：类比人类视觉的选择性注意力机制。翻译时，注意力分配模型分配给不同英文单词的注意力大小不同。[注意力机制（Attention mechanism）基本原理详解及应用 - Jerry_Jin - 博客园 (cnblogs.com)](https://www.cnblogs.com/jins-note/p/13056604.html)
2. 认识Query, Key, Value（q, k, v）：[Attention：注意力机制 - killens - 博客园 (cnblogs.com)](https://www.cnblogs.com/killens/p/16303795.html)
3. 公式和代码实现：[人工智能 - Vision Transformers的注意力层概念解释和代码实现 - deephub - SegmentFault 思否](https://segmentfault.com/a/1190000044678798)
4. self attention和cross attention : [Self -Attention、Multi-Head Attention、Cross-Attention_cross attention-CSDN博客](https://blog.csdn.net/philosophyatmath/article/details/128013258)

### 对应代码
这段话对应的代码，我是在 [attention.py](./models/decoder/attention.py.md) 和 [in_context_correspondence.py](./models/decoder/in_context_correspondence.py.md)  里找到的。

#### attention. py
定义了Attention类，被 [in_context_correspondence.py](./models/decoder/in_context_correspondence.py.md) 中的 `OneWayAttentionBlock` 调用，`OneWayAttentionBlock` 又被 `TrainingCrossAttention` 调用了。

```python
        # 注意力计算
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)  # 缩放
        attn = torch.softmax(attn, dim=-1)  # 应用softmax得到注意力权重
```
大概对应以下公式
![[attention.png|200]]
#### inter-similarity
 [in_context_correspondence.py](./models/decoder/in_context_correspondence.py.md) 中的 `TrainingCrossAttention` 和 `TrainingFreeAttention` 的代码大致相同，不同的是前者多了一个 `OneWayAttentionBlock`. 不太清楚free的含义，是否就是少了 `OneWayAttentionBlock` 的意思。

这两个类基本一致，大概都是实现对"inter-similarity"的计算（`TrainingFreeAttention` 是否也是呢，不太确定）。都有这行：
```python
        sim = einsum('i d, j d -> i j', feature_ref, feature)*scale
```
这行 [einsum](https://zhuanlan.zhihu.com/p/650247173) 的作用其实就是普通的矩阵乘法。对应了Figure 7中inter-similarity里的dot-product.

#### intra-similarity
 [in_context_correspondence.py](./models/decoder/in_context_correspondence.py.md) 定义的 `SemiTrainingAttentionBlocks` 中的 `training_free_self_attention` 应该就是实现"self-attention". Figure 7中intra-similarity里的"element-wise multiplication"逐元素乘法（相同位置的元素相乘， $a_{ij}*b_{ij}$ ）, 体现为：

```python
# propagate , element wise multiplication x_ and self_attn_maps
# 传播，逐元素乘法 x_ 和 self_attn_maps
x_ = x_ * self_attn_map_
```
## 3.5. Matting Head
>The success of ViTMatte [42] implies that the information of original image is important during decoding. Following this practice, in our matting head, the original image is con catenated and decoded with outputs from previous modules. **The guidance map from the in-context similarity module and the intra-features from the backbone** are merged and refined using a **convolutional feature fusion block**, including a series of **convolution, normalization, and activation layers**. The output multi-scale in-context features are progressively merged using a series of **fusion layers which comprise up sampling, concatenation, convolution, normalization, and activation layers**. Then, following ViTMatte [42], details from the original image are extracted and combined with the merged feature in a detail decoder, enhancing the details of alpha matte. This matting head effectively melds contextual information with original image details, yielding the generation of a highly precise and refined alpha matte.

### 在 [in_context_correspondence.py](./models/decoder/in_context_correspondence.py.md) 中
#### SemiTrainingAttentionBlocks类


1. The guidance map from the in-context similarity module（inter-similarity）:
```python        
attn_output = self.attn_module(
            features_cor, features_ref, guidance_on_reference_image)
```

2. the intra-features：
```python
        self_attn_output = self.training_free_self_attention(
            attn_output, ft_attn_of_source_image['attn'])
```

3. The guidance map from the in-context similarity module and the intra-features from the backbone are **merged and refined using a convolutional feature fusion block**, including a series of convolution, normalization, and activation layers. 
```python
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
```

FusionBlock在 [detail_capture.py](./models/decoder/detail_capture.py.md) 中定义

### 在 [detail_capture.py](./models/decoder/detail_capture.py.md) 中
>...merged and refined using a **convolutional feature fusion block**, including a series of **convolution, normalization, and activation layers**. 

```python
class Fusion_Block(nn.Module):
    """
    Simple fusion block to fuse feature from ConvStream and Plain Vision Transformer.
    """
    def __init__(self, in_chans, out_chans):
        super().__init__()
        # 定义一个基本的3x3卷积层
        self.conv = Basic_Conv3x3(in_chans, out_chans, stride=1, padding=1)

    def forward(self, x, D):
        # 上采样操作
        F_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # 将输入特征和上采样特征拼接
        out = torch.cat([D, F_up], dim=1)
        # 通过卷积层
        out = self.conv(out)
        return out  
```
其中 `Basic_Conv3x3` 包含卷积层、归一化层、激活层。


## 4.2 中的Training Details. 

> We employ distinct loss functions for matting and segmentation, respectively. For **matting**, we use a combination of **ℓ1 loss, Laplacian loss, and Gradient loss**. For **segmentation**, we only use the **ℓ1 loss**. To leverage the segmentation dataset while reducing the im pact of imprecise edge annotations, we adopt the approach from HIM [33] that only backpropagates the loss from the confident areas.

体现在：
### eval. yaml
```yaml
model:
  target: icm.models.in_context_matting.InContextMatting
  params:
    learning_rate: 0.0004
    cfg_loss_function:
      target: icm.criterion.loss_function.LossFunction2
      params:
        losses_seg:
          - known_smooth_l1_loss
        losses_matting:
          - unknown_l1_loss
          - known_l1_loss
          - loss_pha_laplacian
          - loss_gradient_penalty
```
还有Training Details中提到
>the learning rate is set to 0.0004 and the batch size is 8. The input images are randomly cropped to a size of 768×768 pixels


同样也能在eval. yaml里找到。

## 4.2中的Evaluation.
>We employ the four widely used matting metrics: SAD, MSE, Grad and Conn [27]. Lower values imply higher-quality mattes. In particular, MSE is scaled by a factor of 1 × 10−3.

在 [criterion/matting_criterion.py](./criterion/matting_criterion.py.md) 

[SAD, MSE, Grad, Conn](./SAD,MSE,Grad,Conn.md)


## 8.3. Ablation on Diffusion Time Steps
>trade-off. As shown in Table 8, our experiments show that performance is suboptimal for rather small t values (e.g., 0) or too large t values (e.g., over 300). **The optimal performance is achieved when t is set to 200**.

### eval. yaml
```yaml
    cfg_feature_extractor:
      #...
        time_steps:
        - 200
```
