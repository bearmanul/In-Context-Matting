# In-Context-Matting
基于上下文图像抠图

==**将我看的陆昊老师的论文中的一些关键概念和操作记录在README中，组员可以看一下，后续如果有遗漏或差错，大家可以继续修改（最好还是要看一看原文）**==

## 论文简略笔记

### 1、introduce

trimaps [7, 17, 1921, 34, 40], scribbles [41], or a even known background

已有的减少抠图不确定性的方法，trimaps，涂鸦，已知背景

Despite their inherent advantages, these automatic matting models are narrowed to specific object categories, such as humans [4, 10, 22, 33], animals [13], and salient objects [12, 45].

一些自动抠图只能服务特定目标

Can the auxiliary inputbased matting be optimized to enhance the efficiency, while also maintaining guidance for matting targets with sufficient automation, thereby harmonizing the two existing matting paradigms?

是否可以优化基于辅助输入的抠图以提高效率，同时保持对抠图目标的指导，从而使现有的两种抠图范式协调一致?

In this work, we approach this challenge as a problem of region-to-region matching.

这个工作将挑战变为区域到区域匹配的问题。

we explore the applicability of pretrained diffusion models for in-context matting.

预训练扩散模型在上下文抠图中的适用性。

However, the matching is often sparse and insufficient to represent the  entire target area. To address this, the intraimage similarity, based on the self-attention maps of Stable Diffusion, is additionally used to  supplement the missing parts. By leveraging both inter- and intra-image  similarities, informative guidance of the matting target would be  acquired. Finally, any off-the-shelf matting heads can be used to  predict the alpha matte.

然而，匹配往往是稀疏的，不足以代表整个目标区域。为了解决这个问题，基于稳定扩散的自关注映射的图像内相似性被额外用于补充缺失的部分。利用图像间和图像内的相似性，可以获得对抠图目标的信息制导。最后，任何现成的抠图头都可以用来预测阿尔法抠图。

### 2、related work

#### 图像抠图：

The network structures used in automatic matting can be divided into two groups: one-stage network with global guidance [26] and parallel  multi-task network [11, 33].

自动抠图被分为全局引导的单阶段网络[26]和并行多任务网络[11,33]

#### 视觉中的上下文学习：

Flamingo [1], a family of visual language models, shows rapid adaptation to a variety of image and video tasks with few-shot learning  capabilities.

Flamingo[1]是一类视觉语言模型，具有快速适应各种图像和视频任务的能力，具有少量的学习能力。

### 3、具有扩散性模型的上下文抠图

#### 3.1 问题建立

Notably, the reference image can either be a part of the input  collection or an entirely separate image. When the input image  collection has only a single image, users can treat that image as the  reference image. In this case, in-context matting degenerates into image matting guided by user interaction.

值得注意的是，参考图像可以是输入集合的一部分，也可以是完全独立的图像。当输入图像集合只*有一张图像时*，用户可以将该图像作为参考图像。在这种情况下，上下文抠图退化为用户交互引导下的图像抠图。

when provided with a reference input, it becomes an automatic matting system targeted towards a specific foreground.

当提供参考输入时，它成为针对特定前景的自动抠图系统。

#### 3.2 总体结构

==（这里最好要看一看论文中的结构图，会更清楚）==

IconMatting is comprised of three components: a feature extractor, an in-context similarity module, and a matting head.

IconMatting由三个组件组成:特征提取器、上下文相似性模块和抠图头。

the features and selfattention maps from the target image.

从目标图中提取特征和自注意图

the former leverages the reference RoI features as an in-context query  to derive a guidance map from the target features; the latter integrates the guidance map with multi-scale self-attention maps to obtain  guidance for the matting head.

前者利用参考RoI特征作为上下文查询，从目标特征派生出引导图;后者将制导图与多尺度自注意图相结合，获得对抠图头的制导。

which, combined with self-attention maps, assists in locating the target object.

引导图与自注意图结合

#### 3.3 特征提取器

Therefore, if the features derived from a backbone naturally possess  correspondence capabilities, referred to as incontext features,

从主干派生的特征自然具有对应能力，称为上下文特征，

we leverage Stable Diffusion as a feature extractor to implement in-context matting.

利用Stable Diffusion作为特征提取器来实现上下文抠图。

IconMatting uses the capabilities of Stable Diffusion and both reference and target images to extract multi-scale features and self-attention  maps to enhance feature representation.

使用稳定扩散以及参考和目标图像的功能来提取多尺度特征和自关注图，以增强特征表示。

#### 3.4 上下文相似性模块

According to our observations, both the reference-target similarity and  target-target similarity matter for locating the potential target  foreground. These correspond to the proposed intersimilarity and  intra-similarity sub-modules.

参考-目标相似度和目标-目标相似度对潜在目标前景的定位都很重要。这些对应于所提出的==Inter==相似性和==Intra==相似性子模块。

##### Observation

one can associate points of the foreground areas between the reference  and target images using the emergent feature correspondence.

利用紧急特征对应将参考图像和目标图像之间的前景区域的点联系起来。

a rigorous one-to-one mapping of all points between the two areas is unfeasible.

严格一对一映射是不可行的。

To address this, we look for other points sharing similar semantic meaning with this subset of points.

寻找与这个点子集共享相似语义的其他点。

the self-attention maps from Stable Diffusion reflect the similarities between different image patches.

来自Stable Diffusion的自注意图反映了不同图像块之间的相似性。

##### ==Inter==-Similarity

The similarity map is denoted by {Sk}K , and the mean of all such  similarity maps yields S, which measures the degree of similarity  between different locations on It and the RoI in Ir, serving as the  first intermediate output of incontext similarity.

相似度图用{Sk}K表示，所有相似度图的平均值为S, S衡量It上不同位置与Ir中的RoI之间的相似程度，作为上下文相似度的第一个中间输出。

##### ==Intra==-Similarity

Here we further design the intra-similarity sub-module to leverage the  internal similarity within It to propagate S into a more precise  representation of the RoI on It.

进一步设计了内部相似性子模块，以利用It内部的相似性将S传播为It上的RoI的更精确表示

{Al}L representing its internal similarity are also retained, serving as the input for intra-similarity

自注意图做输入

This sub-module uses S as a weight to the self-attention maps

使用S作为自关注映射的权重

#### 3.5 抠图头

#### 3.6 Reference-Prompt Extension

Specifically, for each prompt point, the top m points with the highest  responses in their corresponding attention maps are integrated into the  RoI mask additionally, thus enriching the in-context query.

对于每个提示点，将其对应的注意图中响应最高的前m个点额外集成到RoI掩码中，从而丰富了上下文查询

### 4 Results and Discussion

#### 4.1 数据集

In particular, in-context matting requires to organize images into  groups where the annotated foregrounds share categories or instances.  Such organization allows for random selection of reference and target  images within groups during training. In the test set, one or more  images in each group are designated as reference images.

特别是，上下文抠图需要将图像组织成组，其中注释的前景共享类别或实例。这样的组织允许在训练过程中随机选择组内的参考和目标图像。在测试集中，每组中指定一个或多个图像作为参考图像。

#### 4.2 一些细节

U-Net has 11 decoder blocks; we extract feature maps from the 5-th,  8-th, and 11-th blocks as the intra-features and ones from the 5-th  block as the inter-features.

U-Net有11个解码器块;我们从第5、8、11块提取特征映射作为intra特征，从第5块提取特征映射作为inter特征。

For matting, we use a combination of ℓ1 loss, Laplacian loss, and Gradient loss.

对于消光，我们使用了1损失、拉普拉斯损失和梯度损失的组合

We employ the four widely used matting metrics: SAD, MSE, Grad and Conn [27].

采用了四种广泛使用的抠图指标:SAD、MSE、Grad和Conn

If both inter- and intrasimilarity are absent, the model degenerates to  directly predicting the alpha matte from the image, losing the  information source for the specified matting target, and thus the  performance markedly deteriorates.

如果缺少内部相似度和内部相似度，则模型退化为直接从图像中预测alpha哑光，失去了指定抠图目标的信息源，从而导致性能明显下降。
