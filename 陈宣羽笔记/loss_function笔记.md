>库函数说明见笔记1  
## LossFunction类
```py
class LossFunction(nn.Module):
    '''
    Loss function set
    losses=['unknown_l1_loss', 'known_l1_loss',
            'loss_pha_laplacian', 'loss_gradient_penalty',
            'smooth_l1_loss', 'cross_entropy_loss', 'focal_loss']
    '''
```  
&emsp;&emsp;定义了一个名为 LossFunction 的类，该类继承自 nn.Module 类。nn.Module 是 PyTorch 中用于构建神经网络模型的基类。  
&emsp;&emsp;类的注释中说明了这个类是一个损失函数集合，列出了一些损失函数的名称，包括：
* unknown_l1_loss: 一个未知的 L1 损失函数。
* known_l1_loss: 一个已知的 L1 损失函数。
* loss_pha_laplacian: 一个相位拉普拉斯损失函数。
* loss_gradient_penalty: 一个梯度惩罚损失函数。
* smooth_l1_loss: 一个平滑 L1 损失函数。
* cross_entropy_loss: 一个交叉熵损失函数。
* focal_loss: 一个 Focal Loss 损失函数。
 ```py
    def __init__(self,
                 *,
                 losses,
                 ):
        super(LossFunction, self).__init__()
        self.losses = losses
```
&emsp;&emsp;该类的构造函数 __init__ 接受一个参数 losses，它是一个损失函数的列表，包含上述损失函数名称。  
&emsp;&emsp;通过调用 super(LossFunction, self).__init__()，代码会执行 nn.Module 类的构造函数，以便正确地初始化 LossFunction 类的实例。这样做可以继承并利用父类的功能，确保子类的实例在创建时具有父类的属性和方法。
### loss_gradient_penalty
>该方法计算梯度惩罚损失（gradient penalty loss）
```py
    def loss_gradient_penalty(self, sample_map ,preds, targets):
        #方法的输入参数包括 sample_map、preds 和 targets 
        #它们分别表示样本图、预测值和目标值。
        preds = preds['phas']
        targets = targets['phas']
        h,w = sample_map.shape[2:]
        #首先从 preds 和 targets 中提取出 'phas' 对应的张量。
        #然后，通过 sample_map 的形状获取高度和宽度信息。
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            #sample_map for unknown area
            scale = sample_map.shape[0]*262144/torch.sum(sample_map)
        #根据 sample_map 是否全为零，计算一个比例因子 scale。
        #如果 sample_map 全为零，则将 scale 设置为 0；
        #否则，使用公式计算 scale 的值。

"""
        以下定义了两个 Sobel 算子的卷积核 sobel_x_kernel 和 sobel_y_kernel，
        这些卷积核用于计算 preds 和 targets 在 x 和 y 方向上的梯度。

        使用 F.conv2d 函数对 preds 和 targets 进行卷积操作，
        分别得到在 x 方向上的梯度 delta_pred_x、delta_gt_x，
        以及在 y 方向上的梯度 delta_pred_y、delta_gt_y。
"""
        #gradient in x
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(dtype=preds.type())
        delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
        delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)

        #gradient in y 
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(dtype=preds.type())
        delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
        delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)
'''
        根据公式计算梯度惩罚损失 loss，其中包括两部分：
        x 方向上的 L1 损失
        y 方向上的 L1 损失
        x 方向上的绝对值平均值乘以 0.01
        y 方向上的绝对值平均值乘以 0.01
'''
        #loss
        loss = (F.l1_loss(delta_pred_x*sample_map, delta_gt_x*sample_map)* scale + \
            F.l1_loss(delta_pred_y*sample_map, delta_gt_y*sample_map)* scale + \
            0.01 * torch.mean(torch.abs(delta_pred_x*sample_map))* scale +  \
            0.01 * torch.mean(torch.abs(delta_pred_y*sample_map))* scale)

        return dict(loss_gradient_penalty=loss)
```
&emsp;&emsp;最后，将计算得到的梯度惩罚损失值以字典的形式返回，字典的键为 'loss_gradient_penalty'。 

&emsp;&emsp;这个方法的作用是计算梯度惩罚损失，用于在训练过程中约束模型的梯度，以提高模型的鲁棒性和生成效果。
#### if torch.sum(sample_map) == 0:的说明
```py
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            #sample_map for unknown area
            scale = sample_map.shape[0]*262144/torch.sum(sample_map)
```
&emsp;&emsp;这一步的目的是计算一个比例因子 scale，用于根据 sample_map 的值调整梯度惩罚损失的权重。

&emsp;&emsp;在这段代码中，sample_map 表示样本的地图，可能包含未知区域或其他特定区域的标记。通过计算 sample_map 中所有元素的和 torch.sum(sample_map)，可以判断 sample_map 是否全为零。

&emsp;&emsp;如果 sample_map 全为零，说明样本中没有未知区域或其他特定区域的标记，此时梯度惩罚损失对最终的总损失没有贡献，因此将 scale 设置为 0。

&emsp;&emsp;如果 sample_map 不全为零，说明样本中存在未知区域或其他特定区域的标记。在这种情况下，通过公式 sample_map.shape[0]*262144/torch.sum(sample_map) 计算 scale 的值。这个公式的含义是，将 sample_map 的形状中的第一个维度的大小乘以 262144（即 sample_map 的空间尺寸），再除以 sample_map 中所有元素的和，得到一个比例因子。

&emsp;&emsp;这个比例因子 scale 将用于加权计算梯度惩罚损失的各个部分，以考虑样本中未知区域或其他特定区域的重要性。如果这些区域占据了较大的比例，那么梯度惩罚损失在总损失中的权重就会相应增加。

&emsp;&emsp;总结来说，这一步的目的是根据样本图中未知区域或其他特定区域的情况，计算一个比例因子 scale，用于调整梯度惩罚损失的权重。
### loss_pha_laplacian
>&emsp;&emsp;这段代码定义了一个名为 loss_pha_laplacian 的方法，它是 LossFunction 类的一部分。该方法计算相位（phas）的拉普拉斯损失（laplacian loss）。

```py
    def loss_pha_laplacian(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = laplacian_loss(preds['phas'], targets['phas'])

        return dict(loss_pha_laplacian=loss)
```
&emsp;&emsp;方法的输入参数是 preds 和 targets，它们都应包含键为 'phas' 的项，分别表示预测值和目标值中的相位。

&emsp;&emsp;在方法的实现中，首先使用断言（assert）语句确保 preds 和 targets 中都包含 'phas' 这个键。这是为了确保输入的预测值和目标值中都有相位信息。

&emsp;&emsp;然后，调用了一个名为 laplacian_loss 的函数，将预测值 preds['phas'] 和目标值 targets['phas'] 作为参数传入该函数进行计算。laplacian_loss 函数实现了相位的拉普拉斯损失的计算逻辑。

&emsp;&emsp;最后，将计算得到的相位的拉普拉斯损失值以字典的形式返回，字典的键为 'loss_pha_laplacian'。

&emsp;&emsp;这个方法的作用是计算相位的拉普拉斯损失，用于衡量预测值和目标值之间的相位差异。
### unknown_l1_loss
>这段代码定义了一个名为 unknown_l1_loss 的方法，它是 LossFunction 类的一部分。该方法计算未知区域的 L1 损失（L1 loss）。
```py
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
```
&emsp;&emsp;方法的输入参数包括 sample_map、preds 和 targets，它们分别表示样本图、预测值和目标值。

&emsp;&emsp;在方法的实现中，首先从 sample_map 中获取高度和宽度信息。

&emsp;&emsp;接下来，通过判断 sample_map 是否全为零，来确定是否存在未知区域，此处分析同前文。

&emsp;&emsp;然后，计算未知区域的 L1 损失，使用 F.l1_loss 函数计算预测值 preds['phas'] 和目标值 targets['phas'] 在未知区域上的元素差的绝对值，并将其乘以 sample_map，再乘以比例因子 scale。

&emsp;&emsp;最后，将计算得到的未知区域的 L1 损失值以字典的形式返回，字典的键为 'unknown_l1_loss'。

&emsp;&emsp;这个方法的作用是计算预测值和目标值在未知区域上的 L1 损失，用于衡量预测值和目标值之间在未知区域的差异。这种损失可以用于训练模型，帮助模型更好地处理未知区域的预测。
### known_l1_loss
>这段代码定义了一个名为 known_l1_loss 的方法，它是 LossFunction 类的一部分。该方法计算已知区域的 L1 损失（L1 loss）。
```py
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
```
&emsp;&emsp;实现方法与未知区域方法的实现类似。
### smooth_l1_loss 与 known_smooth_l1_loss
>&emsp;&emsp;smooth_l1_loss 方法计算平滑 L1 损失（smooth L1 loss）。输入参数 preds 和 targets 都应包含键为 'phas' 的项，分别表示预测值和目标值中的相位。

>&emsp;&emsp;known_smooth_l1_loss 方法与 known_l1_loss 方法类似，不同之处在于使用的损失函数是平滑 L1 损失函数（F.smooth_l1_loss）。该方法在计算已知区域的平滑 L1 损失时，也考虑了样本的 sample_map。
```py
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
```
#### known_smooth_l1_loss
&emsp;&emsp;首先，创建一个与 sample_map 形状相同的全零张量 new_sample_map，然后将 sample_map 中值为零的位置设置为 1，以表示已知区域。

&emsp;&emsp;接下来，从 sample_map 中获取高度和宽度信息。

&emsp;&emsp;然后，通过判断 new_sample_map 是否全为零，来确定是否存在已知区域。如果 new_sample_map 全为零，说明样本中没有已知区域，此时将比例因子 scale 设置为 0；否则，使用公式 new_sample_map.shape[0]*262144/torch.sum(new_sample_map) 计算 scale 的值。

&emsp;&emsp;最后，计算已知区域的平滑 L1 损失，使用 F.smooth_l1_loss 函数计算预测值 preds['phas'] 和目标值 targets['phas'] 在已知区域上的平滑 L1 损失，并乘以 new_sample_map，再乘以比例因子 scale。

&emsp;&emsp;最终，将计算得到的已知区域的平滑 L1 损失值以字典的形式返回，字典的键为 'known_l1_loss'。

&emsp;&emsp;这两个方法分别用于计算整体相位的平滑 L1 损失和已知区域的平滑 L1 损失。这些损失函数可以用于训练模型，帮助模型学习更准确的相位预测，并在考虑已知区域时进行更加精细的调整。
### cross_entropy_loss
>&emsp;&emsp;计算预测值（preds）和目标值（targets）之间的交叉熵损失。
```py
    def cross_entropy_loss(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = F.binary_cross_entropy_with_logits(preds['phas'], targets['phas'])

        return dict(cross_entropy_loss=loss)
```

### focal_loss
>&emsp;&emsp;使用 Focal Loss 计算预测值（preds）和目标值（targets）之间的损失。
```py
      def focal_loss(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = focal_loss.sigmoid_focal_loss(preds['phas'], targets['phas'], reduction='mean')

        return dict(focal_loss=loss)
```
&emsp;&emsp;Focal Loss是一种用于解决类别不平衡问题的损失函数，最初在目标检测任务中提出。它通过调整易分类样本的权重来减轻类别不平衡对模型训练的影响。Focal Loss相比于传统的交叉熵损失函数，在处理类别不平衡问题时能够更加有效地关注于困难样本。    
Focal Loss的定义如下：
```py
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```
&emsp;&emsp;其中，p_t 是模型的预测概率，alpha_t 是用于平衡正负样本的权重，gamma 是一个可调整的超参数。

&emsp;&emsp;Focal Loss的特点是对易分类样本施加了降低权重的调整，使得模型更加关注于困难样本（预测概率接近于0.5的样本），从而在类别不平衡的情况下提高模型的性能。
### forward
>&emsp;&emsp;用于计算前向传播过程中的损失值。
```py
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
        """
        创建一个空的损失字典 losses，用于存储各个损失函数计算得到的损失值。

        然后，使用 for 循环遍历 self.losses，self.losses 是一个包含损失函数方法名称的列表。

        对于列表中的每个损失函数名称，通过 getattr 函数获取 self 对象中对应的方法，并调用该方法进行损失计算。

        如果损失函数名称是 'unknown_l1_loss'、'known_l1_loss'、'loss_gradient_penalty' 或者 'known_smooth_l1_loss'，
        则调用对应的方法，并传递 sample_map、preds 和 targets 作为参数，将计算得到的损失值更新到 losses 字典中。

        否则，调用对应的方法，并传递 preds 和 targets 作为参数，将计算得到的损失值更新到 losses 字典中。

        最后，将计算得到的所有损失值组成的 losses 字典返回。
        """
```
&emsp;&emsp;这个方法的目的是计算前向传播过程中的损失值。在深度学习模型的训练过程中，损失函数用于衡量模型预测结果与真实标签之间的差异，通过最小化损失函数来优化模型的参数。

&emsp;&emsp;在这个方法中，根据给定的损失函数列表，逐个调用相应的损失函数方法，并传递所需的输入参数（样本图、预测值和目标值）进行计算。计算得到的每个损失值都被存储在一个字典中，以损失函数的名称作为键，最后将这个字典作为输出返回。

&emsp;&emsp;通过在前向传播过程中计算损失值，可以在训练过程中监控模型的性能，并根据损失值的大小调整模型的参数，以使模型的预测结果更接近真实标签。这有助于提高模型的准确性和泛化能力。

&emsp;&emsp;总之，这个方法的目的是计算前向传播过程中各个损失函数的损失值，为模型的训练提供衡量性能的指标。

## LossFunction2类
>&emsp;&emsp;这段代码定义了一个名为 LossFunction2 的自定义损失函数类。该类继承自 nn.Module 类。

>&emsp;&emsp;在类的构造函数 __init__ 中，定义了两个损失函数列表 losses_seg 和 losses_matting，分别用于分割任务（segmentation task）和抠图任务（matting task）。
```py
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

```
&emsp;&emsp;losses_seg 列表包含了用于分割任务的损失函数，其中只包含了一个损失函数 'known_smooth_l1_loss'。

&emsp;&emsp;losses_matting 列表包含了用于抠图任务的损失函数，其中包含了多个损失函数：'unknown_l1_loss'、'known_l1_loss'、'loss_pha_laplacian' 和 'loss_gradient_penalty'。
### loss_gradient_penalty
>&emsp;&emsp;定义了一个名为 loss_gradient_penalty 的损失函数方法，用于计算梯度惩罚损失。
```py
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
```
&emsp;&emsp;这个方法的作用是计算梯度惩罚损失，用于约束模型预测值的梯度与目标值的梯度之间的差异。
>梯度惩罚损失（Gradient Penalty Loss）是一种用于增强生成对抗网络（GAN）训练稳定性和生成样本质量的技巧。它被用于在训练过程中约束生成器网络的梯度，并帮助生成器产生更真实、高质量的样本。

>梯度惩罚损失的基本思想是通过对生成器网络的梯度进行惩罚，使其在生成样本时更加平滑，并且避免生成样本在局部区域出现明显的不连续性或噪声。

>在GAN模型中，通常有一个判别器网络和一个生成器网络。判别器网络用于区分真实样本和生成样本，生成器网络用于生成伪造的样本。梯度惩罚损失通常应用在判别器网络上。

>具体实现梯度惩罚损失的方法是，在真实样本和生成样本之间随机采样一些样本，然后计算这些样本的梯度。通过计算梯度的范数（通常是L2范数）并对其进行惩罚（通常是乘以一个权重因子），将这个惩罚项添加到判别器的损失函数中。

>梯度惩罚损失的作用是使生成器产生更平滑的样本，避免生成样本中出现明显的不连续性或噪声。它有助于提升生成模型的训练稳定性，并生成更逼真的样本。
### 一系列损失函数方法（同）
```py
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
```
* loss_pha_laplacian 方法计算 Laplacian 损失。它接收 preds 和 targets 作为输入，确保这两个字典中都包含了 'phas' 键。然后，调用 laplacian_loss 方法计算预测值和目标值之间的 Laplacian 损失，并将其返回。
* unknown_l1_loss 方法计算未知区域的 L1 损失。它接收 sample_map、preds 和 targets 作为输入。首先，计算 sample_map 的高度 h 和宽度 w。然后，通过判断 sample_map 的总和是否为零来确定 scale 的值。如果总和为零，则 scale 被设置为 0；否则，将计算 scale 的值。接下来，使用 F.l1_loss 函数计算预测值和目标值在未知区域上的 L1 损失，并乘以 scale 进行缩放。最后，将损失值组成的字典返回，字典的键为 'unknown_l1_loss'。
* known_l1_loss 方法计算已知区域的 L1 损失。它接收 sample_map、preds 和 targets 作为输入。首先，创建一个与 sample_map 相同大小的全零张量 new_sample_map，然后将其中 sample_map 为零的位置设置为 1，以获取已知区域的掩码。接着，计算 new_sample_map 的高度 h 和宽度 w。通过判断 new_sample_map 的总和是否为零，确定 scale 的值。然后，使用 F.l1_loss 函数计算预测值和目标值在已知区域上的 L1 损失，并乘以 scale 进行缩放。最后，将损失值组成的字典返回，字典的键为 'known_l1_loss'。
* smooth_l1_loss 方法计算平滑 L1 损失。它接收 preds 和 targets 作为输入，确保这两个字典中都包含了 'phas' 键。然后，使用 F.smooth_l1_loss 函数计算预测值和目标值之间的平滑 L1 损失，并将其返回。
* known_smooth_l1_loss 方法计算已知区域的平滑 L1 损失。它接收 sample_map、preds 和 targets 作为输入，通过创建已知区域的掩码 new_sample_map，计算 scale 的值，并使用 F.smooth_l1_loss 函数计算预测值和目标值在已知区域上的平滑 L1 损失。最后，将损失值组成的字典返回，字典的键为 'known_l1_loss'。
* cross_entropy_loss 方法计算交叉熵损失。它接收 preds 和 targets 作为输入，确保这两个字典中都包含了 'phas' 键。然后，使用 F.binary_cross_entropy_with_logits 函数计算预测值和目标值之间的交叉熵损失，并将其返回。
* focal_loss 方法计算 Focal 损失。它接收 preds 和 targets 作为输入，确保这两个字典中都包含了 'phas' 键。然后，调用 focal_loss.sigmoid_focal_loss 方法计算预测值和目标值之间的 Focal 损失，并将其返回。
### forward_single_sample
>&emsp;&emsp;在单个样本上进行前向传播并计算损失。根据 targets 的内容选择不同的损失函数集合，然后在相应的损失函数集合中循环计算损失并返回损失字典。
```py
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
```
首先检查 targets 是否只包含元素 0 和 1。如果是，则执行以下操作：
1. 将 preds 和 targets 封装成字典形式，其中键为 'phas'，值为对应的张量。
2. 创建一个空字典 losses，用于存储损失值。
3. 对于 self.losses_seg 中的每个损失函数名称 k，执行以下操作：
* 如果损失函数名称是 'unknown_l1_loss'、'known_l1_loss'、'loss_gradient_penalty' 或 'known_smooth_l1_loss'，则调用相应的损失函数方法并传递 sample_map、preds 和 targets 进行计算，并将计算得到的损失添加到 losses 字典中。
* 否则，调用相应的损失函数方法并传递 preds 和 targets 进行计算，并将计算得到的损失添加到 losses 字典中。
4. 返回损失字典 losses。
如果 targets 不仅包含元素 0 和 1，上述操作将被跳过，而是执行以下操作：
#
1. 将 preds 和 targets 封装成字典形式，其中键为 'phas'，值为对应的张量。
2. 创建一个空字典 losses，用于存储损失值。
3. 对于 self.losses_matting 中的每个损失函数名称 k，执行与上述步骤相同的操作，计算相应的损失并将其添加到 losses 字典中。
4. 返回损失字典 losses。
>检查 targets 是否只包含元素 0 和 1的目的:

&emsp;&emsp;检查 targets 是否只包含元素 0 和 1 的目的是确定当前样本的目标值是属于分类任务还是其他类型的任务。这个检查是根据任务的特定需求设计的，可能是因为模型在不同类型的任务上使用了不同的损失函数或其他处理逻辑。

&emsp;&emsp;如果 targets 中的元素**全都是 0 或者全都是 1**，那么可以认为这是一个**分类任务**，目标值只有两个类别。在这种情况下，代码会执行与 self.losses_seg 相关的部分，计算与分割任务相关的损失函数，并返回损失字典。

&emsp;&emsp;如果 targets 中的元素**不仅限于 0 和 1**，那么可以推断这是一个**不同类型的任务**，例如回归任务或多类别分类任务等。在这种情况下，代码会执行与 self.losses_matting 相关的部分，计算与该任务相关的损失函数，并返回损失字典。

&emsp;&emsp;通过检查 targets 中的元素，代码能够根据任务类型动态选择适当的损失函数集合，并进行相应的损失计算。这种灵活性可以使模型适应不同类型的任务，并根据任务需求进行相应的处理。

### forward
>&emsp;&emsp;定义了一个名为 forward 的方法，用于在多个样本上进行前向传播并计算总体损失。
```py
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
```
&emsp;&emsp;这个方法的目的是在多个样本上进行前向传播和损失计算，并将每个样本的损失累加到总体损失中。通过循环遍历样本并调用 forward_single_sample 方法，可以处理整个批次的样本，并返回总体损失字典。

## laplacian_loss 拉普拉斯损失
> 
```py
#-----------------Laplacian Loss-------------------------#
    """
首先，laplacian_loss 函数计算 Laplacian Loss。它接受两个输入参数 pred 和 true，
分别表示预测值和真实值。max_levels 是一个可选参数，用于指定金字塔的最大级别，默认为5。
    """
def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    #调用 gauss_kernel 函数，传入预测值的设备类型和数据类型，获得一个高斯核。
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    #调用 laplacian_pyramid 函数，分别对预测值和真实值构建拉普拉斯金字塔。
    #该函数会使用高斯核和最大级别数作为参数，对输入的预测值和真实值进行多级别的下采样和差值计算，并返回金字塔形式的结果。
    loss = 0
    #初始化一个变量 loss 为0，用于累加每个级别的损失。
     """
    通过迭代每个级别，从金字塔中取出对应级别的预测值和真实值，并计算它们之间的 L1 损失。
    L1 损失是指预测值和真实值之间的绝对差值的平均值。
    在这里，每个级别的损失乘以一个权重因子 (2 ** level)，这个权重因子随着级别的增加而增大，用于加权不同级别的损失。
    """
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels
    #将所有级别的损失累加起来，并除以最大级别数 max_levels，得到平均损失，并将其返回作为函数的输出。s
```
&emsp;&emsp;上述代码针对预测值和真实值之间差异的 Laplacian Loss 的计算，通过对预测值和真实值构建拉普拉斯金字塔，并计算每个级别的 L1 损失并加权求和，得到最终的平均损失。
### laplacian_pyramid
>&emsp;&emsp;实现了通过多级别的下采样、上采样和差值计算构建拉普拉斯金字塔的功能。
```py
"""
该函数接受三个输入参数：
img 表示输入图像，kernel 表示高斯核，max_levels 表示金字塔的最大级别数。
"""
def laplacian_pyramid(img, kernel, max_levels):
    current = img#将输入图像赋值给变量 current
    pyramid = []#创建一个空列表 pyramid，用于存储金字塔的不同级别
    """循环迭代来构建金字塔的各个级别"""
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        #将 current 裁剪为偶数大小,确保图像大小为偶数
        down = downsample(current, kernel)
        #对 current 进行下采样操作，得到下采样后的图像 down
        up = upsample(down, kernel)
        #对 down 进行上采样操作，得到上采样后的图像 up
        diff = current - up
        #计算当前级别的差值图像，即用 current 减去 up
        pyramid.append(diff)
        #将差值图像 diff 添加到金字塔列表 pyramid 中
        current = down
        #将下采样后的图像 down 赋值给 current，用于下一个级别的计算
    return pyramid
    #返回金字塔列表 pyramid，
    #其中包含了从输入图像到最高级别的各个级别的差值图像。
```
### gauss_kernel 高斯核
>&emsp;&emsp;构建高斯核的目的是进行图像模糊或平滑操作。高斯核是一种常用的卷积核，用于对图像进行平滑处理，可以有效地去除图像中的噪声，并模糊图像中的细节，使得图像更加平滑。
```py
def gauss_kernel(device='cpu', dtype=torch.float32):
    """构建一个 5x5 的高斯核，并将其保存在变量 kernel 中。
       高斯核的数值是根据高斯分布函数计算得到的。 """
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    #将高斯核的值除以 256，以进行归一化处理，使得高斯核的和为1。
    kernel /= 256
    """对高斯核进行形状变换，将其添加两个额外的维度，
    使得其形状变为 (1, 1, 5, 5)。这是为了与输入图像进行卷积操作时的要求。"""
    kernel = kernel[None, None, :, :]
    return kernel
```
why 5*5? 

&emsp;&emsp;构建一个 5x5 的高斯核是一种常见的选择，因为它在平滑图像时能够提供适度的模糊效果，同时计算效率较高。

&emsp;&emsp;在构建高斯核时，一般使用高斯分布函数作为权重来定义卷积核的数值。5x5 的高斯核具有足够的大小，可以捕捉到图像中较大范围的细节，并对其进行平滑处理。较大的卷积核可以提供更广阔的感受野，从而更好地对图像进行平滑处理。
### 维度问题
&emsp;&emsp;在构建卷积神经网络（CNN）中，卷积操作要求输入张量和卷积核张量具有相同的维度。对于二维卷积操作，输入张量通常具有四个维度：(batch_size, channels, height, width)，其中 batch_size 表示批量大小，channels 表示通道数，height 表示图像的高度，width 表示图像的宽度。

&emsp;&emsp;对于卷积核张量，它的维度要求是 (out_channels, in_channels, kernel_height, kernel_width)，其中 out_channels 表示输出通道数，in_channels 表示输入通道数，kernel_height 表示卷积核的高度，kernel_width 表示卷积核的宽度。

&emsp;&emsp;在构建高斯核时，为了与输入图像进行卷积操作，需要将高斯核的维度调整为与输入图像的维度相同。因此，在 gauss_kernel 函数中，通过添加两个额外的维度，将高斯核的形状从 (5, 5) 调整为 (1, 1, 5, 5)。

* 添加的第一个维度 None 表示扩展一个维度用于批量大小，因为高斯核只有一个，没有批量的概念，所以批量大小设置为1。

* 添加的第二个维度 None 表示扩展一个维度用于输入通道数，因为高斯核只有一个通道，与输入图像的通道数相匹配。

&emsp;&emsp;通过添加这两个额外的维度，可以使高斯核的形状与输入图像保持一致，从而能够进行有效的卷积操作。这样，每个高斯核的元素将与输入图像的对应元素进行乘法运算和求和，以实现平滑和模糊的效果。

### gauss_convolution 高斯核卷积
```py
def gauss_convolution(img, kernel):
    """获取输入图像的形状，并将其分解为:
    批量大小 B、通道数 C、高度 H 和宽度 W"""
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    #将输入图像进行形状变换，将其变为一个四维张量
    #第一维度表示批量大小和通道数的乘积，
    #第二维度表示通道数（此处为1），第三和第四维度表示图像的高度和宽度。
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    #对图像进行填充操作，将图像的边界扩展
    #填充模式使用 'reflect'，表示使用图像边界像素进行镜像填充
    img = F.conv2d(img, kernel)
    #进行二维卷积操作，将高斯核应用于填充后的图像
    img = img.reshape(B, C, H, W)
    #将卷积结果的形状重新变回与输入图像相同的形状
    return img
```
### downsample下采样
```py
def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    #调用 gauss_convolution 函数对输入图像进行高斯卷积操作，
    #这可以平滑图像并减少噪声，以得到平滑后的图像。
    img = img[:, :, ::2, ::2]
    #对卷积结果进行下采样，
    #切片操作中的 ::2 表示跳过每隔一个像素进行采样，
    #即在水平和垂直方向上将图像的尺寸缩小一半。
    return img
```
&emsp;&emsp;下采样可以帮助提取图像的主要特征，因为在降低图像尺寸的同时，也减少了图像中的细节信息（减少图像的像素数量，降低图像的空间分辨率）。
### upsample上采样
>&emsp;&emsp;该函数的目的是对输入图像进行上采样操作，并在上采样之前先对图像进行高斯卷积。上采样可以增加图像的尺寸和细节信息，而高斯卷积可以平滑图像并减少噪声，从而提供更清晰和更细节的上采样结果。
```py
def upsample(img, kernel):
    B, C, H, W = img.shape
    #获取输入图像的形状，将其分解为批量大小 B、通道数 C、高度 H 和宽度 W
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    #通过 torch.zeros 创建一个形状为 (B, C, H * 2, W * 2) 的全零张量 out，用于存储上采样后的图像。
    #这里的 H * 2 和 W * 2 表示在垂直和水平方向上将图像的尺寸放大两倍。
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out
```
### 上采样和下采样的使用？
&emsp;&emsp;通常情况下，上采样和下采样是成对使用的操作，它们通常在图像处理中相互配合使用。

* 下采样用于减小图像的尺寸，降低计算量，并提取图像的主要特征。下采样可以在一些任务中起到加快计算速度和降低存储需求的作用，例如图像分类或目标检测。

* 而上采样则用于增加图像的尺寸和细节信息。它可以在降低图像尺寸的同时还原图像的细节，使得图像在视觉上更加清晰。上采样常用于任务如图像分割或生成高分辨率图像等。

&emsp;&emsp;因此，在一些应用场景中，可以先使用下采样操作对图像进行降采样，提取主要特征，然后再使用上采样操作将图像恢复到原始尺寸并还原细节信息。
### crop_to_even_size
> 将输入图像裁剪为偶数尺寸
```py
def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]
```