
# ChatGPT的概括
以下是各类和函数的中文概述：

### register_attention_control函数
- **功能**：注册注意力控制。
- **内部函数ca_forward**：实现基于xFormers的内存高效注意力处理器。处理输入的hidden_states、encoder_hidden_states等，计算注意力并返回处理后的hidden_states。
- **MyXFormersAttnProcessor类**：实现内存高效注意力处理器，调用xformers库进行注意力计算。
- **DummyController类**：用于提供默认的注意力控制器。
- **register_recr函数**：递归注册注意力层。
- **流程**：遍历UNet子模块，注册注意力层，设置控制器的注意力层数量。

### MyUNet2DConditionModel类
- **功能**：自定义的UNet2DConditionModel，重写了forward方法。
- **forward方法**：处理输入张量、时间步、编码器隐藏状态等，经过时间嵌入、卷积层、下采样块、中间块和上采样块的处理，返回指定上采样块的输出。

### OneStepSDPipeline类
- **功能**：基于StableDiffusionPipeline的类，进行一步式的图像处理。
- **\_\_call\_\_方法**：对输入图像进行编码，加噪声后用U-Net处理，返回U-Net的输出。

### SDFeaturizer类
- **功能**：用于特征提取的类，加载预训练的UNet模型和OneStepSDPipeline。
- **初始化方法**：加载模型、调度器，并将模型移动到GPU上，启用内存高效注意力。
- **forward方法**：处理输入图像和提示，返回指定上采样块的U-Net特征。
- **forward_feature_extractor方法**：处理一批图像和提示，返回所有指定上采样块的U-Net特征。

# 代码注释
```python
from diffusers import StableDiffusionPipeline  # 导入StableDiffusionPipeline模块
import torch  # 导入torch模块
import torch.nn as nn  # 导入torch.nn模块，并将其简化为nn
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，并将其简化为plt
import numpy as np  # 导入numpy模块，并将其简化为np
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型注释所需的模块
from diffusers.models.unet_2d_condition import UNet2DConditionModel  # 导入UNet2DConditionModel模块
from diffusers import DDIMScheduler  # 导入DDIMScheduler模块
import gc  # 导入gc（垃圾回收）模块
from PIL import Image  # 导入PIL库中的Image模块

from icm.models.feature_extractor.attention_controllers import AttentionStore  # 导入AttentionStore模块
import xformers  # 导入xformers模块


def register_attention_control(model, controller, if_softmax=True, ensemble_size=1):
    # 定义register_attention_control函数，注册注意力控制

    def ca_forward(self, place_in_unet, att_opt_b):
        # 定义ca_forward内部函数，用于处理注意力前向传播

        class MyXFormersAttnProcessor:
            """
            实现基于xFormers的内存高效注意力处理器

            参数:
                attention_op (`Callable`, *optional*, defaults to `None`):
                    用作注意力操作的基础操作符。建议设置为`None`，允许xFormers选择最佳操作符。
            """

            def __init__(self, attention_op=None):
                self.attention_op = attention_op  # 初始化attention_op

            def __call__(
                self,
                attn,
                hidden_states: torch.FloatTensor,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states  # 保存残差连接的输入

                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)  # 如果存在空间归一化，则进行归一化处理

                input_ndim = hidden_states.ndim  # 获取hidden_states的维度

                if input_ndim == 4:
                    # 如果维度为4，调整hidden_states的形状
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(
                        batch_size, channel, height * width).transpose(1, 2)

                batch_size, key_tokens, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )  # 获取batch_size和key_tokens的数量

                attention_mask = attn.prepare_attention_mask(
                    attention_mask, key_tokens, batch_size)  # 准备注意力掩码
                if attention_mask is not None:
                    # 扩展注意力掩码的维度
                    # expand our mask's singleton query_tokens dimension:
                    #   [batch*heads,            1, key_tokens] ->
                    #   [batch*heads, query_tokens, key_tokens]
                    # so that it can be added as a bias onto the attention scores that xformers computes:
                    #   [batch*heads, query_tokens, key_tokens]
                    # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
                    _, query_tokens, _ = hidden_states.shape
                    attention_mask = attention_mask.expand(-1, query_tokens, -1)

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(
                        hidden_states.transpose(1, 2)).transpose(1, 2)  # 如果存在群归一化，则进行归一化处理

                query = attn.to_q(hidden_states)  # 计算query

                is_cross = False if encoder_hidden_states is None else True  # 判断是否为交叉注意力

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states  # 如果encoder_hidden_states为空，则将其设置为hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(
                        encoder_hidden_states)  # 如果存在norm_cross，则进行归一化处理

                key = attn.to_k(encoder_hidden_states)  # 计算key
                value = attn.to_v(encoder_hidden_states)  # 计算value

                query = attn.head_to_batch_dim(query).contiguous()  # 调整query的形状
                key = attn.head_to_batch_dim(key).contiguous()  # 调整key的形状
                value = attn.head_to_batch_dim(value).contiguous()  # 调整value的形状

                # 控制器
                if query.shape[1] in controller.store_res:
                    sim = torch.einsum('b i d, b j d -> b i j',
                                    query, key) * attn.scale  # 计算注意力相似度

                    if if_softmax:
                        sim = sim / if_softmax  # 如果启用softmax，则进行归一化
                        my_attn = sim.softmax(dim=-1).detach()  # 计算softmax并分离
                        del sim  # 删除相似度矩阵
                    else:
                        my_attn = sim.detach()  # 直接分离相似度矩阵

                    controller(my_attn, is_cross, place_in_unet, ensemble_size, batch_size)  # 调用控制器处理注意力

                # 结束控制器

                hidden_states = xformers.ops.memory_efficient_attention(
                    query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
                )  # 使用xformers进行内存高效的注意力计算
                hidden_states = hidden_states.to(query.dtype)  # 转换hidden_states的数据类型
                hidden_states = attn.batch_to_head_dim(hidden_states)  # 调整hidden_states的形状

                # 线性投影
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    # 如果输入维度为4，调整hidden_states的形状
                    hidden_states = hidden_states.transpose(
                        -1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual  # 添加残差连接

                hidden_states = hidden_states / attn.rescale_output_factor  # 缩放输出

                return hidden_states  # 返回hidden_states

        return MyXFormersAttnProcessor(att_opt_b)  # 返回MyXFormersAttnProcessor实例

    class DummyController:

        def __call__(self, *args):
            return args[0]  # 返回第一个参数

        def __init__(self):
            self.num_att_layers = 0  # 初始化注意力层数量为0

    if controller is None:
        controller = DummyController()  # 如果控制器为空，则使用DummyController

    def register_recr(net_, count, place_in_unet):
        # 定义register_recr函数，递归注册注意力层
        if net_.__class__.__name__ == 'Attention':
            net_.processor = ca_forward(
                net_, place_in_unet, net_.processor.attention_op)  # 注册注意力处理器
            return count + 1  # 增加计数
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)  # 递归注册子模块
        return count

    cross_att_count = 0  # 初始化交叉注意力层计数为0
    # sub_nets = model.unet.named_children()
    sub_nets = model.unet.named_children()  # 获取UNet子模块
    # for net in sub_nets:
    #     if "down" in net[0]:
    #         cross_att_count += register_recr(net[1], 0, "down")
    #     elif "up" in net[0]:
    #         cross_att_count += register_recr(net[1], 0, "up")
    #     elif "mid" in net[0]:
    #         cross_att_count += register_recr(net[1], 0, "mid")
    for net in sub_nets:
        if "down_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")  # 注册下采样块中的注意力层
        elif "up_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")  # 注册上采样块中的注意力层
        elif "mid_block" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")  # 注册中间块中的注意力层
    controller.num_att_layers = cross_att_count  # 设置控制器的注意力层数量

class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        参数:
            sample (`torch.FloatTensor`): (batch, channel, height, width) 带噪声的输入张量
            timestep (`torch.FloatTensor` 或 `float` 或 `int`): (batch) 时间步
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) 编码器隐藏状态
            cross_attention_kwargs (`dict`, *optional*):
                一个可选的关键字参数字典，如果指定，则传递给定义在`self.processor`中的`AttnProcessor`，
                参考[diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py)。
        """
        # 默认情况下，样本必须至少是整体上采样因子的倍数。
        # 整体上采样因子等于2的（上采样层数）的幂。
        # 然而，如果有必要，可以在运行时强制上采样插值输出大小以适应任何上采样大小。
        default_overall_up_factor = 2**self.num_upsamplers  # 计算整体上采样因子

        # 当样本不是`default_overall_up_factor`的倍数时，应转发上采样大小
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # 如果样本的高或宽不是整体上采样因子的倍数，则转发上采样大小
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 准备attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)  # 调整attention_mask的形状

        # 0. 如果需要，中心化输入
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0  # 中心化处理

        # 1. 时间
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: 这需要在CPU和GPU之间同步。因此，如果可以，请尝试将时间步作为张量传递
            # 这对`match`语句（Python 3.10+）是一个很好的用例
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor(
                [timesteps], dtype=dtype, device=sample.device)  # 创建时间步张量
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)  # 调整时间步张量的形状

        # 以一种兼容ONNX/Core ML的方式广播到批量维度
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)  # 计算时间嵌入

        # timesteps不包含任何权重，总是返回f32张量
        # 但time_embedding可能实际上以fp16运行，因此我们需要在这里进行类型转换
        # 可能有更好的方法来封装这个
        t_emb = t_emb.to(dtype=self.dtype)  # 转换时间嵌入的类型

        emb = self.time_embedding(t_emb, timestep_cond)  # 计算时间嵌入

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "当num_class_embeds > 0时，应提供class_labels"
                )

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)  # 计算时间步嵌入

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)  # 计算类别嵌入
            emb = emb + class_emb  # 将类别嵌入加到时间嵌入上

        # 2. 预处理
        sample = self.conv_in(sample)  # 通过输入卷积层处理样本

        # 3. 下采样
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )  # 如果存在交叉注意力，则进行相应的处理
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb)  # 否则，仅进行下采样

            down_block_res_samples += res_samples  # 保存下采样的中间结果

        # 4. 中间块
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )  # 通过中间块处理样本

        # 5. 上采样
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):
            # if i > np.max(up_ft_indices):
            #     break

            is_final_block = i == len(self.up_blocks) - 1  # 判断是否为最后一个上采样块

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]  # 获取下采样块的中间结果

            # 如果未到达最后一个块且需要转发上采样大小，则在这里进行
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )  # 如果存在交叉注意力，则进行相应的处理
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )  # 否则，仅进行上采样

            if i in up_ft_indices:
                up_ft[i] = sample.detach()  # 保存上采样块的输出

        output = {}
        output["up_ft"] = up_ft  # 将上采样块的输出保存到字典中
        return output  # 返回输出字典
class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        device = self._execution_device  # 获取执行设备
        latents = (
            self.vae.encode(img_tensor).latent_dist.sample()
            * self.vae.config.scaling_factor
        )  # 对输入图像进行编码，并按缩放因子进行采样
        t = torch.tensor(t, dtype=torch.long, device=device)  # 将时间步t转换为张量
        noise = torch.randn_like(latents).to(device)  # 生成与潜变量形状相同的随机噪声
        latents_noisy = self.scheduler.add_noise(latents, noise, t)  # 将噪声添加到潜变量中
        unet_output = self.unet(
            latents_noisy,
            t,
            up_ft_indices,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )  # 使用U-Net对带噪声的潜变量进行处理
        return unet_output  # 返回U-Net的输出


class SDFeaturizer(nn.Module):
    def __init__(self, sd_id='pretrained_models/stable-diffusion-2-1',
                 load_local=False):
        super().__init__()
        # 初始化SDFeaturizer类

        unet = MyUNet2DConditionModel.from_pretrained(
            sd_id,
            subfolder="unet",
            local_files_only=load_local,
            low_cpu_mem_usage=True,
            use_safetensors=False,
        )  # 加载预训练的UNet模型

        onestep_pipe = OneStepSDPipeline.from_pretrained(
            sd_id,
            unet=unet,
            safety_checker=None,
            local_files_only=load_local,
            low_cpu_mem_usage=True,
            use_safetensors=False,
        )  # 加载OneStepSDPipeline

        onestep_pipe.vae.decoder = None  # 禁用VAE的解码器
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(
            sd_id, subfolder="scheduler"
        )  # 加载预训练的DDIM调度器
        gc.collect()  # 进行垃圾回收

        onestep_pipe = onestep_pipe.to("cuda")  # 将管道移动到GPU上

        onestep_pipe.enable_attention_slicing()  # 启用注意力切片
        onestep_pipe.enable_xformers_memory_efficient_attention()  # 启用内存高效注意力
        self.pipe = onestep_pipe  # 保存管道

        # 注册nn.Module用于分布式数据并行
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        # 冻结VAE和UNet的参数
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, img_tensor, prompt='', t=261, up_ft_index=3, ensemble_size=8):
        """
        参数:
            img_tensor: 应该是形状为[1, C, H, W]或[C, H, W]的单个torch张量
            prompt: 使用的提示，字符串类型
            t: 使用的时间步，应为[0, 1000]范围内的整数
            up_ft_index: 要提取特征的U-Net上采样块，可以选择[0, 1, 2, 3]
            ensemble_size: 批处理中用于提取特征的重复图像数量
        返回:
            unet_ft: 形状为[1, c, h, w]的torch张量
        """
        img_tensor = img_tensor.repeat(
            ensemble_size, 1, 1, 1).cuda()  # 将图像张量重复ensemble_size次并移动到GPU上
        prompt_embeds = self.pipe._encode_prompt(
            prompt=prompt,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )  # 编码提示，生成[1, 77, dim]的张量
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)  # 重复提示嵌入
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds,
        )  # 使用管道处理图像张量
        unet_ft = unet_ft_all["up_ft"][up_ft_index]  # 获取指定上采样块的输出
        unet_ft = unet_ft.mean(0, keepdim=True)  # 计算平均值
        return unet_ft  # 返回U-Net特征
    # index 0: 1280, 24, 24
    # index 1: 1280, 48, 48
    # index 2: 640, 96, 96
    # index 3: 320, 96，96

    @torch.no_grad()
    def forward_feature_extractor(self, uc, img_tensor, t=261, up_ft_index=[0, 1, 2, 3], ensemble_size=8):
        """
        参数:
            img_tensor: 应该是形状为[1, C, H, W]或[C, H, W]的单个torch张量
            prompt: 使用的提示，字符串类型
            t: 使用的时间步，应为[0, 1000]范围内的整数
            up_ft_index: 要提取特征的U-Net上采样块，可以选择[0, 1, 2, 3]
            ensemble_size: 批处理中用于提取特征的重复图像数量
        返回:
            unet_ft: 形状为[1, c, h, w]的torch张量
        """
        batch_size = img_tensor.shape[0]  # 获取批量大小

        img_tensor = img_tensor.unsqueeze(1).repeat(1, ensemble_size, 1, 1, 1)  # 增加维度并重复图像张量
        img_tensor = img_tensor.reshape(-1, *img_tensor.shape[2:])  # 重新调整形状

        prompt_embeds = uc.repeat(
            img_tensor.shape[0], 1, 1).to(img_tensor.device)  # 重复并移动提示嵌入到图像张量设备上
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=up_ft_index,
            prompt_embeds=prompt_embeds,
        )  # 使用管道处理图像张量
        unet_ft = unet_ft_all["up_ft"]  # 获取上采样块的输出

        return unet_ft  # 返回U-Net特征


class FeatureExtractor(nn.Module):  # 定义FeatureExtractor类，继承自nn.Module
    def __init__(self,  # 初始化方法
                 sd_id='stabilityai/stable-diffusion-2-1',  # 稳定扩散模型的ID
                 load_local=False,  # 是否本地加载模型
                 if_softmax=False,  # 是否使用softmax
                 feature_index_cor=1,  # 特征索引（对应特征相关性）
                 feature_index_matting=4,  # 特征索引（对应特征遮蔽）
                 attention_res=32,  # 注意力分辨率
                 set_diag_to_one=True,  # 是否将对角线设置为1
                 time_steps=[0],  # 时间步长
                 extract_feature_inputted_to_layer=False,  # 是否提取输入到层的特征
                 ensemble_size=8):  # 集成大小
        super().__init__()  # 调用父类的初始化方法

        self.dift_sd = SDFeaturizer(sd_id=sd_id, load_local=load_local)  # 初始化SDFeaturizer实例
        self.register_buffer("prompt_embeds", self.dift_sd.pipe._encode_prompt(  # 注册缓冲区以存储提示嵌入
            prompt='',  # 空的提示
            num_images_per_prompt=1,  # 每个提示的图像数量
            do_classifier_free_guidance=False,  # 是否进行无分类指导
            device="cuda",  # 使用CUDA设备
        ))
        del self.dift_sd.pipe.tokenizer  # 删除tokenizer
        del self.dift_sd.pipe.text_encoder  # 删除text_encoder
        gc.collect()  # 进行垃圾回收
        torch.cuda.empty_cache()  # 清空CUDA缓存
        self.feature_index_cor = feature_index_cor  # 初始化feature_index_cor
        self.feature_index_matting = feature_index_matting  # 初始化feature_index_matting
        self.attention_res = attention_res  # 初始化attention_res
        self.set_diag_to_one = set_diag_to_one  # 初始化set_diag_to_one
        self.time_steps = time_steps  # 初始化time_steps
        self.extract_feature_inputted_to_layer = extract_feature_inputted_to_layer  # 初始化extract_feature_inputted_to_layer
        self.ensemble_size = ensemble_size  # 初始化ensemble_size
        self.register_attention_store(if_softmax=if_softmax, attention_res=attention_res)  # 注册注意力存储

    def register_attention_store(self, if_softmax=False, attention_res=[16, 32]):  # 注册注意力存储器的方法
        self.controller = AttentionStore(store_res=attention_res)  # 创建AttentionStore实例
        register_attention_control(self.dift_sd.pipe, self.controller, if_softmax=if_softmax, ensemble_size=self.ensemble_size)  # 注册注意力控制

    def get_trainable_params(self):  # 获取可训练参数的方法
        return []  # 返回一个空列表

    def get_reference_feature(self, images):  # 获取参考特征的方法
        self.controller.reset()  # 重置控制器
        batch_size = images.shape[0]  # 获取批量大小
        features = self.dift_sd.forward_feature_extractor(  # 提取特征
            self.prompt_embeds, images, t=self.time_steps[0], ensemble_size=self.ensemble_size)  # 提取特征

        features = self.ensemble_feature(features, self.feature_index_cor, batch_size)  # 合并特征
        
        return features.detach()  # 返回分离的特征

    def ensemble_feature(self, features, index, batch_size):  # 合并特征的方法
        if isinstance(index, int):  # 如果索引是整数
            features_ = features[index].reshape(  # 重塑特征
                batch_size, self.ensemble_size, *features[index].shape[1:])
            features_ = features_.mean(1, keepdim=False).detach()  # 求均值并分离
        else:  # 如果索引是列表
            index = list(index)  # 转换为列表
            res = ['24','48','96']  # 预定义的分辨率
            res = res[:len(index)]  # 截取合适长度的分辨率列表
            features_ = {}
            for i in range(len(index)):  # 遍历索引
                features_[res[i]] = features[index[i]].reshape(  # 重塑特征
                    batch_size, self.ensemble_size, *features[index[i]].shape[1:])
                features_[res[i]] = features_[res[i]].mean(1, keepdim=False).detach()  # 求均值并分离
        return features_

    def get_source_feature(self, images):  # 获取源特征的方法
        self.controller.reset()  # 重置控制器
        torch.cuda.empty_cache()  # 清空CUDA缓存
        batch_size = images.shape[0]  # 获取批量大小
        
        ft = self.dift_sd.forward_feature_extractor(  # 提取特征
            self.prompt_embeds, images, t=self.time_steps[0], ensemble_size=self.ensemble_size)  # 提取特征

        attention_maps = self.get_feature_attention(batch_size)  # 获取特征注意力图

        output = {"ft_cor": self.ensemble_feature(ft, self.feature_index_cor, batch_size),  # 合并特征并存储到输出字典
                  "attn": attention_maps, 'ft_matting': self.ensemble_feature(ft, self.feature_index_matting, batch_size)}  # 存储注意力图和遮蔽特征
        return output

    def get_feature_attention(self, batch_size):  # 获取特征注意力图的方法
        attention_maps = self.__aggregate_attention(  # 聚合注意力
            from_where=["down", "mid", "up"], is_cross=False, batch_size=batch_size)  # 指定从哪些层聚合注意力

        for attn_map in attention_maps.keys():  # 遍历注意力图
            attention_maps[attn_map] = attention_maps[attn_map].permute(0, 2, 1).reshape(  # 调整注意力图的形状
                (batch_size, -1, int(attn_map), int(attn_map)))  # 调整形状
            attention_maps[attn_map] = attention_maps[attn_map].permute(0, 2, 3, 1)  # 再次调整形状
        return attention_maps

    def __aggregate_attention(self, from_where: List[str], is_cross: bool, batch_size: int):  # 聚合注意力的方法
        out = {}  # 初始化输出字典
        self.controller.between_steps()  # 执行控制器的between_steps方法
        self.controller.cur_step=1  # 设置当前步长为1
        attention_maps = self.controller.get_average_attention()  # 获取平均注意力图
        for res in self.attention_res:  # 遍历注意力分辨率
            out[str(res)] = self.__aggregate_attention_single_res(  # 聚合单个分辨率的注意力
                from_where, is_cross, batch_size, res, attention_maps)  # 聚合单个分辨率的注意力
        return out
    
    def __aggregate_attention_single_res(self, from_where: List[str], is_cross: bool, batch_size: int, res: int, attention_maps):  # 聚合单个分辨率的注意力的方法
        out = []  # 初始化输出列表
        num_pixels = res ** 2  # 计算像素数量
        for location in from_where:  # 遍历指定的层
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:  # 遍历注意力图项
                if item.shape[1] == num_pixels:  # 如果项的形状符合条件
                    cross_maps = item.reshape(  # 重塑注意力图
                        batch_size, -1, res, res, item.shape[-1])
                    out.append(cross_maps)  # 添加到输出列表
        out = torch.cat(out, dim=1)  # 在维度1上拼接输出
        out = out.sum(1) / out.shape[1]  # 求和并求平均值
        out = out.reshape(batch_size, out.shape[-1], out.shape[-1])  # 重塑输出形状

        if self.set_diag_to_one:  # 如果设置对角线为1
            for o in out:  # 遍历输出
                o = o - torch.diag(torch.diag(o)) + \
                    torch.eye(o.shape[0]).to(o.device)  # 修改对角线元素
        return out  # 返回输出

    def __init__(self,
                 sd_id='stabilityai/stable-diffusion-2-1',
                 load_local=False,
                 if_softmax=False,
                 feature_index_cor=1,
                 feature_index_matting=4,
                 attention_res=32,
                 set_diag_to_one=True,
                 time_steps=[0],
                 extract_feature_inputted_to_layer=False,
                 ensemble_size=8):
        super().__init__()

        self.dift_sd = SDFeaturizer(sd_id=sd_id, load_local=load_local)  # 初始化SDFeaturizer实例
        # 注册缓冲区以存储提示嵌入
        self.register_buffer("prompt_embeds", self.dift_sd.pipe._encode_prompt(
            prompt='',
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            device="cuda",
        ))
        # 释放管道的tokenizer和text_encoder
        del self.dift_sd.pipe.tokenizer
        del self.dift_sd.pipe.text_encoder
        gc.collect()  # 进行垃圾回收
        torch.cuda.empty_cache()  # 清空CUDA缓存
        self.feature_index_cor = feature_index_cor
        self.feature_index_matting = feature_index_matting
        self.attention_res = attention_res
        self.set_diag_to_one = set_diag_to_one
        self.time_steps = time_steps
        self.extract_feature_inputted_to_layer = extract_feature_inputted_to_layer
        self.ensemble_size = ensemble_size
        self.register_attention_store(
            if_softmax=if_softmax, attention_res=attention_res)  # 注册注意力存储
    def register_attention_store(self, if_softmax=False, attention_res=[16, 32]):
        # 注册注意力存储器
        self.controller = AttentionStore(store_res=attention_res)

        # 注册注意力控制
        register_attention_control(
            self.dift_sd.pipe, self.controller, if_softmax=if_softmax, ensemble_size=self.ensemble_size)

    def get_trainable_params(self):
        # 获取可训练参数，这里返回一个空列表
        return []

    def get_reference_feature(self, images):
        # 获取参考特征
        self.controller.reset()
        batch_size = images.shape[0]
        # 提取特征
        features = self.dift_sd.forward_feature_extractor(
            self.prompt_embeds, images, t=self.time_steps[0], ensemble_size=self.ensemble_size) # b*e, c, h, w

        # 合并特征
        features = self.ensemble_feature(
            features, self.feature_index_cor, batch_size)
        
        return features.detach()

    def ensemble_feature(self, features, index, batch_size):
        # 合并特征
        if isinstance(index, int):
            # 单个特征索引
            features_ = features[index].reshape(
                batch_size, self.ensemble_size, *features[index].shape[1:])
            features_ = features_.mean(1, keepdim=False).detach()
        else:
            # 多个特征索引
            index = list(index)
            res = ['24','48','96']
            res = res[:len(index)]
            features_ = {}
            for i in range(len(index)):
                features_[res[i]] = features[index[i]].reshape(
                    batch_size, self.ensemble_size, *features[index[i]].shape[1:])
                features_[res[i]] = features_[res[i]].mean(1, keepdim=False).detach()
        return features_

    def get_source_feature(self, images):
        # 获取源特征
        self.controller.reset()
        torch.cuda.empty_cache()
        batch_size = images.shape[0]
        
        ft = self.dift_sd.forward_feature_extractor(
            self.prompt_embeds, images, t=self.time_steps[0], ensemble_size=self.ensemble_size) # b*e, c, h, w

        # 获取特征注意力图
        attention_maps = self.get_feature_attention(batch_size)

        output = {"ft_cor": self.ensemble_feature(ft, self.feature_index_cor, batch_size),
                  "attn": attention_maps, 'ft_matting': self.ensemble_feature(ft, self.feature_index_matting, batch_size)}
        return output

    def get_feature_attention(self, batch_size):
        # 获取特征注意力图
        attention_maps = self.__aggregate_attention(
            from_where=["down", "mid", "up"], is_cross=False, batch_size=batch_size)

        for attn_map in attention_maps.keys():
            attention_maps[attn_map] = attention_maps[attn_map].permute(0, 2, 1).reshape(
                (batch_size, -1, int(attn_map), int(attn_map)))  # [bs, h*w, h, w]
            attention_maps[attn_map] = attention_maps[attn_map].permute(0, 2, 3, 1)  # [bs, h, w, h*w]
        return attention_maps

    def __aggregate_attention(self, from_where: List[str], is_cross: bool, batch_size: int):
        # 聚合注意力
        out = {}
        self.controller.between_steps()
        self.controller.cur_step=1
        attention_maps = self.controller.get_average_attention()
        for res in self.attention_res:
            out[str(res)] = self.__aggregate_attention_single_res(
                from_where, is_cross, batch_size, res, attention_maps)
        return out
    
    def __aggregate_attention_single_res(self, from_where: List[str], is_cross: bool, batch_size: int, res: int, attention_maps):
        # 聚合单个分辨率的注意力
        out = []
        num_pixels = res ** 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(
                        batch_size, -1, res, res, item.shape[-1])
                    out.append(cross_maps)
        out = torch.cat(out, dim=1)
        out = out.sum(1) / out.shape[1]
        out = out.reshape(batch_size, out.shape[-1], out.shape[-1])

        if self.set_diag_to_one:
            for o in out:
                o = o - torch.diag(torch.diag(o)) + \
                    torch.eye(o.shape[0]).to(o.device)
        return out

```