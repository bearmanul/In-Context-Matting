
# 代码注释
```python
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR

# 导入计算损失的函数
from icm.criterion.matting_criterion_eval import compute_mse_loss_torch, compute_sad_loss_torch
# 导入实例化配置的工具函数
from icm.util import instantiate_from_config
from pytorch_lightning.utilities import rank_zero_only
import os
import cv2

class InContextMatting(pl.LightningModule):
    '''
    In Context Matting 模型
    包含一个特征提取器和一个上下文解码器
    使用学习率、调度器和损失函数来训练模型
    '''

    def __init__(
        self,
        cfg_feature_extractor,
        cfg_in_context_decoder,
        cfg_loss_function,
        learning_rate,
        cfg_scheduler=None,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 从配置中实例化特征提取器
        self.feature_extractor = instantiate_from_config(cfg_feature_extractor)
        # 从配置中实例化上下文解码器
        self.in_context_decoder = instantiate_from_config(cfg_in_context_decoder)
        # 从配置中实例化损失函数
        self.loss_function = instantiate_from_config(cfg_loss_function)
        # 设置学习率
        self.learning_rate = learning_rate
        # 设置调度器配置
        self.cfg_scheduler = cfg_scheduler

    def forward(self, reference_images, guidance_on_reference_image, source_images):
        # 获取参考图像的特征
        feature_of_reference_image = self.feature_extractor.get_reference_feature(reference_images)
        # 获取源图像的特征
        feature_of_source_image = self.feature_extractor.get_source_feature(source_images)
        
        # 组合参考图像特征和指导信息
        reference = {'feature': feature_of_reference_image, 'guidance': guidance_on_reference_image}
        # 组合源图像特征和图像
        source = {'feature': feature_of_source_image, 'image': source_images}

        # 通过上下文解码器获取输出和注意力图
        output, cross_map, self_map = self.in_context_decoder(source, reference)

        # 返回输出和注意力图
        return output, cross_map, self_map

    def on_train_epoch_start(self):
        # 在训练epoch开始时记录当前epoch
        self.log("epoch", self.current_epoch, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        # 共享步骤计算损失
        loss_dict, loss, _, _, _ = self.__shared_step(batch)

        # 记录训练损失
        self.__log_loss(loss_dict, loss, "train")

        # 记录学习率
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # 共享步骤计算验证损失
        loss_dict, loss, preds, cross_map, self_map = self.__shared_step(batch)

        # 记录验证损失
        self.__log_loss(loss_dict, loss, "val")
        # 添加注意力图到batch
        batch['cross_map'] = cross_map
        batch['self_map'] = self_map
        return preds, batch

    def __shared_step(self, batch):
        # 获取batch中的图像和标签
        reference_images, guidance_on_reference_image, source_images, labels, trimaps = batch["reference_image"], batch["guidance_on_reference_image"], batch["source_image"], batch["alpha"], batch["trimap"]

        # 前向传播获取输出和注意力图
        outputs, cross_map, self_map = self(reference_images, guidance_on_reference_image, source_images)
        
        # 创建样本图
        sample_map = torch.zeros_like(trimaps)
        sample_map[trimaps==0.5] = 1     
        
        # 计算损失
        loss_dict = self.loss_function(sample_map, outputs, labels)

        # 汇总损失
        loss = sum(loss_dict.values())
        if loss > 1e4 or torch.isnan(loss):
            raise ValueError(f"Loss explosion: {loss}")
        return loss_dict, loss, outputs, cross_map, self_map

    def __log_loss(self, loss_dict, loss, prefix):
        # 记录损失
        loss_dict = {f"{prefix}/{key}": loss_dict.get(key) for key in loss_dict}
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_step_end(self, outputs):
        # 验证步骤结束时处理输出
        preds, batch = outputs
        h, w = batch['alpha_shape']
        
        # 获取cross_map和self_map
        cross_map = batch['cross_map']
        self_map = batch['self_map']
        # 将注意力图resize到和preds相同的大小
        cross_map = torch.nn.functional.interpolate(cross_map, size=preds.shape[2:], mode='bilinear', align_corners=False)
        self_map = torch.nn.functional.interpolate(self_map, size=preds.shape[2:], mode='bilinear', align_corners=False)
        
        # 归一化注意力图
        cross_map = (cross_map - cross_map.min()) / (cross_map.max() - cross_map.min())
        self_map = (self_map - self_map.min()) / (self_map.max() - self_map.min())
        
        # 将注意力图转换为0-255的像素值
        cross_map = cross_map[0].squeeze()*255.0
        self_map = self_map[0].squeeze()*255.0
        
        # 获取batch中的一个样本
        pred = preds[0].squeeze()*255.0
        source_image = batch['source_image'][0]
        label = batch["alpha"][0].squeeze()*255.0
        trimap = batch["trimap"][0].squeeze()*255.0
        trimap[trimap == 127.5] = 128
        reference_image = batch["reference_image"][0]
        guidance_on_reference_image = batch["guidance_on_reference_image"][0]
        dataset_name = batch["dataset_name"][0]
        image_name = batch["image_name"][0].split('.')[0]

        # 如果设置了val_save_path，保存预测结果
        if hasattr(self, 'val_save_path'):
            os.makedirs(self.val_save_path, exist_ok=True)
            pred_ = torch.nn.functional.interpolate(pred.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
            pred_ = pred_.squeeze().cpu().numpy()
            pred_ = pred_.astype('uint8')
            cv2.imwrite(os.path.join(self.val_save_path, image_name+'.png'), pred_)
            
        # 生成带有指导信息的参考图像
        masked_reference_image = reference_image*guidance_on_reference_image

        # 记录图像
        self.__log_image(source_image, masked_reference_image, pred, label, dataset_name, image_name, prefix='val', self_map=self_map, cross_map=cross_map)

    def __compute_and_log_mse_sad_of_one_sample(self, pred, label, trimap, prefix="val"):
        # 计算未知区域像素的MSE和SAD损失
        mse_loss_unknown_ = compute_mse_loss_torch(pred, label, trimap)
        sad_loss_unknown_ = compute_sad_loss_torch(pred, label, trimap)

        # 计算所有像素的MSE和SAD损失
        trimap = torch.ones_like(label)*128
        mse_loss_all_ = compute_mse_loss_torch(pred, label, trimap)
        sad_loss_all_ = compute_sad_loss_torch(pred, label, trimap)

        # 记录损失
        metrics_unknown = {f'{prefix}/mse_unknown': mse_loss_unknown_, f'{prefix}/sad_unknown': sad_loss_unknown_,}
        metrics_all = {f'{prefix}/mse_all': mse_loss_all_, f'{prefix}/sad_all': sad_loss_all_,}

        self.log_dict(metrics_unknown, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log_dict(metrics_all, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def __log_image(self, source_image, masked_reference_image, pred, label, dataset_name, image_name, prefix='val', self_map=None, cross_map=None):
        # 还原图像的归一化
        source_image = self.__revert_normalize(source_image)
        masked_reference_image = self.__revert_normalize(masked_reference_image)
        # 处理预测结果和标签
        pred = torch.stack((pred/255.0,)*3, axis=-1)
        label = torch.stack((label/255.0,)*3, axis=-1)
        self_map = torch.stack((self_map/255.0,)*3, axis=-1)
        cross_map = torch.stack((cross_map/255.0,)*3, axis=-1)
        
        # 将图像拼接在一起
        image_for_log = torch.stack((source_image, masked_reference_image, label, pred, self_map, cross_map), axis=0)

        # 记录图像
        self.logger.experiment.add_images(f'{prefix}-{dataset_name}/{image_name}', image_for_log, self.current_epoch, dataformats='NHWC')

    def __revert_normalize(self, image):
        # 将图像从[C, H, W]转换为[H, W, C]
        image = image.permute(1, 2, 0)
        # 反归一化
        image = image * torch.tensor([0.229, 0.224, 0.225], device=self.device) + torch.tensor([0.485, 0.456, 0.406], device=self.device)
        image = torch.clamp(image, 0, 1)
        return image

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # 在每个验证batch结束后清空显存缓存
        torch.cuda.empty_cache()
        
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # 在每个训练batch结束后清空显存缓存
        torch.cuda.empty_cache()
        
    def test_step(self, batch, batch_idx):
        # 测试步骤
        loss_dict, loss, preds = self.__shared_step(batch)
        return loss_dict, loss, preds

    def configure_optimizers(self):
        # 配置优化器
        params = self.__get_trainable_params()
        opt = torch.optim.Adam(params, lr=self.learning_rate)

        if self.cfg_scheduler is not None:
            scheduler = self.__get_scheduler(opt)
            return [opt], scheduler
        return opt

    def __get_trainable_params(self):
        # 获取可训练参数
        params = []
        params = params + self.in_context_decoder.get_trainable_params() + self.feature_extractor.get_trainable_params()
        return params

    def __get_scheduler(self, opt):
        # 获取学习率调度器
        scheduler = instantiate_from_config(self.cfg_scheduler)
        scheduler = [
            {
                "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                "interval": "step",
                "frequency": 1,
            }
        ]
        return scheduler

from pytorch_lightning.callbacks import ModelCheckpoint

class ModifiedModelCheckpoint(ModelCheckpoint):
    def delete_frozen_params(self, ckpt):
        # 删除冻结参数
        for k in list(ckpt["state_dict"].keys()):
            if "feature_extractor" in k:
                del ckpt["state_dict"][k]
        return ckpt

    def _save_model(self, trainer: "pl.Trainer", filepath: str) -> None:
        # 保存模型
        super()._save_model(trainer, filepath)

        if trainer.is_global_zero:
            # 删除冻结参数后保存
            ckpt = torch.load(filepath)
            ckpt = self.delete_frozen_params(ckpt)
            torch.save(ckpt, filepath)

```