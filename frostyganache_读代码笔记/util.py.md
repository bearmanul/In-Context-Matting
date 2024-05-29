
# 代码注释
```python
import importlib
import torch

def instantiate_from_config(config):
    # 从配置实例化对象
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    # 从字符串中获取对象
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_feature_extractor(cfg):
    # 实例化特征提取器
    model = instantiate_from_config(cfg)
    load_odise_params = cfg.get("load_odise_params", False)
    if load_odise_params:
        # 加载参数
        params = torch.load('ckpt/odise_label_coco_50e-b67d2efc.pth')
        # 用"backbone.feature_extractor."前缀收集参数
        params = {k.replace("backbone.feature_extractor.", ""): v for k, v in params['model'].items() if "backbone.feature_extractor." in k}
        model.load_state_dict(params, strict=False)
        '''
        alpha_cond
        alpha_cond_time_embed
        clip_project.positional_embedding
        clip_project.linear.weight
        clip_project.linear.bias
        time_embed_project.positional_embedding
        time_embed_project.linear.weight
        time_embed_project.linear.bias
        '''
    model.eval()  # 将模型设置为评估模式
    return model

```