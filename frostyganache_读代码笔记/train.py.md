
# 代码注释
```python
if __name__ == '__main__':
    # 导入所需的模块
    import datetime
    import argparse
    from omegaconf import OmegaConf

    import os
    
    from icm.util import instantiate_from_config
    import torch
    from pytorch_lightning import Trainer, seed_everything

    # 定义解析命令行参数的函数
    def parse_args():
        parser = argparse.ArgumentParser()

        # 添加实验名称参数
        parser.add_argument(
            "--experiment_name",
            type=str,
            default="in_context_matting",
        )
        # 添加调试模式参数
        parser.add_argument(
            "--debug",
            type=bool,
            default=False,
        )
        # 添加恢复训练参数
        parser.add_argument(
            "--resume",
            type=str,
            default="",
        )
        # 添加微调模式参数
        parser.add_argument(
            "--fine_tune",
            type=bool,
            default=False,
        )
        # 添加配置文件路径参数
        parser.add_argument(
            "--config",
            type=str,
            default="",
        )
        # 添加日志目录参数
        parser.add_argument(
            "--logdir",
            type=str,
            default="logs",
        )
        # 添加随机种子参数
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
        )

        # 解析命令行参数
        args = parser.parse_args()
        return args

    # 设置多进程启动方式为'spawn'
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    
    # 解析命令行参数
    args = parse_args()
    if args.resume:
        path = args.resume.split('checkpoints')[0]
        # 获取最新的版本文件夹
        all_folder = os.listdir(path)
        all_folder = [os.path.join(path, folder) for folder in all_folder if 'version' in folder]
        all_folder.sort()
        last_version_folder = all_folder[-1]
        # 获取hparams.yaml文件路径
        hparams_path = os.path.join(last_version_folder, 'hparams.yaml')
        cfg = OmegaConf.load(hparams_path)
    else:
        cfg = OmegaConf.load(args.config)

    if args.fine_tune:
        cfg_ft = OmegaConf.load(args.config)
        # 合并配置文件，cfg_ft将覆盖cfg
        cfg = OmegaConf.merge(cfg, cfg_ft)
        
    # 设置随机种子
    seed_everything(args.seed)

    """=== 初始化数据 ==="""
    cfg_data = cfg.get('data')

    # 实例化数据配置
    data = instantiate_from_config(cfg_data)

    """=== 初始化模型 ==="""
    cfg_model = cfg.get('model')

    # 实例化模型配置
    model = instantiate_from_config(cfg_model)

    """=== 初始化训练器 ==="""
    cfg_trainer = cfg.get('trainer')
    # 将omegaconf配置转换为字典
    cfg_trainer = OmegaConf.to_container(cfg_trainer)

    if args.debug:
        cfg_trainer['limit_train_batches'] = 2
        # cfg_trainer['log_every_n_steps'] = 1
        cfg_trainer['limit_val_batches'] = 3
        # cfg_trainer['overfit_batches'] = 2

    # 初始化日志记录器
    cfg_logger = cfg_trainer.pop('cfg_logger')

    if args.resume:
        name = args.resume.split('/')[-3]
    else:
        name = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S")+'-'+args.experiment_name
    cfg_logger['params']['save_dir'] = args.logdir
    cfg_logger['params']['name'] = name
    cfg_trainer['logger'] = instantiate_from_config(cfg_logger)

    # 插件配置
    cfg_plugin = cfg_trainer.pop('plugins')
    cfg_trainer['plugins'] = instantiate_from_config(cfg_plugin)
    
    # 初始化回调函数
    cfg_callbacks = cfg_trainer.pop('cfg_callbacks')
    callbacks = []
    for callback_name in cfg_callbacks:
        if (callback_name == 'modelcheckpoint'):
            cfg_callbacks[callback_name]['params']['dirpath'] = os.path.join(
                args.logdir, name, 'checkpoints')
        callbacks.append(instantiate_from_config(cfg_callbacks[callback_name]))
    cfg_trainer['callbacks'] = callbacks

    if args.resume and not args.fine_tune:
        cfg_trainer['resume_from_checkpoint'] = args.resume
    
    if args.fine_tune:
        # 加载模型权重
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['state_dict'], strict=False)
    # 初始化训练器
    trainer_opt = argparse.Namespace(**cfg_trainer)
    trainer = Trainer.from_argparse_args(trainer_opt)

    # 保存配置到日志
    trainer.logger.log_hyperparams(cfg)

    """=== 开始训练 ==="""

    trainer.fit(model, data)


```