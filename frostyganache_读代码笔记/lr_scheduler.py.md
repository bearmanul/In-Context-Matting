# 余弦退火
$$
\eta_{t}=\eta_{min}^{i}+\frac{1}{2}(\eta^{i}_{max}-\eta_{min}^{i})\left( 1+\cos\left( \frac{T_{cur}}{T_{i}}\pi \right) \right)
$$

# 代码注释
```python
import numpy as np

class LambdaWarmUpCosineScheduler:
    """
    LambdaWarmUpCosineScheduler类：lambda预热余弦调度器
    """
    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        # 初始化函数
        self.lr_warm_up_steps = warm_up_steps  # 学习率预热步数
        self.lr_start = lr_start  # 初始学习率
        self.lr_min = lr_min  # 最小学习率
        self.lr_max = lr_max  # 最大学习率
        self.lr_max_decay_steps = max_decay_steps  # 最大衰减步数
        self.last_lr = 0.  # 上一次学习率
        self.verbosity_interval = verbosity_interval  # 输出间隔

    def schedule(self, n, **kwargs):
        # 调度函数
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: 
                print(f"当前步数: {n}, 最近的学习率乘数: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            # 在预热阶段
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            # 在余弦退火阶段
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

    def __call__(self, n, **kwargs):
        return self.schedule(n,**kwargs)


class LambdaWarmUpCosineScheduler2:
    """
    LambdaWarmUpCosineScheduler2类：支持重复迭代的lambda预热余弦调度器，可通过列表配置
    """
    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps  # 学习率预热步数列表
        self.f_start = f_start  # 初始学习率列表
        self.f_min = f_min  # 最小学习率列表
        self.f_max = f_max  # 最大学习率列表
        self.cycle_lengths = cycle_lengths  # 周期长度列表
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))  # 累积周期列表
        self.last_f = 0.  # 上一次学习率乘数
        self.verbosity_interval = verbosity_interval  # 输出间隔

    def find_in_interval(self, n):
        # 寻找所在周期
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        # 调度函数
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: 
                print(f"当前步数: {n}, 最近的学习率乘数: {self.last_f}, 当前周期 {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            # 在预热阶段
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            # 在余弦退火阶段
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (
                    1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    """
    LambdaLinearScheduler类：线性调度器
    """
    def schedule(self, n, **kwargs):
        # 调度函数
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: 
                print(f"当前步数: {n}, 最近的学习率乘数: {self.last_f}, 当前周期 {cycle}")

        if n < self.lr_warm_up_steps[cycle]:
            # 在预热阶段
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            # 在线性衰减阶段
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n) / (self.cycle_lengths[cycle])
            self.last_f = f
            return f

```