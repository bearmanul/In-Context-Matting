# SAD, MSE, Grad, Conn
[SAD, MSE, Grad, Conn](SAD,%20MSE,%20Grad,%20Conn.md)
# 代码注释
```python
import scipy.ndimage  # 导入SciPy库中的图像处理模块
import numpy as np  # 导入NumPy库
from skimage.measure import label  # 从scikit-image库中导入标签测量函数
import scipy.ndimage.morphology  # 导入SciPy库中的形态学处理模块
import torch  # 导入PyTorch库

def compute_mse_loss(pred, target, trimap):
    """计算均方误差损失"""
    error_map = (pred - target) / 255.0  # 计算预测值和目标值的误差图像
    loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)  # 计算损失

    return loss  # 返回损失值

def compute_sad_loss(pred, target, trimap):
    """计算平均绝对误差损失"""
    error_map = np.abs((pred - target) / 255.0)  # 计算预测值和目标值的绝对误差图像
    loss = np.sum(error_map * (trimap == 128))  # 计算损失
    # 返回损失值和前景像素数目
    return loss / 1000, np.sum(trimap == 128) / 1000
```

## 高斯核

```python
def gauss(x, sigma):
    """高斯函数"""
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y

def dgauss(x, sigma):
    """高斯函数的导数"""
    y = -x * gauss(x, sigma) / (sigma ** 2)
    return y

def gaussgradient(im, sigma):
    """计算图像的高斯梯度"""
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(int)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()

    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')

    return gx, gy
```

高斯核的一维形式可以表示为：

$$
G (x) = \frac{1} {{\sqrt{2\pi}\sigma}} e^{-\frac{x^2}{2\sigma^2}}
$$

其中，$x$ 是距离中心的偏移量，$\sigma$ 是高斯核的标准差，控制着高斯分布的宽度。当 $\sigma$ 较大时，高斯核的分布越宽，平滑效果越明显；当 $\sigma$ 较小时，分布越窄，平滑效果越局部化。

高斯核的 n 维形式是一个多维的高斯分布函数，通常表示为 $N(\mu, \Sigma)$，其中 $\mu$ 是多维高斯分布的均值向量， $\Sigma$ 是协方差矩阵。

对于 n 维输入 \( x \)，多维高斯核的概率密度函数（Probability Density Function，PDF）可以表示为：

$$ f (x) = \frac{1} {{\sqrt{{(2\pi)^n |\Sigma|}} }} \exp\left (-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right) $$

其中，$|\Sigma|$ 表示协方差矩阵 $\Sigma$ 的行列式，$\Sigma^{-1}$ 是协方差矩阵的逆。


```python
def compute_gradient_loss(pred, target, trimap):
    """计算梯度损失"""
    pred = pred / 255.0
    target = target / 255.0

    pred_x, pred_y = gaussgradient(pred, 1.4)
    target_x, target_y = gaussgradient(target, 1.4)

    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)

    error_map = (pred_amp - target_amp) ** 2
    loss = np.sum(error_map[trimap == 128])

    return loss / 1000.

def compute_connectivity_error(pred, target, trimap, step):
    """计算连通性误差"""
    pred = pred / 255.0
    target = target / 255.0
    h, w = pred.shape

    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones_like(pred, dtype=float) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(int)
        target_alpha_thresh = (target >= thresh_steps[i]).astype(int)

        omega = getLargestCC(pred_alpha_thresh * target_alpha_thresh).astype(int)
        flag = ((l_map == -1) & (omega == 0)).astype(int)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(int)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(int)
    loss = np.sum(np.abs(pred_phi - target_phi)[trimap == 128])

    return loss / 1000.

def getLargestCC(segmentation):
    """获取最大连通区域"""
    labels = label(segmentation, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC

def compute_mse_loss_torch(pred, target, trimap):
    """使用PyTorch计算均方误差损失"""
    error_map = (pred - target) / 255.0
    # 使用torch重写损失
    # loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)
    loss = torch.sum((error_map ** 2) * (trimap == 128).float()) / (torch.sum(trimap == 128).float() + 1e-8)

    return loss

def compute_sad_loss_torch(pred, target, trimap):
    """使用PyTorch重写计算平均绝对误差损失"""
    # 使用torch重写误差图
    # error_map = np.abs((pred - target) / 255.0)
    error_map = torch.abs((pred - target) / 255.0)
    # loss = np.sum(error_map * (trimap == 128))
    loss = torch.sum(error_map * (trimap == 128).float())

    return loss / 1000

```