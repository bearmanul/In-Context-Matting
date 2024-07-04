>库函数说明见笔记1 

>定义了一些函数用于计算损失和梯度。
## compute_mse_loss均方误差损失
```py
def compute_mse_loss(pred, target, trimap):
    """计算均方误差损失"""
    error_map = (pred - target) / 255.0  # 计算预测值和目标值的误差图像
    loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)  # 计算损失
    return loss  # 返回损失值
```
## compute_sad_loss平均绝对误差损失
>&emsp;&emsp;根据预测值、目标值和修剪图，计算预测值和目标值之间的平均绝对误差。返回损失值和前景像素数目。
```py
def compute_sad_loss(pred, target, trimap):
    """计算平均绝对误差损失"""
    error_map = np.abs((pred - target) / 255.0)  # 计算预测值和目标值的绝对误差图像
    loss = np.sum(error_map * (trimap == 128))  # 计算损失
    # 返回损失值和前景像素数目
    return loss / 1000, np.sum(trimap == 128) / 1000
```
## gauss高斯函数
>&emsp;&emsp;给定输入值x和标准差sigma，计算高斯函数的值。
```py
def gauss(x, sigma):
    """高斯函数"""
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y
```

## dgauss高斯函数的导数
>&emsp;&emsp;给定输入值x和标准差sigma，计算高斯函数的导数。
```py
def dgauss(x, sigma):
    """高斯函数的导数"""
    y = -x * gauss(x, sigma) / (sigma ** 2)
    return y
```

## gaussgradient图像的高斯梯度
>&emsp;&emsp;给定图像im和标准差sigma，计算图像的高斯梯度。返回x方向和y方向的梯度。
```py
def gaussgradient(im, sigma):
    """计算图像的高斯梯度"""
    epsilon = 1e-2
    # 计算卷积核的半径大小
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(int)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    
    # 生成x方向上的高斯梯度卷积核
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()

    # 对图像进行卷积操作，计算x和y方向上的高斯梯度
    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')

    return gx, gy
```
## compute_gradient_loss梯度损失
```py
def compute_gradient_loss(pred, target, trimap):
    """
    计算梯度损失
    
    参数:
    - pred: 预测值图像
    - target: 目标值图像
    - trimap: 修剪图像
    
    返回值:
    - loss: 梯度损失值
    """
    pred = pred / 255.0  # 归一化预测值图像
    target = target / 255.0  # 归一化目标值图像

    pred_x, pred_y = gaussgradient(pred, 1.4)  
    # 计算预测值图像的x和y方向的高斯梯度
    target_x, target_y = gaussgradient(target, 1.4)  
    # 计算目标值图像的x和y方向的高斯梯度

    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)  
    # 计算预测值图像的梯度幅值
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)  
    # 计算目标值图像的梯度幅值

    error_map = (pred_amp - target_amp) ** 2  
    # 计算梯度幅值之间的差异，并平方得到误差图像
    loss = np.sum(error_map[trimap == 128])  
    # 根据修剪图像中像素值为128的区域，计算误差图像中对应区域的误差值之和

    return loss / 1000.  # 返回归一化后的梯度损失值
```
## compute_connectivity_error
>&emsp;&emsp;计算连通性误差。在图像分割任务中，连通性误差用于衡量预测的透明度图像与目标透明度图像之间的差异。
```py
def compute_connectivity_error(pred, target, trimap, step):
    """
    计算连通性误差
    
    参数:
    - pred: 预测值图像
    - target: 目标值图像
    - trimap: 修剪图像
    - step: 阈值步长
    
    返回值:
    - loss: 连通性误差值
    """
    pred = pred / 255.0  # 归一化预测值图像
    target = target / 255.0  # 归一化目标值图像
    h, w = pred.shape  # 获取图像的高度和宽度

    thresh_steps = list(np.arange(0, 1 + step, step))  
    # 根据步长生成阈值列表
    l_map = np.ones_like(pred, dtype=float) * -1  
    # 创建与预测值图像相同大小的映射图，并初始化为-1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(int)  
        # 根据阈值生成预测值的二值图像
        target_alpha_thresh = (target >= thresh_steps[i]).astype(int)  
        # 根据阈值生成目标值的二值图像

        omega = getLargestCC(pred_alpha_thresh * target_alpha_thresh).astype(int)  
        # 获取预测值和目标值二值图像的最大连通分量
        flag = ((l_map == -1) & (omega == 0)).astype(int)  
        # 根据映射图和连通分量计算标志图像
        l_map[flag == 1] = thresh_steps[i - 1]  
        # 更新映射图的值

    l_map[l_map == -1] = 1  # 将映射图中剩余的-1值设为1

    pred_d = pred - l_map  # 计算预测值图像与映射图的差异
    target_d = target - l_map  # 计算目标值图像与映射图的差异
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(int)  
    # 根据差异计算预测值的相对透明度
    target_phi = 1 - target_d * (target_d >= 0.15).astype(int)  
    # 根据差异计算目标值的相对透明度
    loss = np.sum(np.abs(pred_phi - target_phi)[trimap == 128])  
    # 根据修剪图像计算相对透明度之间的误差值之和

    return loss / 1000.  # 返回归一化后的连通性误差值
```