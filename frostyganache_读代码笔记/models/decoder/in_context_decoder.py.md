

```python
from torch import nn
from icm.util import instantiate_from_config

class InContextDecoder(nn.Module):
    '''
    InContextDecoder 是 InContextMatting 的解码器。

    in-context 解码器：

    get_trainable_params(): 返回可训练的参数列表

    forward(source, reference): 前向传播函数
        reference = {'feature': feature_of_reference_image,
                     'guidance': guidance_on_reference_image}

        source = {'feature': feature_of_source_image, 'image': source_images}

    '''

    def __init__(self,
                 cfg_detail_decoder,
                 cfg_in_context_fusion,
                 freeze_in_context_fusion=False,
                 ):
        super().__init__()

        # 实例化 in-context 融合模块和详细解码器
        self.in_context_fusion = instantiate_from_config(
            cfg_in_context_fusion)
        self.detail_decoder = instantiate_from_config(cfg_detail_decoder)

        # 冻结 in-context 融合模块的参数
        self.freeze_in_context_fusion = freeze_in_context_fusion
        if freeze_in_context_fusion:
            self.__freeze_in_context_fusion()

    def forward(self, source, reference):
        # 获取参考图像的特征和引导信息
        feature_of_reference_image = reference['feature']
        guidance_on_reference_image = reference['guidance']

        # 获取源图像的特征和图像
        feature_of_source_image = source['feature']
        source_images = source['image']

        # 在 in-context 融合模块中进行特征融合
        features = self.in_context_fusion(
            feature_of_reference_image, feature_of_source_image, guidance_on_reference_image)

        # 在详细解码器中生成输出
        output = self.detail_decoder(features, source_images)

        return output, features['mask'], features['trimap']

    def get_trainable_params(self):
        params = []
        # 将详细解码器的参数添加到可训练参数列表中
        params = params + list(self.detail_decoder.parameters())
        # 如果未冻结 in-context 融合模块，则将其参数添加到可训练参数列表中
        if not self.freeze_in_context_fusion:
            params = params + list(self.in_context_fusion.parameters())
        return params

    def __freeze_in_context_fusion(self):
        # 冻结 in-context 融合模块的参数
        for param in self.in_context_fusion.parameters():
            param.requires_grad = False

```