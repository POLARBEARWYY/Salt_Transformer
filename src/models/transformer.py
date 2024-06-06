import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义编码和解码机制，可以使用线性变换或其他复杂的方法进行实现
class SaltEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SaltEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        return self.encoder(x)

# 针对CIFAR-10数据集的图像数据，nn.Embedding 层不合适，因为它主要用于处理离散数据，如文本
# 图像数据通常直接通过卷积层进行处理。在这里，我们可以重新设计模型结构以适应图像数据，特别是CIFAR-10数据集
# 我们可以使用卷积层作为模型的前置处理单元，并在卷积层之后加入Transformer模块
# 我们将添加卷积层来处理图像数据，然后将卷积层的输出传递给Transformer层进行进一步处理
class ConvEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        return x
        
class SaltDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(SaltDecoder, self).__init__()
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        return self.decoder(x)


# 定义带Salt机制的Transformer模型
class TransformerWithSalt(nn.Module):
    def __init__(self, input_channels, model_dim, num_heads, num_layers, num_classes, salt_layer=None):
        super(TransformerWithSalt, self).__init__()
        self.salt_layer = salt_layer
        
        self.conv_encoder = ConvEncoder(input_channels, model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, num_classes)

        # 定义Salt Encoder和Decoder
        self.salt_encoder = SaltEncoder(model_dim, model_dim)
        self.salt_decoder = SaltDecoder(model_dim, model_dim)
        
    def forward(self, x):
        if self.salt_layer == 'input':
            x = self.salt_encoder(x)  # 在输入层进行编码

        x = self.conv_encoder(x)
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height * width).permute(0, 2, 1)
        
        x = self.transformer_encoder(x)
        
        if self.salt_layer == 'output':
            x = self.salt_decoder(x)  # 在输出层进行解码
        
        x = self.fc(x.mean(dim=1))
        return x

