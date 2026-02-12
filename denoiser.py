import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingFCNWithSkip(nn.Module):
    """
    带跳跃连接的去噪网络，使用GroupNorm和GELU激活
    """

    def __init__(self, in_channels=3, out_channels=3, features=64, num_blocks=8):
        super(DenoisingFCNWithSkip, self).__init__()

        self.num_blocks = num_blocks

        # 输入层
        self.input_conv = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.input_gn = nn.GroupNorm(num_groups=16, num_channels=features)

        # 残差块
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(ResBlock(features, groups=16))

        # 输出层
        self.output_conv = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 输入层
        out = F.gelu(self.input_gn(self.input_conv(x)))

        # 残差块
        for block in self.res_blocks:
            out = block(out)

        # 输出预测的噪声
        predicted_noise = self.output_conv(out)

        return predicted_noise


class ResBlock(nn.Module):
    """残差块，使用GroupNorm和GELU激活"""

    def __init__(self, features, groups=16):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=groups, num_channels=features)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=groups, num_channels=features)

    def forward(self, x):
        residual = x
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += residual
        return F.gelu(out)


# 使用示例
if __name__ == "__main__":
    model = DenoisingFCNWithSkip(in_channels=3, out_channels=3, features=128, num_blocks=8)

    # 打印模型信息
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    batch_size = 4
    height, width = 256, 256
    channels = 3

    noisy_image = torch.randn(batch_size, channels, height, width)

    model.eval()
    predicted_noise = model(noisy_image)

    print(f"输入形状: {noisy_image.shape}")
    print(f"预测噪声形状: {predicted_noise.shape}")

