import torch
import torch.nn as nn
import torch.nn.functional as F

# Swish 激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            Swish(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = Swish()

    def forward(self, x):
        return self.activation(x + self.block(x))

# Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, level=3, max_channels=256):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.level = level
        self.max_channels = max_channels

        self.initial = nn.ModuleList([
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            Swish()
        ])

        self.blocks = nn.ModuleList()
        in_ch = base_channels
        for i in range(level):
            # 第一个 block 不扩展通道
            out_ch = in_ch if i == 0 else min(in_ch * 2, max_channels)
            
            block = nn.ModuleDict({
                'down': nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                'res': ResidualBlock(out_ch)
            })
            self.blocks.append(block)
            in_ch = out_ch

        # 最终块：不下采样，通道再扩展一次
        final_ch = min(in_ch * 2, max_channels)
        self.final_conv = nn.Conv2d(in_ch, final_ch, kernel_size=3, padding=1)
        self.final_res = ResidualBlock(final_ch)

    def forward(self, x):
        fea = []
        for layer in self.initial:
            x = layer(x)
        for block in self.blocks:
            x = block['down'](x)
            x = block['res'](x)
            fea.append(x)
        x = self.final_conv(x)
        x = self.final_res(x)
        fea.append(x)
        return fea

if __name__ == "__main__":
    # encoder = Encoder()
    # x = torch.randn(1, 3, 256, 256)
    # output = encoder(x)
    # print("All output shapes:")
    # for i, feat in enumerate(output):
    #     print(f"Feature {i}: {feat.shape}")

    
    # total_params = sum(p.numel() for p in encoder.parameters())
    # print(f"Total parameters: {total_params:,}")


    # from thop import profile, clever_format

    # # 计算FLOPs和参数量
    # flops, params = profile(encoder, inputs=(x,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(f"FLOPs: {flops}, Params: {params}")

    # from torchvision import transforms, models
    # model = models.resnet50()
    # model.fc = torch.nn.Linear(model.fc.in_features, 128)
    from denoiser import get_denoiser
    denoiser = get_denoiser(sigma=1).network

    x = torch.randn(1, 3, 256, 256)

    total_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Total parameters: {total_params:,}")


    from thop import profile, clever_format

    # 计算FLOPs和参数量
    flops, params = profile(denoiser, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Params: {params}")
