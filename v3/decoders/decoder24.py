
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class FeatureReducer3(nn.Module):
    def __init__(self, in_channels, hidden_channels=256):
        super().__init__()
        self.pool = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.pool(x)
        return self.fc(x)  # (B, hidden_channels)


class WeightedConvPool3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3,), num_convs=4):
        super().__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convs = nn.ModuleList()

        for ks in kernel_sizes:
            for _ in range(num_convs):
                padding = ks // 2
                self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=padding))

        self.total_convs = len(self.convs)

        self.feature_reducer = FeatureReducer3(in_channels, hidden_channels=256)

        self.fc_weight = nn.Linear(256, self.num_convs * out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.feature_reducer(x)  # (B, hidden)
        weights = self.fc_weight(feat).view(B, self.num_convs, self.out_channels, 1, 1)  # (B, N, C, 1, 1)

        outputs = [conv(x) for conv in self.convs]  # list of (B, C, H, W)
        stacked = torch.stack(outputs, dim=1)  # (B, N, C, H, W)

        # weights = weights.view(B, self.total_convs, 1, 1, 1)
        out = (stacked * weights).sum(dim=1)
        return out


class FeatureReducer5(nn.Module):
    def __init__(self, in_channels, hidden_channels=256):
        super().__init__()
        self.pool = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.pool(x)
        return self.fc(x)  # (B, hidden_channels)


class WeightedConvPool5(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5,), num_convs=4):
        super().__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convs = nn.ModuleList()

        for ks in kernel_sizes:
            for _ in range(num_convs):
                padding = ks // 2
                self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=padding))

        self.total_convs = len(self.convs)

        self.feature_reducer = FeatureReducer5(in_channels, hidden_channels=256)

        self.fc_weight = nn.Linear(256, self.num_convs * out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.feature_reducer(x)  # (B, hidden)
        weights = self.fc_weight(feat).view(B, self.num_convs, self.out_channels, 1, 1)  # (B, N, C, 1, 1)

        outputs = [conv(x) for conv in self.convs]  # list of (B, C, H, W)
        stacked = torch.stack(outputs, dim=1)  # (B, N, C, H, W)

        # weights = weights.view(B, self.total_convs, 1, 1, 1)
        out = (stacked * weights).sum(dim=1)
        return out


class SubBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 定义操作池
        self.conv_candidates = nn.ModuleList([
            WeightedConvPool3(in_channels, out_channels),
            # WeightedConvPool5(in_channels, out_channels),
            ])

        self.act_candidates = nn.ModuleList([
            nn.LeakyReLU(0.2),
            nn.GELU(),
            nn.SiLU()
            ])

        self.norm_candidates = nn.ModuleList([
            nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.GroupNorm(32, out_channels),
            # nn.LayerNorm(),  ####新加入的
            ])

    def forward(self, x):
        conv_op = random.choice(self.conv_candidates)
        act_op = random.choice(self.act_candidates)
        norm_op = random.choice(self.norm_candidates)

        # 注意顺序随机但不注册 ModuleList
        ops = [conv_op, act_op, norm_op]
        random.shuffle(ops)

        for op in ops:
            x = op(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_subblocks=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subblocks = num_subblocks
        self.subblocks = nn.ModuleList([SubBlock(out_channels, out_channels) for _ in range(num_subblocks)])

        self.upsample_pixelshuffle = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2))        
        self.conv_up = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.upsample_methods = [
            'bilinear',
            'nearest',
            'bicubic',
            'pixelshuffle']

    def forward(self, x):
        x = self.conv_up(x)
        method = random.choice(self.upsample_methods)

        if method == 'bilinear':
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        elif method == 'nearest':
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        elif method == 'bicubic':
            x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        elif method == 'pixelshuffle':
            x = self.upsample_pixelshuffle(x)

        res_x = x

        num_active = random.randint(1, self.num_subblocks)
        for i, block in enumerate(self.subblocks[:num_active]):
            x = block(x)

        return x + res_x



# 512(32)-256(64)-128(128)-64(256)
class Decoder(nn.Module):
    def __init__(self, in_channels=256, base_channels=128, levels=3, num_subblocks=2):
        super().__init__()
        self.in_channels_list = [256, 128, 64]  # Corresponding to Feature 3, 2, 1, 0

        self.blocks = nn.ModuleList()
        for i in range(levels):
            in_channels = self.in_channels_list[i]
            out_channels = base_channels // (2 ** i)
            self.blocks.append(UpsampleBlock(in_channels, out_channels, num_subblocks))  #(32-64)256/128  (64-128)128/64  (128-256)64/32
            in_channels = out_channels

        self.final_conv = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)

    def forward(self, features):
        valid_indices = [0, 1, 3]
        start_idx = random.choice(valid_indices)
        # print(start_idx)
        if start_idx == 3:
            x = features[3]  # Feature 3: torch.Size([1, 256, 32, 32])
            blocks_to_use = self.blocks[0:]  # All 3 blocks
        elif start_idx == 1:
            x = features[1]  # Feature 1: torch.Size([1, 128, 64, 64])
            blocks_to_use = self.blocks[1:]  # Last 2 blocks
        elif start_idx == 0:
            x = features[0]  # Feature 0: torch.Size([1, 64, 128, 128])
            blocks_to_use = self.blocks[2:]  # Last 1 block
        else:
            raise ValueError("Invalid start index chosen.")

        for block in blocks_to_use:
            x = block(x)
        return self.final_conv(x)


if __name__ == "__main__":
    features = [
        torch.randn(1, 64, 128, 128),   # Feature 0
        torch.randn(1, 128, 64, 64),    # Feature 1
        torch.randn(1, 256, 32, 32),    # Feature 2 （如果需要）
        torch.randn(1, 256, 32, 32),    # Feature 3
    ]
    decoder = Decoder()
    output = decoder(features)
    print("Output shape:", output.shape)



    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Total parameters: {total_params:,}")


    from thop import profile, clever_format

    # 计算FLOPs和参数量
    flops, params = profile(decoder, inputs=(features,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, Params: {params}")