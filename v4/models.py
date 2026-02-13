import torch.nn as nn
from transformers import CLIPModel
import loralib as lora
import timm
import torch


def replace_with_lora(module, rank, lora_dropout=0):
    """ 用 lora.Linear 层替换 nn.Linear 层并复制权重。 """
    lora_layer = lora.Linear(
        module.in_features,
        module.out_features,
        r=rank,
        lora_alpha=rank,
        lora_dropout=lora_dropout,
        bias=(module.bias is not None)
    )
    lora_layer.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        lora_layer.bias.data.copy_(module.bias.data)
    return lora_layer


class BaselineModel(nn.Module):
    def __init__(self, args):
        super(BaselineModel, self).__init__()

        print(f"初始化基线模型 (Baseline Model):")

        self.backbone = None
        self.args = args
        self.classification_head = None  # 仅 ViT 需要
        self.fft_backbone = None  # FFT流的backbone
        self.fusion_classifier = None  # 融合后的分类器

        if 'clip' in self.args.model_name:
            print(f"  正在加载 {self.args.model_name} (CLIP ViT 模式)...")

            clip_model = CLIPModel.from_pretrained(self.args.model_name)
            self.backbone = clip_model.vision_model
            feat_dim = clip_model.config.vision_config.hidden_size

            print(f"  应用 LoRA (秩={self.args.lora_rank})...")
            for layer in self.backbone.encoder.layers:
                attn = layer.self_attn
                attn.q_proj = replace_with_lora(attn.q_proj, self.args.lora_rank, self.args.lora_dropout)
                attn.k_proj = replace_with_lora(attn.k_proj, self.args.lora_rank, self.args.lora_dropout)
                attn.v_proj = replace_with_lora(attn.v_proj, self.args.lora_rank, self.args.lora_dropout)
                attn.out_proj = replace_with_lora(attn.out_proj, self.args.lora_rank, self.args.lora_dropout)

            lora.mark_only_lora_as_trainable(self.backbone, bias='lora_only')
            print(f"  CLIP ViT (dim={feat_dim}) + LoRA 准备就绪。")

            # --- 4a. ViT 分类头 ---
            print(f"  创建分类头 (输入 dim={feat_dim})")
            self.classification_head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(256, 1)
            )

            # 确保头部权重是可训练的
            for param in self.classification_head.parameters():
                param.requires_grad = True


        else:
            print(f"  正在加载 {self.args.model_name} (Timm 双流模式)...")
            # RGB流
            self.backbone = timm.create_model(self.args.model_name,pretrained=True,num_classes=0)

            # FFT流 - 共享权重或独立权重
            self.fft_backbone = timm.create_model(self.args.model_name,pretrained=True,num_classes=0)

            # 获取特征维度
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                feat = self.backbone(dummy_input)
                feat_dim = feat.shape[1]  # [B, feat_dim]

            print(f"  特征维度: {feat_dim}")

            # 融合分类器 - 输入是两个特征向量相加后的结果
            self.fusion_classifier = nn.Sequential(
                nn.Linear(feat_dim*2, 256),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(256, 1)
            )

            print(f"  ({self.args.model_name}) 双流架构准备就绪 (RGB + FFT频域)。")

        print("基线模型 (Baseline Model) 初始化完毕。")

    @torch.no_grad()
    def compute_fft_magnitude(self, x):
        fft = torch.fft.fft2(x, dim=(-2, -1))
        magnitude = torch.abs(fft)
        magnitude = torch.log1p(magnitude)
        return magnitude

    def forward(self, x):
        if 'clip' in self.args.model_name:
            feat = self.backbone(x)['pooler_output']
            output = self.classification_head(feat)

        else:
            rgb_feat = self.backbone(x) # RGB流 - 输出 [B, feat_dim]

            fft_input = self.compute_fft_magnitude(x) # FFT流 - 输出 [B, feat_dim]
            fft_feat = self.fft_backbone(fft_input) # 特征concat融合 - [B, feat_dim * 2]

            fused_feat = torch.cat([rgb_feat, fft_feat], dim=1)
            output = self.fusion_classifier(fused_feat)

        return output

