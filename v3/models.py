import torch.nn as nn
from transformers import CLIPModel
import loralib as lora
import timm



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
            print(f"  正在加载 {self.args.model_name} (Timm 模式)...")

            self.backbone = timm.create_model(self.args.model_name, pretrained=True, num_classes=1)
            print(f"  ({self.args.model_name}) 准备就绪 (内置 num_classes=1)。")


        print("基线模型 (Baseline Model) 初始化完毕。")

    def forward(self, x):
        if 'clip' in self.args.model_name:
            feat = self.backbone(x)['pooler_output']
            output = self.classification_head(feat)

        else:
            output = self.backbone(x)

        return output

