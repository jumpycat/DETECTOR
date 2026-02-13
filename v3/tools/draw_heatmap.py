import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import argparse
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm

from data_tools import get_transforms, create_dataset
from models import BaselineModel
from denoiser import DenoisingFCNWithSkip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 创建一个临时的 logger 对象（用于兼容 create_dataset）
class SimpleLogger:
    def info(self, msg):
        print(msg)

class GradCAM:
    """Grad-CAM 实现用于可视化模型关注区域"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子
        # 在 __init__ 方法中，将第 32-33 行改为：
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)  # 改为 register_full_backward_hook

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        """
        生成类激活图
        Args:
            input_image: 输入图像 tensor [1, C, H, W]
            target_class: 目标类别（None表示使用预测类别）
        Returns:
            cam: 热力图 numpy array [H, W]
        """
        # 前向传播
        model_output = self.model(input_image)

        if target_class is None:
            target_class = model_output.squeeze()

        # 反向传播
        self.model.zero_grad()
        target_class.backward(retain_graph=True)

        # 获取梯度和激活
        gradients = self.gradients  # [1, C, H', W']
        activations = self.activations  # [1, C, H', W']

        # 全局平均池化梯度
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # 加权求和
        cam = torch.sum(weights * activations, dim=1).squeeze()  # [H', W']

        # ReLU
        cam = torch.relu(cam)

        # 归一化到 [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def get_target_layer(model):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise ValueError("没有找到Conv2d层")


def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    将热力图叠加到原图上
    Args:
        image: 原始图像 numpy array [H, W, 3] (RGB, 0-255)
        heatmap: 热力图 numpy array [H, W] (0-1)
        alpha: 叠加透明度
        colormap: OpenCV 颜色映射
    Returns:
        叠加后的图像 [H, W, 3]
    """
    # 将热力图调整到与图像相同大小
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # 转换为彩色热力图
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # 叠加
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlayed


def denormalize_image(tensor):
    """
    反归一化图像用于显示
    Args:
        tensor: 图像 tensor [C, H, W]
        model_name: 模型名称，用于判断归一化方式
    Returns:
        反归一化后的 tensor，范围 [0, 1]
    """
    tensor = tensor.clone()
    # 其他模型（EfficientNet等）使用 -1~1 范围
    # 从 [-1, 1] 转换到 [0, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)

    return tensor


def visualize_samples(model, dataloader, args, output_dir, noiser=None, max_samples=50):
    """
    可视化数据集样本的热力图
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        args: 参数
        output_dir: 输出目录
        noiser: 噪声残差网络（可选）
        max_samples: 最大可视化样本数
    """
    model.eval()

    # 创建输出目录
    output_dir = Path(output_dir)
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # 获取目标层并创建 Grad-CAM
    target_layer = get_target_layer(model)
    grad_cam = GradCAM(model, target_layer)

    print(f"开始生成热力图，最多 {max_samples} 个样本...")
    print(f"输出目录: {output_dir}")

    real_count = 0
    fake_count = 0

    for batch_idx, (imgs, labels, video_names, frame_indices) in enumerate(dataloader):
        if real_count >= max_samples // 2 and fake_count >= max_samples // 2:
            break

        imgs = imgs.float().to(device)
        labels = labels.to(device)

        # 如果使用 noiser，先处理图像
        processed_imgs = imgs.clone()
        if args.use_noiser and noiser is not None:
            res = noiser(processed_imgs)
            processed_imgs = res - res.floor()

        # 逐个样本处理
        for i in range(imgs.size(0)):
            label = labels[i].item()

            # 根据标签决定是否继续
            if label == 1 and real_count >= max_samples // 2:
                continue
            if label == 0 and fake_count >= max_samples // 2:
                continue

            # 准备单个样本
            img_tensor = processed_imgs[i:i + 1]
            img_tensor.requires_grad = True

            # 生成热力图
            model.zero_grad()
            cam = grad_cam.generate_cam(img_tensor)

            # 清除梯度，避免累积
            img_tensor = img_tensor.detach()
            img_tensor.requires_grad = False

            # 准备原始图像用于显示
            original_img = imgs[i].cpu()
            original_img = denormalize_image(original_img)
            original_img = (original_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # 叠加热力图
            overlayed = overlay_heatmap(original_img, cam, alpha=0.5)

            # 获取预测结果
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
            pred_label = 1 if prob > 0.5 else 0


            # 创建综合可视化图
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 原图
            axes[0].imshow(original_img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # 热力图
            im = axes[1].imshow(cam, cmap='jet')
            axes[1].set_title('Heatmap')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            # 叠加图
            axes[2].imshow(overlayed)
            true_label_text = "Real" if label == 1 else "Fake"
            pred_label_text = "Real" if pred_label == 1 else "Fake"
            axes[2].set_title(
                f'Overlay\nTrue: {true_label_text}, Pred: {pred_label_text}\nConf: {prob:.3f}'
            )
            axes[2].axis('off')

            plt.tight_layout()

            # 保存
            # 清理 video_name
            if isinstance(video_names[i], str):
                video_name = Path(video_names[i]).stem  # 只取文件名，不含扩展名
                video_name = video_name.replace('\\', '_').replace('/', '_').replace(':', '_')
            else:
                video_name = f"video_{batch_idx}_{i}"

            # 清理 frame_idx
            if hasattr(frame_indices[i], 'item'):
                frame_idx = frame_indices[i].item()
            else:
                frame_idx = frame_indices[i]

            # 如果 frame_idx 是路径字符串，提取文件名
            if isinstance(frame_idx, str):
                frame_idx_clean = Path(frame_idx).stem
                # 移除所有非法字符
                frame_idx_clean = frame_idx_clean.replace('\\', '_').replace('/', '_').replace(':', '_')
            else:
                frame_idx_clean = str(frame_idx)

            # 生成安全的文件名
            filename = f"{video_name}_frame{frame_idx_clean}_prob{prob:.3f}.png"


            if label == 1:
                save_path = real_dir / filename
                real_count += 1
            else:
                save_path = fake_dir / filename
                fake_count += 1

            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"已保存: {save_path.name} (Real: {real_count}, Fake: {fake_count})")

            if real_count >= max_samples // 2 and fake_count >= max_samples // 2:
                break

    print(f"\n完成! 共生成 {real_count + fake_count} 个热力图")
    print(f"Real 样本: {real_count}")
    print(f"Fake 样本: {fake_count}")


def get_parser():
    parser = argparse.ArgumentParser(description="热力图可视化脚本")

    # 模型相关
    parser.add_argument("--checkpoint", type=str, default=r'C:\Users\jumpycat\OneDrive\03-Projects\30 - DETECTOR\v3\runs\20260211181050_efficientnet_b0\checkpoints\ckpt_step_0050000.pt', help="模型 checkpoint 路径")
    parser.add_argument("--model_name", type=str, default="efficientnet_b0", help="模型名称")

    # 数据相关
    parser.add_argument("--data_root", type=str, default=r'E:\00_fjw\00_data\DeepSpeak\deepspeak_exported\train_cropped_faces', help="数据集根目录")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_videos", type=int, default=32, help="每个类别最大视频数")
    parser.add_argument("--max_frames", type=int, default=4, help="每个视频最大帧数")
    parser.add_argument("--fake_types", nargs='+', default=['Deepfakes', 'Face2Face', 'FaceSwap'], help="Fake 类型（用于 FF++ 数据集）")
    parser.add_argument("--ff_split", type=str, default='test', choices=['train', 'val', 'test'])

    # Noiser 相关
    parser.add_argument("--use_noiser", action="store_true", help="是否使用 noiser")
    parser.add_argument("--noiser_resume", type=str, default=None, help="Noiser checkpoint 路径")

    # 可视化相关
    parser.add_argument("--output_dir", type=str, default=r"C:\Users\jumpycat\OneDrive\03-Projects\30 - DETECTOR\v3\runs\20260211181050_efficientnet_b0\heatmap_visualizations", help="输出目录")
    parser.add_argument("--max_samples", type=int, default=128, help="最大可视化样本数（真伪各占一半）")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    # 其他
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"使用设备: {device}")

    # 1. 加载模型
    print(f"正在加载模型: {args.checkpoint}")
    model = BaselineModel(args=args).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("模型加载成功!")

    # 2. 加载 Noiser（如果需要）
    noiser = None
    if args.use_noiser:
        if args.noiser_resume and os.path.exists(args.noiser_resume):
            print(f"正在加载 Noiser: {args.noiser_resume}")
            noiser = DenoisingFCNWithSkip().to(device)
            noiser_state = torch.load(args.noiser_resume, map_location=device)["noiser"]
            noiser.load_state_dict(noiser_state, strict=False)
            noiser.eval()
            for p in noiser.parameters():
                p.requires_grad = False
            print("Noiser 加载成功!")

    # 3. 准备数据
    print("正在加载数据集...")
    transform, normalize = get_transforms(args)

    logger = SimpleLogger()

    dataset = create_dataset(
        data_root=args.data_root,
        max_videos=args.max_videos,
        max_frames=args.max_frames,
        args=args,
        logger=logger,
        transform=transform,
        normalize=normalize,
        augmentation=None  # 可视化时不需要增广
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"数据集大小: {len(dataset)}")

    # 4. 生成可视化
    visualize_samples(
        model=model,
        dataloader=dataloader,
        args=args,
        output_dir=args.output_dir,
        noiser=noiser,
        max_samples=args.max_samples
    )

    print("\n全部完成!")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)