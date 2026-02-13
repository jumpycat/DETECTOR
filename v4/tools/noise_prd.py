import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from denoiser import DenoisingFCNWithSkip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(img_path, size=224):
    """加载图像并预处理"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))

    # 转换为 tensor: [H, W, C] -> [C, H, W]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [0, 1]

    # Normalize到[-1, 1] (与训练时一致)
    img_tensor = img_tensor * 2.0 - 1.0
    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]

    return img_tensor


def save_noise(noise_tensor, save_path):
    """保存噪声图像"""
    # noise_tensor shape: [1, C, H, W], 值域 [0, 1]
    noise = noise_tensor.squeeze(0).cpu().numpy()  # [C, H, W]
    noise = noise.transpose(1, 2, 0)  # [H, W, C]

    # 转换为 0-255
    noise = (noise * 255).astype(np.uint8)
    noise = cv2.cvtColor(noise, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, noise)
    print(f"噪声已保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Denoiser 推理脚本")
    parser.add_argument("--img_path", type=str, default=r'C:\Users\jumpycat\OneDrive\03-Projects\30 - DETECTOR\figs\00095.png', help="输入图像路径")
    parser.add_argument("--output_path", type=str, default=r'C:\Users\jumpycat\OneDrive\03-Projects\30 - DETECTOR\figs\00095_nise.png', help="输出噪声图像路径")
    parser.add_argument("--model_path", type=str, default=r"C:\Users\jumpycat\OneDrive\03-Projects\30 - DETECTOR\v3\pretrained_ckpts/checkpoint-400000.pth", help="Denoiser 模型权重路径")
    parser.add_argument("--img_size", type=int, default=224, help="图像尺寸")
    args = parser.parse_args()

    # 自动生成输出路径
    if args.output_path is None:
        img_name = Path(args.img_path).stem
        args.output_path = f"{img_name}_noise.png"

    print(f"设备: {device}")
    print(f"加载图像: {args.img_path}")

    # 1. 加载模型
    print(f"加载 Denoiser 模型: {args.model_path}")
    noiser = DenoisingFCNWithSkip().to(device)

    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        noiser.load_state_dict(ckpt["noiser"])
        print("模型加载成功")
    else:
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")

    noiser.eval()

    # 2. 加载图像 (normalize到[-1, 1])
    img_tensor = load_image(args.img_path, args.img_size).to(device)

    # 3. 推理
    print("正在预测噪声...")
    with torch.no_grad():
        res = noiser(img_tensor)
        noise = res - res.floor()  # 提取小数部分作为噪声

    # 4. 保存结果
    save_noise(noise, args.output_path)
    print("完成!")


if __name__ == "__main__":
    main()