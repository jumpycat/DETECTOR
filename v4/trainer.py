import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import argparse
import random
import cv2
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from data_tools import DeepSpeakDataset, FFPPDataset, get_transforms, create_dataset
from data_tools_aug import MultipleAugmentation
from models import BaselineModel
import importlib
from utils import setup_experiment, seedall, format_time, evaluate_and_save, worker_init_fn
from denoiser import DenoisingFCNWithSkip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser():
    parser = argparse.ArgumentParser(description="DeepSpeak Training Script")

    # 主要算法级别参数
    parser.add_argument("--ae_prob", type=float, default=0., help="应用 AE 重建增广的概率 (0-1)")
    parser.add_argument("--use_noiser", action="store_true", help="是否启用噪声残差预测作为输入")
    parser.add_argument("--use_blending", action="store_true", help="是否在 AE 增广中启用 Patch Blending (混合指纹)")

    # 数据相关
    parser.add_argument("--train_data_root", type=str, default=r'E:\00_fjw\00_data\DeepSpeak\deepspeak_exported\train_cropped_faces')
    parser.add_argument("--test_data_root", type=str, default=r'E:\00_fjw\00_data\FF++\FaceForensics++_RAW_split_cropped')
    parser.add_argument("--fake_types", nargs='+', default=['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures'])
    parser.add_argument("--ff_split", type=str, default='test')

    parser.add_argument("--img_size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--train_max_videos", type=int, default=-1, help="每个类别最大视频数 (-1为全部)")
    parser.add_argument("--train_max_frames", type=int, default=-1, help="每个视频最大帧数")
    parser.add_argument("--test_max_videos", type=int, default=-1, help="每个类别最大视频数 (-1为全部)")
    parser.add_argument("--test_max_frames", type=int, default=32, help="每个视频最大帧数")

    # 模型相关
    parser.add_argument("--model_name", type=str, default="efficientnet_b0") #efficientnet_b0, openai/clip-vit-large-patch14
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA Rank (仅 CLIP 有效)")
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint路径，用于恢复训练")

    # 训练相关
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_step", type=int, default=5000, help="每多少 step 做一次评估并保存 ckpt/csv")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")

    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning Rate")
    parser.add_argument("--weight_decay", type=float, default=0., help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="梯度累计步数 (虚拟大Batch)")

    parser.add_argument("--output_dir", type=str, default="runs", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # AE配置
    parser.add_argument("--ae_resume", type=str, default="pretrained_ckpts/checkpoint-1000000.pth", help="AE 模型的 checkpoint 路径")
    parser.add_argument("--encoder_module", type=str, default="decoders.encoder2", help="AE Encoder 模块路径")
    parser.add_argument("--decoder_module", type=str, default="decoders.decoder24", help="AE Decoder 模块路径")
    # --- Noiser 噪声残差参数 ---
    parser.add_argument("--noiser_resume", type=str, default="pretrained_ckpts/checkpoint-400000.pth", help="Noiser 模型的 checkpoint 路径")


    # 增广控制参数
    parser.add_argument("--aug_prob", type=float, default=0., help="应用增广的概率 (0-1), 设置为 0 则不启用")
    parser.add_argument("--aug_min_n", type=int, default=1, help="最少应用几种增广")
    parser.add_argument("--aug_max_n", type=int, default=4, help="最多应用几种增广")

    parser.add_argument("--aug_types", nargs='+',
                        default=['gaussian_blur', 'color_saturation', 'color_shift', 'jpeg', 'white_noise', 'brighten', 'jitter', 'quantization', 'linear_contrast_change'],
                        choices=[
                            'gaussian_blur', 'lens_blur', 'color_saturation', 'color_shift',
                            'jpeg', 'white_noise', 'impulse_noise', 'brighten', 'darken',
                            'jitter', 'quantization', 'linear_contrast_change', 'all'
                        ])

    return parser


def main(args):

    # 0. 设置随机种子
    seedall(args)

    # 1. 实验初始化
    exp_dir, ckpt_dir, logger = setup_experiment(args)
    logger.info(f"使用设备 {device}")

    # # 2. 数据准备
    logger.info("正在加载数据...")

    train_transform, train_normalize = get_transforms(args)

    # --- 修改部分开始：初始化增广 ---
    augmentation = None
    if args.aug_prob > 0:
        # 处理 'all' 的情况
        dist_types = None if 'all' in args.aug_types else args.aug_types

        logger.info(f"启用数据增广: Prob={args.aug_prob}, Count={args.aug_min_n}-{args.aug_max_n}")
        logger.info(f"增广类型: {dist_types if dist_types else '全部可用类型'}")

        augmentation = MultipleAugmentation(
            prob=args.aug_prob,
            num_augmentations=(args.aug_min_n, args.aug_max_n),
            distortion_types=dist_types,
            seed=args.seed  # 如果你的类支持传seed最好传进去
        )
    else:
        logger.info("未启用数据增广")

    train_set = create_dataset(
        args.train_data_root,
        args.train_max_videos,
        args.train_max_frames,
        args, logger, train_transform, train_normalize, augmentation
    )

    test_set = create_dataset(
        args.test_data_root,
        args.test_max_videos,
        args.test_max_frames,
        args, logger, train_transform, train_normalize, None
    )


    logger.info(f"数据加载完成: 训练集 {len(train_set)} 张, 验证集 {len(test_set)} 张")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)


    # 3. 模型初始化
    logger.info("正在初始化模型...")
    model = BaselineModel(args=args).to(device)

    # --- 加载 AE 模型 (用于增广) ---
    encoder = None
    decoder = None
    if args.ae_prob > 0 and args.ae_resume:
        logger.info(f"正在加载 AE 增广模型: {args.ae_resume}")
        try:
            # 动态导入模块
            enc_mod = importlib.import_module(args.encoder_module)
            dec_mod = importlib.import_module(args.decoder_module)

            encoder = enc_mod.Encoder().to(device)
            decoder = dec_mod.Decoder().to(device)

            # 加载权重
            ckpt_ae = torch.load(args.ae_resume, map_location=device)
            encoder.load_state_dict(ckpt_ae["encoder"])
            decoder.load_state_dict(ckpt_ae["decoder"])

            # 冻结并设为 eval
            encoder.eval()
            decoder.eval()
            for p in encoder.parameters(): p.requires_grad = False
            for p in decoder.parameters(): p.requires_grad = False

            logger.info("AE 模型加载成功，将用于数据增广。")
        except Exception as e:
            logger.error(f"AE 模型加载失败，跳过 AE 增广: {e}")
            encoder = None
            decoder = None

    # --- 初始化 Noiser (如果启用) ---
    noiser = None
    if args.use_noiser:
        logger.info("正在加载 Noiser 模型 (DenoisingFCNWithSkip)...")
        noiser = DenoisingFCNWithSkip().to(device)

        if args.noiser_resume and os.path.exists(args.noiser_resume):
            logger.info(f"加载 Noiser 权重: {args.noiser_resume}")
            noiser_state = torch.load(args.noiser_resume, map_location=device)["noiser"]
            try:
                noiser.load_state_dict(noiser_state)
            except Exception as e:
                logger.warning(f"Noiser 严格加载失败，尝试 strict=False: {e}")
                noiser.load_state_dict(noiser_state, strict=False)
        else:
            logger.warning("未指定 Noiser 权重或文件不存在，Noiser 将使用随机初始化！")

        noiser.eval()
        for p in noiser.parameters():
            p.requires_grad = False
        # ---------------------------


    # 4. 优化器与 Loss
    # 注意：只优化 requires_grad=True 的参数 (LoRA参数 + Head参数)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=args.lr
        )

    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    criterion = nn.BCEWithLogitsLoss()
    logger.info(f"可训练参数量: {sum(p.numel() for p in trainable_params)}")


    # ===== 加载 checkpoint (如果指定) =====
    start_step = 1
    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"正在加载 checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = ckpt["step"] + 1
            logger.info(f"成功加载! 从 step {start_step} 继续训练")
        else:
            logger.warning(f"checkpoint 不存在: {args.resume}，从头开始训练")

    start_time = time.time()
    model.train()
    data_iter = iter(train_loader)

    for step in range(start_step, args.steps * args.grad_accum_steps  + 1):
        try:
            imgs, original_labels, _, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            imgs, original_labels, _, _ = next(data_iter)

        imgs = imgs.float().to(device)
        original_labels = original_labels.to(device).float()

        # === 准备工作 ===
        # current_imgs 将流转整个处理过程
        current_imgs = imgs.clone()
        # 记录哪些样本被 AE 修改过（用于标签修正）
        is_augmented = torch.zeros(imgs.size(0), dtype=torch.bool, device=device)


        # === [控制 1 & 2] AE 增广逻辑 (ae_prob & use_blending) ===
        # 只有当概率 > 0 且 AE 模型加载成功时才执行
        if args.ae_prob > 0 and encoder is not None:
            with torch.no_grad():
                # 1.1 生成概率掩码
                # 简单逻辑：所有样本都有概率被选中
                prob_mask = torch.rand(imgs.size(0), device=device) < args.ae_prob

                # 获取被选中的索引
                idx_A = torch.where(prob_mask)[0]
                num_aug = len(idx_A)

                if num_aug > 0:
                    perm = torch.randperm(num_aug, device=device)

                    # --- 根据 use_blending 决定分配逻辑 ---
                    if args.use_blending:
                        # [启用 Blending]: 全部做混合指纹(C)
                        idx_B = torch.tensor([], device=device, dtype=torch.long)
                        idx_C = idx_A
                    else:
                        # [不启用 Blending]: 全部只做常规重构(B)
                        idx_B = idx_A
                        idx_C = torch.tensor([], device=device, dtype=torch.long)
                    # ------------------------------------

                    # >>> 处理集合 B: 常规增强 (直接 AE 重构) <<<
                    if len(idx_B) > 0:
                        imgs_B = current_imgs[idx_B]  # 取当前图
                        recon_B = decoder(encoder(imgs_B))
                        current_imgs[idx_B] = recon_B
                        is_augmented[idx_B] = True

                    # >>> 处理集合 C: 混合指纹增强 (Patch Blending) <<<
                    if len(idx_C) > 0:
                        imgs_C = current_imgs[idx_C]  # 取当前图

                        # Pass 1: 底图
                        latents_1 = encoder(imgs_C)
                        recon_base = decoder(latents_1)

                        # Pass 2: 补丁源
                        recon_patch_source = decoder(latents_1)

                        # Patching 逻辑
                        final_C = recon_base.clone()
                        B_size_c, _, H, W = final_C.shape

                        for i in range(B_size_c):
                            # CPU Numpy Mask
                            mask_np = np.zeros((H, W), dtype=np.uint8)

                            # 随机形状
                            shape_type = random.choice(['polygon', 'curve'])
                            center_x = random.randint(0, W)
                            center_y = random.randint(0, H)

                            if shape_type == 'polygon':
                                num_vertices = random.randint(3, 8)
                                radius = random.randint(int(min(H, W) * 0.1), int(min(H, W) * 0.3))
                                points = []
                                for _ in range(num_vertices):
                                    angle = random.uniform(0, 2 * np.pi)
                                    r = radius * random.uniform(0.5, 1.0)
                                    pt_x = np.clip(int(center_x + r * np.cos(angle)), 0, W - 1)
                                    pt_y = np.clip(int(center_y + r * np.sin(angle)), 0, H - 1)
                                    points.append([pt_x, pt_y])
                                cv2.fillPoly(mask_np, np.array([points], dtype=np.int32), 255)
                            else:
                                num_blobs = random.randint(1, 3)
                                for _ in range(num_blobs):
                                    cx = np.clip(center_x + random.randint(-20, 20), 0, W - 1)
                                    cy = np.clip(center_y + random.randint(-20, 20), 0, H - 1)
                                    axis_x = random.randint(int(W * 0.05), int(W * 0.2))
                                    axis_y = random.randint(int(H * 0.05), int(H * 0.2))
                                    angle = random.randint(0, 180)
                                    cv2.ellipse(mask_np, (cx, cy), (axis_x, axis_y), angle, 0, 360, 255, -1)

                            # 高斯模糊 (Soft Edge)
                            blur_ksize = random.choice([7, 9, 11, 13, 15, 17, 19, 21, 29, 35])
                            mask_blurred = cv2.GaussianBlur(mask_np, (blur_ksize, blur_ksize), 0)

                            # Alpha Fusion
                            mask_tensor = torch.from_numpy(mask_blurred).to(device, non_blocking=True).float() / 255.0
                            mask_tensor = mask_tensor.unsqueeze(0)

                            patch_part = recon_patch_source[i] * mask_tensor
                            base_part = recon_base[i] * (1.0 - mask_tensor)
                            final_C[i] = base_part + patch_part

                        current_imgs[idx_C] = final_C
                        is_augmented[idx_C] = True

        # === 标签修正逻辑 ===
        # 逻辑：只有 "纯净真图" (Original Real + 未被AE修改) -> Label 1
        # 其他 (Original Fake, AE Augmented Fake, AE Augmented Real) -> Label 0
        nature_label_input = 1
        final_labels = torch.zeros_like(original_labels, device=device)
        mask_clean_real = (original_labels == nature_label_input) & (~is_augmented)
        final_labels[mask_clean_real] = 1.0

        # === [控制 3] Noiser 处理 (use_noiser) ===
        # 无论是否经过 AE，只要开启 Noiser，就对当前图像进行噪声残差提取
        if args.use_noiser and noiser is not None:
            with torch.no_grad():
                res = noiser(current_imgs)
                current_imgs = res - res.floor()

        # === 前向传播 ===
        outputs = model(current_imgs).float().squeeze(1)
        loss = criterion(outputs, final_labels)

        # ===== 反向 (修改支持梯度累计) =====
        # 1. Loss 缩放
        loss = loss / args.grad_accum_steps
        loss.backward()

        # 2. 梯度累计步数到达或者是最后一步时，更新参数
        if step % args.grad_accum_steps == 0 or step == (args.steps * args.grad_accum_steps):
            optimizer.step()
            optimizer.zero_grad()

        # 日志（用你的 logger，不用 print / 手动写文件）
        if step % (args.log_freq * args.grad_accum_steps) == 0:
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                acc = (predicted == final_labels).float().mean()

            duration = time.time() - start_time
            dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            current_loss = loss.item() * args.grad_accum_steps

            log_msg = (
                f"{dt_string} [{step:07d}] "
                f"Loss: {current_loss:.4f} "  # 使用还原后的 loss
                f"Acc(train): {acc * 100:.2f} "
                f"Time: {format_time(duration)}"
            )
            logger.info(log_msg)

        # ====== 每 save_step 评估 + 保存 ckpt + 保存csv ======
        if step % (args.save_step * args.grad_accum_steps) == 0:
            logger.info(f"开始评估 step {step}...")
            metrics = evaluate_and_save(
                model=model,
                val_loader=val_loader,
                device=device,
                step=step,
                exp_dir=exp_dir,  # 新增这一行
                logger=logger,
                noiser=noiser,  # <--- 新增传参
            )

            ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{step:07d}.pt")
            torch.save(
                {
                    "step": step,
                    "model_name": args.model_name,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                    "metrics": metrics,
                },
                ckpt_path,
            )
            logger.info(f"[CKPT] Saved: {ckpt_path}")
            torch.cuda.empty_cache()  # 清理GPU缓存

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)