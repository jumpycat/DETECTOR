import os
import argparse
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from collections import defaultdict
import glob
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
from PIL import Image

from transformers import CLIPModel
import loralib as lora
import timm
import torchvision.models as models
from datetime import datetime
import json
from datetime import datetime, timedelta
import logging

import csv
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)

def denormVQGAN(x):
    """ 将 VQGAN 的 [-1, 1] 范围反归一化到 [0, 1] """
    return (x + 1.0) / 2.0


def format_time(seconds):
    delta = timedelta(seconds=seconds)
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{days}d {hours}h {minutes}m {seconds}s'


def seedall(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_experiment(args):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    exp_name = f"{timestamp}_{args.model_name.replace('/', '_')}"

    exp_dir = os.path.join(args.output_dir, exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 保存参数
    with open(os.path.join(exp_dir, "args.txt"), "w", encoding="utf-8") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write(f"Command: {' '.join(sys.argv)}\n")

    log_filename = os.path.join(exp_dir, f"{exp_name}.log")

    # ---- logging：稳版配置（不依赖 basicConfig）----
    logger = logging.getLogger(exp_name)   # 给每个实验一个独立 logger 名称
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 防止传到 root 导致重复打印

    # 清掉旧的 handler（关键：避免重复/不换文件）
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 文件输出
    fh = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 控制台输出（可选但很实用）
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"Experiment dir: {exp_dir}")
    return exp_dir, ckpt_dir, logger




@torch.no_grad()
def evaluate_and_save(
    model,
    val_loader,
    device,
    step: int,
    exp_dir: str,  # 新增这个参数
    logger=None,
    noiser=None,  # <--- [新增1] 接收 noiser 参数，默认为 None
):
    model.eval()

    # 如果传入了 noiser，确保它处于 eval 模式
    if noiser is not None:
        noiser.eval()

    all_probs = []
    all_labels = []
    all_paths = []
    all_fake_types = []

    for batch in tqdm(val_loader, desc=f"Eval@{step}", leave=False):
        # 兼容你改造后的 dataset 输出
        imgs, labels, fake_type_strs, paths = batch

        imgs = imgs.to(device).float()
        labels = labels.to(device).float()

        # --- [新增2] 如果有 Noiser，先处理图片 ---
        if noiser is not None:
            residual = noiser(imgs)
            imgs = residual - residual.floor()
        # -------------------------------------

        logits = model(imgs).float().squeeze(1)           # [B]
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels_np)
        all_paths.extend(list(paths))
        all_fake_types.extend(list(fake_type_strs))

    probs = np.concatenate(all_probs, axis=0)  # [N]
    labels = np.concatenate(all_labels, axis=0).astype(np.float32)  # [N]
    preds = (probs > 0.5).astype(np.float32)

    # --- 分 real / fake 的 acc ---
    real_mask = (labels == 1)
    fake_mask = (labels == 0)

    real_acc = float((preds[real_mask] == labels[real_mask]).mean()) if real_mask.any() else float("nan")
    fake_acc = float((preds[fake_mask] == labels[fake_mask]).mean()) if fake_mask.any() else float("nan")
    bal_acc  = float(np.nanmean([real_acc, fake_acc]))

    # --- 二分类整体指标：AUC / AP / F1 ---
    auc = float(roc_auc_score(labels, probs))
    ap  = float(average_precision_score(labels, probs))
    f1 = float(f1_score(labels, preds))

    # --- 按某一类伪造 fake_type 统计 fake acc ---
    fake_metrics_by_type = {}
    if fake_mask.any():
        for ft in sorted(set([t for t, m in zip(all_fake_types, fake_mask.tolist()) if m])):
            # idx_fake: 只包含当前fake_type的fake样本（用于计算acc）
            idx_fake = np.array([(m and (t == ft)) for t, m in zip(all_fake_types, fake_mask.tolist())], dtype=bool)

            # idx_binary: 包含当前fake_type的fake样本 + 所有real样本（用于计算auc/ap/f1）
            idx_binary = np.array([
                (real_mask[i]) or (fake_mask[i] and all_fake_types[i] == ft)
                for i in range(len(all_fake_types))
            ], dtype=bool)

            if idx_fake.any():
                # acc: 只看fake样本的准确率
                ft_acc = float((preds[idx_fake] == labels[idx_fake]).mean())

                # auc/ap/f1: 当前fake_type vs real（二分类）
                ft_auc, ft_ap, ft_f1 = float("nan"), float("nan"), float("nan")
                if idx_binary.sum() > 0 and len(np.unique(labels[idx_binary])) > 1:  # 确保有两个类别
                    try:
                        ft_auc = float(roc_auc_score(labels[idx_binary], probs[idx_binary]))
                        ft_ap = float(average_precision_score(labels[idx_binary], probs[idx_binary]))
                        ft_f1 = float(f1_score(labels[idx_binary], preds[idx_binary]))
                    except Exception:
                        pass

                fake_metrics_by_type[ft] = {
                    "acc": ft_acc,
                    "auc": ft_auc,
                    "ap": ft_ap,
                    "f1": ft_f1
                }

    # --- 保存逐样本预测到 CSV ---
    pred_dir = os.path.join(exp_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    csv_path = os.path.join(pred_dir, f"pred_step_{step:07d}.csv")

    # 然后修改CSV写入部分，增加真实标签列：
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "png", "pred", "label"])  # 新增label列
        for pth, pr, lb in zip(all_paths, probs.tolist(), labels.tolist()):
            folder = os.path.basename(os.path.dirname(pth))
            png = os.path.basename(pth)
            writer.writerow([folder, png, f"{pr:.6f}", int(lb)])  # 新增lb

    # --- 打印/记录 ---
    metrics = {
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "bal_acc": bal_acc,
        "auc": auc,
        "ap": ap,
        "f1": f1,
        "fake_metrics_by_type": fake_metrics_by_type,
        "csv_path": csv_path,
    }

    metrics_csv_path = os.path.join(pred_dir, f"metrics_step_{step:07d}.csv")
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["real_acc", f"{real_acc*100:.2f}"])
        writer.writerow(["fake_acc", f"{fake_acc*100:.2f}"])
        writer.writerow(["bal_acc", f"{bal_acc*100:.2f}"])
        writer.writerow(["auc", f"{auc*100:.2f}"])
        writer.writerow(["ap", f"{ap*100:.2f}"])
        writer.writerow(["f1", f"{f1*100:.2f}"])

        # 保存所有fake_type的指标
        for ft, ft_metrics in sorted(fake_metrics_by_type.items()):
            writer.writerow([f"{ft}_acc", f"{ft_metrics['acc']*100:.2f}"])
            writer.writerow([f"{ft}_auc", f"{ft_metrics['auc']*100:.2f}"])
            writer.writerow([f"{ft}_ap", f"{ft_metrics['ap']*100:.2f}"])
            writer.writerow([f"{ft}_f1", f"{ft_metrics['f1']*100:.2f}"])

    # 找到logger.info部分，替换为：
    if logger is not None:
        logger.info(
            f"[Eval@{step}] real_acc={real_acc * 100:.2f} fake_acc={fake_acc * 100:.2f} "
            f"bal_acc={bal_acc * 100:.2f} auc={auc:.4f} ap={ap:.4f} f1={f1:.4f} "
            f"| saved_csv={csv_path}"
        )
        if len(fake_metrics_by_type) > 0:
            for ft, ft_metrics in sorted(fake_metrics_by_type.items()):
                logger.info(
                    f"[Eval@{step}] {ft}: acc={ft_metrics['acc'] * 100:.2f} "
                    f"auc={ft_metrics['auc']:.4f} ap={ft_metrics['ap']:.4f} f1={ft_metrics['f1']:.4f}"
                )
            logger.info(f"[Eval@{step}] metrics saved to: {metrics_csv_path}")

    model.train()
    return metrics

