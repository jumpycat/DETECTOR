import os
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torch
import random
import numpy as np
from typing import Optional, List, Tuple
from distortions import gaussian_blur, lens_blur, color_saturation, color_shift, jpeg, white_noise, impulse_noise, brighten, darken, jitter, quantization,linear_contrast_change

class DeepSpeakDataset(Dataset):
    """
    DeepSpeak 结构的 Dataset: root/{class}/{video_id}/{frame}.png

    - class: real / fake
    - video_id:
        - real: 任意命名（不含伪造类型）
        - fake: 形如 diff2lip-7076-7066-speechify （用 '-' 分割，第一个字段是伪造类型）
    - frame: *.png / *.jpg ...

    返回:
        img, real_fake_label, fake_type_str

    约定:
        real_fake_label: real=1, fake=0
        fake_type_str: 仅 fake 有意义，real 返回 "real"（你也可以改成 "none"）
    """

    def __init__(self, root=None, max_videos=-1, max_frames=-1, logger=None, transform=None, augmentation=None, normalize=None):
        self.root = root
        self.transform = transform
        self.augmentation = augmentation
        self.normalize = normalize

        self.logger = logger

        if self.logger:
            self.logger.info(f"DeepSpeakDataset 初始化完成，root={root}")

        # samples: [(img_path, real_fake_label, fake_type_str), ...]
        self.samples = []

        # Real=1, Fake=0
        classes = {'real': 1, 'fake': 0}
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        self.logger.info(f"正在构建 DeepSpeak 数据集索引: {root}")
        self.logger.info(f"  采样设置: max_videos={max_videos}, max_frames={max_frames}")

        for cls_name, cls_label in classes.items():
            cls_folder = os.path.join(root, cls_name)
            if not os.path.isdir(cls_folder):
                self.logger.info(f"  警告: 未找到类别文件夹 {cls_folder}")
                continue

            # 获取所有视频文件夹
            video_folders = [
                d for d in os.listdir(cls_folder)
                if os.path.isdir(os.path.join(cls_folder, d))
            ]
            video_folders.sort()

            # 1) 视频级采样
            if max_videos > 0 and len(video_folders) > max_videos:
                rng = random.Random(42)
                selected_videos = rng.sample(video_folders, max_videos)
            else:
                selected_videos = video_folders

            self.logger.info(f"  类别 '{cls_name}': 扫描了 {len(selected_videos)}/{len(video_folders)} 个视频文件夹...")

            count_imgs = 0
            for vid in tqdm(selected_videos, desc=f"Scanning {cls_name}", leave=False):
                vid_path = os.path.join(cls_folder, vid)

                # 2) 解析 fake_type（仅 fake 有）
                if cls_name == "fake":
                    # diff2lip-7076-7066-speechify -> diff2lip
                    fake_type = vid.split('-')[0].strip()
                    if fake_type == "":
                        fake_type = "unknown"
                    fake_type_str = fake_type
                else:
                    fake_type_str = "real"  # 或者改成 "none"

                # 获取所有图片帧
                images = [f for f in os.listdir(vid_path) if f.lower().endswith(valid_extensions)]
                images.sort()

                # 3) 帧级采样
                if max_frames > 0 and len(images) > max_frames:
                    rng = random.Random(42)
                    selected_images = rng.sample(images, max_frames)
                else:
                    selected_images = images

                for img_name in selected_images:
                    img_path = os.path.join(vid_path, img_name)
                    self.samples.append((img_path, cls_label, fake_type_str))
                    count_imgs += 1

            self.logger.info(f"  类别 '{cls_name}': 共加载 {count_imgs} 张图像。")

        # 打印一下发现的 fake_type（可选）
        fake_types = sorted({ft for _, y, ft in self.samples if y == 0})
        if len(fake_types) > 0:
            self.logger.info(f"[统计] 共发现 {len(fake_types)} 种伪造类型: {fake_types}")
        else:
            self.logger.info("[统计] 未发现伪造类型（可能没有 fake 类别或 fake 文件夹为空）。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, fake_type_str = self.samples[index]
        img = Image.open(path).convert('RGB')

        # 1. Resize, Crop, ToTensor
        if self.transform:
            img = self.transform(img)

        # 2. Augmentation (在 [0,1] 上)
        if self.augmentation:
            img = self.augmentation(img, label)

        # 3. Normalize
        if self.normalize:
            img = self.normalize(img)

        return img, label, fake_type_str, path

class FFPPDataset(Dataset):
    """
    FaceForensics++ (FF++) 数据集加载器

    数据集结构:
        root/
        ├── train/
        │   ├── Deepfakes/
        │   ├── Face2Face/
        │   ├── FaceShifter/
        │   ├── FaceSwap/
        │   ├── NeuralTextures/
        │   └── original/
        ├── test/
        └── val/

    每个类型文件夹下包含视频文件夹(如 001_870/)，视频文件夹内是帧图片(*.png)

    参数:
        root (str): 数据集根目录
        split (str): 'train', 'test' 或 'val'
        fake_types (list): 要使用的伪造类型列表，如 ['Deepfakes', 'Face2Face']
        transform: 图像变换
        max_videos (int): 每个类型最多采样的视频数，-1表示全部
        max_frames (int): 每个视频最多采样的帧数，-1表示全部

    返回:
        img: 图像张量
        real_fake_label: 1=real, 0=fake
        fake_type_str: 伪造类型名称或'original'
        path: 图像路径

    真伪平衡策略:
        - 每种fake类型采样 max_videos 个视频（-1表示全部）
        - original 先采样 max_videos 个视频，然后将视频列表重复 len(fake_types) 倍
        - 例如：max_videos=100, fake_types=['A','B']
          → 每种fake采样100个视频
          → original采样100个视频，重复2倍得到200个视频ID（有重复）
          → 最终每个original视频的帧会被使用2次
    """

    def __init__(self, root, split='train', fake_types=None, max_videos=-1, max_frames=-1, logger=None, transform=None, augmentation=None, normalize=None):
        self.root = root
        self.split = split
        self.fake_types = fake_types
        self.transform = transform
        self.augmentation = augmentation
        self.normalize = normalize


        self.max_videos = max_videos
        self.max_frames = max_frames
        self.logger = logger
        if self.logger:
            self.logger.info(f"FFPPDataset 初始化完成，root={root}")

        # samples: [(img_path, real_fake_label, fake_type_str), ...]
        self.samples = []

        # 有效图像扩展名
        self.valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        # 数据集分割目录
        self.split_dir = os.path.join(root, split)
        if not os.path.isdir(self.split_dir):
            raise ValueError(f"分割目录不存在: {self.split_dir}")

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"正在构建 FF++ 数据集: {split} split")
        self.logger.info(f"  根目录: {root}")
        self.logger.info(f"  使用的伪造类型: {fake_types} (共 {len(fake_types)} 种)")
        self.logger.info(f"  采样设置: max_videos={max_videos}, max_frames={max_frames}")
        self.logger.info(f"  真伪平衡: original 视频列表重复 {len(fake_types)} 倍")
        self.logger.info(f"{'=' * 60}\n")

        # 1. 先处理所有fake类型
        for fake_type in fake_types:
            self._load_class(fake_type, label=0)

        # 2. 再处理original (平衡采样 - 重复视频列表)
        self._load_original_balanced()

        # 统计信息
        self._print_statistics()

    def _load_original_balanced(self):
        """
        加载 original 类别，并通过重复视频列表实现平衡采样

        逻辑：
        1. 读取 original 文件夹所有视频
        2. 按 max_videos 采样（如果 max_videos > 0）
        3. 将采样后的视频列表重复 len(fake_types) 倍
        4. 从重复后的列表中加载所有帧
        """
        class_name = 'original'
        class_folder = os.path.join(self.split_dir, class_name)

        if not os.path.isdir(class_folder):
            self.logger.info(f"  ⚠️  警告: 未找到类别文件夹 {class_folder}")
            return

        # 获取所有视频文件夹
        video_folders = [
            d for d in os.listdir(class_folder)
            if os.path.isdir(os.path.join(class_folder, d))
        ]
        video_folders.sort()

        total_videos = len(video_folders)

        # 视频级采样
        if self.max_videos > 0 and total_videos > self.max_videos:
            rng = random.Random(42)
            selected_videos = rng.sample(video_folders, self.max_videos)
        else:
            selected_videos = video_folders

        # 重复视频列表 len(fake_types) 倍
        repeated_videos = selected_videos * len(self.fake_types)

        self.logger.info(f"  📁 类别 '{class_name}': 原始 {len(selected_videos)}/{total_videos} 个视频")
        self.logger.info(f"     → 重复 {len(self.fake_types)} 倍后共 {len(repeated_videos)} 个视频（用于平衡采样）...")

        label = 1  # real
        fake_type_str = 'original'

        count_imgs = 0
        for vid in tqdm(repeated_videos, desc=f"  Loading {class_name}", leave=False):
            vid_path = os.path.join(class_folder, vid)

            # 获取所有图片帧
            images = [
                f for f in os.listdir(vid_path)
                if f.lower().endswith(self.valid_extensions)
            ]
            images.sort()

            if len(images) == 0:
                continue

            # 帧级采样
            if self.max_frames > 0 and len(images) > self.max_frames:
                rng = random.Random(42)
                selected_images = rng.sample(images, self.max_frames)
            else:
                selected_images = images

            for img_name in selected_images:
                img_path = os.path.join(vid_path, img_name)
                self.samples.append((img_path, label, fake_type_str))
                count_imgs += 1

        self.logger.info(f"  ✓  类别 '{class_name}': 共加载 {count_imgs} 张图像\n")

    def _load_class(self, class_name, label):
        """
        加载某个类别的所有样本

        Args:
            class_name: 类别名称 (如 'Deepfakes')
            label: 0=fake, 1=real
        """
        class_folder = os.path.join(self.split_dir, class_name)

        if not os.path.isdir(class_folder):
            self.logger.info(f"  ⚠️  警告: 未找到类别文件夹 {class_folder}")
            return

        # 获取所有视频文件夹
        video_folders = [
            d for d in os.listdir(class_folder)
            if os.path.isdir(os.path.join(class_folder, d))
        ]
        video_folders.sort()

        total_videos = len(video_folders)

        # 视频级采样
        if self.max_videos > 0 and total_videos > self.max_videos:
            rng = random.Random(42)
            selected_videos = rng.sample(video_folders, self.max_videos)
        else:
            selected_videos = video_folders

        self.logger.info(f"  📁 类别 '{class_name}': 扫描 {len(selected_videos)}/{total_videos} 个视频文件夹...")

        # fake_type_str: 对于fake使用类别名，对于real使用'original'
        fake_type_str = class_name

        count_imgs = 0
        for vid in tqdm(selected_videos, desc=f"  Loading {class_name}", leave=False):
            vid_path = os.path.join(class_folder, vid)

            # 获取所有图片帧
            images = [
                f for f in os.listdir(vid_path)
                if f.lower().endswith(self.valid_extensions)
            ]
            images.sort()

            if len(images) == 0:
                continue

            # 帧级采样
            if self.max_frames > 0 and len(images) > self.max_frames:
                rng = random.Random(42)
                selected_images = rng.sample(images, self.max_frames)
            else:
                selected_images = images

            for img_name in selected_images:
                img_path = os.path.join(vid_path, img_name)
                self.samples.append((img_path, label, fake_type_str))
                count_imgs += 1

        self.logger.info(f"  ✓  类别 '{class_name}': 共加载 {count_imgs} 张图像\n")

    def _print_statistics(self):
        """打印数据集统计信息"""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"数据集构建完成!")
        self.logger.info(f"{'=' * 60}")

        # 统计real/fake数量
        real_count = sum(1 for _, label, _ in self.samples if label == 1)
        fake_count = sum(1 for _, label, _ in self.samples if label == 0)

        self.logger.info(f"  总样本数: {len(self.samples)}")
        self.logger.info(f"  Real样本: {real_count} ({real_count / len(self.samples) * 100:.1f}%)")
        self.logger.info(f"  Fake样本: {fake_count} ({fake_count / len(self.samples) * 100:.1f}%)")

        # 统计各伪造类型数量
        fake_type_counts = {}
        for _, label, fake_type in self.samples:
            if fake_type not in fake_type_counts:
                fake_type_counts[fake_type] = 0
            fake_type_counts[fake_type] += 1

        self.logger.info(f"\n  各类型分布:")
        for ft in sorted(fake_type_counts.keys()):
            count = fake_type_counts[ft]
            self.logger.info(f"    - {ft:20s}: {count:6d} ({count / len(self.samples) * 100:.1f}%)")

        self.logger.info(f"{'=' * 60}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, fake_type_str = self.samples[index]
        img = Image.open(path).convert('RGB')

        # 1. Resize, Crop, ToTensor
        if self.transform:
            img = self.transform(img)

        # 2. Augmentation (在 [0,1] 上)
        if self.augmentation:
            img = self.augmentation(img, label)

        # 3. Normalize
        if self.normalize:
            img = self.normalize(img)

        return img, label, fake_type_str, path

class ConditionalResize(object):
    """
    (来自 eval_UniversalFakeDetect.py)
    可序列化的转换
    """

    def __init__(self, size):
        self.size = size
        self.resize_op = transforms.Resize(size)

    def __call__(self, img):
        if min(img.size) < self.size:
            return self.resize_op(img)
        return img


def get_transforms(args):
    model_name = args.model_name.lower()

    if "clip" in model_name:
        # CLIP 官方归一化
        mean = [0.48145466, 0.4578275, 0.40821073]
        std  = [0.26862954, 0.26130258, 0.27577711]
    else:

        mean = [0.5, 0.5, 0.5]
        std  = [0.5, 0.5, 0.5]

    transform_list = [
        ConditionalResize(args.img_size),
        transforms.RandomCrop(args.img_size),
        transforms.ToTensor(),
    ]

    return transforms.Compose(transform_list), transforms.Normalize(mean=mean, std=std)


def get_dataset_type(data_root):
    """根据路径自动识别数据集类型"""
    data_root_lower = data_root.lower()
    if 'deepspeak' in data_root_lower:
        return 'DeepSpeak'
    elif 'faceforensics' in data_root_lower or 'ff++' in data_root_lower or 'ffpp' in data_root_lower:
        return 'FFPP'
    else:
        # 默认或根据其他规则判断
        raise ValueError(f"无法从路径识别数据集类型: {data_root}")


def create_dataset(data_root, max_videos, max_frames, args, logger, transform, normalize, augmentation=None):
    """
    根据data_root自动识别并创建对应的数据集
    """
    dataset_type = get_dataset_type(data_root)

    if dataset_type == 'FFPP':
        return FFPPDataset(
            root=data_root,
            split=args.ff_split,
            fake_types=args.fake_types,
            max_videos=max_videos,
            max_frames=max_frames,
            logger=logger,
            transform=transform,
            augmentation=augmentation,
            normalize=normalize,
        )
    elif dataset_type == 'DeepSpeak':
        return DeepSpeakDataset(
            root=data_root,
            max_videos=max_videos,
            max_frames=max_frames,
            logger=logger,
            transform=transform,
            augmentation=augmentation,
            normalize=normalize,
        )
    else:
        raise ValueError(f"未知数据集类型: {dataset_type}")
