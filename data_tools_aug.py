import torch
import random
import numpy as np
from typing import Optional, List, Tuple
from distortions import gaussian_blur, lens_blur, color_saturation, color_shift, jpeg, white_noise, impulse_noise, brighten, darken, jitter, quantization,linear_contrast_change

class FFPPImageAugmentation:
    """
    适配 FFPPDataset 的图像增广类

    使用方式:
        augmentation = FFPPImageAugmentation(
            prob=0.5,
            distortion_types=['gaussian_blur', 'jpeg', 'white_noise'],
            apply_to_real=True,
            apply_to_fake=True
        )

        # 在 FFPPDataset 的 __getitem__ 中使用
        if self.augmentation:
            img = self.augmentation(img, label, fake_type_str)
    """

    def __init__(
            self,
            prob: float = 0.5,
            distortion_types: Optional[List[str]] = None,
            apply_to_real: bool = True,
            apply_to_fake: bool = True,
            seed: Optional[int] = None
    ):
        """
        参数:
            prob: 对每张图像应用增广的概率 (0-1)
            distortion_types: 要使用的失真类型列表，None表示使用全部
            apply_to_real: 是否对真实图像应用增广
            apply_to_fake: 是否对伪造图像应用增广
            seed: 随机种子
        """
        self.prob = prob
        self.apply_to_real = apply_to_real
        self.apply_to_fake = apply_to_fake

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 定义所有可用的失真类型及其参数范围
        self.available_distortions = {
            'gaussian_blur': {
                'func': gaussian_blur,
                'params': {'blur_sigma': (0.3, 3.0)}
            },
            'lens_blur': {
                'func': lens_blur,
                'params': {'radius': (1, 6)}
            },
            'color_saturation': {
                'func': color_saturation,
                'params': {'factor': (0.3, 2.0)}
            },
            'color_shift': {
                'func': color_shift,
                'params': {'amount': (1, 10)}
            },
            'jpeg': {
                'func': jpeg,
                'params': {'quality': (15, 85)}
            },
            'white_noise': {
                'func': white_noise,
                'params': {'var': (0.001, 0.05), 'clip': True, 'rounds': False}
            },
            'impulse_noise': {
                'func': impulse_noise,
                'params': {'d': (0.001, 0.03), 's_vs_p': 0.5}
            },
            'brighten': {
                'func': brighten,
                'params': {'amount': (0.1, 0.5)}
            },
            'darken': {
                'func': darken,
                'params': {'amount': (0.1, 0.5), 'dolab': False}
            },
            'jitter': {
                'func': jitter,
                'params': {'amount': (1, 5)}
            },
            'quantization': {
                'func': quantization,
                'params': {'levels': (8, 32)}
            },
            'linear_contrast_change': {
                'func': linear_contrast_change,
                'params': {'amount': (0.2, 0.6)}
            }
        }

        # 选择要使用的失真类型
        if distortion_types is None:
            self.distortion_types = list(self.available_distortions.keys())
        else:
            # 验证提供的类型是否有效
            invalid_types = set(distortion_types) - set(self.available_distortions.keys())
            if invalid_types:
                raise ValueError(f"无效的失真类型: {invalid_types}")
            self.distortion_types = distortion_types

    def _sample_params(self, distortion_type: str) -> dict:
        """随机采样失真参数"""
        params_config = self.available_distortions[distortion_type]['params']
        sampled_params = {}

        for param_name, param_range in params_config.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # 数值范围，均匀采样
                if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                    sampled_params[param_name] = random.randint(param_range[0], param_range[1])
                else:
                    sampled_params[param_name] = random.uniform(param_range[0], param_range[1])
            else:
                # 固定值
                sampled_params[param_name] = param_range

        return sampled_params

    def __call__(
            self,
            img: torch.Tensor,
            label: int,
    ) -> torch.Tensor:
        """
        对图像应用增广

        参数:
            img: 图像张量，形状为 (C, H, W)，值域 [0, 1]
            label: 1=real, 0=fake

        返回:
            增广后的图像张量
        """
        # 根据标签决定是否应用增广
        if label == 1 and not self.apply_to_real:
            return img
        if label == 0 and not self.apply_to_fake:
            return img

        # 以概率 prob 应用增广
        if random.random() > self.prob:
            return img

        # 随机选择一种失真类型
        distortion_type = random.choice(self.distortion_types)

        # 采样参数
        params = self._sample_params(distortion_type)

        # 应用失真
        distortion_func = self.available_distortions[distortion_type]['func']

        try:
            # 确保图像在正确的设备上
            device = img.device
            img_augmented = distortion_func(img, **params)

            # 确保输出在相同设备上
            if hasattr(img_augmented, 'to'):
                img_augmented = img_augmented.to(device)

            # 确保值域在 [0, 1]
            img_augmented = torch.clamp(img_augmented, 0, 1)

            return img_augmented
        except Exception as e:
            # 如果增广失败，返回原图
            print(f"警告: 应用 {distortion_type} 失败: {e}")
            return img


class MultipleAugmentation(FFPPImageAugmentation):
    """
    应用多个连续的增广操作
    """

    def __init__(
            self,
            prob: float = 0.5,
            num_augmentations: Tuple[int, int] = (1, 3),
            **kwargs
    ):
        """
        参数:
            num_augmentations: (min, max) 应用增广操作的数量范围
        """
        super().__init__(prob=prob, **kwargs)
        self.num_augmentations = num_augmentations

    def __call__(
            self,
            img: torch.Tensor,
            label: int,
    ) -> torch.Tensor:
        """应用多个增广操作"""
        # 根据标签决定是否应用增广
        if label == 1 and not self.apply_to_real:
            return img
        if label == 0 and not self.apply_to_fake:
            return img

        # 以概率 prob 应用增广
        if random.random() > self.prob:
            return img

        # 随机决定应用几个增广
        n_aug = random.randint(self.num_augmentations[0], self.num_augmentations[1])

        # 随机选择 n_aug 个不重复的失真类型
        selected_distortions = random.sample(
            self.distortion_types,
            min(n_aug, len(self.distortion_types))
        )

        # 依次应用每个增广
        img_augmented = img
        for distortion_type in selected_distortions:
            params = self._sample_params(distortion_type)
            distortion_func = self.available_distortions[distortion_type]['func']

            try:
                device = img_augmented.device
                img_augmented = distortion_func(img_augmented, **params)

                if hasattr(img_augmented, 'to'):
                    img_augmented = img_augmented.to(device)

                img_augmented = torch.clamp(img_augmented, 0, 1)
            except Exception as e:
                print(f"警告: 应用 {distortion_type} 失败: {e}")
                continue

        return img_augmented


# 使用示例
if __name__ == "__main__":
    """
    在 FFPPDataset 中集成增广的示例
    """

    # 方式1: 修改 FFPPDataset 的 __init__ 方法
    # 添加参数: augmentation=None

    # 方式2: 在 __getitem__ 中使用
    """
    def __getitem__(self, index):
        path, label, fake_type_str = self.samples[index]
        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)  # 转为 tensor

        # 应用增广
        if hasattr(self, 'augmentation') and self.augmentation:
            img = self.augmentation(img, label, fake_type_str)

        return img, label, fake_type_str, path
    """

    # 创建增广实例
    aug = FFPPImageAugmentation(
        prob=0.5,  # 50% 概率应用增广
        distortion_types=['gaussian_blur', 'jpeg', 'white_noise', 'brighten'],
        apply_to_real=True,
        apply_to_fake=True
    )

    # 或使用多重增广
    multi_aug = MultipleAugmentation(
        prob=0.7,
        num_augmentations=(1, 2),  # 应用1-2个增广
        distortion_types=['jpeg', 'white_noise', 'gaussian_blur']
    )

    # 在创建数据集时传入
    # dataset = FFPPDataset(
    #     root='/path/to/data',
    #     split='train',
    #     fake_types=['Deepfakes', 'Face2Face'],
    #     transform=your_transform,
    #     augmentation=aug  # 需要修改 FFPPDataset 以接受此参数
    # )