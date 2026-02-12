import os
import cv2
import math
from tqdm import tqdm
import torch
import numpy as np
from facexlib.detection import init_detection_model
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
import time
from collections import defaultdict

# ================= 配置区域 =================
INPUT_ROOT = r"E:\00_fjw\00_data\FF++\FaceForensics++_RAW_split"
OUTPUT_ROOT = r"E:\00_fjw\00_data\FF++\FaceForensics++_RAW_split_cropped"

# FaceForensics++ 数据集结构
# SPLITS = ["train", "test", "val"]
SPLITS = ["test"]

CATEGORIES = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures", "original"]
# CATEGORIES = ["original"]

# 人脸裁剪参数
PADDING_RATIO = 0.2
FRAME_INTERVAL = 5  # 每5帧采样1帧
BATCH_SIZE = 32  # 批量处理帧数
NUM_IO_THREADS = 16  # I/O线程数（保存图片）
NUM_READER_THREADS = 8  # 视频读取线程数
GPU_QUEUE_SIZE = 8  # GPU任务队列大小
WRITE_QUEUE_SIZE = 32  # 写入队列大小


# ===========================================


def _ensure_gpu_or_die():
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU。请确认安装了 CUDA 版 torch。")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_largest_face(faces):
    """获取最大人脸"""
    if not faces:
        return None
    max_area = 0
    best_face_area = None
    for key in faces:
        face = faces[key]
        box = face['facial_area']
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_face_area = box
    return best_face_area


def crop_face_to_square_strict(img, box, padding=0.0):
    """裁剪人脸为正方形"""
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = box

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    w_face = x2 - x1
    h_face = y2 - y1
    max_side = max(w_face, h_face)

    ideal_side = int(max_side * (1 + padding))
    ideal_radius = ideal_side // 2

    dist_left = cx
    dist_right = w_img - cx
    dist_top = cy
    dist_bottom = h_img - cy

    max_allowed_radius = min(dist_left, dist_right, dist_top, dist_bottom)
    final_radius = min(ideal_radius, max_allowed_radius)

    nx1 = cx - final_radius
    nx2 = cx + final_radius
    ny1 = cy - final_radius
    ny2 = cy + final_radius

    return img[max(0, ny1):min(h_img, ny2), max(0, nx1):min(w_img, nx2)]


def detect_faces_with_retinaface(detector, img_bgr, conf_threshold=0.5):
    """使用RetinaFace检测人脸"""
    h, w = img_bgr.shape[:2]

    with torch.no_grad():
        try:
            detections = detector.detect_faces(
                img_bgr,
                conf_threshold=conf_threshold,
                use_origin_size=True
            )
        except Exception as e:
            return None

    if detections is None or len(detections) == 0:
        return None

    faces = {}
    for idx, detection in enumerate(detections, 1):
        x1, y1, x2, y2, score = detection[:5]

        if score < conf_threshold:
            continue

        x1 = max(0, min(w, int(x1)))
        y1 = max(0, min(h, int(y1)))
        x2 = max(0, min(w, int(x2)))
        y2 = max(0, min(h, int(y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        faces[f"face_{idx}"] = {
            "score": float(score),
            "facial_area": [x1, y1, x2, y2]
        }

    return faces if faces else None


def detect_faces_batch(detector, frames, conf_threshold=0.5):
    """批量检测人脸"""
    results = []
    with torch.no_grad():
        for frame in frames:
            faces = detect_faces_with_retinaface(detector, frame, conf_threshold)
            results.append(faces)
    return results


def video_reader_worker(video_queue, batch_queue, stats):
    """视频读取工作线程"""
    while True:
        task = video_queue.get()
        if task is None:  # 结束信号
            break

        video_path, save_folder = task

        # 读取视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            stats['failed_videos'] += 1
            video_queue.task_done()
            continue

        frames = []
        indices = []
        frame_idx = 0

        # 连续读取所有帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_INTERVAL == 0:
                frames.append(frame)
                indices.append(frame_idx)

                # 达到批次大小，放入GPU队列
                if len(frames) >= BATCH_SIZE:
                    batch_queue.put((frames.copy(), indices.copy(), save_folder))
                    stats['batches_queued'] += 1
                    frames = []
                    indices = []

            frame_idx += 1

        # 处理剩余帧
        if frames:
            batch_queue.put((frames, indices, save_folder))
            stats['batches_queued'] += 1

        cap.release()
        stats['videos_read'] += 1
        video_queue.task_done()


def gpu_processor_worker(batch_queue, write_queue, detector, stats):
    """GPU处理工作线程"""
    while True:
        try:
            batch_data = batch_queue.get(timeout=1)
        except queue.Empty:
            continue

        if batch_data is None:  # 结束信号
            break

        frames, indices, save_folder = batch_data

        # GPU批量检测
        faces_batch = detect_faces_batch(detector, frames, conf_threshold=0.5)

        # 处理每一帧
        for frame, faces, frame_idx in zip(frames, faces_batch, indices):
            try:
                if faces and isinstance(faces, dict):
                    box = get_largest_face(faces)
                    if box is not None:
                        face_img = crop_face_to_square_strict(frame, box, padding=PADDING_RATIO)

                        if face_img is not None and face_img.size > 0:
                            save_name = os.path.join(save_folder, f"{frame_idx:05d}.png")
                            write_queue.put((face_img, save_name))
                            stats['faces_detected'] += 1
            except Exception as e:
                pass

        stats['batches_processed'] += 1
        stats['frames_processed'] += len(frames)
        batch_queue.task_done()


def image_writer_worker(write_queue):
    """图片保存线程"""
    while True:
        item = write_queue.get()
        if item is None:
            break

        face_img, save_path = item
        try:
            cv2.imwrite(save_path, face_img)
        except Exception as e:
            pass

        write_queue.task_done()


def progress_monitor(stats, total_videos, pbar):
    """进度监控线程"""
    last_videos = 0

    while stats['videos_read'] + stats['failed_videos'] < total_videos:
        time.sleep(1)

        # 更新进度条
        current_videos = stats['videos_read'] + stats['failed_videos']
        videos_delta = current_videos - last_videos
        if videos_delta > 0:
            pbar.update(videos_delta)
            last_videos = current_videos


def collect_video_tasks():
    """收集所有视频任务"""
    all_tasks = []
    skipped_count = 0  # 统计跳过的视频数

    for split in SPLITS:
        for category in CATEGORIES:
            input_cat_path = os.path.join(INPUT_ROOT, split, category)
            output_cat_path = os.path.join(OUTPUT_ROOT, split, category)

            if not os.path.exists(input_cat_path):
                print(f"⚠ 警告: 目录不存在 {input_cat_path}，跳过")
                continue

            video_files = [f for f in os.listdir(input_cat_path) if f.lower().endswith('.mp4')]

            if not video_files:
                print(f"⚠ 警告: {input_cat_path} 中没有找到 .mp4 文件")
                continue

            for video_file in video_files:
                video_path = os.path.join(input_cat_path, video_file)
                video_name = os.path.splitext(video_file)[0]
                save_folder = os.path.join(output_cat_path, video_name)

                # 检查输出文件夹是否已存在且有文件
                if os.path.exists(save_folder) and os.listdir(save_folder):
                    skipped_count += 1
                    continue  # 跳过已处理的视频

                ensure_dir(save_folder)
                all_tasks.append((video_path, save_folder))

            print(f"📁 {split}/{category}: {len(video_files)} 个视频 (跳过 {skipped_count} 个)")
            skipped_count = 0  # 重置计数

    return all_tasks

def process_videos():
    """主处理函数"""
    _ensure_gpu_or_die()

    print("=" * 60)
    print("FaceForensics++ 人脸裁剪工具")
    print("=" * 60)
    print(f"输入目录: {INPUT_ROOT}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"Padding 比例: {PADDING_RATIO}")
    print(f"采样间隔: 每 {FRAME_INTERVAL} 帧取 1 帧")
    print(f"批处理大小: {BATCH_SIZE}")
    print(f"读取线程数: {NUM_READER_THREADS}")
    print(f"保存线程数: {NUM_IO_THREADS}")
    print(f"GPU队列大小: {GPU_QUEUE_SIZE}")
    print("=" * 60)

    print("\n正在加载 RetinaFace 模型...")
    detector = init_detection_model('retinaface_resnet50')
    detector.eval()
    print("✓ 模型加载完成")

    # 统计信息
    stats = defaultdict(int)

    # 创建三个队列
    video_queue = queue.Queue(maxsize=NUM_READER_THREADS * 2)
    batch_queue = queue.Queue(maxsize=GPU_QUEUE_SIZE)
    write_queue = queue.Queue(maxsize=WRITE_QUEUE_SIZE)

    # 收集所有视频任务
    print("\n正在扫描视频文件...")
    all_video_tasks = collect_video_tasks()
    total_videos = len(all_video_tasks)

    if total_videos == 0:
        print("❌ 未找到任何视频文件")
        return

    print(f"\n总计: {total_videos} 个视频待处理")
    print("=" * 60)

    # 启动写入线程池
    print("\n启动写入线程...")
    write_threads = []
    for _ in range(NUM_IO_THREADS):
        t = threading.Thread(target=image_writer_worker, args=(write_queue,))
        t.daemon = True
        t.start()
        write_threads.append(t)

    # 启动GPU处理线程
    print("启动GPU处理线程...")
    gpu_thread = threading.Thread(
        target=gpu_processor_worker,
        args=(batch_queue, write_queue, detector, stats)
    )
    gpu_thread.daemon = True
    gpu_thread.start()

    # 启动视频读取线程池
    print("启动视频读取线程...")
    reader_threads = []
    for _ in range(NUM_READER_THREADS):
        t = threading.Thread(target=video_reader_worker, args=(video_queue, batch_queue, stats))
        t.daemon = True
        t.start()
        reader_threads.append(t)

    print("\n开始处理...\n")

    # 启动进度条
    pbar = tqdm(total=total_videos, desc="总进度", unit="video")

    # 启动监控线程
    monitor_thread = threading.Thread(target=progress_monitor, args=(stats, total_videos, pbar))
    monitor_thread.daemon = True
    monitor_thread.start()

    # 将所有视频任务放入队列
    for task in all_video_tasks:
        video_queue.put(task)

    # 等待所有视频读取完成
    video_queue.join()

    # 发送结束信号给读取线程
    for _ in range(NUM_READER_THREADS):
        video_queue.put(None)
    for t in reader_threads:
        t.join()

    # 等待所有批次处理完成
    batch_queue.join()

    # 发送结束信号给GPU线程
    batch_queue.put(None)
    gpu_thread.join()

    # 等待所有写入完成
    print("\n等待文件写入完成...")
    write_queue.join()

    # 发送结束信号给写入线程
    for _ in range(NUM_IO_THREADS):
        write_queue.put(None)
    for t in write_threads:
        t.join()

    pbar.close()

    # 打印统计信息
    print(f"\n{'=' * 60}")
    print(f"✓ 处理完成!")
    print(f"{'=' * 60}")
    print(f"  - 成功处理视频数: {stats['videos_read']}")
    print(f"  - 失败视频数: {stats['failed_videos']}")
    print(f"  - 处理批次数: {stats['batches_processed']}")
    print(f"  - 处理帧数: {stats['frames_processed']}")
    print(f"  - 提取人脸数: {stats['faces_detected']}")
    print(f"  - 结果保存在: {OUTPUT_ROOT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    process_videos()