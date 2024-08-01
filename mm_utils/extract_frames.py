import random
import io
import os
import numpy as np
from PIL import Image
from decord import VideoReader
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 自定义函数，用于获取帧索引
def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices

# 修改后的多线程处理视频帧的函数
def read_frames_decord(
        video_path, save_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, clip=None
    ):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if video_path.startswith('s3') or video_path.startswith('p2'):
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=16)

    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    if clip:
        frame_indices = [f + start_index for f in frame_indices]

    try:
        frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    except decord.DECORDError as e:
        print(f'解码错误: {video_path}, {vlen}, {fps}, {duration}')
        print(f'decord.DECORDError报错: {e}')
        return [], [], 0, 0
    except Exception as e:
        print(f'解码错误: {video_path}, {vlen}, {fps}, {duration}')
        print(f'Exception报错: {e}')
        return [], [], 0, 0

    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

    for i in range(frames.shape[0]):
        frame = frames[i]
        # 转换为 (316, 600, 3) 的形状
        frame = frame.permute(1, 2, 0).numpy()
        image = Image.fromarray(frame)
        image.save(os.path.join(save_path, f'frame_{i + 1}.jpg'))

    return frames, frame_indices, float(fps), vlen

# 多线程处理视频的函数
def process_video(video_id, video_path, save_path, num_frames):
    video_path = os.path.join(video_path, video_id + '.mp4')
    save_path = os.path.join(save_path, video_id)
    read_frames_decord(video_path, save_path, num_frames, 'rand')

# 主程序
if __name__ == '__main__':
    video_path = '/data3/whb/data/Moment-10m/videos'
    save_path = '/data3/whb/data/Moment-10m/frames'
    video_ids = [video_id.replace('.mp4', '') for video_id in os.listdir(video_path)]

    num_frames = 128  # 要提取的帧数

    # 创建线程池
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_video, video_id, video_path, save_path, num_frames) for video_id in video_ids]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            future.result()  # 处理异常，如果有的话
