import cv2
import numpy as np
import random
import sys
import argparse
import multiprocessing
import os
import subprocess
import time

# --- 設定項目 ---
INPUT_VIDEO_FG = 'video1.mp4'
INPUT_VIDEO_BG = 'video2.mp4'
OUTPUT_VIDEO = 'final_video.mp4'
# --- 設定はここまで ---

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def apply_pixel_sort_numpy(frame, sort_amount):
    """
    NumPyを使用してピクセルソートを高速に実行する関数
    """
    height, width, _ = frame.shape
    sorted_frame = frame.copy()
    
    num_rows_to_sort = int(height * sort_amount)
    if num_rows_to_sort == 0:
        return sorted_frame
        
    rows_indices = np.random.choice(height, size=num_rows_to_sort, replace=False)
    
    rows_to_sort = sorted_frame[rows_indices, :]
    
    luminance = rows_to_sort @ np.array([0.114, 0.587, 0.299]) # BGRの重み
    
    sorted_indices = np.argsort(luminance, axis=1)
    
    I, J = np.ogrid[:rows_to_sort.shape[0], :rows_to_sort.shape[1]]
    sorted_rows = rows_to_sort[I, sorted_indices]
    
    sorted_frame[rows_indices, :] = sorted_rows
    
    return sorted_frame


def process_frame_chunk(chunk_info):
    """
    動画の特定の部分（チャンク）を処理するワーカー関数。
    """
    start_frame, end_frame, process_id, args, video_props = chunk_info
    
    cap_fg = cv2.VideoCapture(INPUT_VIDEO_FG)
    cap_bg = cv2.VideoCapture(INPUT_VIDEO_BG)
    
    if not cap_fg.isOpened() or not cap_bg.isOpened():
        print(f"[Process {process_id}] Error opening video files.")
        return None

    cap_fg.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_bg.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    temp_output_filename = f"temp_part_{process_id}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_filename, fourcc, video_props['fps'], (video_props['width'], video_props['height']))

    for frame_num in range(start_frame, end_frame):
        ret_fg, frame_fg = cap_fg.read()
        ret_bg, frame_bg = cap_bg.read()

        if not ret_fg or not ret_bg:
            break

        # ★追加: コンソールのスパムを防ぐため、一定間隔でログを出力
        log_interval = 30 # 30フレームごとにログを出力
        if (frame_num - start_frame) % log_interval == 0:
            # プロセスIDと、全体の何フレーム目かを分かりやすく表示
            print(f"[Process {process_id:02d}] Processing frame {frame_num + 1} / {video_props['total_frames']}")

        frame_bg_resized = cv2.resize(frame_bg, (video_props['width'], video_props['height']))
        random_color_bgr = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
        
        lower_bound = np.clip(random_color_bgr - args.tolerance, 0, 255)
        upper_bound = np.clip(random_color_bgr + args.tolerance, 0, 255)
        mask = cv2.inRange(frame_fg, lower_bound, upper_bound)
        mask_inv = cv2.bitwise_not(mask)
        fg_masked = cv2.bitwise_and(frame_fg, frame_fg, mask=mask_inv)
        bg_masked = cv2.bitwise_and(frame_bg_resized, frame_bg_resized, mask=mask)
        output_frame = cv2.add(fg_masked, bg_masked)

        if args.apply_channel_shift:
            b, g, r = cv2.split(output_frame)
            shift = args.shift_intensity
            b = np.roll(b, shift, axis=1)
            r = np.roll(r, -shift, axis=1)
            output_frame = cv2.merge([b, g, r])

        if args.apply_edge:
            gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            edges = cv2.convertScaleAbs(laplacian)
            output_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if args.apply_thermal_vision:
            if len(output_frame.shape) == 2 or output_frame.shape[2] == 1: gray = output_frame
            else: gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
            output_frame = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)

        if args.apply_pixel_sort:
            output_frame = apply_pixel_sort_numpy(output_frame, args.sort_amount)
        
        out.write(output_frame)

    cap_fg.release()
    cap_bg.release()
    out.release()
    
    # ★変更: 以前の進捗表示を削除し、完了ログをより明確に
    print(f"[Process {process_id:02d}] Chunk ({start_frame + 1}-{end_frame}) processing complete.")

    return temp_output_filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='動画にクロマキーと複数の激しい特殊エフェクトを並列処理で適用します。')
    parser.add_argument('--tolerance', type=int, default=50)
    parser.add_argument('--apply-channel-shift', type=str2bool, default=False)
    parser.add_argument('--apply-edge', type=str2bool, default=False)
    parser.add_argument('--apply-thermal-vision', type=str2bool, default=False)
    parser.add_argument('--apply-pixel-sort', type=str2bool, default=False)
    parser.add_argument('--shift-intensity', type=int, default=10)
    parser.add_argument('--sort-amount', type=float, default=0.2)
    
    args = parser.parse_args()

    cap = cv2.VideoCapture(INPUT_VIDEO_FG)
    if not cap.isOpened():
        print(f"エラー: {INPUT_VIDEO_FG} を開けませんでした。")
        sys.exit(1)
        
    video_props = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()

    num_processes = int(os.environ.get('NUM_CORES', multiprocessing.cpu_count()))
    frames_per_process = video_props['total_frames'] // num_processes
    
    chunks = []
    for i in range(num_processes):
        start_frame = i * frames_per_process
        end_frame = (i + 1) * frames_per_process
        if i == num_processes - 1:
            end_frame = video_props['total_frames'] 
        chunks.append((start_frame, end_frame, i, args, video_props))

    print(f"{video_props['total_frames']} フレームを {num_processes} 個のCPUコアで並列処理します...")
    start_time = time.time()

    with multiprocessing.Pool(processes=num_processes) as pool:
        temp_files = pool.map(process_frame_chunk, chunks)
    
    print(f"\nすべてのチャンクの処理が完了しました。結合しています...")
    
    temp_files = [f for f in temp_files if f is not None]
    with open("concat_list.txt", "w") as f:
        for temp_file in temp_files:
            f.write(f"file '{temp_file}'\n")

    subprocess.run(
        ["ffmpeg", "-f", "concat", "-safe", "0", "-i", "concat_list.txt", "-c", "copy", OUTPUT_VIDEO],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    os.remove("concat_list.txt")
    for temp_file in temp_files:
        os.remove(temp_file)

    end_time = time.time()
    print(f"動画処理が完了しました。'{OUTPUT_VIDEO}' として保存されました。")
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")
