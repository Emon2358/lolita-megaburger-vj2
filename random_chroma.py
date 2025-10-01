import cv2
import numpy as np
import random
import sys
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import logging
from multiprocessing import Manager

# --- 設定項目 ---
INPUT_VIDEO_FG = 'video1.mp4'
INPUT_VIDEO_BG = 'video2.mp4'
OUTPUT_VIDEO = 'final_video.mp4'
# --- 設定はここまで ---

# ロガーの設定
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(processName)s] %(message)s', datefmt='%H:%M:%S')

# --- エフェクト関数 ---
def apply_channel_shift(frame, intensity):
    """ BGRチャンネルをずらして色収差のような効果を出す """
    b, g, r = cv2.split(frame)
    b_shifted = np.roll(b, intensity, axis=1)
    r_shifted = np.roll(r, -intensity, axis=1)
    return cv2.merge([b_shifted, g, r_shifted])

def apply_edge_detection(frame):
    """ Cannyアルゴリズムでエッジを検出する """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_thermal_vision(frame):
    """ サーモグラフィのような見た目にする """
    heatmap = cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    return heatmap

# --- フレーム処理のコア関数 ---
def process_frame(frame_index, frame_fg_path, frame_bg_path, args, temp_dir, progress_dict, total_frames):
    """ 1フレーム分の処理を行う """
    try:
        # フレーム画像を読み込む
        frame_fg = cv2.imread(frame_fg_path)
        frame_bg = cv2.imread(frame_bg_path)
        if frame_fg is None or frame_bg is None:
            return None

        height, width, _ = frame_fg.shape
        frame_bg_resized = cv2.resize(frame_bg, (width, height))
        
        # クロマキー処理
        random_color_bgr = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
        tolerance = int(args.tolerance)
        lower_bound = np.clip(random_color_bgr - tolerance, 0, 255)
        upper_bound = np.clip(random_color_bgr + tolerance, 0, 255)
        mask = cv2.inRange(frame_fg, lower_bound, upper_bound)
        mask_inv = cv2.bitwise_not(mask)
        fg_masked = cv2.bitwise_and(frame_fg, frame_fg, mask=mask_inv)
        bg_masked = cv2.bitwise_and(frame_bg_resized, frame_bg_resized, mask=mask)
        processed_frame = cv2.add(fg_masked, bg_masked)

        # 選択されたエフェクトを適用
        if args.apply_channel_shift:
            processed_frame = apply_channel_shift(processed_frame, int(args.shift_intensity))
        if args.apply_edge:
            processed_frame = apply_edge_detection(processed_frame)
        if args.apply_thermal_vision:
            processed_frame = apply_thermal_vision(processed_frame)

        # 処理済みフレームを一時ファイルとして保存
        output_path = os.path.join(temp_dir, f"frame_{frame_index:06d}.png")
        cv2.imwrite(output_path, processed_frame)

        # 進捗を更新してログ出力
        process_name = f"Core-{os.getpid() % 100}"
        with progress_dict['lock']:
            progress_dict['count'] += 1
            count = progress_dict['count']
        
        if count % 20 == 0 or count == total_frames: # 20フレームごと、または最後のフレームでログ出力
             logging.info(f"[{process_name}] Processed frame {frame_index+1} ({count}/{total_frames})")

        return output_path
    except Exception as e:
        logging.error(f"Error processing frame {frame_index}: {e}")
        return None

# --- 動画の前処理・後処理 ---
def extract_frames(video_path, output_dir):
    """ 動画から全フレームを画像として抽出する """
    logging.info(f"Extracting frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path}")
        return 0
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{count:06d}.png"), frame)
        count += 1
    cap.release()
    logging.info(f"Extracted {count} frames.")
    return count

def combine_frames_to_video(frame_dir, output_path, fps, width, height):
    """ 画像フレームを動画に結合する """
    logging.info("Combining frames into final video...")
    # ffmpegを使用して高速に動画を生成
    command = [
        'ffmpeg',
        '-y', # Overwrite output file if it exists
        '-framerate', str(fps),
        '-i', os.path.join(frame_dir, 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-s', f'{width}x{height}',
        output_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info(f"Video saved to {output_path}")

# --- メイン実行関数 ---
def main(args):
    """ メインの処理フロー """
    cap_fg = cv2.VideoCapture(INPUT_VIDEO_FG)
    if not cap_fg.isOpened():
        logging.error(f"Error opening video: {INPUT_VIDEO_FG}")
        return

    width = int(cap_fg.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_fg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_fg.get(cv2.CAP_PROP_FPS)
    total_frames_fg = int(cap_fg.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_fg.release()

    cap_bg = cv2.VideoCapture(INPUT_VIDEO_BG)
    total_frames_bg = int(cap_bg.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_bg.release()
    
    total_frames = min(total_frames_fg, total_frames_bg)

    # 一時ディレクトリを作成
    temp_fg_dir = "temp_fg_frames"
    temp_bg_dir = "temp_bg_frames"
    temp_processed_dir = "temp_processed_frames"
    os.makedirs(temp_fg_dir, exist_ok=True)
    os.makedirs(temp_bg_dir, exist_ok=True)
    os.makedirs(temp_processed_dir, exist_ok=True)
    
    # フレームを抽出
    extract_frames(INPUT_VIDEO_FG, temp_fg_dir)
    extract_frames(INPUT_VIDEO_BG, temp_bg_dir)

    # 並列処理
    num_cores = os.cpu_count() or 2
    logging.info(f"Starting parallel processing on {num_cores} cores...")
    
    with Manager() as manager:
        progress_dict = manager.dict({'count': 0, 'lock': manager.Lock()})
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(process_frame, i, os.path.join(temp_fg_dir, f"frame_{i:06d}.png"), os.path.join(temp_bg_dir, f"frame_{i:06d}.png"), args, temp_processed_dir, progress_dict, total_frames) for i in range(total_frames)]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"A task generated an exception: {e}")

    # フレームを動画に結合
    combine_frames_to_video(temp_processed_dir, OUTPUT_VIDEO, fps, width, height)

    # 一時ディレクトリをクリーンアップ
    logging.info("Cleaning up temporary files...")
    subprocess.run(['rm', '-rf', temp_fg_dir, temp_bg_dir, temp_processed_dir])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply multiple effects to a video with chroma key.')
    parser.add_argument('--tolerance', type=str, default='50')
    # ★変更点: ピクセルソートに関する引数を削除
    parser.add_argument('--apply-channel-shift', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--apply-edge', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--apply-thermal-vision', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--shift-intensity', type=str, default='10')

    args = parser.parse_args()
    main(args)
