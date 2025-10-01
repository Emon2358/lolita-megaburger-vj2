import cv2
import numpy as np
import random
import sys
import argparse # コマンドライン引数を扱うために追加

# --- 設定項目 ---
INPUT_VIDEO_FG = 'video1.mp4'
INPUT_VIDEO_BG = 'video2.mp4'
OUTPUT_VIDEO = 'final_video.mp4'
# --- 設定はここまで ---

# ★追加: ワークフローからの 'true'/'false' 文字列をbool値に変換する関数
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def process_video(args):
    cap_fg = cv2.VideoCapture(INPUT_VIDEO_FG)
    cap_bg = cv2.VideoCapture(INPUT_VIDEO_BG)

    if not cap_fg.isOpened():
        print(f"エラー: {INPUT_VIDEO_FG} を開けませんでした。")
        return
    if not cap_bg.isOpened():
        print(f"エラー: {INPUT_VIDEO_BG} を開けませんでした。")
        return

    width = int(cap_fg.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_fg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_fg.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_fg.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print("動画処理を開始します...")
    
    frame_count = 0
    while True:
        ret_fg, frame_fg = cap_fg.read()
        ret_bg, frame_bg = cap_bg.read()

        if not ret_fg or not ret_bg:
            break
        
        frame_count += 1
        sys.stdout.write(f"\rフレーム処理中: {frame_count} / {total_frames}")
        sys.stdout.flush()

        frame_bg_resized = cv2.resize(frame_bg, (width, height))
        random_color_bgr = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
        
        # クロマキー処理
        lower_bound = np.clip(random_color_bgr - args.tolerance, 0, 255)
        upper_bound = np.clip(random_color_bgr + args.tolerance, 0, 255)
        mask = cv2.inRange(frame_fg, lower_bound, upper_bound)
        mask_inv = cv2.bitwise_not(mask)
        fg_masked = cv2.bitwise_and(frame_fg, frame_fg, mask=mask_inv)
        bg_masked = cv2.bitwise_and(frame_bg_resized, frame_bg_resized, mask=mask)
        chroma_frame = cv2.add(fg_masked, bg_masked)

        output_frame = chroma_frame

        # ★変更点: if/elifから連続したif文に変更し、エフェクトを重ねがけできるようにする
        
        # チャンネルシフトエフェクトを適用
        if args.apply_channel_shift:
            b, g, r = cv2.split(output_frame)
            shift = args.shift_intensity
            b = np.roll(b, shift, axis=1)
            r = np.roll(r, -shift, axis=1)
            output_frame = cv2.merge([b, g, r])

        # エッジ検出エフェクトを適用
        if args.apply_edge:
            gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            edges = cv2.convertScaleAbs(laplacian)
            output_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # サーマルビジョンエフェクト
        if args.apply_thermal_vision:
            # エッジ検出などでグレースケールになっている可能性を考慮
            if len(output_frame.shape) == 2 or output_frame.shape[2] == 1:
                 gray = output_frame
            else:
                 gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
            output_frame = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)

        # ピクセルソートエフェクト
        if args.apply_pixel_sort:
            sorted_frame = output_frame.copy()
            num_rows_to_sort = int(height * args.sort_amount)
            rows_to_sort = random.sample(range(height), num_rows_to_sort)
            for y in rows_to_sort:
                row = sorted_frame[y, :]
                # ピクセルの明るさに基づいてソート
                sorted_row = sorted(row, key=lambda p: p[0]*0.114 + p[1]*0.587 + p[2]*0.299 if len(p) == 3 else p[0], reverse=random.choice([True, False]))
                sorted_frame[y, :] = np.array(sorted_row, dtype=np.uint8)
            output_frame = sorted_frame


        out.write(output_frame)

    print(f"\n動画処理が完了しました。'{OUTPUT_VIDEO}' として保存されました。")

    cap_fg.release()
    cap_bg.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # ★変更点: エフェクト指定を個別のboolフラグに変更
    parser = argparse.ArgumentParser(description='動画にクロマキーと複数の激しい特殊エフェクトを適用します。')
    parser.add_argument('--tolerance', type=int, default=50, help='クロマキーの色の許容度 (0-255)')
    parser.add_argument('--apply-channel-shift', type=str2bool, default=False, help='チャンネルシフトエフェクトを適用するかどうか')
    parser.add_argument('--apply-edge', type=str2bool, default=False, help='エッジ検出エフェクトを適用するかどうか')
    parser.add_argument('--apply-thermal-vision', type=str2bool, default=False, help='サーマルビジョンエフェクトを適用するかどうか')
    parser.add_argument('--apply-pixel-sort', type=str2bool, default=False, help='ピクセルソートエフェクトを適用するかどうか')
    parser.add_argument('--shift-intensity', type=int, default=10, help='チャンネルシフトエフェクトの強さ（ずらすピクセル数）')
    parser.add_argument('--sort-amount', type=float, default=0.2, help='ピクセルソートを適用する行の割合 (0.0-1.0)')
    
    args = parser.parse_args()
    process_video(args)
