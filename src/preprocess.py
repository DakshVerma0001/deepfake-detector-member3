"""
Preprocess pipeline:
- Extract frames from each video (ffmpeg via cv2)
- Sample frames at given fps (or pick N frames)
- Detect & crop faces using facenet-pytorch (MTCNN)
- Save crops organized as data/faces/{split}/{label}/frame_*.jpg
- Generate metadata.csv with columns: video_id,frame_path,timestamp, label

Quick/--quick mode: very small subset useful for CPU-only runs.
"""
import os
import argparse
import csv
from facenet_pytorch import MTCNN
import cv2
from tqdm import tqdm
from pathlib import Path
import math
from datetime import timedelta
import random
from utils import set_seed

set_seed(42)

def extract_frames(video_path, tmp_out_dir, target_fps=1, max_frames=None):
    os.makedirs(tmp_out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / orig_fps if orig_fps>0 else 0
    step = max(1, int(round(orig_fps / target_fps)))
    saved = []
    idx = 0
    out_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            if max_frames and out_idx>=max_frames:
                break
            fname = os.path.join(tmp_out_dir, f"frame_{out_idx:04d}.jpg")
            cv2.imwrite(fname, frame)
            timestamp = idx / orig_fps
            saved.append((fname, timestamp))
            out_idx += 1
        idx += 1
    cap.release()
    return saved, orig_fps

def crop_faces(frames, mtcnn, out_dir, video_id, label, max_faces_per_frame=1, img_size=224):
    os.makedirs(out_dir, exist_ok=True)
    metadata = []
    for fname, ts in frames:
        img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        # facenet-pytorch MTCNN expects PIL-like arrays; it supports np arrays
        boxes, probs = mtcnn.detect(img)
        if boxes is None:
            continue
        # keep top max_faces_per_frame by probability (if probs available)
        if len(boxes) > max_faces_per_frame:
            # sort by area or probability
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            idxs = sorted(range(len(boxes)), key=lambda i: probs[i] if probs is not None else areas[i], reverse=True)[:max_faces_per_frame]
        else:
            idxs = list(range(len(boxes)))
        for j, iidx in enumerate(idxs):
            b = boxes[iidx].astype(int)
            # expand a small margin
            h, w, _ = img.shape
            padx = int(0.2*(b[2]-b[0]))
            pady = int(0.2*(b[3]-b[1]))
            x1 = max(0, b[0]-padx); y1 = max(0, b[1]-pady)
            x2 = min(w, b[2]+padx); y2 = min(h, b[3]+pady)
            crop = img[y1:y2, x1:x2]
            # resize
            import PIL.Image as Image
            crop_pil = Image.fromarray(crop).resize((img_size, img_size))
            save_name = f"{video_id}_t{int(ts*1000)}_f{j}.jpg"
            out_path = os.path.join(out_dir, save_name)
            crop_pil.save(out_path)
            metadata.append((video_id, out_path, ts, label))
    return metadata

def main(input_dir, out_dir, fps=1, img_size=224, quick=False):
    mtcnn = MTCNN(keep_all=True, device='cpu', thresholds=[0.6,0.7,0.7])
    raw = Path(input_dir)
    faces_base = Path(out_dir)
    os.makedirs(faces_base, exist_ok=True)

    video_files = sorted([p for p in raw.glob("*.mp4")])
    # label inference from filename for demo: real_*.mp4 or fake_*.mp4
    if quick:
        video_files = video_files[:4]

    metadata_rows = []
    tmp_frame_dir = Path("tmp_frames")
    if tmp_frame_dir.exists():
        import shutil; shutil.rmtree(tmp_frame_dir)
    tmp_frame_dir.mkdir(exist_ok=True)

    for v in tqdm(video_files, desc="Videos"):
        video_id = v.stem
        label = "real" if 'real' in video_id.lower() else "fake"
        frames, orig_fps = extract_frames(str(v), str(tmp_frame_dir/video_id), target_fps=fps, max_frames=(3 if quick else None))
        if len(frames)==0:
            continue
        out_split = "train"
        # small heuristic: last file pair as val/test
        if 'val' in video_id.lower():
            out_split='val'
        if 'test' in video_id.lower():
            out_split='test'
        out_dir_faces = faces_base / out_split / label
        md = crop_faces(frames, mtcnn, str(out_dir_faces), video_id, label, img_size=img_size)
        metadata_rows += md

    # write metadata.csv
    csv_path = faces_base.parent / "metadata.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["video_id","frame_path","timestamp","label"])
        for r in metadata_rows:
            writer.writerow([r[0], r[1], r[2], r[3]])
    print("Saved metadata to", csv_path)
    print("Faces saved under", faces_base)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw_videos")
    parser.add_argument("--out_dir", default="data/faces")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--quick", action="store_true", help="small subset for CPU")
    args = parser.parse_args()
    main(args.input_dir, args.out_dir, fps=args.fps, img_size=args.img_size, quick=args.quick)
