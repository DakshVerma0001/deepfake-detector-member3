# !/bin/bash commands
# Quick wrapper to run preprocess -> train -> eval
set -e

# create sample demo data (tiny)
python3 src/data/generate_dummy_videos.py --out_dir data/raw_videos --num_pairs 2

# preprocess (use quick mode)
python3 src/preprocess.py --input_dir data/raw_videos --out_dir data/faces --fps 1 --img_size 224 --quick

# train quick
python3 src/train.py --data_dir data/faces --epochs 1 --batch_size 8 --img_size 224 --output_dir model --quick

# evaluate
python3 src/evaluate.py --data_dir data/faces --model_path model/best.pth --meta model/meta.json --output_dir results
