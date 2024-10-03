import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_cell_contents(table_json_path):
    with open(table_json_path) as f:
        data = json.load(f)
    cells = []
    headers = data['headers']
    for header in headers:
        for col in header['columns']:
            content = col['content'].strip()
            if len(content) > 0:
                cells.append(content)
    rows = data['rows']
    for row in rows:
        for col in row:
            content = col['content'].strip()
            if len(content) > 0:
                cells.append(content)
    return cells


def handle_dir(in_dir: Path, out_dir: Path, val_frac=0.005):
    table_paths = list(in_dir.glob("*.json"))
    total = len(table_paths)
    val_len = max(1, int(val_frac * total))
    random.shuffle(table_paths)
    train_file = out_dir / "train.txt"
    val_file = out_dir / "val.txt"
    val_table_paths = table_paths[:val_len]
    tr_table_paths = table_paths[val_len:]
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = 'a' if train_file.exists() else 'w'
    with open(train_file, mode) as f:
        for tr_table_path in tqdm(tr_table_paths, desc="Train"):
            cells = extract_cell_contents(tr_table_path)
            f.write('\n'.join(cells))

    mode = 'a' if val_file.exists() else 'w'
    with open(val_file, mode) as f:
        for val_table_path in tqdm(val_table_paths, desc="Test"):
            cells = extract_cell_contents(val_table_path)
            f.write('\n'.join(cells))


def handle_dirs(in_root_dir: Path, out_dir: Path, val_frac=0.005):
    sub_dirs = [Path(f.path) for f in os.scandir(in_root_dir) if f.is_dir()]
    for sub_dir in sub_dirs:
        handle_dir(sub_dir, out_dir, val_frac=val_frac)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="prepares pretraining data (one cell content per line) from table "
                                                 "JSON files")
    parser.add_argument("-i", metavar="<input-dir>", required=True)
    parser.add_argument("-o", metavar="<output-dir>", required=True)

    args = parser.parse_args()

    in_root = Path(args.i)
    output_dir = Path(args.o)
    handle_dirs(in_root, output_dir)














