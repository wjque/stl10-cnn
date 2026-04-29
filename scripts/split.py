import os
import shutil
import random
import argparse


def split_dataset(data_dir='STL10', val_ratio=0.2, seed=42, dry_run=False):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir):
        print(f'Error: {train_dir} not found')
        return

    classes = sorted(os.listdir(train_dir))
    total_count = 0
    val_count = 0

    for cls in classes:
        cls_train_dir = os.path.join(train_dir, cls)
        cls_val_dir = os.path.join(val_dir, cls)

        if not os.path.isdir(cls_train_dir):
            continue

        files = sorted([f for f in os.listdir(cls_train_dir) if f.endswith('.png')])

        random.seed(seed)
        random.shuffle(files)

        n_val = max(1, int(len(files) * val_ratio))
        val_files = files[:n_val]

        total_count += len(files)
        val_count += n_val

        if not dry_run:
            os.makedirs(cls_val_dir, exist_ok=True)
            for f in val_files:
                src = os.path.join(cls_train_dir, f)
                dst = os.path.join(cls_val_dir, f)
                shutil.move(src, dst)

        print(f'  {cls}: {len(files)} total -> {n_val} moved to val, {len(files) - n_val} kept in train')

    print(f'\nSummary: {val_count}/{total_count} images moved to {val_dir} (ratio={val_count/total_count:.2%})')
    if dry_run:
        print('(Dry run: no files were actually moved)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split train data into train/val')
    parser.add_argument('--data_dir', type=str, default='STL10', help='Dataset root directory')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dry_run', action='store_true', help='Preview without moving files')
    args = parser.parse_args()

    split_dataset(args.data_dir, args.val_ratio, args.seed, args.dry_run)
