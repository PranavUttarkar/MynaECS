import os
import shutil
import random
from pathlib import Path

def split_esc_dataset(source_dir, dest_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42, organize_by_class=True):
    random.seed(seed)

    source = Path(source_dir)
    dest = Path(dest_dir)

    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    wav_files = list(source.glob("*.wav"))
    random.shuffle(wav_files)

    total = len(wav_files)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    splits = {
        "train": wav_files[:train_end],
        "valid": wav_files[train_end:valid_end],
        "test": wav_files[valid_end:]
    }

    for split_name, files in splits.items():
        for file in files:
            filename = file.name
            # Extract target class (after last dash and before .wav)
            target = filename.split("-")[-1].replace(".wav", "")
            if organize_by_class:
                target_folder = dest / split_name / target
            else:
                target_folder = dest / split_name

            target_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy(file, target_folder / filename)

    print(f"✅ Split complete: {len(wav_files)} files → {len(splits['train'])} train / {len(splits['valid'])} valid / {len(splits['test'])} test")

# Example usage
if __name__ == "__main__":
    split_esc_dataset(
        source_dir="audio",     # folder with the 2000 WAV files
        dest_dir="split",             # where train/valid/test folders will be saved
        train_ratio=0.8,
        valid_ratio=0.1,
        test_ratio=0.1
    )
