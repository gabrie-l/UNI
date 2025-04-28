import os
import argparse
import openslide
import cv2
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def read_svs_to_numpy(svs_path):
    slide = openslide.OpenSlide(svs_path)
    level = 0  # highest resolution
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")
    return np.array(img)


def is_foreground(patch, min_foreground_fraction=0.4, brightness_thresh=230):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    mean_brightness = gray.mean()

    if mean_brightness > brightness_thresh:
        return False

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground_fraction = (thresh < 128).mean()
    return foreground_fraction > min_foreground_fraction


def crop_and_save(img, out_dir, patch_size=224):
    h, w, _ = img.shape
    os.makedirs(out_dir, exist_ok=True)
    mask = np.zeros_like(img)

    progress_bar = Progress(
        f"Creating patches for image:",
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )
    total_patches = ((h - patch_size) // patch_size + 1) * (
        (w - patch_size) // patch_size + 1
    )
    count = 0
    with progress_bar:
        task = progress_bar.add_task("Cropping", total=total_patches)
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = img[y : y + patch_size, x : x + patch_size]
                if is_foreground(patch):
                    count += 1
                    filename = f"{x}_{y}.png"
                    path = os.path.join(out_dir, filename)
                    cv2.imwrite(path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                    mask[y : y + patch_size, x : x + patch_size] = [255, 0, 0]
                progress_bar.update(task, advance=1)
    cv2.imwrite("./patch_mask.png", cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
    print(f"Saved {count} patches to {out_dir}")


def process_svs(svs_path, output_root, patch_size=224):
    img = read_svs_to_numpy(svs_path)
    base_name = os.path.splitext(os.path.basename(svs_path))[0]
    out_dir = os.path.join(output_root, base_name)
    crop_and_save(img, out_dir, patch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Directory containing .svs files")
    parser.add_argument("--output_dir", help="Directory to save 224x224 patches")
    args = parser.parse_args()

    for fname in os.listdir(args.input_dir):
        if fname.endswith(".svs"):
            svs_path = os.path.join(args.input_dir, fname)
            process_svs(svs_path, args.output_dir)
