import os
import argparse
import openslide
import cv2
import numpy as np


def read_svs_to_numpy(svs_path):
    slide = openslide.OpenSlide(svs_path)
    level = 0  # highest resolution
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")
    return np.array(img)


def crop_and_save(img, out_dir, patch_size=448):
    h, w, _ = img.shape
    print(h, w)
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = img[y : y + patch_size, x : x + patch_size]
            filename = f"{x}_{y}.png"
            path = os.path.join(out_dir, filename)
            cv2.imwrite(path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            count += 1
    print(f"Saved {count} patches to {out_dir}")


def process_svs(svs_path, output_root, patch_size=448):
    img = read_svs_to_numpy(svs_path)
    base_name = os.path.splitext(os.path.basename(svs_path))[0]
    out_dir = os.path.join(output_root, base_name)
    crop_and_save(img, out_dir, patch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Directory containing .svs files")
    parser.add_argument("--output_dir", help="Directory to save 448x448 patches")
    args = parser.parse_args()

    for fname in os.listdir(args.input_dir):
        if fname.endswith(".svs"):
            svs_path = os.path.join(args.input_dir, fname)
            process_svs(svs_path, args.output_dir)
