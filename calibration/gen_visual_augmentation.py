# Original: evaluation/calibration/_gen_posthoc_visual.py
#!/usr/bin/env python3
"""Generate additional visual OOD posthoc data from normal recordings.

Pure visual corruptions only (frames modified, actions untouched).
Each corruption is applied to all normal NPZ files in data_renewed/processed_64x64/.
Output goes to data_renewed/ood_posthoc/.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from glob import glob
from PIL import Image, ImageFilter, ImageEnhance
import io

SRC_DIR = 'data_renewed/processed_64x64'
OUT_DIR = 'data_renewed/ood_posthoc'

EXISTING_VISUAL = {'blur_5', 'blur_9', 'dark_03', 'dark_05'}


def frames_to_pil(frames_np):
    """Convert (N, 3, 64, 64) uint8 array to list of PIL images."""
    imgs = []
    for f in frames_np:
        arr = np.transpose(f, (1, 2, 0))
        imgs.append(Image.fromarray(arr))
    return imgs


def pil_to_frames(imgs):
    """Convert list of PIL images back to (N, 3, 64, 64) uint8 array."""
    arrs = []
    for img in imgs:
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        arrs.append(np.transpose(arr, (2, 0, 1)))
    return np.stack(arrs, axis=0).astype(np.uint8)


def jpeg_compress(imgs, quality=10):
    out = []
    for img in imgs:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        out.append(Image.open(buf).copy())
    return out


def contrast_low(imgs, factor=0.3):
    return [ImageEnhance.Contrast(img).enhance(factor) for img in imgs]


def contrast_high(imgs, factor=2.5):
    return [ImageEnhance.Contrast(img).enhance(factor) for img in imgs]


def saturate_low(imgs, factor=0.2):
    return [ImageEnhance.Color(img).enhance(factor) for img in imgs]


def fog_effect(imgs, intensity=0.6):
    out = []
    for img in imgs:
        arr = np.array(img).astype(np.float32)
        white = np.full_like(arr, 220.0)
        fogged = arr * (1 - intensity) + white * intensity
        out.append(Image.fromarray(np.clip(fogged, 0, 255).astype(np.uint8)))
    return out


def speckle_noise(imgs, std=0.15):
    out = []
    for img in imgs:
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.randn(*arr.shape).astype(np.float32) * std
        noisy = arr + arr * noise
        out.append(Image.fromarray(np.clip(noisy * 255, 0, 255).astype(np.uint8)))
    return out


def salt_pepper(imgs, amount=0.08):
    out = []
    for img in imgs:
        arr = np.array(img).copy()
        n_pixels = arr.shape[0] * arr.shape[1]
        n_salt = int(n_pixels * amount / 2)
        n_pepper = int(n_pixels * amount / 2)
        ys = np.random.randint(0, arr.shape[0], n_salt)
        xs = np.random.randint(0, arr.shape[1], n_salt)
        arr[ys, xs] = 255
        yp = np.random.randint(0, arr.shape[0], n_pepper)
        xp = np.random.randint(0, arr.shape[1], n_pepper)
        arr[yp, xp] = 0
        out.append(Image.fromarray(arr))
    return out


def pixelate(imgs, block_size=8):
    out = []
    for img in imgs:
        w, h = img.size
        small = img.resize((w // block_size, h // block_size), Image.NEAREST)
        out.append(small.resize((w, h), Image.NEAREST))
    return out


def motion_blur(imgs, kernel_size=7):
    out = []
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    from PIL import ImageFilter
    k_flat = kernel.flatten().tolist()
    filt = ImageFilter.Kernel((kernel_size, kernel_size), k_flat, scale=1, offset=0)
    for img in imgs:
        out.append(img.filter(filt))
    return out


def color_jitter(imgs, hue_shift=30):
    out = []
    for img in imgs:
        hsv = img.convert('HSV')
        arr = np.array(hsv)
        arr[:, :, 0] = (arr[:, :, 0].astype(int) + hue_shift) % 256
        out.append(Image.fromarray(arr, mode='HSV').convert('RGB'))
    return out


def frost_effect(imgs, intensity=0.5):
    out = []
    for img in imgs:
        arr = np.array(img).astype(np.float32)
        frost = np.random.randint(180, 255, arr.shape, dtype=np.uint8).astype(np.float32)
        mask = np.random.random(arr.shape[:2]) < 0.3
        mask = np.stack([mask]*3, axis=-1).astype(np.float32)
        result = arr * (1 - mask * intensity) + frost * mask * intensity
        out.append(Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)))
    return out


def brightness_high(imgs, factor=2.0):
    return [ImageEnhance.Brightness(img).enhance(factor) for img in imgs]


CORRUPTIONS = {
    'jpeg_q10':       lambda imgs: jpeg_compress(imgs, quality=10),
    'jpeg_q5':        lambda imgs: jpeg_compress(imgs, quality=5),
    'saturate_low':   lambda imgs: saturate_low(imgs, factor=0.2),
    'fog':            lambda imgs: fog_effect(imgs, intensity=0.6),
    'speckle':        lambda imgs: speckle_noise(imgs, std=0.15),
    'salt_pepper':    lambda imgs: salt_pepper(imgs, amount=0.08),
    'pixelate':       lambda imgs: pixelate(imgs, block_size=8),
    'color_jitter':   lambda imgs: color_jitter(imgs, hue_shift=30),
    'frost':          lambda imgs: frost_effect(imgs, intensity=0.5),
}


def process_file(src_path, corruption_name, corrupt_fn):
    """Apply corruption to one normal NPZ file, save as posthoc OOD."""
    basename = os.path.basename(src_path)
    out_name = f'ood_{corruption_name}_{basename}'
    out_path = os.path.join(OUT_DIR, out_name)

    if os.path.exists(out_path):
        return out_path, True

    data = np.load(src_path, allow_pickle=True)
    frames = data['frame']

    pil_imgs = frames_to_pil(frames)
    corrupted_pil = corrupt_fn(pil_imgs)
    corrupted_frames = pil_to_frames(corrupted_pil)

    save_dict = {
        'frame': corrupted_frames,
        'anomaly_type': corruption_name,
    }
    for key in data.keys():
        if key not in ('frame', 'anomaly_type'):
            save_dict[key] = data[key]

    if 'actual_actions' in data:
        if 'action' not in data:
            steering = data['actual_actions']
            T = len(steering)
            save_dict['action'] = np.stack([steering, np.full(T, 0.5)], axis=1).astype(np.float32)

    np.savez_compressed(out_path, **save_dict)
    return out_path, False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    src_files = sorted(glob(os.path.join(SRC_DIR, '*.npz')))
    print(f'Source files: {len(src_files)} in {SRC_DIR}')
    print(f'Output dir: {OUT_DIR}')
    print(f'Corruptions to generate: {len(CORRUPTIONS)}')
    for name in sorted(CORRUPTIONS.keys()):
        print(f'  {name}')
    print()

    total_new = 0
    total_skip = 0
    for c_name in sorted(CORRUPTIONS.keys()):
        corrupt_fn = CORRUPTIONS[c_name]
        print(f'--- {c_name} ---')
        for i, src in enumerate(src_files):
            basename = os.path.basename(src)
            print(f'  [{i+1}/{len(src_files)}] {basename}...', end='', flush=True)
            try:
                out_path, skipped = process_file(src, c_name, corrupt_fn)
                if skipped:
                    print(' skipped (exists)')
                    total_skip += 1
                else:
                    print(' done')
                    total_new += 1
            except Exception as e:
                print(f' ERROR: {e}')
        print()

    print(f'Finished! New: {total_new}, Skipped: {total_skip}')
    total_files = len(glob(os.path.join(OUT_DIR, '*.npz')))
    print(f'Total files in {OUT_DIR}: {total_files}')


if __name__ == '__main__':
    main()
