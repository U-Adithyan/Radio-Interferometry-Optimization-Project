import os
import argparse
import numpy as np

from skimage import io, transform

def circular_arc_mask(H, W, radius, theta_start=0.0, theta_end=np.pi):
    y, x = np.ogrid[:H, :W]
    cx = (W - 1) / 2
    cy = (H - 1) / 2

    X = x - cx
    Y = cy - y
    dist_sq = X**2 + Y**2

    ring_mask = (dist_sq <= (radius + 1.0)**2) & (dist_sq >= (radius - 1.0)**2)
    theta = np.mod(np.arctan2(Y, X), 2 * np.pi)

    theta_start = np.mod(theta_start, 2 * np.pi)
    theta_end = np.mod(theta_end, 2 * np.pi)

    if theta_start <= theta_end:
        arc_mask = (theta >= theta_start) & (theta <= theta_end)
    else:
        arc_mask = (theta >= theta_start) | (theta <= theta_end)

    mask = ring_mask & arc_mask

    theta2_start = np.mod(theta_start + np.pi, 2 * np.pi)
    theta2_end   = np.mod(theta_end + np.pi, 2 * np.pi)

    if theta2_start <= theta2_end:
        arc_mask_2 = (theta >= theta2_start) & (theta <= theta2_end)
    else:
        arc_mask_2 = (theta >= theta2_start) | (theta <= theta2_end)

    mask = mask | (ring_mask & arc_mask_2)

    return mask


def generate_mask(H, W, sample_frac=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    mask = np.zeros((H, W), dtype=np.float32)
    arc_length = rng.uniform(np.pi/6, np.pi/4)
    while np.sum(mask) < sample_frac * H * W:
        radius = rng.integers(10, min(H, W) // 2 - 5)
        theta_start = rng.uniform(0, np.pi)
        theta_end = theta_start + arc_length
        new_mask = circular_arc_mask(H, W, radius, theta_start, theta_end)
        
        mask = np.logical_or(mask, new_mask).astype(np.float32)
        
    return mask.astype(np.float32)


def generate_visibilities(file_path, img_size=None, sample_frac=0.10, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    img = io.imread(file_path, as_gray=True)
    if img_size is not None:
        img = transform.resize(img, img_size, anti_aliasing=True)
    else:
        img_size = img.shape
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    H, W = img_size
    mask = generate_mask(H, W, sample_frac=sample_frac, rng=rng)

    fft_full = np.fft.fftshift(np.fft.fft2(img, norm="ortho"))
    visibilities = mask * fft_full
        
    return visibilities, mask, img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help="Path to the Input Folder containing the images")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the Output Folder for saving the visibilities")
    parser.add_argument('--img_size', type=int, nargs=2, default=None, help="Optional image size (H W). If not provided, no resizing is done.")
    parser.add_argument('--sample_frac', type=float, default=0.10, help="Fraction of Fourier coefficients to sample (default: 0.10)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    rng = np.random.default_rng(seed=args.seed)
    for filename in sorted(os.listdir(args.input_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(args.input_path, filename)
            visibilities, mask, image = generate_visibilities(file_path, img_size=args.img_size, sample_frac=args.sample_frac, rng=rng)
            np.savez(os.path.join(args.output_path, f"{os.path.splitext(filename)[0]}.npz"), visibilities=visibilities, mask=mask, image=image)
    