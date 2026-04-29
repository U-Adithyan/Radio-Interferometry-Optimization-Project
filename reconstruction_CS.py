import os
import argparse

import numpy as np
import pywt
from tqdm import tqdm
from skimage import io

def forward(image, sampling_mask):
    fft_img = np.fft.fftshift(np.fft.fft2(image, norm="ortho"))
    return sampling_mask * fft_img

def adjoint(visibility, sampling_mask):
    masked_vis = sampling_mask * visibility
    return np.real(np.fft.ifft2(np.fft.ifftshift(masked_vis), norm="ortho"))


def data_gradient(image, visibility, sampling_mask):
    residual = forward(image, sampling_mask) - visibility
    return adjoint(residual, sampling_mask)


def soft_threshold(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def wavelet_soft_threshold(image, lam, wavelet="db4", level=4):
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, mode="periodization")
    cA, details = coeffs[0], coeffs[1:]

    new_details = []
    for detail_level in details:
        cH, cV, cD = detail_level
        cH = soft_threshold(cH, lam)
        cV = soft_threshold(cV, lam)
        cD = soft_threshold(cD, lam)
        new_details.append((cH, cV, cD))

    new_coeffs = [cA] + new_details
    rec = pywt.waverec2(new_coeffs, wavelet=wavelet, mode="periodization")

    return rec[:image.shape[0], :image.shape[1]]


def reconstruct_fista_cs(visibility, mask, initial_image, wavelet_weight=0.01, step_size=1.0, num_iterations=500, wavelet="sym4", level=3):
    current_image = initial_image.copy()
    momentum_image = current_image.copy()
    t = 1.0

    for _ in tqdm(range(num_iterations), desc="FISTA-CS Reconstruction"):
        # Gradient step
        gradient = data_gradient(momentum_image, visibility, mask)
        gradient_step = momentum_image - step_size * gradient

        # Proximal step
        next_image = wavelet_soft_threshold(gradient_step, lam=wavelet_weight * step_size, wavelet=wavelet, level=level)
        next_image = np.clip(next_image, 0.0, 1.0)

        # FISTA momentum
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
        momentum_image = next_image + ((t - 1.0) / t_new) * (next_image - current_image)

        current_image = next_image
        t = t_new

    return current_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help="Path to the Input Folder containing the visibilities")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the Output Folder for saving the reconstructed images")
    parser.add_argument('--wavelet_weight', type=float, default=5e-3, help="Weight for the TV regularization (default: 5e-3)")
    parser.add_argument('--wavelet', type=str, default="sym4", help="Type of wavelet to use (default: sym4)")
    parser.add_argument('--wavelet_level', type=int, default=3, help="Level of wavelet decomposition (default: 3)")
    parser.add_argument('--step_size', type=float, default=0.01, help="Step size for the gradient descent (default: 1.0)")
    parser.add_argument('--num_iterations', type=int, default=300, help="Number of iterations for the FISTA algorithm (default: 300)")
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    for filename in sorted(os.listdir(args.input_path)):
        if filename.lower().endswith('.npz'):
            data = np.load(os.path.join(args.input_path, filename))
            visibility = data['visibilities']
            mask = data['mask']
            initial_image = np.clip(adjoint(visibility, mask), 0.0, 1.0)           
            reconstructed_image = reconstruct_fista_cs(visibility, mask, initial_image, wavelet_weight=args.wavelet_weight, step_size=args.step_size, num_iterations=args.num_iterations, wavelet=args.wavelet, level=args.wavelet_level)
            final_img = reconstructed_image - reconstructed_image.min()
            if final_img.max() > 0:
               final_img = final_img / final_img.max()
            io.imsave(os.path.join(args.output_path, f"{os.path.splitext(filename)[0]}.png"), (final_img * 255).astype(np.uint8))