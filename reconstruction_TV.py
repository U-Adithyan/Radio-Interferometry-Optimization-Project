import os
import argparse

import numpy as np
from tqdm import tqdm
from skimage import io
from skimage.restoration import denoise_tv_chambolle

def forward(image, sampling_mask):
    fft_img = np.fft.fftshift(np.fft.fft2(image, norm="ortho"))
    return sampling_mask * fft_img

def adjoint(visibility, sampling_mask):
    masked_vis = sampling_mask * visibility
    return np.real(np.fft.ifft2(np.fft.ifftshift(masked_vis), norm="ortho"))


def data_gradient(image, visibility, sampling_mask):
    residual = forward(image, sampling_mask) - visibility
    return adjoint(residual, sampling_mask)

def reconstruct_fista_tv( visibility, mask, initial_image, tv_weight=0.02, step_size=1.0, num_iterations=500):
    current_image = initial_image.copy()
    momentum_image = current_image.copy()
    t = 1.0

    for _ in tqdm(range(num_iterations), desc="FISTA-TV Reconstruction"):
        # Gradient step
        gradient = data_gradient(momentum_image, visibility, mask)
        gradient_step = momentum_image - step_size * gradient

        # Proximal step for TV
        next_image = denoise_tv_chambolle(gradient_step, weight=tv_weight * step_size, channel_axis=None)
        #next_image = np.clip(next_image, 0.0, 1.0)

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
    parser.add_argument('--tv_weight', type=float, default=5e-3, help="Weight for the TV regularization (default: 5e-3)")
    parser.add_argument('--step_size', type=float, default=1.0, help="Step size for the gradient descent (default: 1.0)")
    parser.add_argument('--num_iterations', type=int, default=300, help="Number of iterations for the FISTA algorithm (default: 300)")
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    for filename in sorted(os.listdir(args.input_path)):
        if filename.lower().endswith('.npz'):
            data = np.load(os.path.join(args.input_path, filename))
            visibility = data['visibilities']
            mask = data['mask']
            initial_image = adjoint(visibility, mask)
            
            reconstructed_image = reconstruct_fista_tv(visibility, mask, initial_image, tv_weight=args.tv_weight, step_size=args.step_size, num_iterations=args.num_iterations)
            final_img = reconstructed_image - reconstructed_image.min()
            if final_img.max() > 0:
                final_img = final_img / final_img.max()
            io.imsave(os.path.join(args.output_path, f"{os.path.splitext(filename)[0]}.png"), (final_img * 255).astype(np.uint8))