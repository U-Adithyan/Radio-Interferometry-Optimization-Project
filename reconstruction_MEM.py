import os
import argparse

import numpy as np
from skimage import io
from tqdm import tqdm

def forward(image, sampling_mask):
    fft_img = np.fft.fftshift(np.fft.fft2(image, norm="ortho"))
    return sampling_mask * fft_img

def adjoint(visibility, sampling_mask):
    masked_vis = sampling_mask * visibility
    return np.real(np.fft.ifft2(np.fft.ifftshift(masked_vis), norm="ortho"))


def mem_reconstruct(visibilities, mask, prior=None, weights=None, lambda_entropy=1e-3, step_size=1e-5, n_iters=300):
    eps = 1e-8
    H, W = mask.shape
    
    if prior is None:
        prior = np.ones((H, W), dtype=np.float64)
    prior = np.maximum(prior.astype(np.float64), eps)
    
    if weights is None:
        weights = np.ones_like(visibilities)
    
    dirty_img = np.real(np.fft.ifft2(np.fft.ifftshift(visibilities), norm="ortho"))
    img = dirty_img - dirty_img.min()
    if img.max() > 0:
        img = img / img.max()
    img = np.maximum(img, eps)

    history = []
    for epoch in tqdm(range(n_iters), desc="MEM Reconstruction"):
        pred_vis = forward(img, mask)
        residual = pred_vis - visibilities

        data_loss = np.sum(weights * np.abs(residual) ** 2)
        entropy = -np.sum(img * np.log((img + eps) / prior))

        total_loss = data_loss - lambda_entropy * entropy
        history.append(total_loss)

        grad_data = 2.0 * adjoint( weights * residual, mask)
        grad_entropy = -np.log((img + eps) / prior) - 1.0

        update = step_size * (lambda_entropy * grad_entropy - grad_data)
        update = np.clip(update, -5, 5)
        
        img = img * np.exp(update)
        img = np.maximum(img, eps)

    return img, history



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help="Path to the Input Folder containing the visibilities")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the Output Folder for saving the reconstructed images")
    parser.add_argument('--lambda_entropy', type=float, default=1e-3, help="Weight for the entropy regularization (default: 1e-3)")
    parser.add_argument('--step_size', type=float, default=1e-5, help="Step size for the gradient descent (default: 1e-5)")
    parser.add_argument('--num_iterations', type=int, default=300, help="Number of iterations for the MEM algorithm (default: 300)")
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    for filename in sorted(os.listdir(args.input_path)):
        if filename.lower().endswith('.npz'):
            data = np.load(os.path.join(args.input_path, filename))
            visibility = data['visibilities']
            mask = data['mask']
            
            reconstructed_image, _ = mem_reconstruct(visibility, mask, lambda_entropy=args.lambda_entropy, step_size=args.step_size, n_iters=args.num_iterations) 
            final_img = reconstructed_image - reconstructed_image.min()
            if final_img.max() > 0:
                final_img = final_img / final_img.max()
                
            io.imsave(os.path.join(args.output_path, f"{os.path.splitext(filename)[0]}.png"), (final_img * 255).astype(np.uint8))