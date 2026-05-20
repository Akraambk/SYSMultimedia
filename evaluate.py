import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pipeline.preprocessing import bgr_to_ycbcr, ycbcr_to_bgr
from pipeline.intra_coding import encode_iframe, decode_iframe
from pipeline.inter_coding import encode_pframe, decode_pframe
from encoder import main as run_encoder
from decoder import main as run_decoder

def compute_metrics(original_frames, reconstructed_frames, bin_path):
    original_size = sum(f.nbytes for f in original_frames)
    compressed_size = os.path.getsize(bin_path)
    compression_ratio = original_size / compressed_size

    psnr_values = []
    for orig, recon in zip(original_frames, reconstructed_frames):
        mse = np.mean((orig.astype(float) - recon.astype(float)) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        psnr_values.append(psnr)

    return compression_ratio, psnr_values

def visualize_pipeline(frames_dir, output_png="output/pipeline_visualisation.png"):
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")) +
                         glob.glob(os.path.join(frames_dir, "*.jpg")))
    
    if len(frame_paths) < 2:
        print("Need at least 2 frames for visualisation.")
        return

    frames = [cv2.imread(p) for p in frame_paths[:4]] # Load first 4 frames
    if any(f is None for f in frames):
        raise ValueError("Error loading frames.")

    fig = plt.figure(figsize=(18, 14))
    
    # --- Row 1: Original input frames ---
    for i, frame in enumerate(frames):
        ax = plt.subplot(5, 4, i + 1)
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Original Frame {i}")
        ax.axis("off")
        
    # --- Row 2: Y, Cb, Cr channels of frame 0 ---
    Y, Cb, Cr = bgr_to_ycbcr(frames[0])
    
    ax = plt.subplot(5, 4, 5)
    ax.imshow(Y, cmap="gray")
    ax.set_title("Y Channel")
    ax.axis("off")
    
    ax = plt.subplot(5, 4, 6)
    ax.imshow(Cb, cmap="gray")
    ax.set_title("Cb Channel")
    ax.axis("off")
    
    ax = plt.subplot(5, 4, 7)
    ax.imshow(Cr, cmap="gray")
    ax.set_title("Cr Channel")
    ax.axis("off")
    
    # --- Row 3: One 8x8 block (Raw -> DCT -> Quantised -> Reconstructed) ---
    block = Y[0:8, 0:8]
    import scipy.fft
    dct_block = scipy.fft.dctn(block.astype(np.float32), norm='ortho')
    from pipeline.intra_coding import QUANT_MATRIX
    quant_block = np.round(dct_block / QUANT_MATRIX)
    dequant_block = quant_block * QUANT_MATRIX
    recon_block = scipy.fft.idctn(dequant_block, norm='ortho')
    recon_block = np.clip(recon_block, 0, 255).astype(np.uint8)
    
    ax = plt.subplot(5, 4, 9)
    ax.imshow(block, cmap="gray")
    ax.set_title("8x8 Raw Block")
    ax.axis("off")
    
    ax = plt.subplot(5, 4, 10)
    ax.imshow(np.log(np.abs(dct_block)+1), cmap="hot")
    ax.set_title("DCT Coefficients (Log)")
    ax.axis("off")
    
    ax = plt.subplot(5, 4, 11)
    ax.imshow(quant_block, cmap="jet")
    ax.set_title("Quantised DCT")
    ax.axis("off")
    
    ax = plt.subplot(5, 4, 12)
    ax.imshow(recon_block, cmap="gray")
    ax.set_title("Reconstructed Block")
    ax.axis("off")
    
    # --- Row 4/5: P-Frame Visualization ---
    Y_prev = Y
    Y_curr, _, _ = bgr_to_ycbcr(frames[1])
    encoded_p = encode_pframe(Y_curr, Y_prev, Q=1.0, search_range=8)
    decode_p_y = decode_pframe(encoded_p, Y_prev, Q=1.0)
    
    mvs = encoded_p["motion_vectors"]
    h, w = Y_curr.shape
    
    # Arrow plot for motion vectors
    ax = plt.subplot(5, 4, 13)
    ax.imshow(Y_curr, cmap="gray")
    X, Y_coords, U, V = [], [], [], []
    for i in range(len(mvs)):
        for j in range(len(mvs[i])):
            dx, dy = mvs[i][j]
            if dx != 0 or dy != 0:
                X.append(j * 16 + 8)
                Y_coords.append(i * 16 + 8)
                U.append(dx)
                V.append(dy)
    if U:
        ax.quiver(X, Y_coords, U, V, color="red", scale=1, scale_units='xy', angles='xy')
    ax.set_title("P-Frame with MVs")
    ax.axis("off")
    
    # Residual
    ax = plt.subplot(5, 4, 14)
    residual = Y_curr.astype(int) - decode_p_y.astype(int)
    # scale for visibility
    res_vis = np.clip(128 + residual, 0, 255).astype(np.uint8)
    ax.imshow(res_vis, cmap="gray")
    ax.set_title("Residual Map")
    ax.axis("off")
    
    ax = plt.subplot(5, 4, 15)
    ax.imshow(decode_p_y, cmap="gray")
    ax.set_title("Reconstructed P-Frame")
    ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"Visulisation saved to {output_png}")

if __name__ == "__main__":
    print("Run this file independently with logic added for full evaluation.")
