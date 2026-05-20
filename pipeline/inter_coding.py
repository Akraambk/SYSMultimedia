import numpy as np
from pipeline.intra_coding import pad_channel, encode_iframe, decode_iframe
import scipy.fft

def block_matching(current_block: np.ndarray, ref_frame: np.ndarray,
                   block_x: int, block_y: int, search_range: int = 8) -> tuple[int, int]:
    """
    Returns best (dx, dy) motion vector using Sum of Absolute Differences (SAD).
    block_x, block_y are the top-left coordinates of the current_block.
    """
    best_sad = float('inf')
    best_dx = 0
    best_dy = 0
    h, w = ref_frame.shape
    bh, bw = current_block.shape
    
    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            ref_y = block_y + dy
            ref_x = block_x + dx
            
            if ref_x < 0 or ref_y < 0 or ref_x + bw > w or ref_y + bh > h:
                continue
                
            ref_block = ref_frame[ref_y:ref_y+bh, ref_x:ref_x+bw]
            sad = np.sum(np.abs(current_block.astype(int) - ref_block.astype(int)))
            
            if sad < best_sad:
                best_sad = sad
                best_dx = dx
                best_dy = dy
                
    return (best_dx, best_dy)

def encode_pframe(current_Y: np.ndarray, ref_Y: np.ndarray,
                  Q: float = 1.0, search_range: int = 8) -> dict:
    """
    Returns dict with 'motion_vectors' and 'residual_coeffs'.
    """
    h, w = current_Y.shape
    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16
    
    if pad_h > 0 or pad_w > 0:
        current_padded = np.pad(current_Y, ((0, pad_h), (0, pad_w)), mode='constant')
        ref_padded = np.pad(ref_Y, ((0, pad_h), (0, pad_w)), mode='constant')
    else:
        current_padded = current_Y
        ref_padded = ref_Y
        
    ph, pw = current_padded.shape
    
    motion_vectors = []
    residual = np.zeros_like(current_padded, dtype=np.int16)
    
    for i in range(0, ph, 16):
        row_mvs = []
        for j in range(0, pw, 16):
            current_block = current_padded[i:i+16, j:j+16]
            dx, dy = block_matching(current_block, ref_padded, j, i, search_range)
            row_mvs.append((dx, dy))
            
            ref_block = ref_padded[i+dy:i+dy+16, j+dx:j+dx+16]
            res_block = current_block.astype(np.int16) - ref_block.astype(np.int16)
            residual[i:i+16, j:j+16] = res_block
            
        motion_vectors.append(row_mvs)
        
    residual_coeffs = encode_iframe(residual, Q)
    
    return {
        "motion_vectors": motion_vectors,
        "residual_coeffs": residual_coeffs
    }

def decode_pframe(encoded: dict, ref_Y: np.ndarray, Q: float = 1.0) -> np.ndarray:
    """
    Returns reconstructed Y channel.
    """
    motion_vectors = encoded["motion_vectors"]
    residual_coeffs = encoded["residual_coeffs"]
    
    h, w = ref_Y.shape
    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16
    
    if pad_h > 0 or pad_w > 0:
        ref_padded = np.pad(ref_Y, ((0, pad_h), (0, pad_w)), mode='constant')
    else:
        ref_padded = ref_Y
        
    ph, pw = ref_padded.shape
    
    residual_recon = decode_iframe(residual_coeffs, (ph, pw), Q, is_residual=True)
    reconstructed_padded = np.zeros((ph, pw), dtype=np.float32)
    
    for i in range(0, ph, 16):
        row_mvs = motion_vectors[i // 16]
        for j in range(0, pw, 16):
            dx, dy = row_mvs[j // 16]
            ref_block = ref_padded[i+dy:i+dy+16, j+dx:j+dx+16]
            res_block = residual_recon[i:i+16, j:j+16]
            
            recon_block = ref_block.astype(np.int16) + res_block
            reconstructed_padded[i:i+16, j:j+16] = recon_block
            
    reconstructed_padded = np.clip(reconstructed_padded, 0, 255).astype(np.uint8)
    return reconstructed_padded[:h, :w]
