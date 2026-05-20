import numpy as np
import scipy.fft

# Standard JPEG luminance quantisation matrix
QUANT_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99],
], dtype=np.float32)

def pad_channel(channel: np.ndarray) -> np.ndarray:
    """Pad the channel dimension with zeros so it is a multiple of 8."""
    h, w = channel.shape
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    if pad_h == 0 and pad_w == 0:
        return channel
    return np.pad(channel, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

def encode_iframe(channel: np.ndarray, Q: float = 1.0) -> np.ndarray:
    """
    Returns 2D array of quantised integer DCT coefficients 
    (same shape as input, padded to multiples of 8).
    """
    padded = pad_channel(channel)
    h, w = padded.shape
    coeffs = np.zeros_like(padded, dtype=np.int32)
    quant = QUANT_MATRIX * Q
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            # DCT expects float input
            block = padded[i:i+8, j:j+8].astype(np.float32)
            dct_block = scipy.fft.dctn(block, norm='ortho')
            # Quantise by dividing by quant matrix element-wise and rounding
            quant_block = np.round(dct_block / quant)
            coeffs[i:i+8, j:j+8] = quant_block.astype(np.int32)
            
    return coeffs

def decode_iframe(coeffs: np.ndarray, original_shape: tuple, Q: float = 1.0, is_residual: bool = False) -> np.ndarray:
    """
    Returns reconstructed channel (cropped to original_shape).
    """
    h, w = coeffs.shape
    recon_padded = np.zeros((h, w), dtype=np.float32)
    quant = QUANT_MATRIX * Q
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            quant_block = coeffs[i:i+8, j:j+8].astype(np.float32)
            # Dequantise
            dequant_block = quant_block * quant
            # Apply IDCT
            idct_block = scipy.fft.idctn(dequant_block, norm='ortho')
            recon_padded[i:i+8, j:j+8] = idct_block
            
    if is_residual:
        recon_padded = np.round(recon_padded).astype(np.int16)
    else:
        # Clip to [0, 255] and cast back to uint8
        recon_padded = np.clip(recon_padded, 0, 255).astype(np.uint8)
    
    # Crop to original shape
    orig_h, orig_w = original_shape
    return recon_padded[:orig_h, :orig_w]
